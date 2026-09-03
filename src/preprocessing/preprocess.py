"""Hasta bazlı ECoG ön işleme (Miller 2019, faces_basic).

Pipeline (patient_id parametresiyle çalışır):
    1. `.mat` dosyasını yükle (data, stim, srate)
    2. Welch PSD ile şebeke gürültüsü frekansını (50 Hz / 60 Hz) tespit et
    3. EDA'daki yönteme benzer şekilde bozuk kanalları tespit et ve çıkar
    4. Tespit edilen frekans + ilk 2 harmoniğine notch filtre uygula
    5. Common Average Reference (CAR) uygula
    6. 0.5-200 Hz FIR bandpass filtre (zero-phase, filtfilt) uygula

Kullanım (kütüphane olarak):
    from src.preprocessing.preprocess import preprocess_patient
    result = preprocess_patient("ap")

Kullanım (komut satırından):
    python src/preprocessing/preprocess.py ap
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import filtfilt, firwin, iirnotch, welch
from scipy.stats import kurtosis

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "faces_basic" / "data"
LOCS_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "faces_basic" / "locs"
PROCESSED_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

LINE_FREQ_CANDIDATES = (50, 60)

# `fhpred_master.m` (data/raw/faces_basic/fhpred_master.m) içindeki area_lbls
# hücre dizisinden çıkarıldı. elcode değeri, bu listenin 1-indexed sırasına
# karşılık gelir (Destrieux ve ark. 2010 anatomik parselasyon sistemi).
AREA_LABELS: dict[int, str] = {
    1: "Temporal pole",
    2: "Parahippocampal gyrus",
    3: "Inferior temporal gyrus",
    4: "Middle temporal gyrus",
    5: "fusiform gyrus",
    6: "Lingual gyrus",
    7: "Inferior occipital gyrus",
    8: "Cuneus",
    9: "Post-ventral cingulate gyrus",
    10: "Middle Occipital gyrus",
    11: "occipital pole",
    12: "precuneus",
    13: "Superior occipital gyrus",
    14: "Post-dorsal cingulate gyrus",
    15: " ",
    16: " ",
    17: " ",
    18: " ",
    19: " ",
    20: "Non-included area",
}

# Bölge grupları: hangi elcode'ların "temporal", "fusiform", "occipital"
# kategorisine girdiğini işaretlemek için.
TEMPORAL_CODES = {1, 2, 3, 4}  # Temporal pole, Parahippocampal, Inf./Mid. temporal gyrus
FUSIFORM_CODES = {5}  # fusiform gyrus (lateral occipito-temporal)
OCCIPITAL_CODES = {7, 8, 10, 11, 13}  # Inf./Mid./Sup. occipital gyrus, cuneus, occipital pole


def load_patient(patient_id: str, raw_dir: Path = RAW_DATA_DIR) -> tuple[np.ndarray, np.ndarray, int]:
    """`<patient_id>_faceshouses.mat` dosyasını yükler.

    Döndürür: (data (n_samples x n_channels), stim (n_samples,), srate)
    """
    mat_path = raw_dir / patient_id / f"{patient_id}_faceshouses.mat"
    if not mat_path.exists():
        raise FileNotFoundError(f"Hasta dosyası bulunamadı: {mat_path}")
    mat = sio.loadmat(mat_path)
    data = mat["data"].astype(np.float64)
    stim = mat["stim"].ravel()
    srate = int(mat["srate"].ravel()[0])
    return data, stim, srate


def get_anatomical_labels(patient_id: str, locs_dir: Path = LOCS_DIR) -> pd.DataFrame:
    """`locs/<patient_id>_xslocs.mat` dosyasından elektrotların anatomik
    bölge etiketlerini çıkarır (`elcode` -> `AREA_LABELS`, fh_get_events.m'e
    benzer bir kod-çözme yaklaşımıyla `fhpred_master.m`'deki area_lbls
    listesinden türetildi).

    Döndürür: sütunları `channel` (0-indexed), `elcode`, `area_label`,
    `is_temporal`, `is_fusiform`, `is_occipital` olan bir DataFrame.
    """
    locs_path = locs_dir / f"{patient_id}_xslocs.mat"
    if not locs_path.exists():
        raise FileNotFoundError(f"Koordinat dosyası bulunamadı: {locs_path}")

    locs_mat = sio.loadmat(locs_path)
    elcodes = locs_mat["elcode"].ravel().astype(int)

    df = pd.DataFrame({
        "channel": np.arange(len(elcodes)),
        "elcode": elcodes,
        "area_label": [AREA_LABELS.get(int(c), "Bilinmeyen kod") for c in elcodes],
    })
    df["is_temporal"] = df["elcode"].isin(TEMPORAL_CODES)
    df["is_fusiform"] = df["elcode"].isin(FUSIFORM_CODES)
    df["is_occipital"] = df["elcode"].isin(OCCIPITAL_CODES)
    return df


def compute_welch_psd(data: np.ndarray, srate: int) -> tuple[np.ndarray, np.ndarray]:
    """Kanal ortalamalı Welch PSD. Döndürür: (freqs, mean_psd)."""
    nperseg = min(4 * srate, data.shape[0])
    freqs, psd = welch(data, fs=srate, nperseg=nperseg, axis=0)
    return freqs, psd.mean(axis=1)


def detect_line_noise_freq(
    data: np.ndarray,
    srate: int,
    candidates: tuple[int, ...] = LINE_FREQ_CANDIDATES,
    peak_bw: float = 1.0,
    baseline_bw: float = 5.0,
) -> dict:
    """Welch PSD üzerinden 50 Hz / 60 Hz şebeke gürültüsünü karşılaştırır.

    Her aday frekans için tepe/taban (peak/baseline) güç oranı (SNR) hesaplanır
    -- ham tepe gücü yerine bu oran kullanılır, çünkü ECoG PSD'si 1/f yapısında
    azaldığından ham güç karşılaştırması düşük frekanslı adayı yanıltıcı
    şekilde kayırabilir. En yüksek SNR'a sahip aday "baskın" kabul edilir.
    """
    freqs, mean_psd = compute_welch_psd(data, srate)

    snrs = {}
    for f0 in candidates:
        peak_mask = (freqs >= f0 - peak_bw) & (freqs <= f0 + peak_bw)
        base_mask = ((freqs >= f0 - baseline_bw) & (freqs < f0 - peak_bw)) | (
            (freqs > f0 + peak_bw) & (freqs <= f0 + baseline_bw)
        )
        peak = mean_psd[peak_mask].max() if peak_mask.any() else 0.0
        baseline = np.median(mean_psd[base_mask]) if base_mask.any() else np.nan
        snrs[f0] = float(peak / baseline) if baseline else float("nan")

    dominant = max(snrs, key=snrs.get)
    return {
        "dominant_freq": dominant,
        "snr_by_candidate": snrs,
        "freqs": freqs,
        "mean_psd": mean_psd,
    }


def compute_bad_channels(data: np.ndarray, z_thresh: float = 3.0) -> list[int]:
    """EDA'daki yöntemle aynı: log-varyans ve kurtosis z-score > eşik olan
    kanalları potansiyel bozuk kanal olarak işaretler."""
    variance = data.var(axis=0)
    kurt = kurtosis(data, axis=0, fisher=True)

    log_var = np.log(variance)
    var_z = (log_var - log_var.mean()) / log_var.std()
    kurt_z = (kurt - kurt.mean()) / kurt.std()

    bad = np.where((np.abs(var_z) > z_thresh) | (np.abs(kurt_z) > z_thresh))[0]
    return bad.tolist()


def notch_filter(
    data: np.ndarray, srate: int, base_freq: float, n_harmonics: int = 2, quality: float = 30.0
) -> np.ndarray:
    """Temel şebeke frekansı + ilk `n_harmonics` harmoniğine zero-phase notch filtre uygular."""
    filtered = data.copy()
    for h in range(1, n_harmonics + 2):
        f0 = base_freq * h
        if f0 >= srate / 2:
            break
        b, a = iirnotch(f0, quality, srate)
        filtered = filtfilt(b, a, filtered, axis=0)
    return filtered


def common_average_reference(data: np.ndarray) -> np.ndarray:
    """Common Average Reference: her örnekte kanallar arası ortalamayı çıkarır."""
    return data - data.mean(axis=1, keepdims=True)


def bandpass_filter_fir(
    data: np.ndarray, srate: int, low: float = 0.5, high: float = 200.0, numtaps: Optional[int] = None
) -> np.ndarray:
    """0.5-200 Hz FIR bandpass filtre, zero-phase (filtfilt) uygular."""
    if numtaps is None:
        numtaps = int(srate) + 1
        if numtaps % 2 == 0:
            numtaps += 1
    taps = firwin(numtaps, [low, high], pass_zero=False, fs=srate)
    return filtfilt(taps, [1.0], data, axis=0)


def preprocess_patient(
    patient_id: str,
    raw_dir: Path = RAW_DATA_DIR,
    bad_channels: Optional[list[int]] = None,
    remove_bad_channels: bool = True,
    n_harmonics: int = 2,
    bandpass: tuple[float, float] = (0.5, 200.0),
) -> dict:
    """Bir hasta için tüm ön işleme pipeline'ını çalıştırır.

    Adımlar: line-noise tespiti -> bozuk kanal tespiti/çıkarma -> notch ->
    CAR -> bandpass. Döndürülen sözlük, öncesi/sonrası PSD karşılaştırması
    için gereken tüm ara verileri içerir.
    """
    data, stim, srate = load_patient(patient_id, raw_dir)

    line_info = detect_line_noise_freq(data, srate)
    line_freq = line_info["dominant_freq"]

    if bad_channels is None:
        bad_channels = compute_bad_channels(data)

    n_channels = data.shape[1]
    kept_channels = (
        [ch for ch in range(n_channels) if ch not in set(bad_channels)]
        if remove_bad_channels
        else list(range(n_channels))
    )
    data_kept = data[:, kept_channels]

    processed = notch_filter(data_kept, srate, line_freq, n_harmonics=n_harmonics)
    processed = common_average_reference(processed)
    processed = bandpass_filter_fir(processed, srate, *bandpass)

    freqs_after, psd_after = compute_welch_psd(processed, srate)
    freqs_before_kept, psd_before_kept = compute_welch_psd(data_kept, srate)

    return {
        "patient_id": patient_id,
        "srate": srate,
        "stim": stim,
        "n_channels_total": n_channels,
        "bad_channels": bad_channels,
        "kept_channels": kept_channels,
        "line_freq": line_freq,
        "line_freq_snrs": line_info["snr_by_candidate"],
        "raw_data": data,
        "freqs_before": freqs_before_kept,
        "psd_before": psd_before_kept,
        "processed_data": processed,
        "freqs_after": freqs_after,
        "psd_after": psd_after,
    }


def list_available_patients(raw_dir: Path = RAW_DATA_DIR) -> list[str]:
    """`raw_dir` altındaki `<id>/<id>_faceshouses.mat` yapısına uyan tüm
    hasta kodlarını döndürür."""
    return sorted(
        p.name for p in raw_dir.iterdir()
        if p.is_dir() and (p / f"{p.name}_faceshouses.mat").exists()
    )


def process_and_save_all_patients(
    patients: Optional[list[str]] = None,
    raw_dir: Path = RAW_DATA_DIR,
    locs_dir: Path = LOCS_DIR,
    out_dir: Path = PROCESSED_DATA_DIR,
) -> pd.DataFrame:
    """Verilen (veya bulunan tüm) hastalar için `preprocess_patient()`
    çalıştırır; işlenmiş sinyali + kalan kanalların anatomik etiketlerini
    `data/processed/<patient_id>/` altına kaydeder.

    Kaydedilen dosyalar:
        - `<id>_processed.npz`: processed_data, stim, srate, kept_channels,
          bad_channels, line_freq
        - `<id>_anatomical_labels.csv`: kalan kanallar için elcode/area_label
          ve temporal/fusiform/occipital bayrakları (processed_data ile aynı
          kanal sırasında, `processed_channel` sütunuyla hizalı)

    Döndürür: her hasta için özet satırı içeren bir DataFrame (aynı zamanda
    `data/processed/patients_summary.csv` olarak da kaydedilir).
    """
    if patients is None:
        patients = list_available_patients(raw_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for patient_id in patients:
        result = preprocess_patient(patient_id, raw_dir=raw_dir)
        anat_df = get_anatomical_labels(patient_id, locs_dir=locs_dir)

        kept = result["kept_channels"]
        anat_kept = anat_df.iloc[kept].reset_index(drop=True)
        anat_kept["processed_channel"] = np.arange(len(kept))

        patient_out_dir = out_dir / patient_id
        patient_out_dir.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            patient_out_dir / f"{patient_id}_processed.npz",
            data=result["processed_data"],
            stim=result["stim"],
            srate=result["srate"],
            kept_channels=np.array(kept),
            bad_channels=np.array(result["bad_channels"]),
            line_freq=result["line_freq"],
        )
        anat_kept.to_csv(patient_out_dir / f"{patient_id}_anatomical_labels.csv", index=False)

        summary_rows.append({
            "patient": patient_id,
            "n_channels_total": result["n_channels_total"],
            "n_bad_channels": len(result["bad_channels"]),
            "n_kept_channels": len(kept),
            "line_freq_hz": result["line_freq"],
            "n_temporal": int(anat_kept["is_temporal"].sum()),
            "n_fusiform": int(anat_kept["is_fusiform"].sum()),
            "n_occipital": int(anat_kept["is_occipital"].sum()),
            "has_temporal_or_fusiform": bool(
                anat_kept["is_temporal"].any() or anat_kept["is_fusiform"].any()
            ),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "patients_summary.csv", index=False)
    return summary_df


if __name__ == "__main__":
    patient = sys.argv[1] if len(sys.argv) > 1 else "ap"
    result = preprocess_patient(patient)

    print(f"Hasta: {result['patient_id']}")
    print(f"Örnekleme hızı: {result['srate']} Hz")
    print(f"Toplam kanal sayısı: {result['n_channels_total']}")
    print(f"Tespit edilen şebeke gürültüsü frekansı: {result['line_freq']} Hz "
          f"(SNR: {result['line_freq_snrs']})")
    print(f"Bozuk kanal sayısı: {len(result['bad_channels'])} -> {result['bad_channels']}")
    print(f"Kalan kanal sayısı: {len(result['kept_channels'])}")
    print(f"İşlenmiş veri boyutu: {result['processed_data'].shape}")
