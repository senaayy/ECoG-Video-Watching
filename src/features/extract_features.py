"""Hasta bazlı özellik (feature) çıkarımı (Miller 2019, faces_basic).

`data/processed/<hasta>/`'daki ön işlenmiş sinyal (`*_processed.npz`) ve
etiket (`labels.parquet`) dosyalarını kullanarak, her face/house event'inin
0-400ms penceresi için (sunum süresi labels.parquet'ten otomatik doğrulanır)
kanal bazlı spektral özellikler çıkarır:

    - High-gamma (70-150 Hz) Hilbert zarfının ortalama gücü
    - theta (4-8 Hz), alpha (8-13 Hz), beta (13-30 Hz), gamma (30-70 Hz)
      bant güçleri (Welch PSD)

Yalnızca `get_anatomical_labels()` çıktısına göre temporal/fusiform/
occipital olarak işaretlenmiş kanallar kullanılır.

Kullanım (kütüphane olarak):
    from src.features.extract_features import extract_and_save_features
    result = extract_and_save_features("ap")

Kullanım (komut satırından, tüm hastalar için):
    python src/features/extract_features.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import hilbert, welch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.preprocessing.preprocess import (
    PROCESSED_DATA_DIR,
    bandpass_filter_fir,
    list_available_patients,
)

HIGH_GAMMA_BAND = (70.0, 150.0)
CLASSIC_BANDS = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 70.0),
}
FEATURE_NAMES = ["high_gamma"] + list(CLASSIC_BANDS.keys())


def infer_event_duration_sec(labels_df: pd.DataFrame) -> float:
    """face/house event sürelerini `labels_df`'teki ardışık onset zamanlarından
    otomatik olarak çıkarır (varsayılmaz). Her stimulus event'inin süresi,
    bir sonraki event'in (ISI/baseline) başlangıcına kadar geçen süredir."""
    df = labels_df.sort_values("sample_idx").reset_index(drop=True)
    df["duration_sec"] = df["time_sec"].shift(-1) - df["time_sec"]
    stim_events = df[df["class"].isin(["face", "house"])]
    durations = stim_events["duration_sec"].dropna()
    if durations.empty:
        raise ValueError("Event süresi çıkarılamadı: face/house event'i bulunamadı.")
    duration = float(durations.median())
    if durations.std() > 1e-3:
        raise ValueError(
            f"Event süreleri tutarsız (std={durations.std():.4f}s) -- "
            "otomatik çıkarım güvenilir değil."
        )
    return duration


def compute_high_gamma_envelope_power(
    data: np.ndarray, srate: int, band: tuple[float, float] = HIGH_GAMMA_BAND
) -> np.ndarray:
    """Sürekli sinyali `band` aralığına bandpass filtreler, Hilbert zarfının
    karesini (anlık güç) döndürür. Kısa event pencereleri üzerinde ayrı ayrı
    filtrelemek yerine tüm sinyal üzerinde bir kez hesaplanır (kenar
    etkilerinden kaçınmak için) ve pencereler sonradan bu diziden kesilir."""
    filtered = bandpass_filter_fir(data, srate, band[0], band[1])
    analytic = hilbert(filtered, axis=0)
    envelope = np.abs(analytic)
    return envelope**2


def extract_patient_features(patient_id: str, processed_dir: Path = PROCESSED_DATA_DIR) -> Optional[dict]:
    """Bir hasta için (n_events x n_roi_channels x n_features) özellik
    matrisini ve ilgili meta verileri üretir.

    ROI (temporal/fusiform/occipital) dışı kanallar hesaplamaya hiç dahil
    edilmez. Hastada ROI kanalı yoksa `None` döner.
    """
    patient_dir = processed_dir / patient_id
    npz = np.load(patient_dir / f"{patient_id}_processed.npz")
    data = npz["data"]
    srate = int(npz["srate"])

    anat_df = pd.read_csv(patient_dir / f"{patient_id}_anatomical_labels.csv")
    roi_mask = anat_df["is_temporal"] | anat_df["is_fusiform"] | anat_df["is_occipital"]
    roi_rows = anat_df.loc[roi_mask].sort_values("processed_channel")
    roi_channels = roi_rows["processed_channel"].to_numpy()
    roi_area_labels = roi_rows["area_label"].to_numpy()

    if len(roi_channels) == 0:
        return None

    data_roi = data[:, roi_channels]

    labels_df = pd.read_parquet(patient_dir / "labels.parquet")
    window_sec = infer_event_duration_sec(labels_df)
    window_samples = int(round(window_sec * srate))

    events_df = labels_df[labels_df["class"].isin(["face", "house"])].reset_index(drop=True)

    hg_power_full = compute_high_gamma_envelope_power(data_roi, srate)

    n_events = len(events_df)
    n_channels = len(roi_channels)
    n_features = len(FEATURE_NAMES)

    features = np.full((n_events, n_channels, n_features), np.nan)
    event_labels = np.empty(n_events, dtype=object)
    valid_mask = np.zeros(n_events, dtype=bool)

    for i, row in events_df.iterrows():
        start = int(row["sample_idx"])
        end = start + window_samples
        if end > data_roi.shape[0]:
            continue  # kayıt sonuna taşan event'i atla

        window = data_roi[start:end]

        features[i, :, 0] = hg_power_full[start:end].mean(axis=0)

        freqs, psd = welch(window, fs=srate, nperseg=window_samples, axis=0)
        for b_idx, (lo, hi) in enumerate(CLASSIC_BANDS.values(), start=1):
            band_mask = (freqs >= lo) & (freqs <= hi)
            features[i, :, b_idx] = psd[band_mask].mean(axis=0)

        event_labels[i] = row["class"]
        valid_mask[i] = True

    return {
        "features": features[valid_mask],
        "event_labels": event_labels[valid_mask],
        "feature_names": FEATURE_NAMES,
        "channels": roi_channels,
        "area_labels": roi_area_labels,
        "srate": srate,
        "window_sec": window_sec,
        "n_events_skipped": int((~valid_mask).sum()),
    }


def extract_and_save_features(
    patient_id: str, processed_dir: Path = PROCESSED_DATA_DIR
) -> Optional[dict]:
    """`extract_patient_features()`'ı çalıştırır ve sonucu
    `data/processed/<hasta>/features.npz` olarak kaydeder."""
    result = extract_patient_features(patient_id, processed_dir)
    if result is None:
        return None

    out_path = processed_dir / patient_id / "features.npz"
    np.savez_compressed(
        out_path,
        features=result["features"],
        event_labels=result["event_labels"].astype(str),
        feature_names=np.array(result["feature_names"]),
        channels=result["channels"],
        area_labels=result["area_labels"].astype(str),
        srate=result["srate"],
        window_sec=result["window_sec"],
    )
    return result


def extract_and_save_all_patients(
    patients: Optional[list[str]] = None, processed_dir: Path = PROCESSED_DATA_DIR
) -> pd.DataFrame:
    """Tüm (veya verilen) hastalar için özellik çıkarımını çalıştırır ve
    özet bir DataFrame döndürür (aynı zamanda
    `data/processed/features_summary.csv` olarak kaydedilir)."""
    if patients is None:
        patients = list_available_patients()

    rows = []
    for patient_id in patients:
        result = extract_and_save_features(patient_id, processed_dir)
        if result is None:
            rows.append({
                "patient": patient_id,
                "n_roi_channels": 0,
                "n_events": 0,
                "n_face": 0,
                "n_house": 0,
                "n_events_skipped": 0,
                "window_sec": None,
            })
            continue
        labels = result["event_labels"]
        rows.append({
            "patient": patient_id,
            "n_roi_channels": len(result["channels"]),
            "n_events": len(labels),
            "n_face": int((labels == "face").sum()),
            "n_house": int((labels == "house").sum()),
            "n_events_skipped": result["n_events_skipped"],
            "window_sec": result["window_sec"],
        })

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(processed_dir / "features_summary.csv", index=False)
    return summary_df


if __name__ == "__main__":
    summary = extract_and_save_all_patients()
    print(summary.to_string(index=False))
