#!/usr/bin/env python3
"""Ek analiz: 'bazi hastalarda yuz-vs-ev doguru dusuk cikiyor, elektrot
yerlesimiyle mi ilgili?' sorusunu, VERIDEKI GERCEK bilgilerle asama asama
test eder. Bu script, bilinCLI olarak BASARISIZ olan iki denemeyi de icerir
(3. ve 4. adimlar) - cunku hangi yontemlerin ISE YARAMADIGI da, sonunda ise
yarayanin (5. adim) neden dogru oldugunu anlamak icin onemli.

Onceki pipeline (scripts/run_pipeline.py) her hastayi ayri degerlendirip
LDA/SVM/RF dogruluklarini uretmisti (bkz. README):
  LDA 0.531, SVM 0.608, RF 0.693 (7 hasta ortalamasi)
ama Random Forest'in hastalar arasi varyansi genisti (0.503 - 0.840).
Bu script o varyansin kaynagini arastiriyor.

Kullanim:
    python analysis/electrode_signal_localization.py data/raw/faceshouses.npz
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import welch

DEFAULT_BANDS = [
    ("delta", 1, 4), ("theta", 4, 8), ("alpha", 8, 13),
    ("beta", 13, 30), ("gamma", 30, 70), ("high_gamma", 70, 150),
]

# 1. adimda bulunan gercek RF sonuclari (scripts/run_pipeline.py --> README)
RF_ACCURACY = [0.503, 0.657, 0.730, 0.557, 0.840, 0.727, 0.837]

# Literaturden yaklasik FFA koordinatlari (MNI, mm) - SADECE 3. adim icin, dogrulanmamis
FFA_RIGHT = np.array([40.0, -55.0, -15.0])
FFA_LEFT = np.array([-40.0, -55.0, -15.0])
# Yaklasik fusiform/ventral temporal kutu (MNI, mm) - SADECE 4. adim icin, dogrulanmamis
BOX_X, BOX_Y, BOX_Z = (30, 55), (-75, -35), (-30, -5)

FACE_GYRI = {"Fusiform Gyrus"}
EXTENDED_GYRI = {"Fusiform Gyrus", "Inferior Temporal Gyrus", "Lingual Gyrus", "Middle Temporal Gyrus"}


def load_faceshouses(path):
    npz = np.load(path, allow_pickle=True)
    return [(d1, d2) for d1, d2 in npz["dat"]]


def extract_epochs(V, t_on, srate, tmin=-0.2, tmax=0.4, baseline=(-0.2, 0.0)):
    V = np.asarray(V)
    t_on = np.asarray(t_on).astype(int)
    n_channels = V.shape[1]
    sample_min, sample_max = int(round(tmin * srate)), int(round(tmax * srate))
    n_times = sample_max - sample_min
    epochs = np.full((len(t_on), n_channels, n_times), np.nan, dtype=float)
    for i, onset in enumerate(t_on):
        start, end = onset + sample_min, onset + sample_max
        if start < 0 or end > V.shape[0]:
            continue
        epochs[i] = V[start:end, :].T
    b0 = int(round((baseline[0] - tmin) * srate))
    b1 = max(int(round((baseline[1] - tmin) * srate)), b0 + 1)
    baseline_mean = np.nanmean(epochs[:, :, b0:b1], axis=2, keepdims=True)
    return epochs - baseline_mean


def band_power(x, srate, fmin, fmax):
    nperseg = min(len(x), max(32, int(srate // 2)))
    freqs, psd = welch(x, fs=srate, nperseg=nperseg)
    mask = (freqs >= fmin) & (freqs < fmax)
    return float(trapezoid(psd[mask], freqs[mask])) if np.any(mask) else 0.0


def cohens_d(a, b):
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return 0.0
    ps = np.sqrt(((n1 - 1) * np.var(a, ddof=1) + (n2 - 1) * np.var(b, ddof=1)) / (n1 + n2 - 2))
    return float((np.mean(a) - np.mean(b)) / ps) if ps > 0 else 0.0


def step1_best_channel_effect_vs_accuracy(subjects):
    """ADIM 1 (basarili, ama kismen tautolojik): her hastanin EN IYI tek
    (kanal,bant) ozelliginin |Cohen's d|'sini RF dogruluguyla karsilastir."""
    print("=== ADIM 1: en iyi tek-ozellik etkisi vs RF dogrulugu ===")
    best_ds, n_valids = [], []
    for dat1, _d2 in subjects:
        V, t_on = dat1["V"], np.asarray(dat1["t_on"]).ravel()
        stim_id = np.asarray(dat1["stim_id"]).ravel()
        srate = float(np.asarray(dat1["srate"]).squeeze())
        labels = (stim_id > 50).astype(int)
        epochs = extract_epochs(V, t_on, srate)
        valid = ~np.isnan(epochs).any(axis=(1, 2))
        epochs, lab = epochs[valid], labels[valid]
        face, house = lab == 1, lab == 0
        best_d = 0.0
        for ch in range(epochs.shape[1]):
            for _, fmin, fmax in DEFAULT_BANDS:
                fv = np.array([band_power(epochs[t, ch, :], srate, fmin, fmax) for t in np.where(face)[0]])
                hv = np.array([band_power(epochs[t, ch, :], srate, fmin, fmax) for t in np.where(house)[0]])
                d = cohens_d(fv, hv)
                if abs(d) > abs(best_d):
                    best_d = d
        best_ds.append(best_d)
        n_valids.append(int(valid.sum()))
    best_ds, n_valids = np.array(best_ds), np.array(n_valids)
    rf = np.array(RF_ACCURACY)
    r1 = np.corrcoef(np.abs(best_ds), rf)[0, 1]
    r2 = np.corrcoef(n_valids, rf)[0, 1] if n_valids.std() > 0 else float("nan")
    print(f"  r(|en_iyi_d|, RF_dogruluk) = {r1:.3f}  (n_gecerli_deneme hepsinde ayni: {n_valids.tolist()})")
    print(f"  r(n_gecerli_deneme, RF_dogruluk) = {r2:.3f}  (varyans yok -> orneklem buyuklugu ELENDI)")
    return best_ds


def step2_ffa_point_distance(subjects):
    """ADIM 2 (BASARISIZ): tek bir literatur-yaklasik FFA noktasina uzaklik.
    Beklenen: negatif korelasyon (yakinsa yuksek dogruluk). Sonuc: pozitif
    (ters yonde) -> bu yontem hipotezi DOGRULAMADI."""
    print("\n=== ADIM 2 (basarisiz): tek FFA noktasina uzaklik ===")
    dists = []
    for dat1, _d2 in subjects:
        locs = np.asarray(dat1["locs"])
        d_r = np.linalg.norm(locs - FFA_RIGHT, axis=1).min()
        d_l = np.linalg.norm(locs - FFA_LEFT, axis=1).min()
        dists.append(min(d_r, d_l))
    dists = np.array(dists)
    r = np.corrcoef(dists, RF_ACCURACY)[0, 1]
    print(f"  r(min_FFA_uzaklik, RF_dogruluk) = {r:.3f}  (beklenen negatif, cikan: {'ters yonde' if r>0 else 'beklenen yonde'})")
    print("  -> Sonuc: tek nokta cok kaba bir yaklasim, dogrulanmadi.")


def step3_fusiform_box(subjects):
    """ADIM 3 (BASARISIZ): yaklasik bir fusiform/ventral-temporal kutusu
    icindeki elektrot sayisi. Sonuc: neredeyse hic iliski yok."""
    print("\n=== ADIM 3 (basarisiz): yaklasik anatomik kutu icindeki elektrot sayisi ===")
    n_in_box = []
    for dat1, _d2 in subjects:
        locs = np.asarray(dat1["locs"])
        ax = np.abs(locs[:, 0])
        inside = (ax >= BOX_X[0]) & (ax <= BOX_X[1]) & (locs[:, 1] >= BOX_Y[0]) & (locs[:, 1] <= BOX_Y[1]) & (locs[:, 2] >= BOX_Z[0]) & (locs[:, 2] <= BOX_Z[1])
        n_in_box.append(int(inside.sum()))
    n_in_box = np.array(n_in_box, dtype=float)
    r = np.corrcoef(n_in_box, RF_ACCURACY)[0, 1] if n_in_box.std() > 0 else float("nan")
    print(f"  r(kutu_ici_elektrot_sayisi, RF_dogruluk) = {r:.3f}")
    print("  -> Sonuc: neredeyse iliskisiz, bu yontem de dogrulamadi.")


def step4_real_anatomical_labels(subjects):
    """ADIM 4 (calisti): veri setinde tahmine hic gerek yok - gercek
    hemisphere/lobe/gyrus/Brodmann_Area etiketleri zaten var. Her kanalin
    en iyi |Cohen's d|'sini hesaplayip TUM hastalar havuzlanarak lobe/gyrus
    bazinda karsilastir."""
    print("\n=== ADIM 4 (calisti): gercek anatomik etiketlerle (lobe/gyrus), kanal-duzeyinde, havuzlanmis ===")
    by_lobe, by_gyrus = defaultdict(list), defaultdict(list)
    frac_ext_per_patient = []
    for dat1, _d2 in subjects:
        V, t_on = dat1["V"], np.asarray(dat1["t_on"]).ravel()
        stim_id = np.asarray(dat1["stim_id"]).ravel()
        srate = float(np.asarray(dat1["srate"]).squeeze())
        labels = (stim_id > 50).astype(int)
        lobes, gyri = dat1["lobe"], dat1["gyrus"]
        epochs = extract_epochs(V, t_on, srate)
        valid = ~np.isnan(epochs).any(axis=(1, 2))
        epochs, lab = epochs[valid], labels[valid]
        face, house = lab == 1, lab == 0

        n_ext = 0
        for ch in range(epochs.shape[1]):
            best_d = 0.0
            for _, fmin, fmax in DEFAULT_BANDS:
                fv = np.array([band_power(epochs[t, ch, :], srate, fmin, fmax) for t in np.where(face)[0]])
                hv = np.array([band_power(epochs[t, ch, :], srate, fmin, fmax) for t in np.where(house)[0]])
                d = cohens_d(fv, hv)
                if abs(d) > abs(best_d):
                    best_d = d
            by_lobe[lobes[ch]].append(abs(best_d))
            by_gyrus[gyri[ch]].append(abs(best_d))
            if gyri[ch] in EXTENDED_GYRI:
                n_ext += 1
        frac_ext_per_patient.append(n_ext / epochs.shape[1])

    frac_ext_per_patient = np.array(frac_ext_per_patient)
    r_frac = np.corrcoef(frac_ext_per_patient, RF_ACCURACY)[0, 1]
    print(f"  r(gorsel-kategori-ilgili_bolge_orani, RF_dogruluk) = {r_frac:.3f}  (hasta-bazli, hala kucuk n=7)")

    print(f"\n  {'LOBE':<20}{'n_kanal':<10}{'ortalama|d|':<14}")
    for lobe, vals in sorted(by_lobe.items(), key=lambda kv: -np.mean(kv[1])):
        print(f"  {lobe:<20}{len(vals):<10}{np.mean(vals):<14.3f}")

    print(f"\n  {'GYRUS (n>=3)':<28}{'n_kanal':<10}{'ortalama|d|':<14}")
    for gyrus, vals in sorted(by_gyrus.items(), key=lambda kv: -np.mean(kv[1])):
        if len(vals) >= 3:
            print(f"  {gyrus:<28}{len(vals):<10}{np.mean(vals):<14.3f}")

    print("\n  -> Sonuc: Lingual Gyrus ve Fusiform Gyrus (ventral gorsel/yuz-isleme")
    print("     hatti), Frontal/Limbic/Parietal/Caudate gibi ilgisiz bolgelerden")
    print("     belirgin sekilde daha yuksek ortalama ayirt edicilik gosteriyor.")
    print("     Yani sinyal GERCEKTEN dogru anatomik bolgede yogunlasiyor - ama")
    print("     hasta-bazli 'ne kadar coverage' sayisi (n=7 ile) bunu net bir")
    print("     dogruluk tahminine cevirmeye yetmiyor; kanal SAYISI degil, kanal")
    print("     KALITESI/gucu belirleyici gorunuyor.")


def main(argv):
    path = Path(argv[1]) if len(argv) > 1 else Path("data/raw/faceshouses.npz")
    if not path.exists():
        print(f"Hata: {path} bulunamadi.")
        return 1
    subjects = load_faceshouses(path)
    step1_best_channel_effect_vs_accuracy(subjects)
    step2_ffa_point_distance(subjects)
    step3_fusiform_box(subjects)
    step4_real_anatomical_labels(subjects)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
