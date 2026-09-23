#!/usr/bin/env python3
"""Miller ve ark. (2017, J Neurophysiol)'nin ana bulgusunu kendi verimizle
kucuk olcekte tekrar uretmeyi dener: nöral yanit (broadband/high-gamma güc),
uyaran gurultusu (stim_noise, %0-100) arttikca azaliyor mu, ve algisal esigin
(~%50) UZERINDE taban seviyesine dusup 'hepsi ya da hicbiri' bir orunto mu
gosteriyor?

Bu dat2 (gurultulu tespit gorevi) kullanir - pipeline'in geri kalani sadece
dat1'i (temiz localizer) kullaniyordu.

Yontem (basitlestirilmis, orijinal makalenin TAM istatistigi degil):
  1. Her hastada, veri setinin kendi gyrus etiketlerinden Fusiform+Lingual
     kanallari sec (bkz. electrode_signal_localization.py Adim 4).
  2. Sadece GERCEK yuz uyaranlari (stim_cat==2) kullan.
  3. Uyaran sonrasi 100-400ms penceresinde high-gamma (70-150Hz) gucunu
     hesapla, uyaran-oncesi baseline'a gore.
  4. Gurultu seviyesine (0,5,...,100) gore grupla, hasta-ici z-skorla
     (hastalar arasi genlik farkini normalize etmek icin), hastalar arasi
     havuzla, her gurultu seviyesinde ortalama al.
  5. Ciz: gurultu% vs ortalama normalize high-gamma yaniti.

Kullanim:
    python analysis/noise_threshold_replication.py data/raw/faceshouses.npz
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import welch

FUSIFORM_LINGUAL = {"Fusiform Gyrus", "Lingual Gyrus"}


def extract_epochs(V, t_on, srate, tmin, tmax, baseline=(-0.2, 0.0)):
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
    return epochs - baseline_mean, sample_min


def band_power(x, srate, fmin, fmax):
    nperseg = min(len(x), max(32, int(srate // 2)))
    freqs, psd = welch(x, fs=srate, nperseg=nperseg)
    mask = (freqs >= fmin) & (freqs < fmax)
    return float(trapezoid(psd[mask], freqs[mask])) if np.any(mask) else 0.0


def main(argv):
    path = Path(argv[1]) if len(argv) > 1 else Path("data/raw/faceshouses.npz")
    npz = np.load(path, allow_pickle=True)
    subjects = [(d1, d2) for d1, d2 in npz["dat"]]

    noise_levels = np.arange(0, 101, 5)
    per_patient_curves = []  # her hastanin, gurultu seviyesine gore z-skorlu ort. yaniti

    for pi, (dat1, dat2) in enumerate(subjects, start=1):
        srate = float(np.asarray(dat1["srate"]).squeeze())  # dat2'de ayri srate yok, dat1'inkini kullan
        V, t_on = dat2["V"], np.asarray(dat2["t_on"]).ravel()
        stim_cat = np.asarray(dat2["stim_cat"]).ravel()
        stim_noise = np.asarray(dat2["stim_noise"]).ravel()
        gyri = dat2["gyrus"]

        sel_channels = [ch for ch, g in enumerate(gyri) if g in FUSIFORM_LINGUAL]
        if not sel_channels:
            print(f"Hasta {pi}: Fusiform/Lingual kanal yok, atlandi.")
            continue

        epochs, sample_min = extract_epochs(V, t_on, srate, tmin=-0.2, tmax=0.5)
        valid = ~np.isnan(epochs).any(axis=(1, 2))
        epochs = epochs[valid]
        stim_cat_v = stim_cat[valid]
        stim_noise_v = stim_noise[valid]

        # 100-400ms sonrasi pencere (epoch tmin=-0.2 oldugu icin index kaymasi var)
        idx0 = int(round((0.1 - (-0.2)) * srate))
        idx1 = int(round((0.4 - (-0.2)) * srate))

        face_mask = stim_cat_v == 2
        resp = np.zeros(face_mask.sum())
        face_noise = stim_noise_v[face_mask]
        face_epochs = epochs[face_mask]
        for ti in range(face_epochs.shape[0]):
            vals = [band_power(face_epochs[ti, ch, idx0:idx1], srate, 70, 150) for ch in sel_channels]
            resp[ti] = np.mean(vals)

        if resp.std() == 0 or len(resp) < 10:
            print(f"Hasta {pi}: yeterli/degisken veri yok, atlandi.")
            continue
        resp_z = (resp - resp.mean()) / resp.std()

        curve = np.full(len(noise_levels), np.nan)
        for li, nl in enumerate(noise_levels):
            m = face_noise == nl
            if m.sum() > 0:
                curve[li] = resp_z[m].mean()
        per_patient_curves.append(curve)
        print(f"Hasta {pi}: {len(sel_channels)} Fusiform/Lingual kanal, {face_mask.sum()} yuz-denemesi kullanildi.")

    per_patient_curves = np.array(per_patient_curves)
    pooled_mean = np.nanmean(per_patient_curves, axis=0)
    pooled_sem = np.nanstd(per_patient_curves, axis=0) / np.sqrt(np.sum(~np.isnan(per_patient_curves), axis=0))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(noise_levels, pooled_mean, yerr=pooled_sem, marker="o", color="#2C5F8A", ecolor="#9db6c9", capsize=3)
    ax.axvline(50, color="gray", linestyle="--", alpha=0.7, label="~%50 algisal esik (Miller ve ark. 2017)")
    ax.set_xlabel("Uyaran gurultusu (%)")
    ax.set_ylabel("Normalize high-gamma yaniti (z-skor)")
    ax.set_title("Yuz uyaranlarina high-gamma yaniti vs gurultu seviyesi\n(Fusiform+Lingual kanallar, 7 hasta havuzlanmis)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_dir = Path(__file__).resolve().parent
    out_path = out_dir / "noise_threshold_replication.png"
    plt.savefig(out_path, dpi=200)
    print(f"\nKaydedildi: {out_path}")

    with open(out_dir / "noise_threshold_replication_results.txt", "w", encoding="utf-8") as f:
        f.write("gurultu% ortalama_z_yanit sem\n")
        for nl, m, s in zip(noise_levels, pooled_mean, pooled_sem):
            f.write(f"{nl} {m:.3f} {s:.3f}\n")
    print(f"Kaydedildi: {out_dir / 'noise_threshold_replication_results.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
