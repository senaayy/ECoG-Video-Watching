"""Epoklanmış ECoG verisinden frekans bandı güç özellikleri çıkarma.

ECoG literatüründe (bkz. Miller ve ark. çalışmaları) yüz/ev gibi görsel
kategori ayrımında en bilgilendirici bant genellikle **high-gamma (70-150 Hz,
broadband spektral değişim)** olarak öne çıkar; klasik EEG bantları
(theta/alpha/beta) de karşılaştırma için hesaplanır.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import welch

# (isim, alt_sınır_Hz, üst_sınır_Hz)
DEFAULT_BANDS: list[tuple[str, float, float]] = [
    ("delta", 1, 4),
    ("theta", 4, 8),
    ("alpha", 8, 13),
    ("beta", 13, 30),
    ("gamma", 30, 70),
    ("high_gamma", 70, 150),
]


def band_power(epoch_1d: np.ndarray, srate: float, fmin: float, fmax: float) -> float:
    """Tek bir kanalın tek bir epok'u için, [fmin, fmax) bandındaki
    ortalama gücü Welch PSD ile hesaplar."""
    nperseg = min(len(epoch_1d), max(32, int(srate // 2)))
    freqs, psd = welch(epoch_1d, fs=srate, nperseg=nperseg)
    mask = (freqs >= fmin) & (freqs < fmax)
    if not np.any(mask):
        return 0.0
    return float(trapezoid(psd[mask], freqs[mask]))


def extract_band_features(
    epochs: np.ndarray,
    srate: float,
    bands: list[tuple[str, float, float]] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Epoklardan (n_trials, n_channels, n_times) her kanal x bant
    kombinasyonu için güç özelliği çıkarır.

    Döndürür
    -------
    features : ndarray, shape (n_trials, n_channels * n_bands)
    feature_names : list[str]
        Her sütunun neyi temsil ettiği (ör. "ch03_high_gamma").
    """
    bands = bands or DEFAULT_BANDS
    n_trials, n_channels, _ = epochs.shape

    features = np.zeros((n_trials, n_channels * len(bands)))
    feature_names: list[str] = []
    for ch in range(n_channels):
        for band_name, fmin, fmax in bands:
            feature_names.append(f"ch{ch:02d}_{band_name}")

    for trial in range(n_trials):
        col = 0
        for ch in range(n_channels):
            for _, fmin, fmax in bands:
                features[trial, col] = band_power(epochs[trial, ch, :], srate, fmin, fmax)
                col += 1

    return features, feature_names


def average_band_power_by_name(
    features: np.ndarray, feature_names: list[str], bands: list[tuple[str, float, float]] | None = None
) -> dict[str, float]:
    """Tüm kanallar/denemeler üzerinden, her bant için ortalama gücü
    döndürür (özellik önemini bant bazında özetlemek için kullanışlı)."""
    bands = bands or DEFAULT_BANDS
    result: dict[str, float] = {}
    for band_name, _, _ in bands:
        cols = [i for i, name in enumerate(feature_names) if name.endswith(f"_{band_name}")]
        result[band_name] = float(np.mean(features[:, cols])) if cols else 0.0
    return result
