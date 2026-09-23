"""dat1/dat2 şemasındaki sürekli ECoG voltaj verisini (V: zaman x kanal),
uyaran başlangıç zamanlarına (t_on) göre epoklara (denemelere) ayıran
ön işleme fonksiyonları.

Miller ECoG verisi zaten notch-filtrelenmiş ve z-score normalize edilmiş
geliyor (bkz. data/raw/README.md), bu yüzden burada ekstra filtreleme
yapmıyoruz — sadece epoklama + isteğe bağlı baseline düzeltmesi.
"""
from __future__ import annotations

import numpy as np


def extract_epochs(
    V: np.ndarray,
    t_on: np.ndarray,
    srate: float,
    *,
    tmin: float = -0.2,
    tmax: float = 0.4,
    baseline: tuple[float, float] | None = (-0.2, 0.0),
) -> np.ndarray:
    """Sürekli veriyi (V) her uyaran başlangıcı (t_on) etrafında epoklara ayırır.

    Parametreler
    ----------
    V : ndarray, shape (n_samples, n_channels)
        Sürekli ECoG voltaj verisi.
    t_on : ndarray, shape (n_trials,)
        Her uyaranın başlangıç zamanı (örnek indeksi, samples).
    srate : float
        Örnekleme hızı (Hz).
    tmin, tmax : float
        Uyaran başlangıcına göre epok penceresi (saniye). Örn. tmin=-0.2,
        tmax=0.4 -> uyarandan 200ms önce ile 400ms sonrası arası.
    baseline : (float, float) veya None
        Baseline düzeltmesi için pencere (saniye, uyaran başlangıcına göre).
        None ise baseline düzeltmesi yapılmaz.

    Döndürür
    -------
    epochs : ndarray, shape (n_trials, n_channels, n_times)
        Baseline-düzeltilmiş (istenirse) epok verisi.
    """
    V = np.asarray(V)
    t_on = np.asarray(t_on).astype(int)
    n_channels = V.shape[1]

    sample_min = int(round(tmin * srate))
    sample_max = int(round(tmax * srate))
    n_times = sample_max - sample_min

    n_trials = len(t_on)
    epochs = np.full((n_trials, n_channels, n_times), np.nan, dtype=float)

    for i, onset in enumerate(t_on):
        start = onset + sample_min
        end = onset + sample_max
        if start < 0 or end > V.shape[0]:
            # Kayıt sınırlarının dışına taşan denemeler NaN olarak kalır,
            # sonraki adımda (özellik çıkarımı öncesi) elenmeli.
            continue
        epochs[i] = V[start:end, :].T  # (n_channels, n_times)

    if baseline is not None:
        b0 = int(round((baseline[0] - tmin) * srate))
        b1 = int(round((baseline[1] - tmin) * srate))
        b1 = max(b1, b0 + 1)
        baseline_mean = np.nanmean(epochs[:, :, b0:b1], axis=2, keepdims=True)
        epochs = epochs - baseline_mean

    return epochs


def drop_invalid_epochs(epochs: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kayıt sınırları dışına taştığı için NaN kalan epokları (ve
    karşılık gelen etiketleri) eler."""
    valid = ~np.isnan(epochs).any(axis=(1, 2))
    return epochs[valid], labels[valid]
