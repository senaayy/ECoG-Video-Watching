#!/usr/bin/env python3
"""
Soru: LDA'nin dusuk performansi (0.531), kovaryans matrisini az orneklemle
(n=300 deneme) guvenilir tahmin edememesinden mi kaynaklaniyor? Bunu test
etmek icin ayni pipeline'i, SADECE LDA'nin kovaryans tahminini "shrinkage"
(buzme/regularizasyon) ile yumusatarak tekrar calistiriyoruz. Geri kalan her
sey (epoklama, bant gucu ozellikleri, StandardScaler, Stratified 5-Fold)
run_ecog_pipeline_standalone.py ile BIREBIR AYNI.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import welch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

def load_faceshouses(path):
    npz = np.load(path, allow_pickle=True)
    subjects = npz["dat"]
    return [(dat1, dat2) for dat1, dat2 in subjects]

def extract_epochs(V, t_on, srate, tmin=-0.2, tmax=0.4, baseline=(-0.2, 0.0)):
    V = np.asarray(V); t_on = np.asarray(t_on).astype(int)
    n_channels = V.shape[1]
    sample_min = int(round(tmin * srate)); sample_max = int(round(tmax * srate))
    n_times = sample_max - sample_min
    epochs = np.full((len(t_on), n_channels, n_times), np.nan, dtype=float)
    for i, onset in enumerate(t_on):
        start, end = onset + sample_min, onset + sample_max
        if start < 0 or end > V.shape[0]:
            continue
        epochs[i] = V[start:end, :].T
    b0 = int(round((baseline[0] - tmin) * srate)); b1 = int(round((baseline[1] - tmin) * srate))
    b1 = max(b1, b0 + 1)
    epochs = epochs - np.nanmean(epochs[:, :, b0:b1], axis=2, keepdims=True)
    return epochs

def drop_invalid_epochs(epochs, labels):
    valid = ~np.isnan(epochs).any(axis=(1, 2))
    return epochs[valid], labels[valid]

DEFAULT_BANDS = [("delta",1,4),("theta",4,8),("alpha",8,13),("beta",13,30),("gamma",30,70),("high_gamma",70,150)]

def band_power(epoch_1d, srate, fmin, fmax):
    nperseg = min(len(epoch_1d), max(32, int(srate // 2)))
    freqs, psd = welch(epoch_1d, fs=srate, nperseg=nperseg)
    mask = (freqs >= fmin) & (freqs < fmax)
    if not np.any(mask):
        return 0.0
    return float(trapezoid(psd[mask], freqs[mask]))

def extract_band_features(epochs, srate, bands=None):
    bands = bands or DEFAULT_BANDS
    n_trials, n_channels, _ = epochs.shape
    features = np.zeros((n_trials, n_channels * len(bands)))
    for trial in range(n_trials):
        col = 0
        for ch in range(n_channels):
            for _, fmin, fmax in bands:
                features[trial, col] = band_power(epochs[trial, ch, :], srate, fmin, fmax)
                col += 1
    return features

def prepare_subject(dat1):
    V = dat1["V"]; t_on = np.asarray(dat1["t_on"]).ravel()
    stim_id = np.asarray(dat1["stim_id"]).ravel()
    srate = float(np.asarray(dat1["srate"]).squeeze())
    labels = (stim_id > 50).astype(int)
    epochs = extract_epochs(V, t_on, srate)
    epochs, labels = drop_invalid_epochs(epochs, labels)
    features = extract_band_features(epochs, srate)
    return features, labels

def make_lda_plain():
    return Pipeline([("scaler", StandardScaler()), ("clf", LinearDiscriminantAnalysis())])

def make_lda_shrinkage():
    return Pipeline([("scaler", StandardScaler()),
                      ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))])

def evaluate(model, X, y, seed=42):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    y_pred = cross_val_predict(model, X, y, cv=cv, method="predict")
    return accuracy_score(y, y_pred)

def main(argv):
    path = Path(argv[1])
    subjects = load_faceshouses(path)
    print(f"{len(subjects)} hasta yuklendi.\n")

    plain_accs, shrink_accs = [], []
    for i, (dat1, _dat2) in enumerate(subjects, start=1):
        X, y = prepare_subject(dat1)
        n_features = X.shape[1]
        acc_plain = evaluate(make_lda_plain(), X, y)
        acc_shrink = evaluate(make_lda_shrinkage(), X, y)
        plain_accs.append(acc_plain); shrink_accs.append(acc_shrink)
        print(f"Hasta {i} | n_ozellik={n_features:4d} n_deneme={len(y):4d} | "
              f"LDA (duz)={acc_plain:.3f}  LDA (shrinkage)={acc_shrink:.3f}  "
              f"fark={acc_shrink-acc_plain:+.3f}")

    print("\n=== OZET (hastalar arasi ortalama) ===")
    print(f"LDA (duz):       {np.mean(plain_accs):.3f} +/- {np.std(plain_accs):.3f}")
    print(f"LDA (shrinkage):  {np.mean(shrink_accs):.3f} +/- {np.std(shrink_accs):.3f}")

if __name__ == "__main__":
    raise SystemExit(main(sys.argv))