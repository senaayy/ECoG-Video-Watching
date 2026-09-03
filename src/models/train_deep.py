"""EEGNet (Lawhern ve ark. 2018) ile derin öğrenme taban çizgisi --
face vs house, hasta-bazlı (within-subject).

Girdi, `extract_features.py`'deki spektral özellikler yerine **ham
(0.5-200 Hz bandpass edilmiş) zaman-serisi penceresidir**: her event için
onset'ten itibaren 400ms (@1000Hz = 400 örnek), yalnızca ROI
(temporal/fusiform/occipital) kanalları.

Mimari (Lawhern 2018 EEGNet, F1=8, D=2, F2=16):
    Block1: temporal conv (1xkernel_length) -> BatchNorm
            -> depthwise conv (n_channels x1, D kat) -> BatchNorm -> ELU
            -> AvgPool(1,4) -> Dropout
    Block2: separable conv (depthwise 1x16 + pointwise 1x1) -> BatchNorm
            -> ELU -> AvgPool(1,8) -> Dropout
    Classifier: Linear -> 2 sınıf (face/house)

Değerlendirme: `train_baselines.py` ile birebir aynı zamansal-blok
GroupKFold(n_splits=5) şeması; her fold içinde ayrıca erken durdurma için
train'den ayrılan küçük bir validasyon alt kümesi kullanılır (test fold'a
hiç dokunulmaz). Class-weighted cross-entropy kaybı kullanılır.

Hesaplama maliyeti nedeniyle **14 hastanın tamamı değil**, temsili bir alt
küme üzerinde çalıştırılır: güçlü sinyal (`ap`, `zt`), orta (`aa`), zayıf
(`fp`, `rr`) -- bkz. 04_feature_extraction.ipynb Bölüm 5 / 05_classification
kategorileri.

Kullanım:
    python src/models/train_deep.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold, train_test_split

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.preprocessing.preprocess import PROCESSED_DATA_DIR
from src.features.extract_features import infer_event_duration_sec
from src.models.train_baselines import make_time_block_groups, N_SPLITS

REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"
REPRESENTATIVE_PATIENTS = ["ap", "zt", "aa", "fp", "rr"]  # güçlü: ap,zt / orta: aa / zayıf: fp,rr

RANDOM_STATE = 42
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
BATCH_SIZE = 32
LEARNING_RATE = 1e-3


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Veri yükleme: ham zaman-serisi pencereleri (spektral özellikler değil)
# ---------------------------------------------------------------------------

def load_patient_windows(
    patient_id: str, processed_dir: Path = PROCESSED_DATA_DIR
) -> dict:
    """Bir hasta için, her face/house event'inin 0-400ms ham (bandpass
    edilmiş) sinyal penceresini yükler. Yalnızca ROI (temporal/fusiform/
    occipital) kanalları kullanılır.

    Döndürür: X (n_events x n_channels x n_timepoints, float32),
    y (n_events,, 1=face/0=house), time_sec, channels (processed_channel),
    srate.
    """
    patient_dir = processed_dir / patient_id
    npz = np.load(patient_dir / f"{patient_id}_processed.npz")
    data = npz["data"]
    srate = int(npz["srate"])

    anat_df = pd.read_csv(patient_dir / f"{patient_id}_anatomical_labels.csv")
    roi_mask = anat_df["is_temporal"] | anat_df["is_fusiform"] | anat_df["is_occipital"]
    roi_rows = anat_df.loc[roi_mask].sort_values("processed_channel")
    roi_channels = roi_rows["processed_channel"].to_numpy()

    data_roi = data[:, roi_channels]

    labels_df = pd.read_parquet(patient_dir / "labels.parquet")
    window_sec = infer_event_duration_sec(labels_df)
    window_samples = int(round(window_sec * srate))

    events_df = labels_df[labels_df["class"].isin(["face", "house"])].reset_index(drop=True)

    X_list, y_list, t_list = [], [], []
    for _, row in events_df.iterrows():
        start = int(row["sample_idx"])
        end = start + window_samples
        if end > data_roi.shape[0]:
            continue
        X_list.append(data_roi[start:end].T)  # (n_channels, n_timepoints)
        y_list.append(1 if row["class"] == "face" else 0)
        t_list.append(row["time_sec"])

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    time_sec = np.array(t_list)

    return {
        "X": X,
        "y": y,
        "time_sec": time_sec,
        "channels": roi_channels,
        "srate": srate,
        "window_samples": window_samples,
    }


# ---------------------------------------------------------------------------
# EEGNet mimarisi (Lawhern ve ark. 2018)
# ---------------------------------------------------------------------------

class EEGNet(nn.Module):
    """EEGNet (Lawhern ve ark. 2018), F1/D/F2 hiperparametreleriyle.

    Girdi: (batch, 1, n_channels, n_timepoints)
    Çıktı: (batch, n_classes) -- ham logit'ler (CrossEntropyLoss ile kullanılır)
    """

    def __init__(
        self,
        n_channels: int,
        n_timepoints: int,
        n_classes: int = 2,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: int = 64,
        dropout: float = 0.5,
        pool1: int = 4,
        pool2: int = 8,
    ):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_length), padding="same", bias=False),
            nn.BatchNorm2d(F1),
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, pool1)),
            nn.Dropout(dropout),
        )
        self.separable = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding="same", groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, pool2)),
            nn.Dropout(dropout),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_timepoints)
            flat_size = self._forward_features(dummy).numel()
        self.classifier = nn.Linear(flat_size, n_classes)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.depthwise(x)
        x = self.separable(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_features(x)
        x = x.flatten(1)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Eğitim / değerlendirme
# ---------------------------------------------------------------------------

def _standardize(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kanal bazlı z-score normalizasyonu; istatistikler yalnızca train'den
    hesaplanır (sızıntı yok)."""
    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std = X_train.std(axis=(0, 2), keepdims=True) + 1e-8
    return (X_train - mean) / std, (X_test - mean) / std


def train_one_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
    random_state: int = RANDOM_STATE,
    verbose: bool = False,
) -> dict:
    """Bir fold için: train'den erken durdurma amaçlı validasyon ayır,
    class-weighted CE ile eğit, en iyi val-loss ağırlıklarını test fold'unda
    değerlendir."""
    X_train, X_test = _standardize(X_train, X_test)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=random_state
    )

    n_channels, n_timepoints = X_train.shape[1], X_train.shape[2]
    model = EEGNet(n_channels=n_channels, n_timepoints=n_timepoints).to(device)

    class_counts = np.bincount(y_tr, minlength=2).astype(np.float32)
    class_weights = torch.tensor(class_counts.sum() / (2 * np.maximum(class_counts, 1)), dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    def to_tensor(X, y=None):
        Xt = torch.from_numpy(X).unsqueeze(1).float().to(device)  # (N,1,C,T)
        if y is None:
            return Xt
        yt = torch.from_numpy(y).long().to(device)
        return Xt, yt

    X_tr_t, y_tr_t = to_tensor(X_tr, y_tr)
    X_val_t, y_val_t = to_tensor(X_val, y_val)
    X_test_t, y_test_t = to_tensor(X_test, y_test)

    n_train = len(X_tr_t)
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(n_train)
        for start in range(0, n_train, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            xb, yb = X_tr_t[idx], y_tr_t[idx]
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = criterion(val_logits, y_val_t).item()

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and epoch % 10 == 0:
            print(f"    epoch {epoch}: val_loss={val_loss:.4f} (best={best_val_loss:.4f})")

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            if verbose:
                print(f"    erken durduruldu: epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        test_logits = model(X_test_t)
        test_probs = torch.softmax(test_logits, dim=1)[:, 1].cpu().numpy()
        test_preds = test_logits.argmax(dim=1).cpu().numpy()

    return {
        "y_true": y_test,
        "y_pred": test_preds,
        "y_prob": test_probs,
        "n_epochs_trained": epoch + 1,
    }


def run_patient_deep(
    patient_id: str,
    processed_dir: Path = PROCESSED_DATA_DIR,
    n_splits: int = N_SPLITS,
    random_state: int = RANDOM_STATE,
    verbose: bool = True,
) -> pd.DataFrame:
    """Bir hasta için EEGNet'i zamansal-blok GroupKFold ile değerlendirir."""
    device = get_device()
    dataset = load_patient_windows(patient_id, processed_dir)
    X, y, time_sec = dataset["X"], dataset["y"], dataset["time_sec"]
    groups = make_time_block_groups(time_sec, n_splits)

    gkf = GroupKFold(n_splits=n_splits)
    all_true, all_pred, all_prob = [], [], []

    t0 = time.time()
    for fold_i, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        torch.manual_seed(random_state + fold_i)
        result = train_one_fold(
            X[train_idx], y[train_idx], X[test_idx], y[test_idx], device, random_state, verbose=False
        )
        all_true.append(result["y_true"])
        all_pred.append(result["y_pred"])
        all_prob.append(result["y_prob"])
        if verbose:
            print(f"  {patient_id} fold {fold_i}: n_epochs={result['n_epochs_trained']}")

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    y_prob = np.concatenate(all_prob)
    elapsed = time.time() - t0

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = np.nan

    if verbose:
        print(f"  {patient_id} / EEGNet: acc={accuracy:.3f} f1={macro_f1:.3f} "
              f"auc={roc_auc:.3f} ({elapsed:.1f}s, device={device})")

    return pd.DataFrame([{
        "patient": patient_id,
        "model": "EEGNet",
        "n_roi_channels": X.shape[1],
        "n_events": len(y),
        "n_face": int(y.sum()),
        "n_house": int((1 - y).sum()),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "roc_auc": roc_auc,
        "permutation_true_accuracy": np.nan,
        "permutation_p_value": np.nan,
        "n_permutations": 0,
        "elapsed_sec": elapsed,
    }])


def run_representative_patients(
    patients: Optional[list[str]] = None,
    processed_dir: Path = PROCESSED_DATA_DIR,
    n_splits: int = N_SPLITS,
    random_state: int = RANDOM_STATE,
    out_path: Path = REPORTS_DIR / "deep_results.csv",
    verbose: bool = True,
) -> pd.DataFrame:
    """Temsili hasta alt kümesi için `run_patient_deep()`'i çalıştırır,
    birleşik sonucu `reports/deep_results.csv`'e kaydeder."""
    if patients is None:
        patients = REPRESENTATIVE_PATIENTS

    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for patient_id in patients:
        if verbose:
            print(f"[{patient_id}] EEGNet çalışıyor...")
        all_rows.append(run_patient_deep(patient_id, processed_dir, n_splits, random_state, verbose))

    results_df = pd.concat(all_rows, ignore_index=True)
    results_df.to_csv(out_path, index=False)
    if verbose:
        print(f"Kaydedildi: {out_path}")
    return results_df


if __name__ == "__main__":
    df = run_representative_patients()
    print(df.to_string(index=False))
