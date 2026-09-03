"""Hasta-bazlı (within-subject) face vs house ikili sınıflandırma taban
çizgileri (baselines) — Miller 2019, faces_basic.

`data/processed/<hasta>/features.npz` (özellik matrisi) ve `labels.parquet`
(event zamanları) kullanılarak, **her hasta için ayrı ayrı**, face vs house
ikili sınıflandırması yapılır:

    - Zamansal grup çapraz doğrulama: GroupKFold(n_splits=5), gruplar
      event zamanına göre 5 eşit genişlikte zaman bloğu (baştan sona
      sıralı). Aynı bloktaki event'ler hem train hem test'te bulunmaz --
      bu, deneyin doğal zamansal akışındaki otokorelasyonun (örn. dikkat/
      yorgunluk düzeyi zamanla değişimi, elektrot empedansının kayması)
      sızıntı (leakage) yaratmasını önler.
    - Modeller: LDA (shrinkage='auto'), Lineer SVM (class_weight=
      'balanced'), Random Forest.
    - Metrikler: accuracy, macro-F1, ROC-AUC (fold ortalaması).
    - Permütasyon testi (`sklearn.model_selection.permutation_test_score`,
      n_permutations=200): gerçek accuracy skorunun, etiketler rastgele
      karıştırıldığında elde edilen null dağılıma göre p-değeri.

Kullanım (kütüphane olarak):
    from src.models.train_baselines import run_patient, run_all_patients
    result = run_patient("ap")

Kullanım (komut satırından, tüm hastalar + tüm modeller):
    python src/models/train_baselines.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold, permutation_test_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.preprocessing.preprocess import PROCESSED_DATA_DIR, list_available_patients

N_SPLITS = 5
N_PERMUTATIONS = 200
RANDOM_STATE = 42
REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"


def load_patient_dataset(patient_id: str, processed_dir: Path = PROCESSED_DATA_DIR) -> dict:
    """`features.npz` + `labels.parquet`'i yükler, kanal x özellik matrisini
    düzleştirilmiş bir tasarım matrisine (`X`) ve ikili etikete (`y`,
    1=face, 0=house) çevirir. `time_sec`, zamansal grup bölmesi için
    döndürülür."""
    patient_dir = processed_dir / patient_id
    feat_npz = np.load(patient_dir / "features.npz", allow_pickle=True)
    features = feat_npz["features"]
    event_labels = feat_npz["event_labels"]
    channels = feat_npz["channels"]
    area_labels = feat_npz["area_labels"]
    feature_names = list(feat_npz["feature_names"])

    labels_df = pd.read_parquet(patient_dir / "labels.parquet")
    events_df = (
        labels_df[labels_df["class"].isin(["face", "house"])]
        .sort_values("sample_idx")
        .reset_index(drop=True)
    )

    n = features.shape[0]
    if len(events_df) != n:
        # extract_features.py yalnızca kayıt sonuna taşan (trailing) event'leri
        # atlar; event sırası korunduğu için baştan n satır alarak hizalanır.
        events_df = events_df.iloc[:n].reset_index(drop=True)

    if not (events_df["class"].to_numpy() == event_labels).all():
        raise ValueError(
            f"{patient_id}: labels.parquet ile features.npz etiket sırası uyuşmuyor."
        )

    X = features.reshape(n, -1)
    y = (event_labels == "face").astype(int)
    time_sec = events_df["time_sec"].to_numpy()

    return {
        "X": X,
        "y": y,
        "time_sec": time_sec,
        "channels": channels,
        "area_labels": area_labels,
        "feature_names": feature_names,
        "n_roi_channels": len(channels),
    }


def make_time_block_groups(time_sec: np.ndarray, n_blocks: int = N_SPLITS) -> np.ndarray:
    """`time_sec`'i `n_blocks` eşit genişlikte zaman aralığına böler.
    Aynı gruptaki (zaman bloğundaki) event'ler GroupKFold ile hem train hem
    test'te birden bulunmaz."""
    groups = pd.cut(pd.Series(time_sec), bins=n_blocks, labels=False, include_lowest=True)
    return groups.to_numpy().astype(int)


def build_models(random_state: int = RANDOM_STATE) -> dict[str, Pipeline]:
    """LDA / Lineer SVM / Random Forest pipeline'larını döndürür. LDA ve SVM
    için `StandardScaler` içerilir (fold içinde yalnızca train'e fit edilir,
    Pipeline sayesinde sızıntı olmaz); Random Forest ölçeklemeye ihtiyaç
    duymaz."""
    return {
        "LDA": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ]),
        "Linear_SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="linear", class_weight="balanced", random_state=random_state)),
        ]),
        "RandomForest": Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=100,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=1,  # permutation_test_score zaten dış katmanda n_jobs=-1 ile paralelleştiriyor;
                           # burada da -1 vermek iç içe (nested) joblib paralelliğine ve büyük yavaşlamaya yol açıyor
            )),
        ]),
    }


def _decision_scores(fitted_pipeline: Pipeline, X: np.ndarray) -> np.ndarray:
    """ROC-AUC için karar skoru: `decision_function` varsa onu, yoksa
    `predict_proba`'nın pozitif sınıf olasılığını kullanır."""
    clf = fitted_pipeline.named_steps["clf"]
    if hasattr(clf, "decision_function"):
        return fitted_pipeline.decision_function(X)
    return fitted_pipeline.predict_proba(X)[:, 1]


def cross_validated_metrics(
    pipeline: Pipeline, X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int = N_SPLITS
) -> dict:
    """GroupKFold ile accuracy, macro-F1, ROC-AUC hesaplar (fold ortalaması).
    Her fold'da pipeline yeniden fit edilir (sızıntı yok)."""
    gkf = GroupKFold(n_splits=n_splits)
    accs, f1s, aucs = [], [], []
    for train_idx, test_idx in gkf.split(X, y, groups):
        model = clone(pipeline)
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[test_idx])
        scores = _decision_scores(model, X[test_idx])

        accs.append(accuracy_score(y[test_idx], y_pred))
        f1s.append(f1_score(y[test_idx], y_pred, average="macro"))
        try:
            aucs.append(roc_auc_score(y[test_idx], scores))
        except ValueError:
            aucs.append(np.nan)

    return {
        "accuracy": float(np.mean(accs)),
        "macro_f1": float(np.mean(f1s)),
        "roc_auc": float(np.nanmean(aucs)),
    }


def run_patient(
    patient_id: str,
    processed_dir: Path = PROCESSED_DATA_DIR,
    n_permutations: int = N_PERMUTATIONS,
    n_splits: int = N_SPLITS,
    random_state: int = RANDOM_STATE,
    verbose: bool = False,
) -> pd.DataFrame:
    """Bir hasta için tüm modelleri çalıştırır (CV metrikleri + permütasyon
    testi). Döndürür: her satırı bir (hasta, model) kombinasyonu olan
    DataFrame."""
    dataset = load_patient_dataset(patient_id, processed_dir)
    X, y, time_sec = dataset["X"], dataset["y"], dataset["time_sec"]
    groups = make_time_block_groups(time_sec, n_splits)

    models = build_models(random_state)
    rows = []

    for model_name, pipeline in models.items():
        t0 = time.time()

        metrics = cross_validated_metrics(pipeline, X, y, groups, n_splits)

        gkf = GroupKFold(n_splits=n_splits)
        true_score, perm_scores, pvalue = permutation_test_score(
            clone(pipeline),
            X,
            y,
            groups=groups,
            cv=gkf,
            scoring="accuracy",
            n_permutations=n_permutations,
            n_jobs=-1,
            random_state=random_state,
        )

        elapsed = time.time() - t0
        if verbose:
            print(f"  {patient_id} / {model_name}: acc={metrics['accuracy']:.3f} "
                  f"f1={metrics['macro_f1']:.3f} auc={metrics['roc_auc']:.3f} "
                  f"p={pvalue:.4f} ({elapsed:.1f}s)")

        rows.append({
            "patient": patient_id,
            "model": model_name,
            "n_roi_channels": dataset["n_roi_channels"],
            "n_events": len(y),
            "n_face": int(y.sum()),
            "n_house": int((1 - y).sum()),
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "roc_auc": metrics["roc_auc"],
            "permutation_true_accuracy": float(true_score),
            "permutation_p_value": float(pvalue),
            "n_permutations": n_permutations,
            "elapsed_sec": elapsed,
        })

    return pd.DataFrame(rows)


def run_all_patients(
    patients: Optional[list[str]] = None,
    processed_dir: Path = PROCESSED_DATA_DIR,
    n_permutations: int = N_PERMUTATIONS,
    n_splits: int = N_SPLITS,
    random_state: int = RANDOM_STATE,
    out_path: Path = REPORTS_DIR / "baseline_results.csv",
    verbose: bool = True,
) -> pd.DataFrame:
    """Tüm (veya verilen) hastalar için `run_patient()`'ı çalıştırır, birleşik
    sonucu `reports/baseline_results.csv` olarak kaydeder."""
    if patients is None:
        patients = list_available_patients()

    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for patient_id in patients:
        if verbose:
            print(f"[{patient_id}] çalışıyor...")
        patient_df = run_patient(
            patient_id, processed_dir, n_permutations, n_splits, random_state, verbose
        )
        all_rows.append(patient_df)

    results_df = pd.concat(all_rows, ignore_index=True)
    results_df.to_csv(out_path, index=False)
    if verbose:
        print(f"Kaydedildi: {out_path}")
    return results_df


if __name__ == "__main__":
    df = run_all_patients()
    print(df.to_string(index=False))
