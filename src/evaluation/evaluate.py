"""Stratified K-Fold cross-validation ile model değerlendirme, ve
sonuçları özetleyen/görselleştiren yardımcı fonksiyonlar.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict


def evaluate_model(model, X: np.ndarray, y: np.ndarray, n_splits: int = 5, seed: int = 42) -> dict:
    """Stratified K-Fold ile out-of-fold tahminler üretir (veri sızıntısı
    olmadan), doğruluk ve ROC-AUC hesaplar.

    `cross_val_predict` kullanmamızın sebebi: her deneme yalnızca kendi
    fold'unun dışında eğitilmiş bir modelden tahmin ediliyor — bu, P300
    projesindeki Stratified K-Fold değerlendirmesiyle aynı prensip.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y_pred = cross_val_predict(model, X, y, cv=cv, method="predict")

    result = {
        "accuracy": accuracy_score(y, y_pred),
        "y_true": y,
        "y_pred": y_pred,
        "confusion_matrix": confusion_matrix(y, y_pred),
    }

    try:
        y_proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
        result["roc_auc"] = roc_auc_score(y, y_proba)
        result["y_proba"] = y_proba
    except (AttributeError, ValueError):
        result["roc_auc"] = None

    return result


def plot_confusion_matrix(result: dict, class_names: tuple[str, str], ax=None):
    disp = ConfusionMatrixDisplay(
        confusion_matrix=result["confusion_matrix"], display_labels=class_names
    )
    return disp.plot(ax=ax, cmap="Blues", colorbar=False)


def plot_roc_curve(result: dict, ax=None, label: str = ""):
    if result.get("y_proba") is None:
        return None
    return RocCurveDisplay.from_predictions(
        result["y_true"], result["y_proba"], ax=ax, name=label
    )
