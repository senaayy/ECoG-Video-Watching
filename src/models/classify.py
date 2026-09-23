"""Yüz vs ev sınıflandırması için model tanımları (LDA / SVM / RF).

Her model, StandardScaler + sınıflandırıcıdan oluşan bir sklearn Pipeline
olarak döner; böylece cross-validation sırasında ölçekleme her fold'da
yalnızca eğitim verisiyle öğrenilir (veri sızıntısı olmaz).
"""
from __future__ import annotations

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def make_lda() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LinearDiscriminantAnalysis()),
    ])


def make_svm(kernel: str = "linear", C: float = 1.0) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel=kernel, C=C, probability=True, random_state=42)),
    ])


def make_rf(n_estimators: int = 300) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),  # RF için şart değil ama pipeline tutarlılığı için
        ("clf", RandomForestClassifier(
            n_estimators=n_estimators, random_state=42, n_jobs=-1
        )),
    ])


MODEL_FACTORIES = {
    "LDA": make_lda,
    "SVM (linear)": make_svm,
    "Random Forest": make_rf,
}
