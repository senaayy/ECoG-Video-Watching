"""Model değerlendirme (Miller 2019, faces_basic).

`reports/baseline_results.csv`'e (bkz. `src/models/train_baselines.py`)
dayanarak en iyi performans gösteren modeli seçer ve şu analizleri yapar:

    1. Hasta bazlı, out-of-fold (GroupKFold) confusion matrix'ler
       (normalize edilmiş) ve bunların 14 hastalık bir grid figürde özeti.
    2. Kanal bazlı feature importance (RF) / |katsayı| (LDA, SVM)
       çıkarımı, `get_anatomical_labels()` çıktısıyla eşleştirme, ve
       04_feature_extraction.ipynb Bölüm 5'teki t-test sonuçlarıyla
       karşılaştırma.
    3. `.nii` MRI hacminden (marching cubes ile) yaklaşık bir kortikal/kafa
       yüzeyi çıkarıp elektrotları bu yüzey üzerine feature-importance
       renk kodlamasıyla bindiren 3D görselleştirme.

Kullanım:
    from src.evaluation.evaluate import select_best_model, ...
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.preprocessing.preprocess import PROCESSED_DATA_DIR, RAW_DATA_DIR, list_available_patients
from src.models.train_baselines import (
    build_models,
    load_patient_dataset,
    make_time_block_groups,
    N_SPLITS,
)

REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"
BRAINS_DIR = RAW_DATA_DIR.parent / "brains"
LOCS_DIR = RAW_DATA_DIR.parent / "locs"

CLASS_NAMES = ["house", "face"]  # y=0 -> house, y=1 -> face


# ---------------------------------------------------------------------------
# 1. En iyi model seçimi ve out-of-fold confusion matrix'ler
# ---------------------------------------------------------------------------

def select_best_model(results_csv: Path = REPORTS_DIR / "baseline_results.csv") -> tuple[str, pd.DataFrame]:
    """`baseline_results.csv`'i yükler, ortalama accuracy'ye göre en iyi
    modeli seçer. Döndürür: (model_adı, results_df)."""
    results_df = pd.read_csv(results_csv)
    mean_acc = results_df.groupby("model")["accuracy"].mean().sort_values(ascending=False)
    best_model = mean_acc.index[0]
    return best_model, results_df


def get_oof_predictions(
    patient_id: str, model_name: str, processed_dir: Path = PROCESSED_DATA_DIR, n_splits: int = N_SPLITS
) -> tuple[np.ndarray, np.ndarray]:
    """Bir hasta için, seçilen modelle out-of-fold (GroupKFold) tahminleri
    üretir -- her event tam olarak bir kez, kendi zaman bloğu dışındaki
    verilerle eğitilmiş bir modelden tahmin edilir (sızıntısız)."""
    from sklearn.base import clone
    from sklearn.model_selection import GroupKFold

    dataset = load_patient_dataset(patient_id, processed_dir)
    X, y, time_sec = dataset["X"], dataset["y"], dataset["time_sec"]
    groups = make_time_block_groups(time_sec, n_splits)

    pipeline = build_models()[model_name]
    gkf = GroupKFold(n_splits=n_splits)

    y_true = np.empty_like(y)
    y_pred = np.empty_like(y)
    for train_idx, test_idx in gkf.split(X, y, groups):
        model = clone(pipeline)
        model.fit(X[train_idx], y[train_idx])
        y_true[test_idx] = y[test_idx]
        y_pred[test_idx] = model.predict(X[test_idx])

    return y_true, y_pred


def compute_all_confusion_matrices(
    model_name: str, patients: Optional[list[str]] = None, processed_dir: Path = PROCESSED_DATA_DIR
) -> dict[str, np.ndarray]:
    """Tüm hastalar için normalize edilmiş (satır bazlı, 'true') 2x2
    confusion matrix'leri hesaplar. Döndürür: {patient_id: cm_normalized}."""
    from sklearn.metrics import confusion_matrix

    if patients is None:
        patients = list_available_patients()

    cms = {}
    for patient_id in patients:
        y_true, y_pred = get_oof_predictions(patient_id, model_name, processed_dir)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1], normalize="true")
        cms[patient_id] = cm
    return cms


def confusion_summary_table(cms: dict[str, np.ndarray]) -> pd.DataFrame:
    """Her hasta için house->face ve face->house karışma oranlarını
    (yanlış sınıflandırma oranı) bir tabloya döker."""
    rows = []
    for patient_id, cm in cms.items():
        rows.append({
            "patient": patient_id,
            "house_correct_rate": cm[0, 0],
            "house_misclassified_as_face": cm[0, 1],
            "face_correct_rate": cm[1, 1],
            "face_misclassified_as_house": cm[1, 0],
        })
    return pd.DataFrame(rows).sort_values("patient").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Kanal bazlı feature importance / |katsayı| çıkarımı
# ---------------------------------------------------------------------------

def _get_flat_importance(fitted_pipeline) -> np.ndarray:
    """Fit edilmiş bir pipeline'dan düz (n_channels*n_features,) boyutunda
    bir önem skoru dizisi çıkarır: RF için `feature_importances_`,
    lineer modeller için `|coef_|`."""
    clf = fitted_pipeline.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        return np.asarray(clf.feature_importances_)
    if hasattr(clf, "coef_"):
        return np.abs(np.asarray(clf.coef_)).ravel()
    raise ValueError(f"{type(clf).__name__} için önem skoru çıkarılamıyor.")


def compute_channel_importance(
    patient_id: str,
    model_name: str,
    processed_dir: Path = PROCESSED_DATA_DIR,
    n_splits: int = N_SPLITS,
) -> pd.DataFrame:
    """Seçilen modeli `n_splits` kez (her CV fold'unda train verisiyle)
    fit edip önem skorlarını fold'lar arasında ortalar, düz özellik
    vektörünü (n_channels x n_features) kanal bazında toplayarak
    (`processed_channel` sırasıyla hizalı) tek bir kanal-önemi skoruna
    indirger, ardından `<hasta>_anatomical_labels.csv`'deki anatomik
    etiketlerle birleştirir.

    Döndürür: sütunları `processed_channel`, `channel` (orijinal/locs
    indeksi), `area_label`, `importance` olan, önem skoruna göre azalan
    sırada bir DataFrame.
    """
    from sklearn.base import clone
    from sklearn.model_selection import GroupKFold

    dataset = load_patient_dataset(patient_id, processed_dir)
    X, y, time_sec = dataset["X"], dataset["y"], dataset["time_sec"]
    groups = make_time_block_groups(time_sec, n_splits)
    roi_channels = dataset["channels"]  # processed_channel sırası, X kolonlarıyla hizalı
    n_channels = len(roi_channels)
    n_features_per_channel = len(dataset["feature_names"])

    pipeline = build_models()[model_name]
    gkf = GroupKFold(n_splits=n_splits)

    importances = []
    for train_idx, _ in gkf.split(X, y, groups):
        model = clone(pipeline)
        model.fit(X[train_idx], y[train_idx])
        importances.append(_get_flat_importance(model))

    mean_importance_flat = np.mean(importances, axis=0)
    per_channel_importance = mean_importance_flat.reshape(n_channels, n_features_per_channel).mean(axis=1)

    anat_df = pd.read_csv(processed_dir / patient_id / f"{patient_id}_anatomical_labels.csv")
    importance_df = pd.DataFrame({
        "processed_channel": roi_channels,
        "importance": per_channel_importance,
    })
    merged = importance_df.merge(
        anat_df[["processed_channel", "channel", "area_label"]], on="processed_channel", how="left"
    )
    merged["importance_rank"] = merged["importance"].rank(ascending=False, method="min").astype(int)
    return merged.sort_values("importance", ascending=False).reset_index(drop=True)


def region_of(area_label: str) -> str:
    """Anatomik etiketi kaba bir bölge kategorisine indirger."""
    if not isinstance(area_label, str):
        return "diğer"
    label = area_label.lower()
    if "fusiform" in label:
        return "fusiform"
    if "temporal" in label or "parahippocampal" in label:
        return "temporal"
    if "occipital" in label or "cuneus" in label:
        return "occipital"
    return "diğer"


def compute_ttest_ranking(patient_id: str, processed_dir: Path = PROCESSED_DATA_DIR) -> pd.DataFrame:
    """04_feature_extraction.ipynb Bölüm 5'teki face-vs-house high-gamma
    t-testini bir hasta için yeniden hesaplar (karşılaştırma referansı)."""
    from scipy.stats import ttest_ind

    npz = np.load(processed_dir / patient_id / "features.npz", allow_pickle=True)
    features = npz["features"]
    event_labels = npz["event_labels"]
    channels = npz["channels"]
    area_labels = npz["area_labels"]
    feature_names = list(npz["feature_names"])

    hg_idx = feature_names.index("high_gamma")
    hg = features[:, :, hg_idx]
    face_mask = event_labels == "face"
    house_mask = event_labels == "house"

    t_vals, p_vals = ttest_ind(hg[face_mask], hg[house_mask], axis=0)
    df = pd.DataFrame({
        "processed_channel": channels,
        "area_label": area_labels,
        "t_stat": t_vals,
        "p_value": p_vals,
        "abs_t": np.abs(t_vals),
    })
    df["ttest_rank"] = df["abs_t"].rank(ascending=False, method="min").astype(int)
    return df.sort_values("abs_t", ascending=False).reset_index(drop=True)


def compare_importance_with_ttest(patient_id: str, model_name: str, top_k: int = 3) -> dict:
    """Bir hastada modelin en önemli bulduğu kanallarla, t-testinin en
    ayrıştırıcı bulduğu kanalları karşılaştırır (top-k örtüşme, Spearman
    korelasyonu)."""
    importance_df = compute_channel_importance(patient_id, model_name)
    ttest_df = compute_ttest_ranking(patient_id)

    merged = importance_df.merge(
        ttest_df[["processed_channel", "t_stat", "abs_t", "ttest_rank"]],
        on="processed_channel",
        how="inner",
    )
    spearman_corr = merged["importance_rank"].corr(merged["ttest_rank"], method="spearman")

    top_importance = set(importance_df.head(top_k)["processed_channel"])
    top_ttest = set(ttest_df.head(top_k)["processed_channel"])
    overlap = top_importance & top_ttest

    return {
        "patient": patient_id,
        "n_channels": len(merged),
        "spearman_rank_corr": float(spearman_corr),
        "top_k": top_k,
        "top_importance_channels": sorted(top_importance),
        "top_ttest_channels": sorted(top_ttest),
        "overlap": sorted(overlap),
        "n_overlap": len(overlap),
        "top_importance_region": region_of(
            importance_df.iloc[0]["area_label"]
        ),
        "top_ttest_region": region_of(ttest_df.iloc[0]["area_label"]),
        "merged": merged,
    }


# ---------------------------------------------------------------------------
# 3. Kortikal yüzey (marching cubes) + elektrot bindirme
# ---------------------------------------------------------------------------

def extract_cortex_mesh(
    patient_id: str, brains_dir: Path = BRAINS_DIR, step_size: int = 6
) -> tuple[np.ndarray, np.ndarray]:
    """`brains/<hasta>/<hasta>_mri.nii` hacminden, Otsu eşiğiyle
    (kafa/beyin dış zarfı) marching cubes ile yaklaşık bir yüzey mesh'i
    çıkarır ve köşe noktalarını `.nii` affine'i ile world (mm) uzayına
    taşır -- bu, `locs/<hasta>_xslocs.mat`'teki elektrot koordinatlarıyla
    aynı uzay.

    NOT: Bu veri setinde önceden hesaplanmış bir pial/kortikal yüzey
    (FreeSurfer/SPM segmentasyonu) bulunmuyor -- yalnızca ham T1 `.nii`
    hacmi var. Bu nedenle burada üretilen yüzey, gerçek bir kortikal
    (pial) yüzey değil, **Otsu eşiğine dayalı yaklaşık bir doku zarfı**
    (kafa derisi/beyin dış sınırına yakın) -- yalnızca elektrotların
    göreli 3D konumunu görselleştirmek amacıyla kullanılmalı.

    Döndürür: (world_vertices (n_verts x 3), faces (n_faces x 3))
    """
    import nibabel as nib
    from skimage.filters import threshold_otsu
    from skimage.measure import marching_cubes

    nii_path = brains_dir / patient_id / f"{patient_id}_mri.nii"
    if not nii_path.exists():
        raise FileNotFoundError(f"MRI dosyası bulunamadı: {nii_path}")

    img = nib.load(str(nii_path))
    volume = img.get_fdata()
    level = threshold_otsu(volume[volume > 0])

    verts, faces, _normals, _values = marching_cubes(volume, level=level, step_size=step_size)
    world_verts = nib.affines.apply_affine(img.affine, verts)
    return world_verts, faces


def load_electrode_locations(patient_id: str, locs_dir: Path = LOCS_DIR) -> np.ndarray:
    """`locs/<hasta>_xslocs.mat`'teki elektrot koordinatlarını (n_channels x 3,
    world/mm uzayında) döndürür."""
    import scipy.io as sio

    locs_mat = sio.loadmat(locs_dir / f"{patient_id}_xslocs.mat")
    return locs_mat["locs"]


if __name__ == "__main__":
    best_model, results_df = select_best_model()
    print(f"En iyi model (ortalama accuracy'ye göre): {best_model}")
    print(results_df.groupby("model")[["accuracy", "macro_f1", "roc_auc"]].mean())
