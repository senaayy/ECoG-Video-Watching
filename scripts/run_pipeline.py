#!/usr/bin/env python3
"""Uçtan uca pipeline: faceshouses.npz -> epoklama -> frekans bandı
özellik çıkarımı -> LDA/SVM/RF sınıflandırma -> değerlendirme + görseller.

Kullanım:
    python scripts/run_pipeline.py [data/raw/faceshouses.npz]

Çıktılar reports/figures/ altına, sayısal özet ise stdout'a yazılır.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.load_faceshouses import load_faceshouses
from src.preprocessing.epoching import drop_invalid_epochs, extract_epochs
from src.features.bandpower import DEFAULT_BANDS, average_band_power_by_name, extract_band_features
from src.models.classify import MODEL_FACTORIES
from src.evaluation.evaluate import evaluate_model, plot_confusion_matrix, plot_roc_curve

FIG_DIR = Path(__file__).resolve().parent.parent / "reports" / "figures"


def prepare_subject(dat1: dict, tmin: float = -0.2, tmax: float = 0.4):
    """Bir hastanın dat1 (pasif izleme) verisinden epok + etiket + özellik
    çıkarır. Etiket: 0 = ev, 1 = yüz (stim_id 1-50 ev, 51-100 yüz)."""
    V = dat1["V"]
    t_on = np.asarray(dat1["t_on"]).ravel()
    stim_id = np.asarray(dat1["stim_id"]).ravel()
    srate = float(np.asarray(dat1["srate"]).squeeze())

    labels = (stim_id > 50).astype(int)  # 0=ev, 1=yüz

    epochs = extract_epochs(V, t_on, srate, tmin=tmin, tmax=tmax)
    epochs, labels = drop_invalid_epochs(epochs, labels)

    features, feature_names = extract_band_features(epochs, srate)
    return features, labels, feature_names, srate


def run_per_subject(subjects) -> dict:
    """Her hasta için ayrı ayrı (within-subject) Stratified K-Fold ile
    değerlendirme yapar; P300 projesindeki 'tek denek' değerlendirmesinin
    ECoG karşılığı."""
    all_results: dict[str, list[float]] = {name: [] for name in MODEL_FACTORIES}
    per_subject_band_means: list[dict[str, float]] = []

    for i, (dat1, _dat2) in enumerate(subjects, start=1):
        features, labels, feature_names, srate = prepare_subject(dat1)
        # Kanal sayısı hastadan hastaya değiştiği için (gerçek ECoG verisinde
        # normal) özellik matrislerini doğrudan vstack'lemiyoruz; her hastanın
        # bant-gücü özetini ayrı çıkarıp sonra hastalar arası ortalıyoruz.
        per_subject_band_means.append(average_band_power_by_name(features, feature_names, DEFAULT_BANDS))

        for name, factory in MODEL_FACTORIES.items():
            model = factory()
            result = evaluate_model(model, features, labels)
            all_results[name].append(result["accuracy"])
            print(f"  Hasta {i} | {name}: doğruluk={result['accuracy']:.3f}"
                  + (f", ROC-AUC={result['roc_auc']:.3f}" if result["roc_auc"] else ""))

    return {
        "per_model_accuracies": all_results,
        "per_subject_band_means": per_subject_band_means,
    }


def plot_band_importance(per_subject_band_means: list[dict[str, float]]):
    """Hastalar arası ortalama bant gücünü çizer — hangi frekans bandının
    görsel kategori ayrımında öne çıktığını gösterir (Tasnim'in 1.lik alan
    takımının vurguladığı analiz türü)."""
    band_names = list(per_subject_band_means[0].keys())
    band_means = {
        band: float(np.mean([subj[band] for subj in per_subject_band_means]))
        for band in band_names
    }

    fig, ax = plt.subplots(figsize=(7, 4.5))
    names = list(band_means.keys())
    values = [band_means[n] for n in names]
    ax.bar(names, values, color="#4C72B0", edgecolor="black")
    ax.set_ylabel("Ortalama bant gücü (tüm kanal/deneme)")
    ax.set_title("Frekans Bandına Göre Ortalama ECoG Gücü (Faces vs Houses)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "band_power_summary.png"
    plt.savefig(out, dpi=200)
    plt.close(fig)
    print(f"\nKaydedildi: {out}")


def plot_accuracy_summary(per_model_accuracies: dict[str, list[float]]):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    names = list(per_model_accuracies.keys())
    means = [np.mean(per_model_accuracies[n]) for n in names]
    stds = [np.std(per_model_accuracies[n]) for n in names]
    ax.bar(names, means, yerr=stds, capsize=6, color="#55A868", edgecolor="black")
    ax.axhline(0.5, color="gray", linestyle="--", label="Rastgele (%50)")
    ax.set_ylabel("Ortalama doğruluk (hastalar arası)")
    ax.set_title("Model Karşılaştırması — Within-Subject Doğruluk")
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "model_accuracy_summary.png"
    plt.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Kaydedildi: {out}")


def main(argv: list[str]) -> int:
    path = Path(argv[1]) if len(argv) > 1 else Path("data/raw/faceshouses.npz")
    if not path.exists():
        print(f"Hata: {path} bulunamadı. data/raw/README.md dosyasındaki adımları izleyin.")
        return 1

    subjects = load_faceshouses(path)
    print(f"{len(subjects)} hasta yüklendi.\n")

    print("=== Within-subject değerlendirme (her hasta ayrı Stratified 5-Fold) ===")
    outcome = run_per_subject(subjects)

    print("\n=== Özet (hastalar arası ortalama) ===")
    for name, accs in outcome["per_model_accuracies"].items():
        print(f"  {name}: {np.mean(accs):.3f} ± {np.std(accs):.3f}")

    plot_band_importance(outcome["per_subject_band_means"])
    plot_accuracy_summary(outcome["per_model_accuracies"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
