"""Hasta bazlı stim/etiket (event) çıkarımı (Miller 2019, faces_basic).

`stim` kodlama şemasının (0/1-50/51-100/101) 14 hastanın tamamında aynı
olduğu doğrulandı (bkz. notebooks/03_label_alignment.ipynb, Bölüm 1). Bu
modül, notebooks/01_eda.ipynb'de `ap` hastası için yapılan onset tespiti +
face/house/baseline sınıflandırma mantığını tüm hastalara uygulanabilir
şekilde genelleştirir.

Kullanım (kütüphane olarak):
    from src.preprocessing.align_labels import build_labels_for_patient
    labels_df = build_labels_for_patient("ap")

Kullanım (komut satırından, tüm hastalar için):
    python src/preprocessing/align_labels.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.preprocessing.preprocess import (
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    list_available_patients,
    load_patient,
)

# README_faces_basic_dataset_notes.docx ve fh_get_events.m ile doğrulanan
# stim kodlama şeması (bkz. notebooks/01_eda.ipynb, Bölüm 5):
#   0        -> baseline (pre/post task run)
#   1-50     -> house
#   51-100   -> face
#   101      -> baseline (interstimulus interval)
EXPECTED_STIM_VALUES = frozenset(range(0, 102))


def stim_to_class(code: int) -> str:
    """Tek bir stim kodunu face/house/baseline sınıfına çevirir."""
    if code == 0 or code == 101:
        return "baseline"
    elif 1 <= code <= 50:
        return "house"
    elif 51 <= code <= 100:
        return "face"
    return "unknown"


def validate_stim_scheme(stim: np.ndarray) -> dict:
    """Bir hastanın `stim` dizisinin beklenen 0/1-50/51-100/101 şemasına
    uyup uymadığını kontrol eder.

    Döndürür: {"conforms": bool, "unique_values": sorted list,
               "unexpected_values": sorted list}
    """
    unique_values = sorted(int(v) for v in np.unique(stim))
    unexpected = sorted(set(unique_values) - EXPECTED_STIM_VALUES)
    return {
        "conforms": len(unexpected) == 0,
        "unique_values": unique_values,
        "unexpected_values": unexpected,
    }


def extract_events(stim: np.ndarray, srate: int) -> pd.DataFrame:
    """`stim` dizisindeki değişim noktalarından (onset) bir event DataFrame'i
    üretir: sample_idx, time_sec, stim_code, class (face/house/baseline).

    01_eda.ipynb'deki `ap`-özel onset tespit mantığıyla birebir aynıdır.
    """
    change_idx = np.where(np.diff(stim) != 0)[0] + 1
    onset_idx = np.concatenate(([0], change_idx))

    events_df = pd.DataFrame({
        "sample_idx": onset_idx,
        "time_sec": onset_idx / srate,
        "stim_code": stim[onset_idx],
    })
    events_df["class"] = events_df["stim_code"].apply(stim_to_class)
    return events_df


def build_labels_for_patient(
    patient_id: str, raw_dir: Path = RAW_DATA_DIR
) -> pd.DataFrame:
    """Bir hasta için stim şemasını doğrular ve event/etiket DataFrame'ini üretir.

    Şema beklenmedik bir değer içeriyorsa ValueError fırlatır (bu veri
    setinde 14 hastanın tamamı şemaya uyduğu için normal koşullarda
    tetiklenmez; ileride farklı bir görev/hastayla kullanılırsa güvenlik
    ağı görevi görür).
    """
    _, stim, srate = load_patient(patient_id, raw_dir)

    validation = validate_stim_scheme(stim)
    if not validation["conforms"]:
        raise ValueError(
            f"{patient_id}: stim şeması beklenenden farklı -- "
            f"beklenmeyen değerler: {validation['unexpected_values']}"
        )

    labels_df = extract_events(stim, srate)
    labels_df.insert(0, "patient", patient_id)
    return labels_df


def build_and_save_all_labels(
    patients: Optional[list[str]] = None,
    raw_dir: Path = RAW_DATA_DIR,
    out_dir: Path = PROCESSED_DATA_DIR,
) -> pd.DataFrame:
    """Tüm (veya verilen) hastalar için `build_labels_for_patient()`
    çalıştırır, her hastanın etiket DataFrame'ini
    `data/processed/<id>/labels.parquet` olarak kaydeder.

    Döndürür: hasta bazlı sınıf dağılımı özet DataFrame'i (aynı zamanda
    `data/processed/labels_class_distribution.csv` olarak da kaydedilir).
    """
    if patients is None:
        patients = list_available_patients(raw_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for patient_id in patients:
        labels_df = build_labels_for_patient(patient_id, raw_dir)

        patient_out_dir = out_dir / patient_id
        patient_out_dir.mkdir(parents=True, exist_ok=True)
        labels_df.to_parquet(patient_out_dir / "labels.parquet", index=False)

        counts = labels_df["class"].value_counts()
        summary_rows.append({
            "patient": patient_id,
            "n_events": len(labels_df),
            "n_baseline": int(counts.get("baseline", 0)),
            "n_house": int(counts.get("house", 0)),
            "n_face": int(counts.get("face", 0)),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "labels_class_distribution.csv", index=False)
    return summary_df


if __name__ == "__main__":
    summary = build_and_save_all_labels()
    print(summary.to_string(index=False))
