"""Neuromatch Academy'nin işlenmiş Miller (2019) Faces-vs-Houses ECoG
veri setini (`faceshouses.npz`) yükleyen ve özetleyen script.

Veri kaynağı: https://osf.io/argh7/download (bkz. data/raw/README.md)

`faceshouses.npz` içinde tek bir dizi vardır: `dat` (dtype=object), uzunluğu
7 (hasta sayısı). Her eleman `(dat1, dat2)` şeklinde iki sözlük içeren bir
tuple/array'dir:

    dat1: pasif izleme deneyi   -> V, srate, t_on, t_off, stim_id, locs
    dat2: gürültülü tespit görevi -> V, srate, t_on, t_off, stim_cat,
                                      stim_noise, key_press, locs

Kullanım:
    python src/data/load_faceshouses.py [data/raw/faceshouses.npz]
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np


def load_faceshouses(path: Path) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """`faceshouses.npz` dosyasını yükler, her hasta için (dat1, dat2)
    sözlük çiftlerinden oluşan bir liste döndürür."""
    npz = np.load(path, allow_pickle=True)
    if "dat" not in npz:
        raise KeyError(
            f"Beklenmeyen dosya içeriği: anahtarlar={list(npz.keys())} "
            "('dat' anahtarı bulunamadı; dosya bozuk veya farklı bir sürüm olabilir)"
        )
    subjects = npz["dat"]
    return [(dat1, dat2) for dat1, dat2 in subjects]


def _describe(value: Any) -> str:
    if isinstance(value, np.ndarray):
        return f"ndarray, shape={value.shape}, dtype={value.dtype}"
    return f"{type(value).__name__}: {value!r}"


def summarize(subjects: list[tuple[dict[str, Any], dict[str, Any]]]) -> None:
    print(f"Toplam hasta sayısı: {len(subjects)}\n")
    dat1, dat2 = subjects[0]

    print("=== Hasta 1 / Deney 1 (dat1, pasif izleme) ===")
    for key in ("V", "srate", "t_on", "t_off", "stim_id", "locs"):
        if key in dat1:
            print(f"  - {key}: {_describe(dat1[key])}")
    n_stim = len(dat1["stim_id"])
    n_houses = int(np.sum(np.asarray(dat1["stim_id"]) <= 50))
    n_faces = n_stim - n_houses
    print(f"  -> {n_stim} uyaran ({n_houses} ev, {n_faces} yüz), "
          f"{dat1['V'].shape[1]} kanal")

    print("\n=== Hasta 1 / Deney 2 (dat2, gürültülü tespit) ===")
    for key in ("V", "srate", "t_on", "t_off", "stim_cat", "stim_noise", "key_press", "locs"):
        if key in dat2:
            print(f"  - {key}: {_describe(dat2[key])}")
    n_stim2 = len(dat2["stim_cat"])
    n_houses2 = int(np.sum(np.asarray(dat2["stim_cat"]) == 1))
    n_faces2 = int(np.sum(np.asarray(dat2["stim_cat"]) == 2))
    print(f"  -> {n_stim2} uyaran ({n_houses2} ev, {n_faces2} yüz)")

    print("\n=== Tüm hastalarda kanal/uyaran sayıları ===")
    for i, (d1, d2) in enumerate(subjects, start=1):
        print(f"  Hasta {i}: dat1 kanal={d1['V'].shape[1]}, "
              f"dat1 uyaran={len(d1['stim_id'])}, "
              f"dat2 uyaran={len(d2['stim_cat'])}")


def main(argv: list[str]) -> int:
    path = Path(argv[1]) if len(argv) > 1 else Path("data/raw/faceshouses.npz")
    if not path.exists():
        print(f"Hata: {path} bulunamadı.")
        print("Önce data/raw/README.md içindeki adımları izleyerek dosyayı indirin.")
        return 1

    subjects = load_faceshouses(path)
    summarize(subjects)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
