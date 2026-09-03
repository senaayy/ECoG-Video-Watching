"""data/raw altındaki Miller (2019) Faces vs Houses ECoG .mat dosyalarını
yükleyip içeriğini (değişkenler, boyutlar, örnekleme hızı, hasta/oturum
sayısı) özetleyen script.

Yükleme sırası: scipy.io.loadmat -> (v7.3 dosyalar için) h5py -> mat73.

Kullanım:
    python src/data/load_dataset.py [data/raw]
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

# .mat dosyasında örnekleme hızını taşıyabilecek olası anahtar isimleri.
SRATE_KEYS = ("srate", "fs", "Fs", "sr", "sampling_rate", "sample_rate")


def _is_matlab_meta_key(key: str) -> bool:
    return key.startswith("__") and key.endswith("__")


def load_mat_any(path: Path) -> tuple[dict[str, Any], str]:
    """Bir .mat dosyasını mümkün olan en uygun yöntemle yükler.

    Döndürür: (değişken adı -> değer sözlüğü, kullanılan yöntemin adı)
    """
    try:
        from scipy.io import loadmat

        data = loadmat(str(path), squeeze_me=True, struct_as_record=False)
        data = {k: v for k, v in data.items() if not _is_matlab_meta_key(k)}
        return data, "scipy.io.loadmat"
    except NotImplementedError:
        # scipy, MATLAB v7.3 (HDF5 tabanlı) dosyalarını yükleyemiyor.
        pass
    except Exception as exc:  # noqa: BLE001 - kasıtlı olarak geniş: fallback zinciri
        print(f"  [uyarı] scipy.io.loadmat başarısız oldu ({exc}); h5py deneniyor...")

    try:
        import h5py

        data: dict[str, Any] = {}
        with h5py.File(path, "r") as f:
            for key in f.keys():
                if _is_matlab_meta_key(key):
                    continue
                item = f[key]
                data[key] = np.asarray(item) if isinstance(item, h5py.Dataset) else item
        return data, "h5py"
    except ImportError:
        print("  [uyarı] h5py kurulu değil (pip install h5py); mat73 deneniyor...")
    except Exception as exc:  # noqa: BLE001
        print(f"  [uyarı] h5py başarısız oldu ({exc}); mat73 deneniyor...")

    try:
        import mat73

        data = mat73.loadmat(str(path))
        return data, "mat73"
    except ImportError as exc:
        raise RuntimeError(
            f"{path} yüklenemedi: scipy.io.loadmat, h5py ve mat73 hiçbiri işe "
            "yaramadı. 'pip install h5py mat73' ile fallback bağımlılıklarını "
            "kurup tekrar deneyin."
        ) from exc


def _describe_value(value: Any) -> str:
    if isinstance(value, np.ndarray):
        return f"ndarray, shape={value.shape}, dtype={value.dtype}"
    if isinstance(value, (list, tuple)):
        return f"{type(value).__name__}, len={len(value)}"
    return f"{type(value).__name__}: {value!r}"


def find_sampling_rate(data: dict[str, Any]) -> float | None:
    for key in SRATE_KEYS:
        if key in data:
            value = data[key]
            try:
                return float(np.asarray(value).squeeze())
            except (TypeError, ValueError):
                continue
    return None


def summarize_file(path: Path) -> dict[str, Any]:
    data, method = load_mat_any(path)
    srate = find_sampling_rate(data)

    print(f"\n=== {path} ===")
    print(f"Yükleme yöntemi: {method}")
    print(f"Değişkenler ({len(data)}):")
    for key, value in data.items():
        print(f"  - {key}: {_describe_value(value)}")
    print(f"Örnekleme hızı: {srate if srate is not None else 'bulunamadı'}")

    return {"path": path, "method": method, "variables": list(data.keys()), "srate": srate}


def discover_mat_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.mat"))


def summarize_sessions(mat_files: list[Path]) -> None:
    """Dosya adı önekine (ör. 'bp1_...', 'sub-01_...') göre hasta/oturum
    sayısını kabaca tahmin eder. Gerçek klasör yapısı farklıysa bu sadece
    bir yaklaşıklamadır; kesin sayı için data/raw içeriğine bakılmalı."""
    patients = {}
    for f in mat_files:
        prefix = f.stem.split("_")[0]
        patients.setdefault(prefix, []).append(f)

    print(f"\nToplam .mat dosyası: {len(mat_files)}")
    print(f"Dosya adı önekine göre tahmini hasta/grup sayısı: {len(patients)}")
    for prefix, files in sorted(patients.items()):
        print(f"  - {prefix}: {len(files)} dosya")


def main(argv: list[str]) -> int:
    root = Path(argv[1]) if len(argv) > 1 else Path("data/raw")
    if not root.exists():
        print(f"Hata: {root} bulunamadı.")
        return 1

    mat_files = discover_mat_files(root)
    if not mat_files:
        print(f"{root} altında .mat dosyası bulunamadı.")
        print("Önce scripts/download_faces_houses.sh ile veriyi indirin.")
        return 1

    summarize_sessions(mat_files)
    for f in mat_files:
        try:
            summarize_file(f)
        except RuntimeError as exc:
            print(f"\n=== {f} ===\n[HATA] {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
