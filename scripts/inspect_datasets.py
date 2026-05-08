"""
IrisFlow — scripts/inspect_datasets.py

Inspeciona os datasets e imprime um resumo para confirmar que os dados
estão no formato correto antes de rodar train_iris.py.

Rodar com:
    .venv\\Scripts\\python.exe scripts/inspect_datasets.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent

OPENEDS_IMG_DIR  = ROOT / "datasets" / "OpenEDS" / "openEDS" / "openEDS"
OPENEDS_BBOX_DIR = ROOT / "datasets" / "OpenEDS" / "bbox" / "bbox"
MPIIGAZE_NORM    = ROOT / "datasets" / "MPIIGaze" / "archive" / "MPIIGaze" / "Data" / "Normalized"


def _sep() -> None:
    print("-" * 60)


def inspect_openeds() -> None:
    _sep()
    print("OpenEDS")
    _sep()

    if not OPENEDS_IMG_DIR.exists():
        print(f"  ERRO: diretório não encontrado: {OPENEDS_IMG_DIR}")
        return

    subj_dirs = [d for d in sorted(OPENEDS_IMG_DIR.iterdir()) if d.is_dir()]
    print(f"  Sujeitos (S_*): {len(subj_dirs)}")

    total_images = 0
    for sd in subj_dirs:
        imgs = list(sd.glob("*.npy")) + list(sd.glob("*.png"))
        total_images += len(imgs)
    print(f"  Total de imagens (.npy + .png): {total_images}")

    # Inspecionar a primeira imagem disponível
    first_img = None
    for sd in subj_dirs:
        candidates = sorted(sd.glob("*.npy"))
        if candidates:
            first_img = candidates[0]
            break
    if first_img:
        try:
            arr = np.load(str(first_img))
            print(f"  Exemplo de imagem .npy: {first_img.relative_to(ROOT)}")
            print(f"    shape={arr.shape}  dtype={arr.dtype}  min={arr.min()}  max={arr.max()}")
        except Exception as e:
            print(f"  Falha ao carregar imagem exemplo: {e}")

    # Inspecionar bbox
    bbox_files = list(OPENEDS_BBOX_DIR.glob("S_*.txt")) if OPENEDS_BBOX_DIR.exists() else []
    print(f"\n  Arquivos bbox (S_*.txt): {len(bbox_files)}")

    if bbox_files:
        first_bbox = sorted(bbox_files)[0]
        print(f"  Exemplo: {first_bbox.relative_to(ROOT)}")
        with open(first_bbox) as f:
            lines = [f.readline().rstrip() for _ in range(3)]
        print("  Primeiras 3 linhas:")
        for ln in lines:
            print(f"    {ln}")
        parts = lines[0].split()
        if len(parts) >= 4:
            x_min, x_max, y_min, y_max = (int(v) for v in parts[:4])
            cx = (x_min + x_max) / 2
            cy = (y_min + y_max) / 2
            print(f"  Formato: x_min x_max y_min y_max")
            print(f"  Linha 1: x=[{x_min},{x_max}]  y=[{y_min},{y_max}]  center=({cx:.1f},{cy:.1f})")


def inspect_mpiigaze() -> None:
    _sep()
    print("MPIIGaze")
    _sep()

    if not MPIIGAZE_NORM.exists():
        print(f"  ERRO: diretório não encontrado: {MPIIGAZE_NORM}")
        return

    mat_files = sorted(MPIIGAZE_NORM.rglob("*.mat"))
    print(f"  Arquivos .mat (Normalized): {len(mat_files)}")

    if not mat_files:
        print("  Nenhum .mat encontrado.")
        return

    first_mat = mat_files[0]
    print(f"  Exemplo: {first_mat.relative_to(ROOT)}")

    try:
        import scipy.io
    except ImportError:
        print("  ERRO: scipy não instalado — pip install scipy")
        return

    try:
        mat = scipy.io.loadmat(str(first_mat))
        data = mat["data"][0, 0]
        print("  Campos em data['left']:")
        left = data["left"][0, 0]
        for field in left.dtype.names or []:
            val = left[field]
            info = f"shape={val.shape}  dtype={val.dtype}" if hasattr(val, "shape") else repr(val)
            print(f"    [{field}]  {info}")
    except Exception as e:
        print(f"  Falha ao inspecionar .mat: {e}")


def main() -> None:
    print()
    print("IrisFlow — Inspeção de Datasets")
    print()
    inspect_openeds()
    print()
    inspect_mpiigaze()
    _sep()
    print()


if __name__ == "__main__":
    main()
