"""
IrisFlow — scripts/build_prior.py

Processa o MPIIGaze (Data/Normalized + Data/Original) e gera engine/prior_gaze.npy:
um prior compacto (~50KB) com a distribuição estatística de vetores
de olhar humanos mapeados para posições de tela normalizadas [0,1].

Rodar UMA VEZ antes de usar o sistema:
    .venv\\Scripts\\python.exe scripts/build_prior.py

Requisitos:
    pip install scipy numpy
    Dataset em: datasets/MPIIGaze/archive/MPIIGaze/

Saída:
    engine/prior_gaze.npy  (~50KB, já commitado no repositório)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import scipy.io
except ImportError:
    print("ERRO: pip install scipy")
    sys.exit(1)

# ── Caminhos ──────────────────────────────────────────────────────────────────

ROOT         = Path(__file__).parent.parent   # raiz do projeto (scripts/../)
NORM_DIR     = ROOT / "datasets" / "MPIIGaze" / "archive" / "MPIIGaze" / "Data" / "Normalized"
ORIG_DIR     = ROOT / "datasets" / "MPIIGaze" / "archive" / "MPIIGaze" / "Data" / "Original"
OUT_PATH     = ROOT / "engine" / "prior_gaze.npy"

# ── Configuração ──────────────────────────────────────────────────────────────

# Quantas amostras por arquivo .mat usar (None = todas)
MAX_PER_FILE = 200

# Peso do prior no GazeModel — bem fraco para não sobrescrever calibração real
PRIOR_WEIGHT = 0.03

# Resolução de referência do laptop MPIIGaze (pixels)
SCREEN_W = 1280
SCREEN_H = 800


# ── Extração ──────────────────────────────────────────────────────────────────

def _load_pair(
    mat_path: Path,
    annot_path: Path,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Lê um par (day.mat, annotation.txt) e retorna (features, screen_positions).

    Features: (N, 2) — [gaze_x, gaze_y] do vetor 3D normalizado (≈ yaw/pitch)
    Posições: (N, 2) — posição de tela normalizada [0, 1]

    A annotation.txt do MPIIGaze tem por linha:
        24 valores de landmarks faciais, depois:
        campo[24] = screen_x (pixels), campo[25] = screen_y (pixels)
    """
    # ── Posições de tela ──────────────────────────────────────────────────────
    try:
        screen_pos = []
        with open(annot_path, "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 26:
                    continue
                sx = float(parts[24])
                sy = float(parts[25])
                screen_pos.append((sx, sy))
        if not screen_pos:
            return None
        pos_arr = np.array(screen_pos, dtype=np.float64)
    except Exception as e:
        print(f"  Aviso annotation {annot_path.name}: {e}")
        return None

    # ── Vetores de gaze (Normalized .mat) ────────────────────────────────────
    try:
        mat = scipy.io.loadmat(str(mat_path))
        # Usa olho esquerdo (mais consistente no MPIIGaze)
        gaze_3d = mat["data"][0, 0]["left"][0, 0]["gaze"]   # (N, 3)
    except Exception as e:
        print(f"  Aviso mat {mat_path.name}: {e}")
        return None

    n = min(len(gaze_3d), len(pos_arr))
    if n == 0:
        return None

    gaze_3d = gaze_3d[:n]
    pos_arr = pos_arr[:n]

    # Extrai os dois primeiros componentes do vetor unitário (≈ yaw/pitch em radianos)
    feats = gaze_3d[:, :2].astype(np.float64)

    # Normaliza posição para [0, 1]
    pos_norm = np.column_stack([
        np.clip(pos_arr[:, 0] / SCREEN_W, 0.0, 1.0),
        np.clip(pos_arr[:, 1] / SCREEN_H, 0.0, 1.0),
    ])

    return feats, pos_norm


def _find_annot(participant: str, day: str) -> Optional[Path]:
    """Localiza o annotation.txt correspondente a um participante/dia."""
    p = ORIG_DIR / participant / day / "annotation.txt"
    return p if p.exists() else None


# ── Main ──────────────────────────────────────────────────────────────────────

def build_prior() -> None:
    if not NORM_DIR.exists():
        print(f"ERRO: dataset não encontrado em {NORM_DIR}")
        print("Verifique se o MPIIGaze está extraído corretamente.")
        sys.exit(1)

    mat_files = sorted(NORM_DIR.rglob("*.mat"))
    if not mat_files:
        print(f"ERRO: nenhum .mat encontrado em {NORM_DIR}")
        sys.exit(1)

    print(f"Encontrados {len(mat_files)} arquivos .mat")
    print(f"Processando (máx {MAX_PER_FILE} amostras por arquivo)...\n")

    all_features:  List[np.ndarray] = []
    all_positions: List[np.ndarray] = []
    skipped = 0

    for i, mat_path in enumerate(mat_files):
        participant = mat_path.parent.name          # e.g. "p00"
        day         = mat_path.stem                 # e.g. "day01"
        annot_path  = _find_annot(participant, day)

        if annot_path is None:
            skipped += 1
            continue

        result = _load_pair(mat_path, annot_path)
        if result is None:
            skipped += 1
            continue

        feats, pos = result
        n = min(len(feats), MAX_PER_FILE)

        # Amostragem uniforme (não pegar só o início)
        indices = np.linspace(0, len(feats) - 1, n, dtype=int)
        all_features.append(feats[indices])
        all_positions.append(pos[indices])

        if (i + 1) % 50 == 0 or (i + 1) == len(mat_files):
            total = sum(len(f) for f in all_features)
            print(f"  [{i+1}/{len(mat_files)}] {total} pontos coletados")

    if skipped:
        print(f"  ({skipped} arquivos pulados por falta de annotation.txt)")

    if not all_features:
        print("ERRO: pontos insuficientes extraídos. Verifique o formato do dataset.")
        sys.exit(1)

    feats = np.vstack(all_features)
    poses = np.vstack(all_positions)

    print(f"\nTotal bruto: {len(feats)} pontos")

    # ── Clustering para compactar ─────────────────────────────────────────────
    # Grid 20×20 no espaço de features — mantém 1 ponto por célula (mediana)
    print("Compactando via grid quantization (20×20)...")

    gx_min, gx_max = float(feats[:, 0].min()), float(feats[:, 0].max())
    gy_min, gy_max = float(feats[:, 1].min()), float(feats[:, 1].max())

    GRID = 20
    compact_feats: List[np.ndarray]          = []
    compact_poses: List[Tuple[float, float]] = []

    for gxi in range(GRID):
        for gyi in range(GRID):
            x0 = gx_min + (gx_max - gx_min) * gxi       / GRID
            x1 = gx_min + (gx_max - gx_min) * (gxi + 1) / GRID
            y0 = gy_min + (gy_max - gy_min) * gyi       / GRID
            y1 = gy_min + (gy_max - gy_min) * (gyi + 1) / GRID

            mask = (
                (feats[:, 0] >= x0) & (feats[:, 0] < x1) &
                (feats[:, 1] >= y0) & (feats[:, 1] < y1)
            )
            if mask.sum() == 0:
                continue

            compact_feats.append(np.median(feats[mask], axis=0))
            compact_poses.append(tuple(np.median(poses[mask], axis=0).tolist()))

    print(f"Compactado: {len(compact_feats)} pontos representativos")

    # ── Salvar ────────────────────────────────────────────────────────────────
    prior = {
        "features":  np.array(compact_feats, dtype=np.float64),
        "positions": np.array(compact_poses, dtype=np.float64),
        "weight":    np.float64(PRIOR_WEIGHT),
        "gx_min":    np.float64(gx_min),
        "gx_max":    np.float64(gx_max),
        "gy_min":    np.float64(gy_min),
        "gy_max":    np.float64(gy_max),
        "n_source":  np.int64(len(feats)),
        "n_compact": np.int64(len(compact_feats)),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT_PATH, prior)

    size_kb = OUT_PATH.stat().st_size / 1024
    print(f"\nSalvo em: {OUT_PATH}")
    print(f"Tamanho:  {size_kb:.1f} KB")
    print(f"Pontos:   {len(compact_feats)}")
    print(f"Peso:     {PRIOR_WEIGHT} (prior fraco — calibração real sempre domina)")
    print("\nPronto! O prior será carregado automaticamente pelo calibration.py.")


if __name__ == "__main__":
    build_prior()
