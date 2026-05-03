"""
IrisFlow — Persistência de dados de calibração (SQLite)

Armazena pares (feature_normalizada, posição_tela, peso) de cada sessão
de calibração e os disponibiliza com decaimento exponencial de peso para
o GazeModel.fit(), enriquecendo o ajuste com histórico do próprio usuário.

API pública
-----------
save_session(profile, feats1, pos1, feats2, pos2, error_px) -> int
load_historical(max_sessions, decay, max_error_px) -> (feats, pos, weights)
list_sessions(limit) -> list[dict]
"""
from __future__ import annotations

import os
import sqlite3
import time
from typing import List, Optional, Tuple

import numpy as np

_DB_PATH = os.path.join(os.path.dirname(__file__), "calibration.db")

# ── Esquema ────────────────────────────────────────────────────────────────────

_DDL = """
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  REAL    NOT NULL,   -- Unix timestamp
    error_px    REAL,               -- erro de validação em pixels (NULL = não validado)
    dominant    TEXT,               -- 'left' | 'right'
    has_glasses INTEGER DEFAULT 0,  -- 0 | 1
    gaze_x_min  REAL,
    gaze_x_max  REAL,
    gaze_y_min  REAL,
    gaze_y_max  REAL,
    variance    REAL,
    n_phase1    INTEGER DEFAULT 0,
    n_phase2    INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS points (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    phase       INTEGER NOT NULL,   -- 1 | 2
    feature_x   REAL    NOT NULL,   -- feature normalizada pelo GazeProfile (eixo X)
    feature_y   REAL    NOT NULL,   -- feature normalizada pelo GazeProfile (eixo Y)
    pos_x       REAL    NOT NULL,   -- posição de tela normalizada [0, 1]
    pos_y       REAL    NOT NULL,
    weight      REAL    NOT NULL    -- 1.0 (fase 1) | 0.15 (fase 2)
);

CREATE INDEX IF NOT EXISTS idx_points_session ON points(session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_date  ON sessions(created_at DESC);
"""


# ── Conexão ────────────────────────────────────────────────────────────────────

def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.executescript(_DDL)
    return conn


# ── Escrita ────────────────────────────────────────────────────────────────────

def save_session(
    profile,
    feats1: List[np.ndarray],
    pos1:   List[Tuple[float, float]],
    feats2: List[np.ndarray],
    pos2:   List[Tuple[float, float]],
    error_px: Optional[float] = None,
) -> int:
    """
    Persiste uma sessão completa de calibração.

    Parâmetros
    ----------
    profile   : GazeProfile com gaze_x_min/max, gaze_y_min/max, dominant, has_glasses
    feats1/2  : listas de np.ndarray([fx, fy]) já normalizados pelo GazeProfile
    pos1/2    : posições de tela correspondentes, normalizadas [0, 1]
    error_px  : erro médio de validação em pixels (None se não validado)

    Retorna
    -------
    id da sessão criada no banco
    """
    conn = _connect()
    with conn:
        cur = conn.execute(
            """INSERT INTO sessions
               (created_at, error_px, dominant, has_glasses,
                gaze_x_min, gaze_x_max, gaze_y_min, gaze_y_max,
                variance, n_phase1, n_phase2)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                time.time(),
                error_px,
                getattr(profile, "dominant",    "left"),
                int(getattr(profile, "has_glasses", False)),
                getattr(profile, "gaze_x_min",  -0.20),
                getattr(profile, "gaze_x_max",   0.20),
                getattr(profile, "gaze_y_min",  -0.15),
                getattr(profile, "gaze_y_max",   0.15),
                getattr(profile, "variance",     0.001),
                len(feats1),
                len(feats2),
            ),
        )
        sid = cur.lastrowid

        rows: list = []
        for feat, pos in zip(feats1, pos1):
            rows.append((sid, 1, float(feat[0]), float(feat[1]),
                          float(pos[0]), float(pos[1]), 1.0))
        for feat, pos in zip(feats2, pos2):
            rows.append((sid, 2, float(feat[0]), float(feat[1]),
                          float(pos[0]), float(pos[1]), 0.15))

        conn.executemany(
            """INSERT INTO points
               (session_id, phase, feature_x, feature_y, pos_x, pos_y, weight)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
    conn.close()
    return sid


# ── Leitura ────────────────────────────────────────────────────────────────────

def load_historical(
    max_sessions: int   = 5,
    decay:        float = 0.60,
    max_error_px: float = 120.0,
) -> Tuple[List[np.ndarray], List[Tuple[float, float]], List[float]]:
    """
    Carrega pontos de calibrações anteriores com decaimento exponencial de peso.

    Sessões mais antigas recebem pesos menores:
        weight_histórico = peso_original × decay^(rank)   (rank 0 = mais recente)

    Parâmetros
    ----------
    max_sessions : quantas sessões passadas usar (padrão 5)
    decay        : fator de decaimento por sessão (padrão 0.60)
    max_error_px : ignora sessões com erro > este valor (padrão 120 px)

    Retorna
    -------
    (features, positions, weights) prontos para GazeModel.fit()
    Retorna listas vazias se banco não existe ou não há sessões.
    """
    if not os.path.exists(_DB_PATH):
        return [], [], []

    conn = _connect()
    sessions = conn.execute(
        """SELECT id FROM sessions
           WHERE error_px IS NULL OR error_px <= ?
           ORDER BY created_at DESC
           LIMIT ?""",
        (max_error_px, max_sessions),
    ).fetchall()

    features:  List[np.ndarray]            = []
    positions: List[Tuple[float, float]]   = []
    weights:   List[float]                 = []

    for rank, row in enumerate(sessions):
        age_factor = decay ** rank
        pts = conn.execute(
            """SELECT feature_x, feature_y, pos_x, pos_y, weight
               FROM points WHERE session_id = ?""",
            (row["id"],),
        ).fetchall()
        for p in pts:
            features.append(np.array([p["feature_x"], p["feature_y"]], dtype=np.float64))
            positions.append((p["pos_x"], p["pos_y"]))
            weights.append(p["weight"] * age_factor)

    conn.close()
    return features, positions, weights


# ── Listagem ───────────────────────────────────────────────────────────────────

def list_sessions(limit: int = 20) -> List[dict]:
    """
    Retorna lista das sessões mais recentes para exibição no dashboard.

    Cada dict contém: id, created_at (Unix ts), error_px, dominant,
    has_glasses, n_phase1, n_phase2.
    """
    if not os.path.exists(_DB_PATH):
        return []
    conn = _connect()
    rows = conn.execute(
        """SELECT id, created_at, error_px, dominant, has_glasses,
                  n_phase1, n_phase2
           FROM sessions
           ORDER BY created_at DESC
           LIMIT ?""",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
