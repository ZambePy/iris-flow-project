"""
IrisFlow — Camadas 4-5: Calibração em 3 Fases + Saída Suave

Camada 4  Calibração:
  Fase 0 — Assinatura ocular (varredura H+V, aprende range, detecta óculos/dominante)
  Fase 1 — 9 pontos centrais (grade 3×3, zona 65% da tela)
  Fase 2 — Trajetórias (cruz, espiral, figura-8, borda circular)
  Modelo — RBFInterpolator thin-plate-spline (scipy)

Camada 5  Saída suave:
  Filtro de Kalman 2D (filterpy) + EMA deadzone adaptativa + LERP 0.08
  Cursor mantém última posição quando íris não detectada
"""

import json
import os
import random
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger

try:
    from scipy.interpolate import RBFInterpolator
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False
    logger.warning("scipy não encontrado — pip install scipy")

try:
    from filterpy.kalman import KalmanFilter as _FPKalman
    _HAS_FILTERPY = True
except ImportError:
    _HAS_FILTERPY = False
    logger.warning("filterpy não encontrado — pip install filterpy")

from iris_tracker import IrisFrame, IrisTracker

_PROFILE_PATH = os.path.join(os.path.dirname(__file__), "user_profile.json")


# ── GazeProfile ───────────────────────────────────────────────────────────────

@dataclass
class GazeProfile:
    """
    Perfil ocular: range do vetor de olhar compensado pela pose da cabeça.
    Aprendido na Fase 0 (varredura H+V).
    """
    gaze_x_min:  float = -0.20
    gaze_x_max:  float =  0.20
    gaze_y_min:  float = -0.15
    gaze_y_max:  float =  0.15
    variance:    float =  0.001
    dominant:    str   = "left"
    has_glasses: bool  = False
    # Geometria física do setup
    monitor_inches:    float = 15.6
    camera_position:   str   = "top"
    distance_cm:       float = 60.0
    vertical_offset:   float = 0.0
    horizontal_offset: float = 0.0
    setup_configured:  bool  = False

    def normalize(self, gaze: np.ndarray) -> np.ndarray:
        """Escala gaze_feature para [0, 1] baseado no range aprendido."""
        x = (gaze[0] - self.gaze_x_min) / max(self.gaze_x_max - self.gaze_x_min, 1e-5)
        y = (gaze[1] - self.gaze_y_min) / max(self.gaze_y_max - self.gaze_y_min, 1e-5)
        return np.array([
            float(np.clip(x, -0.25, 1.25)),   # tolerância maior para olhares extremos
            float(np.clip(y, -0.25, 1.25)),
        ])

    def save(self) -> None:
        with open(_PROFILE_PATH, "w") as f:
            json.dump(asdict(self), f, indent=2)
        logger.info("Perfil salvo em {}", _PROFILE_PATH)

    @classmethod
    def load(cls) -> Optional["GazeProfile"]:
        if not os.path.exists(_PROFILE_PATH):
            return None
        try:
            with open(_PROFILE_PATH) as f:
                data = json.load(f)
            # Verifica se é o formato novo (gaze_x_min) ou antigo (left/right EyeProfile)
            if "gaze_x_min" not in data:
                return None
            return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        except Exception as e:
            logger.warning("Falha ao carregar perfil: {}", e)
            return None


# ── KalmanFilter2D ────────────────────────────────────────────────────────────

class KalmanFilter2D:
    """
    Filtro de Kalman 2D (estado: [x, y, vx, vy]) com amortecimento.
    Usa filterpy quando disponível; fallback manual caso contrário.
    """

    def __init__(
        self,
        dt:   float = 1 / 30,
        damp: float = 0.85,
        q:    float = 1.5,
        r:    float = 25.0,
    ) -> None:
        F = np.array([
            [1, 0, dt,   0   ],
            [0, 1, 0,    dt  ],
            [0, 0, damp, 0   ],
            [0, 0, 0,    damp],
        ], dtype=float)
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        Q = np.eye(4, dtype=float) * q
        R = np.eye(2, dtype=float) * r
        P = np.eye(4, dtype=float) * 500.0

        if _HAS_FILTERPY:
            kf       = _FPKalman(dim_x=4, dim_z=2)
            kf.F, kf.H, kf.Q, kf.R, kf.P = F, H, Q, R, P
            kf.x     = np.zeros((4, 1), dtype=float)
            self._kf = kf
            self._fp = True
        else:
            self._F, self._H, self._Q = F, H, Q
            self._R, self._P = R, P
            self._x  = np.zeros((4, 1), dtype=float)
            self._fp = False

        self._init = False

    def step(self, zx: float, zy: float) -> Tuple[float, float]:
        z = np.array([[zx], [zy]], dtype=float)
        if not self._init:
            if self._fp:
                self._kf.x[:2] = z
            else:
                self._x[:2] = z
            self._init = True
            return zx, zy
        if self._fp:
            self._kf.predict()
            self._kf.update(z)
            x = self._kf.x
        else:
            xp = self._F @ self._x
            Pp = self._F @ self._P @ self._F.T + self._Q
            S  = self._H @ Pp @ self._H.T + self._R
            K  = Pp @ self._H.T @ np.linalg.inv(S)
            self._x = xp + K @ (z - self._H @ xp)
            self._P = (np.eye(4) - K @ self._H) @ Pp
            x = self._x
        return float(x[0, 0]), float(x[1, 0])

    def predict_only(self) -> Tuple[float, float]:
        if not self._init:
            return 0.0, 0.0
        if self._fp:
            self._kf.predict()
            return float(self._kf.x[0, 0]), float(self._kf.x[1, 0])
        self._x = self._F @ self._x
        self._P = self._F @ self._P @ self._F.T + self._Q
        return float(self._x[0, 0]), float(self._x[1, 0])

    def reset(self) -> None:
        if self._fp:
            self._kf.x = np.zeros((4, 1), dtype=float)
            self._kf.P = np.eye(4, dtype=float) * 500.0
        else:
            self._x = np.zeros((4, 1), dtype=float)
            self._P = np.eye(4, dtype=float) * 500.0
        self._init = False


# ── AdaptiveDeadzone ──────────────────────────────────────────────────────────

class AdaptiveDeadzone:
    """Trava o cursor quando o olho está parado; libera após MAX_FROZEN frames."""

    BASE       = 0.030
    MIN        = 0.008
    MAX        = 0.055
    VEL_WIN    = 6
    MAX_FROZEN = 25

    def __init__(self) -> None:
        self._locked: Optional[Tuple[float, float]] = None
        self._thresh  = self.BASE
        self._frozen  = 0
        self._hist:   List[Tuple[float, float]] = []

    def check(self, pos: Tuple[float, float]) -> bool:
        self._hist.append(pos)
        if len(self._hist) > self.VEL_WIN:
            self._hist.pop(0)

        if self._locked is None:
            self._locked = pos
            self._frozen = 0
            return True

        dr = float(np.hypot(pos[0] - self._locked[0], pos[1] - self._locked[1]))

        if len(self._hist) >= 2:
            vel = float(np.mean([
                np.hypot(self._hist[i][0] - self._hist[i - 1][0],
                         self._hist[i][1] - self._hist[i - 1][1])
                for i in range(1, len(self._hist))
            ]))
            if vel > 0.035:
                self._thresh = max(self._thresh * 0.80, self.MIN)
            else:
                self._thresh = min(self._thresh * 1.05, self.MAX)

        if dr >= self._thresh or self._frozen >= self.MAX_FROZEN:
            self._locked = pos
            self._frozen = 0
            return True
        self._frozen += 1
        return False

    def reset(self) -> None:
        self._locked  = None
        self._thresh  = self.BASE
        self._frozen  = 0
        self._hist.clear()


# ── GazeModel ─────────────────────────────────────────────────────────────────

class GazeModel:
    """
    Mapeia feature normalizada → posição na tela.

    Modelo: RBFInterpolator (thin-plate spline) sobre gaze features normalizadas.
    Pós-processamento: Kalman 2D → deadzone adaptativa → LERP 0.08.
    """

    LERP         = 0.08
    TRAJ_SMOOTH  = 0.05
    FIXED_SMOOTH = 1e-3

    def __init__(self) -> None:
        self._rbf_x   = None
        self._rbf_y   = None
        self._fitted  = False
        self._fmean   = np.zeros(2)
        self._fstd    = np.ones(2)
        self._kalman  = KalmanFilter2D()
        self._dz      = AdaptiveDeadzone()
        self._cx      = -1.0    # -1 = não inicializado; definido na 1ª chamada de update()
        self._cy      = -1.0
        self._hx      = -1.0
        self._hy      = -1.0

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(
        self,
        features:  List[np.ndarray],
        positions: List[Tuple[float, float]],
        weights:   Optional[List[float]] = None,
    ) -> bool:
        if not _HAS_SCIPY:
            logger.error("scipy necessário — pip install scipy")
            return False
        n = len(features)
        if n < 4:
            logger.error("Mínimo 4 pontos para calibração (recebido {}).", n)
            return False

        X = np.array(features,  dtype=float)
        Y = np.array(positions, dtype=float)
        w = np.asarray(weights, dtype=float) if weights else np.ones(n)

        # Z-score das features
        self._fmean = X.mean(axis=0)
        self._fstd  = np.maximum(X.std(axis=0), 1e-6)
        Xn = (X - self._fmean) / self._fstd

        # Remove outliers de trajetória (resíduo > 2.5× mediana)
        traj_mask = w < 1.0
        if traj_mask.sum() >= 20:
            A  = np.column_stack([Xn, np.ones(n)])
            cx, *_ = np.linalg.lstsq(A * w[:, None], Y[:, 0] * w, rcond=None)
            cy, *_ = np.linalg.lstsq(A * w[:, None], Y[:, 1] * w, rcond=None)
            res = np.hypot(Y[:, 0] - A @ cx, Y[:, 1] - A @ cy)
            med = float(np.median(res[traj_mask]))
            bad = traj_mask & (res > 2.5 * max(med, 1e-6))
            if bad.sum():
                logger.info("Outliers removidos: {}/{}", int(bad.sum()), int(traj_mask.sum()))
            Xn, Y, w = Xn[~bad], Y[~bad], w[~bad]

        smooth = np.where(w >= 1.0, self.FIXED_SMOOTH, self.TRAJ_SMOOTH)

        try:
            self._rbf_x = RBFInterpolator(
                Xn, Y[:, 0], kernel="thin_plate_spline", smoothing=smooth, degree=1)
            self._rbf_y = RBFInterpolator(
                Xn, Y[:, 1], kernel="thin_plate_spline", smoothing=smooth, degree=1)
            self._fitted = True
            logger.info("GazeModel ajustado com {} pontos.", len(Xn))
            return True
        except Exception as e:
            logger.error("Falha no RBF: {}", e)
            return False

    def predict_raw(self, feat: np.ndarray) -> Optional[Tuple[float, float]]:
        if not self._fitted:
            return None
        fn = ((feat - self._fmean) / self._fstd).reshape(1, -1)
        x  = float(np.clip(self._rbf_x(fn)[0], -0.1, 1.1))
        y  = float(np.clip(self._rbf_y(fn)[0], -0.1, 1.1))
        return x, y

    def update(self, feat: np.ndarray, sw: int, sh: int) -> Tuple[int, int]:
        """
        Atualiza posição do cursor: Kalman + deadzone + LERP.
        Se íris não detectada, cursor permanece na última posição.
        """
        # Inicializa posição no centro da tela na primeira chamada
        if self._cx < 0:
            self._cx = self._hx = float(sw) / 2
            self._cy = self._hy = float(sh) / 2

        pred = self.predict_raw(feat)
        if pred is not None:
            if self._dz.check(pred):
                tx, ty = self._kalman.step(pred[0] * sw, pred[1] * sh)
                self._cx += self.LERP * (tx - self._cx)
                self._cy += self.LERP * (ty - self._cy)
                self._hx, self._hy = self._cx, self._cy
            else:
                self._kalman.predict_only()
                self._cx, self._cy = self._hx, self._hy
        return int(np.clip(self._cx, 0, sw - 1)), int(np.clip(self._cy, 0, sh - 1))

    def reset_smoothing(self) -> None:
        self._kalman.reset()
        self._dz.reset()
        self._cx = self._cy = self._hx = self._hy = -1.0   # reinicia posição


# ── Helpers de UI ─────────────────────────────────────────────────────────────

def _text_center(canvas, text, y, scale=0.85, color=(200, 200, 200), thick=1):
    sw = canvas.shape[1]
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    cv2.putText(canvas, text, ((sw - tw) // 2, y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)


def _progress_bar(canvas, x, y, w, h, progress, color=(0, 150, 255)):
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (45, 45, 45), -1)
    cv2.rectangle(canvas, (x, y), (x + int(w * progress), y + h), color, -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (80, 80, 80), 2)


def _thumb(canvas, frame, tw=192, th=144):
    t = cv2.resize(cv2.flip(frame, 1), (tw, th))
    canvas[8:8 + th, 8:8 + tw] = t


# ── FASE 0 — Assinatura Ocular ────────────────────────────────────────────────

def _run_phase0(
    tracker: IrisTracker,
    sw: int, sh: int,
    win: str,
) -> GazeProfile:
    """
    Varredura H (5s) + V (5s): aprende range de gaze, detecta óculos e olho dominante.
    """

    SWEEP_FRAMES = 90       # ~3s @ 30fps
    MARGIN       = 0.05

    for ct in range(3, 0, -1):
        canvas = np.zeros((sh, sw, 3), np.uint8)
        _text_center(canvas, f"Fase 0 — Assinatura ocular em {ct}...",
                     sh // 2 - 30, 1.2, (0, 200, 255), 2)
        _text_center(canvas, "Siga o ponto com os olhos",
                     sh // 2 + 40, 0.85, (150, 150, 150), 1)
        cv2.imshow(win, canvas)
        cv2.waitKey(1000)

    def _collect_sweep(h_sweep: bool):
        targets, features, l_gazes, r_gazes = [], [], [], []
        last_annotated = None
        fi = 0

        while fi < SWEEP_FRAMES:
            frame = tracker.read_frame(timeout=0.010)

            t   = fi / max(SWEEP_FRAMES - 1, 1)
            pos = MARGIN + t * (1.0 - 2.0 * MARGIN)
            tx, ty = (pos, 0.5) if h_sweep else (0.5, pos)

            if frame is not None:
                iris_frame, last_annotated = tracker.process_frame(frame, draw=True)
                if iris_frame is not None:
                    targets.append((tx, ty))
                    features.append(iris_frame.gaze_feature.copy())
                    l_gazes.append(iris_frame.left_gaze[:2].copy())
                    r_gazes.append(iris_frame.right_gaze[:2].copy())
                fi += 1

            canvas = np.zeros((sh, sw, 3), np.uint8)
            px, py = int(tx * sw), int(ty * sh)
            pulse  = 0.5 + 0.5 * np.sin(time.time() * 6)
            cv2.circle(canvas, (px, py), int(18 + 6 * pulse), (0, 200, 255), 3)
            cv2.circle(canvas, (px, py), 10, (255, 255, 255), -1)
            label  = "Fase 0: horizontal — siga o ponto" if h_sweep else "Fase 0: vertical — siga o ponto"
            _text_center(canvas, label, 50, 0.85, (180, 180, 180), 1)
            bw, bx = 500, (sw - 500) // 2
            _progress_bar(canvas, bx, sh - 60, bw, 12, fi / SWEEP_FRAMES)
            if last_annotated is not None:
                _thumb(canvas, last_annotated)
            cv2.imshow(win, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        return targets, features, l_gazes, r_gazes

    # ── Sweep H ───────────────────────────────────────────────────────────────
    tgts_h, feats_h, l_h, r_h = _collect_sweep(h_sweep=True)

    for _ in range(45):
        canvas = np.zeros((sh, sw, 3), np.uint8)
        _text_center(canvas, "Agora vertical...", sh // 2, 1.1, (0, 200, 255), 2)
        cv2.imshow(win, canvas)
        cv2.waitKey(33)

    # ── Sweep V ───────────────────────────────────────────────────────────────
    tgts_v, feats_v, l_v, r_v = _collect_sweep(h_sweep=False)

    # ── Range por percentil (robusto a usuários que rastreiam com a cabeça) ──────
    # Usa TODOS os frames de ambas as varreduras: H sweep cobre bem gx,
    # V sweep cobre bem gy, mas cruzar os dados garante um piso mínimo razoável.

    def _percentile_range(vals, min_range: float = 0.05):
        if len(vals) < 10:
            return -min_range / 2, min_range / 2, 0.001
        v   = np.array(vals, dtype=float)
        p2  = float(np.percentile(v, 2))
        p98 = float(np.percentile(v, 98))
        actual = p98 - p2
        # Garante range mínimo centrado na mediana
        if actual < min_range:
            mid = float(np.median(v))
            p2  = mid - min_range / 2
            p98 = mid + min_range / 2
            actual = min_range
        margin = 0.10 * actual     # 10% de margem extra em cada lado
        return p2 - margin, p98 + margin, float(np.var(v))

    all_feats = feats_h + feats_v
    gx_vals   = [f[0] for f in all_feats]
    gy_vals   = [f[1] for f in all_feats]

    gx_min, gx_max, gx_var = _percentile_range(gx_vals, min_range=0.06)
    gy_min, gy_max, gy_var = _percentile_range(gy_vals, min_range=0.06)

    # ── Detecção de olho dominante (maior correlação com target) ──────────────
    dominant = "left"
    n_h = min(len(tgts_h), len(l_h), len(r_h))
    if n_h >= 10:
        t_arr = np.array([v[0] for v in tgts_h[:n_h]])
        lx    = np.array([v[0] for v in l_h[:n_h]])
        rx    = np.array([v[0] for v in r_h[:n_h]])
        cl    = abs(float(np.corrcoef(t_arr, lx)[0, 1])) if np.std(lx) > 1e-8 else 0.0
        cr    = abs(float(np.corrcoef(t_arr, rx)[0, 1])) if np.std(rx) > 1e-8 else 0.0
        dominant = "left" if cl >= cr else "right"

    # ── Detecção de óculos (variância relativa alta) ──────────────────────────
    avg_var     = (gx_var + gy_var) / 2
    gx_range    = max(gx_max - gx_min, 1e-5)
    gy_range    = max(gy_max - gy_min, 1e-5)
    rel_var     = (gx_var / gx_range ** 2 + gy_var / gy_range ** 2) / 2
    has_glasses = rel_var > 0.06

    profile = GazeProfile(
        gaze_x_min  = gx_min,
        gaze_x_max  = gx_max,
        gaze_y_min  = gy_min,
        gaze_y_max  = gy_max,
        variance    = avg_var,
        dominant    = dominant,
        has_glasses = has_glasses,
    )

    # ── Tela de resultado ─────────────────────────────────────────────────────
    canvas = np.zeros((sh, sw, 3), np.uint8)
    _text_center(canvas, "Fase 0 concluída!", sh // 2 - 100, 1.2, (0, 200, 255), 2)
    _text_center(canvas, f"Olho dominante: {dominant.upper()}", sh // 2 - 20, 0.85, (200, 200, 200), 1)
    gl_txt = "Óculos: detectado" if has_glasses else "Óculos: não detectado"
    _text_center(canvas, gl_txt, sh // 2 + 30, 0.80, (180, 180, 180), 1)
    _text_center(canvas, "[ SPACE ] continuar", sh // 2 + 100, 0.85, (100, 220, 100), 1)
    cv2.imshow(win, canvas)
    while cv2.waitKey(50) & 0xFF != ord(" "):
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break

    profile.save()
    logger.info(
        "Fase 0: dominante={} óculos={} gx=[{:.4f},{:.4f}] gy=[{:.4f},{:.4f}]",
        dominant, has_glasses, gx_min, gx_max, gy_min, gy_max,
    )
    return profile


# ── FASE 1 — 9 Pontos Centrais ────────────────────────────────────────────────

def _run_phase1(
    tracker:  IrisTracker,
    profile:  GazeProfile,
    sw: int, sh: int,
    win: str,
) -> Tuple[List[np.ndarray], List[Tuple[float, float]]]:
    """Grade 3×3 na zona central (65% da tela). 2s contagem + 1.5s coleta por ponto."""

    M = 0.175   # margem para zona central: (1-0.65)/2
    POINTS = [
        (M,      M),     (0.5,    M),     (1 - M, M),
        (M,      0.5),   (0.5,    0.5),   (1 - M, 0.5),
        (M,      1 - M), (0.5,    1 - M), (1 - M, 1 - M),
    ]
    COUNTDOWN = 1.2
    COLLECT   = 1.0

    feats:    List[np.ndarray]          = []
    pos_list: List[Tuple[float, float]] = []
    done:     List[int]                 = []

    def _draw_dots(canvas, active_idx):
        for i, p in enumerate(POINTS):
            cx2, cy2 = int(p[0] * sw), int(p[1] * sh)
            if i in done:
                cv2.circle(canvas, (cx2, cy2), 10, (0, 200, 80),  -1)
                cv2.circle(canvas, (cx2, cy2), 14, (0, 140, 60),   2)
            elif i != active_idx:
                cv2.circle(canvas, (cx2, cy2), 10, (55, 55, 55),  -1)
                cv2.circle(canvas, (cx2, cy2), 14, (40, 40, 40),   2)

    for idx, pos in enumerate(POINTS):
        px, py        = int(pos[0] * sw), int(pos[1] * sh)
        last_annotated = None

        # ── Contagem regressiva ───────────────────────────────────────────────
        t_end = time.time() + COUNTDOWN
        while time.time() < t_end:
            frame = tracker.read_frame(timeout=0.010)
            if frame is not None:
                _, last_annotated = tracker.process_frame(frame, draw=True)
            progress = 1.0 - (t_end - time.time()) / COUNTDOWN
            canvas   = np.zeros((sh, sw, 3), np.uint8)
            _draw_dots(canvas, idx)
            cv2.ellipse(canvas, (px, py), (42, 42), -90, 0, int(360 * progress), (0, 180, 255), 5)
            cv2.circle(canvas, (px, py), 12, (255, 255, 255), -1)
            cv2.circle(canvas, (px, py),  5, (0,  100, 255), -1)
            _text_center(canvas, f"Olhe para o ponto  {idx + 1} / {len(POINTS)}",
                         46, 0.88, (180, 180, 180), 2)
            if last_annotated is not None:
                _thumb(canvas, last_annotated)
            cv2.imshow(win, canvas)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return feats, pos_list

        # ── Coleta ───────────────────────────────────────────────────────────
        collected: List[np.ndarray] = []
        t_end = time.time() + COLLECT
        while time.time() < t_end:
            frame = tracker.read_frame(timeout=0.010)
            if frame is not None:
                iris_frame, last_annotated = tracker.process_frame(frame, draw=True)
                if iris_frame is not None:
                    collected.append(profile.normalize(iris_frame.gaze_feature))
            canvas = np.zeros((sh, sw, 3), np.uint8)
            _draw_dots(canvas, idx)
            cv2.circle(canvas, (px, py), 12, (255, 255, 255), -1)
            cv2.circle(canvas, (px, py),  5, (0,  100, 255), -1)
            _text_center(canvas, "Capturando...", 46, 0.88, (0, 255, 100), 2)
            if last_annotated is not None:
                _thumb(canvas, last_annotated)
            cv2.imshow(win, canvas)
            cv2.waitKey(1)

        if collected:
            stacked = np.array(collected)
            median  = np.median(stacked, axis=0)
            feats.append(median)
            pos_list.append(pos)
            done.append(idx)
            logger.info("Ponto {} capturado: feat=({:.4f}, {:.4f})",
                        idx + 1, float(median[0]), float(median[1]))
        else:
            logger.warning("Ponto {} sem amostras — pulado.", idx + 1)

        if idx < len(POINTS) - 1:
            cv2.waitKey(400)

    return feats, pos_list


# ── FASE 2 — Trajetórias ──────────────────────────────────────────────────────

def _run_phase2(
    tracker: IrisTracker,
    profile: GazeProfile,
    sw: int, sh: int,
    win: str,
) -> Tuple[List[np.ndarray], List[Tuple[float, float]]]:
    """
    4 trajetórias: cruz H, cruz V, espiral e figura-8.
    Lag oculomotor compensado com buffer de 4 frames (~130ms).
    """

    CROSS_F  = 180
    SPIRAL_F = 300
    INF_F    = 240
    LAG      = 4
    SAMPLE_N = 2

    def _hcross():
        return [(0.05 + 0.90 * i / (CROSS_F - 1), 0.50) for i in range(CROSS_F)]

    def _vcross():
        return [(0.50, 0.05 + 0.90 * i / (CROSS_F - 1)) for i in range(CROSS_F)]

    def _spiral():
        pts, mr = [], min(sw, sh) * 0.42
        for i in range(SPIRAL_F):
            t = i / SPIRAL_F
            θ = t * 4 * np.pi
            pts.append((
                float(np.clip(0.5 + t * mr * np.cos(θ) / sw, 0.05, 0.95)),
                float(np.clip(0.5 + t * mr * np.sin(θ) / sh, 0.05, 0.95)),
            ))
        return pts

    def _figure8():
        pts = []
        for i in range(INF_F):
            t = np.pi / 2 + i / INF_F * 2 * np.pi
            pts.append((
                float(np.clip(0.5 + 0.40 * np.cos(t),      0.05, 0.95)),
                float(np.clip(0.5 + 0.28 * np.sin(2 * t),  0.05, 0.95)),
            ))
        return pts

    phases = [
        ("Fase 2a — Cruz horizontal: siga o alvo", _hcross()),
        ("Fase 2b — Cruz vertical: siga o alvo",   _vcross()),
        ("Fase 2c — Espiral: siga o alvo",         _spiral()),
        ("Fase 2d — Figura-8: siga o alvo",        _figure8()),
    ]

    all_feats: List[np.ndarray]          = []
    all_pos:   List[Tuple[float, float]] = []

    for label, traj in phases:
        feat_buf = deque(maxlen=LAG + 1)
        fi, skip = 0, False

        while fi < len(traj):
            frame = tracker.read_frame(timeout=0.010)

            target = traj[min(fi, len(traj) - 1)]

            if frame is not None:
                iris_frame, annotated = tracker.process_frame(frame, draw=False)
                feat = None
                if iris_frame is not None:
                    feat = profile.normalize(iris_frame.gaze_feature)
                feat_buf.append(feat)

                if fi % SAMPLE_N == 0 and len(feat_buf) == LAG + 1:
                    lagged = feat_buf[0]
                    if lagged is not None:
                        all_feats.append(lagged)
                        all_pos.append(target)
                fi += 1

            # ── Desenho ──────────────────────────────────────────────────────
            canvas = np.zeros((sh, sw, 3), np.uint8)
            for j in range(fi, min(fi + 40, len(traj))):
                fx2, fy2 = int(traj[j][0] * sw), int(traj[j][1] * sh)
                fade = 1.0 - (j - fi) / 40
                c    = int(50 * fade)
                cv2.circle(canvas, (fx2, fy2), 3, (c, c * 2, c * 3), -1)
            tx2, ty2 = int(target[0] * sw), int(target[1] * sh)
            pulse    = 0.5 + 0.5 * np.sin(time.time() * 7)
            cv2.circle(canvas, (tx2, ty2), int(20 + 6 * pulse), (0, 150, 255), 3)
            cv2.circle(canvas, (tx2, ty2), 12, (255, 255, 255), -1)
            cv2.circle(canvas, (tx2, ty2),  5, (0,  100, 255), -1)
            bw, bx = 480, (sw - 480) // 2
            _progress_bar(canvas, bx, sh - 58, bw, 12, fi / len(traj))
            _text_center(canvas, label, 46, 0.78, (180, 180, 180), 1)
            cv2.putText(canvas, f"Amostras: {len(all_feats)}",
                        (20, sh - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (70, 70, 70), 1)
            _text_center(canvas, "[ SPACE ] próxima fase   [ Q ] sair",
                         sh - 20, 0.50, (60, 60, 60), 1)
            cv2.imshow(win, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return all_feats, all_pos
            if key == ord(" "):
                skip = True
                break

        if skip:
            continue

    return all_feats, all_pos


# ── VALIDAÇÃO ─────────────────────────────────────────────────────────────────

def _run_validation(
    tracker: IrisTracker,
    profile: GazeProfile,
    model:   GazeModel,
    sw: int, sh: int,
    win: str,
) -> Tuple[float, float, float]:
    """
    3 pontos fora do grid de calibração; mede erro médio em pixels.
    Retorna (avg_err, sys_dx, sys_dy) onde sys_dx/dy são offsets normalizados
    detectados como erro sistemático (positivo = cursor à direita/abaixo do alvo).
    """

    VAL_POOL = [
        (0.20, 0.25), (0.80, 0.20), (0.50, 0.12),
        (0.15, 0.65), (0.85, 0.72), (0.45, 0.82),
        (0.72, 0.48), (0.28, 0.78), (0.62, 0.18),
    ]
    pts     = random.sample(VAL_POOL, 3)
    errors: List[float]              = []
    err_vx: List[float]              = []
    err_vy: List[float]              = []

    for i, pos in enumerate(pts):
        px, py    = int(pos[0] * sw), int(pos[1] * sh)
        collected: List[Tuple[float, float]] = []
        t_end     = time.time() + 2.0

        while time.time() < t_end:
            frame = tracker.read_frame(timeout=0.010)
            if frame is not None:
                iris_frame, _ = tracker.process_frame(frame, draw=False)
                if iris_frame is not None:
                    feat = profile.normalize(iris_frame.gaze_feature)
                    pred = model.predict_raw(feat)
                    if pred is not None:
                        collected.append(pred)
            progress = 1.0 - (t_end - time.time()) / 2.0
            canvas   = np.zeros((sh, sw, 3), np.uint8)
            _text_center(canvas, f"Validação {i + 1}/3 — olhe para o ponto",
                         46, 0.88, (180, 180, 180), 2)
            cv2.ellipse(canvas, (px, py), (40, 40), -90, 0, int(360 * progress), (0, 180, 255), 4)
            cv2.circle(canvas, (px, py), 12, (255, 255, 255), -1)
            cv2.circle(canvas, (px, py),  5, (0,  100, 255), -1)
            cv2.imshow(win, canvas)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return 9999.0, 0.0, 0.0

        if collected:
            med = np.median(np.array(collected), axis=0)
            ex  = float(med[0]) - pos[0]   # normalizado [0,1]
            ey  = float(med[1]) - pos[1]
            err_vx.append(ex)
            err_vy.append(ey)
            errors.append(float(np.hypot(ex * sw, ey * sh)))

    avg = float(np.mean(errors)) if errors else 9999.0

    # Detecção de erro sistemático: mediana do vetor de erro
    sys_dx = float(np.median(err_vx)) if err_vx else 0.0
    sys_dy = float(np.median(err_vy)) if err_vy else 0.0

    # Direção do desvio para exibição
    direction_lines: List[str] = []
    if abs(sys_dy) > 0.05:
        direction_lines.append("cursor desviando para " + ("CIMA" if sys_dy < 0 else "BAIXO"))
    if abs(sys_dx) > 0.05:
        direction_lines.append("cursor desviando para " + ("ESQUERDA" if sys_dx < 0 else "DIREITA"))
    systematic = bool(direction_lines)

    canvas = np.zeros((sh, sw, 3), np.uint8)
    _text_center(canvas, "Resultado da Validação", sh // 2 - 120, 1.1, (0, 200, 255), 2)
    for j, err in enumerate(errors):
        col = (0, 220, 80) if err < 80 else ((0, 140, 255) if err < 150 else (0, 60, 220))
        _text_center(canvas, f"Ponto {j + 1}: {err:.0f} px",
                     sh // 2 - 30 + j * 48, 0.9, col, 2)
    avg_c = (0, 220, 80) if avg < 80 else ((0, 140, 255) if avg < 150 else (0, 60, 220))
    _text_center(canvas, f"Erro médio: {avg:.0f} px", sh // 2 + 120, 1.0, avg_c, 2)

    if systematic:
        for k, line in enumerate(direction_lines):
            _text_center(canvas, line, sh // 2 + 168 + k * 32, 0.72, (0, 170, 255), 1)
        _text_center(canvas, "[ A ] aplicar correção automática",
                     sh // 2 + 240, 0.72, (0, 220, 130), 1)

    _text_center(canvas, "[ SPACE ] continuar   [ R ] recalibrar",
                 sh // 2 + 280, 0.75, (100, 220, 100), 1)
    cv2.imshow(win, canvas)

    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == ord(" "):
            return avg, 0.0, 0.0
        if key == ord("r"):
            return 9999.0, 0.0, 0.0
        if key == ord("a") and systematic:
            logger.info("Correção automática aplicada: dx={:.3f} dy={:.3f}", sys_dx, sys_dy)
            return avg, sys_dx, sys_dy
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            return avg, 0.0, 0.0


# ── Correção geométrica ────────────────────────────────────────────────────────

def _apply_geometric_correction(
    positions: List[Tuple[float, float]],
    profile: "GazeProfile",
) -> List[Tuple[float, float]]:
    """
    Corrige posições de calibração pela geometria física do setup.
    Aplica offset de câmera e escala vertical para monitores grandes.
    """
    v_off = profile.vertical_offset
    if profile.camera_position == "top":
        v_off += 0.08
    elif profile.camera_position == "bottom":
        v_off -= 0.06

    h_off = profile.horizontal_offset

    v_scale = 1.0
    if profile.monitor_inches >= 23.0:
        v_scale = 1.0 + 0.15 * (profile.monitor_inches - 15.6) / 10.0

    corrected = []
    for px, py in positions:
        new_y = py * v_scale + v_off
        new_x = px + h_off
        corrected.append((
            float(np.clip(new_x, 0.0, 1.0)),
            float(np.clip(new_y, 0.0, 1.0)),
        ))
    return corrected


# ── CalibrationSession ────────────────────────────────────────────────────────

class CalibrationSession:
    """
    Orquestra o fluxo completo:
        setup → Fase 0 → Fase 1 → Fase 2 → modelo → validação → teclado
    """

    WIN         = "IrisFlow"
    TRAJ_WEIGHT = 0.15

    def __init__(self, tracker: IrisTracker, settings=None, bridge=None) -> None:
        self.tracker      = tracker
        self._settings    = settings
        self._bridge      = bridge
        self._start_event = threading.Event()
        self._sw, self._sh = self._screen_size()
        self.model        = GazeModel()

    @staticmethod
    def _screen_size() -> Tuple[int, int]:
        try:
            import tkinter as tk
            root = tk.Tk()
            root.withdraw()
            w, h = root.winfo_screenwidth(), root.winfo_screenheight()
            root.destroy()
            return w, h
        except Exception:
            return 1280, 720

    def _alive(self) -> bool:
        return cv2.getWindowProperty(self.WIN, cv2.WND_PROP_VISIBLE) >= 1

    # ── Setup Físico ──────────────────────────────────────────────────────────

    def _physical_setup_screen(self, profile: "GazeProfile") -> "GazeProfile":
        """
        Tela de configuração do setup físico (monitor/câmera/distância).
        Navegada por teclado: [1–4] monitor, [5–7] câmera, [8–0] distância, [ENTER] confirmar.
        Exibida sempre que setup_configured=False; nas seguintes mostra valores salvos.
        """
        _MONITORS  = [("15\"–17\"", 16.0), ("19\"–22\"", 20.0),
                      ("23\"–27\"", 24.0), ("28\"+",      30.0)]
        _CAMERAS   = [("Em cima",   "top"), ("Embaixo", "bottom"), ("Lateral", "side")]
        _DISTANCES = [("Perto ~40cm", 40.0), ("Normal ~60cm", 60.0), ("Longe ~80cm+", 85.0)]

        mon_idx  = next((i for i, (_, v) in enumerate(_MONITORS)  if v == profile.monitor_inches), 0)
        cam_idx  = next((i for i, (_, v) in enumerate(_CAMERAS)   if v == profile.camera_position), 0)
        dist_idx = next((i for i, (_, v) in enumerate(_DISTANCES) if v == profile.distance_cm), 1)

        sw, sh = self._sw, self._sh
        win    = self.WIN

        while True:
            canvas = np.zeros((sh, sw, 3), np.uint8)
            _text_center(canvas, "IrisFlow — Setup Físico", 50, 1.1, (0, 200, 255), 2)

            def _row(label, options, sel, y0, key_offset):
                cv2.putText(canvas, label, (sw // 2 - 380, y0),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.70, (150, 150, 150), 1)
                bw = 160
                gap = 14
                total_w = len(options) * bw + (len(options) - 1) * gap
                bx = (sw - total_w) // 2
                for i, (txt, _) in enumerate(options):
                    x0b = bx + i * (bw + gap)
                    bg  = (0, 80, 30) if i == sel else (45, 45, 45)
                    bd  = (0, 220, 80) if i == sel else (80, 80, 80)
                    cv2.rectangle(canvas, (x0b, y0 - 28), (x0b + bw, y0 + 12), bg, -1)
                    cv2.rectangle(canvas, (x0b, y0 - 28), (x0b + bw, y0 + 12), bd, 2)
                    num = str(key_offset + i + 1) if key_offset + i < 9 else "0"
                    cv2.putText(canvas, f"[{num}] {txt}",
                                (x0b + 8, y0 + 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.56,
                                (0, 255, 100) if i == sel else (200, 200, 200), 1)

            _row("Monitor",  _MONITORS,  mon_idx,  sh // 2 - 100, 0)
            _row("Câmera",   _CAMERAS,   cam_idx,  sh // 2,       4)
            _row("Distância", _DISTANCES, dist_idx, sh // 2 + 100, 7)

            _text_center(canvas, "[ ENTER ] confirmar e continuar",
                         sh // 2 + 190, 0.85, (100, 220, 100), 2)
            if profile.setup_configured:
                _text_center(canvas, "Setup salvo anteriormente — altere se necessário",
                             sh // 2 - 160, 0.70, (120, 120, 120), 1)
            cv2.imshow(win, canvas)

            key = cv2.waitKey(50) & 0xFF
            if key == ord("1"):   mon_idx  = 0
            elif key == ord("2"): mon_idx  = 1
            elif key == ord("3"): mon_idx  = 2
            elif key == ord("4"): mon_idx  = 3
            elif key == ord("5"): cam_idx  = 0
            elif key == ord("6"): cam_idx  = 1
            elif key == ord("7"): cam_idx  = 2
            elif key == ord("8"): dist_idx = 0
            elif key == ord("9"): dist_idx = 1
            elif key == ord("0"): dist_idx = 2
            elif key == 13:  # ENTER
                profile.monitor_inches   = _MONITORS[mon_idx][1]
                profile.camera_position  = _CAMERAS[cam_idx][1]
                profile.distance_cm      = _DISTANCES[dist_idx][1]
                profile.setup_configured = True
                profile.save()
                logger.info(
                    "Setup físico: {}\" câmera={} dist={}cm",
                    profile.monitor_inches, profile.camera_position, profile.distance_cm,
                )
                return profile
            elif key == ord("q") or not self._alive():
                return profile

    # ── Auto-recalibração silenciosa ──────────────────────────────────────────

    def _silent_recalibrate(self, feat: np.ndarray, cx: int, cy: int) -> None:
        """Adiciona uma amostra pseudo-rotulada e re-ajusta o modelo sem interromper."""
        if not hasattr(self, "_calib_all_feats"):
            return
        try:
            new_pos = (cx / max(self._sw, 1), cy / max(self._sh, 1))
            new_feats   = self._calib_all_feats   + [feat]
            new_pos_all = self._calib_all_pos     + [new_pos]
            new_weights = self._calib_all_weights + [0.3]
            self.model.fit(new_feats, new_pos_all, new_weights)
            self._calib_all_feats   = new_feats
            self._calib_all_pos     = new_pos_all
            self._calib_all_weights = new_weights
            logger.debug("Auto-recal silenciosa: {} pontos totais.", len(new_feats))
        except Exception as e:
            logger.debug("Auto-recal silenciosa falhou: {}", e)

    # ── Tela de Setup ─────────────────────────────────────────────────────────

    def _setup_screen(self) -> bool:
        last_annotated = None
        while True:
            frame = self.tracker.read_frame(timeout=0.010)
            iris_data = None
            if frame is not None:
                iris_data, last_annotated = self.tracker.process_frame(frame, draw=True)

            gray     = cv2.cvtColor(frame if frame is not None else
                                    np.zeros((480, 640, 3), np.uint8),
                                    cv2.COLOR_BGR2GRAY)
            light    = float(np.mean(gray))
            face_ok  = iris_data is not None
            light_ok = 60 <= light <= 200

            pw  = min(int(self._sw * 0.54), 800)
            ph  = int(pw * 9 / 16)
            px0 = (self._sw - pw) // 2
            py0 = 80

            canvas = np.zeros((self._sh, self._sw, 3), np.uint8)
            if last_annotated is not None:
                thumb = cv2.resize(cv2.flip(last_annotated, 1), (pw, ph))
                canvas[py0:py0 + ph, px0:px0 + pw] = thumb

            oval_c = (0, 220, 80) if (face_ok and light_ok) else (80, 80, 80)
            cv2.ellipse(canvas, (self._sw // 2, py0 + ph // 2),
                        (int(pw * 0.21), int(ph * 0.42)), 0, 0, 360, oval_c, 3)

            checks = [("Íris detectada", face_ok), ("Iluminação adequada", light_ok)]
            all_ok = all(ok for _, ok in checks)
            by     = py0 + ph + 52
            for i, (lbl, ok) in enumerate(checks):
                col = (0, 220, 80) if ok else (60, 80, 220)
                cv2.circle(canvas, (self._sw // 2 - 250, by + i * 44 - 6), 9, col, -1)
                cv2.putText(canvas, lbl, (self._sw // 2 - 224, by + i * 44),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.70, col, 2)

            btn_y = by + len(checks) * 44 + 30
            if all_ok:
                cv2.rectangle(canvas,
                              (self._sw // 2 - 250, btn_y),
                              (self._sw // 2 + 250, btn_y + 48), (0, 55, 20), -1)
                cv2.rectangle(canvas,
                              (self._sw // 2 - 250, btn_y),
                              (self._sw // 2 + 250, btn_y + 48), (0, 200, 70), 2)
                _text_center(canvas, "[ ENTER ] Iniciar calibração",
                             btn_y + 33, 0.90, (0, 255, 100), 2)
            else:
                _text_center(canvas, "Ajuste o setup acima", btn_y + 26, 0.80, (100, 100, 100), 1)

            _text_center(canvas, "IrisFlow — Setup", 44, 1.0, (0, 200, 255), 2)
            _text_center(canvas, "[ SPACE ] pular calibração   [ Q ] sair",
                         self._sh - 18, 0.52, (60, 60, 60), 1)
            cv2.imshow(self.WIN, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or not self._alive():
                return False
            if key == 13 and all_ok:
                return True
            if key == ord(" "):
                return True

    # ── Loop de Rastreamento ──────────────────────────────────────────────────

    def _run_tracking(self, profile: GazeProfile) -> bool:
        """
        Loop de rastreamento ao vivo. Retorna True=recalibrar, False=sair.
        Cursor mantém última posição quando íris não detectada (sem estado cinza).
        Tenta rodar a 60fps.
        """
        from virtual_keyboard import VirtualKeyboard

        _speak = None
        try:
            from tts import TTSEngine
            _tts = TTSEngine()
            def _speak(text: str) -> None:
                _tts.speak_async(text)
        except Exception:
            pass

        keyboard = VirtualKeyboard(
            screen_width  = self._sw,
            screen_height = self._sh,
            dwell_time    = 1.5,
            on_speak      = _speak,
        )
        self.model.reset_smoothing()

        TH_W, TH_H          = 200, 150
        DRAW_DT              = 1.0 / 60
        AUTO_RECAL_INTERVAL  = 300
        last_draw            = 0.0
        last_annotated       = None
        cx, cy               = self._sw // 2, self._sh // 2
        frame_count          = 0

        logger.info("Rastreamento ativo.  R=recalibrar  Q=sair")

        while True:
            frame = self.tracker.read_frame(timeout=0.010)
            if frame is not None:
                iris_frame, last_annotated = self.tracker.process_frame(frame, draw=True)
                if iris_frame is not None:
                    feat   = profile.normalize(iris_frame.gaze_feature)
                    cx, cy = self.model.update(feat, self._sw, self._sh)
                    frame_count += 1
                    if frame_count % AUTO_RECAL_INTERVAL == 0:
                        self._silent_recalibrate(feat, cx, cy)

            keyboard.update(cx, cy)

            now = time.time()
            if now - last_draw >= DRAW_DT:
                last_draw = now
                canvas    = np.zeros((self._sh, self._sw, 3), np.uint8)
                keyboard.draw(canvas)

                r = max(18, self._sh // 38)
                cv2.circle(canvas, (cx, cy), r,     (0, 110, 255), 3)
                cv2.circle(canvas, (cx, cy), r // 4, (0, 200, 255), -1)

                if last_annotated is not None:
                    tx0 = max(self._sw - TH_W - 12, 10)
                    th  = cv2.resize(cv2.flip(last_annotated, 1), (TH_W, TH_H))
                    canvas[6:6 + TH_H + 4, tx0 - 2:tx0 + TH_W + 2] = (40, 40, 40)
                    canvas[8:8 + TH_H,     tx0:tx0 + TH_W]           = th

                cv2.putText(canvas, "[ R ] recalibrar   [ Q ] sair",
                            (self._sw - 440, max(keyboard.keyboard_top - 8, 120)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (70, 70, 70), 1)
                cv2.imshow(self.WIN, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or not self._alive():
                return False
            if key == ord("r") or self._start_event.is_set():
                self._start_event.clear()
                return True

    # ── Fluxo Principal ───────────────────────────────────────────────────────

    def run(self) -> None:
        """Setup físico → Setup câmera → Fase 0 → Fase 1 → Fase 2 → modelo → validação → teclado."""
        if not self.tracker.open_camera():
            return
        try:
            cv2.namedWindow(self.WIN, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(self.WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            if not self._setup_screen():
                return

            # ── Setup físico (uma vez por sessão) ─────────────────────────────
            saved_profile = GazeProfile.load()
            _base_profile = saved_profile if saved_profile is not None else GazeProfile()
            _base_profile = self._physical_setup_screen(_base_profile)

            while True:
                # ── Fase 0 ────────────────────────────────────────────────────
                profile = _run_phase0(self.tracker, self._sw, self._sh, self.WIN)

                # Preserva configurações de setup físico do perfil base
                profile.monitor_inches    = _base_profile.monitor_inches
                profile.camera_position   = _base_profile.camera_position
                profile.distance_cm       = _base_profile.distance_cm
                profile.vertical_offset   = _base_profile.vertical_offset
                profile.horizontal_offset = _base_profile.horizontal_offset
                profile.setup_configured  = _base_profile.setup_configured

                self.tracker.set_dominant_eye(profile.dominant)
                self.tracker.set_glasses_mode(profile.has_glasses)

                # ── Fase 1 ────────────────────────────────────────────────────
                feats1, pos1 = _run_phase1(
                    self.tracker, profile, self._sw, self._sh, self.WIN)

                if len(feats1) < 4:
                    logger.error("Fase 1 incompleta ({} pontos).", len(feats1))
                    return

                weights = [1.0] * len(feats1)

                # ── Fase 2 intro (pulada por padrão) ─────────────────────────
                canvas = np.zeros((self._sh, self._sw, 3), np.uint8)
                _text_center(canvas, "Fase 2 — Trajetórias (opcional)",
                             self._sh // 2 - 70, 1.2, (0, 200, 255), 2)
                _text_center(canvas, "Melhora a precisão nas bordas da tela.",
                             self._sh // 2,      0.80, (200, 200, 200), 1)
                _text_center(canvas, "[ T ] ativar trajetórias   [ SPACE ] pular",
                             self._sh // 2 + 70, 0.80, (100, 220, 100), 1)
                cv2.imshow(self.WIN, canvas)

                do_phase2 = False
                while True:
                    key = cv2.waitKey(50) & 0xFF
                    if key == ord("t"):
                        do_phase2 = True
                        break
                    if key == ord(" ") or key == ord("q") or not self._alive():
                        break

                feats2, pos2 = [], []
                if do_phase2:
                    feats2, pos2 = _run_phase2(
                        self.tracker, profile, self._sw, self._sh, self.WIN)
                    weights += [self.TRAJ_WEIGHT] * len(feats2)
                    logger.info("Fase 2: {} amostras.", len(feats2))

                # ── Ajuste do modelo com prior MPIIGaze + histórico ──────────
                from calib_store import load_historical, save_session

                # 1. Histórico de sessões anteriores do mesmo paciente
                hist_feats, hist_pos, hist_weights = load_historical(
                    max_sessions=5,
                    decay=0.60,
                    max_error_px=120.0,
                )
                logger.info("Histórico: {} pontos de sessões anteriores.", len(hist_feats))

                # 2. Prior MPIIGaze com filtragem por setup físico
                prior_feats:   list = []
                prior_pos:     list = []
                prior_weights: list = []

                _prior_path = os.path.join(os.path.dirname(__file__), "prior_gaze.npy")
                if os.path.exists(_prior_path):
                    try:
                        prior_data = np.load(_prior_path, allow_pickle=True).item()
                        p_feats    = prior_data["features"]
                        p_poses    = prior_data["positions"]
                        p_weight   = float(prior_data["weight"])

                        gx_range   = max(profile.gaze_x_max - profile.gaze_x_min, 1e-5)
                        gy_range   = max(profile.gaze_y_max - profile.gaze_y_min, 1e-5)
                        p_gx_range = max(float(prior_data["gx_max"]) - float(prior_data["gx_min"]), 1e-5)
                        p_gy_range = max(float(prior_data["gy_max"]) - float(prior_data["gy_min"]), 1e-5)

                        for feat, pos in zip(p_feats, p_poses):
                            pos_y = float(pos[1])
                            # Monitor grande + câmera em cima: usar só região central-inferior
                            if (profile.monitor_inches >= 23.0
                                    and profile.camera_position == "top"
                                    and pos_y <= 0.3):
                                continue

                            nx = (feat[0] - float(prior_data["gx_min"])) / p_gx_range
                            ny = (feat[1] - float(prior_data["gy_min"])) / p_gy_range

                            mapped = np.array([
                                nx * gx_range + profile.gaze_x_min,
                                ny * gy_range + profile.gaze_y_min,
                            ], dtype=np.float64)
                            norm = profile.normalize(mapped)

                            prior_feats.append(norm)
                            prior_pos.append((float(pos[0]), float(pos[1])))
                            prior_weights.append(p_weight)

                        logger.info("Prior MPIIGaze carregado: {} pontos (peso={}).",
                                    len(prior_feats), p_weight)
                    except Exception as e:
                        logger.warning("Falha ao carregar prior MPIIGaze: {}", e)
                else:
                    logger.info("prior_gaze.npy não encontrado — rodando sem prior. "
                                "Execute scripts/build_prior.py para gerar.")

                # 3. Aplica correção geométrica nas posições da sessão atual
                pos1_corr  = _apply_geometric_correction(pos1,  profile)
                pos2_corr  = _apply_geometric_correction(pos2,  profile)
                hist_pos_c = _apply_geometric_correction(hist_pos, profile)

                # 4. Combina: sessão corrigida + histórico corrigido + prior
                all_feats   = feats1 + feats2 + hist_feats   + prior_feats
                all_pos     = pos1_corr + pos2_corr + hist_pos_c + prior_pos
                all_weights = weights           + hist_weights + prior_weights

                logger.info(
                    "model.fit(): {} pts sessão + {} histórico + {} prior = {} total",
                    len(feats1) + len(feats2),
                    len(hist_feats),
                    len(prior_feats),
                    len(all_feats),
                )

                if not self.model.fit(all_feats, all_pos, all_weights):
                    logger.error("Falha no ajuste do modelo.")
                    return
                self.model.reset_smoothing()

                # Armazena dados da calibração para auto-recalibração silenciosa
                self._calib_all_feats   = list(all_feats)
                self._calib_all_pos     = list(all_pos)
                self._calib_all_weights = list(all_weights)

                # ── Validação ─────────────────────────────────────────────────
                avg_err, sys_dx, sys_dy = _run_validation(
                    self.tracker, profile, self.model,
                    self._sw, self._sh, self.WIN)

                # Aplica correção automática de offset sistemático se solicitado
                if abs(sys_dx) > 0.01 or abs(sys_dy) > 0.01:
                    profile.horizontal_offset -= sys_dx
                    profile.vertical_offset   -= sys_dy
                    _base_profile.horizontal_offset = profile.horizontal_offset
                    _base_profile.vertical_offset   = profile.vertical_offset
                    pos1_corr  = _apply_geometric_correction(pos1,  profile)
                    pos2_corr  = _apply_geometric_correction(pos2,  profile)
                    hist_pos_c = _apply_geometric_correction(hist_pos, profile)
                    all_pos    = pos1_corr + pos2_corr + hist_pos_c + prior_pos
                    self.model.fit(all_feats, all_pos, all_weights)
                    self.model.reset_smoothing()
                    self._calib_all_pos = list(all_pos)
                    logger.info("Offset corrigido: dx={:.3f} dy={:.3f}",
                                profile.horizontal_offset, profile.vertical_offset)

                if avg_err > 80:
                    canvas = np.zeros((self._sh, self._sw, 3), np.uint8)
                    _text_center(canvas, f"Erro médio: {avg_err:.0f} px  (>80px)",
                                 self._sh // 2 - 40, 1.0, (0, 60, 220), 2)
                    _text_center(canvas, "[ R ] recalibrar   [ SPACE ] continuar assim mesmo",
                                 self._sh // 2 + 30, 0.78, (100, 200, 100), 1)
                    cv2.imshow(self.WIN, canvas)
                    if cv2.waitKey(0) & 0xFF == ord("r"):
                        continue

                # ── Salvar sessão no histórico ────────────────────────────────
                try:
                    save_session(
                        profile  = profile,
                        feats1   = feats1,
                        pos1     = pos1,
                        feats2   = feats2,
                        pos2     = pos2,
                        error_px = avg_err,
                    )
                    logger.info("Sessão salva no histórico (erro={:.0f}px).", avg_err)
                except Exception as e:
                    logger.warning("Falha ao salvar sessão: {}", e)

                # ── Rastreamento ao vivo ──────────────────────────────────────
                if not self._run_tracking(profile):
                    break

        finally:
            self.tracker.stop()
