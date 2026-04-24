"""
IrisFlow — Rastreador de íris
Usa MediaPipe Face Mesh + OpenCV para detectar e rastrear o movimento da íris em tempo real.

Técnicas implementadas:
  - Filtro de Kalman 2D (modelo de velocidade constante + amortecimento)
  - EMA α=0.3 para suavização adaptativa dos ratios
  - Feature combinada: 0.7 × ratio no olho + 0.3 × ratio na face
  - Calibração de 9 pontos (grade 3×3) → modelo quadrático
  - Dead zone adaptativa: trava cursor quando olho está parado
"""

import time
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple
from loguru import logger

from calibration import LinearCalibrator


# ---------------------------------------------------------------------------
# Landmarks MediaPipe Face Mesh (478 pontos com refine_landmarks=True)
# ---------------------------------------------------------------------------

LEFT_IRIS  = [473, 474, 475, 476, 477]   # 473 = centro da íris esquerda
RIGHT_IRIS = [468, 469, 470, 471, 472]   # 468 = centro da íris direita

LEFT_EYE   = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE  = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# Cantos ósseos do olho — NÃO seguem rotação do globo ocular
LEFT_EYE_CORNER_OUTER  = 33
LEFT_EYE_CORNER_INNER  = 133
RIGHT_EYE_CORNER_OUTER = 362
RIGHT_EYE_CORNER_INNER = 263

# Referências faciais para a feature composta (span ~3× mais largo que um olho)
FACE_LEFT   = 33    # canto lateral esquerdo da face
FACE_RIGHT  = 362   # canto lateral direito da face
NOSE_TIP    = 4

# Referências verticais: pálpebras e altura da face
LEFT_EYE_TOP     = 159   # pálpebra superior esquerda (centro)
LEFT_EYE_BOTTOM  = 145   # pálpebra inferior esquerda (centro)
RIGHT_EYE_TOP    = 386   # pálpebra superior direita (centro)
RIGHT_EYE_BOTTOM = 374   # pálpebra inferior direita (centro)
FACE_TOP         = 10    # testa (centro) — landmark estável, acima das sobrancelhas
FACE_BOTTOM      = 152   # queixo (centro)


# ---------------------------------------------------------------------------
# Filtro de Kalman 2D (numpy-only)
# ---------------------------------------------------------------------------

class KalmanFilter2D:
    """
    Filtro de Kalman 2D com modelo de velocidade constante + amortecimento.

    Estado: [px, py, vx, vy]
    Observação: [px, py]

    velocity_damp=0.85 garante que após ~1 s de eye parado a velocidade
    residual é < 1% da original — cursor para sem trepidação.
    """

    def __init__(
        self,
        dt: float = 1 / 30,
        velocity_damp: float = 0.85,
        process_noise: float = 1.5,
        meas_noise: float = 25.0,
    ) -> None:
        d = velocity_damp
        self._F = np.array([
            [1, 0, dt,  0],
            [0, 1,  0, dt],
            [0, 0,  d,  0],
            [0, 0,  0,  d],
        ], dtype=float)
        self._H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        self._Q = np.eye(4) * process_noise
        self._R = np.eye(2) * meas_noise
        self._x = np.zeros((4, 1))
        self._P = np.eye(4) * 500.0
        self._initialized = False

    def initialize(self, px: float, py: float) -> None:
        self._x = np.array([[px], [py], [0.0], [0.0]])
        self._P = np.eye(4) * 500.0
        self._initialized = True

    def step(self, z_x: float, z_y: float) -> Tuple[float, float]:
        """Prediz + atualiza com nova observação. Retorna posição suavizada."""
        if not self._initialized:
            self.initialize(z_x, z_y)
            return z_x, z_y
        x_pred = self._F @ self._x
        P_pred = self._F @ self._P @ self._F.T + self._Q
        z = np.array([[z_x], [z_y]])
        S = self._H @ P_pred @ self._H.T + self._R
        K = P_pred @ self._H.T @ np.linalg.inv(S)
        self._x = x_pred + K @ (z - self._H @ x_pred)
        self._P = (np.eye(4) - K @ self._H) @ P_pred
        return float(self._x[0, 0]), float(self._x[1, 0])

    def predict_only(self) -> Tuple[float, float]:
        """Aplica apenas predição (sem nova medição) — decai suavemente."""
        if not self._initialized:
            return 0.0, 0.0
        self._x = self._F @ self._x
        self._P = self._F @ self._P @ self._F.T + self._Q
        return float(self._x[0, 0]), float(self._x[1, 0])

    def reset(self) -> None:
        self._initialized = False
        self._x = np.zeros((4, 1))
        self._P = np.eye(4) * 500.0


# ---------------------------------------------------------------------------
# Dead zone adaptativa
# ---------------------------------------------------------------------------

class AdaptiveDeadZone:
    """
    Dead zone adaptativa em espaço normalizado de tela (0–1).

    Opera APÓS a calibração para que os thresholds tenham significado
    uniforme nos dois eixos (1% de tela = 1% de tela).

    Válvula de escape: se o cursor ficar travado por mais de MAX_FROZEN
    frames enquanto a posição continua mudando, libera o lock para evitar
    que movimentos lentos sejam bloqueados indefinidamente.
    """

    BASE: float      = 0.035   # 3.5 % da tela — threshold de repouso
    MIN: float       = 0.010   # 1.0 % da tela — limite inferior (saccades rápidas)
    MAX: float       = 0.055   # 5.5 % da tela — limite superior (tremor extremo)
    VEL_WIN: int     = 6
    FAST_VEL: float  = 0.040   # velocidade alta em coords normalizadas
    MAX_FROZEN: int  = 25      # frames máximos travado antes de forçar unlock

    def __init__(self) -> None:
        self._history: deque = deque(maxlen=self.VEL_WIN)
        self._threshold: float = self.BASE
        self._locked: Optional[Tuple[float, float]] = None
        self._frozen: int = 0

    def check(self, pos: Tuple[float, float]) -> bool:
        """Retorna True se o movimento supera o threshold atual."""
        self._history.append(pos)

        if self._locked is None:
            self._locked = pos
            self._frozen = 0
            return True

        dr = ((pos[0] - self._locked[0]) ** 2
              + (pos[1] - self._locked[1]) ** 2) ** 0.5

        if len(self._history) >= 2:
            deltas = [
                ((self._history[i][0] - self._history[i - 1][0]) ** 2
                 + (self._history[i][1] - self._history[i - 1][1]) ** 2) ** 0.5
                for i in range(1, len(self._history))
            ]
            avg_vel = float(np.mean(deltas))
        else:
            avg_vel = 0.0

        if avg_vel > self.FAST_VEL:
            self._threshold = max(self._threshold * 0.80, self.MIN)
        else:
            self._threshold = min(self._threshold * 1.05, self.MAX)

        if dr >= self._threshold or self._frozen >= self.MAX_FROZEN:
            self._locked = pos
            self._frozen = 0
            return True

        self._frozen += 1
        return False

    def reset(self) -> None:
        self._history.clear()
        self._threshold = self.BASE
        self._locked = None
        self._frozen = 0


# ---------------------------------------------------------------------------
# IrisData + IrisTracker
# ---------------------------------------------------------------------------

@dataclass
class GazePoint:
    x: float
    y: float
    confidence: float


@dataclass
class IrisData:
    left_center:  Optional[Tuple[float, float]]
    right_center: Optional[Tuple[float, float]]
    left_ratio:   Optional[Tuple[float, float]]
    right_ratio:  Optional[Tuple[float, float]]


class IrisTracker:
    """
    Rastreia o movimento da íris em tempo real usando MediaPipe Face Mesh.

    Feature combinada por frame:
        ratio = 0.7 × eye_ratio + 0.3 × face_ratio

    eye_ratio: posição da íris relativa aos cantos ósseos do próprio olho.
    face_ratio: posição da íris relativa ao span lateral de toda a face
                (baseline ~3× mais largo → mais estável, menos ruído ocular).
    Ambos são suavizados por EMA (α=0.3) antes da combinação.
    """

    EMA_ALPHA: float = 0.20

    def __init__(
        self,
        camera_index: int = 0,
        target_fps: int = 30,
        show_preview: bool = True,
    ) -> None:
        self.camera_index = camera_index
        self.target_fps   = target_fps
        self.show_preview = show_preview

        self._cap: Optional[cv2.VideoCapture] = None
        self.has_glasses: bool = False
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._running = False
        self._ema_left:  Optional[Tuple[float, float]] = None
        self._ema_right: Optional[Tuple[float, float]] = None
        logger.info("IrisTracker inicializado (câmera {})", camera_index)

    def set_glasses_mode(self, has_glasses: bool) -> None:
        """Reajusta o FaceMesh para usuários com óculos (maior confiança mínima)."""
        if self.has_glasses == has_glasses:
            return
        self.has_glasses = has_glasses
        conf = 0.70 if has_glasses else 0.50
        self._face_mesh.close()
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=conf,
            min_tracking_confidence=conf,
        )
        logger.info("Modo óculos: {} (confiança={:.2f})", has_glasses, conf)

    def _open_camera(self) -> bool:
        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            logger.error("Não foi possível abrir a câmera {}", self.camera_index)
            return False
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        logger.info("Câmera {} aberta (640×480 @ {}fps)", self.camera_index, self.target_fps)
        return True

    def _ema(
        self,
        prev: Optional[Tuple[float, float]],
        new: Tuple[float, float],
    ) -> Tuple[float, float]:
        a = self.EMA_ALPHA
        if prev is None:
            return new
        return (a * new[0] + (1 - a) * prev[0],
                a * new[1] + (1 - a) * prev[1])

    def _extract_iris_data(self, landmarks, frame_w: int, frame_h: int) -> IrisData:
        """
        Calcula ratio combinado (eye + face) com EMA para cada olho.

        eye_ratio X: 0 = canto lateral, 1 = canto medial (dentro do olho)
        eye_ratio Y: 0.5 = linha média dos cantos; deslocamento normalizado
                     pela largura do olho (invariante à rotação do globo)

        face_ratio: íris relativa ao span entre os cantos externos dos dois
                    olhos — baseline ~3× mais largo isola melhor a íris do globo
        """
        def get_iris_center(indices: List[int]) -> Tuple[float, float]:
            pts = np.array(
                [(landmarks[i].x * frame_w, landmarks[i].y * frame_h) for i in indices]
            )
            return float(pts[:, 0].mean()), float(pts[:, 1].mean())

        def eye_ratio(
            center: Tuple[float, float],
            outer: int,
            inner: int,
            top_lm: int,
            bottom_lm: int,
        ) -> Optional[Tuple[float, float]]:
            xa    = landmarks[outer].x * frame_w
            xb    = landmarks[inner].x * frame_w
            eye_w = abs(xb - xa)
            if eye_w == 0:
                return None
            top_y  = landmarks[top_lm].y    * frame_h
            bot_y  = landmarks[bottom_lm].y * frame_h
            eye_h  = max(abs(bot_y - top_y), 1.0)
            eye_cy = (top_y + bot_y) / 2.0
            rx = (center[0] - min(xa, xb)) / eye_w
            ry = 0.5 + (center[1] - eye_cy) / eye_h
            return rx, ry

        def face_ratio(center: Tuple[float, float]) -> Optional[Tuple[float, float]]:
            fl_x       = landmarks[FACE_LEFT].x    * frame_w
            fr_x       = landmarks[FACE_RIGHT].x   * frame_w
            face_top_y = landmarks[FACE_TOP].y     * frame_h
            face_bot_y = landmarks[FACE_BOTTOM].y  * frame_h
            face_w = abs(fr_x - fl_x)
            face_h = max(abs(face_bot_y - face_top_y), 1.0)
            face_cy = (face_top_y + face_bot_y) / 2.0
            if face_w == 0:
                return None
            rx = (center[0] - min(fl_x, fr_x)) / face_w
            ry = 0.5 + (center[1] - face_cy) / face_h
            return rx, ry

        left_center  = get_iris_center(LEFT_IRIS)
        right_center = get_iris_center(RIGHT_IRIS)

        le = eye_ratio(left_center,  LEFT_EYE_CORNER_OUTER,  LEFT_EYE_CORNER_INNER,  LEFT_EYE_TOP,  LEFT_EYE_BOTTOM)
        re = eye_ratio(right_center, RIGHT_EYE_CORNER_OUTER, RIGHT_EYE_CORNER_INNER, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM)
        lf = face_ratio(left_center)
        rf = face_ratio(right_center)

        def combine_ema(
            er: Optional[Tuple[float, float]],
            fr: Optional[Tuple[float, float]],
            prev: Optional[Tuple[float, float]],
        ) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
            if er is None or fr is None:
                return None, prev
            combined = (0.7 * er[0] + 0.3 * fr[0], 0.7 * er[1] + 0.3 * fr[1])
            new_ema = self._ema(prev, combined)
            return new_ema, new_ema

        left_ratio,  self._ema_left  = combine_ema(le, lf, self._ema_left)
        right_ratio, self._ema_right = combine_ema(re, rf, self._ema_right)

        return IrisData(
            left_center=left_center,
            right_center=right_center,
            left_ratio=left_ratio,
            right_ratio=right_ratio,
        )

    def process_frame(self, frame: np.ndarray) -> Tuple[Optional[IrisData], np.ndarray]:
        frame_h, frame_w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None, frame

        landmarks = results.multi_face_landmarks[0].landmark
        iris_data = self._extract_iris_data(landmarks, frame_w, frame_h)

        if self.show_preview:
            frame = self._draw_landmarks(frame, landmarks, frame_w, frame_h, iris_data)

        return iris_data, frame

    def _draw_landmarks(self, frame, landmarks, frame_w, frame_h, iris_data: IrisData):
        for idx in LEFT_EYE + RIGHT_EYE:
            cv2.circle(frame,
                       (int(landmarks[idx].x * frame_w), int(landmarks[idx].y * frame_h)),
                       1, (0, 160, 0), -1)
        for idx in (LEFT_EYE_CORNER_OUTER, LEFT_EYE_CORNER_INNER,
                    RIGHT_EYE_CORNER_OUTER, RIGHT_EYE_CORNER_INNER):
            cv2.circle(frame,
                       (int(landmarks[idx].x * frame_w), int(landmarks[idx].y * frame_h)),
                       4, (255, 200, 0), -1)
        for idx in LEFT_IRIS + RIGHT_IRIS:
            cv2.circle(frame,
                       (int(landmarks[idx].x * frame_w), int(landmarks[idx].y * frame_h)),
                       2, (0, 140, 255), -1)
        if iris_data.left_center:
            cv2.circle(frame,
                       (int(iris_data.left_center[0]), int(iris_data.left_center[1])),
                       5, (0, 0, 255), -1)
        if iris_data.right_center:
            cv2.circle(frame,
                       (int(iris_data.right_center[0]), int(iris_data.right_center[1])),
                       5, (0, 0, 255), -1)
        return frame

    def start(self, on_iris_data=None) -> None:
        if not self._open_camera():
            return
        self._running = True
        logger.info("Iniciando rastreamento de íris. Pressione Q para sair.")
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                logger.warning("Frame não capturado.")
                break
            iris_data, annotated = self.process_frame(frame)
            if iris_data and on_iris_data:
                on_iris_data(iris_data)
            if self.show_preview:
                cv2.imshow("IrisFlow — Rastreamento", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("Encerrado pelo usuário.")
                    break
        self.stop()

    def stop(self) -> None:
        self._running = False
        if self._cap:
            self._cap.release()
        cv2.destroyAllWindows()
        self._face_mesh.close()
        logger.info("IrisTracker encerrado.")


# ---------------------------------------------------------------------------
# Detecção de óculos por heurística de reflexo
# ---------------------------------------------------------------------------

def _detect_glasses_heuristic(
    frame: np.ndarray,
    landmarks,
    frame_w: int,
    frame_h: int,
) -> bool:
    """
    Detecta óculos verificando pixels muito brilhantes (reflexo de lente)
    na região de cada olho.  Threshold empírico: se >8 % dos pixels do
    bounding‑box ocular estiverem acima de 210 em escala de cinza, provável
    reflexo de óculos.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bright_fracs: List[float] = []
    for outer, inner, top, bot in [
        (LEFT_EYE_CORNER_OUTER,  LEFT_EYE_CORNER_INNER,  LEFT_EYE_TOP,  LEFT_EYE_BOTTOM),
        (RIGHT_EYE_CORNER_OUTER, RIGHT_EYE_CORNER_INNER, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM),
    ]:
        x1 = int(min(landmarks[outer].x, landmarks[inner].x) * frame_w) - 6
        x2 = int(max(landmarks[outer].x, landmarks[inner].x) * frame_w) + 6
        y1 = int(min(landmarks[top].y,   landmarks[bot].y)   * frame_h) - 6
        y2 = int(max(landmarks[top].y,   landmarks[bot].y)   * frame_h) + 6
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame_w, x2), min(frame_h, y2)
        if x2 > x1 and y2 > y1:
            roi = gray[y1:y2, x1:x2]
            bright_fracs.append(float(np.mean(roi > 210)))
    return (max(bright_fracs) > 0.08) if bright_fracs else False


# ---------------------------------------------------------------------------
# CalibrationSession — 9 pontos + Kalman + dead zone adaptativa
# ---------------------------------------------------------------------------

class CalibrationSession:
    """
    Sessão interativa de calibração (9 pontos, grade 3×3) + rastreamento ao vivo.

    Pipeline de suavização no rastreamento:
      1. EMA α=0.3 no IrisTracker (ratio combinado eye+face por frame)
      2. Dead zone adaptativa no espaço do ratio (suprime micro-tremor)
      3. Filtro de Kalman 2D em coordenadas de pixel (suaviza trajetória)

    Teclas: SPACE = iniciar / pular contagem  |  R = recalibrar  |  Q = sair
    """

    CALIB_POSITIONS: List[Tuple[float, float]] = [
        (0.10, 0.15), (0.50, 0.15), (0.90, 0.15),
        (0.10, 0.50), (0.50, 0.50), (0.90, 0.50),
        (0.10, 0.85), (0.50, 0.85), (0.90, 0.85),
    ]
    CENTER_POSITION:       Tuple[float, float] = (0.50, 0.50)
    CENTER_WEIGHT:         float               = 3.0
    CENTER_COLLECT_FRAMES: int                 = 60
    COUNTDOWN_SECS: float = 2.5
    COLLECT_FRAMES: int   = 30
    LERP_FACTOR:    float = 0.10   # interpolação linear cursor→alvo (0.1 = lento/suave)
    TRAJ_WEIGHT:    float = 0.10   # peso por amostra de trajetória (mais ruidoso que ponto fixo)
    WIN_NAME: str         = "IrisFlow"

    def __init__(self, tracker: "IrisTracker") -> None:
        self.tracker    = tracker
        self.calibrator = LinearCalibrator()
        self._kalman    = KalmanFilter2D()
        self._dead_zone = AdaptiveDeadZone()
        self._sw, self._sh = self._get_screen_size()
        self._done_points: List[int] = []

    @staticmethod
    def _get_screen_size() -> Tuple[int, int]:
        try:
            import tkinter as tk
            root = tk.Tk()
            root.withdraw()
            w, h = root.winfo_screenwidth(), root.winfo_screenheight()
            root.destroy()
            return w, h
        except Exception:
            return 1280, 720

    def _canvas(self) -> np.ndarray:
        return np.zeros((self._sh, self._sw, 3), dtype=np.uint8)

    def _px(self, norm: Tuple[float, float]) -> Tuple[int, int]:
        return int(norm[0] * self._sw), int(norm[1] * self._sh)

    def _text_center(
        self, canvas, text, y, scale=0.9,
        color=(200, 200, 200), thickness=1,
    ) -> None:
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        cv2.putText(canvas, text, ((self._sw - tw) // 2, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

    def _draw_thumb(self, canvas, frame) -> None:
        th, tw = 162, 216
        thumb = cv2.resize(cv2.flip(frame, 1), (tw, th))
        canvas[8:12 + th, 8:12 + tw] = (40, 40, 40)
        canvas[10:10 + th, 10:10 + tw] = thumb

    def _avg_ratio(self, data: IrisData) -> Optional[Tuple[float, float]]:
        l, r = data.left_ratio, data.right_ratio
        if l and r:
            return (l[0] + r[0]) / 2, (l[1] + r[1]) / 2
        return l or r

    def _window_alive(self) -> bool:
        return cv2.getWindowProperty(self.WIN_NAME, cv2.WND_PROP_VISIBLE) >= 1

    def _draw_all_points(
        self,
        canvas: np.ndarray,
        active_idx: Optional[int] = None,
        pulse: float = 0.0,
    ) -> None:
        """Cinza = pendente | verde = feito | anel animado = atual."""
        for i, pos in enumerate(self.CALIB_POSITIONS):
            px, py = self._px(pos)
            if i in self._done_points:
                cv2.circle(canvas, (px, py), 10, (0, 220, 80), -1)
                cv2.circle(canvas, (px, py), 14, (0, 160, 60), 2)
            elif i == active_idx:
                r = int(14 + 6 * pulse)
                cv2.circle(canvas, (px, py), r,  (0, 180, 255), 3)
                cv2.circle(canvas, (px, py), 10, (255, 255, 255), -1)
                cv2.circle(canvas, (px, py),  5, (0, 100, 255),  -1)
            else:
                cv2.circle(canvas, (px, py), 10, (70, 70, 70), -1)
                cv2.circle(canvas, (px, py), 14, (50, 50, 50),  2)

    def _show_intro(self) -> bool:
        lines = [
            ("IrisFlow — Calibracao", 1.3, (0, 200, 255), 2),
            ("", 0.8, (200, 200, 200), 1),
            ("Voce vera 9 pontos na tela (grade 3x3).", 0.8, (200, 200, 200), 1),
            ("Olhe fixamente para cada ponto e aguarde.", 0.8, (200, 200, 200), 1),
            ("O sistema captura automaticamente.", 0.8, (200, 200, 200), 1),
            ("", 0.8, (200, 200, 200), 1),
            ("[ SPACE ] iniciar   |   [ Q ] sair", 0.85, (100, 220, 100), 1),
        ]
        base_y = self._sh // 2 - len(lines) * 24
        while True:
            canvas = self._canvas()
            self._draw_all_points(canvas)
            for i, (text, scale, color, thick) in enumerate(lines):
                self._text_center(canvas, text, base_y + i * 52, scale, color, thick)
            cv2.imshow(self.WIN_NAME, canvas)
            key = cv2.waitKey(50) & 0xFF
            if key == ord(" "):
                return True
            if key == ord("q") or not self._window_alive():
                return False

    def _show_setup_check(self) -> bool:
        """
        Tela de verificação de setup antes da calibração.
        Verifica: rosto detectado, centralização, altura dos olhos e iluminação.
        Quando tudo OK, exibe botão [ ENTER ] para o usuário prosseguir.
        [G] alterna modo óculos manualmente.
        Retorna False se o usuário sair (Q).
        """
        glasses_on: bool = False

        prev_w = min(int(self._sw * 0.54), 800)
        prev_h = int(prev_w * 9 / 16)
        prev_x = (self._sw - prev_w) // 2
        prev_y = 85
        oval_cx = self._sw // 2
        oval_cy = prev_y + prev_h // 2

        while True:
            ret, frame = self.tracker._cap.read()
            if not ret:
                return False

            frame_h, frame_w = frame.shape[:2]
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.tracker._face_mesh.process(rgb)

            # Iluminação — independe de face detectada
            gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_br  = float(np.mean(gray))
            light_ok = 65.0 <= mean_br <= 195.0
            light_hint = (
                "" if light_ok
                else (" — ambiente muito escuro" if mean_br < 65 else " — luz excessiva")
            )

            face_ok = center_ok = height_ok = False
            face_hint = center_hint = height_hint = ""

            if results.multi_face_landmarks:
                face_ok = True
                lm = results.multi_face_landmarks[0].landmark

                nose_x = lm[NOSE_TIP].x
                nose_y = lm[NOSE_TIP].y
                center_ok = abs(nose_x - 0.5) < 0.22 and abs(nose_y - 0.5) < 0.22
                if not center_ok:
                    center_hint = " — use o oval como guia"

                avg_eye_y = (lm[LEFT_IRIS[0]].y + lm[RIGHT_IRIS[0]].y) / 2
                height_ok = 0.38 <= avg_eye_y <= 0.62
                if not height_ok:
                    height_hint = (
                        " — camera muito baixa" if avg_eye_y < 0.38
                        else " — camera muito alta"
                    )
            else:
                face_hint = " — posicione o rosto na camera"

            all_ok = face_ok and center_ok and height_ok and light_ok

            # --- desenho ---
            canvas = self._canvas()

            # Preview da câmera espelhado
            thumb = cv2.resize(cv2.flip(frame, 1), (prev_w, prev_h))
            canvas[prev_y:prev_y + prev_h, prev_x:prev_x + prev_w] = thumb

            # Oval de posicionamento do rosto
            oval_color = (0, 220, 80) if all_ok else (120, 120, 120)
            cv2.ellipse(canvas, (oval_cx, oval_cy),
                        (int(prev_w * 0.21), int(prev_h * 0.42)),
                        0, 0, 360, oval_color, 3)

            # Indicadores de status
            checks = [
                ("Rosto detectado"              + face_hint,   face_ok),
                ("Rosto centralizado"           + center_hint,  center_ok),
                ("Olhos entre 40%-60% do frame" + height_hint, height_ok),
                ("Iluminacao adequada"          + light_hint,   light_ok),
            ]

            base_y = prev_y + prev_h + 52
            dot_x  = self._sw // 2 - 255
            txt_x  = dot_x + 26

            for i, (label, ok) in enumerate(checks):
                row_y     = base_y + i * 44
                dot_color = (0, 220, 80) if ok else (60, 80, 220)
                txt_color = (0, 220, 80) if ok else (140, 140, 220)
                cv2.circle(canvas, (dot_x, row_y - 6), 9, dot_color, -1)
                cv2.putText(canvas, label, (txt_x, row_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.70, txt_color, 2)

            # Óculos — toggle manual [G]
            glasses_row_y = base_y + len(checks) * 44
            if glasses_on:
                g_txt = "[ G ] Uso oculos: SIM — ajustes aplicados"
                g_dot = (0, 180, 255)
                g_txt_color = (0, 180, 255)
            else:
                g_txt = "[ G ] Uso oculos: NAO"
                g_dot = (70, 70, 70)
                g_txt_color = (100, 100, 100)
            cv2.circle(canvas, (dot_x, glasses_row_y - 6), 9, g_dot, -1)
            cv2.putText(canvas, g_txt, (txt_x, glasses_row_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, g_txt_color, 1)

            # Área do botão — abaixo da linha de óculos
            btn_y = base_y + (len(checks) + 1) * 44 + 28
            btn_len = 480
            btn_x0  = self._sw // 2 - btn_len // 2

            if all_ok:
                # Botão verde de prosseguir
                cv2.rectangle(canvas,
                              (btn_x0, btn_y - 4),
                              (btn_x0 + btn_len, btn_y + 46),
                              (0, 55, 20), -1)
                cv2.rectangle(canvas,
                              (btn_x0, btn_y - 4),
                              (btn_x0 + btn_len, btn_y + 46),
                              (0, 200, 70), 2)
                self._text_center(canvas, "[ ENTER ]  Prosseguir",
                                  btn_y + 30, 1.0, (0, 255, 100), 2)
            else:
                self._text_center(canvas, "Ajuste o setup acima",
                                  btn_y + 24, 0.85, (120, 120, 120), 1)

            self._text_center(canvas, "IrisFlow — Verificacao de Setup",
                              44, 1.1, (0, 200, 255), 2)
            self._text_center(
                canvas,
                "[ G ] oculos   |   [ SPACE ] pular   |   [ Q ] sair",
                self._sh - 18, 0.55, (65, 65, 65), 1,
            )

            cv2.imshow(self.WIN_NAME, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or not self._window_alive():
                return False
            if key == 13 and all_ok:   # Enter — só avança se tudo OK
                return True
            if key == ord(" "):        # Space — pula a verificação
                return True
            if key == ord("g"):
                glasses_on = not glasses_on
                self.tracker.set_glasses_mode(glasses_on)

    def _show_success(self) -> None:
        canvas = self._canvas()
        self._draw_all_points(canvas)
        self._text_center(canvas, "Calibracao concluida!", self._sh // 2 - 28,
                          1.3, (0, 255, 120), 2)
        self._text_center(canvas, "Iniciando rastreamento...", self._sh // 2 + 28,
                          0.9, (200, 200, 200), 1)
        cv2.imshow(self.WIN_NAME, canvas)
        cv2.waitKey(1400)

    def _calibrate_point(
        self,
        pos_norm: Tuple[float, float],
        idx: int,
        n_collect: Optional[int] = None,
        label_override: Optional[str] = None,
    ) -> bool:
        fps = max(self.tracker.target_fps, 15)
        countdown_total = int(self.COUNTDOWN_SECS * fps)
        n_collect = n_collect or self.COLLECT_FRAMES
        px, py = self._px(pos_norm)
        label = label_override or f"Olhe para o ponto  {idx} / {len(self.CALIB_POSITIONS)}"
        active_i = idx - 1

        # Fase 1: contagem regressiva
        for i in range(countdown_total):
            ret, frame = self.tracker._cap.read()
            if not ret:
                return False
            _, annotated = self.tracker.process_frame(frame)
            progress = i / countdown_total
            pulse = np.sin(progress * 6 * np.pi) * 0.5 + 0.5

            canvas = self._canvas()
            self._draw_all_points(canvas, active_idx=active_i, pulse=pulse)
            self._text_center(canvas, label, 50, 1.0, (180, 180, 180), 2)
            cv2.ellipse(canvas, (px, py), (44, 44), -90, 0,
                        int(360 * progress), (0, 180, 255), 5)
            self._draw_thumb(canvas, annotated)
            cv2.imshow(self.WIN_NAME, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or not self._window_alive():
                return False
            if key == ord(" "):
                break

        # Fase 2: coleta de amostras (mediana para robustez a outliers)
        collected: List[Tuple[float, float]] = []
        for _ in range(n_collect):
            ret, frame = self.tracker._cap.read()
            if not ret:
                break
            iris_data, annotated = self.tracker.process_frame(frame)
            if iris_data:
                ratio = self._avg_ratio(iris_data)
                if ratio:
                    collected.append(ratio)

            canvas = self._canvas()
            self._draw_all_points(canvas, active_idx=active_i, pulse=1.0)
            self._text_center(canvas, "Capturando...", 50, 1.0, (0, 255, 100), 2)
            self._draw_thumb(canvas, annotated)
            cv2.imshow(self.WIN_NAME, canvas)
            cv2.waitKey(1)

        if not collected:
            logger.warning("Nenhuma amostra capturada no ponto {}.", idx)
            return False

        median_x = float(np.median([r[0] for r in collected]))
        median_y = float(np.median([r[1] for r in collected]))
        self.calibrator.add_calibration_point((median_x, median_y), pos_norm)
        self._done_points.append(active_i)
        logger.info("Ponto {} capturado: íris=({:.3f}, {:.3f})", idx, median_x, median_y)
        return True

    def _run_gaze_tracking(self) -> bool:
        """
        Loop de rastreamento ao vivo.
        Retorna True = recalibrar (R pressionado), False = sair (Q pressionado).

        Pipeline:
          ratio EMA → dead zone → Kalman step (se moveu) ou predict_only (parado)
        """
        self._kalman.reset()
        self._dead_zone.reset()
        cx, cy = float(self._sw // 2), float(self._sh // 2)
        hold_x, hold_y = cx, cy
        logger.info("Rastreamento ativo.  R = recalibrar  |  Q = sair")

        while True:
            ret, frame = self.tracker._cap.read()
            if not ret:
                return False

            iris_data, annotated = self.tracker.process_frame(frame)
            canvas = self._canvas()

            if iris_data:
                ratio = self._avg_ratio(iris_data)
                if ratio and not (0.10 <= ratio[0] <= 0.90):
                    ratio = None
                if ratio:
                    pred = self.calibrator.predict(ratio)
                    if pred:
                        sz_x = pred[0] * self._sw
                        sz_y = pred[1] * self._sh
                        if self._dead_zone.check(pred):
                            target_x, target_y = self._kalman.step(sz_x, sz_y)
                            cx = cx + self.LERP_FACTOR * (target_x - cx)
                            cy = cy + self.LERP_FACTOR * (target_y - cy)
                            hold_x, hold_y = cx, cy
                        else:
                            self._kalman.predict_only()
                            cx, cy = hold_x, hold_y

            draw_x, draw_y = int(cx), int(cy)
            cv2.circle(canvas, (draw_x, draw_y), 28, (0, 110, 255), 3)
            cv2.circle(canvas, (draw_x, draw_y),  7, (0, 200, 255), -1)

            self._draw_thumb(canvas, annotated)
            cv2.putText(
                canvas, "[ R ] recalibrar   [ Q ] sair",
                (self._sw - 360, self._sh - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (70, 70, 70), 1,
            )
            cv2.imshow(self.WIN_NAME, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or not self._window_alive():
                return False
            if key == ord("r"):
                return True

    def _show_trajectory_intro(self) -> bool:
        """
        Tela de introdução para a Fase 2 (calibração por trajetória).
        Retorna True = iniciar, False = pular fase (Q).
        """
        lines = [
            ("IrisFlow — Fase 2: Calibracao por Trajetoria", 1.1, (0, 200, 255), 2),
            ("", 0.7, (200, 200, 200), 1),
            ("Fase 1 concluida!  Agora um alvo se movera lentamente.", 0.75, (200, 200, 200), 1),
            ("Siga o alvo com os olhos da melhor forma possivel.", 0.75, (200, 200, 200), 1),
            ("Isso melhora a precisao nos cantos e bordas da tela.", 0.75, (180, 180, 180), 1),
            ("", 0.7, (200, 200, 200), 1),
            ("[ SPACE ] iniciar fase 2   |   [ Q ] pular esta fase", 0.80, (100, 220, 100), 1),
        ]
        base_y = self._sh // 2 - len(lines) * 26
        while True:
            canvas = self._canvas()
            for i, (text, scale, color, thick) in enumerate(lines):
                self._text_center(canvas, text, base_y + i * 52, scale, color, thick)
            cv2.imshow(self.WIN_NAME, canvas)
            key = cv2.waitKey(50) & 0xFF
            if key == ord(" "):
                return True
            if key == ord("q") or not self._window_alive():
                return False

    def _run_trajectory_calibration(
        self,
    ) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """
        Fase 2: 6 padrões de trajetória cobrindo toda a tela.
          2a Cruz horizontal | 2b Cruz vertical | 2c Diagonal Z
          2d Figura-8        | 2e Borda da tela | 2f Espiral

        Compensa lag oculomotor (~130 ms) pareando a íris de LAG_FRAMES
        frames atrás com o alvo atual.
        [ SPACE ] avança de fase  |  [ Q ] aborta e retorna o já coletado.
        """
        CROSS_FRAMES  = 120   # ~4 s a 30 fps
        Z_FRAMES      = 120   # ~4 s
        INF_FRAMES    = 180   # ~6 s
        BORDER_FRAMES = 180   # ~6 s
        SPIRAL_FRAMES = 300   # ~10 s
        SAMPLE_EVERY  = 2     # 1 amostra a cada 2 frames
        LAG_FRAMES    = 4     # ~130 ms de compensação de lag

        # Cruz — horizontal
        h_cross: List[Tuple[float, float]] = [
            (0.05 + 0.90 * i / (CROSS_FRAMES - 1), 0.50) for i in range(CROSS_FRAMES)
        ]
        # Cruz — vertical
        v_cross: List[Tuple[float, float]] = [
            (0.50, 0.05 + 0.90 * i / (CROSS_FRAMES - 1)) for i in range(CROSS_FRAMES)
        ]
        # Diagonal em Z: topo L→R, diagonal top-right→bottom-left, base L→R
        seg = Z_FRAMES // 3
        z_traj: List[Tuple[float, float]] = []
        for i in range(seg):
            t = i / (seg - 1)
            z_traj.append((0.05 + 0.90 * t, 0.12))
        for i in range(seg):
            t = i / (seg - 1)
            z_traj.append((0.95 - 0.90 * t, 0.12 + 0.76 * t))
        for i in range(seg):
            t = i / (seg - 1)
            z_traj.append((0.05 + 0.90 * t, 0.88))
        # Figura-8 (lemniscata) — começa do centro, 1 volta completa
        infinity: List[Tuple[float, float]] = []
        for i in range(INF_FRAMES):
            t = np.pi / 2 + i / INF_FRAMES * 2 * np.pi
            x = max(0.05, min(0.95, 0.5 + 0.40 * np.cos(t)))
            y = max(0.08, min(0.92, 0.5 + 0.30 * np.sin(2 * t)))
            infinity.append((x, y))
        # Borda da tela — sentido horário
        n4 = BORDER_FRAMES // 4
        border: List[Tuple[float, float]] = (
            [(0.05 + 0.90 * i / (n4 - 1), 0.10) for i in range(n4)]
            + [(0.95, 0.10 + 0.80 * i / (n4 - 1)) for i in range(n4)]
            + [(0.95 - 0.90 * i / (n4 - 1), 0.90) for i in range(n4)]
            + [(0.05, 0.90 - 0.80 * i / (n4 - 1)) for i in range(n4)]
        )
        # Espiral circular do centro → bordas, 2 voltas
        max_r = min(self._sw, self._sh) * 0.44
        spiral: List[Tuple[float, float]] = []
        for i in range(SPIRAL_FRAMES):
            t = i / SPIRAL_FRAMES
            r = t * max_r
            theta = t * 4 * np.pi
            x = max(0.05, min(0.95, 0.5 + r * np.cos(theta) / self._sw))
            y = max(0.05, min(0.95, 0.5 + r * np.sin(theta) / self._sh))
            spiral.append((x, y))

        phases = [
            ("Fase 2a — Cruz horizontal: siga o alvo",  h_cross),
            ("Fase 2b — Cruz vertical: siga o alvo",    v_cross),
            ("Fase 2c — Diagonal em Z: siga o alvo",    z_traj),
            ("Fase 2d — Figura-8: siga o alvo",         infinity),
            ("Fase 2e — Borda da tela: siga o alvo",    border),
            ("Fase 2f — Espiral: siga o alvo",          spiral),
        ]

        all_samples: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []

        for phase_label, traj in phases:
            ratio_buf: deque = deque(maxlen=LAG_FRAMES + 1)

            for frame_idx, target_pos in enumerate(traj):
                ret, frame = self.tracker._cap.read()
                if not ret:
                    return all_samples

                iris_data, annotated = self.tracker.process_frame(frame)

                cur_ratio: Optional[Tuple[float, float]] = None
                if iris_data:
                    cur_ratio = self._avg_ratio(iris_data)
                ratio_buf.append(cur_ratio)

                if (frame_idx % SAMPLE_EVERY == 0
                        and len(ratio_buf) == LAG_FRAMES + 1):
                    lagged = ratio_buf[0]
                    if lagged is not None and 0.10 <= lagged[0] <= 0.90:
                        all_samples.append((lagged, target_pos))

                canvas = self._canvas()

                for j in range(frame_idx + 1, min(frame_idx + 45, len(traj))):
                    fx, fy = self._px(traj[j])
                    fade = 1.0 - (j - frame_idx) / 45
                    c = int(55 * fade)
                    cv2.circle(canvas, (fx, fy), 3, (c, c * 2, c * 3), -1)

                tx, ty = self._px(target_pos)
                pulse  = 0.5 + 0.5 * np.sin(time.time() * 7)
                cv2.circle(canvas, (tx, ty), int(22 + 7 * pulse), (0, 150, 255), 3)
                cv2.circle(canvas, (tx, ty), 13, (255, 255, 255), -1)
                cv2.circle(canvas, (tx, ty),  5, (0, 100, 255),  -1)

                bar_w  = 480
                bar_x0 = self._sw // 2 - bar_w // 2
                bar_y  = self._sh - 68
                prog   = frame_idx / len(traj)
                cv2.rectangle(canvas, (bar_x0, bar_y),
                              (bar_x0 + bar_w, bar_y + 12), (45, 45, 45), -1)
                cv2.rectangle(canvas, (bar_x0, bar_y),
                              (bar_x0 + int(bar_w * prog), bar_y + 12), (0, 150, 255), -1)
                cv2.rectangle(canvas, (bar_x0, bar_y),
                              (bar_x0 + bar_w, bar_y + 12), (80, 80, 80), 2)

                self._text_center(canvas, phase_label, 44, 0.85, (180, 180, 180), 1)
                cv2.putText(
                    canvas,
                    f"Amostras: {len(all_samples)}",
                    (20, self._sh - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (70, 70, 70), 1,
                )
                self._text_center(
                    canvas,
                    "[ SPACE ] proxima fase   |   [ Q ] pular trajetoria",
                    self._sh - 18, 0.55, (65, 65, 65), 1,
                )

                self._draw_thumb(canvas, annotated)
                cv2.imshow(self.WIN_NAME, canvas)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or not self._window_alive():
                    return all_samples
                if key == ord(" "):
                    break

        return all_samples

    def run(self) -> None:
        """Fluxo completo: setup check → intro → calibração 9 pontos → rastreamento ao vivo."""
        if not self.tracker._open_camera():
            return
        try:
            cv2.namedWindow(self.WIN_NAME, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(
                self.WIN_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
            )
            if not self._show_setup_check():
                return
            if not self._show_intro():
                return

            while True:
                self.calibrator.model.clear()
                self._done_points.clear()
                self.tracker._ema_left  = None
                self.tracker._ema_right = None

                # 9 pontos da grade — peso 1.0 cada
                weights: List[float] = []
                for i, pos in enumerate(self.CALIB_POSITIONS):
                    if not self._calibrate_point(pos, i + 1):
                        return
                    weights.append(1.0)
                    if i < len(self.CALIB_POSITIONS) - 1:
                        canvas = self._canvas()
                        self._draw_all_points(canvas)
                        self._text_center(
                            canvas, "Proximo ponto...",
                            self._sh // 2, 1.0, (0, 200, 100), 2,
                        )
                        cv2.imshow(self.WIN_NAME, canvas)
                        cv2.waitKey(700)

                # 10º ponto: centro dedicado com peso CENTER_WEIGHT e 2× amostras
                canvas = self._canvas()
                self._draw_all_points(canvas)
                self._text_center(
                    canvas, "Ponto central — olhe para o centro da tela",
                    self._sh // 2, 1.0, (0, 200, 255), 2,
                )
                cv2.imshow(self.WIN_NAME, canvas)
                cv2.waitKey(800)
                if not self._calibrate_point(
                    self.CENTER_POSITION, 10,
                    n_collect=self.CENTER_COLLECT_FRAMES,
                    label_override="Ponto central (10/10) — olhe para o centro",
                ):
                    return
                weights.append(self.CENTER_WEIGHT)

                # Fase 2 — calibração por trajetória (opcional)
                if self._show_trajectory_intro():
                    traj_samples = self._run_trajectory_calibration()
                    for iris_ratio, screen_pos in traj_samples:
                        self.calibrator.add_calibration_point(iris_ratio, screen_pos)
                        weights.append(self.TRAJ_WEIGHT)
                    logger.info(
                        "Trajetória: {} amostras adicionadas (peso={}).",
                        len(traj_samples), self.TRAJ_WEIGHT,
                    )

                if not self.calibrator.calibrate(weights=weights):
                    logger.error("Falha ao calcular a calibração.")
                    return

                self._show_success()

                if not self._run_gaze_tracking():
                    break  # Q → encerra

        finally:
            self.tracker.stop()


# --- Execução direta ---
if __name__ == "__main__":
    tracker = IrisTracker(show_preview=True)
    CalibrationSession(tracker).run()
