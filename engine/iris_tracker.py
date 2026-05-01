"""
IrisFlow — Camadas 1-3: Captura · Head Pose · Vetor de Olhar

Camada 1  Captura threaded com queue.Queue — sem perda de frames
Camada 2  Head pose estimation 6-DoF via solvePnP (principal diferencial do Beam)
Camada 3  Vetor de olhar 3D compensado pela rotação da cabeça
"""

import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from loguru import logger

# ── Índices de landmarks (MediaPipe Face Mesh, refine_landmarks=True) ─────────

# 6 pontos para estimativa de head pose (nariz, queixo, cantos olhos, cantos boca)
_HP_IDX = [1, 152, 33, 263, 61, 291]

# Modelo 3D genérico de face em mm (+x = direita do viewer, +y = cima, -z = profundidade)
_FACE_3D = np.array([
    [ 0.0,    0.0,    0.0  ],   # 1   ponta do nariz
    [ 0.0,  -63.6,  -12.5  ],   # 152 queixo
    [-43.3,  32.7,  -26.0  ],   # 33  canto ext. olho esq (viewer)
    [ 43.3,  32.7,  -26.0  ],   # 263 canto ext. olho dir (viewer)
    [-28.9, -28.9,  -24.1  ],   # 61  canto esq. boca
    [ 28.9, -28.9,  -24.1  ],   # 291 canto dir. boca
], dtype=np.float64)

# Íris — 4 pontos de borda (perspectiva do usuário)
_IRIS_L = [474, 475, 476, 477]   # olho esquerdo do usuário
_IRIS_R = [469, 470, 471, 472]   # olho direito do usuário

# 4 landmarks de canto do socket ocular (âncoras ósseas estáveis, não pálpebra)
# Usar só 4 pontos evita que movimento de pálpebra cancele o sinal de gaze vertical
_EYE_L = [362, 263, 386, 374]   # outer, inner, top, bottom (olho esq. usuário)
_EYE_R = [ 33, 133, 159, 145]   # outer, inner, top, bottom (olho dir. usuário)


# ── IrisFrame ─────────────────────────────────────────────────────────────────

@dataclass
class IrisFrame:
    """Dados de um frame processado: vetor de olhar compensado + metadados."""
    gaze_feature:  np.ndarray                  # [gaze_x, gaze_y] em espaço da cabeça
    left_gaze:     np.ndarray                  # vetor 3D olho esq. (pós-compensação)
    right_gaze:    np.ndarray                  # vetor 3D olho dir. (pós-compensação)
    head_euler:    Tuple[float, float, float]  # (yaw°, pitch°, roll°)
    left_iris_px:  Tuple[float, float]         # centro íris esq. em pixels
    right_iris_px: Tuple[float, float]         # centro íris dir. em pixels
    head_ok:       bool                        # False = solvePnP falhou
    timestamp:     float


# ── Camada 1: Captura threaded com queue ──────────────────────────────────────

class _FrameQueue(threading.Thread):
    """Thread de captura: produz frames na queue; descarta o mais antigo se cheia."""

    def __init__(self, cap: cv2.VideoCapture, maxsize: int = 4) -> None:
        super().__init__(daemon=True, name="cam-capture")
        self._cap     = cap
        self._q: queue.Queue = queue.Queue(maxsize=maxsize)
        self._running = False

    def run(self) -> None:
        self._running = True
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.002)
                continue
            if self._q.full():
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    pass
            try:
                self._q.put_nowait(frame)
            except queue.Full:
                pass

    def get(self, timeout: float = 0.05) -> Optional[np.ndarray]:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        self._running = False


# ── Camada 2: Head Pose Estimation (6-DoF) ───────────────────────────────────

class _HeadPoseEstimator:
    """
    Estima pose da cabeça em 6 graus de liberdade via solvePnP.

    solvePnP retorna (rvec, tvec):
      - rvec: vetor de rotação (Rodrigues) → converte para Euler (yaw, pitch, roll)
      - tvec: translação XYZ em mm (escala do modelo 3D)
    """

    def __init__(self, frame_w: int = 640, frame_h: int = 480) -> None:
        f = float(frame_w)
        self._cam = np.array(
            [[f, 0, frame_w / 2],
             [0, f, frame_h / 2],
             [0, 0, 1           ]], dtype=np.float64)
        self._dist = np.zeros((4, 1), dtype=np.float64)

    def estimate(
        self, lm, fw: int, fh: int
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Retorna (rvec, tvec) ou None se o solvePnP falhar."""
        pts2d = np.array(
            [[lm[i].x * fw, lm[i].y * fh] for i in _HP_IDX],
            dtype=np.float64,
        )
        try:
            ok, rvec, tvec = cv2.solvePnP(
                _FACE_3D, pts2d, self._cam, self._dist,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            return (rvec, tvec) if ok else None
        except cv2.error:
            return None

    @staticmethod
    def euler(rvec: np.ndarray) -> Tuple[float, float, float]:
        """Converte rvec (Rodrigues) → (yaw°, pitch°, roll°). Convenção ZYX."""
        R, _ = cv2.Rodrigues(rvec)
        sy = float(np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
        if sy > 1e-6:
            pitch = float(np.degrees(np.arctan2(-R[2, 0], sy)))
            yaw   = float(np.degrees(np.arctan2( R[1, 0], R[0, 0])))
            roll  = float(np.degrees(np.arctan2( R[2, 1], R[2, 2])))
        else:
            pitch = float(np.degrees(np.arctan2(-R[2, 0], sy)))
            yaw   = 0.0
            roll  = float(np.degrees(np.arctan2(-R[1, 2], R[1, 1])))
        return yaw, pitch, roll


# ── Camada 3: Vetor de Olhar 3D com Compensação de Head Pose ─────────────────

class _GazeExtractor:
    """
    Calcula vetores de olhar 3D com compensação parcial de head pose.

    Algoritmo:
      1. iris_3d = média dos 4 landmarks de borda da íris [x, y, z]
      2. eye_3d  = média dos 4 landmarks de canto do socket (âncoras estáveis)
      3. gaze_cam = normalizar(iris_3d - eye_3d)
      4. gaze_head = R.T @ gaze_cam  (compensação total de rotação da cabeça)
      5. gaze_blend = COMP * gaze_head + (1-COMP) * gaze_cam
         → blend parcial: preserva sinal mesmo quando usuário rastreia com a cabeça
      6. combined_2d = média ponderada dos dois olhos (dominante 1.5×)
    """

    COMP = 0.6   # força da compensação de head pose [0=sem, 1=total]

    def __init__(self, dominant: str = "left") -> None:
        self.dominant = dominant    # "left" | "right"

    def extract(
        self,
        lm,
        rvec: Optional[np.ndarray],
        fw: int, fh: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Retorna (left_gaze_3d, right_gaze_3d, combined_2d).
        combined_2d = [gaze_x, gaze_y]; blend parcial preserva sinal de cabeça.
        """
        R_inv = None
        if rvec is not None:
            R, _ = cv2.Rodrigues(rvec)
            R_inv = R.T     # câmera → espaço da cabeça

        def _vec(iris_idx, eye_idx) -> np.ndarray:
            iris = np.mean([[lm[i].x, lm[i].y, lm[i].z] for i in iris_idx], axis=0)
            eye  = np.mean([[lm[i].x, lm[i].y, lm[i].z] for i in eye_idx],  axis=0)
            d    = iris - eye
            n    = float(np.linalg.norm(d))
            v_cam = (d / n if n > 1e-8 else np.array([0.0, 0.0, 1.0])).astype(np.float64)
            if R_inv is None:
                return v_cam
            v_head = (R_inv @ v_cam).astype(np.float64)
            # Blend: COMP × compensado + (1-COMP) × câmera
            return (self.COMP * v_head + (1 - self.COMP) * v_cam).astype(np.float64)

        lg = _vec(_IRIS_L, _EYE_L)
        rg = _vec(_IRIS_R, _EYE_R)

        wl, wr   = (1.5, 1.0) if self.dominant == "left" else (1.0, 1.5)
        combined = ((wl * lg[:2] + wr * rg[:2]) / (wl + wr)).astype(np.float64)
        return lg, rg, combined


# ── IrisTracker ───────────────────────────────────────────────────────────────

class IrisTracker:
    """
    Orquestra as Camadas 1–3.

    Uso típico:
        tracker = IrisTracker()
        tracker.open_camera()
        frame = tracker.read_frame()
        iris_frame, annotated = tracker.process_frame(frame)
        tracker.stop()
    """

    _CONF_NORMAL  = 0.50
    _CONF_GLASSES = 0.70

    def __init__(self, camera_index: int = 0, fps: int = 30) -> None:
        self.camera_index = camera_index
        self.fps          = fps
        self._cap:        Optional[cv2.VideoCapture] = None
        self._cam_thread: Optional[_FrameQueue]       = None
        self._head_pose   = _HeadPoseEstimator(640, 480)
        self._gaze        = _GazeExtractor("left")
        self._face_mesh   = self._build_mesh(self._CONF_NORMAL)
        logger.info("IrisTracker inicializado (câmera {})", camera_index)

    # ── Controles ─────────────────────────────────────────────────────────────

    def set_dominant_eye(self, eye: str) -> None:
        self._gaze.dominant = eye

    def set_glasses_mode(self, enabled: bool) -> None:
        conf = self._CONF_GLASSES if enabled else self._CONF_NORMAL
        self._face_mesh.close()
        self._face_mesh = self._build_mesh(conf)
        logger.info("Modo óculos: {} (conf mín {:.2f})", enabled, conf)

    # ── Câmera ────────────────────────────────────────────────────────────────

    def open_camera(self) -> bool:
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            logger.error("Câmera {} não disponível.", self.camera_index)
            return False
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS,          self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        self._cap        = cap
        self._cam_thread = _FrameQueue(cap, maxsize=4)
        self._cam_thread.start()
        logger.info("Câmera {} aberta (640×480 @ {}fps)", self.camera_index, self.fps)
        return True

    def read_frame(self, timeout: float = 0.05) -> Optional[np.ndarray]:
        """Retorna próximo frame da queue (bloqueia até timeout)."""
        return self._cam_thread.get(timeout) if self._cam_thread else None

    # ── Processamento ─────────────────────────────────────────────────────────

    def process_frame(
        self, frame: np.ndarray, draw: bool = True
    ) -> Tuple[Optional[IrisFrame], np.ndarray]:
        """
        Camada 1→2→3: face mesh → head pose → gaze.
        Retorna (IrisFrame, frame_anotado) ou (None, frame) se sem face.
        """
        fh, fw = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self._face_mesh.process(rgb)

        if not res.multi_face_landmarks:
            return None, frame

        lm = res.multi_face_landmarks[0].landmark

        # Camada 2: head pose
        pose  = self._head_pose.estimate(lm, fw, fh)
        rvec  = pose[0] if pose else None
        euler = _HeadPoseEstimator.euler(rvec) if rvec is not None else (0.0, 0.0, 0.0)

        # Centros da íris em pixels (para anotação e diagnóstico)
        def _px(idx):
            return (
                float(np.mean([lm[i].x * fw for i in idx])),
                float(np.mean([lm[i].y * fh for i in idx])),
            )
        l_px = _px(_IRIS_L)
        r_px = _px(_IRIS_R)

        # Camada 3: vetor de olhar compensado
        l_gaze, r_gaze, combined = self._gaze.extract(lm, rvec, fw, fh)

        if draw:
            for i in _IRIS_L + _IRIS_R:
                cv2.circle(frame, (int(lm[i].x * fw), int(lm[i].y * fh)), 2, (0, 140, 255), -1)
            cv2.circle(frame, (int(l_px[0]), int(l_px[1])), 5, (0, 0, 220), -1)
            cv2.circle(frame, (int(r_px[0]), int(r_px[1])), 5, (0, 0, 220), -1)
            if rvec is not None:
                yaw, pitch, roll = euler
                cv2.putText(
                    frame,
                    f"Y:{yaw:+.0f} P:{pitch:+.0f} R:{roll:+.0f}",
                    (4, fh - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (60, 200, 60), 1,
                )

        return IrisFrame(
            gaze_feature  = combined,
            left_gaze     = l_gaze,
            right_gaze    = r_gaze,
            head_euler    = euler,
            left_iris_px  = l_px,
            right_iris_px = r_px,
            head_ok       = rvec is not None,
            timestamp     = time.time(),
        ), frame

    # ── Encerramento ──────────────────────────────────────────────────────────

    def stop(self) -> None:
        if self._cam_thread:
            self._cam_thread.stop()
        if self._cap:
            self._cap.release()
        self._face_mesh.close()
        cv2.destroyAllWindows()
        logger.info("IrisTracker encerrado.")

    # ── Helper privado ────────────────────────────────────────────────────────

    @staticmethod
    def _build_mesh(conf: float) -> mp.solutions.face_mesh.FaceMesh:
        return mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=conf,
            min_tracking_confidence=conf,
        )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from calibration import CalibrationSession
    tracker = IrisTracker(camera_index=0)
    CalibrationSession(tracker).run()
