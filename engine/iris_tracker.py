"""
IrisFlow — Captura · Detecção · Gaze via CNN

Camada 1  Captura threaded com queue.Queue — sem perda de frames
Camada 2  Detecção de rosto via haarcascade (OpenCV)
Camada 3  CNN end-to-end (IrisGazeNet): crop dos olhos + head pose → posição na tela
"""

import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from loguru import logger

from iris_model import IrisGazeNet, get_head_pose, load_model

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "iris_gaze_model.pth")
_EYE_H      = 64    # altura de entrada da CNN
_EYE_W      = 128   # largura de entrada da CNN
_EMA_ALPHA  = 0.20  # suavização EMA aplicada sobre o output da CNN


# ── IrisFrame ─────────────────────────────────────────────────────────────────

@dataclass
class IrisFrame:
    """Dados de um frame processado pelo pipeline CNN."""
    gaze_feature:  np.ndarray                  # [x, y] em [0, 1] — saída da CNN pós-EMA
    left_gaze:     np.ndarray                  # zeros (mantido por compatibilidade)
    right_gaze:    np.ndarray                  # zeros (mantido por compatibilidade)
    head_euler:    Tuple[float, float, float]  # (yaw°, pitch°, roll°)
    left_iris_px:  Tuple[float, float]         # centro estimado olho esq. em pixels
    right_iris_px: Tuple[float, float]         # centro estimado olho dir. em pixels
    head_ok:       bool                        # False = rosto não detectado
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


# ── IrisTracker ───────────────────────────────────────────────────────────────

class IrisTracker:
    """
    Orquestra o pipeline CNN de rastreamento ocular.

    Uso típico:
        tracker = IrisTracker()
        tracker.open_camera()
        frame = tracker.read_frame()
        iris_frame, annotated = tracker.process_frame(frame)
        tracker.stop()
    """

    def __init__(self, camera_index: int = 0, fps: int = 30) -> None:
        self.camera_index = camera_index
        self.fps          = fps
        self._cap:        Optional[cv2.VideoCapture] = None
        self._cam_thread: Optional[_FrameQueue]       = None

        # Detecção de rosto
        cascade_path   = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade  = cv2.CascadeClassifier(cascade_path)

        # CNN
        self._model: Optional[IrisGazeNet] = load_model(_MODEL_PATH)
        self._device = torch.device("cpu")

        # Estado EMA
        self._ema_x: Optional[float] = None
        self._ema_y: Optional[float] = None

        if self._model is None:
            logger.warning(
                "IrisTracker: usando fallback (centro do frame). "
                "Execute scripts/train_iris.py para treinar o modelo."
            )
        logger.info("IrisTracker inicializado (câmera {})", camera_index)

    # ── Controles (mantidos por compatibilidade) ──────────────────────────────

    def set_dominant_eye(self, eye: str) -> None:
        pass  # CNN processa a região dos dois olhos simultaneamente

    def set_glasses_mode(self, enabled: bool) -> None:
        pass  # Sem impacto no pipeline CNN

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
        Pipeline: haarcascade → crop dos olhos → CNN → EMA.
        Retorna (IrisFrame, frame_anotado) ou (None, frame) se rosto não detectado.
        """
        fh, fw = frame.shape[:2]
        gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self._cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
        if len(faces) == 0:
            return None, frame

        # Maior rosto detectado
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

        # Crop da região dos olhos: terço superior do rosto
        ey1 = max(0,  y + int(h * 0.15))
        ey2 = min(fh, y + int(h * 0.55))
        ex1 = max(0,  x)
        ex2 = min(fw, x + w)

        eye_crop = frame[ey1:ey2, ex1:ex2]
        if eye_crop.size == 0:
            return None, frame

        # Prepara tensor para a CNN: resize → RGB → [0,1] → (1,3,H,W)
        eye_resized = cv2.resize(eye_crop, (_EYE_W, _EYE_H))
        eye_rgb     = cv2.cvtColor(eye_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_tensor  = torch.from_numpy(eye_rgb.transpose(2, 0, 1)).unsqueeze(0)

        # Head pose
        yaw, pitch, roll = get_head_pose(frame)
        hp_tensor = torch.tensor(
            [[yaw / 90.0, pitch / 60.0, roll / 30.0]], dtype=torch.float32
        )

        # Inferência CNN ou fallback
        if self._model is not None:
            with torch.no_grad():
                xy = self._model(img_tensor, hp_tensor)[0].numpy()
        else:
            xy = np.array([0.5, 0.5], dtype=np.float32)

        gaze_x, gaze_y = float(xy[0]), float(xy[1])

        # EMA α=0.20
        if self._ema_x is None:
            self._ema_x, self._ema_y = gaze_x, gaze_y
        else:
            self._ema_x += _EMA_ALPHA * (gaze_x - self._ema_x)
            self._ema_y += _EMA_ALPHA * (gaze_y - self._ema_y)

        gaze_feature = np.array([self._ema_x, self._ema_y], dtype=np.float64)

        # Centros estimados dos olhos em pixels (para anotação e diagnóstico)
        l_px = (float(x + w * 0.28), float(y + h * 0.35))
        r_px = (float(x + w * 0.72), float(y + h * 0.35))

        if draw:
            cv2.rectangle(frame, (ex1, ey1), (ex2, ey2), (0, 140, 255), 2)
            cv2.circle(frame, (int(l_px[0]), int(l_px[1])), 5, (0, 0, 220), -1)
            cv2.circle(frame, (int(r_px[0]), int(r_px[1])), 5, (0, 0, 220), -1)
            cv2.putText(
                frame,
                f"Y:{yaw:+.0f} P:{pitch:+.0f} R:{roll:+.0f}",
                (4, fh - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (60, 200, 60), 1,
            )

        return IrisFrame(
            gaze_feature  = gaze_feature,
            left_gaze     = np.zeros(3, dtype=np.float64),
            right_gaze    = np.zeros(3, dtype=np.float64),
            head_euler    = (yaw, pitch, roll),
            left_iris_px  = l_px,
            right_iris_px = r_px,
            head_ok       = True,
            timestamp     = time.time(),
        ), frame

    # ── Encerramento ──────────────────────────────────────────────────────────

    def stop(self) -> None:
        if self._cam_thread:
            self._cam_thread.stop()
        if self._cap:
            self._cap.release()
        cv2.destroyAllWindows()
        logger.info("IrisTracker encerrado.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from calibration import CalibrationSession
    tracker = IrisTracker(camera_index=0)
    CalibrationSession(tracker).run()
