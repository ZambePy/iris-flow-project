"""
IrisFlow — engine/iris_model.py

CNN end-to-end para estimativa de posição de olhar (gaze) na tela.
Input:  imagem do olho recortada (3, 64, 128) + head_pose (3,) = yaw, pitch, roll
Output: (x, y) normalizado em [0, 1] = posição na tela
"""

import os
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


_MODEL_PATH = os.path.join(os.path.dirname(__file__), "iris_gaze_model.pth")

# Pontos 3D genéricos da face para solvePnP (mm, origem no nariz)
_FACE_3D = np.array([
    [ 0.0,    0.0,    0.0  ],   # ponta do nariz
    [ 0.0,  -63.6,  -12.5  ],   # queixo
    [-43.3,  32.7,  -26.0  ],   # canto ext. olho esq.
    [ 43.3,  32.7,  -26.0  ],   # canto ext. olho dir.
    [-28.9, -28.9,  -24.1  ],   # canto esq. boca
    [ 28.9, -28.9,  -24.1  ],   # canto dir. boca
], dtype=np.float64)

# Índices dos landmarks dlib (68 pts) correspondentes ao modelo 3D acima
_DLIB_HP_IDX = [30, 8, 36, 45, 48, 54]


# ── IrisGazeNet ───────────────────────────────────────────────────────────────

class IrisGazeNet(nn.Module):
    """
    CNN end-to-end: imagem do olho (3, 224, 224) + head_pose (3,) → (x, y) em [0, 1].
    Backbone: MobileNetV2 pré-treinado no ImageNet.
    """

    def __init__(self) -> None:
        super().__init__()
        base = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        self.backbone = base.features  # saída: (B, 1280, 7, 7) para input 224×224

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # (B, 1280, 1, 1)
            nn.Flatten(),             # (B, 1280)
        )
        self.regressor = nn.Sequential(
            nn.Linear(1280 + 3, 256),  # +3 para head_pose
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
            nn.Sigmoid(),
        )

    def forward(self, img: torch.Tensor, head_pose: torch.Tensor) -> torch.Tensor:
        """
        img:       (B, 3, 224, 224)
        head_pose: (B, 3) — yaw, pitch, roll normalizados em [-1, 1]
        retorna:   (B, 2) — x, y em [0, 1]
        """
        x = self.backbone(img)                  # (B, 1280, 7, 7)
        x = self.head(x)                        # (B, 1280)
        x = torch.cat([x, head_pose], dim=1)   # (B, 1283)
        return self.regressor(x)                # (B, 2)


def load_model(path: str = _MODEL_PATH) -> Optional[IrisGazeNet]:
    """Carrega modelo treinado do arquivo .pth. Retorna None se não encontrado."""
    if not os.path.exists(path):
        logger.warning(
            "Modelo não encontrado em {}. "
            "Execute scripts/train_iris.py para treinar.",
            path,
        )
        return None
    try:
        model = IrisGazeNet()
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
        model.load_state_dict(state)
        model.eval()
        if isinstance(ckpt, dict) and "epoch" in ckpt:
            logger.info("Modelo IrisGazeNet carregado de {} (epoch {})", path, ckpt["epoch"])
        else:
            logger.info("Modelo IrisGazeNet carregado de {}", path)
        return model
    except Exception as e:
        logger.error("Falha ao carregar modelo IrisGazeNet: {}", e)
        return None


# ── Head pose sem MediaPipe ───────────────────────────────────────────────────

_face_cascade:  Optional[cv2.CascadeClassifier] = None
_dlib_detector  = None
_dlib_predictor = None


def _ensure_cascade() -> cv2.CascadeClassifier:
    global _face_cascade
    if _face_cascade is None:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_cascade = cv2.CascadeClassifier(path)
        if _face_cascade.empty():
            logger.error("Haarcascade não encontrado em {}", path)
    return _face_cascade


def _try_load_dlib() -> bool:
    """Tenta carregar dlib com 68 landmarks. Retorna True se bem-sucedido."""
    global _dlib_detector, _dlib_predictor
    if _dlib_detector is not None:
        return _dlib_predictor is not None
    try:
        import dlib  # type: ignore
        _dlib_detector = dlib.get_frontal_face_detector()
        dat = os.path.join(
            os.path.dirname(__file__), "..",
            "shape_predictor_68_face_landmarks.dat",
        )
        if os.path.exists(dat):
            _dlib_predictor = dlib.shape_predictor(dat)
            logger.info("dlib 68-landmarks carregado.")
            return True
        logger.info(
            "shape_predictor_68_face_landmarks.dat não encontrado — "
            "head pose via haarcascade."
        )
        return False
    except ImportError:
        logger.debug("dlib não instalado — head pose via haarcascade.")
        return False


def _rvec_to_euler(rvec: np.ndarray) -> Tuple[float, float, float]:
    """Rodrigues → (yaw°, pitch°, roll°). Convenção ZYX."""
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


def get_head_pose(frame: np.ndarray) -> Tuple[float, float, float]:
    """
    Estima (yaw°, pitch°, roll°) sem MediaPipe.

    Tentativa 1: dlib 68 landmarks + solvePnP.
    Fallback:    haarcascade → estimativa grosseira por posição da face.
    Fallback 2:  (0.0, 0.0, 0.0) se face não detectada.
    """
    fh, fw = frame.shape[:2]
    cam  = np.array(
        [[float(fw), 0, fw / 2], [0, float(fw), fh / 2], [0, 0, 1]],
        dtype=np.float64,
    )
    dist = np.zeros((4, 1), dtype=np.float64)

    # Tentativa 1: dlib
    if _try_load_dlib() and _dlib_predictor is not None:
        try:
            import dlib  # type: ignore
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = _dlib_detector(gray, 0)
            if faces:
                shp   = _dlib_predictor(gray, faces[0])
                pts2d = np.array(
                    [[shp.part(i).x, shp.part(i).y] for i in _DLIB_HP_IDX],
                    dtype=np.float64,
                )
                ok, rvec, _ = cv2.solvePnP(
                    _FACE_3D, pts2d, cam, dist, flags=cv2.SOLVEPNP_ITERATIVE
                )
                if ok:
                    return _rvec_to_euler(rvec)
        except Exception:
            pass

    # Fallback: haarcascade
    cascade = _ensure_cascade()
    if not cascade.empty():
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        if len(faces) > 0:
            x, y, w, h = faces[0]
            cx    = float(x + w / 2) / fw
            cy    = float(y + h / 2) / fh
            # Estimativa grosseira: desvio do centro → ângulos em graus
            yaw   = (cx - 0.5) * 60.0
            pitch = (cy - 0.5) * 40.0
            return yaw, pitch, 0.0

    return 0.0, 0.0, 0.0
