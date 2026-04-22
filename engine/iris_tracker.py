"""
IrisFlow — Rastreador de íris
Usa MediaPipe Face Mesh + OpenCV para detectar e rastrear o movimento da íris em tempo real.
"""

import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from loguru import logger


# Índices dos landmarks do olho no MediaPipe Face Mesh (478 pontos)
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]


@dataclass
class GazePoint:
    """Representa um ponto do olhar na tela."""
    x: float           # Coordenada X normalizada (0.0 a 1.0)
    y: float           # Coordenada Y normalizada (0.0 a 1.0)
    confidence: float  # Confiança da detecção (0.0 a 1.0)


@dataclass
class IrisData:
    """Dados brutos da íris detectada."""
    left_center: Optional[Tuple[float, float]]   # Centro da íris esquerda
    right_center: Optional[Tuple[float, float]]  # Centro da íris direita
    left_ratio: Optional[Tuple[float, float]]    # Posição relativa no olho esquerdo
    right_ratio: Optional[Tuple[float, float]]   # Posição relativa no olho direito


class IrisTracker:
    """
    Rastreia o movimento da íris em tempo real usando MediaPipe Face Mesh.

    Exemplo de uso:
        tracker = IrisTracker()
        tracker.start()
    """

    def __init__(
        self,
        camera_index: int = 0,
        target_fps: int = 30,
        show_preview: bool = True,
    ) -> None:
        self.camera_index = camera_index
        self.target_fps = target_fps
        self.show_preview = show_preview

        self._cap: Optional[cv2.VideoCapture] = None
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,  # Necessário para landmarks da íris
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._running = False
        logger.info("IrisTracker inicializado (câmera {})", camera_index)

    def _open_camera(self) -> bool:
        """Abre a câmera e verifica se está funcionando."""
        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            logger.error("Não foi possível abrir a câmera {}", self.camera_index)
            return False
        self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        logger.info("Câmera {} aberta com sucesso", self.camera_index)
        return True

    def _extract_iris_data(self, landmarks, frame_w: int, frame_h: int) -> IrisData:
        """
        Extrai os dados da íris a partir dos landmarks do MediaPipe.

        Retorna:
            IrisData com centros e ratios de ambas as íris
        """
        def get_center(indices):
            pts = [(landmarks[i].x * frame_w, landmarks[i].y * frame_h) for i in indices]
            cx = np.mean([p[0] for p in pts])
            cy = np.mean([p[1] for p in pts])
            return (cx, cy)

        def get_eye_bbox(indices):
            xs = [landmarks[i].x * frame_w for i in indices]
            ys = [landmarks[i].y * frame_h for i in indices]
            return min(xs), max(xs), min(ys), max(ys)

        def calc_ratio(iris_center, eye_indices):
            x_min, x_max, y_min, y_max = get_eye_bbox(eye_indices)
            eye_w = x_max - x_min
            eye_h = y_max - y_min
            if eye_w == 0 or eye_h == 0:
                return None
            ratio_x = (iris_center[0] - x_min) / eye_w
            ratio_y = (iris_center[1] - y_min) / eye_h
            return (ratio_x, ratio_y)

        left_center = get_center(LEFT_IRIS)
        right_center = get_center(RIGHT_IRIS)
        left_ratio = calc_ratio(left_center, LEFT_EYE)
        right_ratio = calc_ratio(right_center, RIGHT_EYE)

        return IrisData(
            left_center=left_center,
            right_center=right_center,
            left_ratio=left_ratio,
            right_ratio=right_ratio,
        )

    def process_frame(self, frame: np.ndarray) -> Tuple[Optional[IrisData], np.ndarray]:
        """
        Processa um frame da câmera e detecta a íris.

        Args:
            frame: Frame BGR do OpenCV

        Retorna:
            Tupla (IrisData ou None, frame com anotações)
        """
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
        """Desenha os landmarks no frame para visualização."""
        # Desenha contorno dos olhos
        for idx in LEFT_EYE + RIGHT_EYE:
            x = int(landmarks[idx].x * frame_w)
            y = int(landmarks[idx].y * frame_h)
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        # Desenha centros das íris
        if iris_data.left_center:
            cx, cy = int(iris_data.left_center[0]), int(iris_data.left_center[1])
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

        if iris_data.right_center:
            cx, cy = int(iris_data.right_center[0]), int(iris_data.right_center[1])
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

        return frame

    def start(self, on_iris_data=None) -> None:
        """
        Inicia o loop de captura e processamento.

        Args:
            on_iris_data: Callback chamado a cada frame com IrisData detectado.
                          Assinatura: on_iris_data(iris_data: IrisData) -> None
        """
        if not self._open_camera():
            return

        self._running = True
        logger.info("Iniciando rastreamento de íris. Pressione Q para sair.")

        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                logger.warning("Frame não capturado — verificando câmera...")
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
        """Encerra o rastreador e libera recursos."""
        self._running = False
        if self._cap:
            self._cap.release()
        cv2.destroyAllWindows()
        self._face_mesh.close()
        logger.info("IrisTracker encerrado.")


# --- Execução direta para teste ---
if __name__ == "__main__":
    def on_data(data: IrisData):
        if data.left_ratio:
            print(f"Olho esq: ({data.left_ratio[0]:.2f}, {data.left_ratio[1]:.2f})", end="\r")

    tracker = IrisTracker(show_preview=True)
    tracker.start(on_iris_data=on_data)
