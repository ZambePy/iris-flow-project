"""
IrisFlow — Calibração
Mapeia as posições da íris para coordenadas de tela.
Calibração manual (Mês 1) e adaptativa por LSTM (Mês 3+).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from loguru import logger


@dataclass
class CalibrationPoint:
    """Par (posição do olhar) → (coordenada de tela)."""
    iris_ratio: Tuple[float, float]   # (ratio_x, ratio_y) da íris
    screen_pos: Tuple[float, float]   # (x, y) na tela, normalizados 0.0–1.0


@dataclass
class CalibrationModel:
    """Modelo de calibração com pontos coletados."""
    points: List[CalibrationPoint] = field(default_factory=list)
    is_calibrated: bool = False
    user_id: Optional[str] = None

    def add_point(self, iris_ratio: Tuple[float, float], screen_pos: Tuple[float, float]) -> None:
        self.points.append(CalibrationPoint(iris_ratio=iris_ratio, screen_pos=screen_pos))
        logger.debug("Ponto de calibração adicionado: íris={}, tela={}", iris_ratio, screen_pos)

    def clear(self) -> None:
        self.points.clear()
        self.is_calibrated = False
        logger.info("Calibração resetada.")


class LinearCalibrator:
    """
    Calibrador polinomial adaptativo.

    Escolhe o modelo com base no número de pontos disponíveis:
        4–5 pontos  → afim       [x, y, 1]               (3 parâmetros)
        6–8 pontos  → bilinear   [x, y, x·y, 1]          (4 parâmetros)
        9+ pontos   → quadrático [x, y, x², y², x·y, 1]  (6 parâmetros)

    Todos resolvidos por mínimos quadrados (lstsq), portanto o sistema é
    sempre sobredeterminado quando há mais pontos que parâmetros.
    """

    MIN_POINTS = 4

    def __init__(self) -> None:
        self.model   = CalibrationModel()
        self._coeffs_x: Optional[np.ndarray] = None
        self._coeffs_y: Optional[np.ndarray] = None
        self._degree:   str = "affine"
        # Normalização de saída: estica o range de calibração para [0, 1]
        self._sx_min: float = 0.0
        self._sx_max: float = 1.0
        self._sy_min: float = 0.0
        self._sy_max: float = 1.0

    def add_calibration_point(
        self,
        iris_ratio: Tuple[float, float],
        screen_pos: Tuple[float, float],
    ) -> None:
        """Adiciona um ponto de calibração."""
        self.model.add_point(iris_ratio, screen_pos)

    def calibrate(self, weights: Optional[np.ndarray] = None) -> bool:
        """
        Ajusta o modelo polinomial com os pontos coletados.

        weights: array de pesos por ponto (mesmo comprimento que model.points).
                 Usa mínimos quadrados ponderados — pesos maiores forçam o modelo
                 a ajustar melhor aquele ponto (ex.: centro com peso 2.5×).

        Após o ajuste, aplica normalização linear que estende o range de
        calibração ao intervalo [0, 1] com margem de 10% para os cantos,
        permitindo extrapolação até as bordas reais da tela.
        """
        n = len(self.model.points)
        if n < self.MIN_POINTS:
            logger.warning(
                "Calibração requer {} pontos, {} fornecidos.",
                self.MIN_POINTS, n,
            )
            return False

        ix = np.array([p.iris_ratio[0] for p in self.model.points])
        iy = np.array([p.iris_ratio[1] for p in self.model.points])
        sx = np.array([p.screen_pos[0] for p in self.model.points])
        sy = np.array([p.screen_pos[1] for p in self.model.points])

        if n >= 9:
            A = np.column_stack([ix, iy, ix ** 2, iy ** 2, ix * iy, np.ones(n)])
            self._degree = "quadratic"
        elif n >= 6:
            A = np.column_stack([ix, iy, ix * iy, np.ones(n)])
            self._degree = "bilinear"
        else:
            A = np.column_stack([ix, iy, np.ones(n)])
            self._degree = "affine"

        # Mínimos quadrados ponderados
        if weights is not None:
            W = np.sqrt(np.asarray(weights, dtype=float))
            A_fit  = A  * W[:, np.newaxis]
            sx_fit = sx * W
            sy_fit = sy * W
        else:
            A_fit, sx_fit, sy_fit = A, sx, sy

        self._coeffs_x, _, _, _ = np.linalg.lstsq(A_fit, sx_fit, rcond=None)
        self._coeffs_y, _, _, _ = np.linalg.lstsq(A_fit, sy_fit, rcond=None)

        # Normalização de saída: [sx_min, sx_max] → [0, 1] com 10% de margem
        # A margem permite que o cursor alcance as bordas da tela por extrapolação.
        MARGIN = 0.10
        sx_rng = sx.max() - sx.min()
        sy_rng = sy.max() - sy.min()
        self._sx_min = sx.min() - MARGIN * sx_rng
        self._sx_max = sx.max() + MARGIN * sx_rng
        self._sy_min = sy.min() - MARGIN * sy_rng
        self._sy_max = sy.max() + MARGIN * sy_rng

        self.model.is_calibrated = True
        logger.info(
            "Calibração concluída com {} pontos (modelo: {}).", n, self._degree
        )
        return True

    def predict(self, iris_ratio: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """
        Converte ratio da íris em posição na tela.

        Args:
            iris_ratio: (ratio_x, ratio_y) da íris

        Retorna:
            (screen_x, screen_y) normalizados 0–1, ou None se não calibrado.
        """
        if not self.model.is_calibrated:
            logger.warning("Modelo não calibrado. Chame calibrate() primeiro.")
            return None

        ix, iy = iris_ratio[0], iris_ratio[1]

        if self._degree == "quadratic":
            feat = np.array([ix, iy, ix ** 2, iy ** 2, ix * iy, 1.0])
        elif self._degree == "bilinear":
            feat = np.array([ix, iy, ix * iy, 1.0])
        else:
            feat = np.array([ix, iy, 1.0])

        raw_x = float(feat @ self._coeffs_x)
        raw_y = float(feat @ self._coeffs_y)

        # Aplica normalização: estica range de calibração para [0, 1]
        rng_x = self._sx_max - self._sx_min
        rng_y = self._sy_max - self._sy_min
        sx = (raw_x - self._sx_min) / rng_x if rng_x > 1e-6 else raw_x
        sy = (raw_y - self._sy_min) / rng_y if rng_y > 1e-6 else raw_y

        return (max(0.0, min(1.0, sx)), max(0.0, min(1.0, sy)))


# TODO (Mês 3): Implementar LSTMCalibrator que usa TensorFlow Lite
# para calibração adaptativa — aprende o padrão de movimento de cada
# usuário ao longo das sessões, sem necessidade de recalibrar manualmente.
