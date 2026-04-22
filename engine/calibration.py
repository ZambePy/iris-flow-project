"""
IrisFlow — Calibração
Mapeia as posições da íris para coordenadas de tela.
Calibração manual (Mês 1) e adaptativa por LSTM (Mês 3+).
"""

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
    Calibração linear simples para o MVP (Mês 1–2).
    Mapeia ratio da íris → posição na tela usando interpolação linear.
    Requer pelo menos 4 pontos de calibração (cantos da tela).
    """

    MIN_POINTS = 4

    def __init__(self) -> None:
        self.model = CalibrationModel()
        self._x_coeffs: Optional[Tuple[float, float]] = None  # (slope, intercept)
        self._y_coeffs: Optional[Tuple[float, float]] = None

    def add_calibration_point(
        self,
        iris_ratio: Tuple[float, float],
        screen_pos: Tuple[float, float],
    ) -> None:
        """Adiciona um ponto de calibração."""
        self.model.add_point(iris_ratio, screen_pos)

    def calibrate(self) -> bool:
        """
        Calcula os coeficientes de mapeamento linear.

        Retorna:
            True se a calibração foi bem-sucedida.
        """
        if len(self.model.points) < self.MIN_POINTS:
            logger.warning(
                "Calibração requer {} pontos, apenas {} fornecidos.",
                self.MIN_POINTS,
                len(self.model.points),
            )
            return False

        iris_xs = [p.iris_ratio[0] for p in self.model.points]
        iris_ys = [p.iris_ratio[1] for p in self.model.points]
        screen_xs = [p.screen_pos[0] for p in self.model.points]
        screen_ys = [p.screen_pos[1] for p in self.model.points]

        # Regressão linear simples: y = mx + b
        self._x_coeffs = self._linear_regression(iris_xs, screen_xs)
        self._y_coeffs = self._linear_regression(iris_ys, screen_ys)
        self.model.is_calibrated = True

        logger.info("Calibração concluída com {} pontos.", len(self.model.points))
        return True

    def predict(self, iris_ratio: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """
        Converte ratio da íris em posição na tela.

        Args:
            iris_ratio: (ratio_x, ratio_y) da íris

        Retorna:
            (screen_x, screen_y) normalizados, ou None se não calibrado.
        """
        if not self.model.is_calibrated or not self._x_coeffs or not self._y_coeffs:
            logger.warning("Modelo não calibrado. Chame calibrate() primeiro.")
            return None

        screen_x = self._apply(iris_ratio[0], self._x_coeffs)
        screen_y = self._apply(iris_ratio[1], self._y_coeffs)

        # Garante que fica dentro dos limites da tela
        screen_x = max(0.0, min(1.0, screen_x))
        screen_y = max(0.0, min(1.0, screen_y))

        return (screen_x, screen_y)

    @staticmethod
    def _linear_regression(xs: List[float], ys: List[float]) -> Tuple[float, float]:
        n = len(xs)
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        numerator = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n))
        denominator = sum((xs[i] - mean_x) ** 2 for i in range(n))
        slope = numerator / denominator if denominator != 0 else 1.0
        intercept = mean_y - slope * mean_x
        return (slope, intercept)

    @staticmethod
    def _apply(x: float, coeffs: Tuple[float, float]) -> float:
        slope, intercept = coeffs
        return slope * x + intercept


# TODO (Mês 3): Implementar LSTMCalibrator que usa TensorFlow Lite
# para calibração adaptativa — aprende o padrão de movimento de cada
# usuário ao longo das sessões, sem necessidade de recalibrar manualmente.
