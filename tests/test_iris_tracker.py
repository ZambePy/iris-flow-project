"""
IrisFlow — Testes do engine de rastreamento de íris.
"""

import numpy as np
import pytest
from engine.calibration import LinearCalibrator, CalibrationPoint
from engine.iris_tracker import IrisData


class TestLinearCalibrator:
    """Testes da calibração linear."""

    def test_calibrate_requires_minimum_points(self):
        cal = LinearCalibrator()
        result = cal.calibrate()
        assert result is False

    def test_calibrate_with_four_points(self):
        cal = LinearCalibrator()
        # 4 cantos da tela
        cal.add_calibration_point((0.2, 0.2), (0.0, 0.0))  # Canto superior esquerdo
        cal.add_calibration_point((0.8, 0.2), (1.0, 0.0))  # Canto superior direito
        cal.add_calibration_point((0.2, 0.8), (0.0, 1.0))  # Canto inferior esquerdo
        cal.add_calibration_point((0.8, 0.8), (1.0, 1.0))  # Canto inferior direito
        result = cal.calibrate()
        assert result is True
        assert cal.model.is_calibrated is True

    def test_predict_before_calibration_returns_none(self):
        cal = LinearCalibrator()
        result = cal.predict((0.5, 0.5))
        assert result is None

    def test_predict_center_after_calibration(self):
        cal = LinearCalibrator()
        cal.add_calibration_point((0.2, 0.2), (0.0, 0.0))
        cal.add_calibration_point((0.8, 0.2), (1.0, 0.0))
        cal.add_calibration_point((0.2, 0.8), (0.0, 1.0))
        cal.add_calibration_point((0.8, 0.8), (1.0, 1.0))
        cal.calibrate()

        result = cal.predict((0.5, 0.5))
        assert result is not None
        # Centro da íris deve mapear para ~centro da tela
        assert abs(result[0] - 0.5) < 0.1
        assert abs(result[1] - 0.5) < 0.1

    def test_predict_clamps_to_screen_bounds(self):
        cal = LinearCalibrator()
        cal.add_calibration_point((0.2, 0.2), (0.0, 0.0))
        cal.add_calibration_point((0.8, 0.2), (1.0, 0.0))
        cal.add_calibration_point((0.2, 0.8), (0.0, 1.0))
        cal.add_calibration_point((0.8, 0.8), (1.0, 1.0))
        cal.calibrate()

        # Posição extrema deve ser limitada a [0, 1]
        result = cal.predict((0.0, 0.0))
        assert result is not None
        assert 0.0 <= result[0] <= 1.0
        assert 0.0 <= result[1] <= 1.0

    def test_clear_resets_calibration(self):
        cal = LinearCalibrator()
        cal.add_calibration_point((0.2, 0.2), (0.0, 0.0))
        cal.model.is_calibrated = True
        cal.model.clear()
        assert cal.model.is_calibrated is False
        assert len(cal.model.points) == 0


class TestIrisData:
    """Testes da estrutura de dados da íris."""

    def test_iris_data_creation(self):
        data = IrisData(
            left_center=(320.0, 240.0),
            right_center=(960.0, 240.0),
            left_ratio=(0.5, 0.5),
            right_ratio=(0.5, 0.5),
        )
        assert data.left_center == (320.0, 240.0)
        assert data.right_ratio == (0.5, 0.5)

    def test_iris_data_can_be_none(self):
        data = IrisData(
            left_center=None,
            right_center=None,
            left_ratio=None,
            right_ratio=None,
        )
        assert data.left_center is None
