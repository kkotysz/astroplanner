from __future__ import annotations

import math

from matplotlib.figure import Figure
import numpy as np
from PySide6.QtGui import QColor

from astroplanner.models import Target
from astroplanner.visibility_matplotlib import VisibilityMatplotlibCoordinator


class _DummyPlanner:
    def __init__(self) -> None:
        self._plot_airmass = False

    def _theme_qcolor(self, _key: str, fallback: str) -> QColor:
        return QColor(fallback)

    @staticmethod
    def _qcolor_rgba_mpl(color: QColor, alpha: float) -> tuple[float, float, float, float]:
        return (color.redF(), color.greenF(), color.blueF(), float(alpha))


def test_visibility_matplotlib_target_color_key_prefers_stable_ids() -> None:
    source_target = Target(
        name="Friendly name",
        ra=10.0,
        dec=20.0,
        source_object_id=" Gaia DR3 123 ",
    )
    named_target = Target(name=" M 31 ", ra=10.0, dec=20.0)
    coord_target = Target(name="", ra=10.1234567, dec=-20.7654321)

    assert VisibilityMatplotlibCoordinator._target_color_key(source_target) == "id:gaia dr3 123"
    assert VisibilityMatplotlibCoordinator._target_color_key(named_target) == "name:m 31"
    assert VisibilityMatplotlibCoordinator._target_color_key(coord_target) == "coord:10.123457,-20.765432"


def test_visibility_matplotlib_airmass_values_mask_invalid_altitudes() -> None:
    coordinator = VisibilityMatplotlibCoordinator(_DummyPlanner())  # type: ignore[arg-type]
    values = coordinator._airmass_from_altitude([90.0, 30.0, 0.0, -5.0])

    assert math.isclose(float(values[0]), 1.0, rel_tol=0.02)
    assert float(values[1]) > 1.9
    assert np.isnan(values[2])
    assert np.isnan(values[3])


def test_visibility_matplotlib_polar_rgba_array_clamps_alpha() -> None:
    rgba = VisibilityMatplotlibCoordinator._polar_rgba_array(QColor("#336699"), np.array([-1.0, 0.5, 2.0]))

    assert rgba.shape == (3, 4)
    assert rgba[0, 3] == 0.0
    assert rgba[1, 3] == 0.5
    assert rgba[2, 3] == 1.0


def test_visibility_selected_line_uses_layered_neon_glow() -> None:
    coordinator = VisibilityMatplotlibCoordinator(_DummyPlanner())  # type: ignore[arg-type]
    ax = Figure().add_subplot()
    (line,) = ax.plot([0, 1], [10, 20], color="#59f3ff")

    coordinator._apply_visibility_line_style(line, is_over=True, is_selected=True)

    assert line.get_alpha() == 1.0
    assert line.get_linewidth() == 2.8
    assert line.get_zorder() == 80
    assert len(line.get_path_effects()) == 5

    coordinator._apply_visibility_line_style(line, is_over=False, is_selected=False)
    assert line.get_path_effects() == []
    assert line.get_zorder() == 1


def test_visibility_selected_base_line_does_not_get_dashed_glow() -> None:
    coordinator = VisibilityMatplotlibCoordinator(_DummyPlanner())  # type: ignore[arg-type]
    ax = Figure().add_subplot()
    (line,) = ax.plot([0, 1], [10, 20], color="#59f3ff", linestyle="--")

    coordinator._apply_visibility_line_style(line, is_over=False, is_selected=True)

    assert line.get_linewidth() == 1.4
    assert line.get_alpha() == 0.42
    assert line.get_zorder() == 2
    assert line.get_path_effects() == []
