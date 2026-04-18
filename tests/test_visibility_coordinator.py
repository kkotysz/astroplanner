from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QItemSelectionModel
from PySide6.QtWidgets import QApplication, QWidget

from astroplanner.models import Target
from astroplanner.ui.common import TargetTableView
from astroplanner.ui.targets import TargetTableModel
from astroplanner.visibility_coordinator import VisibilityCoordinator


class _DummyVisibilityPlanner(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.targets = [
            Target(name="M31", ra=10.684, dec=41.269),
            Target(name="M42", ra=83.822, dec=-5.391),
        ]
        self.table_model = TargetTableModel(self.targets, site=None, parent=self)
        self.table_view = TargetTableView()
        self.table_view.setModel(self.table_model)
        self._defer_startup_preview_updates = False
        self._cutout_updates: list[Target | None] = []

    def _update_cutout_preview_for_target(self, target: Target | None) -> None:
        self._cutout_updates.append(target)


class _DummyWebPage:
    def __init__(self) -> None:
        self.scripts: list[str] = []

    def runJavaScript(self, script: str) -> None:
        self.scripts.append(script)


class _DummyWebView:
    def __init__(self) -> None:
        self._page = _DummyWebPage()

    def page(self) -> _DummyWebPage:
        return self._page


def test_visibility_coordinator_selection_and_cutout_smoke() -> None:
    app = QApplication.instance() or QApplication([])
    assert app is not None

    planner = _DummyVisibilityPlanner()
    coordinator = VisibilityCoordinator(planner)
    coordinator.bind()

    idx = planner.table_model.index(1, TargetTableModel.COL_NAME)
    planner.table_view.selectionModel().select(idx, QItemSelectionModel.Select | QItemSelectionModel.Rows)

    assert coordinator.selected_target_names() == ["M42"]

    coordinator.schedule_selected_cutout_update(planner.targets[1])
    coordinator.flush_selected_cutout_update()
    assert planner._cutout_updates == [planner.targets[1]]


def test_visibility_coordinator_caps_web_html_cache() -> None:
    app = QApplication.instance() or QApplication([])
    assert app is not None

    planner = _DummyVisibilityPlanner()
    coordinator = VisibilityCoordinator(planner)
    coordinator.bind()

    for idx in range(10):
        coordinator.store_visibility_web_html_cache(f"key-{idx}", f"html-{idx}")

    assert len(planner._visibility_web_html_cache) == 8
    assert "key-0" not in planner._visibility_web_html_cache
    assert "key-1" not in planner._visibility_web_html_cache
    assert planner._visibility_web_html_cache["key-9"] == "html-9"


def test_visibility_web_selection_style_moves_selected_traces_to_front() -> None:
    app = QApplication.instance() or QApplication([])
    assert app is not None

    planner = _DummyVisibilityPlanner()
    planner._use_visibility_web = True
    planner._visibility_web_has_content = True
    planner.visibility_web = _DummyWebView()
    coordinator = VisibilityCoordinator(planner)

    coordinator.apply_visibility_web_selection_style({"M42"})

    script = planner.visibility_web.page().scripts[-1]
    assert "Plotly.moveTraces" in script
    assert "selectedHighIndices.map((_,j)=>gd.data.length-selectedHighIndices.length+j)" in script
    assert "widths.push(1.4)" in script
    assert "widths.push(isSelected?4.1:1.9)" in script
    assert "drop-shadow(0 0 15px" in script
    assert "7.6" not in script


def test_polar_axes_path_line_keeps_moon_path_dashed() -> None:
    app = QApplication.instance() or QApplication([])
    assert app is not None

    planner = _DummyVisibilityPlanner()
    from matplotlib.figure import Figure
    planner.polar_ax = Figure().add_subplot(projection="polar")
    coordinator = VisibilityCoordinator(planner)

    line = coordinator.add_polar_axes_path_line(
        np.array([0.2, 0.4]),
        np.array([0.8, 0.7]),
        color="silver",
        linewidth=1.15,
        linestyle=VisibilityCoordinator.MOON_PATH_LINESTYLE,
        alpha=0.88,
        zorder=2.4,
    )

    assert line.is_dashed()
    assert line.get_gapcolor() is None


def test_build_polar_visible_path_splits_below_horizon_without_horizon_arc() -> None:
    theta, radius = VisibilityCoordinator.build_polar_visible_path(
        alt_series=[-1.0, 20.0, 25.0, -2.0, 30.0, 35.0],
        az_series=[0.0, 350.0, 10.0, 20.0, 355.0, 5.0],
    )

    assert theta.size == radius.size
    assert theta.size > 0
    assert np.isnan(theta[:-1]).any()
    assert np.all(radius[np.isfinite(radius)] <= 90.0)
    assert 90.0 in set(radius[np.isfinite(radius)])


def test_observing_pass_selects_single_rise_to_set_segment_overlapping_night() -> None:
    selected_alt, selected_az = VisibilityCoordinator.observing_pass_series(
        times_series=[0.0, 6.0, 12.0, 18.0, 24.0],
        alt_series=[20.0, 25.0, -10.0, -12.0, 22.0],
        az_series=[80.0, 100.0, 140.0, 180.0, 220.0],
        night_start_num=18.1,
        night_end_num=23.9,
    )

    assert selected_alt.size == selected_az.size
    assert selected_alt.size >= 2
    assert np.all(selected_az > 180.0)
    assert 90.0 in set(90.0 - selected_alt)


def test_build_polar_visible_path_unwraps_azimuth_without_breaking() -> None:
    theta, radius = VisibilityCoordinator.build_polar_visible_path(
        alt_series=[20.0, 25.0, 30.0],
        az_series=[350.0, 10.0, 20.0],
    )

    assert theta.size == radius.size
    finite_theta = theta[np.isfinite(theta)]
    assert finite_theta.size == 3
    assert np.rad2deg(finite_theta[1]) > 360.0


def test_build_polar_axes_path_splits_below_horizon_without_horizon_arc() -> None:
    x, y = VisibilityCoordinator.build_polar_axes_path(
        alt_series=[20.0, -10.0, 25.0],
        az_series=[350.0, 0.0, 10.0],
    )

    assert x.size == y.size
    finite_x = x[np.isfinite(x)]
    finite_y = y[np.isfinite(y)]
    assert finite_x.size == 4
    assert finite_y.size == 4
    assert np.isnan(x[:-1]).any()
    assert np.all((finite_x >= 0.0) & (finite_x <= 1.0))
    assert np.all((finite_y >= 0.0) & (finite_y <= 1.0))
    assert 0.98 < finite_y.max() < 1.0


def test_polar_moon_path_uses_dashed_line() -> None:
    assert VisibilityCoordinator.MOON_PATH_LINESTYLE == "--"
    assert VisibilityCoordinator.SUN_PATH_LINESTYLE == "--"
