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


def test_build_polar_visible_path_splits_horizon_and_wraps() -> None:
    theta, radius = VisibilityCoordinator.build_polar_visible_path(
        alt_series=[-1.0, 20.0, 25.0, -2.0, 30.0, 35.0],
        az_series=[0.0, 350.0, 10.0, 20.0, 355.0, 5.0],
    )

    assert theta.size == radius.size
    assert theta.size > 0
    assert np.isnan(theta).any()
    assert np.all(radius[np.isfinite(radius)] < 90.0)
