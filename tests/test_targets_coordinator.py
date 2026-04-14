from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

from PySide6.QtWidgets import QApplication, QWidget

from astroplanner.models import Target
from astroplanner.storage import AppStorage, SettingsAdapter
from astroplanner.targets_coordinator import TargetTableCoordinator
from astroplanner.ui.common import TargetTableView
from astroplanner.ui.targets import TargetTableModel


class _DummyPlanner(QWidget):
    def __init__(self, settings: SettingsAdapter) -> None:
        super().__init__()
        self.settings = settings
        self.targets = [Target(name="M31", ra=10.684, dec=41.269)]
        self.table_model = TargetTableModel(self.targets, site=None, parent=self)
        self.table_view = TargetTableView()
        self.color_blind_mode = False
        self._theme_name = "night"
        self._dark_enabled = False
        self._ensure_primary_target_selected_calls = 0
        self._autosave_calls = 0

    def _build_deterministic_observation_order(self):
        return ([{"row_index": 0}], {})

    def _delete_selected_targets(self) -> None:
        return

    def _schedule_plan_autosave(self) -> None:
        self._autosave_calls += 1

    def _open_table_context_menu(self, *_args) -> None:
        return

    def _on_table_double_click(self, *_args) -> None:
        return

    def _ensure_primary_target_selected(self) -> None:
        self._ensure_primary_target_selected_calls += 1


def _build_settings(tmp_path: Path) -> SettingsAdapter:
    return SettingsAdapter(AppStorage(tmp_path / "settings"))


def test_target_table_coordinator_smoke(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    assert app is not None

    planner = _DummyPlanner(_build_settings(tmp_path))
    coordinator = TargetTableCoordinator(planner)
    coordinator.bind()
    coordinator.apply_table_settings()
    coordinator.recompute_recommended_order_cache()
    coordinator.apply_column_preset("observation")
    coordinator.schedule_primary_target_selection()
    app.processEvents()

    assert planner.table_view.model() is planner.table_model
    assert coordinator.table_matches_observation_preset() is True
    assert planner.table_model.order_values == [1]
    assert planner._ensure_primary_target_selected_calls >= 1

    coordinator.clear_table_dynamic_cache()
    assert planner.table_model.order_values == []
