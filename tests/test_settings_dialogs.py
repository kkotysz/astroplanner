from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import astro_planner
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QApplication, QComboBox, QWidget

from astroplanner.storage import AppStorage, SettingsAdapter
from astroplanner.ui.settings import GeneralSettingsDialog, TableSettingsDialog


class _DummySettingsParent(QWidget):
    darkModeChanged = Signal(bool)

    def __init__(self, settings: SettingsAdapter) -> None:
        super().__init__()
        self.settings = settings
        self._theme_name = astro_planner.DEFAULT_UI_THEME
        self._dark_enabled = False
        self.color_blind_mode = False
        self._accent_overrides: dict[str, str] = {}
        self._table_apply_calls = 0
        self._general_apply_calls = 0
        self._run_plan_calls = 0

        site = astro_planner.Site(name="WRO", latitude=51.1, longitude=17.0, elevation=120.0)
        self.observatories = {"WRO": site}
        self.obs_combo = QComboBox(self)
        self.obs_combo.addItem("WRO")
        self.table_model = astro_planner.TargetTableModel(
            [astro_planner.Target(name="M31", ra=10.684, dec=41.269)]
        )

    def _apply_table_settings(self) -> None:
        self._table_apply_calls += 1

    def _apply_general_settings(self) -> None:
        self._general_apply_calls += 1

    def _run_plan(self) -> None:
        self._run_plan_calls += 1

    def _set_dark_mode_enabled(self, enabled: bool, *, persist: bool = True) -> None:
        self._dark_enabled = bool(enabled)
        if persist:
            self.settings.setValue("general/darkMode", bool(enabled))
        self.darkModeChanged.emit(bool(enabled))

    def _load_accent_secondary_override(self, theme_key: str) -> str:
        return self._accent_overrides.get(theme_key, "")

    def _save_accent_secondary_override(self, theme_key: str, color: str) -> None:
        self._accent_overrides[str(theme_key)] = str(color or "")


def _build_parent(tmp_path: Path) -> _DummySettingsParent:
    storage = AppStorage(tmp_path / "settings")
    settings = SettingsAdapter(storage)
    return _DummySettingsParent(settings)


def test_settings_dialogs_smoke_accept(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    assert app is not None

    parent = _build_parent(tmp_path)

    table_dialog = TableSettingsDialog(parent)
    table_dialog.accept()
    assert parent._table_apply_calls == 1

    general_dialog = GeneralSettingsDialog(parent, initial_tab="ai")
    general_dialog.accept()
    assert parent._general_apply_calls == 1
