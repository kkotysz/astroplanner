from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import astro_planner
from PySide6.QtCore import QDate, Signal
from PySide6.QtWidgets import QApplication, QWidget

from astroplanner.models import Site
from astroplanner.storage import AppStorage, SettingsAdapter
from astroplanner.ui.weather import WeatherDialog


class _DummyWeatherParent(QWidget):
    darkModeChanged = Signal(bool)

    def __init__(self, storage: AppStorage) -> None:
        super().__init__()
        self.app_storage = storage
        self.settings = SettingsAdapter(storage)
        self._theme_name = astro_planner.DEFAULT_UI_THEME
        self._dark_enabled = False


def test_weather_dialog_smoke_rebuild_without_network(tmp_path) -> None:
    app = QApplication.instance() or QApplication([])
    assert app is not None

    storage = AppStorage(tmp_path / "weather-settings")
    parent = _DummyWeatherParent(storage)
    dialog = WeatherDialog(parent)
    dialog._start_live_refresh = lambda *args, **kwargs: None

    dialog.set_context(
        site=Site(name="WRO", latitude=51.1, longitude=17.0, elevation=120.0),
        obs_name="WRO",
        date=QDate(2026, 4, 14),
        sun_alt_limit=-18.0,
        local_time_text="2026-04-14 22:15:00",
        utc_time_text="2026-04-14 20:15:00",
        sunrise_text="2026-04-15 05:42:00",
        sunset_text="2026-04-14 19:31:00",
        moonrise_text="2026-04-14 23:10:00",
        moonset_text="2026-04-15 06:20:00",
        moon_phase_pct=63,
        rebuild=True,
    )

    assert dialog.tabs.count() == 4
    assert dialog.conditions_tab_widget is not None
    assert dialog.obs_label.text() == "Obs: WRO"
