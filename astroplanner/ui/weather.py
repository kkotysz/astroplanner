from __future__ import annotations

import base64
import io
import logging
import math
import re
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Optional
from urllib.parse import quote

import matplotlib.dates as mdates
from astropy import units as u
from matplotlib.figure import Figure
from PySide6.QtCore import QDate, QSignalBlocker, QTimer, QUrl, Qt, Slot
from PySide6.QtGui import QColor, QDesktopServices, QFont, QFontMetrics, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QStackedLayout,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from astroplanner.i18n import current_language, localize_widget_tree, set_translated_text, set_translated_tooltip
from astroplanner.models import Site
from astroplanner.theme import DEFAULT_DARK_MODE, DEFAULT_UI_THEME, normalize_theme_key, resolve_theme_tokens
from astroplanner.ui.common import SkeletonShimmerWidget, _fit_dialog_to_screen
from astroplanner.ui.theme_utils import (
    _set_button_icon_kind,
    _set_button_variant,
    _theme_color_from_widget,
    _theme_tokens_from_widget,
)
from astroplanner.weather import WeatherLiveWorker

try:
    from PySide6.QtWebEngineWidgets import QWebEngineView

    _HAS_QTWEBENGINE = True
except Exception:  # pragma: no cover - optional runtime dependency
    QWebEngineView = None  # type: ignore[assignment]
    _HAS_QTWEBENGINE = False

try:
    import plotly
    import plotly.graph_objects as go
    from plotly.io import to_html as plotly_to_html
    from plotly.subplots import make_subplots as plotly_make_subplots

    _plotly_base_dir = Path(plotly.__file__).resolve().parent / "package_data"
    _PLOTLY_JS_BASE_DIR = str(_plotly_base_dir) if (_plotly_base_dir / "plotly.min.js").exists() else ""
    _HAS_PLOTLY = True
except Exception:  # pragma: no cover - optional runtime dependency
    plotly_to_html = None  # type: ignore[assignment]
    plotly_make_subplots = None  # type: ignore[assignment]
    go = None  # type: ignore[assignment]
    _PLOTLY_JS_BASE_DIR = ""
    _HAS_PLOTLY = False

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

    _HAS_MPL_CANVAS = True
except Exception:  # pragma: no cover - optional runtime dependency
    FigureCanvas = None  # type: ignore[assignment]
    _HAS_MPL_CANVAS = False


logger = logging.getLogger(__name__)


def _safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: object) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _configure_tab_widget(tab_widget: QTabWidget, *, document_mode: bool = True) -> QTabWidget:
    """Normalize tab widgets so native tab-bar chrome doesn't leak through custom themes."""
    tab_widget.setDocumentMode(document_mode)
    try:
        tab_widget.setUsesScrollButtons(True)
    except Exception:
        pass
    try:
        tab_widget.setElideMode(Qt.TextElideMode.ElideNone)
    except Exception:
        pass
    bar = tab_widget.tabBar()
    bar.setExpanding(False)
    bar.setElideMode(Qt.TextElideMode.ElideNone)
    if hasattr(bar, "setUsesScrollButtons"):
        try:
            bar.setUsesScrollButtons(True)
        except Exception:
            pass
    tab_widget.tabBar().setDrawBase(False)
    return tab_widget


_DISPLAY_FONT_PATH = Path(__file__).resolve().parents[2] / "assets" / "fonts" / "Rajdhani-SemiBold.ttf"
_DISPLAY_FONT_CSS_CACHE: Optional[str] = None
_DISPLAY_WEB_FONT_FAMILY = "Rajdhani Web"
_DISPLAY_FONT_QT_FAMILY: Optional[str] = None


def _preferred_display_font_family() -> str:
    return str(_DISPLAY_FONT_QT_FAMILY or "Rajdhani").strip() or "Rajdhani"


def _split_font_families(value: object) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    families: list[str] = []
    for chunk in text.split(","):
        name = str(chunk or "").strip().strip('"').strip("'").strip()
        if name:
            families.append(name)
    return families


def _platform_ui_font_candidates() -> list[str]:
    if sys.platform == "darwin":
        return [".AppleSystemUIFont", "SF Pro Text", "Avenir Next", "Helvetica Neue", "Arial"]
    if sys.platform.startswith("win"):
        return ["Segoe UI", "Arial"]
    return ["Noto Sans", "DejaVu Sans", "Liberation Sans", "Arial"]


def _embedded_display_font_css() -> str:
    global _DISPLAY_FONT_CSS_CACHE
    if _DISPLAY_FONT_CSS_CACHE is not None:
        return _DISPLAY_FONT_CSS_CACHE
    if not _DISPLAY_FONT_PATH.exists():
        _DISPLAY_FONT_CSS_CACHE = ""
        return _DISPLAY_FONT_CSS_CACHE
    try:
        encoded = base64.b64encode(_DISPLAY_FONT_PATH.read_bytes()).decode("ascii")
    except Exception:
        _DISPLAY_FONT_CSS_CACHE = ""
        return _DISPLAY_FONT_CSS_CACHE
    _DISPLAY_FONT_CSS_CACHE = (
        "@font-face{" 
        f"font-family:'{_DISPLAY_WEB_FONT_FAMILY}';"
        f"src:url(data:font/ttf;base64,{encoded}) format('truetype');"
        "font-style:normal;"
        "font-weight:600;"
        "font-display:swap;"
        "}"
    )
    return _DISPLAY_FONT_CSS_CACHE


def _plot_font_css_stack(theme_tokens: Optional[dict[str, str]] = None) -> str:
    fallback_families: list[str] = [_preferred_display_font_family(), "Rajdhani"]
    if theme_tokens:
        for body_font in _split_font_families(theme_tokens.get("font_family", "")):
            if body_font and body_font not in fallback_families:
                fallback_families.append(body_font)
    fallback_families.extend(_platform_ui_font_candidates())
    fallback_families.extend(["Arial", "Helvetica"])
    deduped: list[str] = []
    for family in fallback_families:
        normalized = str(family or "").strip()
        if normalized and normalized not in deduped and normalized != "Sans Serif":
            deduped.append(normalized)
    fallback = ", ".join(f'"{family}"' for family in deduped) or '"Arial"'
    if _DISPLAY_FONT_PATH.exists():
        return f'"{_DISPLAY_WEB_FONT_FAMILY}", {fallback}'
    return fallback


def _mix_qcolors_for_theme(first: QColor, second: QColor, first_ratio: float) -> QColor:
    ratio = max(0.0, min(1.0, float(first_ratio)))
    other_ratio = 1.0 - ratio
    return QColor(
        int(round(first.red() * ratio + second.red() * other_ratio)),
        int(round(first.green() * ratio + second.green() * other_ratio)),
        int(round(first.blue() * ratio + second.blue() * other_ratio)),
    )


def _qcolor_rgba_css_for_theme(color: QColor, alpha: float) -> str:
    use_color = QColor(color)
    if not use_color.isValid():
        use_color = QColor("#59f3ff")
    return f"rgba({use_color.red()}, {use_color.green()}, {use_color.blue()}, {max(0.0, min(1.0, float(alpha))):.3f})"


def _qcolor_rgba_mpl_for_theme(color: QColor, alpha: float) -> tuple[float, float, float, float]:
    use_color = QColor(color)
    if not use_color.isValid():
        use_color = QColor("#59f3ff")
    return (
        float(use_color.redF()),
        float(use_color.greenF()),
        float(use_color.blueF()),
        max(0.0, min(1.0, float(alpha))),
    )


def _softened_plot_grid_qcolor_from_tokens(theme_tokens: Optional[dict[str, str]]) -> QColor:
    tokens = theme_tokens if isinstance(theme_tokens, dict) else {}
    base = QColor(str(tokens.get("plot_grid", "#2f4666")))
    panel = QColor(str(tokens.get("plot_panel_bg", "#162334")))
    if not base.isValid():
        base = QColor("#2f4666")
    if not panel.isValid():
        panel = QColor("#162334")
    return _mix_qcolors_for_theme(base, panel, 0.34)


def _softened_plot_grid_css_from_tokens(theme_tokens: Optional[dict[str, str]], *, alpha: float = 1.0) -> str:
    return _qcolor_rgba_css_for_theme(_softened_plot_grid_qcolor_from_tokens(theme_tokens), alpha)


def _softened_plot_grid_rgba_from_tokens(theme_tokens: Optional[dict[str, str]], *, alpha: float = 1.0) -> tuple[float, float, float, float]:
    return _qcolor_rgba_mpl_for_theme(_softened_plot_grid_qcolor_from_tokens(theme_tokens), alpha)


class WeatherDialog(QDialog):
    """Weather workspace with provider-based live data, cloud analysis and interactive meteograms."""

    _AUTO_REFRESH_MIN_INTERVAL_S = 120.0
    _MONTH_CHOICES = [(idx + 1, name) for idx, name in enumerate(WeatherLiveWorker.MONTH_NAMES)]
    _FORECAST_SOURCE_CHOICES = [
        ("open_meteo", "Open-Meteo"),
        ("meteoblue", "meteoblue"),
        ("windy", "Windy"),
        ("meteo_icm", "Meteo ICM"),
    ]
    _CONDITION_SOURCE_CHOICES = [
        ("average", "Average (real measurements)"),
        ("metar", "Nearest METAR"),
        ("custom", "Custom URL"),
    ]
    _METEOGRAM_SOURCE_CHOICES = [
        ("open_meteo", "Open-Meteo"),
        ("meteoblue", "meteoblue"),
        ("windy", "Windy"),
        ("meteo_icm", "Meteo ICM"),
        ("metar", "Nearest METAR"),
        ("custom", "Custom URL"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName(self.__class__.__name__)
        self.setWindowTitle("Weather")
        self.setModal(False)
        self._theme_tokens = dict(_theme_tokens_from_widget(self))
        self._theme_name = normalize_theme_key(
            str(getattr(parent, "_theme_name", DEFAULT_UI_THEME) if parent is not None else DEFAULT_UI_THEME)
        )
        self._dark_enabled = bool(getattr(parent, "_dark_enabled", False))
        self._storage = getattr(parent, "app_storage", None)

        self._site: Optional[Site] = None
        self._obs_name = "-"
        self._date = QDate.currentDate()
        self._sun_alt_limit = -10.0
        self._local_time_text = "-"
        self._utc_time_text = "-"
        self._sunrise_text = "-"
        self._sunset_text = "-"
        self._moonrise_text = "-"
        self._moonset_text = "-"
        self._moon_phase_pct = 0

        self._weather_default_source = "average"
        self._weather_auto_refresh_s = float(self._AUTO_REFRESH_MIN_INTERVAL_S)
        self._weather_custom_url = ""
        self._weather_wunderground_url = ""
        self._weather_weathercloud_url = ""
        self._weather_local_links_raw = ""
        self._weather_cloud_source = "earthenv"
        self._weather_cloud_month_mode = "session_month"

        self._live_worker: Optional[WeatherLiveWorker] = None
        self._live_payload: dict[str, object] = {}
        self._last_live_context_key = ""
        self._last_live_fetch_perf = 0.0
        self.conditions_tab_widget: Optional[QWidget] = None

        self._conditions_raw_pixmap = QPixmap()
        self._meteogram_raw_pixmap = QPixmap()
        self._cloud_map_raw_pixmap = QPixmap()
        self._satellite_raw_pixmap = QPixmap()

        self._resize_debounce = QTimer(self)
        self._resize_debounce.setSingleShot(True)
        self._resize_debounce.timeout.connect(self._rescale_preview_pixmaps)

        self._conditions_poll_timer = QTimer(self)
        self._conditions_poll_timer.setSingleShot(False)
        self._conditions_poll_timer.timeout.connect(self._on_conditions_poll_timeout)
        self._weather_plot_stacks: dict[str, QStackedLayout] = {}
        self._weather_plot_placeholders: dict[str, QWidget] = {}
        self._weather_plot_hosts: dict[str, QWidget] = {}
        self._weather_plot_loaded: dict[str, bool] = {}
        self._weather_image_stacks: dict[str, QStackedLayout] = {}
        self._weather_image_placeholders: dict[str, QWidget] = {}
        self._weather_image_hosts: dict[str, QWidget] = {}
        self._weather_image_labels: dict[str, QLabel] = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        context_row = QWidget(self)
        context_l = QHBoxLayout(context_row)
        context_l.setContentsMargins(0, 0, 0, 0)
        context_l.setSpacing(8)
        self.obs_label = QLabel("Obs: -", context_row)
        self.coords_label = QLabel("Lat/Lon: -", context_row)
        self.sun_limit_label = QLabel("Sun ≤ -", context_row)
        for lbl in (self.obs_label, self.coords_label, self.sun_limit_label):
            lbl.setObjectName("SectionHint")
            context_l.addWidget(lbl)
        context_l.addSpacing(10)
        src_lbl = QLabel("Source:", context_row)
        src_lbl.setObjectName("SectionHint")
        context_l.addWidget(src_lbl, 0)

        self.live_source_combo = QComboBox(context_row)
        for key, label in self._CONDITION_SOURCE_CHOICES:
            self.live_source_combo.addItem(label, key)
        self.live_source_combo.setMinimumWidth(220)
        self.live_source_combo.setMaximumWidth(280)
        context_l.addWidget(self.live_source_combo, 0)

        self.live_refresh_btn = QPushButton("Refresh", context_row)
        self.live_refresh_btn.setMinimumHeight(30)
        _set_button_variant(self.live_refresh_btn, "secondary")
        _set_button_icon_kind(self.live_refresh_btn, "refresh")
        context_l.addWidget(self.live_refresh_btn, 0)

        self.live_progress = QProgressBar(context_row)
        self.live_progress.setTextVisible(False)
        self.live_progress.setMinimumWidth(72)
        self.live_progress.setMaximumWidth(120)
        self.live_progress.hide()
        context_l.addWidget(self.live_progress, 0)

        self.live_status_label = QLabel("Idle", context_row)
        self.live_status_label.setObjectName("SectionHint")
        self.live_status_label.setWordWrap(False)
        self.live_status_label.setMaximumWidth(260)
        context_l.addWidget(self.live_status_label, 0)
        context_l.addStretch(1)
        root.addWidget(context_row)

        summary_row = QWidget(self)
        summary_l = QHBoxLayout(summary_row)
        summary_l.setContentsMargins(0, 0, 0, 0)
        summary_l.setSpacing(6)

        self.live_temp_label = QLabel("T -", summary_row)
        self.live_wind_label = QLabel("W -", summary_row)
        self.live_cloud_label = QLabel("C -", summary_row)
        self.live_rh_label = QLabel("RH -", summary_row)
        self.live_pressure_label = QLabel("P -", summary_row)
        self.date_chip_label = QLabel("Date -", summary_row)
        self.local_time_label = QLabel("L -", summary_row)
        self.utc_time_label = QLabel("U -", summary_row)
        self.sunrise_info_label = QLabel("Sun↑ -", summary_row)
        self.sunset_info_label = QLabel("Sun↓ -", summary_row)
        self.moonrise_info_label = QLabel("Moon↑ -", summary_row)
        self.moonset_info_label = QLabel("Moon↓ -", summary_row)
        self.moon_phase_info_bar = QProgressBar(summary_row)
        self.moon_phase_info_bar.setRange(0, 100)
        self.moon_phase_info_bar.setValue(0)
        self.moon_phase_info_bar.setTextVisible(True)
        self.moon_phase_info_bar.setAlignment(Qt.AlignCenter)
        self.moon_phase_info_bar.setFormat("Phase %p%%")
        self.moon_phase_info_bar.setProperty("weather_chip", True)
        self.moon_phase_info_bar.setProperty("weather_phase_chip", True)
        self.moon_phase_info_bar.setProperty("weather_chip_role", "lunar")
        self.moon_phase_info_bar.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        summary_chips = (
            (self.live_temp_label, "weather"),
            (self.live_wind_label, "weather"),
            (self.live_cloud_label, "weather"),
            (self.live_rh_label, "weather"),
            (self.live_pressure_label, "weather"),
            (self.date_chip_label, "context"),
            (self.local_time_label, "clock"),
            (self.utc_time_label, "clock"),
            (self.sunrise_info_label, "solar"),
            (self.sunset_info_label, "solar"),
            (self.moonrise_info_label, "lunar"),
            (self.moonset_info_label, "lunar"),
        )
        for lbl, role in summary_chips:
            lbl.setObjectName("SectionHint")
            lbl.setProperty("weather_chip", True)
            lbl.setProperty("weather_chip_role", role)
            lbl.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
            lbl.setTextFormat(Qt.PlainText)
            summary_l.addWidget(lbl, 0)
        self.live_temp_label.setProperty("weather_chip_series", "temp")
        self.live_wind_label.setProperty("weather_chip_series", "wind")
        self.live_cloud_label.setProperty("weather_chip_series", "cloud")
        self.live_rh_label.setProperty("weather_chip_series", "humidity")
        self.live_pressure_label.setProperty("weather_chip_series", "pressure")
        summary_l.addWidget(self.moon_phase_info_bar, 0)
        summary_l.addStretch(1)
        root.addWidget(summary_row)

        self.tabs = _configure_tab_widget(QTabWidget(self))
        self.tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        root.addWidget(self.tabs, 1)
        self.tabs.currentChanged.connect(self._on_tab_changed)

        self.live_source_combo.currentIndexChanged.connect(self._update_live_views)
        self.live_source_combo.currentIndexChanged.connect(lambda _=0: self._sync_conditions_source_combo())
        self.live_refresh_btn.clicked.connect(lambda: self._start_live_refresh(force=True))
        _fit_dialog_to_screen(
            self,
            preferred_width=1380,
            preferred_height=900,
            min_width=1080,
            min_height=720,
        )
        self._update_summary_chip_widths()
        localize_widget_tree(self, current_language())

    def _sync_theme_state_from_parent(self) -> None:
        parent = self.parent()
        self._theme_tokens = dict(_theme_tokens_from_widget(self))
        self._theme_name = normalize_theme_key(
            str(getattr(parent, "_theme_name", getattr(self, "_theme_name", DEFAULT_UI_THEME)) or DEFAULT_UI_THEME)
        )
        self._dark_enabled = bool(getattr(parent, "_dark_enabled", getattr(self, "_dark_enabled", False)))
        self._refresh_theme_surfaces()

    def _refresh_theme_surfaces(self) -> None:
        plot_bg = str(getattr(self, "_theme_tokens", {}).get("plot_bg", "#121b29"))
        for host in self._weather_plot_hosts.values():
            if isinstance(host, QWidget):
                host.setStyleSheet(f"background:{plot_bg};")
        for host in self._weather_image_hosts.values():
            if isinstance(host, QWidget):
                host.setStyleSheet(f"background:{plot_bg};")
        for placeholder in self._weather_plot_placeholders.values():
            if isinstance(placeholder, QWidget):
                placeholder.setStyleSheet(f"background:{plot_bg};")
        for placeholder in self._weather_image_placeholders.values():
            if isinstance(placeholder, QWidget):
                placeholder.setStyleSheet(f"background:{plot_bg};")

    def closeEvent(self, event):
        self._conditions_poll_timer.stop()
        self._stop_live_worker()
        super().closeEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._resize_debounce.start(120)

    def set_context(
        self,
        *,
        site: Optional[Site],
        obs_name: str,
        date: QDate,
        sun_alt_limit: float,
        local_time_text: str,
        utc_time_text: str,
        sunrise_text: str,
        sunset_text: str,
        moonrise_text: str,
        moonset_text: str,
        moon_phase_pct: int,
        rebuild: bool = True,
    ) -> None:
        self._sync_theme_state_from_parent()
        self._site = site
        self._obs_name = str(obs_name or "-")
        self._date = date if isinstance(date, QDate) and date.isValid() else QDate.currentDate()
        self._sun_alt_limit = float(sun_alt_limit)
        self._local_time_text = str(local_time_text or "-")
        self._utc_time_text = str(utc_time_text or "-")
        self._sunrise_text = str(sunrise_text or "-")
        self._sunset_text = str(sunset_text or "-")
        self._moonrise_text = str(moonrise_text or "-")
        self._moonset_text = str(moonset_text or "-")
        self._moon_phase_pct = max(0, min(100, int(moon_phase_pct)))
        if rebuild:
            self._reload_weather_settings()
            self._rebuild()
            return
        self._apply_summary_labels()

    def _reload_weather_settings(self) -> None:
        parent = self.parent()
        self._sync_theme_state_from_parent()
        settings = getattr(parent, "settings", None)
        if settings is None:
            return
        self._weather_default_source = str(
            settings.value("weather/defaultConditionsSource", "average", type=str) or "average"
        ).strip() or "average"
        self._weather_auto_refresh_s = max(
            30.0,
            min(
                1800.0,
                float(
                    settings.value(
                        "weather/autoRefreshSec",
                        self._AUTO_REFRESH_MIN_INTERVAL_S,
                        type=int,
                    )
                ),
            ),
        )
        site_custom_url = (
            str(getattr(self._site, "custom_conditions_url", "") or "").strip()
            if isinstance(self._site, Site)
            else ""
        )
        self._weather_custom_url = site_custom_url
        self._weather_wunderground_url = str(settings.value("weather/wundergroundUrl", "", type=str) or "").strip()
        self._weather_weathercloud_url = str(settings.value("weather/weathercloudUrl", "", type=str) or "").strip()
        self._weather_local_links_raw = str(settings.value("weather/localConditionsLinks", "", type=str) or "")
        self._weather_cloud_source = str(settings.value("weather/cloudMapSource", "earthenv", type=str) or "earthenv").strip()
        self._weather_cloud_month_mode = str(
            settings.value("weather/cloudMapMonthMode", "session_month", type=str) or "session_month"
        ).strip()
        self._conditions_poll_timer.setInterval(int(max(30.0, self._weather_auto_refresh_s) * 1000))
        idx = self.live_source_combo.findData(self._weather_default_source)
        if idx >= 0:
            blocker = QSignalBlocker(self.live_source_combo)
            self.live_source_combo.setCurrentIndex(idx)
            del blocker

    def _site_coordinates(self) -> tuple[float, float, float]:
        if isinstance(self._site, Site):
            return float(self._site.latitude), float(self._site.longitude), float(self._site.elevation)
        return 0.0, 0.0, 0.0

    def _default_cloud_month(self) -> int:
        mode = str(self._weather_cloud_month_mode or "session_month").strip().lower()
        if mode == "current_month":
            return int(QDate.currentDate().month())
        return int(self._date.month())

    def _selected_cloud_month(self) -> int:
        if hasattr(self, "cloud_month_combo"):
            month = _safe_int(self.cloud_month_combo.currentData())
            if month is not None:
                return max(1, min(12, int(month)))
        return self._default_cloud_month()

    @staticmethod
    def _open_url(url: str, parent: Optional[QWidget] = None) -> None:
        qurl = QUrl(str(url))
        if not qurl.isValid():
            QMessageBox.warning(parent, "Weather Link", f"Invalid URL: {url}")
            return
        if not QDesktopServices.openUrl(qurl):
            QMessageBox.warning(parent, "Weather Link", f"Unable to open URL:\n{url}")

    def _open_urls(self, urls: list[str]) -> None:
        for url in urls:
            self._open_url(url, self)

    @staticmethod
    def _normalize_optional_station_url(value: str, *, base_url: str = "") -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        if text.startswith("http://") or text.startswith("https://"):
            return text
        if base_url:
            return f"{base_url.rstrip('/')}/{quote(text)}"
        return text

    @staticmethod
    def _label_from_url(url: str) -> str:
        text = str(url or "").strip()
        if not text:
            return "Local Source"
        clean = text.replace("https://", "").replace("http://", "")
        domain = clean.split("/", 1)[0]
        domain = domain.replace("www.", "").strip()
        if not domain:
            return "Local Source"
        return domain[:40]

    def _parse_local_conditions_links(self) -> list[tuple[str, str, str]]:
        entries: list[tuple[str, str, str]] = []
        lines = str(self._weather_local_links_raw or "").splitlines()
        for raw in lines:
            line = raw.strip()
            if not line:
                continue
            label = ""
            url = ""
            if "|" in line:
                left, right = line.split("|", 1)
                label = left.strip()
                url = right.strip()
            else:
                url = line
            if not url:
                continue
            if not (url.startswith("http://") or url.startswith("https://")):
                continue
            if not label:
                label = self._label_from_url(url)
            entries.append((label, url, "Custom local weather source."))
        return entries

    @staticmethod
    def _fmt_value(value: Optional[float], unit: str, precision: int = 1) -> str:
        if value is None or not math.isfinite(float(value)):
            return "N/A"
        return f"{float(value):.{precision}f} {unit}".strip()

    @staticmethod
    def _short_provider_note(value: object, max_len: int = 66) -> str:
        text = str(value or "").replace("\n", " ").strip()
        if len(text) <= max_len:
            return text
        return text[: max_len - 1].rstrip() + "…"

    def _source_urls(self) -> dict[str, list[tuple[str, str, str]]]:
        lat, lon, elev = self._site_coordinates()
        lat_txt = f"{lat:.5f}"
        lon_txt = f"{lon:.5f}"
        elev_txt = f"{elev:.0f}"
        windy_center = f"{lat:.5f},{lon:.5f},8"
        local_conditions_entries: list[tuple[str, str, str]] = []
        wu_url = self._normalize_optional_station_url(
            self._weather_wunderground_url,
            base_url="https://www.wunderground.com/dashboard/pws",
        )
        if wu_url:
            local_conditions_entries.append(
                ("Wunderground PWS", wu_url, "Public local station page (configured in Settings).")
            )
        wc_url = self._normalize_optional_station_url(
            self._weather_weathercloud_url,
            base_url="https://app.weathercloud.net",
        )
        if wc_url:
            local_conditions_entries.append(
                ("WeatherCloud", wc_url, "Public local station page (configured in Settings).")
            )
        local_conditions_entries.extend(self._parse_local_conditions_links())
        return {
            "forecast": [
                ("Meteo ICM", "https://www.meteo.pl/", "Forecast portal and meteorogram ecosystem."),
                (
                    "meteoblue",
                    f"https://www.meteoblue.com/en/weather/week?lat={lat_txt}&lon={lon_txt}&asl={elev_txt}",
                    "Model week forecast for current coordinates.",
                ),
                (
                    "Windy",
                    f"https://www.windy.com/{lat_txt}/{lon_txt}?{windy_center}",
                    "Interactive multi-model forecast map.",
                ),
                ("wetterzentrale", "https://www.wetterzentrale.de/en/topkarten.php?model=gfs", "Synoptic model charts."),
            ],
            "conditions": local_conditions_entries + [
                ("Open-Meteo", f"https://open-meteo.com/en/docs#latitude={lat_txt}&longitude={lon_txt}", "Public no-key API docs."),
                ("AviationWeather METAR", "https://aviationweather.gov/data/api/", "Nearest METAR station feed."),
                ("WeatherCloud Portal", "https://weathercloud.net/", "Station platform."),
                ("Wunderground Portal", "https://www.wunderground.com/weatherstation/overview", "Station platform."),
                ("Windy", f"https://www.windy.com/-Weather-radar-radar?radar,{lat_txt},{lon_txt},7", "Nowcast overlays."),
            ],
            "cloud": [
                ("EarthEnv", "https://www.earthenv.org/cloud", "Monthly cloud frequency dataset."),
                (
                    "ECMWF OpenCharts",
                    "https://charts.ecmwf.int/products/medium-clouds",
                    "Operational cloud charts.",
                ),
                ("DWD RCC-CM", "https://www.dwd.de/EN/ourservices/rcccm/int/rcccm_int_cfc.html", "Cloud climatology service."),
                ("CHELSA", "https://www.chelsa-climate.org/datasets/chelsa_monthly", "Monthly climate datasets."),
            ],
            "satellite": [
                ("Windy satellite", f"https://www.windy.com/-Satellite-satellite?satellite,{lat_txt},{lon_txt},7", "Satellite layer."),
                ("NASA GIBS WMS", "https://nasa-gibs.github.io/gibs-api-docs/gis-usage/", "Satellite map service used for centered site cutouts."),
                ("MET geosatellite", "https://api.met.no/weatherapi/geosatellite/1.4/documentation", "Operational geostationary imagery API."),
                ("NASA Worldview", "https://worldview.earthdata.nasa.gov/", "Satellite layers and browse UI."),
                ("wetterzentrale", "https://www.wetterzentrale.de/en/reanalysis.php?map=1&model=sat&var=44", "Alternative reference source."),
                ("meteoblue maps", f"https://www.meteoblue.com/en/weather/maps/index?lat={lat_txt}&lon={lon_txt}&asl={elev_txt}&map=satellite", "Satellite map view."),
            ],
        }

    def _append_links(
        self,
        layout: QVBoxLayout,
        entries: list[tuple[str, str, str]],
        *,
        max_cols: int = 5,
        min_button_w: int = 96,
        max_button_w: int = 156,
    ) -> None:
        if not entries:
            return
        urls: list[str] = []
        box = QFrame(self)
        box_l = QGridLayout(box)
        box_l.setContentsMargins(0, 0, 0, 0)
        box_l.setHorizontalSpacing(6)
        box_l.setVerticalSpacing(6)
        cols = max(1, int(max_cols))
        for idx, (source_name, url, desc) in enumerate(entries):
            urls.append(url)
            btn = QPushButton(source_name, box)
            btn.setProperty("weather_link", True)
            btn.setToolTip(f"{desc}\n{url}")
            btn.setMinimumHeight(24)
            btn.setMaximumHeight(24)
            btn.setMinimumWidth(max(70, int(min_button_w)))
            btn.setMaximumWidth(max(90, int(max_button_w)))
            _set_button_variant(btn, "ghost")
            _set_button_icon_kind(btn, "link", 13)
            btn.clicked.connect(lambda _checked=False, u=url: self._open_url(u, self))
            row = idx // cols
            col = idx % cols
            box_l.addWidget(btn, row, col)
        layout.addWidget(box, 0)
        open_all_btn = QPushButton("Open all", self)
        open_all_btn.setProperty("weather_link", True)
        open_all_btn.setMinimumHeight(24)
        open_all_btn.setMaximumHeight(24)
        open_all_btn.setMinimumWidth(90)
        _set_button_variant(open_all_btn, "secondary")
        _set_button_icon_kind(open_all_btn, "open", 13)
        open_all_btn.clicked.connect(lambda: self._open_urls(urls))
        layout.addWidget(open_all_btn, 0, Qt.AlignLeft)

    def _append_links_inline(
        self,
        layout: QHBoxLayout,
        entries: list[tuple[str, str, str]],
        *,
        min_button_w: int = 88,
        max_button_w: int = 132,
        add_open_all: bool = True,
    ) -> None:
        if not entries:
            return
        urls: list[str] = []
        for source_name, url, desc in entries:
            urls.append(url)
            btn = QPushButton(source_name, self)
            btn.setProperty("weather_link", True)
            btn.setToolTip(f"{desc}\n{url}")
            btn.setMinimumHeight(24)
            btn.setMaximumHeight(24)
            btn.setMinimumWidth(max(70, int(min_button_w)))
            btn.setMaximumWidth(max(90, int(max_button_w)))
            _set_button_variant(btn, "ghost")
            _set_button_icon_kind(btn, "link", 13)
            btn.clicked.connect(lambda _checked=False, u=url: self._open_url(u, self))
            layout.addWidget(btn, 0)
        if add_open_all:
            open_all_btn = QPushButton("Open all", self)
            open_all_btn.setProperty("weather_link", True)
            open_all_btn.setMinimumHeight(24)
            open_all_btn.setMaximumHeight(24)
            open_all_btn.setMinimumWidth(90)
            _set_button_variant(open_all_btn, "secondary")
            _set_button_icon_kind(open_all_btn, "open", 13)
            open_all_btn.clicked.connect(lambda: self._open_urls(urls))
            layout.addWidget(open_all_btn, 0)

    def _build_weather_plot_placeholder(self, title: str, message: str) -> QWidget:
        widget = QWidget(self)
        widget.setAttribute(Qt.WA_StyledBackground, True)
        widget.setStyleSheet(f"background:{_theme_color_from_widget(self, 'plot_bg', '#121b29')};")
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)
        layout.addStretch(1)
        card = QFrame(widget)
        card.setObjectName("VisibilityLoadingCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(24, 22, 24, 22)
        card_layout.setSpacing(12)
        title_label = QLabel(title, card)
        title_label.setObjectName("SectionTitle")
        title_label.setAlignment(Qt.AlignCenter)
        skeleton = SkeletonShimmerWidget("plot", card)
        skeleton.setMinimumHeight(230)
        hint_label = QLabel(message, card)
        hint_label.setObjectName("SectionHint")
        hint_label.setAlignment(Qt.AlignCenter)
        hint_label.setWordWrap(True)
        hint_label.setMinimumWidth(320)
        hint_label.setMaximumWidth(460)
        hint_label.setMinimumHeight(42)
        card_layout.addWidget(title_label, 0, Qt.AlignHCenter)
        card_layout.addWidget(skeleton, 1)
        card_layout.addWidget(hint_label, 0, Qt.AlignHCenter)
        layout.addWidget(card, 0, Qt.AlignCenter)
        layout.addStretch(1)
        widget._loading_hint_label = hint_label  # type: ignore[attr-defined]
        widget._loading_skeleton = skeleton  # type: ignore[attr-defined]
        return widget

    def _create_weather_plot_stack(
        self,
        parent: QWidget,
        *,
        kind: str,
        title: str,
        min_height: int,
    ) -> tuple[QWidget, "QWebEngineView"]:
        host = QWidget(parent)
        host.setAttribute(Qt.WA_StyledBackground, True)
        host.setStyleSheet(f"background:{_theme_color_from_widget(self, 'plot_bg', '#121b29')};")
        host.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        host.setMinimumHeight(min_height)
        stack = QStackedLayout(host)
        stack.setContentsMargins(0, 0, 0, 0)
        placeholder = self._build_weather_plot_placeholder(title, "Loading weather plot…")
        web_view = QWebEngineView(host)
        web_view.setMinimumHeight(min_height)
        web_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        web_view.setProperty("weather_plot_kind", kind)
        web_view.setStyleSheet(f"background:{_theme_color_from_widget(self, 'plot_bg', '#121b29')};")
        web_view.loadFinished.connect(lambda ok, k=kind: self._on_weather_plot_load_finished(k, ok))
        stack.addWidget(placeholder)
        stack.addWidget(web_view)
        stack.setCurrentWidget(placeholder)
        self._weather_plot_stacks[kind] = stack
        self._weather_plot_placeholders[kind] = placeholder
        self._weather_plot_hosts[kind] = host
        self._weather_plot_loaded[kind] = False
        return host, web_view

    def _create_weather_image_stack(
        self,
        parent: QWidget,
        *,
        kind: str,
        title: str,
        min_height: int,
    ) -> tuple[QWidget, QLabel]:
        host = QWidget(parent)
        host.setAttribute(Qt.WA_StyledBackground, True)
        host.setStyleSheet(f"background:{_theme_color_from_widget(self, 'plot_bg', '#121b29')};")
        host.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        host.setMinimumHeight(min_height)
        stack = QStackedLayout(host)
        stack.setContentsMargins(0, 0, 0, 0)
        placeholder = self._build_weather_plot_placeholder(title, "Fetching preview…")
        skeleton = getattr(placeholder, "_loading_skeleton", None)
        if isinstance(skeleton, SkeletonShimmerWidget):
            skeleton._variant = "image"
            skeleton.update()
        image_label = QLabel("", host)
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setWordWrap(True)
        image_label.setMinimumHeight(min_height)
        image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        image_label.setObjectName("CutoutImage")
        stack.addWidget(placeholder)
        stack.addWidget(image_label)
        stack.setCurrentWidget(placeholder)
        self._weather_image_stacks[kind] = stack
        self._weather_image_placeholders[kind] = placeholder
        self._weather_image_hosts[kind] = host
        self._weather_image_labels[kind] = image_label
        return host, image_label

    def _set_weather_image_loading(self, kind: str, message: str, *, visible: bool = True) -> None:
        stack = self._weather_image_stacks.get(kind)
        placeholder = self._weather_image_placeholders.get(kind)
        image_label = self._weather_image_labels.get(kind)
        if stack is None or placeholder is None or image_label is None:
            return
        hint_label = getattr(placeholder, "_loading_hint_label", None)
        if isinstance(hint_label, QLabel):
            hint_label.setText(message)
        if visible:
            stack.setCurrentWidget(placeholder)
        else:
            stack.setCurrentWidget(image_label)

    def _set_weather_plot_loading(self, kind: str, message: str, *, visible: bool = True) -> None:
        stack = self._weather_plot_stacks.get(kind)
        placeholder = self._weather_plot_placeholders.get(kind)
        if stack is None or placeholder is None:
            return
        hint_label = getattr(placeholder, "_loading_hint_label", None)
        if isinstance(hint_label, QLabel):
            hint_label.setText(message)
        if visible:
            stack.setCurrentWidget(placeholder)
        else:
            widget = stack.widget(1)
            if widget is not None:
                stack.setCurrentWidget(widget)

    @Slot(bool)
    def _on_weather_plot_load_finished(self, kind: str, ok: bool) -> None:
        self._weather_plot_loaded[kind] = bool(ok)
        if ok:
            self._set_weather_plot_loading(kind, "", visible=False)
            return
        self._set_weather_plot_loading(kind, "Unable to render interactive chart.", visible=True)

    def _build_forecast_tab(self, entries: list[tuple[str, str, str]]) -> QWidget:
        page = QWidget(self.tabs)
        page_l = QVBoxLayout(page)
        page_l.setContentsMargins(10, 10, 10, 10)
        page_l.setSpacing(6)
        intro = QLabel(
            "Forecast: model-driven look ahead for the next hours/nights (different from live current conditions).",
            page,
        )
        intro.setWordWrap(True)
        intro.setObjectName("SectionHint")
        page_l.addWidget(intro)
        self.forecast_summary_label = QLabel("Forecast providers: -", page)
        self.forecast_summary_label.setObjectName("SectionHint")
        self.forecast_summary_label.setWordWrap(True)
        page_l.addWidget(self.forecast_summary_label)

        src_row = QWidget(page)
        src_row_l = QHBoxLayout(src_row)
        src_row_l.setContentsMargins(0, 0, 0, 0)
        src_row_l.setSpacing(6)
        src_lbl = QLabel("Forecast source:", src_row)
        src_lbl.setObjectName("SectionHint")
        src_row_l.addWidget(src_lbl, 0)
        self.forecast_source_combo = QComboBox(src_row)
        for key, label in self._FORECAST_SOURCE_CHOICES:
            self.forecast_source_combo.addItem(label, key)
        src_row_l.addWidget(self.forecast_source_combo, 0)
        src_row_l.addStretch(1)
        page_l.addWidget(src_row)

        if _HAS_QTWEBENGINE and _HAS_PLOTLY and QWebEngineView is not None:
            self.forecast_web = QWebEngineView(page)
            self.forecast_web.setMinimumHeight(280)
            self.forecast_web.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            page_l.addWidget(self.forecast_web, 1)
        else:
            self.forecast_web = None
            self.forecast_image_label = QLabel("", page)
            self.forecast_canvas = None
            if _HAS_MPL_CANVAS and FigureCanvas is not None:
                self.forecast_figure = Figure(figsize=(8.8, 3.8), dpi=120)
                self.forecast_canvas = FigureCanvas(self.forecast_figure)
                self.forecast_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                self.forecast_canvas.setMinimumHeight(280)
                page_l.addWidget(self.forecast_canvas, 1)
            else:
                self.forecast_image_label.setAlignment(Qt.AlignCenter)
                self.forecast_image_label.setWordWrap(True)
                self.forecast_image_label.setMinimumHeight(280)
                self.forecast_image_label.setObjectName("CutoutImage")
                self.forecast_image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                self.forecast_image_label.setText("No forecast data loaded yet.")
                page_l.addWidget(self.forecast_image_label, 1)

        self.forecast_provider_details = QLabel("-", page)
        self.forecast_provider_details.setObjectName("SectionHint")
        self.forecast_provider_details.setWordWrap(False)
        self.forecast_provider_details.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.forecast_provider_details.setMaximumHeight(24)
        page_l.addWidget(self.forecast_provider_details)

        self.forecast_source_combo.currentIndexChanged.connect(self._update_live_views)
        src_hint = QLabel("Sources:", page)
        src_hint.setObjectName("SectionHint")
        page_l.addWidget(src_hint)
        self._append_links(page_l, entries, max_cols=5, min_button_w=90, max_button_w=146)
        page_l.addStretch(1)
        return page

    def _build_conditions_tab(self, entries: list[tuple[str, str, str]]) -> QWidget:
        page = QWidget(self.tabs)
        page_l = QVBoxLayout(page)
        page_l.setContentsMargins(10, 10, 10, 10)
        page_l.setSpacing(6)
        self.conditions_summary_label = QLabel("ok 0 · err 0", page)
        self.conditions_summary_label.setObjectName("SectionHint")
        self.conditions_summary_label.setWordWrap(False)

        src_row = QWidget(page)
        src_row_l = QHBoxLayout(src_row)
        src_row_l.setContentsMargins(0, 0, 0, 0)
        src_row_l.setSpacing(6)
        src_row_l.addWidget(self.conditions_summary_label, 0)

        self.conditions_updated_label = QLabel("UTC -", src_row)
        self.conditions_updated_label.setObjectName("SectionHint")
        src_row_l.addWidget(self.conditions_updated_label, 0)
        src_lbl = QLabel("Source:", src_row)
        src_lbl.setObjectName("SectionHint")
        src_row_l.addWidget(src_lbl, 0)
        self.conditions_source_combo = QComboBox(src_row)
        self.conditions_source_combo.setMinimumWidth(240)
        self.conditions_source_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        for key, label in self._CONDITION_SOURCE_CHOICES:
            self.conditions_source_combo.addItem(label, key)
        src_row_l.addWidget(self.conditions_source_combo, 1)
        self.conditions_polling_label = QLabel("Poll off", src_row)
        self.conditions_polling_label.setObjectName("SectionHint")
        src_row_l.addWidget(self.conditions_polling_label, 0)
        self._append_links_inline(src_row_l, entries, min_button_w=82, max_button_w=122, add_open_all=True)
        src_row_l.addStretch(1)
        page_l.addWidget(src_row)
        self.conditions_source_combo.currentIndexChanged.connect(self._on_conditions_source_combo_changed)
        self._sync_conditions_source_combo()

        if _HAS_QTWEBENGINE and _HAS_PLOTLY and QWebEngineView is not None:
            host, web_view = self._create_weather_plot_stack(
                page,
                kind="conditions",
                title="Conditions Trend",
                min_height=280,
            )
            self.conditions_trend_web = web_view
            page_l.addWidget(host, 1)
        else:
            self.conditions_trend_web = None
            self.conditions_trend_image_label = QLabel("", page)
            self.conditions_trend_canvas = None
            if _HAS_MPL_CANVAS and FigureCanvas is not None:
                self.conditions_trend_figure = Figure(figsize=(8.0, 3.8), dpi=120)
                self.conditions_trend_canvas = FigureCanvas(self.conditions_trend_figure)
                self.conditions_trend_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                self.conditions_trend_canvas.setMinimumHeight(280)
                page_l.addWidget(self.conditions_trend_canvas, 1)
            else:
                self.conditions_trend_image_label.setAlignment(Qt.AlignCenter)
                self.conditions_trend_image_label.setWordWrap(True)
                self.conditions_trend_image_label.setMinimumHeight(280)
                self.conditions_trend_image_label.setObjectName("CutoutImage")
                self.conditions_trend_image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                self.conditions_trend_image_label.setText("No conditions trend data loaded yet.")
                page_l.addWidget(self.conditions_trend_image_label, 1)
        return page

    def _build_cloud_analysis_tab(self, entries: list[tuple[str, str, str]]) -> QWidget:
        page = QWidget(self.tabs)
        page_l = QVBoxLayout(page)
        page_l.setContentsMargins(10, 10, 10, 10)
        page_l.setSpacing(8)

        top = QWidget(page)
        top_l = QVBoxLayout(top)
        top_l.setContentsMargins(0, 0, 0, 0)
        top_l.setSpacing(8)

        self.cloud_live_label = QLabel("Live cloud (selected source): -", top)
        self.cloud_live_label.setObjectName("SectionHint")
        self.cloud_live_label.setProperty("weather_chip", True)
        self.cloud_live_label.setProperty("weather_chip_role", "weather")
        self.cloud_annual_label = QLabel("Annual cloud average: -", top)
        self.cloud_annual_label.setObjectName("SectionHint")
        self.cloud_annual_label.setProperty("weather_chip", True)
        self.cloud_annual_label.setProperty("weather_chip_role", "context")
        self.cloud_map_source_label = QLabel("Source EarthEnv", top)
        self.cloud_map_source_label.setObjectName("SectionHint")
        self.cloud_map_source_label.setProperty("weather_chip", True)
        self.cloud_map_source_label.setProperty("weather_chip_role", "clock")

        summary_row = QWidget(top)
        summary_l = QHBoxLayout(summary_row)
        summary_l.setContentsMargins(0, 0, 0, 0)
        summary_l.setSpacing(6)
        summary_l.addWidget(self.cloud_live_label, 0)
        summary_l.addWidget(self.cloud_annual_label, 0)
        summary_l.addWidget(self.cloud_map_source_label, 0)
        summary_l.addStretch(1)
        top_l.addWidget(summary_row)

        self.cloud_low_spin = QSpinBox(top)
        self.cloud_mid_spin = QSpinBox(top)
        self.cloud_high_spin = QSpinBox(top)
        for spin, init in ((self.cloud_low_spin, 35), (self.cloud_mid_spin, 20), (self.cloud_high_spin, 10)):
            spin.setRange(0, 100)
            spin.setSuffix(" %")
            spin.setValue(init)
            spin.setFixedWidth(92)
            spin.valueChanged.connect(self._recompute_cloud_cover)

        weights_row = QWidget(top)
        weights_l = QHBoxLayout(weights_row)
        weights_l.setContentsMargins(0, 0, 0, 0)
        weights_l.setSpacing(8)
        weights_l.addWidget(QLabel("Low:", weights_row))
        weights_l.addWidget(self.cloud_low_spin)
        weights_l.addWidget(QLabel("Mid:", weights_row))
        weights_l.addWidget(self.cloud_mid_spin)
        weights_l.addWidget(QLabel("High:", weights_row))
        weights_l.addWidget(self.cloud_high_spin)
        month_lbl = QLabel("Month:", weights_row)
        month_lbl.setObjectName("SectionHint")
        weights_l.addWidget(month_lbl)
        self.cloud_month_combo = QComboBox(weights_row)
        for key, label in self._MONTH_CHOICES:
            self.cloud_month_combo.addItem(label, key)
        self.cloud_month_combo.setMinimumWidth(118)
        self.cloud_month_combo.setMaximumWidth(150)
        weights_l.addWidget(self.cloud_month_combo, 0)
        weights_l.addStretch(1)
        top_l.addWidget(weights_row)

        calc_row = QWidget(top)
        calc_l = QHBoxLayout(calc_row)
        calc_l.setContentsMargins(0, 0, 0, 0)
        calc_l.setSpacing(8)
        self.cloud_sunny_bar = QProgressBar(calc_row)
        self.cloud_sunny_bar.setRange(0, 100)
        self.cloud_sunny_bar.setTextVisible(True)
        self.cloud_sunny_bar.setFormat("%p%% clear")
        self.cloud_sunny_bar.setMinimumWidth(260)
        self.cloud_sunny_bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.cloud_result_label = QLabel("-", calc_row)
        self.cloud_result_label.setWordWrap(False)
        self.cloud_result_label.setObjectName("SectionHint")
        self.cloud_result_label.setProperty("weather_chip", True)
        self.cloud_result_label.setProperty("weather_chip_role", "solar")
        calc_l.addWidget(self.cloud_sunny_bar, 1)
        calc_l.addWidget(self.cloud_result_label, 0)
        top_l.addWidget(calc_row)

        page_l.addWidget(top)

        cloud_host, self.cloud_map_image_label = self._create_weather_image_stack(
            page,
            kind="cloud_map",
            title="Cloud Map",
            min_height=320,
        )
        page_l.addWidget(cloud_host, 1)

        self.cloud_map_info_label = QLabel("-", page)
        self.cloud_map_info_label.setWordWrap(False)
        self.cloud_map_info_label.setObjectName("SectionHint")
        page_l.addWidget(self.cloud_map_info_label)

        self._append_links(page_l, entries, max_cols=4, min_button_w=88, max_button_w=138)
        self.cloud_month_combo.currentIndexChanged.connect(self._on_cloud_month_changed)
        self._recompute_cloud_cover()
        return page

    def _build_meteogram_tab(self, entries: list[tuple[str, str, str]]) -> QWidget:
        page = QWidget(self.tabs)
        page_l = QVBoxLayout(page)
        page_l.setContentsMargins(10, 10, 10, 10)
        page_l.setSpacing(6)

        row = QWidget(page)
        row_l = QHBoxLayout(row)
        row_l.setContentsMargins(0, 0, 0, 0)
        row_l.setSpacing(8)
        lbl = QLabel("Meteogram source:", row)
        lbl.setObjectName("SectionHint")
        row_l.addWidget(lbl, 0)
        self.meteogram_source_combo = QComboBox(row)
        self.meteogram_source_combo.setMinimumWidth(240)
        self.meteogram_source_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        for key, label in self._METEOGRAM_SOURCE_CHOICES:
            self.meteogram_source_combo.addItem(label, key)
        row_l.addWidget(self.meteogram_source_combo, 1)
        row_l.addSpacing(8)
        self._append_links_inline(row_l, entries, min_button_w=82, max_button_w=128, add_open_all=True)
        row_l.addStretch(1)
        page_l.addWidget(row)

        if _HAS_QTWEBENGINE and _HAS_PLOTLY and QWebEngineView is not None:
            host, web_view = self._create_weather_plot_stack(
                page,
                kind="meteogram",
                title="Meteogram",
                min_height=320,
            )
            self.meteogram_web = web_view
            page_l.addWidget(host, 1)
        else:
            self.meteogram_web = None
            self.meteogram_image_label = QLabel("", page)
            self.meteogram_canvas = None
            if _HAS_MPL_CANVAS and FigureCanvas is not None:
                self.meteogram_figure = Figure(figsize=(9.0, 4.0), dpi=120)
                self.meteogram_canvas = FigureCanvas(self.meteogram_figure)
                self.meteogram_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                self.meteogram_canvas.setMinimumHeight(320)
                page_l.addWidget(self.meteogram_canvas, 1)
            else:
                self.meteogram_image_label.setAlignment(Qt.AlignCenter)
                self.meteogram_image_label.setWordWrap(True)
                self.meteogram_image_label.setMinimumHeight(320)
                self.meteogram_image_label.setObjectName("CutoutImage")
                self.meteogram_image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                self.meteogram_image_label.setText("No meteogram data loaded yet.")
                page_l.addWidget(self.meteogram_image_label, 1)

        self.meteogram_source_combo.currentIndexChanged.connect(self._update_live_views)
        return page

    def _build_satellite_tab(self, entries: list[tuple[str, str, str]]) -> QWidget:
        page = QWidget(self.tabs)
        page_l = QVBoxLayout(page)
        page_l.setContentsMargins(10, 10, 10, 10)
        page_l.setSpacing(8)

        sat_host, self.satellite_image_label = self._create_weather_image_stack(
            page,
            kind="satellite",
            title="Satellite",
            min_height=340,
        )
        page_l.addWidget(sat_host, 1)

        self._append_links(page_l, entries, max_cols=4, min_button_w=88, max_button_w=138)
        return page

    @Slot()
    def _recompute_cloud_cover(self) -> None:
        if not hasattr(self, "cloud_low_spin"):
            return
        low = float(self.cloud_low_spin.value())
        mid = float(self.cloud_mid_spin.value())
        high = float(self.cloud_high_spin.value())
        effective_cloud = max(0.0, min(100.0, 0.65 * low + 0.25 * mid + 0.10 * high))
        clear_rate = max(0.0, min(100.0, 100.0 - effective_cloud))
        self.cloud_sunny_bar.setValue(int(round(clear_rate)))
        if clear_rate >= 80.0:
            rating = "excellent"
        elif clear_rate >= 60.0:
            rating = "good"
        elif clear_rate >= 40.0:
            rating = "fair"
        else:
            rating = "poor"
        self.cloud_result_label.setText(
            f"{clear_rate:.1f}% clear · {rating} · cloud {effective_cloud:.1f}%"
        )

    @Slot(str, int, int)
    def _on_live_progress(self, status: str, step: int, total: int) -> None:
        current = self.tabs.currentWidget() if hasattr(self, "tabs") else None
        in_cloud_tab = current is self.tabs.widget(2) if hasattr(self, "tabs") and self.tabs.count() >= 3 else False
        in_sat_tab = current is self.tabs.widget(3) if hasattr(self, "tabs") and self.tabs.count() >= 4 else False
        status_text = status or "Loading live data..."
        if ("monthly cloud map" in status_text.lower()) and not in_cloud_tab:
            status_text = "Refreshing weather providers..."
        if ("satellite preview" in status_text.lower()) and not in_sat_tab:
            status_text = "Refreshing weather providers..."
        self.live_status_label.setText(self._compact_live_status(status_text))
        self.live_status_label.setToolTip(status_text)
        if total > 0:
            self.live_progress.setRange(0, int(total))
            self.live_progress.setValue(max(0, min(int(total), int(step))))
        else:
            self.live_progress.setRange(0, 0)
        if self.live_progress.isHidden():
            self.live_progress.show()

    @Slot(dict)
    def _on_live_partial(self, payload: dict) -> None:
        if isinstance(payload, dict):
            self._live_payload = self._merged_live_payload(payload)
            self._update_live_views()

    @Slot(dict)
    def _on_live_completed(self, payload: dict) -> None:
        self._live_payload = self._merged_live_payload(payload if isinstance(payload, dict) else {})
        self._last_live_fetch_perf = perf_counter()
        self.live_progress.hide()
        self._update_live_views()

    def _merged_live_payload(self, payload: dict[str, object]) -> dict[str, object]:
        if not isinstance(payload, dict):
            return {}
        prev = self._live_payload if isinstance(self._live_payload, dict) else {}
        if not prev:
            return payload
        merged = dict(prev)
        merged.update(payload)

        new_cloud_map = payload.get("cloud_map")
        if new_cloud_map is None and isinstance(prev.get("cloud_map"), dict):
            merged["cloud_map"] = prev.get("cloud_map")
        new_sat = payload.get("satellite")
        if new_sat is None and isinstance(prev.get("satellite"), dict):
            merged["satellite"] = prev.get("satellite")
        return merged

    @Slot(int)
    def _on_tab_changed(self, _index: int) -> None:
        current = self.tabs.currentWidget()
        is_conditions_tab = bool(current is getattr(self, "conditions_tab_widget", None))
        if is_conditions_tab:
            if hasattr(self, "conditions_polling_label"):
                interval_s = int(max(30.0, self._weather_auto_refresh_s))
                self.conditions_polling_label.setText(f"Poll {interval_s}s")
                self.conditions_polling_label.setToolTip(f"Polling every {interval_s} s")
            self._conditions_poll_timer.start(int(max(30.0, self._weather_auto_refresh_s) * 1000))
            self._start_live_refresh(force=False, include_cloud_map=False, include_satellite=False)
            return

        self._conditions_poll_timer.stop()
        if hasattr(self, "conditions_polling_label"):
            self.conditions_polling_label.setText("Poll off")
            self.conditions_polling_label.setToolTip("Polling off")

        is_cloud_tab = bool(hasattr(self, "tabs") and self.tabs.count() >= 3 and current is self.tabs.widget(2))
        if is_cloud_tab and not self._has_cloud_map_for_current_context():
            self._start_live_refresh(force=False, include_cloud_map=True, include_satellite=False)
            return

        is_satellite_tab = bool(hasattr(self, "tabs") and self.tabs.count() >= 4 and current is self.tabs.widget(3))
        if is_satellite_tab and not self._has_satellite_for_current_site():
            self._start_live_refresh(force=False, include_cloud_map=False, include_satellite=True)
            return

    @Slot()
    def _on_conditions_poll_timeout(self) -> None:
        if not self.isVisible():
            return
        current = self.tabs.currentWidget()
        if current is not getattr(self, "conditions_tab_widget", None):
            return
        if isinstance(self._live_worker, WeatherLiveWorker) and self._live_worker.isRunning():
            return
        self._start_live_refresh(force=False, include_cloud_map=False, include_satellite=False)

    def _stop_live_worker(self) -> None:
        worker = self._live_worker
        self._live_worker = None
        if not isinstance(worker, WeatherLiveWorker):
            return
        if worker.isRunning():
            worker.requestInterruption()
            if not worker.wait(1200):
                worker.terminate()
                worker.wait(300)

    def _live_context_key(self) -> str:
        lat, lon, elev = self._site_coordinates()
        return "|".join(
            (
                f"{lat:.4f}",
                f"{lon:.4f}",
                f"{elev:.0f}",
                self._date.toString("yyyy-MM-dd"),
                f"{self._sun_alt_limit:.0f}",
                self._weather_custom_url.strip(),
                str(self._selected_cloud_month()),
                self._weather_cloud_source,
            )
        )

    def _start_live_refresh(
        self,
        *,
        force: bool,
        include_cloud_map: Optional[bool] = None,
        include_satellite: Optional[bool] = None,
    ) -> None:
        lat, lon, elev = self._site_coordinates()
        key = self._live_context_key()
        if include_cloud_map is None:
            current = self.tabs.currentWidget() if hasattr(self, "tabs") else None
            include_cloud_map = bool(
                current is self.tabs.widget(2) if hasattr(self, "tabs") and self.tabs.count() >= 3 else False
            )
            if include_cloud_map and self._has_cloud_map_for_current_context() and not force:
                include_cloud_map = False
        if include_satellite is None:
            current = self.tabs.currentWidget() if hasattr(self, "tabs") else None
            include_satellite = bool(
                current is self.tabs.widget(3) if hasattr(self, "tabs") and self.tabs.count() >= 4 else False
            )
            if include_satellite and self._has_satellite_for_current_site() and not force:
                include_satellite = False
        needs_cloud_fetch = bool(include_cloud_map) and not self._has_cloud_map_for_current_context()
        needs_sat_fetch = bool(include_satellite) and not self._has_satellite_for_current_site()
        now_perf = perf_counter()
        if not force and key == self._last_live_context_key and self._live_payload:
            if (
                now_perf - float(self._last_live_fetch_perf) < float(self._weather_auto_refresh_s)
                and not needs_cloud_fetch
                and not needs_sat_fetch
            ):
                return
        self._last_live_context_key = key
        self._stop_live_worker()
        self.live_progress.setRange(0, 0)
        self.live_progress.show()
        self.live_status_label.setText("Loading sources…")
        self.live_status_label.setToolTip("Loading live weather providers...")
        if bool(include_cloud_map):
            self._set_weather_image_loading("cloud_map", "Loading cloud climatology…", visible=True)
        if bool(include_satellite):
            self._set_weather_image_loading("satellite", "Loading satellite preview…", visible=True)
        worker = WeatherLiveWorker(
            lat=lat,
            lon=lon,
            elev=elev,
            custom_conditions_url=self._weather_custom_url,
            cloud_map_source=self._weather_cloud_source,
            cloud_map_month=self._selected_cloud_month(),
            force_refresh=force,
            include_cloud_map=bool(include_cloud_map),
            include_satellite=bool(include_satellite),
            storage=self._storage,
            parent=self,
        )
        worker.progress.connect(self._on_live_progress)
        worker.partial.connect(self._on_live_partial)
        worker.completed.connect(self._on_live_completed)
        worker.finished.connect(lambda: self.live_progress.hide())
        self._live_worker = worker
        worker.start()

    @staticmethod
    def _fallback_plot_bytes(
        series: dict[str, object],
        title: str,
        *,
        max_points: int = 72,
        dark: bool = True,
        theme_tokens: Optional[dict[str, str]] = None,
    ) -> bytes:
        ts_raw = series.get("timestamps")
        ts = [int(v) for v in ts_raw if isinstance(v, (int, float))] if isinstance(ts_raw, list) else []
        temp = [float(v) for v in series.get("temp_c", []) if isinstance(v, (int, float))]
        wind = [float(v) for v in series.get("wind_ms", []) if isinstance(v, (int, float))]
        cloud = [float(v) for v in series.get("cloud_pct", []) if isinstance(v, (int, float))]
        rh = [float(v) for v in series.get("rh_pct", []) if isinstance(v, (int, float))]
        pressure = [float(v) for v in series.get("pressure_hpa", []) if isinstance(v, (int, float))]
        n = max(len(temp), len(wind), len(cloud), len(rh), len(pressure))
        if n <= 1:
            return b""
        n = min(int(max_points), n)
        if ts and len(ts) >= n:
            x_axis = [datetime.fromtimestamp(int(v), tz=timezone.utc).astimezone() for v in ts[-n:]]
        else:
            base = datetime.now().replace(minute=0, second=0, microsecond=0) - timedelta(hours=n - 1)
            x_axis = [base + timedelta(hours=i) for i in range(n)]
        use_tokens = theme_tokens or resolve_theme_tokens(DEFAULT_UI_THEME, dark_enabled=dark)
        plot_bg = str(use_tokens.get("plot_bg", "#121b29"))
        plot_panel_bg = str(use_tokens.get("plot_panel_bg", plot_bg))
        plot_text = str(use_tokens.get("plot_text", "#d7e4f0" if dark else "#253347"))
        plot_grid = str(use_tokens.get("plot_grid", "#2f4666" if dark else "#d6deea"))
        soft_grid_css = _softened_plot_grid_css_from_tokens(use_tokens, alpha=0.42)
        plot_font = _plot_font_css_stack(use_tokens)

        fig = Figure(figsize=(9.0, 4.4), dpi=120)
        fig.patch.set_facecolor(plot_bg)
        axes = fig.subplots(
            3,
            1,
            sharex=True,
            gridspec_kw={"height_ratios": [0.44, 0.33, 0.23], "hspace": 0.12},
        )
        ax_temp = axes[0]
        ax_cloud = axes[1]
        ax_press = axes[2]
        for axis in (ax_temp, ax_cloud, ax_press):
            axis.set_facecolor(plot_panel_bg)
        handles: list[Any] = []
        labels: list[str] = []

        def add_line(axis, values: list[float], name: str, color: str, style: str = "-") -> None:
            if len(values) < n:
                return
            (line,) = axis.plot(x_axis, values[-n:], color=color, linewidth=1.8, linestyle=style, label=name)
            handles.append(line)
            labels.append(name)

        add_line(ax_temp, temp, "Temp (°C)", str(use_tokens.get("plot_series_temp", "#f08a24")))
        add_line(ax_temp, wind, "Wind (m/s)", str(use_tokens.get("plot_series_wind", "#4aa7ff")))
        add_line(ax_cloud, cloud, "Cloud (%)", str(use_tokens.get("plot_series_cloud", "#5cd2ff")))
        add_line(ax_cloud, rh, "Humidity (%)", str(use_tokens.get("plot_series_humidity", "#86d37f")), "--")
        add_line(ax_press, pressure, "Pressure (hPa)", str(use_tokens.get("plot_series_pressure", "#b29bff")), "-.")

        ax_temp.set_title(title, color=plot_text)
        ax_temp.set_ylabel("Temp / Wind", color=plot_text)
        ax_cloud.set_ylabel("Cloud / RH", color=plot_text)
        ax_press.set_ylabel("Pressure (hPa)", color=plot_text)
        ax_press.set_xlabel("Time (local)", color=plot_text)
        for axis in (ax_temp, ax_cloud, ax_press):
            axis.tick_params(axis="x", colors=plot_text)
            axis.tick_params(axis="y", colors=plot_text)
            axis.grid(True, alpha=0.28, linestyle="--", linewidth=0.7, color=plot_grid)
            for spine in axis.spines.values():
                spine.set_color(plot_grid)
            axis.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        fig.autofmt_xdate(rotation=20)
        if handles:
            fig.legend(
                handles,
                labels,
                loc="lower center",
                ncol=min(5, len(handles)),
                frameon=False,
                bbox_to_anchor=(0.5, -0.02),
                labelcolor=plot_text,
            )
        out = io.BytesIO()
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        fig.savefig(out, format="png")
        return out.getvalue()

    @staticmethod
    def _build_plotly_html(
        series: dict[str, object],
        title: str,
        *,
        max_points: int = 96,
        dark: bool = True,
        theme_tokens: Optional[dict[str, str]] = None,
        show_legend: bool = False,
    ) -> Optional[str]:
        if not (_HAS_PLOTLY and plotly_to_html is not None and plotly_make_subplots is not None and go is not None):
            return None
        ts_raw = series.get("timestamps")
        ts = [int(v) for v in ts_raw if isinstance(v, (int, float))] if isinstance(ts_raw, list) else []
        temp = [float(v) for v in series.get("temp_c", []) if isinstance(v, (int, float))]
        wind = [float(v) for v in series.get("wind_ms", []) if isinstance(v, (int, float))]
        cloud = [float(v) for v in series.get("cloud_pct", []) if isinstance(v, (int, float))]
        rh = [float(v) for v in series.get("rh_pct", []) if isinstance(v, (int, float))]
        pressure = [float(v) for v in series.get("pressure_hpa", []) if isinstance(v, (int, float))]
        n = max(len(temp), len(wind), len(cloud), len(rh), len(pressure))
        if n <= 1:
            return None
        n = min(int(max_points), n)
        if ts and len(ts) >= n:
            x = [datetime.fromtimestamp(int(v), tz=timezone.utc).astimezone() for v in ts[-n:]]
        else:
            base = datetime.now().replace(minute=0, second=0, microsecond=0) - timedelta(hours=n - 1)
            x = [base + timedelta(hours=i) for i in range(n)]
        use_tokens = theme_tokens or resolve_theme_tokens(DEFAULT_UI_THEME, dark_enabled=dark)
        plot_bg = str(use_tokens.get("plot_bg", "#121b29"))
        plot_panel_bg = str(use_tokens.get("plot_panel_bg", plot_bg))
        plot_text = str(use_tokens.get("plot_text", "#d7e4f0" if dark else "#253347"))
        plot_grid = str(use_tokens.get("plot_grid", "#2f4666" if dark else "#d6deea"))
        soft_grid_css = _softened_plot_grid_css_from_tokens(use_tokens, alpha=0.42)
        plot_font = _plot_font_css_stack(use_tokens)

        fig = plotly_make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            row_heights=[0.44, 0.33, 0.23],
            specs=[
                [{"secondary_y": False}],
                [{"secondary_y": False}],
                [{"secondary_y": False}],
            ],
        )

        def add(values: list[float], name: str, color: str, *, row: int, dash: str = "solid") -> None:
            if len(values) < n:
                return
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=values[-n:],
                    mode="lines",
                    name=name,
                    line={"color": color, "width": 2, "dash": dash},
                ),
                row=row,
                col=1,
            )

        add(temp, "Temp (°C)", str(use_tokens.get("plot_series_temp", "#f08a24")), row=1)
        add(wind, "Wind (m/s)", str(use_tokens.get("plot_series_wind", "#4aa7ff")), row=1)
        add(cloud, "Cloud (%)", str(use_tokens.get("plot_series_cloud", "#00d0ff")), row=2)
        add(rh, "Humidity (%)", str(use_tokens.get("plot_series_humidity", "#76dc7a")), row=2, dash="dash")
        add(pressure, "Pressure (hPa)", str(use_tokens.get("plot_series_pressure", "#b29bff")), row=3, dash="dot")

        fig.update_layout(
            template="plotly_dark" if dark else "plotly_white",
            title=title,
            margin={"l": 46, "r": 14, "t": 36, "b": 34 if not show_legend else 64},
            showlegend=bool(show_legend),
            legend={"orientation": "h", "y": -0.22, "x": 0.5, "xanchor": "center"},
            hovermode="x unified",
            paper_bgcolor=plot_bg,
            plot_bgcolor=plot_panel_bg,
            font={"color": plot_text, "family": plot_font},
        )
        fig.update_xaxes(showgrid=True, gridcolor=soft_grid_css, zerolinecolor=soft_grid_css, color=plot_text, row=1, col=1)
        fig.update_xaxes(showgrid=True, gridcolor=soft_grid_css, zerolinecolor=soft_grid_css, color=plot_text, row=2, col=1)
        fig.update_xaxes(showgrid=True, gridcolor=soft_grid_css, zerolinecolor=soft_grid_css, color=plot_text, row=3, col=1)
        fig.update_yaxes(showgrid=True, gridcolor=soft_grid_css, zerolinecolor=soft_grid_css, color=plot_text, row=1, col=1)
        fig.update_yaxes(showgrid=True, gridcolor=soft_grid_css, zerolinecolor=soft_grid_css, color=plot_text, row=2, col=1)
        fig.update_yaxes(showgrid=True, gridcolor=soft_grid_css, zerolinecolor=soft_grid_css, color=plot_text, row=3, col=1)
        fig.update_xaxes(title_text="Time (local)", row=3, col=1)
        fig.update_yaxes(title_text="Temp / Wind", row=1, col=1)
        fig.update_yaxes(title_text="Cloud / RH", row=2, col=1)
        fig.update_yaxes(title_text="Pressure (hPa)", row=3, col=1)
        html_fragment = plotly_to_html(
            fig,
            include_plotlyjs=False,
            full_html=False,
            config={"displaylogo": False, "responsive": True, "scrollZoom": True},
        )
        html_fragment = html_fragment.replace("<div>", "<div id='plot-host' style='width:100%;height:100%;min-height:100%;'>", 1)
        graph_id_match = re.search(r'<div id="([^"]+)" class="plotly-graph-div"', html_fragment)
        graph_id = graph_id_match.group(1) if graph_id_match else ""
        resize_script = ""
        if graph_id:
            resize_script = (
                "<script>"
                "(function(){"
                f"const gd=document.getElementById('{graph_id}');"
                "if(!gd||!window.Plotly){return;}"
                "const resize=()=>{"
                "gd.style.width='100%';"
                "gd.style.height='100%';"
                "if(window.Plotly&&Plotly.Plots){Plotly.Plots.resize(gd);}"
                "};"
                "window.addEventListener('resize', resize);"
                "if(window.ResizeObserver){new ResizeObserver(resize).observe(document.body);}"
                "setTimeout(resize,0);"
                "setTimeout(resize,120);"
                "})();"
                "</script>"
            )
        font_face_css = _embedded_display_font_css()
        html_head = (
            "<!doctype html><html><head><meta charset='utf-8'>"
            "<style>"
            f"{font_face_css}"
            f"html,body,#plot-host{{margin:0;width:100%;height:100%;overflow:hidden;background:{plot_bg};}}"
            f"html,body,#plot-host,.plotly-graph-div{{font-family:{plot_font};}}"
            ".plotly-graph-div{width:100%!important;height:100%!important;}"
            "</style>"
        )
        if _PLOTLY_JS_BASE_DIR:
            return (
                f"{html_head}<script src='plotly.min.js'></script></head><body>"
                f"{html_fragment}{resize_script}</body></html>"
            )
        return (
            f"{html_head}<script src='https://cdn.plot.ly/plotly-3.4.0.min.js'></script></head><body>"
            f"{html_fragment}{resize_script}</body></html>"
        )

    @staticmethod
    def _render_series_on_canvas(
        figure: Figure,
        canvas: "FigureCanvas",
        series: dict[str, object],
        *,
        title: str,
        max_points: int,
        dark: bool,
        theme_tokens: Optional[dict[str, str]] = None,
        show_legend: bool = False,
    ) -> bool:
        ts_raw = series.get("timestamps")
        ts = [int(v) for v in ts_raw if isinstance(v, (int, float))] if isinstance(ts_raw, list) else []
        temp = [float(v) for v in series.get("temp_c", []) if isinstance(v, (int, float))]
        wind = [float(v) for v in series.get("wind_ms", []) if isinstance(v, (int, float))]
        cloud = [float(v) for v in series.get("cloud_pct", []) if isinstance(v, (int, float))]
        rh = [float(v) for v in series.get("rh_pct", []) if isinstance(v, (int, float))]
        pressure = [float(v) for v in series.get("pressure_hpa", []) if isinstance(v, (int, float))]

        n = max(len(temp), len(wind), len(cloud), len(rh), len(pressure))
        if n <= 1:
            figure.clear()
            ax = figure.add_subplot(111)
            ax.set_axis_off()
            use_tokens = theme_tokens or resolve_theme_tokens(DEFAULT_UI_THEME, dark_enabled=dark)
            ax.text(
                0.5,
                0.5,
                "No series data.",
                ha="center",
                va="center",
                fontsize=10,
                color=str(use_tokens.get("plot_text", "#d9e2f0" if dark else "#2e3c4f")),
            )
            canvas.draw_idle()
            return False
        n = min(max_points, n)
        if ts and len(ts) >= n:
            x_axis = [datetime.fromtimestamp(int(v), tz=timezone.utc).astimezone() for v in ts[-n:]]
        else:
            base = datetime.now().replace(minute=0, second=0, microsecond=0) - timedelta(hours=n - 1)
            x_axis = [base + timedelta(hours=i) for i in range(n)]
        use_tokens = theme_tokens or resolve_theme_tokens(DEFAULT_UI_THEME, dark_enabled=dark)
        plot_bg = str(use_tokens.get("plot_bg", "#0f1825" if dark else "#f7f9fc"))
        plot_panel_bg = str(use_tokens.get("plot_panel_bg", "#162334" if dark else "#ffffff"))
        plot_grid = str(use_tokens.get("plot_grid", "#2f4666" if dark else "#d6deea"))
        plot_text = str(use_tokens.get("plot_text", "#d9e2f0" if dark else "#253347"))
        soft_grid_rgba = _softened_plot_grid_rgba_from_tokens(use_tokens, alpha=0.42)
        soft_grid_css = _softened_plot_grid_qcolor_from_tokens(use_tokens).name()

        figure.clear()
        figure.patch.set_facecolor(plot_bg)
        axes = figure.subplots(
            3,
            1,
            sharex=True,
            gridspec_kw={"height_ratios": [0.44, 0.33, 0.23], "hspace": 0.10},
        )
        ax_temp = axes[0]
        ax_cloud = axes[1]
        ax_press = axes[2]
        ax_temp.set_facecolor(plot_panel_bg)
        ax_cloud.set_facecolor(plot_panel_bg)
        ax_press.set_facecolor(plot_panel_bg)

        handles: list[Any] = []
        labels: list[str] = []

        def _plot(axis, values: list[float], name: str, color: str, style: str = "-") -> None:
            if len(values) < n:
                return
            (line,) = axis.plot(x_axis, values[-n:], color=color, linewidth=1.9, linestyle=style, label=name)
            handles.append(line)
            labels.append(name)

        _plot(ax_temp, temp, "Temp (°C)", str(use_tokens.get("plot_series_temp", "#f08a24")))
        _plot(ax_temp, wind, "Wind (m/s)", str(use_tokens.get("plot_series_wind", "#4aa7ff")))
        _plot(ax_cloud, cloud, "Cloud (%)", str(use_tokens.get("plot_series_cloud", "#00d0ff")))
        _plot(ax_cloud, rh, "Humidity (%)", str(use_tokens.get("plot_series_humidity", "#76dc7a")), "--")
        _plot(ax_press, pressure, "Pressure (hPa)", str(use_tokens.get("plot_series_pressure", "#b29bff")), "-.")

        ax_temp.set_title(title, color=plot_text, fontsize=11, pad=6)
        ax_temp.set_ylabel("Temp / Wind", color=plot_text)
        ax_cloud.set_ylabel("Cloud / RH", color=plot_text)
        ax_press.set_ylabel("Pressure (hPa)", color=plot_text)
        ax_press.set_xlabel("Time (local)", color=plot_text)
        for axis in (ax_temp, ax_cloud, ax_press):
            axis.tick_params(axis="x", colors=plot_text)
            axis.tick_params(axis="y", colors=plot_text)
            axis.grid(True, linestyle="--", linewidth=0.7, color=soft_grid_rgba)
            axis.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        for spine in ax_temp.spines.values():
            spine.set_color(soft_grid_css)
        for spine in ax_cloud.spines.values():
            spine.set_color(soft_grid_css)
        for spine in ax_press.spines.values():
            spine.set_color(soft_grid_css)

        if handles and show_legend:
            figure.legend(
                handles,
                labels,
                loc="lower center",
                ncol=min(5, len(handles)),
                frameon=False,
                bbox_to_anchor=(0.5, 0.01),
                labelcolor=plot_text,
            )
        figure.subplots_adjust(left=0.06, right=0.992, top=0.94, bottom=0.11 if not show_legend else 0.15, hspace=0.10)
        canvas.draw_idle()
        return True

    def _rescale_preview_pixmaps(self) -> None:
        if hasattr(self, "conditions_trend_image_label") and not self._conditions_raw_pixmap.isNull():
            size = self.conditions_trend_image_label.size()
            if size.width() > 2 and size.height() > 2:
                self.conditions_trend_image_label.setPixmap(
                    self._conditions_raw_pixmap.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )
        if hasattr(self, "meteogram_image_label") and not self._meteogram_raw_pixmap.isNull():
            size = self.meteogram_image_label.size()
            if size.width() > 2 and size.height() > 2:
                self.meteogram_image_label.setPixmap(
                    self._meteogram_raw_pixmap.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )
        if hasattr(self, "cloud_map_image_label") and not self._cloud_map_raw_pixmap.isNull():
            size = self.cloud_map_image_label.size()
            if size.width() > 2 and size.height() > 2:
                self.cloud_map_image_label.setPixmap(
                    self._cloud_map_raw_pixmap.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )
        if hasattr(self, "satellite_image_label") and not self._satellite_raw_pixmap.isNull():
            size = self.satellite_image_label.size()
            if size.width() > 2 and size.height() > 2:
                self.satellite_image_label.setPixmap(
                    self._satellite_raw_pixmap.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )

    def _series_for_source(self, source_key: str, series_map: dict[str, dict[str, object]]) -> Optional[dict[str, object]]:
        row = series_map.get(source_key)
        if isinstance(row, dict):
            return row
        fallback = series_map.get("open_meteo")
        if isinstance(fallback, dict):
            return fallback
        for key in ("metar", "custom", "meteoblue", "windy", "meteo_icm"):
            value = series_map.get(key)
            if isinstance(value, dict):
                return value
        return None

    def _series_for_conditions_source(
        self,
        source_key: str,
        series_map: dict[str, dict[str, object]],
    ) -> Optional[dict[str, object]]:
        if source_key != "average":
            row = series_map.get(source_key)
            return row if isinstance(row, dict) else None
        for key in ("custom", "metar"):
            candidate = series_map.get(key)
            if self._series_has_points(candidate, min_points=1):
                return candidate
        return None

    @staticmethod
    def _trim_series_for_recent(series: Optional[dict[str, object]], max_points: int) -> Optional[dict[str, object]]:
        if not isinstance(series, dict):
            return series
        ts_raw = series.get("timestamps")
        ts = [int(v) for v in ts_raw if isinstance(v, (int, float))] if isinstance(ts_raw, list) else []
        if not ts or max_points <= 0:
            return series
        now_ts = int(datetime.now(timezone.utc).timestamp())
        idx = 0
        for i, t in enumerate(ts):
            if t <= now_ts:
                idx = i
            else:
                break
        start = max(0, idx - max_points + 1)
        end = min(len(ts), start + max_points)

        def _slice(values: object) -> object:
            if not isinstance(values, list):
                return values
            return values[start:end]

        trimmed = dict(series)
        trimmed["timestamps"] = ts[start:end]
        for key in ("temp_c", "wind_ms", "cloud_pct", "rh_pct", "pressure_hpa"):
            trimmed[key] = _slice(series.get(key))
        return trimmed

    def _set_plot_html(self, web_view: Optional["QWebEngineView"], html: str) -> None:
        if web_view is None or not html:
            return
        kind = str(web_view.property("weather_plot_kind") or "").strip().lower()
        if kind:
            self._set_weather_plot_loading(kind, "Rendering interactive chart…", visible=True)
        if _PLOTLY_JS_BASE_DIR:
            web_view.setHtml(html, QUrl.fromLocalFile(str(_PLOTLY_JS_BASE_DIR) + "/"))
            return
        web_view.setHtml(html)

    @staticmethod
    def _series_has_points(series: object, *, min_points: int = 2) -> bool:
        if not isinstance(series, dict):
            return False
        for key in ("temp_c", "wind_ms", "cloud_pct", "rh_pct", "pressure_hpa"):
            values = series.get(key)
            if isinstance(values, list) and len(values) >= int(min_points):
                return True
        return False

    def _series_for_forecast_source(
        self, preferred_key: str, series_map: dict[str, dict[str, object]]
    ) -> tuple[Optional[str], Optional[dict[str, object]]]:
        preferred = series_map.get(preferred_key)
        if self._series_has_points(preferred):
            return preferred_key, preferred
        for key in ("open_meteo", "meteoblue", "windy", "meteo_icm"):
            candidate = series_map.get(key)
            if self._series_has_points(candidate):
                return key, candidate
        return None, None

    def _has_cloud_map_for_current_context(self) -> bool:
        payload = self._live_payload if isinstance(self._live_payload, dict) else {}
        cloud_map = payload.get("cloud_map")
        if not isinstance(cloud_map, dict):
            return False
        data = cloud_map.get("image_bytes")
        if not isinstance(data, (bytes, bytearray)) or not data:
            return False
        month_match = int(_safe_int(cloud_map.get("month")) or self._selected_cloud_month()) == self._selected_cloud_month()
        source_match = str(cloud_map.get("source") or "").strip().lower() == str(self._weather_cloud_source or "").strip().lower()
        return bool(month_match and source_match)

    def _has_satellite_for_current_site(self) -> bool:
        payload = self._live_payload if isinstance(self._live_payload, dict) else {}
        sat = payload.get("satellite")
        if not isinstance(sat, dict):
            return False
        data = sat.get("image_bytes")
        return bool(isinstance(data, (bytes, bytearray)) and data)

    def _update_live_views(self) -> None:
        payload = self._live_payload if isinstance(self._live_payload, dict) else {}
        providers = payload.get("providers") if isinstance(payload.get("providers"), dict) else {}
        series_map = payload.get("series") if isinstance(payload.get("series"), dict) else {}
        averages = payload.get("averages") if isinstance(payload.get("averages"), dict) else {}
        sections = payload.get("sections") if isinstance(payload.get("sections"), dict) else {}
        errors = payload.get("errors") if isinstance(payload.get("errors"), list) else []

        def _avg_local(values: list[float]) -> Optional[float]:
            if not values:
                return None
            return float(sum(values) / len(values))

        source_key = str(self.live_source_combo.currentData() or "average")
        current_row: dict[str, object] = {}
        if source_key == "average":
            current_row = {
                "temp_c": _safe_float((averages.get("temp_c") or {}).get("value")) if isinstance(averages.get("temp_c"), dict) else None,
                "wind_ms": _safe_float((averages.get("wind_ms") or {}).get("value")) if isinstance(averages.get("wind_ms"), dict) else None,
                "cloud_pct": _safe_float((averages.get("cloud_pct") or {}).get("value")) if isinstance(averages.get("cloud_pct"), dict) else None,
                "rh_pct": _safe_float((averages.get("rh_pct") or {}).get("value")) if isinstance(averages.get("rh_pct"), dict) else None,
                "pressure_hpa": _safe_float((averages.get("pressure_hpa") or {}).get("value")) if isinstance(averages.get("pressure_hpa"), dict) else None,
                "note": "Average values from available measured sources.",
                "updated_utc": "-",
            }
            if all(current_row.get(k) is None for k in ("temp_c", "wind_ms", "cloud_pct", "rh_pct", "pressure_hpa")):
                # Fallback: compute averages from provider rows when aggregate payload is empty.
                values_by_metric: dict[str, list[float]] = {k: [] for k in ("temp_c", "wind_ms", "cloud_pct", "rh_pct", "pressure_hpa")}
                for key, row in providers.items():
                    if not isinstance(row, dict):
                        continue
                    if not WeatherLiveWorker._is_measured_conditions_provider(str(key), row):
                        continue
                    status = str(row.get("status") or "").strip().lower()
                    if status not in {"ok", "partial"}:
                        continue
                    for metric in values_by_metric:
                        value = _safe_float(row.get(metric))
                        if value is None or not math.isfinite(float(value)):
                            continue
                        values_by_metric[metric].append(float(value))
                current_row.update(
                    {
                        "temp_c": _avg_local(values_by_metric["temp_c"]),
                        "wind_ms": _avg_local(values_by_metric["wind_ms"]),
                        "cloud_pct": _avg_local(values_by_metric["cloud_pct"]),
                        "rh_pct": _avg_local(values_by_metric["rh_pct"]),
                        "pressure_hpa": _avg_local(values_by_metric["pressure_hpa"]),
                        "note": "Average values from available measured sources.",
                    }
                )
        else:
            row = providers.get(source_key)
            if isinstance(row, dict):
                current_row = row

        self._set_weather_chip_label(self.live_temp_label, "T", self._fmt_value(_safe_float(current_row.get('temp_c')), '°C'))
        self._set_weather_chip_label(self.live_wind_label, "W", self._fmt_value(_safe_float(current_row.get('wind_ms')), 'm/s'))
        self._set_weather_chip_label(self.live_cloud_label, "C", self._fmt_value(_safe_float(current_row.get('cloud_pct')), '%'))
        self._set_weather_chip_label(self.live_rh_label, "RH", self._fmt_value(_safe_float(current_row.get('rh_pct')), '%'))
        self._set_weather_chip_label(
            self.live_pressure_label,
            "P",
            self._fmt_value(_safe_float(current_row.get('pressure_hpa')), 'hPa'),
        )
        self._update_summary_chip_widths()

        logger.info(
            "Weather chips: src=%s T=%s W=%s C=%s RH=%s P=%s providers=%d avg_keys=%s",
            source_key,
            current_row.get("temp_c"),
            current_row.get("wind_ms"),
            current_row.get("cloud_pct"),
            current_row.get("rh_pct"),
            current_row.get("pressure_hpa"),
            len(providers),
            ",".join(sorted(averages.keys())) if isinstance(averages, dict) else "-",
        )

        base_status = str(current_row.get("note") or "Live data: idle").strip()
        if errors:
            base_status = f"{base_status} | Partial errors: {len(errors)}"
        full_status = base_status or "Live data: idle"
        self.live_status_label.setText(self._compact_live_status(full_status))
        self.live_status_label.setToolTip(full_status)

        updated_txt = str(current_row.get("updated_utc") or "-").strip()
        if hasattr(self, "conditions_updated_label"):
            self.conditions_updated_label.setText(f"UTC {self._compact_weather_update(updated_txt)}")
        forecast_rows: list[str] = []
        conditions_ok = 0
        conditions_err = 0
        forecast_ok = 0
        forecast_err = 0
        for key, row in providers.items():
            if not isinstance(row, dict):
                continue
            categories = row.get("categories") if isinstance(row.get("categories"), list) else []
            status = str(row.get("status") or "-").strip().lower()
            note = self._short_provider_note(row.get("note"), 54)
            label = str(row.get("label") or key)
            if "forecast" in categories:
                if status in {"ok", "partial"}:
                    forecast_ok += 1
                elif status == "error":
                    forecast_err += 1
                forecast_rows.append(f"{label}: {status}{f' ({note})' if note else ''}")
            if WeatherLiveWorker._is_measured_conditions_provider(str(key), row):
                if status in {"ok", "partial"}:
                    conditions_ok += 1
                elif status == "error":
                    conditions_err += 1
        if hasattr(self, "conditions_summary_label"):
            cond_status = sections.get("conditions") if isinstance(sections.get("conditions"), dict) else {}
            cond_msg = str(cond_status.get("message") or "").strip()
            if cond_msg:
                compact_msg = (
                    cond_msg.replace("Measured conditions providers", "")
                    .replace("Conditions providers", "")
                    .replace("providers", "")
                    .strip(" .:")
                )
                self.conditions_summary_label.setText(compact_msg or cond_msg)
                self.conditions_summary_label.setToolTip(cond_msg)
            else:
                summary_text = f"ok {conditions_ok} · err {conditions_err}"
                self.conditions_summary_label.setText(summary_text)
                self.conditions_summary_label.setToolTip(summary_text)
        if hasattr(self, "forecast_summary_label"):
            forecast_status = sections.get("forecast") if isinstance(sections.get("forecast"), dict) else {}
            forecast_msg = str(forecast_status.get("message") or "").strip()
            if forecast_msg:
                self.forecast_summary_label.setText(f"Forecast: {forecast_msg}")
            else:
                self.forecast_summary_label.setText(f"Forecast status: ok {forecast_ok}, errors {forecast_err}.")
        if hasattr(self, "forecast_provider_details"):
            details = " | ".join(forecast_rows[:3]) if forecast_rows else "-"
            if len(forecast_rows) > 3:
                details = f"{details} | … +{len(forecast_rows) - 3} more"
            self.forecast_provider_details.setText(details)

        if hasattr(self, "cloud_live_label"):
            self.cloud_live_label.setText(
                f"Live cloud {self._fmt_value(_safe_float(current_row.get('cloud_pct')), '%')}"
            )
        if hasattr(self, "cloud_annual_label"):
            annual = _safe_float(payload.get("annual_cloud_pct"))
            if annual is None:
                self.cloud_annual_label.setText("Annual avg -")
            else:
                self.cloud_annual_label.setText(f"Annual avg {annual:.1f}%")

        # Conditions trend chart
        selected_series = self._series_for_conditions_source(source_key, series_map)
        parent = self.parent()
        dark = bool(getattr(parent, "settings", None).value("general/darkMode", DEFAULT_DARK_MODE, type=bool)) if hasattr(parent, "settings") else DEFAULT_DARK_MODE

        # Forecast chart
        if hasattr(self, "forecast_source_combo"):
            forecast_key = str(self.forecast_source_combo.currentData() or "open_meteo")
            resolved_key, forecast_series = self._series_for_forecast_source(forecast_key, series_map)
            if resolved_key and resolved_key != forecast_key:
                idx = self.forecast_source_combo.findData(resolved_key)
                if idx >= 0:
                    blocker = QSignalBlocker(self.forecast_source_combo)
                    self.forecast_source_combo.setCurrentIndex(idx)
                    del blocker
                    forecast_key = resolved_key
            if self._series_has_points(forecast_series):
                html = self._build_plotly_html(
                    forecast_series,
                    f"Forecast: {forecast_key}",
                    max_points=72,
                    dark=dark,
                    theme_tokens=getattr(self, "_theme_tokens", None),
                )
                if html and hasattr(self, "forecast_web") and self.forecast_web is not None:
                    self._set_plot_html(self.forecast_web, html)
                elif hasattr(self, "forecast_canvas") and self.forecast_canvas is not None:
                    self._render_series_on_canvas(
                        self.forecast_figure,
                        self.forecast_canvas,
                        forecast_series,
                        title=f"Forecast: {forecast_key}",
                        max_points=72,
                        dark=dark,
                        theme_tokens=getattr(self, "_theme_tokens", None),
                    )
                elif hasattr(self, "forecast_image_label"):
                    chart_bytes = self._fallback_plot_bytes(
                        forecast_series,
                        f"Forecast: {forecast_key}",
                        max_points=72,
                        dark=dark,
                        theme_tokens=getattr(self, "_theme_tokens", None),
                    )
                    pix = QPixmap()
                    if chart_bytes and pix.loadFromData(chart_bytes):
                        self.forecast_image_label.setText("")
                        self.forecast_image_label.setPixmap(
                            pix.scaled(
                                self.forecast_image_label.size(),
                                Qt.KeepAspectRatio,
                                Qt.SmoothTransformation,
                            )
                        )
                    else:
                        self.forecast_image_label.setPixmap(QPixmap())
                        self.forecast_image_label.setText("Unable to render forecast chart.")
            elif hasattr(self, "forecast_canvas") and self.forecast_canvas is not None:
                self._render_series_on_canvas(
                    self.forecast_figure,
                    self.forecast_canvas,
                    {},
                    title="Forecast",
                    max_points=72,
                    dark=dark,
                    theme_tokens=getattr(self, "_theme_tokens", None),
                )
            elif hasattr(self, "forecast_image_label"):
                self.forecast_image_label.setPixmap(QPixmap())
                self.forecast_image_label.setText("No forecast data available yet.")

        trimmed_series = self._trim_series_for_recent(selected_series, max_points=12)
        if self._series_has_points(trimmed_series):
            html = self._build_plotly_html(
                trimmed_series,
                f"Conditions trend: {source_key}",
                max_points=12,
                dark=dark,
                theme_tokens=getattr(self, "_theme_tokens", None),
                show_legend=True,
            )
            if html and hasattr(self, "conditions_trend_web") and self.conditions_trend_web is not None:
                self._set_plot_html(self.conditions_trend_web, html)
            elif hasattr(self, "conditions_trend_web") and self.conditions_trend_web is not None:
                self._weather_plot_loaded["conditions"] = False
                self._set_weather_plot_loading("conditions", "Unable to render interactive chart.", visible=True)
            elif hasattr(self, "conditions_trend_canvas") and self.conditions_trend_canvas is not None:
                self._render_series_on_canvas(
                    self.conditions_trend_figure,
                    self.conditions_trend_canvas,
                    trimmed_series,
                    title=f"Conditions trend: {source_key}",
                    max_points=18,
                    dark=dark,
                    theme_tokens=getattr(self, "_theme_tokens", None),
                    show_legend=True,
                )
            elif hasattr(self, "conditions_trend_image_label"):
                chart_bytes = self._fallback_plot_bytes(
                    trimmed_series,
                    f"Conditions trend: {source_key}",
                    max_points=12,
                    dark=dark,
                    theme_tokens=getattr(self, "_theme_tokens", None),
                    show_legend=True,
                )
                pix = QPixmap()
                if chart_bytes and pix.loadFromData(chart_bytes):
                    self._conditions_raw_pixmap = pix
                    self.conditions_trend_image_label.setText("")
                    self._rescale_preview_pixmaps()
                else:
                    self._conditions_raw_pixmap = QPixmap()
                    self.conditions_trend_image_label.setPixmap(QPixmap())
                    self.conditions_trend_image_label.setText("Unable to render conditions trend.")
        elif hasattr(self, "conditions_trend_canvas") and self.conditions_trend_canvas is not None:
            self._render_series_on_canvas(
                self.conditions_trend_figure,
                self.conditions_trend_canvas,
                {},
                title="Conditions trend",
                max_points=18,
                dark=dark,
                theme_tokens=getattr(self, "_theme_tokens", None),
            )
        elif hasattr(self, "conditions_trend_web") and self.conditions_trend_web is not None:
            self._weather_plot_loaded["conditions"] = False
            self._set_weather_plot_loading("conditions", "No conditions trend data for selected source.", visible=True)
        elif hasattr(self, "conditions_trend_image_label"):
            self._conditions_raw_pixmap = QPixmap()
            self.conditions_trend_image_label.setPixmap(QPixmap())
            self.conditions_trend_image_label.setText("No conditions trend data for selected source.")

        # Meteogram chart
        if hasattr(self, "meteogram_source_combo"):
            meteo_key = str(self.meteogram_source_combo.currentData() or "open_meteo")
            meteo_series = self._series_for_source(meteo_key, series_map)
            if self._series_has_points(meteo_series):
                html = self._build_plotly_html(
                    meteo_series,
                    f"Meteogram: {meteo_key}",
                    max_points=96,
                    dark=dark,
                    theme_tokens=getattr(self, "_theme_tokens", None),
                    show_legend=True,
                )
                if html and hasattr(self, "meteogram_web") and self.meteogram_web is not None:
                    self._set_plot_html(self.meteogram_web, html)
                elif hasattr(self, "meteogram_web") and self.meteogram_web is not None:
                    self._weather_plot_loaded["meteogram"] = False
                    self._set_weather_plot_loading("meteogram", "Unable to render interactive chart.", visible=True)
                elif hasattr(self, "meteogram_canvas") and self.meteogram_canvas is not None:
                    self._render_series_on_canvas(
                        self.meteogram_figure,
                        self.meteogram_canvas,
                        meteo_series,
                        title=f"Meteogram: {meteo_key}",
                        max_points=96,
                        dark=dark,
                        theme_tokens=getattr(self, "_theme_tokens", None),
                        show_legend=True,
                    )
                elif hasattr(self, "meteogram_image_label"):
                    chart_bytes = self._fallback_plot_bytes(
                        meteo_series,
                        f"Meteogram: {meteo_key}",
                        max_points=96,
                        dark=dark,
                        theme_tokens=getattr(self, "_theme_tokens", None),
                        show_legend=True,
                    )
                    pix = QPixmap()
                    if chart_bytes and pix.loadFromData(chart_bytes):
                        self._meteogram_raw_pixmap = pix
                        self.meteogram_image_label.setText("")
                        self._rescale_preview_pixmaps()
                    else:
                        self._meteogram_raw_pixmap = QPixmap()
                        self.meteogram_image_label.setPixmap(QPixmap())
                        self.meteogram_image_label.setText("Unable to render meteogram.")
            elif hasattr(self, "meteogram_canvas") and self.meteogram_canvas is not None:
                self._render_series_on_canvas(
                    self.meteogram_figure,
                    self.meteogram_canvas,
                    {},
                    title="Meteogram",
                    max_points=96,
                    dark=dark,
                )
            elif hasattr(self, "meteogram_web") and self.meteogram_web is not None:
                self._weather_plot_loaded["meteogram"] = False
                self._set_weather_plot_loading("meteogram", "No meteogram data for selected source.", visible=True)
            elif hasattr(self, "meteogram_image_label"):
                self._meteogram_raw_pixmap = QPixmap()
                self.meteogram_image_label.setPixmap(QPixmap())
                self.meteogram_image_label.setText("No meteogram data for selected source.")

        # Cloud map
        if hasattr(self, "cloud_map_image_label"):
            cloud_map_obj = payload.get("cloud_map")
            if isinstance(cloud_map_obj, dict):
                cm_bytes = cloud_map_obj.get("image_bytes")
                cm_caption = str(cloud_map_obj.get("caption", "-")).strip()
                cm_url = str(cloud_map_obj.get("url", "")).strip()
                approx = _safe_float(cloud_map_obj.get("approx_cloud_pct"))
                month_name = str(cloud_map_obj.get("month_name") or "").strip()
                pix = QPixmap()
                if isinstance(cm_bytes, (bytes, bytearray)) and cm_bytes and pix.loadFromData(bytes(cm_bytes)):
                    self._cloud_map_raw_pixmap = pix
                    self.cloud_map_image_label.setText("")
                    self._set_weather_image_loading("cloud_map", "", visible=False)
                    self._rescale_preview_pixmaps()
                    info_parts: list[str] = []
                    if month_name:
                        info_parts.append(month_name)
                    if approx is not None:
                        info_parts.append(f"marker cloud {approx:.1f}%")
                    if cm_url:
                        info_parts.append("EarthEnv")
                    info_text = " | ".join(info_parts) if info_parts else "Monthly cloud map loaded."
                    self.cloud_map_info_label.setText(info_text)
                    if hasattr(self, "cloud_map_source_label"):
                        source_name = "EarthEnv" if cm_url else "Source -"
                        self.cloud_map_source_label.setText(source_name)
                elif cm_caption:
                    self._set_weather_image_loading("cloud_map", "", visible=False)
                    if self._cloud_map_raw_pixmap.isNull():
                        self.cloud_map_image_label.setPixmap(QPixmap())
                        self.cloud_map_image_label.setText("Monthly cloud map unavailable.")
                    self.cloud_map_info_label.setText(cm_caption)
                    if hasattr(self, "cloud_map_source_label"):
                        self.cloud_map_source_label.setText("Source -")

        # Satellite preview
        if hasattr(self, "satellite_image_label"):
            sat_obj = payload.get("satellite")
            if isinstance(sat_obj, dict):
                sat_bytes = sat_obj.get("image_bytes")
                sat_caption = str(sat_obj.get("caption", "-")).strip()
                sat_url = str(sat_obj.get("url", "")).strip()
                pix = QPixmap()
                if isinstance(sat_bytes, (bytes, bytearray)) and sat_bytes and pix.loadFromData(bytes(sat_bytes)):
                    self._satellite_raw_pixmap = pix
                    self.satellite_image_label.setText("")
                    self._set_weather_image_loading("satellite", "", visible=False)
                    self._rescale_preview_pixmaps()
                    tip_parts = [part for part in (sat_caption, sat_url) if part]
                    self.satellite_image_label.setToolTip("\n".join(tip_parts))
                elif sat_caption:
                    self._set_weather_image_loading("satellite", "", visible=False)
                    if self._satellite_raw_pixmap.isNull():
                        self.satellite_image_label.setPixmap(QPixmap())
                        self.satellite_image_label.setText("Satellite preview unavailable.")
                    self.satellite_image_label.setToolTip(sat_caption or "")
        localize_widget_tree(self, current_language())

    @Slot()
    def _on_cloud_month_changed(self) -> None:
        self._start_live_refresh(force=False, include_cloud_map=True, include_satellite=False)

    @Slot(int)
    def _on_conditions_source_combo_changed(self, _idx: int) -> None:
        if not hasattr(self, "conditions_source_combo"):
            return
        selected = self.conditions_source_combo.currentData()
        main_idx = self.live_source_combo.findData(selected)
        if main_idx < 0 or self.live_source_combo.currentIndex() == main_idx:
            return
        blocker = QSignalBlocker(self.live_source_combo)
        self.live_source_combo.setCurrentIndex(main_idx)
        del blocker
        self._update_live_views()

    @Slot()
    def _sync_conditions_source_combo(self) -> None:
        if not hasattr(self, "conditions_source_combo"):
            return
        selected = self.live_source_combo.currentData()
        idx = self.conditions_source_combo.findData(selected)
        if idx < 0:
            return
        blocker = QSignalBlocker(self.conditions_source_combo)
        self.conditions_source_combo.setCurrentIndex(idx)
        del blocker

    @staticmethod
    def _parse_weather_datetime(value: str) -> Optional[datetime]:
        text = str(value or "").strip()
        if not text or text == "-":
            return None
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            pass
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                continue
        return None

    @staticmethod
    def _weather_chip_markup(prefix: str, value: str) -> str:
        prefix_text = str(prefix or "").strip()
        value_text = str(value or "-").strip() or "-"
        return f"{prefix_text} {value_text}".strip()

    def _set_weather_chip_label(
        self,
        label: QLabel,
        prefix: str,
        value: str,
        *,
        tooltip: Optional[str] = None,
    ) -> None:
        value_text = str(value or "-").strip() or "-"
        set_translated_text(label, self._weather_chip_markup(prefix, value_text), current_language())
        set_translated_tooltip(label, str(tooltip or f"{prefix} {value_text}").strip(), current_language())

    def _update_summary_chip_widths(self) -> None:
        chip_font = QFont(self.local_time_label.font())
        value_font = QFont(chip_font)
        value_font.setBold(True)
        chip_metrics = QFontMetrics(chip_font)
        value_metrics = QFontMetrics(value_font)
        horizontal_padding = 26

        # Live condition chips (ensure values are visible, not clipped).
        live_min_widths = (
            chip_metrics.horizontalAdvance("T") + value_metrics.horizontalAdvance("88.8°C") + horizontal_padding,
            chip_metrics.horizontalAdvance("W") + value_metrics.horizontalAdvance("88.8 m/s") + horizontal_padding,
            chip_metrics.horizontalAdvance("C") + value_metrics.horizontalAdvance("100%") + horizontal_padding,
            chip_metrics.horizontalAdvance("RH") + value_metrics.horizontalAdvance("100%") + horizontal_padding,
            chip_metrics.horizontalAdvance("P") + value_metrics.horizontalAdvance("1088.8 hPa") + horizontal_padding,
        )
        for lbl, width in zip(
            (
                self.live_temp_label,
                self.live_wind_label,
                self.live_cloud_label,
                self.live_rh_label,
                self.live_pressure_label,
            ),
            live_min_widths,
        ):
            lbl.setMinimumWidth(max(int(width), 84))

        local_width = max(
            chip_metrics.horizontalAdvance("L") + value_metrics.horizontalAdvance("88:88:88") + horizontal_padding,
            118,
        )
        utc_width = max(
            chip_metrics.horizontalAdvance("U") + value_metrics.horizontalAdvance("88:88:88") + horizontal_padding,
            118,
        )
        phase_width = max(
            chip_metrics.horizontalAdvance("Phase")
            + value_metrics.horizontalAdvance("100%")
            + horizontal_padding
            + 6,
            132,
        )
        self.local_time_label.setFixedWidth(local_width)
        self.utc_time_label.setFixedWidth(utc_width)
        self.moon_phase_info_bar.setFixedWidth(phase_width)

    def _compact_weather_time(self, value: str) -> str:
        dt = self._parse_weather_datetime(value)
        if dt is None:
            return "-"
        return dt.strftime('%H:%M:%S')

    def _compact_weather_event(self, value: str) -> str:
        dt = self._parse_weather_datetime(value)
        if dt is None:
            return "-"
        obs_date = self._date.toPython() if isinstance(self._date, QDate) and self._date.isValid() else date.today()
        delta_days = (dt.date() - obs_date).days
        suffix = f" +{delta_days}d" if delta_days > 0 else (f" {delta_days}d" if delta_days < 0 else "")
        return f"{suffix.strip()} {dt.strftime('%H:%M')}".strip()

    def _compact_weather_update(self, value: str) -> str:
        dt = self._parse_weather_datetime(value)
        if dt is None:
            text = str(value or "").strip()
            return text or "-"
        return dt.strftime("%H:%M:%S")

    @staticmethod
    def _compact_live_status(text: str) -> str:
        value = str(text or "").strip()
        if not value:
            return "Idle"
        lowered = value.lower()
        if "average values from available measured sources" in lowered:
            return "Avg measurements"
        if "average values from available conditions providers" in lowered:
            return "Avg measurements"
        if "loading live weather providers" in lowered:
            return "Loading sources…"
        if "refreshing weather providers" in lowered:
            return "Refreshing…"
        if "live data: idle" in lowered:
            return "Idle"
        if "partial errors" in lowered:
            match = re.search(r"partial errors:\s*(\d+)", lowered)
            if match:
                return f"Partial errors: {match.group(1)}"
        return value[:40].rstrip() + "…" if len(value) > 41 else value

    def _apply_summary_labels(self) -> None:
        lat, lon, _ = self._site_coordinates()
        self._update_summary_chip_widths()
        self.obs_label.setText(f"Obs: {self._obs_name}")
        self.coords_label.setText(f"Lat/Lon: {lat:.5f}, {lon:.5f}")
        self.sun_limit_label.setText(f"Sun ≤ {self._sun_alt_limit:.0f}°")
        self._set_weather_chip_label(self.date_chip_label, "Date", self._date.toString('yyyy-MM-dd'))
        self._set_weather_chip_label(self.local_time_label, "L", self._compact_weather_time(self._local_time_text))
        self._set_weather_chip_label(self.utc_time_label, "U", self._compact_weather_time(self._utc_time_text))
        self._set_weather_chip_label(self.sunrise_info_label, "Sun↑", self._compact_weather_event(self._sunrise_text))
        self._set_weather_chip_label(self.sunset_info_label, "Sun↓", self._compact_weather_event(self._sunset_text))
        self._set_weather_chip_label(self.moonrise_info_label, "Moon↑", self._compact_weather_event(self._moonrise_text))
        self._set_weather_chip_label(self.moonset_info_label, "Moon↓", self._compact_weather_event(self._moonset_text))
        self.moon_phase_info_bar.setValue(self._moon_phase_pct)
        self.moon_phase_info_bar.setFormat(f"Phase {self._moon_phase_pct}%")
        self.moon_phase_info_bar.setToolTip(f"Moon phase {self._moon_phase_pct}%")
        localize_widget_tree(self, current_language())

    def _rebuild(self) -> None:
        self._apply_summary_labels()

        while self.tabs.count() > 0:
            widget = self.tabs.widget(0)
            self.tabs.removeTab(0)
            if widget is not None:
                widget.deleteLater()

        sources = self._source_urls()
        self.tabs.addTab(self._build_meteogram_tab(sources["forecast"]), "Meteograms")
        self.conditions_tab_widget = self._build_conditions_tab(sources["conditions"])
        self.tabs.addTab(self.conditions_tab_widget, "Conditions")
        self.tabs.addTab(self._build_cloud_analysis_tab(sources["cloud"]), "Cloud Analysis")
        self.tabs.addTab(self._build_satellite_tab(sources["satellite"]), "Satellite")

        if hasattr(self, "cloud_month_combo"):
            default_month = self._default_cloud_month()
            idx = self.cloud_month_combo.findData(int(default_month))
            if idx >= 0:
                blocker = QSignalBlocker(self.cloud_month_combo)
                self.cloud_month_combo.setCurrentIndex(idx)
                del blocker

        idx = self.live_source_combo.findData(self._weather_default_source)
        if idx >= 0:
            blocker = QSignalBlocker(self.live_source_combo)
            self.live_source_combo.setCurrentIndex(idx)
            del blocker
        self._sync_conditions_source_combo()

        self._on_tab_changed(self.tabs.currentIndex())
        self._update_live_views()
        self._start_live_refresh(force=False, include_cloud_map=True, include_satellite=True)
        localize_widget_tree(self, current_language())

__all__ = ["WeatherDialog"]
