"""
Astronomical Observation Planner GUI
-----------------------------------
Python 3.12 | PySide6

This is a simple GUI application for planning astronomical observations. It allows you to:
- Load celestial targets from a JSON file.
- Select an observation site and date.
- Calculate visibility curves for each target.
- Plot the results using Matplotlib.
- Display current altitudes, azimuths, and separations from the Moon.

Dependencies
===========
- Python 3.12+
- PySide6
- Astroplan
- Astropy
- Matplotlib
- PyEphem
- Pydantic
- Astroquery
- TimezoneFinder
- NumPy

Run
===
```bash
python astro_planner.py
```

A tiny sample JSON file with targets is included below.  Save it as `example_targets.json` and load it.
```
[
    {"name": "M31", "ra": 10.684, "dec": 41.269},
    {"name": "Sirius", "ra": 6.752, "dec": -16.716},
    {"name": "Betelgeuse", "ra": 88.792939, "dec": 7.407064}
]
```
"""

# --- Imports --------------------------------------
from __future__ import annotations

# Standard library imports
import argparse
import copy
import csv
import hashlib
import html as html_module
import io
import json
import locale
import math
import os
import re
import sys
import threading
from urllib.parse import quote
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, List, Optional
from time import perf_counter

import numpy as np
import pytz
import ephem
from astropy import units as u
from astropy.coordinates import SkyCoord, Angle
from astropy.time import Time
from astroplan import FixedTarget, Observer
from astroplan.observer import TargetAlwaysUpWarning

from astroquery.simbad import Simbad
try:
    from astroquery.exceptions import NoResultsWarning
except Exception:  # pragma: no cover - fallback only for older astroquery variants
    class NoResultsWarning(Warning):
        pass
from pydantic import ValidationError
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import warnings
import logging
import shiboken6 as shb   # PySide6 helper (isValid,objId)
from astroplanner.ai import (
    AIIntent,
    AI_CHAT_SPACING_CHOICES,
    AI_CHAT_TINT_CHOICES,
    AI_CHAT_WIDTH_CHOICES,
    ClassQuerySpec,
    CompareQuerySpec,
    KNOWLEDGE_DIR,
    KnowledgeNote,
    LLMConfig,
    LLMModelDiscoveryWorker,
    LLMWorker,
    ObjectQuerySpec,
    _format_knowledge_note_snippet,
    _knowledge_note_family,
    _load_knowledge_note,
    _looks_like_object_class_query,
    _looks_like_object_scoped_query,
    _looks_like_observing_guidance_query,
    _mentions_supernova_term,
    _normalize_knowledge_tag,
    _question_action_flags,
    _question_bhtom_type_markers,
    _requested_marker_family,
    _requested_marker_label,
    _requested_object_class_marker,
    _truncate_ai_memory_text,
    _type_label_class_family,
    _type_matches_requested_class,
)
from astroplanner.ai_context import AIContextCoordinator
from astroplanner.ai_panel_coordinator import AIPanelCoordinator
from astroplanner.app_config import (
    APP_SETTINGS_APP,
    APP_SETTINGS_ENV_KEY,
    APP_SETTINGS_ORG,
    OBSOLETE_APP_SETTINGS_KEYS,
    OBSOLETE_APP_STATE_KEYS,
    _cleanup_obsolete_settings,
    _config_root_dir,
    _create_app_settings,
    _resolve_settings_dir,
    _settings_dir_env_override,
    _settings_dir_for_org,
)
from astroplanner.exporters import export_metrics_csv, export_observation_ics
from astroplanner.bhtom import (
    BHTOM_SUGGESTION_MIN_IMPORTANCE,
    BhtomCandidatePrefetchWorker,
    BhtomObservatoryPresetWorker,
    BhtomSuggestionWorker,
    _bhtom_suggestion_source_message,
)
from astroplanner.astronomy import AstronomyWorker, ClockWorker
from astroplanner.models import (
    CalcRunStats,
    DEFAULT_LIMITING_MAGNITUDE,
    SessionSettings,
    Site,
    Target,
    targets_match as _targets_match,
)
from astroplanner.parsing import parse_dec_to_deg, parse_ra_dec_query, parse_ra_to_deg
from astroplanner.scoring import TargetNightMetrics
from astroplanner.seestar import (
    SEESTAR_ALP_LP_FILTER_AUTO,
    SEESTAR_ALP_LP_FILTER_OFF,
    SEESTAR_ALP_LP_FILTER_ON,
    SEESTAR_ALP_DEFAULT_BASE_URL,
    SEESTAR_ALP_DEFAULT_CLIENT_ID,
    SEESTAR_ALP_DEFAULT_DEVICE_NUM,
    SEESTAR_ALP_DEFAULT_GAIN,
    SEESTAR_ALP_DEFAULT_HONOR_QUEUE_TIMES,
    SEESTAR_ALP_DEFAULT_WAIT_UNTIL_LOCAL_TIME,
    SEESTAR_ALP_DEFAULT_STARTUP_SEQUENCE,
    SEESTAR_ALP_DEFAULT_STARTUP_POLAR_ALIGN,
    SEESTAR_ALP_DEFAULT_STARTUP_AUTO_FOCUS,
    SEESTAR_ALP_DEFAULT_STARTUP_DARK_FRAMES,
    SEESTAR_ALP_DEFAULT_CAPTURE_FLATS,
    SEESTAR_ALP_DEFAULT_FLATS_WAIT_S,
    SEESTAR_ALP_AF_MODE_OFF,
    SEESTAR_ALP_AF_MODE_PER_RUN,
    SEESTAR_ALP_AF_MODE_PER_TARGET,
    SEESTAR_ALP_DEFAULT_SCHEDULE_AUTOFOCUS_MODE,
    SEESTAR_ALP_DEFAULT_SCHEDULE_AUTOFOCUS,
    SEESTAR_ALP_DEFAULT_AUTOFOCUS_TRY_COUNT,
    SEESTAR_ALP_DEFAULT_DEW_HEATER_VALUE,
    SEESTAR_ALP_DEFAULT_PARK_AFTER_SESSION,
    SEESTAR_ALP_DEFAULT_SHUTDOWN_AFTER_SESSION,
    SEESTAR_ALP_DEFAULT_NUM_TRIES,
    SEESTAR_ALP_DEFAULT_PANEL_OVERLAP_PERCENT,
    SEESTAR_ALP_DEFAULT_RETRY_WAIT_S,
    SEESTAR_ALP_DEFAULT_STACK_EXPOSURE_MS,
    SEESTAR_ALP_DEFAULT_TIMEOUT_S,
    SEESTAR_ALP_DEFAULT_USE_AUTOFOCUS,
    SEESTAR_CAMPAIGN_PRESET_BL_LAC,
    SEESTAR_DEFAULT_BLOCK_MINUTES,
    SEESTAR_DEVICE_PROFILE_ALP,
    SEESTAR_DEVICE_PROFILE,
    SEESTAR_METHOD_ALP,
    SEESTAR_METHOD_GUIDED,
    SEESTAR_TEMPLATE_SCOPE_MULTI_TARGET,
    SEESTAR_TEMPLATE_SCOPE_SINGLE_TARGET,
    NightQueueSiteSnapshot,
    SeestarAlpAdapter,
    SeestarSessionTemplate,
    build_alp_web_ui_url,
    build_session_queue,
    default_science_checklist_items,
    normalize_seestar_alp_schedule_autofocus_mode,
    render_alp_backend_status_text,
    SeestarAlpClient,
    SeestarAlpConfig,
    SeestarGuidedAdapter,
)
from astroplanner.theme import (
    DEFAULT_DARK_MODE,
    DEFAULT_UI_THEME,
    THEME_CHOICES,
    build_stylesheet,
    highlight_palette_for_theme,
    normalize_theme_key,
    resolve_theme_tokens,
)
from astroplanner.i18n import (
    SUPPORTED_UI_LANGUAGES,
    current_language,
    localize_widget_tree,
    resolve_language_choice,
    set_current_language,
    set_translated_text,
    set_translated_tooltip,
    translate_text,
)
from astroplanner.bhtom_coordinator import BhtomCoordinator
from astroplanner.main_window_coordinator import MainWindowCoordinator
from astroplanner.observatory_coordinator import ObservatoryCoordinator
from astroplanner.plan_coordinator import PlanCoordinator
from astroplanner.resolvers import (
    MetadataLookupWorker,
    SIMBAD_COMPACT_CACHE_TTL_S,
    SIMBAD_COMPACT_NEGATIVE_CACHE_TTL_S,
    TARGET_SEARCH_SOURCES,
    TARGET_SOURCE_LABELS,
    TNS_ENDPOINT_BASE_URLS,
    TNS_ENDPOINT_CHOICES,
    TargetResolver,
    _airmass_from_altitude_values,
    _build_tns_marker,
    _decode_simbad_value,
    _extract_simbad_compact_measurements,
    _extract_simbad_metadata,
    _extract_simbad_name,
    _extract_simbad_photometry,
    _normalize_catalog_display_name,
    _normalize_catalog_token,
    _normalize_tns_endpoint_key,
    _object_type_is_unknown,
    _safe_float,
    _safe_int,
    _simbad_best_row_index,
    _simbad_cell,
    _simbad_column,
    _simbad_has_row,
    _simbad_row_coord,
    _target_magnitude_label,
    _target_source_label,
    _tns_api_base_url,
)
from astroplanner.preview_coordinator import (
    CUTOUT_ALADIN_CONTEXT_STEP,
    CUTOUT_ALADIN_FETCH_MARGIN,
    CUTOUT_ALADIN_FETCH_MAX_EDGE_PX,
    CUTOUT_ALADIN_FETCH_MIN_ARCMIN,
    CUTOUT_ALADIN_FETCH_MIN_SHORT_PX,
    CUTOUT_ALADIN_FETCH_RES_MULT,
    CUTOUT_ALADIN_FETCH_TELESCOPE_MARGIN,
    CUTOUT_ALADIN_FETCH_TELESCOPE_MAX_ARCMIN,
    CUTOUT_CACHE_MAX,
    CUTOUT_DEFAULT_FOV_ARCMIN,
    CUTOUT_DEFAULT_SIZE_PX,
    CUTOUT_DEFAULT_SURVEY_KEY,
    CUTOUT_DEFAULT_VIEW_KEY,
    CUTOUT_MAX_FOV_ARCMIN,
    CUTOUT_MAX_SIZE_PX,
    CUTOUT_MIN_FOV_ARCMIN,
    CUTOUT_MIN_SIZE_PX,
    CUTOUT_SURVEY_CHOICES,
    CUTOUT_VIEW_CHOICES,
    FINDER_RETRY_COOLDOWN_S,
    FINDER_WORKER_TIMEOUT_MS,
    PreviewCoordinator,
    _cutout_survey_hips,
    _cutout_survey_label,
    _normalize_cutout_survey_key,
    _normalize_cutout_view_key,
    _sanitize_cutout_fov_arcmin,
    _sanitize_cutout_size_px,
)
from astroplanner.targets_coordinator import TargetTableCoordinator
from astroplanner.visibility_coordinator import VisibilityCoordinator
from astroplanner.visibility_matplotlib import VisibilityMatplotlibCoordinator
from astroplanner.ui.common import (
    SkeletonShimmerWidget,
    TargetTableView,
    _distribute_extra_table_width,
    _fit_dialog_to_screen,
)
from astroplanner.ui.add_target import AddTargetDialog, FinderChartWorker
from astroplanner.ui.observatory import AddObservatoryDialog, ObservatoryLookupWorker, ObservatoryManagerDialog
from astroplanner.ui.settings import GeneralSettingsDialog, TableSettingsDialog
from astroplanner.ui.seestar import SeestarSessionPlanDialog
from astroplanner.ui.suggestions import SuggestedTargetsDialog
from astroplanner.ui.targets import TargetTableModel
from astroplanner.ui.preview import (
    build_cutout_loading_placeholder,
    create_cutout_image_stack,
    set_cutout_image_loading,
)
from astroplanner.ui.theme_utils import (
    UI_FONT_SIZE_MAX,
    UI_FONT_SIZE_MIN,
    _embedded_display_font_css,
    _ensure_display_font_loaded,
    _normalized_css_color,
    _pick_font_family,
    _pick_matplotlib_font_family,
    _plot_font_css_stack,
    _preferred_display_font_family,
    _sanitize_ui_font_size,
    _swatch_text_color,
)
from astroplanner.ui.widgets import CoverImageLabel, NeonToggleSwitch, NoSelectBackgroundDelegate
from astroplanner.weather import WeatherLiveWorker
from astroplanner.ui.weather import WeatherDialog
from astroplanner.visibility_plotly import (
    HAS_PLOTLY as _HAS_PLOTLY,
    PLOTLY_JS_BASE_DIR as _PLOTLY_JS_BASE_DIR,
    VisibilityPlotlyRequest,
    build_visibility_plotly_html,
)

try:
    from PIL import Image, ImageDraw
except Exception:  # pragma: no cover - optional runtime dependency
    Image = None  # type: ignore[assignment]
    ImageDraw = None  # type: ignore[assignment]

# ── Logging configuration ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,                              # default level
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

QUICK_TARGETS_DEFAULT_COUNT = 10
QUICK_TARGETS_MIN_COUNT = 1
QUICK_TARGETS_MAX_COUNT = 50



# GUI imports (PySide6)
from PySide6.QtCore import (
    QDate,
    QElapsedTimer,
    QEasingCurve,
    QEvent,
    QModelIndex,
    QObject,
    QPropertyAnimation,
    QSignalBlocker,
    Qt,
    QThread,
    Signal,
    Slot,
    QSize,
    QTimer,
    QItemSelectionModel,
    QUrl,
    QLocale,
)
from PySide6.QtGui import (
    QAction,
    QActionGroup,
    QBrush,
    QColor,
    QDoubleValidator,
    QFont,
    QFontMetrics,
    QDesktopServices,
    QIcon,
    QKeySequence,
    QLinearGradient,
    QPalette,
    QPainter,
    QPen,
    QPixmap,
    QShortcut,
    QTextDocument,
)
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkReply
try:
    from PySide6.QtWebEngineWidgets import QWebEngineView

    _HAS_QTWEBENGINE = True
except Exception:  # pragma: no cover - optional runtime dependency
    QWebEngineView = None  # type: ignore[assignment]
    _HAS_QTWEBENGINE = False
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDoubleSpinBox,
    QFrame,
    QFormLayout,
    QGridLayout,
    QHeaderView,
    QFileDialog,
    QGraphicsDropShadowEffect,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QSplitter,
    QStackedLayout,
    QSpinBox,
    QSlider,
    QStyle,
    QSizePolicy,
    QTableView,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QColorDialog,
    QDialog,
    QDialogButtonBox,
    QTextEdit,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt import NavigationToolbar2QT as NavigationToolbar
_HAS_MPL_CANVAS = True

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


def _repolish_widget(widget: Optional[QWidget]) -> None:
    if widget is None:
        return
    style = widget.style()
    if style is None:
        widget.update()
        return
    style.unpolish(widget)
    style.polish(widget)
    widget.update()


def _set_dynamic_property(widget: Optional[QWidget], name: str, value: object) -> None:
    if widget is None:
        return
    widget.setProperty(name, value)


def _set_button_variant(button: Optional[QWidget], variant: str) -> None:
    _set_dynamic_property(button, "variant", str(variant))


def _set_label_tone(label: Optional[QWidget], tone: str) -> None:
    _set_dynamic_property(label, "tone", str(tone))


def _set_widget_invalid(widget: Optional[QWidget], invalid: bool) -> None:
    _set_dynamic_property(widget, "invalid", bool(invalid))


def _theme_tokens_from_widget(widget: object) -> dict[str, str]:
    current = widget
    for _ in range(6):
        tokens = getattr(current, "_theme_tokens", None)
        if isinstance(tokens, dict):
            return tokens
        parent = getattr(current, "parent", None)
        if not callable(parent):
            break
        current = parent()
        if current is None:
            break
    return {}


def _theme_color_from_widget(widget: object, key: str, fallback: str) -> str:
    tokens = _theme_tokens_from_widget(widget)
    value = tokens.get(key) if isinstance(tokens, dict) else None
    return str(value or fallback)


_CSS_RGBA_RE = re.compile(
    r"^rgba?\(\s*([0-9]{1,3})\s*,\s*([0-9]{1,3})\s*,\s*([0-9]{1,3})(?:\s*,\s*([0-9]*\.?[0-9]+))?\s*\)$",
    re.IGNORECASE,
)


def _parse_qcolor_token(value: object) -> QColor:
    if isinstance(value, QColor):
        color = QColor(value)
        return color if color.isValid() else QColor()

    text = str(value or "").strip()
    if not text:
        return QColor()

    color = QColor(text)
    if color.isValid():
        return color

    match = _CSS_RGBA_RE.match(text)
    if not match:
        return QColor()

    red, green, blue = (max(0, min(255, int(part))) for part in match.groups()[:3])
    alpha_token = match.group(4)
    if alpha_token is None or alpha_token == "":
        alpha = 255
    else:
        alpha_value = float(alpha_token)
        if alpha_value <= 1.0:
            alpha = int(round(max(0.0, min(1.0, alpha_value)) * 255.0))
        else:
            alpha = int(round(max(0.0, min(255.0, alpha_value))))
    return QColor(red, green, blue, alpha)


def _qcolor_from_token(value: object, fallback: object | None = None) -> QColor:
    color = _parse_qcolor_token(value)
    if color.isValid():
        return color
    if fallback is not None:
        fallback_color = _parse_qcolor_token(fallback)
        if fallback_color.isValid():
            return fallback_color
    return QColor()


def _theme_qcolor_from_widget(widget: object, key: str, fallback: str) -> QColor:
    color = _qcolor_from_token(_theme_color_from_widget(widget, key, fallback), fallback)
    if color.isValid():
        return color
    return _qcolor_from_token(fallback)


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


def _composite_qcolor(foreground: QColor, background: QColor) -> QColor:
    fg = QColor(foreground)
    bg = QColor(background)
    if not fg.isValid():
        return bg if bg.isValid() else QColor()
    if not bg.isValid() or fg.alpha() >= 255:
        return fg
    alpha = fg.alphaF()
    out = QColor(
        round(fg.red() * alpha + bg.red() * (1.0 - alpha)),
        round(fg.green() * alpha + bg.green() * (1.0 - alpha)),
        round(fg.blue() * alpha + bg.blue() * (1.0 - alpha)),
    )
    return out


def _contrast_text_for_background(background: QColor, *, table_bg: Optional[QColor] = None) -> QColor:
    bg = QColor(background)
    if not bg.isValid():
        return QColor("#f6fbff")
    if table_bg is not None and table_bg.isValid() and bg.alpha() < 255:
        bg = _composite_qcolor(bg, table_bg)
    luminance = (
        0.2126 * float(bg.red()) +
        0.7152 * float(bg.green()) +
        0.0722 * float(bg.blue())
    ) / 255.0
    return QColor("#061118" if luminance > 0.58 else "#f6fbff")


def _icon_pen(color: QColor, width: float) -> QPen:
    pen = QPen(color)
    pen.setWidthF(width)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    return pen


def _button_icon_palette(widget: object) -> tuple[QColor, QColor, QColor]:
    variant = ""
    if hasattr(widget, "property"):
        try:
            variant = str(widget.property("variant") or "")
        except Exception:
            variant = ""
    if variant == "primary":
        fg = _theme_qcolor_from_widget(widget, "btn_text", "#f7fbff")
    elif variant == "ghost":
        fg = _theme_qcolor_from_widget(widget, "strip_label", "#a4b4c8")
    else:
        fg = _theme_qcolor_from_widget(widget, "section_title", "#eef4fc")
    accent = _theme_qcolor_from_widget(widget, "accent_primary", "#59f3ff")
    accent_secondary = _theme_qcolor_from_widget(widget, "accent_secondary", "#ff4da6")
    return fg, accent, accent_secondary


def _build_button_icon_pixmap(kind: str, fg: QColor, accent: QColor, accent_secondary: QColor, size: int = 18) -> QPixmap:
    px = QPixmap(size, size)
    px.fill(Qt.GlobalColor.transparent)
    painter = QPainter(px)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    stroke = max(1.5, float(size) * 0.11)
    painter.setPen(_icon_pen(fg, stroke))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    kind = str(kind or "").strip().lower()
    s = float(size)
    m = s * 0.2

    def line(x1: float, y1: float, x2: float, y2: float) -> None:
        painter.drawLine(int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))

    def ellipse(x: float, y: float, w: float, h: float) -> None:
        painter.drawEllipse(int(round(x)), int(round(y)), int(round(w)), int(round(h)))

    def arc(x: float, y: float, w: float, h: float, start_deg: float, span_deg: float) -> None:
        painter.drawArc(
            int(round(x)),
            int(round(y)),
            int(round(w)),
            int(round(h)),
            int(round(start_deg * 16)),
            int(round(span_deg * 16)),
        )

    def dot(x: float, y: float, r: float = 1.7) -> None:
        painter.save()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(accent_secondary)
        painter.drawEllipse(int(round(x - r)), int(round(y - r)), int(round(r * 2)), int(round(r * 2)))
        painter.restore()

    if kind in {"add", "new"}:
        line(s * 0.5, m, s * 0.5, s - m)
        line(m, s * 0.5, s - m, s * 0.5)
        dot(s * 0.77, s * 0.23)
    elif kind in {"quick", "flash"}:
        line(s * 0.56, m, s * 0.33, s * 0.48)
        line(s * 0.33, s * 0.48, s * 0.50, s * 0.48)
        line(s * 0.50, s * 0.48, s * 0.42, s - m)
        line(s * 0.42, s - m, s * 0.68, s * 0.54)
        line(s * 0.68, s * 0.54, s * 0.52, s * 0.54)
    elif kind in {"suggest", "spark", "ai"}:
        line(s * 0.5, m, s * 0.5, s - m)
        line(m, s * 0.5, s - m, s * 0.5)
        line(s * 0.26, s * 0.26, s * 0.74, s * 0.74)
        line(s * 0.74, s * 0.26, s * 0.26, s * 0.74)
        dot(s * 0.76, s * 0.24)
    elif kind in {"toggle", "observed", "ok", "apply"}:
        line(s * 0.23, s * 0.56, s * 0.42, s * 0.74)
        line(s * 0.42, s * 0.74, s * 0.78, s * 0.30)
        dot(s * 0.76, s * 0.24)
    elif kind in {"cancel", "clear"}:
        line(m, m, s - m, s - m)
        line(s - m, m, m, s - m)
    elif kind in {"remove", "delete"}:
        line(m, s * 0.5, s - m, s * 0.5)
    elif kind in {"edit", "pencil"}:
        line(s * 0.28, s * 0.72, s * 0.72, s * 0.28)
        line(s * 0.62, s * 0.24, s * 0.76, s * 0.38)
        line(s * 0.24, s * 0.76, s * 0.38, s * 0.62)
        dot(s * 0.78, s * 0.22)
    elif kind in {"load", "open"}:
        line(m, s * 0.72, s - m, s * 0.72)
        line(m, s * 0.72, m, s * 0.58)
        line(s - m, s * 0.72, s - m, s * 0.58)
        line(s * 0.5, m, s * 0.5, s * 0.58)
        line(s * 0.5, s * 0.58, s * 0.34, s * 0.42)
        line(s * 0.5, s * 0.58, s * 0.66, s * 0.42)
    elif kind in {"save"}:
        painter.drawRoundedRect(int(round(m)), int(round(m)), int(round(s - 2 * m)), int(round(s - 2 * m)), 3, 3)
        line(s * 0.32, m + 1, s * 0.32, s * 0.45)
        line(s * 0.32, s * 0.45, s * 0.68, s * 0.45)
        line(s * 0.28, s * 0.67, s * 0.72, s * 0.67)
    elif kind in {"weather", "cloud"}:
        ellipse(s * 0.22, s * 0.42, s * 0.28, s * 0.22)
        ellipse(s * 0.40, s * 0.32, s * 0.30, s * 0.26)
        ellipse(s * 0.56, s * 0.44, s * 0.22, s * 0.18)
        line(s * 0.24, s * 0.59, s * 0.72, s * 0.59)
        dot(s * 0.76, s * 0.28)
    elif kind in {"lookup", "resolve"}:
        ellipse(s * 0.22, s * 0.22, s * 0.36, s * 0.36)
        line(s * 0.52, s * 0.52, s * 0.74, s * 0.74)
        line(s * 0.40, s * 0.28, s * 0.40, s * 0.52)
        line(s * 0.28, s * 0.40, s * 0.52, s * 0.40)
    elif kind in {"send"}:
        line(m, s * 0.54, s - m, m)
        line(s - m, m, s * 0.62, s - m)
        line(s * 0.62, s - m, s * 0.50, s * 0.56)
        line(s * 0.50, s * 0.56, m, s * 0.54)
    elif kind in {"seestar", "target"}:
        ellipse(s * 0.22, s * 0.22, s * 0.56, s * 0.56)
        line(s * 0.5, m, s * 0.5, s * 0.34)
        line(s * 0.5, s * 0.66, s * 0.5, s - m)
        line(m, s * 0.5, s * 0.34, s * 0.5)
        line(s * 0.66, s * 0.5, s - m, s * 0.5)
        dot(s * 0.5, s * 0.5, 1.8)
    elif kind in {"refresh"}:
        painter.drawArc(int(round(m)), int(round(m)), int(round(s - 2 * m)), int(round(s - 2 * m)), int(25 * 16), int(280 * 16))
        line(s * 0.72, s * 0.18, s * 0.82, s * 0.18)
        line(s * 0.82, s * 0.18, s * 0.82, s * 0.30)
    elif kind in {"clock", "localtime"}:
        ellipse(s * 0.18, s * 0.18, s * 0.64, s * 0.64)
        line(s * 0.50, s * 0.50, s * 0.50, s * 0.30)
        line(s * 0.50, s * 0.50, s * 0.66, s * 0.58)
        dot(s * 0.74, s * 0.24)
    elif kind in {"utc", "globe"}:
        ellipse(s * 0.18, s * 0.18, s * 0.64, s * 0.64)
        arc(s * 0.18, s * 0.18, s * 0.64, s * 0.64, 0, 180)
        arc(s * 0.18, s * 0.18, s * 0.64, s * 0.64, 180, 180)
        line(s * 0.50, s * 0.20, s * 0.50, s * 0.80)
        arc(s * 0.30, s * 0.18, s * 0.40, s * 0.64, 90, 180)
        arc(s * 0.30, s * 0.18, s * 0.40, s * 0.64, 270, 180)
    elif kind in {"sidereal", "star"}:
        line(s * 0.50, m, s * 0.50, s - m)
        line(m, s * 0.50, s - m, s * 0.50)
        line(s * 0.26, s * 0.26, s * 0.74, s * 0.74)
        line(s * 0.74, s * 0.26, s * 0.26, s * 0.74)
        dot(s * 0.78, s * 0.24)
    elif kind in {"sun", "sun_alt"}:
        ellipse(s * 0.31, s * 0.31, s * 0.38, s * 0.38)
        line(s * 0.50, m, s * 0.50, s * 0.20)
        line(s * 0.50, s * 0.80, s * 0.50, s - m)
        line(m, s * 0.50, s * 0.20, s * 0.50)
        line(s * 0.80, s * 0.50, s - m, s * 0.50)
        line(s * 0.25, s * 0.25, s * 0.34, s * 0.34)
        line(s * 0.66, s * 0.66, s * 0.75, s * 0.75)
        line(s * 0.66, s * 0.34, s * 0.75, s * 0.25)
        line(s * 0.25, s * 0.75, s * 0.34, s * 0.66)
    elif kind in {"moon", "moon_alt"}:
        arc(s * 0.18, s * 0.16, s * 0.52, s * 0.68, 58, 244)
        arc(s * 0.34, s * 0.16, s * 0.36, s * 0.68, 110, 140)
        dot(s * 0.78, s * 0.28)
    elif kind in {"sunrise"}:
        line(m, s * 0.70, s - m, s * 0.70)
        arc(s * 0.28, s * 0.40, s * 0.44, s * 0.44, 0, 180)
        line(s * 0.50, s * 0.18, s * 0.50, s * 0.34)
        line(s * 0.34, s * 0.30, s * 0.40, s * 0.38)
        line(s * 0.66, s * 0.30, s * 0.60, s * 0.38)
    elif kind in {"sunset"}:
        line(m, s * 0.70, s - m, s * 0.70)
        arc(s * 0.28, s * 0.48, s * 0.44, s * 0.40, 180, 180)
        line(s * 0.50, s * 0.22, s * 0.50, s * 0.38)
        line(s * 0.50, s * 0.38, s * 0.42, s * 0.30)
        line(s * 0.50, s * 0.38, s * 0.58, s * 0.30)
    elif kind in {"moonphase", "phase"}:
        ellipse(s * 0.22, s * 0.18, s * 0.56, s * 0.56)
        arc(s * 0.34, s * 0.18, s * 0.34, s * 0.56, 90, 180)
        dot(s * 0.76, s * 0.26)
    elif kind in {"moonrise"}:
        arc(s * 0.18, s * 0.18, s * 0.52, s * 0.64, 58, 244)
        arc(s * 0.34, s * 0.18, s * 0.36, s * 0.64, 110, 140)
        line(s * 0.72, s * 0.74, s * 0.72, s * 0.36)
        line(s * 0.72, s * 0.36, s * 0.64, s * 0.44)
        line(s * 0.72, s * 0.36, s * 0.80, s * 0.44)
    elif kind in {"moonset"}:
        arc(s * 0.18, s * 0.18, s * 0.52, s * 0.64, 58, 244)
        arc(s * 0.34, s * 0.18, s * 0.36, s * 0.64, 110, 140)
        line(s * 0.72, s * 0.30, s * 0.72, s * 0.68)
        line(s * 0.72, s * 0.68, s * 0.64, s * 0.60)
        line(s * 0.72, s * 0.68, s * 0.80, s * 0.60)
    elif kind in {"window", "best_window"}:
        line(s * 0.24, s * 0.28, s * 0.24, s * 0.72)
        line(s * 0.76, s * 0.28, s * 0.76, s * 0.72)
        line(s * 0.24, s * 0.28, s * 0.76, s * 0.28)
        line(s * 0.24, s * 0.72, s * 0.76, s * 0.72)
        line(s * 0.50, s * 0.34, s * 0.50, s * 0.50)
        line(s * 0.50, s * 0.50, s * 0.62, s * 0.58)
    elif kind in {"notes", "note"}:
        painter.drawRoundedRect(int(round(s * 0.24)), int(round(s * 0.18)), int(round(s * 0.50)), int(round(s * 0.62)), 3, 3)
        line(s * 0.34, s * 0.36, s * 0.64, s * 0.36)
        line(s * 0.34, s * 0.50, s * 0.64, s * 0.50)
        line(s * 0.34, s * 0.64, s * 0.56, s * 0.64)
    elif kind in {"link", "open-link"}:
        line(s * 0.28, s * 0.72, s * 0.72, s * 0.28)
        line(s * 0.50, s * 0.28, s * 0.72, s * 0.28)
        line(s * 0.72, s * 0.28, s * 0.72, s * 0.50)
        line(s * 0.30, s * 0.34, s * 0.30, s * 0.70)
        line(s * 0.30, s * 0.70, s * 0.66, s * 0.70)
    elif kind in {"describe"}:
        painter.drawRoundedRect(int(round(m)), int(round(s * 0.24)), int(round(s - 2 * m)), int(round(s * 0.40)), 4, 4)
        line(s * 0.34, s * 0.66, s * 0.46, s * 0.56)
        line(s * 0.46, s * 0.56, s * 0.56, s * 0.66)
        dot(s * 0.76, s * 0.30)
    else:
        ellipse(m, m, s - 2 * m, s - 2 * m)
        dot(s * 0.72, s * 0.28)

    painter.end()
    return px


def _build_button_icon(button: object, kind: str, size: int = 18) -> QIcon:
    fg, accent, accent_secondary = _button_icon_palette(button)
    disabled = _theme_qcolor_from_widget(button, "state_disabled", "#7f8ca3")
    icon = QIcon()
    icon.addPixmap(_build_button_icon_pixmap(kind, fg, accent, accent_secondary, size), QIcon.Mode.Normal, QIcon.State.Off)
    icon.addPixmap(_build_button_icon_pixmap(kind, fg, accent_secondary, accent, size), QIcon.Mode.Active, QIcon.State.Off)
    icon.addPixmap(_build_button_icon_pixmap(kind, disabled, disabled, disabled, size), QIcon.Mode.Disabled, QIcon.State.Off)
    return icon


def _set_button_icon_kind(button: Optional[QWidget], kind: str, size: int = 16) -> None:
    if button is None or not hasattr(button, "setIcon"):
        return
    button.setProperty("icon_kind", str(kind))
    button.setProperty("icon_size", int(size))
    button.setIcon(_build_button_icon(button, kind, size))
    if hasattr(button, "setIconSize"):
        button.setIconSize(QSize(int(size), int(size)))


def _refresh_button_icons(root: Optional[QWidget]) -> None:
    if root is None:
        return
    widgets = [root, *root.findChildren(QWidget)]
    for widget in widgets:
        if isinstance(widget, QLabel):
            kind = widget.property("detail_icon_kind")
            if kind:
                size = int(widget.property("detail_icon_size") or 16)
                fg = _theme_qcolor_from_widget(widget, "section_title", "#eef4fc")
                accent = _theme_qcolor_from_widget(widget, "accent_primary", "#59f3ff")
                accent_secondary = _theme_qcolor_from_widget(widget, "accent_secondary", "#ff4da6")
                widget.setPixmap(_build_button_icon_pixmap(str(kind), fg, accent, accent_secondary, size))
                widget.setFixedSize(size, size)
                widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
                continue
        if not hasattr(widget, "property") or not hasattr(widget, "setIcon"):
            continue
        kind = widget.property("icon_kind")
        if not kind:
            continue
        size = int(widget.property("icon_size") or 16)
        widget.setIcon(_build_button_icon(widget, str(kind), size))
        if hasattr(widget, "setIconSize"):
            widget.setIconSize(QSize(size, size))
    _refresh_button_hover_glow(root)


_BUTTON_HOVER_GLOW_FILTER: Optional[QObject] = None


class _ButtonHoverGlowFilter(QObject):
    def eventFilter(self, watched, event):  # noqa: D401
        if not isinstance(watched, QWidget):
            return False
        event_type = event.type()
        if event_type in (QEvent.Type.Enter, QEvent.Type.HoverEnter):
            _apply_button_hover_glow(watched, True)
        elif event_type in (
            QEvent.Type.Leave,
            QEvent.Type.HoverLeave,
            QEvent.Type.Hide,
            QEvent.Type.EnabledChange,
            QEvent.Type.PaletteChange,
            QEvent.Type.StyleChange,
        ):
            _apply_button_hover_glow(watched, watched.isEnabled() and watched.underMouse())
        return False


def _apply_button_hover_glow(widget: QWidget, enabled: bool) -> None:
    if not isinstance(widget, QPushButton):
        return
    if bool(widget.property("weather_link")):
        enabled = False
    current = widget.graphicsEffect()
    if not enabled or not widget.isEnabled():
        if isinstance(current, QGraphicsDropShadowEffect):
            widget.setGraphicsEffect(None)
        return
    color = _theme_qcolor_from_widget(widget, "button_hover_glow", "#ff7ecf")
    aura = _theme_qcolor_from_widget(widget, "button_hover_aura", "#ff7ecf")
    effect: QGraphicsDropShadowEffect
    if isinstance(current, QGraphicsDropShadowEffect):
        effect = current
    else:
        effect = QGraphicsDropShadowEffect(widget)
        effect.setOffset(0.0, 0.0)
        widget.setGraphicsEffect(effect)
    glow_color = QColor(aura if aura.isValid() else color)
    if not glow_color.isValid():
        glow_color = QColor("#ff7ecf")
    variant = str(widget.property("variant") or "")
    blur_radius = 36.0 if variant in {"primary", "secondary"} else 32.0
    alpha_floor = 210 if variant in {"primary", "secondary"} else 188
    if glow_color.alpha() < alpha_floor:
        glow_color.setAlpha(alpha_floor)
    glow_color = QColor(
        max(glow_color.red(), color.red()),
        max(glow_color.green(), color.green()),
        max(glow_color.blue(), color.blue()),
        glow_color.alpha(),
    )
    effect.setBlurRadius(blur_radius)
    effect.setColor(glow_color)


def _refresh_button_hover_glow(root: Optional[QWidget]) -> None:
    global _BUTTON_HOVER_GLOW_FILTER
    if root is None:
        return
    if _BUTTON_HOVER_GLOW_FILTER is None:
        _BUTTON_HOVER_GLOW_FILTER = _ButtonHoverGlowFilter(QApplication.instance())
    widgets = [root, *root.findChildren(QWidget)]
    for widget in widgets:
        if not isinstance(widget, QPushButton):
            continue
        if not bool(widget.property("_hover_glow_installed")):
            widget.installEventFilter(_BUTTON_HOVER_GLOW_FILTER)
            widget.setProperty("_hover_glow_installed", True)
        _apply_button_hover_glow(widget, widget.isEnabled() and widget.underMouse())


def _set_detail_icon_kind(label: Optional[QLabel], kind: str, size: int = 16) -> None:
    if label is None:
        return
    label.setProperty("detail_icon_kind", str(kind))
    label.setProperty("detail_icon_size", int(size))
    fg = _theme_qcolor_from_widget(label, "section_title", "#eef4fc")
    accent = _theme_qcolor_from_widget(label, "accent_primary", "#59f3ff")
    accent_secondary = _theme_qcolor_from_widget(label, "accent_secondary", "#ff4da6")
    label.setPixmap(_build_button_icon_pixmap(kind, fg, accent, accent_secondary, size))
    label.setFixedSize(size, size)
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)


_LEGACY_TABLE_HIGHLIGHT_DEFAULTS = {
    "below": "#ffe0ea",
    "limit": "#fff2cf",
    "above": "#d9ffe9",
}
_LEGACY_TABLE_HIGHLIGHT_COLORBLIND = {
    "below": "#ffb18e",
    "limit": "#ffe680",
    "above": "#9bf985",
}


def _canonical_color_token(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    color = _qcolor_from_token(text)
    if color.isValid():
        return color.name(QColor.NameFormat.HexArgb).lower()
    return re.sub(r"\s+", "", text.lower())


def _resolve_table_highlight_color(settings: object, key: str, default_color: str) -> str:
    if settings is None or not hasattr(settings, "value"):
        return str(default_color)
    raw = str(settings.value(f"table/color/{key}", default_color) or "").strip()
    if not raw:
        return str(default_color)
    normalized = _canonical_color_token(raw)
    auto_colors = {
        _canonical_color_token(_LEGACY_TABLE_HIGHLIGHT_DEFAULTS.get(key, "")),
        _canonical_color_token(_LEGACY_TABLE_HIGHLIGHT_COLORBLIND.get(key, "")),
    }
    for theme_key, _label in THEME_CHOICES:
        for dark_enabled in (False, True):
            palette = highlight_palette_for_theme(theme_key, dark_enabled=dark_enabled, color_blind=False)
            auto_colors.update({
                _canonical_color_token(palette.below),
                _canonical_color_token(palette.limit),
                _canonical_color_token(palette.above),
            })
    colorblind_palette = highlight_palette_for_theme(DEFAULT_UI_THEME, dark_enabled=False, color_blind=True)
    auto_colors.update({
        _canonical_color_token(colorblind_palette.below),
        _canonical_color_token(colorblind_palette.limit),
        _canonical_color_token(colorblind_palette.above),
    })
    if normalized in auto_colors:
        return str(default_color)
    return raw


def _site_runtime_signature(site: Optional["Site"]) -> Optional[tuple[str, float, float, float, str]]:
    if not isinstance(site, Site):
        return None
    return (
        str(site.name or "").strip(),
        round(float(site.latitude), 6),
        round(float(site.longitude), 6),
        round(float(site.elevation), 2),
        str(site.timezone_name or "").strip(),
    )


def _same_runtime_site(first: Optional["Site"], second: Optional["Site"]) -> bool:
    first_sig = _site_runtime_signature(first)
    second_sig = _site_runtime_signature(second)
    return first_sig is not None and first_sig == second_sig


def _should_apply_observatory_change(
    current_name: str,
    current_site: Optional["Site"],
    next_name: str,
    next_site: Optional["Site"],
) -> bool:
    normalized_next = str(next_name or "").strip()
    if not normalized_next or not isinstance(next_site, Site):
        return False
    if str(current_name or "").strip() != normalized_next:
        return True
    return not _same_runtime_site(current_site, next_site)


def _style_dialog_button_box(
    button_box: Optional[QDialogButtonBox],
    *,
    ok: str = "secondary",
    cancel: str = "ghost",
    apply: str = "secondary",
    close: str = "secondary",
) -> None:
    if button_box is None:
        return
    mapping = {
        QDialogButtonBox.Ok: ok,
        QDialogButtonBox.Cancel: cancel,
        QDialogButtonBox.Apply: apply,
        QDialogButtonBox.Close: close,
    }
    icon_mapping = {
        QDialogButtonBox.Ok: "ok",
        QDialogButtonBox.Cancel: "cancel",
        QDialogButtonBox.Apply: "apply",
        QDialogButtonBox.Close: "cancel",
    }
    for role, variant in mapping.items():
        btn = button_box.button(role)
        if btn is not None:
            _set_button_variant(btn, variant)
            _set_button_icon_kind(btn, icon_mapping.get(role, "spark"), 14)


_VALID_TABLE_COLOR_MODES = {"background", "text_glow"}


def _normalize_table_color_mode(value: object, *, default: str = "background") -> str:
    mode = str(value or "").strip().lower()
    if mode in _VALID_TABLE_COLOR_MODES:
        return mode
    return default


# --- Table Settings Dialog ---




# Number of time samples for visibility curve (lower = faster)
TIME_SAMPLES = 240 


# --------------------------------------------------
# --- Main Window ----------------------------------
# --------------------------------------------------
class MainWindow(QMainWindow):
    darkModeChanged = Signal(bool)
    bhtom_observatory_presets_changed = Signal(list, str)
    bhtom_observatory_presets_loading = Signal(bool, str)

    def eventFilter(self, watched, event):  # noqa: D401
        if hasattr(self, "ai_output") and watched is self.ai_output:
            if event.type() == QEvent.Type.Resize:
                new_width = self._ai_message_layout_width()
                last_width = int(getattr(self, "_ai_output_last_viewport_width", 0))
                if abs(new_width - last_width) >= 8:
                    self._ai_output_last_viewport_width = new_width
                    if getattr(self, "_ai_messages", None):
                        QTimer.singleShot(0, self._render_ai_messages)
        return super().eventFilter(watched, event)

    def _update_night_details_constraints(self):
        if not hasattr(self, "info_widget") or not hasattr(self, "info_card"):
            return
        self.info_widget.adjustSize()
        content_hint = self.info_widget.minimumSizeHint()
        content_h = max(self.info_widget.sizeHint().height(), content_hint.height())
        title_h = self.info_title_label.sizeHint().height() if hasattr(self, "info_title_label") else 20
        card_min_h = max(220, content_h + title_h + 30)
        card_min_w = max(330, int(getattr(self, "_night_details_fixed_min_width", 330)))
        self.info_card.setMinimumHeight(card_min_h)
        self.info_card.setMinimumWidth(card_min_w)
        if hasattr(self, "info_scroll"):
            self.info_scroll.setMinimumHeight(content_h + 6)
            self.info_scroll.setMinimumWidth(max(260, card_min_w - 18))
        if hasattr(self, "right_dashboard"):
            dashboard_base_min = max(460, int(getattr(self, "_right_dashboard_base_min_width", 460)))
            self.right_dashboard.setMinimumWidth(max(dashboard_base_min, card_min_w + 22))

    def _ensure_display_font_loaded(self) -> None:
        _ensure_display_font_loaded()

    def _pick_font_family(self, candidates: list[str]) -> str:
        return _pick_font_family(candidates)

    def _pick_matplotlib_font_family(self, candidates: list[str]) -> str:
        return _pick_matplotlib_font_family(candidates)

    def _set_dark_mode_enabled(self, enabled: bool, persist: bool = True):
        enabled = bool(enabled)
        if enabled == getattr(self, "_dark_enabled", False):
            if persist:
                self.settings.setValue("general/darkMode", enabled)
            self._sync_dark_mode_menu_action()
            return
        self._dark_enabled = enabled
        if persist:
            self.settings.setValue("general/darkMode", self._dark_enabled)
        self._apply_styles()
        self._sync_dark_mode_menu_action()
        self.darkModeChanged.emit(self._dark_enabled)

    def _sync_dark_mode_menu_action(self) -> None:
        if not hasattr(self, "dark_act"):
            return
        blocker = QSignalBlocker(self.dark_act)
        self.dark_act.setChecked(bool(getattr(self, "_dark_enabled", False)))
        del blocker

    def _theme_color(self, key: str, fallback: str) -> str:
        tokens = getattr(self, "_theme_tokens", {})
        value = tokens.get(key) if isinstance(tokens, dict) else None
        return str(value or fallback)

    def _theme_qcolor(self, key: str, fallback: str) -> QColor:
        raw = self._theme_color(key, fallback)
        match = re.fullmatch(
            r"rgba?\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})(?:\s*,\s*([0-9.]+))?\s*\)",
            raw,
        )
        if match:
            red = max(0, min(255, int(match.group(1))))
            green = max(0, min(255, int(match.group(2))))
            blue = max(0, min(255, int(match.group(3))))
            alpha_group = match.group(4)
            alpha = 255
            if alpha_group is not None:
                try:
                    alpha_value = float(alpha_group)
                    alpha = int(round(255 * alpha_value)) if alpha_value <= 1.0 else int(round(alpha_value))
                except ValueError:
                    alpha = 255
                alpha = max(0, min(255, alpha))
            return QColor(red, green, blue, alpha)
        color = QColor(raw)
        if color.isValid():
            return color
        return QColor(fallback)

    def _accent_secondary_settings_key(self, theme_key: str | None = None) -> str:
        normalized = normalize_theme_key(theme_key or getattr(self, "_theme_name", DEFAULT_UI_THEME))
        return f"general/accentSecondaryColorByTheme/{normalized}"

    def _load_accent_secondary_override(self, theme_key: str | None = None) -> str:
        normalized = normalize_theme_key(theme_key or getattr(self, "_theme_name", DEFAULT_UI_THEME))
        per_theme_key = self._accent_secondary_settings_key(normalized)
        stored = _normalized_css_color(self.settings.value(per_theme_key, "", type=str))
        if stored:
            return stored
        return ""

    def _save_accent_secondary_override(self, theme_key: str, color: str) -> None:
        normalized = normalize_theme_key(theme_key)
        per_theme_key = self._accent_secondary_settings_key(normalized)
        normalized_color = _normalized_css_color(color)
        if normalized_color:
            self.settings.setValue(per_theme_key, normalized_color)
        else:
            self.settings.remove(per_theme_key)
        self.settings.remove("general/accentSecondaryColor")

    def _refresh_mpl_theme(self) -> None:
        self._visibility_matplotlib._refresh_mpl_theme()

    def _apply_styles(self):
        """Apply a custom stylesheet, fonts, and default icon sizes."""
        self._ensure_display_font_loaded()
        display_family = str(getattr(self, "_display_font_family", "") or _preferred_display_font_family()).strip()
        self._display_font_family = display_family
        body_family = display_family
        font_size = _sanitize_ui_font_size(getattr(self, "_ui_font_size", 11))
        app_font = QFont(body_family)
        app_font.setPointSize(font_size)
        QApplication.setFont(app_font)
        plt.rcParams["font.family"] = [
            _preferred_display_font_family(),
            "Rajdhani",
            "DejaVu Sans",
            "Arial",
            "Helvetica",
        ]
        self._theme_name = normalize_theme_key(getattr(self, "_theme_name", DEFAULT_UI_THEME))
        theme_overrides: dict[str, str] = {}
        if getattr(self, "_accent_secondary_override", ""):
            theme_overrides["btn2"] = str(self._accent_secondary_override)
        self._theme_tokens = resolve_theme_tokens(
            self._theme_name,
            dark_enabled=getattr(self, "_dark_enabled", False),
            ui_font_size=font_size,
            font_family=body_family,
            display_font_family=display_family,
            overrides=theme_overrides or None,
        )
        if hasattr(self, "_visibility_web_html_cache"):
            self._visibility_web_html_cache.clear()
        stylesheet = build_stylesheet(
            self._theme_name,
            dark_enabled=getattr(self, "_dark_enabled", False),
            ui_font_size=font_size,
            font_family=body_family,
            display_font_family=display_family,
            overrides=theme_overrides or None,
        )
        self.setStyleSheet(stylesheet)
        ai_window = getattr(self, "ai_window", None)
        if isinstance(ai_window, QDialog):
            ai_window.setStyleSheet(stylesheet)
        weather_window = getattr(self, "weather_window", None)
        if isinstance(weather_window, QDialog):
            weather_window.setStyleSheet(stylesheet)
        if getattr(self, "visibility_web", None) is not None:
            try:
                self.visibility_web.setStyleSheet(f"background:{self._theme_color('plot_bg', '#0f192b')};")
                self.visibility_web.page().setBackgroundColor(self._theme_qcolor("plot_bg", "#0f192b"))
            except Exception:
                pass
        if getattr(self, "visibility_plot_host", None) is not None:
            try:
                self.visibility_plot_host.setStyleSheet(f"background:{self._theme_color('plot_bg', '#0f192b')};")
            except Exception:
                pass
        if getattr(self, "visibility_loading_widget", None) is not None:
            try:
                self.visibility_loading_widget.setStyleSheet(f"background:{self._theme_color('plot_bg', '#0f192b')};")
            except Exception:
                pass
        bootstrap_placeholder = getattr(self, "_visibility_bootstrap_placeholder", None)
        if isinstance(bootstrap_placeholder, QWidget):
            try:
                bootstrap_placeholder.setStyleSheet(f"background:{self._theme_color('plot_bg', '#0f192b')};")
            except Exception:
                pass
        for placeholder in getattr(self, "_cutout_image_placeholders", {}).values():
            if isinstance(placeholder, QWidget):
                try:
                    placeholder.setStyleSheet(f"background:{self._theme_color('plot_bg', '#0f192b')};")
                except Exception:
                    pass
        for stack in getattr(self, "_cutout_image_stacks", {}).values():
            host = stack.parentWidget() if isinstance(stack, QStackedLayout) else None
            if isinstance(host, QWidget):
                try:
                    host.setStyleSheet(f"background:{self._theme_color('plot_bg', '#0f192b')};")
                except Exception:
                    pass
        self._refresh_target_color_map()
        self._refresh_mpl_theme()
        self._refresh_date_nav_icons()
        self._refresh_plot_mode_switch()
        self._update_plot_mode_label_metrics()
        if isinstance(getattr(self, "last_payload", None), dict):
            self._render_visibility_web_plot(self.last_payload)
        if hasattr(self, "table_model") and hasattr(self, "settings"):
            palette = highlight_palette_for_theme(
                getattr(self, "_theme_name", DEFAULT_UI_THEME),
                dark_enabled=bool(getattr(self, "_dark_enabled", False)),
                color_blind=bool(getattr(self, "color_blind_mode", False)),
            )
            default_colors = {"below": palette.below, "limit": palette.limit, "above": palette.above}
            self.table_model.highlight_colors = {
                key: _qcolor_from_token(_resolve_table_highlight_color(self.settings, key, default_colors[key]), default_colors[key])
                for key in default_colors
            }
            if hasattr(self, "_emit_table_data_changed"):
                self._emit_table_data_changed()
        if isinstance(weather_window, WeatherDialog):
            self._refresh_weather_window_context()
        _refresh_button_icons(self)
        if isinstance(ai_window, QDialog):
            _refresh_button_icons(ai_window)
        if isinstance(weather_window, QDialog):
            _refresh_button_icons(weather_window)
        self._update_night_details_constraints()
        # Apply icon size globally
        self.setIconSize(QSize(20, 20))

    def _build_date_nav_icon_pixmap(self, kind: str, stroke: QColor, size: int = 20) -> QPixmap:
        px = QPixmap(size, size)
        px.fill(Qt.GlobalColor.transparent)
        painter = QPainter(px)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        pen = QPen(stroke)
        pen.setWidthF(max(1.5, size * 0.11))
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        if kind == "prev":
            painter.drawLine(int(size * 0.60), int(size * 0.24), int(size * 0.37), int(size * 0.50))
            painter.drawLine(int(size * 0.60), int(size * 0.76), int(size * 0.37), int(size * 0.50))
        elif kind == "next":
            painter.drawLine(int(size * 0.40), int(size * 0.24), int(size * 0.63), int(size * 0.50))
            painter.drawLine(int(size * 0.40), int(size * 0.76), int(size * 0.63), int(size * 0.50))
        else:
            m = size * 0.22
            arc = int(size - 2 * m)
            painter.drawArc(int(m), int(m), arc, arc, int(35 * 16), int(290 * 16))
            tip_x = int(size * 0.77)
            tip_y = int(size * 0.30)
            painter.drawLine(tip_x, tip_y, int(size * 0.64), int(size * 0.30))
            painter.drawLine(tip_x, tip_y, int(size * 0.77), int(size * 0.43))
        painter.end()
        return px

    def _build_date_nav_icon(self, kind: str) -> QIcon:
        base = self.palette().color(QPalette.ColorRole.ButtonText)
        if not base.isValid():
            base = self._theme_qcolor("section_title", "#d8e6f5")
        normal = QColor(base)
        normal.setAlpha(225)
        active = QColor(base)
        active.setAlpha(255)
        disabled = QColor(base)
        disabled.setAlpha(95)
        icon = QIcon()
        icon.addPixmap(self._build_date_nav_icon_pixmap(kind, normal), QIcon.Mode.Normal, QIcon.State.Off)
        icon.addPixmap(self._build_date_nav_icon_pixmap(kind, active), QIcon.Mode.Active, QIcon.State.Off)
        icon.addPixmap(self._build_date_nav_icon_pixmap(kind, disabled), QIcon.Mode.Disabled, QIcon.State.Off)
        return icon

    def _refresh_date_nav_icons(self):
        for attr, kind in (
            ("prev_day_btn", "prev"),
            ("today_btn", "today"),
            ("next_day_btn", "next"),
        ):
            btn = getattr(self, attr, None)
            if isinstance(btn, QToolButton):
                btn.setIcon(self._build_date_nav_icon(kind))

    def _build_web_loading_placeholder(self, title: str, message: str) -> QWidget:
        widget = QWidget(self)
        widget.setAttribute(Qt.WA_StyledBackground, True)
        widget.setStyleSheet(f"background:{self._theme_color('plot_bg', '#121b29')};")
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)
        layout.addStretch(1)
        card = QFrame(widget)
        card.setObjectName("VisibilityLoadingCard")
        card.setMinimumWidth(420)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(30, 26, 30, 26)
        card_layout.setSpacing(14)
        title_label = QLabel(title, card)
        title_label.setObjectName("SectionTitle")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setMinimumWidth(360)
        skeleton = SkeletonShimmerWidget("plot", card)
        skeleton.setMinimumHeight(260)
        hint_label = QLabel(message, card)
        hint_label.setObjectName("SectionHint")
        hint_label.setAlignment(Qt.AlignCenter)
        hint_label.setWordWrap(True)
        hint_label.setMinimumWidth(380)
        hint_label.setMaximumWidth(520)
        hint_label.setMinimumHeight(54)
        hint_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.MinimumExpanding)
        card_layout.addWidget(title_label, 0, Qt.AlignHCenter)
        card_layout.addWidget(skeleton, 0)
        card_layout.addWidget(hint_label, 0, Qt.AlignHCenter)
        layout.addWidget(card, 0, Qt.AlignCenter)
        layout.addStretch(1)
        widget._loading_hint_label = hint_label  # type: ignore[attr-defined]
        widget._loading_skeleton = skeleton  # type: ignore[attr-defined]
        return widget

    def _build_cutout_loading_placeholder(self, title: str, message: str) -> QWidget:
        return build_cutout_loading_placeholder(self, title, message)

    def _ensure_visibility_plot_widgets(self) -> None:
        if bool(getattr(self, "_visibility_plot_widgets_ready", False)):
            return

        if self.plot_canvas is None or self.ax_alt is None:
            self.plot_canvas = FigureCanvas(Figure(figsize=(6, 4), tight_layout=True))
            self.plot_canvas.setContentsMargins(0, 0, 0, 0)
            self.plot_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.ax_alt = self.plot_canvas.figure.subplots()
            self.ax_alt.margins(x=0, y=0)

        if self._use_visibility_web and QWebEngineView is not None:
            self.visibility_web = QWebEngineView(self)
            self.visibility_web.setMinimumHeight(320)
            self.visibility_web.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.visibility_web.setStyleSheet(f"background:{self._theme_color('plot_bg', '#121b29')};")
            self.visibility_web.setToolTip("Interactive visibility chart. Use Plotly controls in the top-right to zoom, pan and reset.")
            self.visibility_loading_widget = self._build_web_loading_placeholder(
                "Visibility Plot",
                "Computing night tracks…",
            )
            self.visibility_plot_stack = QStackedLayout()
            self.visibility_plot_stack.setContentsMargins(0, 0, 0, 0)
            self.visibility_plot_host = QWidget(self)
            self.visibility_plot_host.setAttribute(Qt.WA_StyledBackground, True)
            self.visibility_plot_host.setStyleSheet(f"background:{self._theme_color('plot_bg', '#121b29')};")
            self.visibility_plot_host.setLayout(self.visibility_plot_stack)
            self.visibility_plot_stack.addWidget(self.visibility_loading_widget)
            self.visibility_plot_stack.addWidget(self.visibility_web)
            self.visibility_plot_stack.setCurrentWidget(self.visibility_loading_widget)
            self.visibility_web.loadFinished.connect(self._on_visibility_web_load_finished)
            plot_content_widget = self.visibility_plot_host
        else:
            self.plot_toolbar = NavigationToolbar(self.plot_canvas, self)
            self.plot_toolbar.setIconSize(QSize(16, 16))
            self.plot_toolbar.layout().setContentsMargins(0, 0, 0, 0)
            self.plot_toolbar.layout().setSpacing(0)
            self.plot_toolbar.addSeparator()
            self.plot_toolbar.addWidget(self.plot_mode_widget)
            self.plot_toolbar.setMaximumHeight(self.plot_toolbar.sizeHint().height())
            plot_content_widget = self.plot_canvas

        placeholder = getattr(self, "_visibility_bootstrap_placeholder", None)
        plot_layout = getattr(self, "plot_card_layout", None)
        if isinstance(plot_layout, QVBoxLayout):
            if isinstance(placeholder, QWidget):
                plot_layout.removeWidget(placeholder)
                placeholder.hide()
            if self.plot_toolbar is not None and plot_layout.indexOf(self.plot_toolbar) < 0:
                plot_layout.insertWidget(0, self.plot_toolbar)
            if isinstance(plot_content_widget, QWidget) and plot_layout.indexOf(plot_content_widget) < 0:
                plot_layout.addWidget(plot_content_widget, 1)

        self._visibility_plot_widgets_ready = True
        self._refresh_mpl_theme()

    def _ensure_visibility_plot_placeholder_message(self, message: str) -> None:
        placeholder = getattr(self, "_visibility_bootstrap_placeholder", None)
        hint_label = getattr(placeholder, "_loading_hint_label", None)
        if isinstance(hint_label, QLabel):
            hint_label.setText(message)

    def _create_cutout_image_stack(
        self,
        parent: QWidget,
        *,
        kind: str,
        title: str,
    ) -> tuple[QWidget, CoverImageLabel]:
        return create_cutout_image_stack(self, parent, kind=kind, title=title)

    def _set_cutout_image_loading(self, kind: str, message: str, *, visible: bool = True) -> None:
        set_cutout_image_loading(self, kind, message, visible=visible)

    def _set_visibility_loading_state(self, message: str, *, visible: bool = True) -> None:
        stack = getattr(self, "visibility_plot_stack", None)
        placeholder = getattr(self, "visibility_loading_widget", None)
        web_view = getattr(self, "visibility_web", None)
        if stack is None or placeholder is None or web_view is None:
            return
        hint_label = getattr(placeholder, "_loading_hint_label", None)
        if isinstance(hint_label, QLabel):
            hint_label.setText(message)
        if visible:
            stack.setCurrentWidget(placeholder)
        else:
            stack.setCurrentWidget(web_view)

    @Slot(bool)
    def _on_visibility_web_load_finished(self, ok: bool) -> None:
        if ok:
            self._visibility_web_has_content = True
            self._set_visibility_loading_state("", visible=False)
            self._apply_visibility_web_selection_style()
            return
        self._visibility_web_has_content = False
        self._set_visibility_loading_state("Unable to render interactive chart.", visible=True)

    def _line_palette(self) -> list[str]:
        return self._visibility_matplotlib._line_palette()

    @staticmethod
    def _target_color_key(target: Target) -> str:
        return VisibilityMatplotlibCoordinator._target_color_key(target)

    def _ensure_auto_target_color_palette(self, palette: Optional[list[str]] = None) -> list[str]:
        return self._visibility_matplotlib._ensure_auto_target_color_palette(palette)

    def _target_plot_color_css(
        self,
        target: Target,
        index: int,
        palette: Optional[list[str]] = None,
    ) -> str:
        return self._visibility_matplotlib._target_plot_color_css(target, index, palette)

    def _airmass_from_altitude(self, altitude_deg: object) -> np.ndarray:
        return self._visibility_matplotlib._airmass_from_altitude(altitude_deg)

    def _plot_y_values(self, altitude_deg: object) -> np.ndarray:
        return self._visibility_matplotlib._plot_y_values(altitude_deg)

    def _plot_limit_value(self) -> float:
        return self._visibility_matplotlib._plot_limit_value()

    def _visibility_time_window(
        self,
        data: dict,
        tz,
    ) -> tuple[datetime, datetime, dict[str, datetime]]:
        return self._visibility_matplotlib._visibility_time_window(data, tz)

    def _visibility_grid_color(self, *, alpha: float = 1.0) -> str:
        return self._visibility_matplotlib._visibility_grid_color(alpha=alpha)

    def _visibility_grid_rgba(self, *, alpha: float = 1.0) -> tuple[float, float, float, float]:
        return self._visibility_matplotlib._visibility_grid_rgba(alpha=alpha)

    def _visibility_context_key_from_parts(
        self,
        *,
        site_name: str,
        latitude: float,
        longitude: float,
        elevation: float,
        obs_date,
        time_samples: int,
        limit_altitude: float,
    ) -> str:
        return self._visibility_matplotlib._visibility_context_key_from_parts(
            site_name=site_name,
            latitude=latitude,
            longitude=longitude,
            elevation=elevation,
            obs_date=obs_date,
            time_samples=time_samples,
            limit_altitude=limit_altitude,
        )

    def _current_visibility_context_key(self) -> str:
        return self._visibility_matplotlib._current_visibility_context_key()

    def _show_visibility_matplotlib_placeholder(self, title: str, message: str) -> None:
        self._visibility_matplotlib._show_visibility_matplotlib_placeholder(title, message)

    def _begin_visibility_refresh(self, message: str) -> None:
        self._visibility_matplotlib._begin_visibility_refresh(message)

    def _configure_main_plot_y_axis(self) -> None:
        self._visibility_matplotlib._configure_main_plot_y_axis()

    def _update_plot_mode_label_metrics(self) -> None:
        self._visibility_matplotlib._update_plot_mode_label_metrics()

    def _animate_plot_mode_switch(self) -> None:
        self._visibility_matplotlib._animate_plot_mode_switch()

    def _refresh_plot_mode_switch(self) -> None:
        self._visibility_matplotlib._refresh_plot_mode_switch()

    @staticmethod
    def _polar_rgba_array(color: QColor, alphas: np.ndarray) -> np.ndarray:
        return VisibilityMatplotlibCoordinator._polar_rgba_array(color, alphas)

    def _ensure_radar_echo_artists(self, count: int) -> None:
        self._visibility_matplotlib._ensure_radar_echo_artists(count)

    def _build_radar_sweep_cmap(self):
        return self._visibility_matplotlib._build_radar_sweep_cmap()

    @staticmethod
    def _radar_sector_vertices(theta_start: float, theta_end: float, outer_radius: float = 90.0, samples: int = 12) -> np.ndarray:
        return VisibilityMatplotlibCoordinator._radar_sector_vertices(theta_start, theta_end, outer_radius, samples)

    def _refresh_radar_sweep_artists(self, *, redraw: bool = True, delta_s: Optional[float] = None) -> None:
        self._visibility_matplotlib._refresh_radar_sweep_artists(redraw=redraw, delta_s=delta_s)

    def _update_radar_sweep_state(self) -> None:
        self._visibility_matplotlib._update_radar_sweep_state()

    @Slot()
    def _advance_radar_sweep(self) -> None:
        self._visibility_matplotlib._advance_radar_sweep()

    def _reset_plot_navigation_home(self) -> None:
        self._visibility_matplotlib._reset_plot_navigation_home()

    def _selected_target_names(self) -> list[str]:
        return self._visibility_coordinator.selected_target_names()

    def _schedule_selected_cutout_update(self, target: Optional[Target]) -> None:
        self._visibility_coordinator.schedule_selected_cutout_update(target)

    @Slot()
    def _flush_selected_cutout_update(self) -> None:
        self._visibility_coordinator.flush_selected_cutout_update()

    def _refresh_visibility_matplotlib_mode_only(self, data: Optional[dict] = None) -> None:
        self._visibility_matplotlib._refresh_visibility_matplotlib_mode_only(data)

    def _visibility_web_render_signature(
        self,
        data: dict,
        *,
        now_override: Optional[datetime] = None,
    ) -> str:
        return self._visibility_coordinator.visibility_web_render_signature(
            data,
            now_override=now_override,
        )

    def _store_visibility_web_html_cache(self, cache_key: str, html: str) -> None:
        self._visibility_coordinator.store_visibility_web_html_cache(cache_key, html)

    def _schedule_visibility_plot_refresh(self, *, delay_ms: int = 0) -> None:
        self._visibility_coordinator.schedule_visibility_plot_refresh(delay_ms=delay_ms)

    @Slot()
    def _flush_visibility_plot_refresh(self) -> None:
        self._visibility_coordinator.flush_visibility_plot_refresh()

    def _apply_visibility_web_selection_style(self, selected_names: Optional[set[str]] = None) -> None:
        self._visibility_coordinator.apply_visibility_web_selection_style(selected_names)

    def _apply_visibility_line_style(self, line: object, *, is_over: bool, is_selected: bool) -> None:
        self._visibility_matplotlib._apply_visibility_line_style(line, is_over=is_over, is_selected=is_selected)

    def _build_visibility_plotly_html(
        self,
        data: dict,
        *,
        now_override: Optional[datetime] = None,
    ) -> Optional[str]:
        try:
            tz = pytz.timezone(str(data.get("tz", "UTC") or "UTC"))
        except Exception:
            tz = pytz.UTC
        try:
            times = [t.astimezone(tz) for t in mdates.num2date(data["times"])]
        except Exception:
            return None
        if len(times) < 2:
            return None

        use_tokens = getattr(
            self,
            "_theme_tokens",
            resolve_theme_tokens(
                getattr(self, "_theme_name", DEFAULT_UI_THEME),
                dark_enabled=bool(getattr(self, "_dark_enabled", False)),
            ),
        )
        line_palette = self._line_palette()
        target_colors = [
            self._target_plot_color_css(tgt, idx, line_palette)
            for idx, tgt in enumerate(self.targets)
        ]
        row_enabled = list(getattr(self.table_model, "row_enabled", []))
        if len(row_enabled) != len(self.targets):
            row_enabled = [True] * len(self.targets)
        start_dt, end_dt, event_map = self._visibility_time_window(data, tz)
        fallback_label_dt = now_override if isinstance(now_override, datetime) else data.get("now_local")
        if not isinstance(fallback_label_dt, datetime):
            fallback_label_dt = datetime.now(tz)
        if fallback_label_dt.tzinfo is None:
            fallback_label_dt = tz.localize(fallback_label_dt)
        else:
            fallback_label_dt = fallback_label_dt.astimezone(tz)
        date_label = (
            self.date_edit.date().toString("yyyy-MM-dd")
            if hasattr(self, "date_edit")
            else fallback_label_dt.strftime("%Y-%m-%d")
        )
        request = VisibilityPlotlyRequest(
            data=data,
            targets=list(self.targets),
            row_enabled=row_enabled,
            target_colors=target_colors,
            limit_altitude=float(self.limit_spin.value()),
            sun_alt_limit=self._sun_alt_limit(),
            plot_airmass=bool(self._plot_airmass),
            show_sun=bool(self.sun_check.isChecked()),
            show_moon=bool(self.moon_check.isChecked()),
            date_label=date_label,
            theme_tokens=use_tokens,
            dark_enabled=bool(getattr(self, "_dark_enabled", False)),
            start_dt=start_dt,
            end_dt=end_dt,
            event_map=event_map,
            grid_css=self._visibility_grid_color(alpha=0.42),
            guide_css=self._visibility_grid_color(alpha=0.24),
            plot_font=_plot_font_css_stack(use_tokens),
            font_face_css=_embedded_display_font_css(),
            use_local_plotly_js=bool(_PLOTLY_JS_BASE_DIR),
        )
        return build_visibility_plotly_html(
            request,
            now_override=now_override,
        )

    def _render_visibility_web_plot(
        self,
        data: Optional[dict] = None,
        *,
        now_override: Optional[datetime] = None,
    ) -> None:
        self._visibility_coordinator.render_visibility_web_plot(
            data,
            now_override=now_override,
        )

    def _refresh_target_color_map(self, palette: Optional[list[str]] = None):
        self._visibility_matplotlib._refresh_target_color_map(palette)

    def _cutout_render_dimensions_px(self, label: Optional[QLabel] = None) -> tuple[int, int]:
        return self._preview_coordinator._cutout_render_dimensions_px(label)

    def _cutout_fov_axes_arcmin(self, width_px: int, height_px: int) -> tuple[float, float]:
        return self._preview_coordinator._cutout_fov_axes_arcmin(width_px, height_px)

    def _aladin_fetch_fov_axes_arcmin(self, width_px: int, height_px: int) -> tuple[float, float]:
        return self._preview_coordinator._aladin_fetch_fov_axes_arcmin(width_px, height_px)

    def _aladin_fetch_min_short_axis_arcmin(self) -> float:
        return self._preview_coordinator._aladin_fetch_min_short_axis_arcmin()

    def _aladin_fetch_dimensions_px(self, width_px: int, height_px: int) -> tuple[int, int]:
        return self._preview_coordinator._aladin_fetch_dimensions_px(width_px, height_px)

    def _site_telescope_fov_arcmin(self, site: Optional[Site] = None) -> Optional[tuple[float, float]]:
        return self._preview_coordinator._site_telescope_fov_arcmin(site)

    def _telescope_overlay_signature(self, site: Optional[Site] = None) -> str:
        return self._preview_coordinator._telescope_overlay_signature(site)

    def _fit_cutout_base_fov_to_telescope(self, tel_fov_x: float, tel_fov_y: float, width_px: int, height_px: int) -> int:
        return self._preview_coordinator._fit_cutout_base_fov_to_telescope(tel_fov_x, tel_fov_y, width_px, height_px)

    def _sync_cutout_fov_to_site(self, site: Optional[Site] = None, persist: bool = False) -> bool:
        return self._preview_coordinator._sync_cutout_fov_to_site(site, persist)

    def _telescope_overlay_rect(
        self,
        width_px: int,
        height_px: int,
        margin_px: int = 0,
        fov_axes: Optional[tuple[float, float]] = None,
    ) -> Optional[tuple[int, int, int, int]]:
        return self._preview_coordinator._telescope_overlay_rect(width_px, height_px, margin_px, fov_axes)

    def _paint_telescope_fov_overlay(
        self,
        painter: QPainter,
        w: int,
        h: int,
        fill: bool = True,
        color: Optional[QColor] = None,
        offset_x: int = 0,
        offset_y: int = 0,
        fov_axes: Optional[tuple[float, float]] = None,
        min_margin_px: int = 4,
    ) -> None:
        return self._preview_coordinator._paint_telescope_fov_overlay(painter, w, h, fill, color, offset_x, offset_y, fov_axes, min_margin_px)

    def _cutout_fov_text(
        self,
        width_px: int,
        height_px: int,
        fetch_margin: bool = False,
        fov_axes: Optional[tuple[float, float]] = None,
    ) -> str:
        return self._preview_coordinator._cutout_fov_text(width_px, height_px, fetch_margin, fov_axes)

    def _cutout_key_for_target(self, target: Target, width_px: int, height_px: int) -> str:
        return self._preview_coordinator._cutout_key_for_target(target, width_px, height_px)

    def _aladin_visible_image_rect(
        self,
        widget_w: int,
        widget_h: int,
        image_x: int,
        image_y: int,
        image_w: int,
        image_h: int,
    ) -> tuple[int, int, int, int]:
        return self._preview_coordinator._aladin_visible_image_rect(widget_w, widget_h, image_x, image_y, image_w, image_h)

    def _aladin_visible_fov_axes_arcmin(
        self,
        widget_w: int,
        widget_h: int,
        image_x: int,
        image_y: int,
        image_w: int,
        image_h: int,
    ) -> tuple[float, float]:
        return self._preview_coordinator._aladin_visible_fov_axes_arcmin(widget_w, widget_h, image_x, image_y, image_w, image_h)

    def _cutout_resize_signature_for_target(self, target: Optional[Target]) -> Optional[tuple]:
        return self._preview_coordinator._cutout_resize_signature_for_target(target)

    @Slot(int, int)
    def _schedule_cutout_resize_refresh(self, *_):
        return self._preview_coordinator._schedule_cutout_resize_refresh(*_)

    @Slot()
    def _on_cutout_resize_timeout(self):
        return self._preview_coordinator._on_cutout_resize_timeout()

    def _set_cutout_placeholder(self, text: str):
        return self._preview_coordinator._set_cutout_placeholder(text)

    def _paint_aladin_static_overlay(
        self,
        painter: QPainter,
        w: int,
        h: int,
        image_x: Optional[int] = None,
        image_y: Optional[int] = None,
        image_w: Optional[int] = None,
        image_h: Optional[int] = None,
    ) -> None:
        return self._preview_coordinator._paint_aladin_static_overlay(painter, w, h, image_x, image_y, image_w, image_h)

    def _build_aladin_overlay_pixmap(self, source: QPixmap) -> QPixmap:
        return self._preview_coordinator._build_aladin_overlay_pixmap(source)

    def _build_finder_overlay_pixmap(self, source: QPixmap) -> QPixmap:
        return self._preview_coordinator._build_finder_overlay_pixmap(source)

    def _pixmap_to_png_bytes(self, pixmap: QPixmap) -> bytes:
        return self._preview_coordinator._pixmap_to_png_bytes(pixmap)

    def _load_pixmap_from_storage_cache(self, namespace: str, key: str) -> Optional[QPixmap]:
        return self._preview_coordinator._load_pixmap_from_storage_cache(namespace, key)

    def _persist_pixmap_to_storage_cache(self, namespace: str, key: str, pixmap: QPixmap) -> None:
        return self._preview_coordinator._persist_pixmap_to_storage_cache(namespace, key, pixmap)

    def _cache_cutout_pixmap(self, key: str, pixmap: QPixmap, *, persist: bool = True):
        return self._preview_coordinator._cache_cutout_pixmap(key, pixmap, persist=persist)

    def _cache_finder_pixmap(self, key: str, pixmap: QPixmap, *, persist: bool = True):
        return self._preview_coordinator._cache_finder_pixmap(key, pixmap, persist=persist)

    def _find_cutout_cache_variant(self, target: Target) -> Optional[tuple[str, QPixmap]]:
        return self._preview_coordinator._find_cutout_cache_variant(target)

    def _find_finder_cache_variant(self, target: Target) -> Optional[tuple[str, QPixmap]]:
        return self._preview_coordinator._find_finder_cache_variant(target)

    def _show_finder_aladin_fallback(self, key: str, text_if_missing: str) -> bool:
        return self._preview_coordinator._show_finder_aladin_fallback(key, text_if_missing)

    def _ensure_aladin_pan_ready(self) -> None:
        return self._preview_coordinator._ensure_aladin_pan_ready()

    def _set_finder_status(self, text: str, busy: bool = False):
        return self._preview_coordinator._set_finder_status(text, busy)

    def _set_aladin_status(self, text: str, busy: bool = False):
        return self._preview_coordinator._set_aladin_status(text, busy)

    def _finder_prefetch_done_status(self) -> None:
        return self._preview_coordinator._finder_prefetch_done_status()

    def _aladin_prefetch_done_status(self) -> None:
        return self._preview_coordinator._aladin_prefetch_done_status()

    def _set_bhtom_status(self, text: str, busy: bool = False):
        if not hasattr(self, "status_bhtom_label") or not hasattr(self, "status_bhtom_progress"):
            return
        self.status_bhtom_label.setText(text)
        if busy:
            self.status_bhtom_progress.setRange(0, 0)
            self.status_bhtom_progress.show()
            return
        self.status_bhtom_progress.hide()
        self.status_bhtom_progress.setRange(0, 1)
        self.status_bhtom_progress.setValue(0)

    def _stop_finder_workers(self, aggressive: bool = False):
        return self._preview_coordinator._stop_finder_workers(aggressive)

    def _cancel_finder_chart_worker(self):
        return self._preview_coordinator._cancel_finder_chart_worker()

    @Slot(int, str, bytes, str)
    def _on_finder_chart_completed(self, request_id: int, key: str, payload: bytes, err: str):
        return self._preview_coordinator._on_finder_chart_completed(request_id, key, payload, err)

    def _on_finder_chart_worker_finished(self, worker: FinderChartWorker):
        return self._preview_coordinator._on_finder_chart_worker_finished(worker)

    @Slot()
    def _on_finder_chart_timeout(self):
        return self._preview_coordinator._on_finder_chart_timeout()

    def _update_finder_chart_for_target(
        self,
        target: Target,
        key: str,
        *,
        background: bool = False,
        cache_only: bool = False,
    ):
        return self._preview_coordinator._update_finder_chart_for_target(target, key, background=background, cache_only=cache_only)

    def _enqueue_finder_prefetch(self, target: Optional[Target], key: str) -> str:
        return self._preview_coordinator._enqueue_finder_prefetch(target, key)

    def _drain_finder_prefetch_queue(self) -> None:
        return self._preview_coordinator._drain_finder_prefetch_queue()

    def _prefetch_finder_charts_for_all_targets(self, prioritize: Optional[Target] = None) -> None:
        return self._preview_coordinator._prefetch_finder_charts_for_all_targets(prioritize)

    def _start_cutout_request(
        self,
        target: Target,
        key: str,
        *,
        render_w: int,
        render_h: int,
        fetch_w: int,
        fetch_h: int,
        background: bool = False,
    ) -> None:
        return self._preview_coordinator._start_cutout_request(target, key, render_w=render_w, render_h=render_h, fetch_w=fetch_w, fetch_h=fetch_h, background=background)

    def _enqueue_cutout_prefetch(
        self,
        target: Optional[Target],
        key: str,
        *,
        render_w: int,
        render_h: int,
        fetch_w: int,
        fetch_h: int,
    ) -> str:
        return self._preview_coordinator._enqueue_cutout_prefetch(target, key, render_w=render_w, render_h=render_h, fetch_w=fetch_w, fetch_h=fetch_h)

    def _drain_cutout_prefetch_queue(self) -> None:
        return self._preview_coordinator._drain_cutout_prefetch_queue()

    def _prefetch_cutouts_for_all_targets(self, prioritize: Optional[Target] = None) -> None:
        return self._preview_coordinator._prefetch_cutouts_for_all_targets(prioritize)

    def _update_cutout_preview_for_target(self, target: Optional[Target], *, cache_only: bool = False):
        return self._preview_coordinator._update_cutout_preview_for_target(target, cache_only=cache_only)

    @Slot(QNetworkReply)
    def _on_cutout_reply(self, reply: QNetworkReply):
        return self._preview_coordinator._on_cutout_reply(reply)

    def _clear_cutout_cache(self):
        return self._preview_coordinator._clear_cutout_cache()

    def _selected_target_or_none(self) -> Optional[Target]:
        return self._preview_coordinator._selected_target_or_none()

    @Slot(int)
    def _on_cutout_tab_changed(self, index: int):
        return self._preview_coordinator._on_cutout_tab_changed(index)

    @Slot()
    def _aladin_zoom_in(self):
        return self._preview_coordinator._aladin_zoom_in()

    def _aladin_expand_context(self) -> None:
        return self._preview_coordinator._aladin_expand_context()

    @Slot()
    def _aladin_zoom_out(self):
        return self._preview_coordinator._aladin_zoom_out()

    @Slot()
    def _aladin_zoom_reset(self):
        return self._preview_coordinator._aladin_zoom_reset()

    @Slot(str)
    def _on_obs_change(self, name: str, *, defer_replot: bool = False):
        logger.info("Observatory switched to %s", name)
        """Populate site fields when an observatory is selected."""
        site = self.observatories.get(name)
        if site is None:
            return
        if hasattr(self, "settings"):
            self.settings.setValue("general/defaultSite", name)
        self._prime_selected_observatory(name)
        # Update the table model and replot with debounce
        self._clear_table_dynamic_cache()
        self.target_metrics.clear()
        self.target_windows.clear()
        self.last_payload = None
        # Reset color mapping until new calculation assigns fresh colors
        self.table_model.color_map.clear()
        self.table_model.layoutChanged.emit()
        self._validate_site_inputs()
        if not defer_replot:
            self._begin_visibility_refresh(f"Updating view for {name}…")
            self._replot_timer.start()
            if hasattr(self, "aladin_image_label"):
                self._set_cutout_image_loading("aladin", f"Loading Aladin preview for {name}…", visible=True)
            if hasattr(self, "finder_image_label"):
                self._set_cutout_image_loading("finder", f"Loading finder chart for {name}…", visible=True)
            weather_window = getattr(self, "weather_window", None)
            if isinstance(weather_window, WeatherDialog):
                weather_window._set_weather_plot_loading("conditions", "Updating conditions…", visible=True)
                weather_window._set_weather_plot_loading("meteogram", "Updating meteogram…", visible=True)
                weather_window._set_weather_image_loading("cloud_map", "Updating cloud climatology…", visible=True)
                weather_window._set_weather_image_loading("satellite", "Updating satellite preview…", visible=True)
        self._obs_change_finalize_pending = True
        self._schedule_plan_autosave()
        QTimer.singleShot(0, lambda n=str(name): self._finalize_observatory_change(n))

    def _prime_selected_observatory(self, name: str) -> bool:
        site = self.observatories.get(name)
        if site is None:
            return False
        self.lat_edit.setText(f"{site.latitude}")
        self.lon_edit.setText(f"{site.longitude}")
        self.elev_edit.setText(f"{site.elevation}")
        self.table_model.site = site
        self._validate_site_inputs()
        return True

    def _finalize_observatory_change(self, name: str) -> None:
        if not hasattr(self, "obs_combo") or self.obs_combo.currentText() != name:
            return
        site = self.observatories.get(name)
        if site is None:
            return
        self._obs_change_finalize_pending = False
        # Start the lightweight realtime worker first so the UI can repaint
        # immediately; heavier panel rebuilds are deferred one more tick.
        self._start_clock_worker()
        QTimer.singleShot(0, lambda n=str(name): self._refresh_observatory_dependent_views(n))

    def _refresh_observatory_dependent_views(self, name: str) -> None:
        if not hasattr(self, "obs_combo") or self.obs_combo.currentText() != name:
            return
        site = self.observatories.get(name)
        if site is None:
            return
        self._sync_cutout_fov_to_site(site)
        if not getattr(self, "_defer_startup_preview_updates", False):
            self._update_cutout_preview_for_target(self._selected_target_or_none())
        self._refresh_weather_window_context()

    def _current_limiting_magnitude(self) -> float:
        site = None
        if hasattr(self, "obs_combo") and hasattr(self, "observatories"):
            site = self.observatories.get(self.obs_combo.currentText())
        if site is None and hasattr(self, "table_model"):
            site = self.table_model.site
        if site is not None:
            value = _safe_float(getattr(site, "limiting_magnitude", None))
            if value is not None and math.isfinite(value):
                return float(value)
        return DEFAULT_LIMITING_MAGNITUDE

    def _quick_targets_config(self) -> dict[str, object]:
        count = self.settings.value("general/quickTargetsCount", QUICK_TARGETS_DEFAULT_COUNT, type=int)
        min_importance = self.settings.value(
            "general/quickTargetsMinImportance",
            BHTOM_SUGGESTION_MIN_IMPORTANCE,
            type=float,
        )
        try:
            count = int(count)
        except Exception:
            count = QUICK_TARGETS_DEFAULT_COUNT
        count = max(QUICK_TARGETS_MIN_COUNT, min(QUICK_TARGETS_MAX_COUNT, count))
        try:
            min_importance = float(min_importance)
        except Exception:
            min_importance = float(BHTOM_SUGGESTION_MIN_IMPORTANCE)
        if not math.isfinite(min_importance) or min_importance < 0.0:
            min_importance = float(BHTOM_SUGGESTION_MIN_IMPORTANCE)
        return {
            "count": count,
            "min_importance": min_importance,
            "use_score_filter": self.settings.value("general/quickTargetsUseScoreFilter", True, type=bool),
            "use_moon_filter": self.settings.value("general/quickTargetsUseMoonFilter", True, type=bool),
            "use_limiting_mag": self.settings.value("general/quickTargetsUseLimitingMag", True, type=bool),
        }

    def _update_quick_targets_button_tooltip(self) -> None:
        if not hasattr(self, "quick_targets_btn"):
            return
        cfg = self._quick_targets_config()
        count = int(cfg["count"])
        use_score = bool(cfg["use_score_filter"])
        use_moon = bool(cfg["use_moon_filter"])
        use_lim_mag = bool(cfg["use_limiting_mag"])
        parts: list[str] = []
        if use_score:
            parts.append("Score")
        if use_moon:
            parts.append("Moon Sep")
        if use_lim_mag:
            parts.append("Limiting mag")
        filters_txt = ", ".join(parts) if parts else "no extra filters"
        self.quick_targets_btn.setToolTip(
            f"Add top {count} suggested targets sorted by score ({filters_txt})."
        )

    def _seestar_config(self) -> dict[str, object]:
        method = str(self.settings.value("general/seestarMethod", SEESTAR_METHOD_GUIDED, type=str) or SEESTAR_METHOD_GUIDED).strip().lower()
        if method not in {SEESTAR_METHOD_GUIDED, SEESTAR_METHOD_ALP}:
            method = SEESTAR_METHOD_GUIDED
        alp_base_url = (
            str(self.settings.value("general/seestarAlpBaseUrl", SEESTAR_ALP_DEFAULT_BASE_URL, type=str) or SEESTAR_ALP_DEFAULT_BASE_URL)
            .strip()
            .rstrip("/")
        ) or SEESTAR_ALP_DEFAULT_BASE_URL
        alp_device_num = self.settings.value("general/seestarAlpDeviceNum", SEESTAR_ALP_DEFAULT_DEVICE_NUM, type=int)
        alp_client_id = self.settings.value("general/seestarAlpClientId", SEESTAR_ALP_DEFAULT_CLIENT_ID, type=int)
        alp_timeout_s = self.settings.value("general/seestarAlpTimeoutSec", SEESTAR_ALP_DEFAULT_TIMEOUT_S, type=float)
        alp_gain = self.settings.value("general/seestarAlpGain", SEESTAR_ALP_DEFAULT_GAIN, type=int)
        alp_panel_overlap = self.settings.value(
            "general/seestarAlpPanelOverlapPercent",
            SEESTAR_ALP_DEFAULT_PANEL_OVERLAP_PERCENT,
            type=int,
        )
        alp_use_autofocus = self.settings.value(
            "general/seestarAlpUseAutofocus",
            SEESTAR_ALP_DEFAULT_USE_AUTOFOCUS,
            type=bool,
        )
        alp_num_tries = self.settings.value("general/seestarAlpNumTries", SEESTAR_ALP_DEFAULT_NUM_TRIES, type=int)
        alp_retry_wait_s = self.settings.value(
            "general/seestarAlpRetryWaitSec",
            SEESTAR_ALP_DEFAULT_RETRY_WAIT_S,
            type=int,
        )
        alp_stack_exposure_ms = self.settings.value(
            "general/seestarAlpStackExposureMs",
            SEESTAR_ALP_DEFAULT_STACK_EXPOSURE_MS,
            type=int,
        )
        alp_lp_filter_mode = self.settings.value(
            "general/seestarAlpLpFilterMode",
            SEESTAR_ALP_LP_FILTER_AUTO,
            type=str,
        )
        alp_honor_queue_times = self.settings.value(
            "general/seestarAlpHonorQueueTimes",
            SEESTAR_ALP_DEFAULT_HONOR_QUEUE_TIMES,
            type=bool,
        )
        alp_wait_until_local_time = self.settings.value(
            "general/seestarAlpWaitUntilLocalTime",
            SEESTAR_ALP_DEFAULT_WAIT_UNTIL_LOCAL_TIME,
            type=str,
        )
        alp_startup_enabled = self.settings.value(
            "general/seestarAlpStartupEnabled",
            SEESTAR_ALP_DEFAULT_STARTUP_SEQUENCE,
            type=bool,
        )
        alp_startup_polar_align = self.settings.value(
            "general/seestarAlpStartupPolarAlign",
            SEESTAR_ALP_DEFAULT_STARTUP_POLAR_ALIGN,
            type=bool,
        )
        alp_startup_auto_focus = self.settings.value(
            "general/seestarAlpStartupAutoFocus",
            SEESTAR_ALP_DEFAULT_STARTUP_AUTO_FOCUS,
            type=bool,
        )
        alp_startup_dark_frames = self.settings.value(
            "general/seestarAlpStartupDarkFrames",
            SEESTAR_ALP_DEFAULT_STARTUP_DARK_FRAMES,
            type=bool,
        )
        alp_capture_flats = self.settings.value(
            "general/seestarAlpCaptureFlatsBeforeSession",
            SEESTAR_ALP_DEFAULT_CAPTURE_FLATS,
            type=bool,
        )
        alp_flats_wait_s = self.settings.value(
            "general/seestarAlpFlatsWaitSec",
            SEESTAR_ALP_DEFAULT_FLATS_WAIT_S,
            type=int,
        )
        alp_schedule_autofocus_legacy = self.settings.value(
            "general/seestarAlpScheduleAutofocusBeforeEachTarget",
            SEESTAR_ALP_DEFAULT_SCHEDULE_AUTOFOCUS,
            type=bool,
        )
        alp_schedule_autofocus_mode = normalize_seestar_alp_schedule_autofocus_mode(
            self.settings.value(
                "general/seestarAlpScheduleAutofocusMode",
                SEESTAR_ALP_DEFAULT_SCHEDULE_AUTOFOCUS_MODE,
                type=str,
            ),
            legacy_enabled=bool(alp_schedule_autofocus_legacy),
        )
        alp_schedule_autofocus_try_count = self.settings.value(
            "general/seestarAlpScheduleAutofocusTryCount",
            SEESTAR_ALP_DEFAULT_AUTOFOCUS_TRY_COUNT,
            type=int,
        )
        alp_dew_heater_value = self.settings.value(
            "general/seestarAlpDewHeaterValue",
            SEESTAR_ALP_DEFAULT_DEW_HEATER_VALUE,
            type=int,
        )
        alp_park_after_session = self.settings.value(
            "general/seestarAlpParkAfterSession",
            SEESTAR_ALP_DEFAULT_PARK_AFTER_SESSION,
            type=bool,
        )
        alp_shutdown_after_session = self.settings.value(
            "general/seestarAlpShutdownAfterSession",
            SEESTAR_ALP_DEFAULT_SHUTDOWN_AFTER_SESSION,
            type=bool,
        )
        session_template_key = self.settings.value(
            "general/seestarSessionTemplateKey",
            "",
            type=str,
        )
        session_repeat_count = self.settings.value(
            "general/seestarSessionRepeatCount",
            1,
            type=int,
        )
        session_minutes_per_run = self.settings.value(
            "general/seestarSessionMinutesPerRun",
            SEESTAR_DEFAULT_BLOCK_MINUTES,
            type=int,
        )
        session_gap_seconds = self.settings.value(
            "general/seestarSessionGapSeconds",
            0,
            type=int,
        )
        session_require_checklist = self.settings.value(
            "general/seestarSessionRequireChecklist",
            False,
            type=bool,
        )
        session_checklist_text = self.settings.value(
            "general/seestarSessionChecklistText",
            "",
            type=str,
        )
        session_notes = self.settings.value("general/seestarSessionTemplateNotes", "", type=str)
        try:
            alp_device_num = int(alp_device_num)
        except Exception:
            alp_device_num = SEESTAR_ALP_DEFAULT_DEVICE_NUM
        try:
            alp_client_id = int(alp_client_id)
        except Exception:
            alp_client_id = SEESTAR_ALP_DEFAULT_CLIENT_ID
        try:
            alp_timeout_s = float(alp_timeout_s)
        except Exception:
            alp_timeout_s = SEESTAR_ALP_DEFAULT_TIMEOUT_S
        try:
            alp_gain = int(alp_gain)
        except Exception:
            alp_gain = SEESTAR_ALP_DEFAULT_GAIN
        try:
            alp_panel_overlap = int(alp_panel_overlap)
        except Exception:
            alp_panel_overlap = SEESTAR_ALP_DEFAULT_PANEL_OVERLAP_PERCENT
        try:
            alp_num_tries = int(alp_num_tries)
        except Exception:
            alp_num_tries = SEESTAR_ALP_DEFAULT_NUM_TRIES
        try:
            alp_retry_wait_s = int(alp_retry_wait_s)
        except Exception:
            alp_retry_wait_s = SEESTAR_ALP_DEFAULT_RETRY_WAIT_S
        try:
            alp_stack_exposure_ms = int(alp_stack_exposure_ms)
        except Exception:
            alp_stack_exposure_ms = SEESTAR_ALP_DEFAULT_STACK_EXPOSURE_MS
        try:
            alp_flats_wait_s = int(alp_flats_wait_s)
        except Exception:
            alp_flats_wait_s = SEESTAR_ALP_DEFAULT_FLATS_WAIT_S
        try:
            alp_schedule_autofocus_try_count = int(alp_schedule_autofocus_try_count)
        except Exception:
            alp_schedule_autofocus_try_count = SEESTAR_ALP_DEFAULT_AUTOFOCUS_TRY_COUNT
        try:
            alp_dew_heater_value = int(alp_dew_heater_value)
        except Exception:
            alp_dew_heater_value = SEESTAR_ALP_DEFAULT_DEW_HEATER_VALUE
        try:
            session_repeat_count = int(session_repeat_count)
        except Exception:
            session_repeat_count = 1
        try:
            session_minutes_per_run = int(session_minutes_per_run)
        except Exception:
            session_minutes_per_run = SEESTAR_DEFAULT_BLOCK_MINUTES
        try:
            session_gap_seconds = int(session_gap_seconds)
        except Exception:
            session_gap_seconds = 0
        return {
            "method": method,
            "alp": SeestarAlpConfig(
                base_url=alp_base_url,
                device_num=max(0, alp_device_num),
                client_id=max(1, alp_client_id),
                timeout_s=max(1.0, alp_timeout_s),
                gain=max(0, alp_gain),
                panel_overlap_percent=max(0, alp_panel_overlap),
                use_autofocus=bool(alp_use_autofocus),
                num_tries=max(1, alp_num_tries),
                retry_wait_s=max(0, alp_retry_wait_s),
                target_integration_override_min=0,
                stack_exposure_ms=max(0, alp_stack_exposure_ms),
                lp_filter_mode=str(alp_lp_filter_mode or SEESTAR_ALP_LP_FILTER_AUTO).strip().lower(),
                honor_queue_times=bool(alp_honor_queue_times),
                wait_until_local_time=str(alp_wait_until_local_time or "").strip(),
                startup_enabled=bool(alp_startup_enabled),
                startup_polar_align=bool(alp_startup_polar_align),
                startup_auto_focus=bool(alp_startup_auto_focus),
                startup_dark_frames=bool(alp_startup_dark_frames),
                capture_flats_before_session=bool(alp_capture_flats),
                flats_wait_s=max(0, alp_flats_wait_s),
                schedule_autofocus_mode=alp_schedule_autofocus_mode,
                schedule_autofocus_before_each_target=(alp_schedule_autofocus_mode == SEESTAR_ALP_AF_MODE_PER_RUN),
                schedule_autofocus_try_count=max(1, alp_schedule_autofocus_try_count),
                dew_heater_value=alp_dew_heater_value if alp_dew_heater_value >= 0 else SEESTAR_ALP_DEFAULT_DEW_HEATER_VALUE,
                park_after_session=bool(alp_park_after_session),
                shutdown_after_session=bool(alp_shutdown_after_session),
            ),
            "template": SeestarSessionTemplate(
                key=str(session_template_key or "").strip(),
                name="",
                scope=SEESTAR_TEMPLATE_SCOPE_MULTI_TARGET,
                repeat_count=max(1, session_repeat_count),
                minutes_per_run=max(1, session_minutes_per_run),
                gap_seconds=max(0, session_gap_seconds),
                require_science_checklist=bool(session_require_checklist),
                science_checklist_items=[
                    line.strip()
                    for line in str(session_checklist_text or "").splitlines()
                    if line.strip()
                ],
                template_notes=str(session_notes or "").strip(),
                lp_filter_mode=str(alp_lp_filter_mode or SEESTAR_ALP_LP_FILTER_AUTO).strip().lower(),
                gain=max(0, alp_gain),
                panel_overlap_percent=max(0, alp_panel_overlap),
                use_autofocus=bool(alp_use_autofocus),
                num_tries=max(1, alp_num_tries),
                retry_wait_s=max(0, alp_retry_wait_s),
                target_integration_override_min=0,
                stack_exposure_ms=max(0, alp_stack_exposure_ms),
                honor_queue_times=bool(alp_honor_queue_times),
                wait_until_local_time=str(alp_wait_until_local_time or "").strip(),
                startup_enabled=bool(alp_startup_enabled),
                startup_polar_align=bool(alp_startup_polar_align),
                startup_auto_focus=bool(alp_startup_auto_focus),
                startup_dark_frames=bool(alp_startup_dark_frames),
                capture_flats_before_session=bool(alp_capture_flats),
                flats_wait_s=max(0, alp_flats_wait_s),
                schedule_autofocus_mode=alp_schedule_autofocus_mode,
                schedule_autofocus_before_each_target=(alp_schedule_autofocus_mode == SEESTAR_ALP_AF_MODE_PER_RUN),
                schedule_autofocus_try_count=max(1, alp_schedule_autofocus_try_count),
                dew_heater_value=alp_dew_heater_value if alp_dew_heater_value >= 0 else SEESTAR_ALP_DEFAULT_DEW_HEATER_VALUE,
                park_after_session=bool(alp_park_after_session),
                shutdown_after_session=bool(alp_shutdown_after_session),
            ),
        }

    def _update_seestar_button_tooltip(self) -> None:
        if not hasattr(self, "seestar_session_btn"):
            return
        cfg = self._seestar_config()
        backend_label = "ALP service" if str(cfg["method"]) == SEESTAR_METHOD_ALP else "guided handoff"
        template = cfg["template"]
        tooltip = (
            "Build a Seestar S50 session from the current Targets table order. "
            f"Backend: {backend_label}, defaults: {int(template.repeat_count)}x {int(template.minutes_per_run)} min, "
            f"gap {int(template.gap_seconds)} s."
        )
        self.seestar_session_btn.setToolTip(tooltip)
        if hasattr(self, "seestar_session_act"):
            self.seestar_session_act.setToolTip(tooltip)
            self.seestar_session_act.setStatusTip(tooltip)

    def _current_site_snapshot(self) -> NightQueueSiteSnapshot:
        selected_name = self.obs_combo.currentText() if hasattr(self, "obs_combo") else ""
        preset_site = self.observatories.get(selected_name) if hasattr(self, "observatories") else None
        tz_name = "UTC"
        if isinstance(getattr(self, "last_payload", None), dict):
            tz_name = str(self.last_payload.get("tz", "UTC"))
        elif preset_site is not None:
            tz_name = preset_site.timezone_name
        return NightQueueSiteSnapshot(
            name=selected_name or getattr(preset_site, "name", "") or "Current site",
            latitude=self._read_site_float(self.lat_edit),
            longitude=self._read_site_float(self.lon_edit),
            elevation=self._read_site_float(self.elev_edit),
            timezone=tz_name,
            limiting_magnitude=self._current_limiting_magnitude(),
            telescope_diameter_mm=float(getattr(preset_site, "telescope_diameter_mm", 0.0) or 0.0),
            focal_length_mm=float(getattr(preset_site, "focal_length_mm", 0.0) or 0.0),
            pixel_size_um=float(getattr(preset_site, "pixel_size_um", 0.0) or 0.0),
            detector_width_px=int(getattr(preset_site, "detector_width_px", 0) or 0),
            detector_height_px=int(getattr(preset_site, "detector_height_px", 0) or 0),
        )

    def _seestar_night_bounds(self) -> Optional[tuple[datetime, datetime, str]]:
        payload = self.last_payload if isinstance(getattr(self, "last_payload", None), dict) else None
        if payload is None or "times" not in payload:
            return None
        tz_name = str(payload.get("tz", "UTC"))
        try:
            tz = pytz.timezone(tz_name)
        except Exception:
            tz = pytz.UTC
            tz_name = "UTC"
        try:
            times = [t.astimezone(tz) for t in mdates.num2date(payload["times"])]
        except Exception:
            return None
        if not times:
            return None
        sun_alt_series = np.array(payload.get("sun_alt", np.full(len(times), np.nan)), dtype=float)
        if sun_alt_series.shape[0] != len(times):
            return None
        sun_mask = np.isfinite(sun_alt_series) & (sun_alt_series <= self._sun_alt_limit())
        if not sun_mask.any():
            return None
        indices = np.where(sun_mask)[0]
        start_idx = int(indices[0])
        end_idx = min(int(indices[-1]) + 1, len(times) - 1)
        start_dt = times[start_idx]
        end_dt = times[end_idx]
        if end_dt <= start_dt:
            return None
        return start_dt, end_dt, tz_name

    @Slot()
    def _open_seestar_session(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(self, "Seestar Session", "Wait for the current visibility calculation to finish.")
            return
        night_bounds = self._seestar_night_bounds()
        if night_bounds is None:
            QMessageBox.information(
                self,
                "Seestar Session",
                "Run a visibility calculation first so the Seestar queue uses the current night bounds.",
            )
            return
        session_dialog = SeestarSessionPlanDialog(self)
        if session_dialog.exec() != int(QDialog.Accepted):
            return
        self._update_seestar_button_tooltip()
        night_start, night_end, tz_name = night_bounds
        method = str(session_dialog.method() or SEESTAR_METHOD_GUIDED).strip().lower()
        session_template = session_dialog.session_template()
        session_items = [item for item in session_dialog.session_items() if item.enabled]
        alp_config = session_dialog.alp_config()
        if not session_items:
            QMessageBox.information(self, "Seestar Session", "Enable at least one target in the session table.")
            return
        device_profile = SEESTAR_DEVICE_PROFILE_ALP if method == SEESTAR_METHOD_ALP else SEESTAR_DEVICE_PROFILE
        site_snapshot = self._current_site_snapshot()
        if (
            str(session_template.scope or SEESTAR_TEMPLATE_SCOPE_MULTI_TARGET) == SEESTAR_TEMPLATE_SCOPE_SINGLE_TARGET
            and len(session_items) != 1
        ):
            QMessageBox.information(
                self,
                "Seestar Session",
                "The selected template is single-target. Enable exactly one target in the session table.",
            )
            return
        if session_template.require_science_checklist and not session_template.science_checklist_items:
            session_template = session_template.model_copy(update={"science_checklist_items": default_science_checklist_items()})
        queue = build_session_queue(
            session_items,
            session_template=session_template,
            site_snapshot=site_snapshot,
            night_start_local=night_start,
            night_end_local=night_end,
            timezone=tz_name,
            device_profile=device_profile,
            start_cursor_local=datetime.now(night_start.tzinfo) if night_start.tzinfo is not None else datetime.now(),
        )
        if not queue.blocks:
            QMessageBox.information(self, "Seestar Session", "No session blocks were generated from the current Targets table.")
            return
        adapter = (
            SeestarAlpAdapter(alp_config)
            if method == SEESTAR_METHOD_ALP
            else SeestarGuidedAdapter()
        )
        bundle = adapter.build_handoff_bundle(queue)
        adapter.open_handoff_dialog(bundle, parent=self)

    def _observatories_config_path(self) -> Path:
        return self._observatory_coordinator.observatories_config_path()

    def _update_obs_combo_widths(self):
        self._observatory_coordinator.update_obs_combo_widths()

    def _parse_custom_observatories_payload(self, payload: object) -> tuple[dict[str, Site], dict[str, str]]:
        return self._observatory_coordinator.parse_custom_observatories_payload(payload)

    def _load_custom_observatories(self) -> tuple[dict[str, Site], dict[str, str]]:
        return self._observatory_coordinator.load_custom_observatories()

    def _save_custom_observatories(
        self,
        custom_sites: Optional[dict[str, Site]] = None,
        *,
        preset_keys: Optional[dict[str, str]] = None,
    ):
        self._observatory_coordinator.save_custom_observatories(
            custom_sites,
            preset_keys=preset_keys,
        )

    def _refresh_observatory_combo(
        self,
        selected_name: Optional[str] = None,
        *,
        emit_selection_change: bool = False,
    ):
        self._observatory_coordinator.refresh_observatory_combo(
            selected_name=selected_name,
            emit_selection_change=emit_selection_change,
        )

    def _lookup_observatory_coordinates(self, query: str) -> tuple[float, float, Optional[float], str]:
        return self._observatory_coordinator.lookup_observatory_coordinates(query)

    @Slot()
    def _open_observatory_manager(self):
        current_name = self.obs_combo.currentText() if hasattr(self, "obs_combo") else ""
        dlg = ObservatoryManagerDialog(
            self.observatories,
            self,
            preset_keys=getattr(self, "_observatory_preset_keys", {}),
            selected_name=current_name,
        )
        if dlg.exec() != QDialog.Accepted:
            return
        updated = dlg.observatories()
        updated_preset_keys = dlg.preset_keys()
        if not updated:
            QMessageBox.warning(self, "No Observatories", "At least one observatory must remain configured.")
            return
        previous = self.obs_combo.currentText() if hasattr(self, "obs_combo") else ""
        self.observatories = updated
        self._observatory_preset_keys = {
            name: str(updated_preset_keys.get(name, "custom") or "custom")
            for name in self.observatories
        }
        self._save_custom_observatories(self.observatories, preset_keys=self._observatory_preset_keys)
        selected = previous if previous in self.observatories else next(iter(self.observatories.keys()), "")
        self._refresh_observatory_combo(selected_name=selected)
        if selected:
            self.settings.setValue("general/defaultSite", selected)
            self._on_obs_change(selected)

    @Slot()
    def _open_add_observatory_dialog(self):
        template_site = self.observatories.get(self.obs_combo.currentText())
        dlg = AddObservatoryDialog(
            existing_names=set(self.observatories.keys()),
            template_site=template_site,
            default_limiting_magnitude=self._current_limiting_magnitude(),
            lookup_resolver=self._lookup_observatory_coordinates,
            parent=self,
        )
        if dlg.exec() != QDialog.Accepted:
            return
        site = dlg.site()
        if site is None:
            return
        name = site.name
        self.observatories[name] = site
        self._observatory_preset_keys[name] = "custom"
        self._save_custom_observatories()
        self._refresh_observatory_combo(selected_name=name)
        self.settings.setValue("general/defaultSite", name)
        self._on_obs_change(name)

    def __init__(self):
        super().__init__()
        self.setObjectName(self.__class__.__name__)
        self._ensure_display_font_loaded()
        fallback_family = _preferred_display_font_family()
        if fallback_family not in {"", "Sans Serif"}:
            QFont.insertSubstitution("Sans Serif", fallback_family)
            QFont.insertSubstitution("SF Pro Text", fallback_family)
            QFont.insertSubstitution("SF Pro Display", fallback_family)
            bootstrap_font = QFont(fallback_family)
            bootstrap_font.setPointSize(11)
            QApplication.setFont(bootstrap_font)
        # Initialize clock_worker early so callbacks can safely reference it
        self.clock_worker = None
        # Keep references to every ClockWorker we create so they can all be stopped
        self._clock_workers: list[ClockWorker] = []
        self.last_payload = None
        # Prevent new workers once we start shutting down
        self._shutting_down = False
        self.setWindowTitle("Astronomical Observation Planner")
        self.resize(1360, 820)
        self.setMinimumSize(1180, 720)

        # Persistent user settings
        self.settings = _create_app_settings()
        self.app_storage = getattr(self.settings, "storage", None)
        self._observatory_coordinator = ObservatoryCoordinator(self)
        try:
            self._system_locale_name = str(QLocale.system().name() or "")
        except Exception:
            self._system_locale_name = str(locale.getdefaultlocale()[0] or "")
        self._ui_language_choice = str(self.settings.value("general/uiLanguage", "system", type=str) or "system")
        self._ui_language = set_current_language(self._ui_language_choice, system_locale=self._system_locale_name)
        self._workspace_plan_id = ""
        self._active_plan_id = ""
        self._active_plan_kind = "workspace"
        self._active_plan_name = "Workspace"
        self._plan_autosave_timer = QTimer(self)
        self._plan_autosave_timer.setSingleShot(True)
        self._plan_autosave_timer.timeout.connect(self._flush_plan_autosave)
        self._suspend_plan_autosave = False
        self._plan_coordinator = PlanCoordinator(self)
        self._defer_startup_preview_updates = False
        self._startup_restore_pending = True
        self._obs_change_finalize_pending = False
        logger.info("Using settings file: %s", self.settings.fileName())
        self._migrate_table_settings_schema()
        self._dark_enabled = self.settings.value("general/darkMode", DEFAULT_DARK_MODE, type=bool)
        self._theme_name = normalize_theme_key(self.settings.value("general/uiTheme", DEFAULT_UI_THEME, type=str))
        self._ui_font_size = _sanitize_ui_font_size(self.settings.value("general/uiFontSize", 11, type=int))
        self._display_font_family = _preferred_display_font_family()
        startup_font = QFont(self._display_font_family)
        startup_font.setPointSize(self._ui_font_size)
        QApplication.setFont(startup_font)

        # ── Polar-path visibility flags (persisted) ───────────────────────────
        self.show_sun_path  = self.settings.value("general/showSunPath",  True, type=bool)
        self.show_moon_path = self.settings.value("general/showMoonPath", True, type=bool)
        self.show_obj_path  = self.settings.value("general/showObjPath",  True, type=bool)
        
        # state holders
        self.targets: List[Target] = []
        self._auto_target_color_map: dict[str, str] = {}
        self._auto_target_color_palette_signature: tuple[str, ...] = tuple()
        self.worker: Optional[AstronomyWorker] = None  # keep reference!
        self.target_metrics: dict[str, TargetNightMetrics] = {}
        self.target_windows: dict[str, tuple[datetime, datetime]] = {}
        self._simbad_meta_cache: dict[str, tuple[Optional[float], str]] = {}
        self._simbad_compact_cache: dict[str, dict[str, object]] = {}
        self._knowledge_notes_cache: Optional[list[KnowledgeNote]] = None
        self._gaia_alerts_cache: dict[str, dict[str, str]] = {}
        self._gaia_alerts_cache_loaded_at = 0.0
        self._meta_worker: Optional[MetadataLookupWorker] = None
        self._meta_request_id = 0
        self._target_resolver = TargetResolver(self)
        self._calc_started_at = 0.0
        self._last_calc_stats = CalcRunStats(duration_s=0.0, visible_targets=0, total_targets=0)
        self._queued_plan_run = False
        self._pending_visibility_context_key = ""
        self._clock_polar_tick = 0
        self.color_blind_mode = self.settings.value("general/colorBlindMode", False, type=bool)
        self._accent_secondary_override = self._load_accent_secondary_override(self._theme_name)
        self._radar_sweep_enabled = self.settings.value("general/radarSweepAnimation", False, type=bool)
        self._radar_sweep_speed = max(40, min(260, self.settings.value("general/radarSweepSpeed", 140, type=int)))
        self._radar_sweep_angle = 0.0
        self._plot_airmass = self.settings.value("general/plotAirmass", False, type=bool)
        self._plot_mode_animations: list[QPropertyAnimation] = []
        self._cutout_view_key = _normalize_cutout_view_key(
            self.settings.value("general/cutoutView", CUTOUT_DEFAULT_VIEW_KEY, type=str)
        )
        self._cutout_survey_key = _normalize_cutout_survey_key(
            self.settings.value("general/cutoutSurvey", CUTOUT_DEFAULT_SURVEY_KEY, type=str)
        )
        self._cutout_fov_arcmin = _sanitize_cutout_fov_arcmin(
            self.settings.value("general/cutoutFovArcmin", CUTOUT_DEFAULT_FOV_ARCMIN, type=int)
        )
        self._cutout_size_px = _sanitize_cutout_size_px(
            self.settings.value("general/cutoutSizePx", CUTOUT_DEFAULT_SIZE_PX, type=int)
        )
        self._aladin_context_factor = 1.0
        self.llm_config = LLMConfig(
            url=self.settings.value("llm/serverUrl", LLMConfig.DEFAULT_URL, type=str),
            model=self.settings.value("llm/model", LLMConfig.DEFAULT_MODEL, type=str),
            timeout_s=self.settings.value("llm/timeoutSec", LLMConfig.DEFAULT_TIMEOUT_S, type=int),
            max_tokens=self.settings.value("llm/maxTokens", LLMConfig.DEFAULT_MAX_TOKENS, type=int),
            enable_thinking=self.settings.value("llm/enableThinking", LLMConfig.DEFAULT_ENABLE_THINKING, type=bool),
            enable_chat_memory=self.settings.value(
                "llm/enableChatMemory",
                LLMConfig.DEFAULT_ENABLE_CHAT_MEMORY,
                type=bool,
            ),
        )
        self._llm_chat_font_size_pt = max(
            9,
            int(self.settings.value("llm/chatFontSizePt", LLMConfig.DEFAULT_CHAT_FONT_PT, type=int)),
        )
        self._ai_chat_spacing = str(
            self.settings.value("llm/chatSpacing", LLMConfig.DEFAULT_CHAT_SPACING, type=str)
            or LLMConfig.DEFAULT_CHAT_SPACING
        ).strip().lower()
        self._ai_chat_tint_strength = str(
            self.settings.value("llm/chatTintStrength", LLMConfig.DEFAULT_CHAT_TINT_STRENGTH, type=str)
            or LLMConfig.DEFAULT_CHAT_TINT_STRENGTH
        ).strip().lower()
        self._ai_chat_width = str(
            self.settings.value("llm/chatMessageWidth", LLMConfig.DEFAULT_CHAT_WIDTH, type=str)
            or LLMConfig.DEFAULT_CHAT_WIDTH
        ).strip().lower()
        self._ai_status_error_clear_s = max(
            0,
            int(self.settings.value("llm/statusErrorClearSec", LLMConfig.DEFAULT_STATUS_ERROR_CLEAR_S, type=int)),
        )
        self._llm_worker: Optional[LLMWorker] = None
        self._llm_active_tag = ""
        self._llm_active_requested_class = ""
        self._llm_active_primary_target: Optional[Target] = None
        self._llm_warmup_silent = False
        self._llm_startup_warmup_attempted = False
        self._ai_last_knowledge_titles: list[str] = []
        self._llm_last_warmup_at = 0.0
        self._llm_last_warmup_key: tuple[str, str] = ("", "")
        self._ai_runtime_status = ""
        self._ai_runtime_status_tone = "info"
        self._ai_messages: list[dict[str, Any]] = []
        self._ai_message_widget_refs: list[dict[str, Any]] = []
        self._ai_stream_message_index: Optional[int] = None
        self._ai_scroll_bottom_pending = False
        self._ai_stream_render_timer = QTimer(self)
        self._ai_stream_render_timer.setSingleShot(True)
        self._ai_stream_render_timer.timeout.connect(self._flush_ai_stream_render)
        self._ai_status_clear_timer = QTimer(self)
        self._ai_status_clear_timer.setSingleShot(True)
        self._ai_status_clear_timer.timeout.connect(self._clear_ai_runtime_status)
        self._ai_context_coordinator = AIContextCoordinator(self)
        self._ai_panel_coordinator = AIPanelCoordinator(self)
        self._preview_coordinator = PreviewCoordinator(self)
        self._cutout_manager = QNetworkAccessManager(self)
        self._cutout_manager.finished.connect(self._on_cutout_reply)
        self._cutout_request_id = 0
        self._cutout_pending_key = ""
        self._cutout_pending_name = ""
        self._cutout_pending_background = False
        self._cutout_displayed_key = ""
        self._cutout_reply: Optional[QNetworkReply] = None
        self._cutout_cache: dict[str, QPixmap] = {}
        self._cutout_cache_order: list[str] = []
        self._cutout_prefetch_queue: list[tuple[Target, str, int, int, int, int]] = []
        self._cutout_prefetch_enqueued_keys: set[str] = set()
        self._cutout_prefetch_total = 0
        self._cutout_prefetch_completed = 0
        self._cutout_prefetch_cached = 0
        self._cutout_prefetch_active = False
        self._finder_cache: dict[str, QPixmap] = {}
        self._finder_cache_order: list[str] = []
        self._bhtom_candidate_cache_key: Optional[tuple[str, str]] = None
        self._bhtom_candidate_cache: Optional[list[dict[str, object]]] = None
        self._bhtom_candidate_cache_loaded_at = 0.0
        self._bhtom_last_network_fetch_key: Optional[tuple[str, str]] = None
        self._bhtom_ranked_suggestions_cache: list[dict[str, object]] = []
        self._bhtom_candidate_prefetch_worker: Optional[BhtomCandidatePrefetchWorker] = None
        self._bhtom_candidate_prefetch_request_id = 0
        self._bhtom_observatory_cache_key: Optional[tuple[str, str]] = None
        self._bhtom_observatory_cache: Optional[list[dict[str, object]]] = None
        self._bhtom_observatory_cache_loaded_at = 0.0
        self._bhtom_observatory_worker: Optional[BhtomObservatoryPresetWorker] = None
        self._bhtom_observatory_worker_request_id = 0
        self._bhtom_observatory_loading_message = ""
        self._observatory_preset_keys: dict[str, str] = {}
        self._bhtom_worker: Optional[BhtomSuggestionWorker] = None
        self._bhtom_worker_request_id = 0
        self._bhtom_worker_mode = ""
        self._bhtom_worker_cache_key: Optional[tuple[str, str]] = None
        self._bhtom_worker_source = ""
        self._bhtom_dialog: Optional[SuggestedTargetsDialog] = None
        self._bhtom_coordinator = BhtomCoordinator(self)
        self._finder_workers: list[FinderChartWorker] = []
        self._finder_worker: Optional[FinderChartWorker] = None
        self._finder_request_id = 0
        self._finder_pending_key = ""
        self._finder_pending_name = ""
        self._finder_pending_background = False
        self._finder_displayed_key = ""
        self._finder_prefetch_queue: list[tuple[Target, str]] = []
        self._finder_prefetch_enqueued_keys: set[str] = set()
        self._finder_prefetch_total = 0
        self._finder_prefetch_completed = 0
        self._finder_prefetch_cached = 0
        self._finder_prefetch_active = False
        self._finder_retry_after: dict[str, float] = {}
        self._cutout_image_stacks: dict[str, QStackedLayout] = {}
        self._cutout_image_placeholders: dict[str, QWidget] = {}
        self._cutout_image_labels: dict[str, CoverImageLabel] = {}
        self._finder_timeout_timer = QTimer(self)
        self._finder_timeout_timer.setSingleShot(True)
        self._finder_timeout_timer.setInterval(FINDER_WORKER_TIMEOUT_MS)
        self._finder_timeout_timer.timeout.connect(self._on_finder_chart_timeout)
        self._startup_weather_worker: Optional[WeatherLiveWorker] = None
        self._cutout_resize_timer = QTimer(self)
        self._cutout_resize_timer.setSingleShot(True)
        self._cutout_resize_timer.setInterval(260)
        self._cutout_resize_timer.timeout.connect(self._on_cutout_resize_timeout)
        self._cutout_last_resize_signature: Optional[tuple] = None
        self._visibility_matplotlib = VisibilityMatplotlibCoordinator(self)
        self._visibility_coordinator = VisibilityCoordinator(self)
        self._visibility_coordinator.bind()
        self._main_window_coordinator = MainWindowCoordinator(self)

        # UI ------------------------------------------------
        self.table_model = TargetTableModel(self.targets, site=None, parent=self)
        self.table_view = TargetTableView()
        self._table_coordinator = TargetTableCoordinator(self)
        self._table_coordinator.bind()

        # Polar plot for alt-az projection
        self.polar_canvas = FigureCanvas(Figure(figsize=(4, 4)))
        self.polar_ax = self.polar_canvas.figure.add_subplot(projection='polar')
        self.polar_ax.set_theta_zero_location('N')
        self.polar_ax.set_theta_direction(-1)
        # Plot placeholders: targets, selected target, sun, moon
        self.polar_scatter = self.polar_ax.scatter([], [], c='blue', marker='o', s=10, label='Targets', alpha=0.5, picker=True)
        self.selected_scatter = self.polar_ax.scatter([], [], c='red', marker='x', s=40, alpha=1, label='Selected')
        self.radar_echo_glow_scatter = self.polar_ax.scatter([], [], c='cyan', marker='o', s=10, alpha=1.0, linewidths=3.2, zorder=3)
        self.radar_echo_scatter = self.polar_ax.scatter([], [], c='cyan', marker='o', s=10, alpha=1.0, linewidths=1.35, zorder=4)
        self.radar_echo_glow_scatter.set_visible(False)
        self.radar_echo_scatter.set_visible(False)
        self._radar_echo_artists: list[Any] = []
        self.radar_sweep_glow_line, = self.polar_ax.plot(
            [],
            [],
            color='cyan',
            linewidth=3.2,
            alpha=0.42,
            zorder=2,
            solid_capstyle='round',
            solid_joinstyle='round',
            antialiased=True,
        )
        self.radar_sweep_line, = self.polar_ax.plot(
            [],
            [],
            color='magenta',
            linewidth=0.9,
            alpha=0.0,
            zorder=3,
            solid_capstyle='round',
            solid_joinstyle='round',
            antialiased=True,
        )
        self.radar_sweep_core = self.polar_ax.scatter([], [], c='magenta', marker='o', s=58, alpha=0.0, linewidths=0.0, zorder=3)
        self.radar_sweep_core.set_visible(False)
        self._radar_sweep_theta_edges = np.linspace(0.0, 2.0 * math.pi, 721)
        self._radar_sweep_theta_centers = (self._radar_sweep_theta_edges[:-1] + self._radar_sweep_theta_edges[1:]) * 0.5
        self._radar_sweep_radius_edges = np.linspace(0.0, 90.0, 9)
        self._radar_sweep_mesh_values = np.zeros(
            (len(self._radar_sweep_radius_edges) - 1, len(self._radar_sweep_theta_edges) - 1),
            dtype=float,
        )
        self.radar_sweep_mesh = self.polar_ax.pcolormesh(
            self._radar_sweep_theta_edges,
            self._radar_sweep_radius_edges,
            self._radar_sweep_mesh_values,
            shading='flat',
            cmap=self._build_radar_sweep_cmap(),
            vmin=0.0,
            vmax=1.0,
            antialiased=False,
            zorder=1,
        )
        self.radar_sweep_mesh.set_visible(False)
        # Placeholder for selected-object path trace
        self.selected_trace_line = None
        # Placeholder for altitude limit circle
        self.limit_circle = None
        # Placeholder for celestial pole marker
        self.pole_marker = None
        self.sun_marker = self.polar_ax.scatter([], [], c='orange', marker='o', s=100, label='Sun')
        self.moon_marker = self.polar_ax.scatter([], [], c='silver', marker='o', s=100, label='Moon')
        # Place-holders for path lines
        self.sun_path_line  = None
        self.moon_path_line = None
        self.polar_ax.set_rlim(0, 90)
        self.polar_ax.set_rlabel_position(135)
        # Will map scatter points to target indices for picking
        self.polar_indices: list[int] = []
        # Label cardinal directions on the polar plot
        self.polar_ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
        self.polar_ax.set_xticklabels(['N', 'E', 'S', 'W'])
        # Draw cardinal labels in white for visibility on dark background
        self.polar_ax.tick_params(axis='x', colors='white', pad=0)
        # Radial ticks for altitudes 20°, 40°, 60°, and 80° (r = 90 - altitude), labels hidden
        alt_ticks = [20, 40, 60, 80]
        r_ticks = [90 - a for a in alt_ticks]
        self.polar_ax.set_yticks(r_ticks)
        # Hide radial tick labels for a cleaner look
        self.polar_ax.set_yticklabels([])
        self.polar_ax.tick_params(axis='y', pad=1)
        # Keep the polar axes background white
        self.polar_ax.set_facecolor('white')
        self.polar_canvas.figure.patch.set_alpha(0)
        # Make the canvas widget itself transparent
        self.polar_canvas.setAttribute(Qt.WA_TranslucentBackground)
        self.polar_canvas.setStyleSheet("background: transparent;")
        # Update polar when table selection changes
        self.table_view.selectionModel().selectionChanged.connect(self._update_polar_selection)
        # Also highlight visibility curves on selection
        self.table_view.selectionModel().selectionChanged.connect(self._update_vis_selection)
        self.table_view.selectionModel().selectionChanged.connect(self._update_selected_details)
        # Holder for visibility line artists per target, with over-limit flag
        self.vis_lines: list[tuple[str, Any, bool]] = []
        self.limit_line = None
        # Connect pick event for polar scatter
        self.polar_canvas.mpl_connect('pick_event', self._on_polar_pick)
        self.polar_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.polar_canvas.setMinimumWidth(250)
        self.polar_canvas.setMinimumHeight(220)
        self.polar_canvas.setMaximumWidth(16777215)
        # Keep enough edge room so cardinal labels are not clipped, but let the
        # polar plot fill more of the available canvas area on large screens.
        self.polar_canvas.figure.subplots_adjust(left=0.03, right=0.97, bottom=0.06, top=0.97)
        self._radar_target_coords = np.empty((0, 2), dtype=float)
        self._radar_echo_strengths = np.zeros(0, dtype=float)
        self._radar_sweep_timer = QTimer(self)
        self._radar_sweep_timer.setInterval(16)
        try:
            self._radar_sweep_timer.setTimerType(Qt.PreciseTimer)
        except Exception:
            pass
        self._radar_sweep_timer.timeout.connect(self._advance_radar_sweep)
        self._radar_sweep_clock = QElapsedTimer()

        # Debounce frequent requests for plotting
        self._replot_timer = QTimer(self)
        self._replot_timer.setSingleShot(True)
        self._replot_timer.setInterval(300)  # ms
        self._replot_timer.timeout.connect(self._run_plan)

        self.date_edit = QDateEdit()
        # Initialize to current observing night
        self._change_to_today()
        self.date_edit.setCalendarPopup(True)
        self.date_edit.setMinimumWidth(130)
        self.date_edit.setMaximumWidth(130)

        # Site coordinate inputs and limit-altitude
        self.lat_edit = QLineEdit("-24.59", self)
        self.lat_edit.setMinimumWidth(92)
        self.lat_edit.setMaximumWidth(95)
        self.lat_edit.setValidator(QDoubleValidator(-90.0, 90.0, 6, self))
        self.lon_edit = QLineEdit("-70.19", self)
        self.lon_edit.setMinimumWidth(92)
        self.lon_edit.setMaximumWidth(95)
        self.lon_edit.setValidator(QDoubleValidator(-180.0, 180.0, 6, self))
        self.elev_edit = QLineEdit("2800", self)
        self.elev_edit.setMinimumWidth(70)
        self.elev_edit.setMaximumWidth(70)
        self.elev_edit.setValidator(QDoubleValidator(-1000.0, 20000.0, 2, self))
        self.limit_spin = QSpinBox(minimum=-90, maximum=90, value=35)
        self.limit_spin.setMinimumWidth(74)
        self.limit_spin.setMaximumWidth(74)
        init_sun_alt_limit = self.settings.value("general/sunAltLimit", -10, type=int)
        self.sun_alt_limit_spin = QSpinBox(minimum=-90, maximum=90, value=init_sun_alt_limit)
        self.sun_alt_limit_spin.setMinimumWidth(84)
        self.sun_alt_limit_spin.setMaximumWidth(90)
        self.sun_alt_limit_spin.setToolTip("Sun altitude threshold for start/end of observing window (deg)")

        # Observatory selection
        self._builtin_observatories = {
            "OCM": Site(
                name="OCM",
                latitude=-24.59,
                longitude=-70.19,
                elevation=2800,
                limiting_magnitude=DEFAULT_LIMITING_MAGNITUDE,
            ),
            "Białków": Site(
                name="Białków",
                latitude=51.474248,
                longitude=16.657821,
                elevation=128,
                limiting_magnitude=DEFAULT_LIMITING_MAGNITUDE,
            ),
            "Roque de los Muchachos": Site(
                name="Roque de los Muchachos",
                latitude=28.4522,
                longitude=-17.5330,
                elevation=2426,
                limiting_magnitude=DEFAULT_LIMITING_MAGNITUDE,
            ),
        }
        loaded_observatories, loaded_preset_keys = self._load_custom_observatories()
        if loaded_observatories:
            # SQLite is the source of truth for user observatories.
            self.observatories = dict(loaded_observatories)
            self._observatory_preset_keys = {
                name: str(loaded_preset_keys.get(name, "custom") or "custom")
                for name in self.observatories
            }
        else:
            self.observatories = dict(self._builtin_observatories)
            self._observatory_preset_keys = {name: "custom" for name in self.observatories}
            self._save_custom_observatories(self.observatories, preset_keys=self._observatory_preset_keys)
        self.obs_combo = QComboBox()
        self.obs_combo.addItems(self.observatories.keys())
        init_site = self.settings.value("general/defaultSite", "OCM", type=str)
        if init_site not in self.observatories:
            init_site = "OCM" if "OCM" in self.observatories else (next(iter(self.observatories.keys()), ""))
        self.obs_combo.blockSignals(True)
        self.obs_combo.setCurrentText(init_site)
        self.obs_combo.blockSignals(False)
        self.obs_combo.currentTextChanged.connect(self._on_obs_change)
        # Prime the initial site fields without kicking background work before
        # workspace restore decides what the real startup context should be.
        if init_site:
            self._prime_selected_observatory(init_site)
        self.obs_combo.setMinimumContentsLength(26)
        self.obs_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.obs_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._update_obs_combo_widths()
        self.add_obs_btn = QToolButton()
        self.add_obs_btn.setText("+")
        self.add_obs_btn.setToolTip("Open observatory manager")
        self.add_obs_btn.setFixedSize(24, 24)
        self.add_obs_btn.clicked.connect(self._open_observatory_manager)

        # Debounced connections for date and limit spin
        self.date_edit.dateChanged.connect(lambda _: self._replot_timer.start())
        self.date_edit.dateChanged.connect(lambda _: self._refresh_weather_window_context())
        self.date_edit.dateChanged.connect(lambda *_: self._schedule_plan_autosave())
        self.limit_spin.valueChanged.connect(self._update_limit)
        self.sun_alt_limit_spin.valueChanged.connect(self._on_sun_alt_limit_changed)
        # Re-plot when latitude, longitude, or elevation is changed
        self.lat_edit.editingFinished.connect(self._on_site_inputs_changed)
        self.lon_edit.editingFinished.connect(self._on_site_inputs_changed)
        self.elev_edit.editingFinished.connect(self._on_site_inputs_changed)

        self._use_visibility_web = bool(_HAS_QTWEBENGINE and _HAS_PLOTLY and QWebEngineView is not None)
        self._visibility_web_minute_key = ""
        self._visibility_web_has_content = False
        self.plot_canvas = None
        self.ax_alt = None
        self.plot_toolbar = None
        self.visibility_web = None
        self.visibility_loading_widget = None
        self.visibility_plot_stack = None
        self.visibility_plot_host = None
        self._visibility_plot_widgets_ready = False
        self._visibility_bootstrap_placeholder = self._build_web_loading_placeholder(
            "Visibility Plot",
            "Preparing chart…",
        )
        self._visibility_bootstrap_placeholder.setMinimumHeight(320)

        self.plot_mode_widget = QWidget(self)
        self.plot_mode_widget.setObjectName("PlotModeWidget")
        self.plot_mode_widget.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        plot_mode_layout = QHBoxLayout(self.plot_mode_widget)
        plot_mode_layout.setContentsMargins(12, 2, 12, 2)
        plot_mode_layout.setSpacing(12)
        self.plot_mode_alt_label = QLabel("Altitude", self.plot_mode_widget)
        self.plot_mode_airmass_label = QLabel("Airmass", self.plot_mode_widget)
        for label in (self.plot_mode_alt_label, self.plot_mode_airmass_label):
            effect = QGraphicsOpacityEffect(label)
            effect.setOpacity(1.0)
            label.setGraphicsEffect(effect)
            label._opacity_effect = effect  # type: ignore[attr-defined]
        self.airmass_toggle_btn = NeonToggleSwitch(self.plot_mode_widget)
        self.airmass_toggle_btn.setObjectName("PlotModeSwitch")
        self.airmass_toggle_btn.setToolTip("Switch the main plot Y-axis between altitude and airmass")
        self.airmass_toggle_btn.setChecked(bool(self._plot_airmass))
        self.airmass_toggle_btn.toggled.connect(self._on_plot_mode_switch_changed)
        plot_mode_layout.addWidget(self.plot_mode_alt_label)
        plot_mode_layout.addWidget(self.airmass_toggle_btn)
        plot_mode_layout.addWidget(self.plot_mode_airmass_label)
        self._refresh_plot_mode_switch()
        self._update_plot_mode_label_metrics()
        if self._use_visibility_web:
            self.plot_controls_bar = QWidget(self)
            self.plot_controls_bar.setObjectName("PlotControlsBar")
            plot_controls_layout = QHBoxLayout(self.plot_controls_bar)
            plot_controls_layout.setContentsMargins(6, 2, 6, 2)
            plot_controls_layout.setSpacing(8)
            self.plot_hint_label = QLabel(
                "Interactive visibility chart. Use Plotly controls in the top-right for zoom and pan.",
                self.plot_controls_bar,
            )
            self.plot_hint_label.setObjectName("SectionHint")
            plot_controls_layout.addWidget(self.plot_hint_label, 1)
            plot_controls_layout.addWidget(self.plot_mode_widget, 0, Qt.AlignRight)
        else:
            self.plot_controls_bar = None
            self.plot_hint_label = None

        # top controls
        # Date selector with previous/next day buttons
        self.prev_day_btn = QToolButton()
        self.prev_day_btn.setObjectName("DateNavButton")
        self.prev_day_btn.setToolTip("Previous day")
        self.prev_day_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.prev_day_btn.setIconSize(QSize(17, 17))
        self.prev_day_btn.setFixedSize(30, 30)
        self.prev_day_btn.clicked.connect(lambda: self._change_date(-1))

        self.today_btn = QToolButton()
        self.today_btn.setObjectName("DateNavButton")
        self.today_btn.setToolTip("Today")
        self.today_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.today_btn.setIconSize(QSize(17, 17))
        self.today_btn.setFixedSize(30, 30)
        self.today_btn.clicked.connect(lambda: self._change_to_today())

        self.next_day_btn = QToolButton()
        self.next_day_btn.setObjectName("DateNavButton")
        self.next_day_btn.setToolTip("Next day")
        self.next_day_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.next_day_btn.setIconSize(QSize(17, 17))
        self.next_day_btn.setFixedSize(30, 30)
        self.next_day_btn.clicked.connect(lambda: self._change_date(1))
        self._refresh_date_nav_icons()

        # --- Compact date chooser ---
        date_widget = QWidget()
        date_layout = QHBoxLayout()
        date_layout.setContentsMargins(0, 0, 0, 0)
        date_layout.setSpacing(4)
        date_layout.addWidget(self.prev_day_btn)
        date_layout.addWidget(self.today_btn)
        date_layout.addWidget(self.date_edit)
        date_layout.addWidget(self.next_day_btn)
        date_widget.setLayout(date_layout)

        self.coord_error_label = QLabel("")
        self.coord_error_label.setWordWrap(True)
        _set_label_tone(self.coord_error_label, "error")

        session_layout = QVBoxLayout()
        session_layout.setContentsMargins(2, 2, 2, 2)
        session_layout.setSpacing(3)
        session_row_top = QHBoxLayout()
        session_row_top.setSpacing(5)
        session_row_top.addWidget(QLabel("Obs"))
        session_row_top.addWidget(self.obs_combo, 2)
        session_row_top.addWidget(self.add_obs_btn)
        session_row_top.addWidget(QLabel("Date"))
        session_row_top.addWidget(date_widget, 1)
        session_row_top.addWidget(QLabel("Alt"))
        session_row_top.addWidget(self.limit_spin)
        session_row_top.addWidget(QLabel("Sun ≤"))
        session_row_top.addWidget(self.sun_alt_limit_spin)
        session_row_top.addStretch(1)
        session_layout.addLayout(session_row_top)

        # Coordinates are managed by observatory selection/add dialog.
        # Keep editors hidden as internal data holders for calculations.
        self.lat_edit.setVisible(False)
        self.lon_edit.setVisible(False)
        self.elev_edit.setVisible(False)

        self.coord_error_label.setVisible(False)

        self.min_moon_sep_spin = QSpinBox(minimum=0, maximum=180, value=self.settings.value("general/minMoonSep", 0, type=int))
        self.min_moon_sep_spin.setMinimumWidth(84)
        self.min_moon_sep_spin.setMaximumWidth(92)

        self.min_score_spin = QSpinBox(minimum=0, maximum=100, value=self.settings.value("general/minScore", 0, type=int))
        self.min_score_spin.setMinimumWidth(84)
        self.min_score_spin.setMaximumWidth(92)

        self.hide_observed_chk = QCheckBox("Hide observed")
        self.hide_observed_chk.setChecked(self.settings.value("general/hideObserved", False, type=bool))
        self.min_moon_sep_spin.valueChanged.connect(self._on_filter_change)
        self.min_score_spin.valueChanged.connect(self._on_filter_change)
        self.hide_observed_chk.stateChanged.connect(lambda _: self._on_filter_change())

        # Sun/Moon visibility toggles
        self.sun_check = QCheckBox("Show Sun")
        self.sun_check.setChecked(False)
        self.sun_check.stateChanged.connect(self._toggle_visibility)

        self.moon_check = QCheckBox("Show Moon")
        self.moon_check.setChecked(True)
        self.moon_check.stateChanged.connect(self._toggle_visibility)

        for chk, min_w in (
            (self.hide_observed_chk, 132),
            (self.sun_check, 102),
            (self.moon_check, 112),
        ):
            chk.setMinimumWidth(min_w)
            chk.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)

        filters_layout = QVBoxLayout()
        filters_layout.setContentsMargins(2, 2, 2, 2)
        filters_layout.setSpacing(3)

        filters_row_top = QHBoxLayout()
        filters_row_top.setSpacing(6)
        filters_row_top.addWidget(QLabel("Moon ≥"))
        filters_row_top.addWidget(self.min_moon_sep_spin)
        filters_row_top.addSpacing(6)
        filters_row_top.addWidget(QLabel("Score ≥"))
        filters_row_top.addWidget(self.min_score_spin)
        filters_row_top.addStretch(1)
        filters_layout.addLayout(filters_row_top)

        filters_row_bottom = QHBoxLayout()
        filters_row_bottom.setSpacing(8)
        filters_row_bottom.addWidget(self.hide_observed_chk)
        filters_row_bottom.addWidget(self.sun_check)
        filters_row_bottom.addWidget(self.moon_check)
        filters_row_bottom.addStretch(1)
        filters_layout.addLayout(filters_row_bottom)

        add_btn = QPushButton("Add")
        add_btn.setMinimumHeight(24)
        add_btn.clicked.connect(self._add_target_dialog)
        add_btn.setToolTip("Add a new target to the plan")
        _set_button_variant(add_btn, "primary")
        _set_button_icon_kind(add_btn, "add")

        toggle_obs_btn = QPushButton("Observed")
        toggle_obs_btn.setMinimumHeight(24)
        toggle_obs_btn.clicked.connect(self._toggle_observed_selected)
        toggle_obs_btn.setToolTip("Toggle observed state for the selected target")
        _set_button_variant(toggle_obs_btn, "neutral")
        _set_button_icon_kind(toggle_obs_btn, "toggle")
        self.edit_object_btn = QPushButton("Edit")
        self.edit_object_btn.setMinimumHeight(24)
        self.edit_object_btn.clicked.connect(self._edit_object_for_selected)
        self.edit_object_btn.setEnabled(False)
        self.edit_object_btn.setToolTip("Edit the selected target")
        _set_button_variant(self.edit_object_btn, "neutral")
        _set_button_icon_kind(self.edit_object_btn, "edit")

        self.open_saved_plan_btn = QPushButton("Open Plan")
        self.open_saved_plan_btn.setMinimumHeight(24)
        self.open_saved_plan_btn.clicked.connect(self._open_saved_plan)
        self.open_saved_plan_btn.setToolTip("Open a saved plan from local storage")
        _set_button_variant(self.open_saved_plan_btn, "secondary")
        _set_button_icon_kind(self.open_saved_plan_btn, "load")
        self.save_plan_as_btn = QPushButton("Save Plan")
        self.save_plan_as_btn.setMinimumHeight(24)
        self.save_plan_as_btn.clicked.connect(self._save_plan_as)
        self.save_plan_as_btn.setToolTip("Save the current plan under a chosen name")
        _set_button_variant(self.save_plan_as_btn, "secondary")
        _set_button_icon_kind(self.save_plan_as_btn, "save")
        suggest_targets_btn = QPushButton("Suggest")
        suggest_targets_btn.setMinimumHeight(24)
        suggest_targets_btn.clicked.connect(self._ai_suggest_targets)
        suggest_targets_btn.setToolTip(
            "Check cached BHTOM targets first (TTL 1h). "
            "A fresh fetch starts only when the cache is missing or expired."
        )
        _set_button_variant(suggest_targets_btn, "primary")
        _set_button_icon_kind(suggest_targets_btn, "suggest")
        self.quick_targets_btn = QPushButton("Quick")
        self.quick_targets_btn.setMinimumHeight(24)
        self.quick_targets_btn.clicked.connect(self._quick_add_suggested_targets)
        self._update_quick_targets_button_tooltip()
        _set_button_variant(self.quick_targets_btn, "primary")
        _set_button_icon_kind(self.quick_targets_btn, "quick")
        self.seestar_session_btn = QPushButton("Seestar")
        self.seestar_session_btn.setMinimumHeight(24)
        self.seestar_session_btn.clicked.connect(self._open_seestar_session)
        self.seestar_session_btn.setToolTip("Open Seestar session settings")
        _set_button_variant(self.seestar_session_btn, "neutral")
        _set_button_icon_kind(self.seestar_session_btn, "seestar")
        self.weather_btn = QPushButton("Weather")
        self.weather_btn.setMinimumHeight(24)
        self.weather_btn.setToolTip("Open weather workspace (forecast/conditions/cloud/satellite tabs)")
        self.weather_btn.clicked.connect(self._open_weather_window)
        _set_button_variant(self.weather_btn, "secondary")
        _set_button_icon_kind(self.weather_btn, "weather")
        self.ai_toggle_btn = QPushButton("AI")
        self.ai_toggle_btn.setMinimumHeight(24)
        self.ai_toggle_btn.setCheckable(True)
        self.ai_toggle_btn.setChecked(False)
        self.ai_toggle_btn.setToolTip("Show or hide the AI assistant window")
        self.ai_toggle_btn.toggled.connect(self._toggle_ai_panel)
        _set_button_variant(self.ai_toggle_btn, "secondary")
        _set_button_icon_kind(self.ai_toggle_btn, "ai")

        session_strip = QWidget()
        session_strip.setObjectName("SessionStrip")
        session_strip.setLayout(session_layout)
        filters_strip = QWidget()
        filters_strip.setObjectName("FiltersStrip")
        filters_strip.setLayout(filters_layout)

        top_controls = QFrame()
        top_controls.setObjectName("TopControlsBar")
        top_controls_l = QHBoxLayout()
        top_controls_l.setContentsMargins(6, 4, 6, 4)
        top_controls_l.setSpacing(8)
        top_controls_l.addWidget(session_strip, 3)
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setFrameShadow(QFrame.Sunken)
        top_controls_l.addWidget(sep)
        top_controls_l.addWidget(filters_strip, 2)
        top_controls.setLayout(top_controls_l)
        top_controls.setMinimumHeight(70)
        top_controls.setMaximumHeight(140)
        _set_dynamic_property(top_controls, "accented", True)

        actions_bar = QFrame()
        actions_bar.setObjectName("ActionsBar")
        actions_l = QHBoxLayout()
        actions_l.setContentsMargins(10, 4, 10, 4)
        actions_l.setSpacing(8)
        actions_l.addWidget(add_btn)
        actions_l.addWidget(suggest_targets_btn)
        actions_l.addWidget(self.quick_targets_btn)
        actions_l.addWidget(self.seestar_session_btn)
        actions_l.addWidget(self.weather_btn)
        actions_l.addWidget(toggle_obs_btn)
        actions_l.addWidget(self.edit_object_btn)
        actions_l.addWidget(self.open_saved_plan_btn)
        actions_l.addWidget(self.save_plan_as_btn)
        actions_l.addWidget(self.ai_toggle_btn)
        actions_l.addStretch(1)
        actions_bar.setLayout(actions_l)
        actions_bar.setMinimumHeight(40)
        actions_bar.setMaximumHeight(50)
        self._update_seestar_button_tooltip()
        _set_dynamic_property(actions_bar, "accented", True)

        plot_card = QFrame()
        plot_card.setObjectName("PlotCard")
        plot_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        _set_dynamic_property(plot_card, "accented", True)
        plot_l = QVBoxLayout()
        plot_l.setContentsMargins(2, 2, 2, 2)
        plot_l.setSpacing(2)
        if self._use_visibility_web and self.plot_controls_bar is not None:
            if self.plot_controls_bar is not None:
                plot_l.addWidget(self.plot_controls_bar)
        plot_l.addWidget(self._visibility_bootstrap_placeholder, 1)
        plot_card.setLayout(plot_l)
        self.plot_card = plot_card
        self.plot_card_layout = plot_l

        self.table_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.table_view.setAlternatingRowColors(True)

        # Info panel for sun/moon events
        self.info_widget = QWidget()
        info_layout = QHBoxLayout()
        info_layout.setContentsMargins(0, 0, 0, 0)
        info_layout.setSpacing(10)

        left_info_widget = QWidget()
        left_info_form = QFormLayout()
        left_info_form.setContentsMargins(0, 0, 0, 0)
        left_info_form.setHorizontalSpacing(6)
        left_info_form.setVerticalSpacing(2)
        left_info_form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        left_info_widget.setLayout(left_info_form)

        info_sep = QFrame()
        info_sep.setFrameShape(QFrame.VLine)
        info_sep.setFrameShadow(QFrame.Sunken)

        right_info_widget = QWidget()
        right_info_form = QFormLayout()
        right_info_form.setContentsMargins(0, 0, 0, 0)
        right_info_form.setHorizontalSpacing(6)
        right_info_form.setVerticalSpacing(2)
        right_info_form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        right_info_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        right_info_widget.setLayout(right_info_form)

        info_layout.addWidget(left_info_widget, 1)
        info_layout.addWidget(info_sep)
        info_layout.addWidget(right_info_widget, 1)
        self.info_widget.setLayout(info_layout)

        self.sunrise_label = QLabel("-")
        self.sunset_label = QLabel("-")
        self.moonrise_label = QLabel("-")
        self.moonset_label = QLabel("-")
        self.moonphase_bar = QProgressBar(self)
        self.moonphase_bar.setRange(0, 100)
        self.moonphase_bar.setValue(0)
        self.moonphase_bar.setFormat("%p%%")
        self.moonphase_bar.setTextVisible(True)
        self.moonphase_bar.setMinimumWidth(130)
        self.moonphase_bar.setMaximumWidth(190)
        self.moonphase_bar.setMinimumHeight(20)
        # Current time labels
        self.localtime_label = QLabel("-")
        self.utctime_label = QLabel("-")
        # Fonts for info panel: labels bold, values italic
        label_font = QFont(self.font())
        label_font.setPointSize(10)
        label_font.setWeight(QFont.Weight.DemiBold)
        value_font = QFont(self.font())
        value_font.setPointSize(10)
        self.info_label_font = label_font
        self.info_value_font = value_font
        value_line_h = max(16, QFontMetrics(self.info_value_font).lineSpacing())
        self.sun_alt_label = QLabel("-")
        self.moon_alt_label = QLabel("-")
        self.sidereal_label = QLabel("-")

        # Selected target details
        self.sel_name_label = QLabel("-")
        self.sel_type_label = QLabel("-")
        self.sel_score_label = QLabel("-")
        self.sel_window_label = QLabel("-")
        self.sel_notes_label = QLabel("-")
        for dynamic_value_label in (self.sel_name_label, self.sel_type_label, self.sel_notes_label):
            dynamic_value_label.setWordWrap(True)
            dynamic_value_label.setMinimumWidth(0)
            dynamic_value_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
            dynamic_value_label.setMinimumHeight(value_line_h + 2)
        # Reserve two text lines for target/type so short values don't collapse the row
        # and medium-length values don't cause visible "jumping" between selections.
        reserved_two_lines_h = value_line_h * 2 + 4
        self.sel_name_label.setMinimumHeight(reserved_two_lines_h)
        self.sel_type_label.setMinimumHeight(reserved_two_lines_h)

        details_row = QWidget()
        details_layout = QVBoxLayout()
        details_layout.setContentsMargins(0, 0, 0, 0)
        details_layout.setSpacing(2)
        details_layout.addWidget(self.sel_notes_label)
        details_row.setLayout(details_layout)

        def _add_info_row(form: QFormLayout, row_idx: int, icon_kind: str, title: str, value_widget: QWidget):
            title_widget = QWidget()
            title_layout = QHBoxLayout(title_widget)
            title_layout.setContentsMargins(0, 0, 0, 0)
            title_layout.setSpacing(7)
            icon_label = QLabel(title_widget)
            _set_detail_icon_kind(icon_label, icon_kind, 16)
            title_label = QLabel(title, title_widget)
            title_label.setFont(self.info_label_font)
            title_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            title_layout.addWidget(icon_label, 0, Qt.AlignVCenter)
            title_layout.addWidget(title_label, 0, Qt.AlignVCenter)
            title_layout.addStretch(1)
            if isinstance(value_widget, QLabel):
                value_widget.setFont(self.info_value_font)
                value_widget.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            form.insertRow(row_idx, title_widget, value_widget)
            return title_label

        _add_info_row(left_info_form, 0, "clock", "Local time:", self.localtime_label)
        _add_info_row(left_info_form, 1, "utc", "UTC time:", self.utctime_label)
        _add_info_row(left_info_form, 2, "sidereal", "Sidereal time:", self.sidereal_label)
        _add_info_row(left_info_form, 3, "sunset", "Sunset:", self.sunset_label)
        _add_info_row(left_info_form, 4, "sunrise", "Sunrise:", self.sunrise_label)
        _add_info_row(left_info_form, 5, "moonphase", "Moon phase:", self.moonphase_bar)
        _add_info_row(left_info_form, 6, "moonrise", "Moonrise:", self.moonrise_label)
        _add_info_row(left_info_form, 7, "moonset", "Moonset:", self.moonset_label)

        _add_info_row(right_info_form, 0, "sun", "Sun altitude:", self.sun_alt_label)
        _add_info_row(right_info_form, 1, "moon", "Moon altitude:", self.moon_alt_label)
        _add_info_row(right_info_form, 2, "target", "Selected target:", self.sel_name_label)
        self.sel_type_title_label = _add_info_row(right_info_form, 3, "describe", "Type / Last Mag:", self.sel_type_label)
        _add_info_row(right_info_form, 4, "sidereal", "Score:", self.sel_score_label)
        _add_info_row(right_info_form, 5, "window", "Best window:", self.sel_window_label)
        _add_info_row(right_info_form, 6, "notes", "Notes:", details_row)
        self.sel_name_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.sel_type_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.sel_notes_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        self.info_widget.setContentsMargins(0, 0, 0, 0)

        table_card = QFrame()
        table_card.setObjectName("TableCard")
        table_card.setMinimumHeight(180)
        _set_dynamic_property(table_card, "accented", True)
        table_l = QVBoxLayout()
        table_l.setContentsMargins(2, 2, 2, 2)
        table_l.setSpacing(2)
        table_title_row = QWidget()
        table_title_row_l = QHBoxLayout(table_title_row)
        table_title_row_l.setContentsMargins(0, 0, 0, 0)
        table_title_row_l.setSpacing(8)
        table_title = QLabel("Targets")
        table_title.setObjectName("SectionTitle")
        table_hint = QLabel("Left-click: select row | Double left-click (Name): edit object | Right-click: more options")
        table_hint.setObjectName("SectionHint")
        table_title_row_l.addWidget(table_title)
        table_title_row_l.addWidget(table_hint)
        table_title_row_l.addStretch(1)
        table_l.addWidget(table_title_row)
        table_l.addWidget(self.table_view)
        table_card.setLayout(table_l)

        polar_card = QFrame()
        polar_card.setObjectName("PolarCard")
        _set_dynamic_property(polar_card, "accented", True)
        polar_l = QVBoxLayout()
        polar_l.setContentsMargins(2, 2, 2, 2)
        polar_l.setSpacing(1)
        polar_title = QLabel("Sky View")
        polar_title.setObjectName("SectionTitle")
        polar_l.addWidget(polar_title)

        cutout_frame = QFrame()
        cutout_frame.setObjectName("CutoutFrame")
        _set_dynamic_property(cutout_frame, "accented", True)
        cutout_frame.setMinimumWidth(220)
        cutout_frame.setMaximumWidth(560)
        cutout_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        cutout_l = QVBoxLayout()
        cutout_l.setContentsMargins(0, 0, 0, 0)
        cutout_l.setSpacing(0)
        self.cutout_tabs = _configure_tab_widget(QTabWidget(self))
        self.cutout_tabs.setTabPosition(QTabWidget.North)
        self.cutout_tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        aladin_host, self.aladin_image_label = self._create_cutout_image_stack(
            self.cutout_tabs,
            kind="aladin",
            title="Aladin",
        )
        self.aladin_image_label.set_zoom_enabled(True)
        self.aladin_image_label.set_overlay_painter(self._paint_aladin_static_overlay)
        self.aladin_image_label.setToolTip(
            "Mouse wheel: zoom (wheel down at 1x loads wider field) | Drag: pan | Double left-click: reset zoom"
        )
        self.aladin_image_label.zoomOutLimitReached.connect(self._aladin_expand_context)
        self.aladin_image_label.resized.connect(self._schedule_cutout_resize_refresh)
        # Backward-compatible alias used by existing cutout helpers.
        self.cutout_image_label = self.aladin_image_label

        finder_host, self.finder_image_label = self._create_cutout_image_stack(
            self.cutout_tabs,
            kind="finder",
            title="Finder Chart",
        )
        self.finder_image_label.resized.connect(self._schedule_cutout_resize_refresh)

        aladin_tab = QWidget(self.cutout_tabs)
        aladin_tab.setProperty("cutout_page", True)
        aladin_tab.setAttribute(Qt.WA_StyledBackground, True)
        aladin_tab_l = QHBoxLayout(aladin_tab)
        aladin_tab_l.setContentsMargins(0, 0, 0, 0)
        aladin_tab_l.setSpacing(2)
        aladin_tab_l.addWidget(aladin_host, 1)
        aladin_zoom_col = QWidget(aladin_tab)
        aladin_zoom_col.setProperty("cutout_tool_col", True)
        aladin_zoom_col.setAttribute(Qt.WA_StyledBackground, True)
        aladin_zoom_col_l = QVBoxLayout(aladin_zoom_col)
        aladin_zoom_col_l.setContentsMargins(2, 2, 2, 2)
        aladin_zoom_col_l.setSpacing(4)
        zoom_btn_size = 30
        aladin_zoom_in_btn = QToolButton(aladin_zoom_col)
        aladin_zoom_in_btn.setText("+")
        aladin_zoom_in_btn.setToolTip("Zoom in")
        aladin_zoom_in_btn.setFixedSize(zoom_btn_size, zoom_btn_size)
        aladin_zoom_in_btn.clicked.connect(self._aladin_zoom_in)
        aladin_zoom_out_btn = QToolButton(aladin_zoom_col)
        aladin_zoom_out_btn.setText("−")
        aladin_zoom_out_btn.setToolTip("Zoom out")
        aladin_zoom_out_btn.setFixedSize(zoom_btn_size, zoom_btn_size)
        aladin_zoom_out_btn.clicked.connect(self._aladin_zoom_out)
        aladin_zoom_reset_btn = QToolButton(aladin_zoom_col)
        aladin_zoom_reset_btn.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
        aladin_zoom_reset_btn.setIconSize(QSize(14, 14))
        aladin_zoom_reset_btn.setToolTip("Reset zoom to 1:1")
        aladin_zoom_reset_btn.setFixedSize(zoom_btn_size, zoom_btn_size)
        aladin_zoom_reset_btn.clicked.connect(self._aladin_zoom_reset)
        aladin_zoom_col_l.addWidget(aladin_zoom_in_btn)
        aladin_zoom_col_l.addWidget(aladin_zoom_out_btn)
        aladin_zoom_col_l.addWidget(aladin_zoom_reset_btn)
        aladin_zoom_col_l.addStretch(1)
        aladin_tab_l.addWidget(aladin_zoom_col, 0, Qt.AlignTop)
        finder_tab = QWidget(self.cutout_tabs)
        finder_tab.setProperty("cutout_page", True)
        finder_tab.setAttribute(Qt.WA_StyledBackground, True)
        finder_tab_l = QVBoxLayout(finder_tab)
        finder_tab_l.setContentsMargins(0, 0, 0, 0)
        finder_tab_l.addWidget(finder_host, 1)
        self.cutout_tabs.addTab(aladin_tab, "Aladin")
        self.cutout_tabs.addTab(finder_tab, "Finder chart")
        self.cutout_tabs.currentChanged.connect(self._on_cutout_tab_changed)
        self.cutout_tabs.setCurrentIndex(1 if self._cutout_view_key == "finderchart" else 0)
        cutout_l.addWidget(self.cutout_tabs, 1)
        cutout_frame.setLayout(cutout_l)

        sky_row = QHBoxLayout()
        sky_row.setContentsMargins(0, 0, 0, 0)
        sky_row.setSpacing(2)
        sky_row.addWidget(self.polar_canvas, 2)
        sky_row.addWidget(cutout_frame, 3)
        polar_l.addLayout(sky_row, stretch=1)
        polar_card.setLayout(polar_l)

        info_card = QFrame()
        info_card.setObjectName("InfoCard")
        _set_dynamic_property(info_card, "accented", True)
        info_l = QVBoxLayout()
        info_l.setContentsMargins(2, 2, 2, 2)
        info_l.setSpacing(2)
        info_title = QLabel("Night Details")
        info_title.setObjectName("SectionTitle")
        self.info_title_label = info_title
        info_l.addWidget(info_title)
        info_scroll = QScrollArea()
        info_scroll.setWidgetResizable(True)
        info_scroll.setFrameShape(QFrame.NoFrame)
        info_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        info_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        info_scroll.setWidget(self.info_widget)
        self.info_scroll = info_scroll
        info_l.addWidget(info_scroll, stretch=1)
        info_card.setLayout(info_l)
        self.info_card = info_card
        info_card.setMinimumHeight(220)
        self._night_details_fixed_min_width = max(330, self.info_widget.minimumSizeHint().width() + 26)

        right_dashboard = QSplitter(Qt.Vertical)
        right_dashboard.addWidget(polar_card)
        right_dashboard.addWidget(info_card)
        right_dashboard.setHandleWidth(1)
        right_dashboard.setStretchFactor(0, 3)
        right_dashboard.setStretchFactor(1, 2)
        right_dashboard.setSizes([390, 300])
        right_dashboard.setMinimumWidth(460)
        self._right_dashboard_base_min_width = 460
        right_dashboard.setMaximumWidth(16777215)
        right_dashboard.setChildrenCollapsible(False)
        self.right_dashboard = right_dashboard
        self._update_night_details_constraints()

        top_splitter = QSplitter(Qt.Horizontal)
        top_splitter.addWidget(plot_card)
        top_splitter.addWidget(right_dashboard)
        top_splitter.setHandleWidth(1)
        top_splitter.setStretchFactor(0, 8)
        top_splitter.setStretchFactor(1, 1)
        top_splitter.setSizes([1000, 660])
        top_splitter.setChildrenCollapsible(False)

        main_vertical = QSplitter(Qt.Vertical)
        main_vertical.addWidget(top_splitter)
        main_vertical.addWidget(table_card)
        main_vertical.setHandleWidth(1)
        main_vertical.setStretchFactor(0, 4)
        main_vertical.setStretchFactor(1, 3)
        main_vertical.setSizes([500, 300])
        main_vertical.setChildrenCollapsible(False)

        main_area = QWidget()
        main_l = QVBoxLayout()
        main_l.setContentsMargins(0, 0, 0, 0)
        main_l.setSpacing(0)
        main_l.addWidget(main_vertical)
        main_area.setLayout(main_l)
        self.ai_window: Optional[QDialog] = None
        self.weather_window: Optional[WeatherDialog] = None

        container = QWidget()
        container.setObjectName("RootContainer")
        container_l = QVBoxLayout()
        container_l.setContentsMargins(0, 0, 0, 0)
        container_l.setSpacing(2)
        container_l.addWidget(top_controls)
        container_l.addWidget(main_area, 1)
        container_l.addWidget(actions_bar)
        container.setLayout(container_l)
        self.setCentralWidget(container)
        required_main_width = max(top_controls.minimumSizeHint().width(), actions_bar.minimumSizeHint().width()) + 24
        self.setMinimumWidth(max(self.minimumWidth(), required_main_width))
        # Apply persisted settings once all widgets exist
        self._load_settings()

        # Apply custom GUI styles
        self._apply_styles()
        # Build the menu bar and shortcuts
        self._build_actions()
        self._main_window_coordinator.setup_status_bar_and_startup_hooks()
        self._apply_localization()

    def _apply_localization(self) -> None:
        lang = resolve_language_choice(
            getattr(self, "_ui_language", current_language()),
            system_locale=getattr(self, "_system_locale_name", ""),
        )
        set_current_language(lang, system_locale=getattr(self, "_system_locale_name", ""))
        self._ui_language = lang
        localize_widget_tree(self, lang)
        if hasattr(self, "table_model"):
            try:
                self.table_model.headerDataChanged.emit(Qt.Horizontal, 0, self.table_model.columnCount() - 1)
            except Exception:
                pass
        for dialog in (
            getattr(self, "ai_window", None),
            getattr(self, "weather_window", None),
            getattr(self, "_bhtom_dialog", None),
        ):
            if not isinstance(dialog, QDialog):
                continue
            localize_widget_tree(dialog, lang)
            model = getattr(dialog, "table_model", None)
            if model is not None and hasattr(model, "columnCount") and hasattr(model, "headerDataChanged"):
                try:
                    model.headerDataChanged.emit(Qt.Horizontal, 0, model.columnCount() - 1)
                except Exception:
                    pass

    def _start_clock_worker(self):
        # Don’t create new workers while exiting
        if getattr(self, "_shutting_down", False):
            return
        # Determine site, fallback to default if none
        site = self.table_model.site or Site(name="Default", latitude=0.0, longitude=0.0, elevation=0.0)
        existing_worker = self.clock_worker
        if (
            existing_worker is not None
            and existing_worker.isRunning()
            and _same_runtime_site(getattr(existing_worker, "site", None), site)
        ):
            logger.debug("Reusing existing ClockWorker for site %s", site.name)
            return
        # Stop any existing clock worker
        if existing_worker:
            self.clock_worker = None
            try:
                existing_worker.request_stop()
            except Exception:
                try:
                    existing_worker.stop()
                except Exception:
                    pass
        logger.info("Launching new ClockWorker for site %s", site.name)
        # Create and track new worker instance
        worker = ClockWorker(site, self.targets, self)
        self.clock_worker = worker
        self._clock_workers.append(worker)
        # Connect updates only from this specific worker to avoid stale threads
        worker.updated.connect(lambda data, w=worker: self._handle_clock_update(data) if self.clock_worker is w else None)
        # Start the new worker
        worker.start()
        # Ensure finished threads clean themselves up
        worker.finished.connect(lambda w=worker: self._clock_workers.remove(w) if w in self._clock_workers else None)
        worker.finished.connect(worker.deleteLater)

    # --------------------------------------------------
    # Qt close handler – ensure threads are stopped
    # --------------------------------------------------
    def closeEvent(self, event):
        """Stop background threads before the window closes."""
        try:
            if hasattr(self, "_plan_autosave_timer") and self._plan_autosave_timer.isActive():
                self._plan_autosave_timer.stop()
            self._flush_plan_autosave()
        except Exception:
            pass
        try:
            self._cleanup_threads()
        except Exception as exc:
            logger.exception("Error during thread cleanup: %s", exc)
        try:
            self.settings.sync()
        except Exception:
            pass
        super().closeEvent(event)

    # --------------------------------------------------
    # Qt close handler – ensure threads are stopped
    # --------------------------------------------------

    # .....................................................
    # menu & shortcuts
    # .....................................................
    def _build_actions(self) -> None:  # noqa: D401
        return self._main_window_coordinator._build_actions()

    def _load_settings(self):
        return self._main_window_coordinator._load_settings()

    def _migrate_table_settings_schema(self) -> None:
        coordinator = getattr(self, "_main_window_coordinator", None)
        if coordinator is None:
            coordinator = MainWindowCoordinator(self)
        return coordinator._migrate_table_settings_schema()

    def _recompute_recommended_order_cache(self) -> None:
        coordinator = getattr(self, "_table_coordinator", None)
        if coordinator is not None:
            coordinator.recompute_recommended_order_cache()

    def _apply_table_settings(self):
        coordinator = getattr(self, "_table_coordinator", None)
        if coordinator is not None:
            coordinator.apply_table_settings()

    def _table_matches_observation_preset(self) -> bool:
        coordinator = getattr(self, "_table_coordinator", None)
        if coordinator is None:
            return False
        return coordinator.table_matches_observation_preset()

    def _apply_column_preset(self, preset: str, save: bool = True):
        coordinator = getattr(self, "_table_coordinator", None)
        if coordinator is not None:
            coordinator.apply_column_preset(preset, save=save)

    def _schedule_table_column_width_refresh(self, reset_widths: bool = False) -> None:
        coordinator = getattr(self, "_table_coordinator", None)
        if coordinator is not None:
            coordinator.schedule_table_column_width_refresh(reset_widths=reset_widths)

    def _refresh_table_column_widths(self) -> None:
        coordinator = getattr(self, "_table_coordinator", None)
        if coordinator is not None:
            coordinator.refresh_table_column_widths()

    def open_table_settings(self):
        """Open the Table Settings dialog."""
        dlg = TableSettingsDialog(self)
        dlg.exec()

    def _apply_general_settings(self):
        return self._main_window_coordinator._apply_general_settings()

    def open_general_settings(self, initial_tab: Optional[str] = None):
        dlg = GeneralSettingsDialog(self, initial_tab=initial_tab)
        dlg.exec()

    @Slot()
    def _open_ai_settings(self) -> None:
        self._ai_panel_coordinator.open_settings()

    def _refresh_ai_context_hint(self) -> None:
        self._ai_panel_coordinator.refresh_context_hint()

    def _refresh_ai_knowledge_hint(self, titles: Optional[list[str]] = None) -> None:
        self._ai_panel_coordinator.refresh_knowledge_hint(titles)

    @Slot()
    def _focus_ai_input(self) -> None:
        self._ai_panel_coordinator.focus_input()

    def _sun_alt_limit(self) -> float:
        if not hasattr(self, "sun_alt_limit_spin"):
            return -10.0
        return float(self.sun_alt_limit_spin.value())

    @Slot(int)
    def _on_sun_alt_limit_changed(self, value: int):
        self.settings.setValue("general/sunAltLimit", int(value))
        self._update_status_bar()
        self._refresh_weather_window_context()
        self._schedule_plan_autosave()
        if hasattr(self, "progress"):
            self._replot_timer.start()

    @Slot()
    def _on_filter_change(self, *_):
        self.settings.setValue("general/minMoonSep", self.min_moon_sep_spin.value())
        self.settings.setValue("general/minScore", self.min_score_spin.value())
        self.settings.setValue("general/hideObserved", self.hide_observed_chk.isChecked())
        self.table_model.row_enabled = self._recompute_row_enabled_from_current()
        self._apply_table_row_visibility()
        self._emit_table_data_changed()
        self._update_status_bar()
        self._schedule_plan_autosave()
        if hasattr(self, "progress"):
            self._replot_timer.start()

    def _passes_active_filters(self, target: Target, score: float, moon_sep_now: float) -> bool:
        min_moon_sep = float(self.min_moon_sep_spin.value()) if hasattr(self, "min_moon_sep_spin") else 0.0
        min_score = float(self.min_score_spin.value()) if hasattr(self, "min_score_spin") else 0.0
        hide_observed = self.hide_observed_chk.isChecked() if hasattr(self, "hide_observed_chk") else False
        if hide_observed and target.observed:
            return False
        if score < min_score:
            return False
        if not np.isfinite(moon_sep_now):
            return min_moon_sep <= 0.0
        # Keep filter semantics aligned with table display precision (0.1 deg).
        moon_sep_cmp = round(float(moon_sep_now), 1)
        if moon_sep_cmp < min_moon_sep:
            return False
        return True

    def _recompute_row_enabled_from_current(self) -> list[bool]:
        row_enabled: list[bool] = []
        for idx, tgt in enumerate(self.targets):
            score = float(self.table_model.scores[idx]) if idx < len(self.table_model.scores) else 0.0
            moon_sep_now = (
                float(self.table_model.current_seps[idx])
                if idx < len(self.table_model.current_seps)
                else float("nan")
            )
            row_enabled.append(self._passes_active_filters(tgt, score, moon_sep_now))
        return row_enabled

    def _selected_rows(self) -> list[int]:
        sel_model = self.table_view.selectionModel()
        if sel_model is None:
            return []
        return sorted({idx.row() for idx in sel_model.selectedRows()})

    def _schedule_primary_target_selection(self) -> None:
        coordinator = getattr(self, "_table_coordinator", None)
        if coordinator is not None:
            coordinator.schedule_primary_target_selection()

    @Slot()
    def _ensure_primary_target_selected(self) -> None:
        if not hasattr(self, "table_view") or not hasattr(self, "table_model"):
            return
        rows = self.table_model.rowCount()
        if rows <= 0:
            self._update_selected_details()
            return
        current_rows = [row for row in self._selected_rows() if 0 <= row < rows and not self.table_view.isRowHidden(row)]
        if current_rows:
            return
        target_row = next((row for row in range(rows) if not self.table_view.isRowHidden(row)), None)
        if target_row is None:
            self._update_selected_details()
            return
        sel_model = self.table_view.selectionModel()
        if sel_model is None:
            return
        idx = self.table_model.index(target_row, TargetTableModel.COL_NAME)
        if not idx.isValid():
            return
        sel_model.select(idx, QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows)
        self.table_view.setCurrentIndex(idx)
        self.table_view.scrollTo(idx, QTableView.PositionAtCenter)

    def _update_status_bar(self):
        return self._main_window_coordinator._update_status_bar()

    @Slot(object, object)
    def _update_selected_details(self, *_):
        rows = self._selected_rows()
        if not rows:
            self.sel_name_label.setText("-")
            set_translated_text(self.sel_type_title_label, "Type / Mag:", current_language())
            self.sel_type_label.setText("-")
            self.sel_score_label.setText("-")
            self.sel_window_label.setText("-")
            self.sel_notes_label.setText("-")
            self.edit_object_btn.setEnabled(False)
            if hasattr(self, "ai_describe_act"):
                self.ai_describe_act.setEnabled(False)
            self._schedule_selected_cutout_update(None)
            self._update_status_bar()
            return

        row = rows[0]
        if not (0 <= row < len(self.targets)):
            self._schedule_selected_cutout_update(None)
            return
        tgt = self.targets[row]
        self._ensure_known_target_type(tgt)
        self.sel_name_label.setText(tgt.name)
        set_translated_text(
            self.sel_type_title_label,
            "Type / Last Mag:" if _target_magnitude_label(tgt) == "Last Mag" else "Type / Mag:",
            current_language(),
        )
        mag_txt = f"{tgt.magnitude:.2f}" if tgt.magnitude is not None else "-"
        type_txt = tgt.object_type or "-"
        self.sel_type_label.setText(f"{type_txt} / {_target_magnitude_label(tgt)} {mag_txt}")

        metrics = self.target_metrics.get(tgt.name)
        if metrics:
            self.sel_score_label.setText(f"{metrics.score:.1f} | {metrics.hours_above_limit:.1f}h")
        else:
            self.sel_score_label.setText("-")

        window = self.target_windows.get(tgt.name)
        if window:
            tz_name = "UTC"
            if isinstance(getattr(self, "last_payload", None), dict):
                tz_name = str(self.last_payload.get("tz", "UTC"))
            try:
                tz = pytz.timezone(tz_name)
            except Exception:
                tz = pytz.UTC
            start = window[0].astimezone(tz).strftime("%H:%M")
            end = window[1].astimezone(tz).strftime("%H:%M")
            self.sel_window_label.setText(f"{start} - {end}")
        else:
            self.sel_window_label.setText("-")
        self.sel_notes_label.setText(tgt.notes or "-")
        self.edit_object_btn.setEnabled(True)
        if hasattr(self, "ai_describe_act"):
            self.ai_describe_act.setEnabled(True)
        self._schedule_selected_cutout_update(tgt)
        self._update_status_bar()

    def _edit_notes_for_row(self, row: int):
        if not (0 <= row < len(self.targets)):
            return
        tgt = self.targets[row]
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Notes: {tgt.name}")
        layout = QVBoxLayout(dlg)
        notes_edit = QTextEdit(dlg)
        notes_edit.setPlainText(tgt.notes or "")
        notes_edit.setMinimumHeight(120)
        layout.addWidget(notes_edit)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dlg)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        _style_dialog_button_box(buttons)
        layout.addWidget(buttons)
        _fit_dialog_to_screen(
            dlg,
            preferred_width=760,
            preferred_height=520,
            min_width=560,
            min_height=340,
        )
        if dlg.exec() != QDialog.Accepted:
            return
        tgt.notes = notes_edit.toPlainText().strip()
        idx = self.table_model.index(row, TargetTableModel.COL_NAME)
        self.table_model.dataChanged.emit(idx, idx, [Qt.ToolTipRole, Qt.DisplayRole])
        self._update_selected_details()
        self._schedule_plan_autosave()

    def _edit_object_for_row(self, row: int):
        if not (0 <= row < len(self.targets)):
            return
        tgt = self.targets[row]
        old_name = tgt.name

        dlg = QDialog(self)
        dlg.setWindowTitle(f"Edit Object: {tgt.name}")
        dlg.setObjectName("EditObjectDialog")
        form = QFormLayout(dlg)

        name_edit = QLineEdit(dlg)
        name_edit.setText(tgt.name)
        ra_edit = QLineEdit(dlg)
        ra_edit.setText(Angle(tgt.ra, u.deg).to_string(unit=u.hourangle, sep=":", pad=True, precision=2))
        ra_edit.setPlaceholderText("HH:MM:SS or degrees")
        dec_edit = QLineEdit(dlg)
        dec_edit.setText(Angle(tgt.dec, u.deg).to_string(unit=u.deg, sep=":", alwayssign=True, pad=True, precision=2))
        dec_edit.setPlaceholderText("±DD:MM:SS or degrees")
        type_edit = QLineEdit(dlg)
        type_edit.setText(tgt.object_type or "")
        mag_edit = QLineEdit(dlg)
        mag_edit.setPlaceholderText("Optional")
        mag_edit.setValidator(QDoubleValidator(-40.0, 40.0, 4, dlg))
        if tgt.magnitude is not None:
            mag_edit.setText(f"{tgt.magnitude:.2f}")
        line_palette = self._line_palette()
        color_edit = QLineEdit(dlg)
        color_edit.setPlaceholderText("Auto (palette) or #RRGGBB")
        color_edit.setText(_normalized_css_color(tgt.plot_color))
        color_preview = QLabel("Auto", dlg)
        color_preview.setAlignment(Qt.AlignCenter)
        color_preview.setMinimumWidth(112)
        color_preview.setFrameShape(QFrame.StyledPanel)
        pick_color_btn = QPushButton("Pick...", dlg)
        pick_color_btn.setMinimumWidth(78)
        auto_color_btn = QPushButton("Auto", dlg)
        auto_color_btn.setMinimumWidth(70)

        def _default_color_for_row() -> str:
            mapped = self.table_model.color_map.get(tgt.name)
            if isinstance(mapped, QColor) and mapped.isValid():
                return mapped.name()
            return self._target_plot_color_css(tgt, row, line_palette)

        def _refresh_color_preview():
            color_css = _normalized_css_color(color_edit.text())
            solid_border = self._theme_color("panel_border", "#50627a")
            dashed_border = self._theme_color("section_hint", solid_border)
            if color_css:
                sample = QColor(color_css)
                fg = "#111111" if sample.lightness() > 145 else "#f5f7fb"
                color_preview.setText(color_css.upper())
                color_preview.setStyleSheet(
                    f"background:{color_css}; color:{fg}; border:1px solid {solid_border}; border-radius:4px; padding:2px 6px;"
                )
                return
            fallback = _default_color_for_row()
            sample = QColor(fallback)
            fg = "#111111" if sample.lightness() > 145 else "#f5f7fb"
            color_preview.setText("Auto")
            color_preview.setStyleSheet(
                f"background:{fallback}; color:{fg}; border:1px dashed {dashed_border}; border-radius:4px; padding:2px 6px;"
            )

        def _pick_plot_color():
            seed = _normalized_css_color(color_edit.text()) or _default_color_for_row()
            chosen = QColorDialog.getColor(QColor(seed), dlg, "Pick object color")
            if chosen.isValid():
                color_edit.setText(chosen.name())

        color_edit.textChanged.connect(lambda _text: _refresh_color_preview())
        pick_color_btn.clicked.connect(_pick_plot_color)
        auto_color_btn.clicked.connect(color_edit.clear)
        color_row = QHBoxLayout()
        color_row.setContentsMargins(0, 0, 0, 0)
        color_row.setSpacing(8)
        color_row.addWidget(color_edit, 1)
        color_row.addWidget(pick_color_btn)
        color_row.addWidget(auto_color_btn)
        color_row.addWidget(color_preview)
        color_widget = QWidget(dlg)
        color_widget.setLayout(color_row)
        _refresh_color_preview()
        priority_spin = QSpinBox(dlg)
        priority_spin.setRange(1, 5)
        priority_spin.setValue(tgt.priority)
        observed_chk = QCheckBox("Observed", dlg)
        observed_chk.setChecked(tgt.observed)
        notes_edit = QTextEdit(dlg)
        notes_edit.setPlainText(tgt.notes or "")
        notes_edit.setMinimumHeight(120)

        form.addRow("Name:", name_edit)
        form.addRow("RA:", ra_edit)
        form.addRow("Dec:", dec_edit)
        form.addRow("Type:", type_edit)
        form.addRow("Magnitude:", mag_edit)
        form.addRow("Color:", color_widget)
        form.addRow("Priority:", priority_spin)
        form.addRow("", observed_chk)
        form.addRow("Notes:", notes_edit)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dlg)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        _style_dialog_button_box(buttons)
        form.addRow(buttons)
        _fit_dialog_to_screen(
            dlg,
            preferred_width=920,
            preferred_height=760,
            min_width=620,
            min_height=480,
        )

        while dlg.exec() == QDialog.Accepted:
            name = name_edit.text().strip()
            if not name:
                QMessageBox.warning(dlg, "Invalid Name", "Name cannot be empty.")
                continue
            try:
                ra_deg = parse_ra_to_deg(ra_edit.text().strip())
                dec_deg = parse_dec_to_deg(dec_edit.text().strip())
            except Exception as exc:
                QMessageBox.warning(dlg, "Invalid Coordinates", f"Cannot parse RA/Dec: {exc}")
                continue
            if not -90.0 <= dec_deg <= 90.0:
                QMessageBox.warning(dlg, "Invalid Coordinates", "Declination must be between -90 and +90 degrees.")
                continue

            mag_txt = mag_edit.text().strip()
            magnitude: Optional[float] = None
            if mag_txt:
                try:
                    magnitude = float(mag_txt)
                except ValueError:
                    QMessageBox.warning(dlg, "Invalid Magnitude", "Magnitude must be a numeric value.")
                    continue
            color_txt = color_edit.text().strip()
            plot_color = _normalized_css_color(color_txt)
            if color_txt and not plot_color:
                QMessageBox.warning(
                    dlg,
                    "Invalid Color",
                    "Color must be a valid CSS color (for example #4a86e8 or steelblue).",
                )
                continue

            tgt.name = name
            tgt.ra = float(ra_deg) % 360.0
            tgt.dec = float(dec_deg)
            tgt.object_type = type_edit.text().strip()
            tgt.magnitude = magnitude
            tgt.plot_color = plot_color
            tgt.priority = priority_spin.value()
            was_observed = bool(tgt.observed)
            tgt.observed = observed_chk.isChecked()
            tgt.notes = notes_edit.toPlainText().strip()

            # Metrics/windows are coordinate and condition dependent; refresh on next run.
            self.target_metrics.pop(old_name, None)
            self.target_metrics.pop(tgt.name, None)
            self.target_windows.pop(old_name, None)
            self.target_windows.pop(tgt.name, None)
            if old_name != tgt.name:
                self.table_model.color_map.pop(old_name, None)
            self._refresh_target_color_map()
            self._recompute_recommended_order_cache()

            self._emit_table_data_changed()
            self._update_selected_details()
            self._update_status_bar()
            self._record_observation_if_needed(tgt, was_observed=was_observed, source="edit_object")
            self._schedule_plan_autosave()
            self._replot_timer.start()
            return

    @Slot()
    def _edit_object_for_selected(self):
        rows = self._selected_rows()
        if not rows:
            QMessageBox.information(self, "No target selected", "Select a target row first.")
            return
        self._edit_object_for_row(rows[0])

    @Slot(QModelIndex)
    def _on_table_double_click(self, index: QModelIndex):
        if not index.isValid():
            return
        if index.column() == TargetTableModel.COL_OBSERVED:
            self._toggle_observed_row(index.row())
            return
        if index.column() == TargetTableModel.COL_NAME:
            self._edit_object_for_row(index.row())

    @Slot(object)
    def _open_table_context_menu(self, pos):
        index = self.table_view.indexAt(pos)
        if not index.isValid():
            return
        sel_model = self.table_view.selectionModel()
        if sel_model is None:
            return
        if not sel_model.isRowSelected(index.row(), QModelIndex()):
            sel_model.select(index, QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows)
        rows = self._selected_rows()
        if not rows:
            return
        target = self.targets[rows[0]]

        menu = QMenu(self.table_view)
        menu.setAttribute(Qt.WA_TranslucentBackground, False)
        menu.setWindowOpacity(1.0)
        menu.setMinimumWidth(240)
        menu_font = QFont(menu.font())
        menu_font.setPointSize(max(menu_font.pointSize(), 11))
        menu.setFont(menu_font)
        edit_obj_act = menu.addAction("Edit object...")
        open_bhtom_act = None
        if _normalize_catalog_token(target.source_catalog) == "bhtom":
            open_bhtom_act = menu.addAction("Open in BHTOM")
        toggle_act = menu.addAction("Toggle observed")
        menu.addSeparator()
        delete_act = menu.addAction("Remove selected")
        chosen = menu.exec(self.table_view.viewport().mapToGlobal(pos))
        if chosen == edit_obj_act:
            self._edit_object_for_row(rows[0])
        elif open_bhtom_act is not None and chosen == open_bhtom_act:
            self._open_target_in_bhtom(target)
        elif chosen == toggle_act:
            self._toggle_observed_selected()
        elif chosen == delete_act:
            self._delete_selected_targets()

    def _open_target_in_bhtom(self, target: Target) -> None:
        if _normalize_catalog_token(target.source_catalog) != "bhtom":
            return
        slug = quote(target.name.strip(), safe="")
        if not slug:
            return
        QDesktopServices.openUrl(QUrl(f"{self._bhtom_api_base_url()}/targets/{slug}/"))

    def _read_site_float(self, edit: QLineEdit) -> float:
        return float(edit.text().strip().replace(",", "."))

    def _validate_site_inputs(self) -> bool:
        invalid: list[QLineEdit] = []
        errors: list[str] = []
        try:
            lat = self._read_site_float(self.lat_edit)
            if not -90.0 <= lat <= 90.0:
                raise ValueError("Latitude must be within [-90, 90].")
        except Exception:
            invalid.append(self.lat_edit)
            errors.append("Invalid latitude")
        try:
            lon = self._read_site_float(self.lon_edit)
            if not -180.0 <= lon <= 180.0:
                raise ValueError("Longitude must be within [-180, 180].")
        except Exception:
            invalid.append(self.lon_edit)
            errors.append("Invalid longitude")
        try:
            self._read_site_float(self.elev_edit)
        except Exception:
            invalid.append(self.elev_edit)
            errors.append("Invalid elevation")

        if hasattr(self, "coord_error_label"):
            self.coord_error_label.setText("; ".join(errors))
            self.coord_error_label.setVisible(bool(errors))
        for edit in (self.lat_edit, self.lon_edit, self.elev_edit):
            _set_widget_invalid(edit, edit in invalid)
        return not invalid

    @Slot()
    def _on_site_inputs_changed(self):
        if not self._validate_site_inputs():
            return
        try:
            site = self._build_runtime_site_from_inputs()
        except (ValidationError, ValueError):
            return
        self.table_model.site = site
        self._start_clock_worker()
        self._schedule_plan_autosave()
        self._replot_timer.start()

    def _build_runtime_site_from_inputs(self) -> Site:
        selected_name = self.obs_combo.currentText() if hasattr(self, "obs_combo") else ""
        template: Optional[Site] = None
        if selected_name and hasattr(self, "observatories"):
            maybe = self.observatories.get(selected_name)
            if isinstance(maybe, Site):
                template = maybe
        if template is None:
            current_site = getattr(self.table_model, "site", None) if hasattr(self, "table_model") else None
            if isinstance(current_site, Site):
                template = current_site
        return Site(
            name=selected_name or (template.name if isinstance(template, Site) else "Custom"),
            latitude=self._read_site_float(self.lat_edit),
            longitude=self._read_site_float(self.lon_edit),
            elevation=self._read_site_float(self.elev_edit),
            limiting_magnitude=self._current_limiting_magnitude(),
            telescope_diameter_mm=float(getattr(template, "telescope_diameter_mm", 0.0) or 0.0),
            focal_length_mm=float(getattr(template, "focal_length_mm", 0.0) or 0.0),
            pixel_size_um=float(getattr(template, "pixel_size_um", 0.0) or 0.0),
            detector_width_px=int(getattr(template, "detector_width_px", 0) or 0),
            detector_height_px=int(getattr(template, "detector_height_px", 0) or 0),
            custom_conditions_url=str(getattr(template, "custom_conditions_url", "") or "").strip(),
        )

    def _cleanup_threads(self):
        """Stop timer and threads cleanly on exit (order matters)."""
        logger.info("Begin cleanup_threads")
        # Signal that shutdown has begun
        self._shutting_down = True
        cutout_resize_timer = getattr(self, "_cutout_resize_timer", None)
        if cutout_resize_timer is not None:
            try:
                cutout_resize_timer.stop()
            except Exception:
                pass
        # 1) Kill the periodic timer *first* so it can’t spawn new workers
        clock_timer = getattr(self, "_clock_timer", None)
        if clock_timer is not None:
            try:
                # Disconnect to avoid late-queued timeouts spawning new workers
                if hasattr(clock_timer, "timeout"):
                    clock_timer.timeout.disconnect(self._update_clock)
                clock_timer.stop()
            except Exception:
                pass
            logger.debug("Clock timer stopped")
            self._clock_timer = None

        # Stop pending cutout network request.
        cutout_reply = getattr(self, "_cutout_reply", None)
        if cutout_reply is not None and not cutout_reply.isFinished():
            try:
                cutout_reply.abort()
            except Exception:
                pass
        self._cutout_reply = None
        self._cutout_pending_key = ""
        self._cutout_pending_name = ""
        self._cutout_pending_background = False
        self._cutout_prefetch_queue.clear()
        self._cutout_prefetch_enqueued_keys.clear()
        self._cutout_prefetch_total = 0
        self._cutout_prefetch_completed = 0
        self._cutout_prefetch_cached = 0
        self._cutout_prefetch_active = False

        # 2) Stop *all* ClockWorkers created during this session
        for w in list(getattr(self, "_clock_workers", [])):
            try:
                # Skip if C++ object is already destroyed
                if shb.isValid(w) and w.isRunning():
                    # Ask it to stop only if still alive
                    w.running = False
                    if hasattr(w, "_wait_cond"):
                        w._wait_cond.wakeAll()
                    w.quit()
                    w.wait()
            except RuntimeError:
                # Already deleted; nothing to do
                pass
            except Exception as exc:
                logger.exception("Failed to stop ClockWorker: %s", exc)
        logger.debug("All ClockWorkers stopped (%d total)", len(getattr(self, "_clock_workers", [])))
        # Clear list
        self._clock_workers = []
        self.clock_worker = None

        # 3) Stop the AstronomyWorker if it’s still running
        worker = getattr(self, "worker", None)
        if worker is not None and worker.isRunning():
            try:
                worker.quit()
                worker.wait()
                logger.debug("AstronomyWorker stopped")
            except Exception as exc:
                logger.exception("Failed to stop AstronomyWorker: %s", exc)
        llm_worker = getattr(self, "_llm_worker", None)
        if llm_worker is not None:
            try:
                if llm_worker.isRunning():
                    llm_worker.requestInterruption()
                    llm_worker.quit()
                    if not llm_worker.wait(1200):
                        llm_worker.terminate()
                        llm_worker.wait(400)
            except Exception as exc:
                logger.exception("Failed to stop LLMWorker: %s", exc)
            self._llm_worker = None
        meta_worker = getattr(self, "_meta_worker", None)
        if meta_worker is not None:
            try:
                if meta_worker.isRunning():
                    meta_worker.cancel()
                    meta_worker.quit()
                    meta_worker.wait()
            except Exception as exc:  # noqa: BLE001
                logger.exception("Failed to stop MetadataLookupWorker: %s", exc)
            self._meta_worker = None
        startup_weather_worker = getattr(self, "_startup_weather_worker", None)
        if startup_weather_worker is not None:
            try:
                if startup_weather_worker.isRunning():
                    startup_weather_worker.requestInterruption()
                    startup_weather_worker.quit()
                    if not startup_weather_worker.wait(1500):
                        startup_weather_worker.terminate()
                        startup_weather_worker.wait(400)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Failed to stop startup WeatherLiveWorker: %s", exc)
            self._startup_weather_worker = None
        bhtom_worker = getattr(self, "_bhtom_worker", None)
        if bhtom_worker is not None:
            try:
                if bhtom_worker.isRunning():
                    bhtom_worker.requestInterruption()
                    bhtom_worker.quit()
                    if not bhtom_worker.wait(1500):
                        bhtom_worker.terminate()
                        bhtom_worker.wait(400)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Failed to stop BhtomSuggestionWorker: %s", exc)
            self._bhtom_worker = None
            self._bhtom_worker_mode = ""
            self._bhtom_worker_cache_key = None
            self._bhtom_worker_source = ""
            self._bhtom_dialog = None
        cand_worker = getattr(self, "_bhtom_candidate_prefetch_worker", None)
        if cand_worker is not None:
            try:
                if cand_worker.isRunning():
                    cand_worker.requestInterruption()
                    cand_worker.quit()
                    if not cand_worker.wait(1500):
                        cand_worker.terminate()
                        cand_worker.wait(400)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Failed to stop BhtomCandidatePrefetchWorker: %s", exc)
            self._bhtom_candidate_prefetch_worker = None
        obs_worker = getattr(self, "_bhtom_observatory_worker", None)
        if obs_worker is not None:
            try:
                if obs_worker.isRunning():
                    obs_worker.requestInterruption()
                    obs_worker.quit()
                    if not obs_worker.wait(1500):
                        obs_worker.terminate()
                        obs_worker.wait(400)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Failed to stop BhtomObservatoryPresetWorker: %s", exc)
            self._bhtom_observatory_worker = None
            self._bhtom_observatory_loading_message = ""
        self._stop_finder_workers(aggressive=True)
        self._finder_worker = None
        self._finder_pending_key = ""
        self._finder_pending_name = ""
        self._finder_pending_background = False
        self._finder_prefetch_queue.clear()
        self._finder_prefetch_enqueued_keys.clear()
        self._finder_prefetch_total = 0
        self._finder_prefetch_completed = 0
        self._finder_prefetch_cached = 0
        self._finder_prefetch_active = False
        self._finder_retry_after.clear()

    def _apply_default_sort(self):
        """Apply default sort column from settings."""
        default_sort = self.settings.value("table/defaultSortColumn", TargetTableModel.COL_SCORE, type=int)
        cols = self.table_model.columnCount()
        if 0 <= default_sort < cols:
            self.table_model.sort(default_sort, Qt.AscendingOrder)
            self.table_view.horizontalHeader().setSortIndicator(default_sort, Qt.AscendingOrder)

    def _reapply_current_table_sort(self) -> None:
        if not hasattr(self, "table_view") or not hasattr(self, "table_model"):
            return
        header = self.table_view.horizontalHeader()
        section = header.sortIndicatorSection()
        order = header.sortIndicatorOrder()
        cols = self.table_model.columnCount()
        if 0 <= section < cols:
            self.table_model.sort(section, order)
            header.setSortIndicator(section, order)
            return
        self._apply_default_sort()

    def _plan_target_key(self, target: Target) -> str:
        return self._plan_coordinator.plan_target_key(target)

    def _active_plan_storage_id(self) -> str:
        return self._plan_coordinator.active_plan_storage_id()

    def _serialize_current_targets(self) -> list[dict[str, object]]:
        return self._plan_coordinator.serialize_current_targets()

    def _current_site_snapshot(self) -> dict[str, object]:
        return self._plan_coordinator.current_site_snapshot()

    def _current_plan_snapshot(self) -> dict[str, object]:
        return self._plan_coordinator.current_plan_snapshot()

    def _set_plan_context(self, *, plan_id: str, plan_kind: str, plan_name: str) -> None:
        self._plan_coordinator.set_plan_context(
            plan_id=plan_id,
            plan_kind=plan_kind,
            plan_name=plan_name,
        )

    def _serialize_ai_messages_for_storage(self) -> list[dict[str, object]]:
        return self._plan_coordinator.serialize_ai_messages_for_storage()

    def _deserialize_ai_messages_from_storage(self, rows: list[dict[str, object]]) -> list[dict[str, Any]]:
        return self._plan_coordinator.deserialize_ai_messages_from_storage(rows)

    def _load_ai_messages_for_active_plan(self) -> None:
        self._plan_coordinator.load_ai_messages_for_active_plan()

    def _preload_cached_runtime_state_on_startup(self) -> None:
        return self._main_window_coordinator._preload_cached_runtime_state_on_startup()

    def _preload_cached_preview_images_on_startup(self) -> None:
        return self._main_window_coordinator._preload_cached_preview_images_on_startup()

    def _run_startup_sequence(self) -> None:
        return self._main_window_coordinator._run_startup_sequence()

    def _prefetch_bhtom_candidates_on_startup(self) -> None:
        return self._bhtom_coordinator._prefetch_bhtom_candidates_on_startup()

    def _start_bhtom_candidate_prefetch(self, *, force_refresh: bool = False) -> bool:
        return self._bhtom_coordinator._start_bhtom_candidate_prefetch(force_refresh=force_refresh)

    @Slot(int, list, str)
    def _on_bhtom_candidate_prefetch_completed(self, request_id: int, candidates: list, err: str) -> None:
        return self._bhtom_coordinator._on_bhtom_candidate_prefetch_completed(request_id, candidates, err)

    def _on_bhtom_candidate_prefetch_finished(self, worker: BhtomCandidatePrefetchWorker) -> None:
        return self._bhtom_coordinator._on_bhtom_candidate_prefetch_finished(worker)

    def _refresh_cached_bhtom_suggestions(self) -> None:
        return self._bhtom_coordinator._refresh_cached_bhtom_suggestions()

    def _prefetch_weather_on_startup(self) -> None:
        return self._main_window_coordinator._prefetch_weather_on_startup()

    @Slot(str, int, int)
    def _on_startup_weather_progress(self, status: str, _step: int, _total: int) -> None:
        return self._main_window_coordinator._on_startup_weather_progress(
            status,
            _step,
            _total,
        )

    @Slot(dict)
    def _on_startup_weather_completed(self, payload: dict) -> None:
        return self._main_window_coordinator._on_startup_weather_completed(
            payload,
        )

    def _on_startup_weather_finished(self, worker: WeatherLiveWorker) -> None:
        return self._main_window_coordinator._on_startup_weather_finished(
            worker,
        )

    def _persist_ai_messages_to_storage(self, *, allow_empty_clear: bool = False) -> None:
        self._plan_coordinator.persist_ai_messages_to_storage(allow_empty_clear=allow_empty_clear)

    def _persist_workspace_now(self) -> None:
        self._plan_coordinator.persist_workspace_now()

    def _persist_active_plan_now(self) -> None:
        self._plan_coordinator.persist_active_plan_now()

    def _schedule_plan_autosave(self) -> None:
        self._plan_coordinator.schedule_plan_autosave()

    def _flush_plan_autosave(self) -> None:
        self._plan_coordinator.flush_plan_autosave()

    def _ensure_named_site_available(self, site_snapshot: dict[str, object], *, preferred_name: str) -> None:
        self._plan_coordinator.ensure_named_site_available(site_snapshot, preferred_name=preferred_name)

    def _apply_plan_payload(
        self,
        snapshot: dict[str, object],
        target_payloads: list[dict[str, object]],
        *,
        plan_id: str,
        plan_kind: str,
        plan_name: str,
        defer_visual_refresh: bool = False,
        apply_snapshot_date: bool = True,
    ) -> None:
        self._plan_coordinator.apply_plan_payload(
            snapshot,
            target_payloads,
            plan_id=plan_id,
            plan_kind=plan_kind,
            plan_name=plan_name,
            defer_visual_refresh=defer_visual_refresh,
            apply_snapshot_date=apply_snapshot_date,
        )

    def _restore_plan_record(
        self,
        record: dict[str, object],
        *,
        defer_visual_refresh: bool = False,
        apply_snapshot_date: bool = False,
    ) -> None:
        self._plan_coordinator.restore_plan_record(
            record,
            defer_visual_refresh=defer_visual_refresh,
            apply_snapshot_date=apply_snapshot_date,
        )

    def _restore_workspace_on_startup(self, *, defer_visual_refresh: bool = False) -> bool:
        return self._plan_coordinator.restore_workspace_on_startup(defer_visual_refresh=defer_visual_refresh)

    def _load_plan_from_json_path(self, file_path: str, *, persist_workspace: bool = True) -> None:
        self._plan_coordinator.load_plan_from_json_path(file_path, persist_workspace=persist_workspace)

    @Slot()
    def _import_plan_json(self):
        self._plan_coordinator.import_plan_json()

    @Slot()
    def _load_plan(self):
        self._plan_coordinator.load_plan()

    @Slot()
    def _save_plan_as(self) -> None:
        self._plan_coordinator.save_plan_as()

    @Slot()
    def _open_saved_plan(self) -> None:
        self._plan_coordinator.open_saved_plan()

    @Slot()
    def _show_observation_history(self) -> None:
        self._plan_coordinator.show_observation_history()

    def _record_observation_if_needed(self, target: Target, *, was_observed: bool, source: str) -> None:
        self._plan_coordinator.record_observation_if_needed(target, was_observed=was_observed, source=source)

    # .....................................................
    # file / target helpers
    # .....................................................
    @Slot()
    def _load_targets(self):  # noqa: D401
        """Load a CSV/TSV/MPC target list file."""
        fn, _ = QFileDialog.getOpenFileName(
            self,
            "Open target list",
            str(Path.cwd()),
            "Text files (*.csv *.tsv *.txt)",
        )
        if not fn:
            return

        try:
            loaded_targets: list[Target] = []
            self.target_metrics.clear()
            self.target_windows.clear()
            with open(fn, newline="", encoding="utf-8") as fh:
                dialect = csv.Sniffer().sniff(fh.read(2048))
                fh.seek(0)
                rdr = csv.DictReader(fh, dialect=dialect)
                for row in rdr:
                    priority = int(row.get("priority", 3) or 3)
                    priority = min(5, max(1, priority))
                    magnitude = row.get("magnitude") or None
                    size_arcmin = row.get("size_arcmin") or None
                    observed_raw = str(row.get("observed", "")).strip().lower()
                    loaded_targets.append(
                        Target(
                            name=row["name"],
                            ra=parse_ra_to_deg(row["ra"]),
                            dec=parse_dec_to_deg(row["dec"]),
                            object_type=row.get("object_type", row.get("type", "")) or "",
                            magnitude=float(magnitude) if magnitude is not None else None,
                            size_arcmin=float(size_arcmin) if size_arcmin is not None else None,
                            priority=priority,
                            observed=observed_raw in {"1", "true", "yes", "y", "tak"},
                            notes=row.get("notes", "") or "",
                        )
                    )
            self.table_model.reset_targets(loaded_targets)
        except (FileNotFoundError, KeyError, ValueError, ValidationError, csv.Error) as exc:
            QMessageBox.critical(self, "Load error", str(exc))
            return

        # Refresh table view after successful load
        # Reapply settings & default sort after layout change
        self._apply_table_settings()
        self._apply_default_sort()
        self._fetch_missing_magnitudes_async()

    @Slot()
    def _add_target_dialog(self):
        """Prompt user for a target name or coordinate string and append it."""
        default_source = _normalize_catalog_token(
            self.settings.value("general/targetSearchSource", "simbad", type=str)
        )
        dlg = AddTargetDialog(
            self._resolve_target,
            self._enrich_target_metadata_for_dialog,
            self,
            source_options=TARGET_SEARCH_SOURCES,
            default_source=default_source,
        )
        if dlg.exec() != QDialog.Accepted:
            return
        self.settings.setValue("general/targetSearchSource", dlg.selected_source())
        try:
            target = dlg.build_target()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Resolve error", str(exc))
            return
        self._append_target_to_plan(target)

    def _plan_contains_target(self, target: Target) -> bool:
        return any(_targets_match(existing, target) for existing in self.targets)

    def _try_fast_append_visibility_update(self, target: Target) -> bool:
        if getattr(self, "worker", None) is not None and self.worker.isRunning():
            return False
        payload = getattr(self, "last_payload", None)
        if not isinstance(payload, dict):
            return False
        current_key = self._current_visibility_context_key()
        payload_key = str(payload.get("site_key", "") or "")
        if not current_key or not payload_key or payload_key != current_key:
            return False
        if target.name in payload:
            return True
        if "times" not in payload or "moon_ra" not in payload or "moon_dec" not in payload:
            return False
        if not self._validate_site_inputs():
            return False
        try:
            site = self._build_runtime_site_from_inputs()
            tz_name = str(payload.get("tz", site.timezone_name) or site.timezone_name)
            observer = Observer(location=site.to_earthlocation(), timezone=tz_name)
            times_num = np.array(payload["times"], dtype=float)
            times = Time(mdates.num2date(times_num))
            fixed = FixedTarget(name=target.name, coord=target.skycoord)
            altaz = observer.altaz(times, fixed)
            moon_ra = np.array(payload.get("moon_ra", []), dtype=float)
            moon_dec = np.array(payload.get("moon_dec", []), dtype=float)
            if moon_ra.size != times_num.size or moon_dec.size != times_num.size:
                return False
            moon_coords = SkyCoord(ra=moon_ra * u.deg, dec=moon_dec * u.deg)
            moon_sep = target.skycoord.separation(moon_coords).deg
            merged = dict(payload)
            merged[target.name] = {
                "altitude": np.array(altaz.alt.deg, dtype=float),  # type: ignore[arg-type]
                "azimuth": np.array(altaz.az.deg, dtype=float),  # type: ignore[arg-type]
                "moon_sep": np.array(moon_sep, dtype=float),
            }
            self._update_plot(merged)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.info("Fast append visibility update failed for %s: %s", target.name, exc)
            return False

    def _append_target_to_plan(
        self,
        target: Target,
        refresh: bool = True,
        notify_duplicate: bool = True,
    ) -> bool:
        if self._plan_contains_target(target):
            if notify_duplicate:
                QMessageBox.information(self, "Already in plan", f"{target.name} is already present in the current plan.")
            return False

        target_copy = Target(**target.model_dump())
        self._ensure_known_target_type(target_copy)
        self.table_model.append_target(target_copy)
        if refresh:
            self._recompute_recommended_order_cache()
            self._apply_table_settings()
            self._apply_default_sort()
            self._refresh_target_color_map()
            self._emit_table_data_changed()
            self._fetch_missing_magnitudes_async()
            if not self._try_fast_append_visibility_update(target_copy):
                self._replot_timer.start()

            for row_idx, existing in enumerate(self.targets):
                if existing is target_copy or _targets_match(existing, target_copy):
                    self.table_view.selectRow(row_idx)
                    self.table_view.scrollTo(self.table_model.index(row_idx, TargetTableModel.COL_NAME))
                    break
            self._update_selected_details()
            self._prefetch_cutouts_for_all_targets(prioritize=self._selected_target_or_none())
            self._prefetch_finder_charts_for_all_targets(prioritize=self._selected_target_or_none())
        self._schedule_plan_autosave()
        return True

    @Slot()
    def _run_plan(self):
        """Kick off the worker thread unless one is already running."""
        logger.info("Starting new visibility calculation …")
        self._ensure_visibility_plot_widgets()
        if self.worker and self.worker.isRunning():
            self._queued_plan_run = True
            try:
                self.worker.requestInterruption()
            except Exception:
                pass
            self._begin_visibility_refresh("Updating visibility for the new context…")
            return
        if not self._validate_site_inputs():
            return
        try:
            site = self._build_runtime_site_from_inputs()
            settings = SessionSettings(
                date=self.date_edit.date(),
                site=site,
                limit_altitude=float(self.limit_spin.value()),
                time_samples=self.settings.value("general/timeSamples", 240, type=int),
            )
        except (ValidationError, ValueError) as exc:
            QMessageBox.critical(self, "Invalid input", str(exc))
            return

        self.table_model.site = site
        self._queued_plan_run = False
        self._pending_visibility_context_key = self._visibility_context_key_from_parts(
            site_name=site.name,
            latitude=float(site.latitude),
            longitude=float(site.longitude),
            elevation=float(site.elevation),
            obs_date=settings.date,
            time_samples=int(settings.time_samples),
            limit_altitude=float(settings.limit_altitude),
        )
        # Update the observation limit for table coloring
        self.table_model.limit = settings.limit_altitude
        self._emit_table_data_changed()
        self._calc_started_at = perf_counter()
        self._update_status_bar()

        self.worker = AstronomyWorker(self.targets, settings, parent=self)
        self.worker.aborted.connect(self._on_astronomy_worker_aborted)
        self.worker.finished.connect(self._update_plot)
        self.worker.finished.connect(lambda _: self.plot_canvas.setEnabled(True))
        if getattr(self, "visibility_web", None) is not None:
            self.worker.finished.connect(lambda _: self.visibility_web.setEnabled(True))
        # disable canvas during computation
        self.plot_canvas.setEnabled(False)
        if getattr(self, "visibility_web", None) is not None:
            self.visibility_web.setEnabled(False)
        self._begin_visibility_refresh("Computing night tracks…")
        self.worker.start()

    def _remove_target(self, row: int):
        if not (0 <= row < len(self.targets)):
            return
        removed_name = self.targets[row].name
        self.target_metrics.pop(removed_name, None)
        self.target_windows.pop(removed_name, None)
        self.table_model.remove_rows([row])
        self._recompute_recommended_order_cache()
        # Reapply settings & default sort after layout change
        self._apply_table_settings()
        self._apply_default_sort()
        self._update_selected_details()
        # Debounce and update plot after removing a target
        self._schedule_plan_autosave()
        self._replot_timer.start()

    @Slot()
    def _delete_selected_targets(self):
        """Delete all currently selected targets via Ctrl/Cmd+Delete."""
        # Collect selected rows, sort descending to avoid reindexing issues
        rows = self._selected_rows()
        if not rows:
            return
        removed = self.table_model.remove_rows(rows)
        for tgt in removed:
            self.target_metrics.pop(tgt.name, None)
            self.target_windows.pop(tgt.name, None)
        self._recompute_recommended_order_cache()
        self._apply_table_settings()
        self._apply_default_sort()
        self._update_selected_details()
        self._schedule_plan_autosave()
        self._replot_timer.start()

    def _toggle_observed_row(self, row: int):
        if 0 <= row < len(self.targets):
            was_observed = bool(self.targets[row].observed)
            self.targets[row].observed = not self.targets[row].observed
            self._emit_table_data_changed()
            self._update_selected_details()
            self._record_observation_if_needed(self.targets[row], was_observed=was_observed, source="toggle_row")
            self._schedule_plan_autosave()
            if self.last_payload is not None:
                self._update_plot(self.last_payload)
            else:
                self._replot_timer.start()

    @Slot()
    def _toggle_observed_selected(self):
        rows = self._selected_rows()
        if not rows:
            return
        for row in rows:
            if 0 <= row < len(self.targets):
                target = self.targets[row]
                was_observed = bool(target.observed)
                target.observed = not target.observed
                self._record_observation_if_needed(target, was_observed=was_observed, source="toggle_selected")
        self._emit_table_data_changed()
        self._update_selected_details()
        self._schedule_plan_autosave()
        if self.last_payload is not None:
            self._update_plot(self.last_payload)
        else:
            self._replot_timer.start()

    def _emit_table_data_changed(self):
        coordinator = getattr(self, "_table_coordinator", None)
        if coordinator is not None:
            coordinator.emit_table_data_changed()

    def _apply_table_row_visibility(self):
        coordinator = getattr(self, "_table_coordinator", None)
        if coordinator is not None:
            coordinator.apply_table_row_visibility()

    def _clear_table_dynamic_cache(self):
        coordinator = getattr(self, "_table_coordinator", None)
        if coordinator is not None:
            coordinator.clear_table_dynamic_cache()

    def _refresh_table_buttons(self):
        # Row widgets were replaced by a context menu to keep the table fast.
        return

    @Slot()
    def _on_astronomy_worker_aborted(self) -> None:
        worker = getattr(self, "worker", None)
        if worker is not None and worker.isRunning():
            QTimer.singleShot(40, self._on_astronomy_worker_aborted)
            return
        self.worker = None
        if self._queued_plan_run:
            self._queued_plan_run = False
            QTimer.singleShot(0, self._run_plan)

    @Slot(dict)
    def _update_plot(self, data: dict):
        self._visibility_matplotlib._update_plot(data, sender=self.sender())

    @Slot()
    def _toggle_visibility(self):
        self._visibility_coordinator.toggle_visibility()

    @Slot(bool)
    def _on_plot_mode_switch_changed(self, checked: bool):
        self._visibility_coordinator.on_plot_mode_switch_changed(checked)

    @Slot()
    def _update_clock(self):
        # Skip if we’re in the middle of shutdown
        if self._shutting_down:
            return
        if self.clock_worker is None and self.table_model.site:
            self._start_clock_worker()


    @Slot(dict)
    def _handle_clock_update(self, data):
        self.localtime_label.setText(data["now_local"].strftime("%Y-%m-%d %H:%M:%S"))
        self.utctime_label.setText(data["now_utc"].strftime("%Y-%m-%d %H:%M:%S"))
        self._refresh_weather_window_context(rebuild=False)
        self.sun_alt_label.setText(f"{data['sun_alt']:.1f}°")
        self.moon_alt_label.setText(f"{data['moon_alt']:.1f}°")
        self.table_model.current_alts = data["alts"]
        self.table_model.current_azs = data["azs"]
        self.table_model.current_seps = data["seps"]
        self.table_model.row_enabled = self._recompute_row_enabled_from_current()
        self._reapply_current_table_sort()
        self._apply_table_row_visibility()
        self._emit_table_data_changed()
        self._update_status_bar()

        # Update sidereal time based on local time and site longitude
        if hasattr(self, 'last_payload') and self.last_payload:
            now = data["now_local"]
            # sidereal time calculation
            from astropy.time import Time
            if self.table_model.site is not None:
                sidereal = Time(data["now_local"]).sidereal_time('apparent', self.table_model.site.to_earthlocation().lon)
                # Format as HH:MM:SS
                self.sidereal_label.setText(sidereal.to_string(unit=u.hour, sep=":", pad=True, precision=0))
            else:
                self.sidereal_label.setText("-")
            if hasattr(self, 'now_line') and self.now_line:
                self.now_line.set_xdata([float(mdates.date2num(now)), float(mdates.date2num(now))])
            else:
                self.now_line = self.ax_alt.axvline(
                    float(mdates.date2num(now)), color="magenta", linestyle=":", linewidth=1.2, label="Now"
                )
            self.plot_canvas.draw_idle()
            current_minute_key = now.strftime("%Y-%m-%d %H:%M")
            if getattr(self, "_visibility_web_minute_key", "") != current_minute_key:
                self._visibility_web_minute_key = current_minute_key
                self.last_payload["now_local"] = now
                self._render_visibility_web_plot(self.last_payload, now_override=now)
            self._update_polar_positions(data, dynamic_only=True)


    @Slot()
    def _update_polar_selection(self, selected, deselected):
        self._visibility_coordinator.update_polar_selection(selected, deselected)

    @Slot(object)
    def _on_polar_pick(self, event):
        self._visibility_coordinator.on_polar_pick(event)

    @Slot(object, object)
    def _update_vis_selection(self, selected, deselected):
        self._visibility_coordinator.update_vis_selection(selected, deselected)

    @staticmethod
    def _build_polar_visible_path(
        alt_series: object,
        az_series: object,
    ) -> tuple[np.ndarray, np.ndarray]:
        return VisibilityCoordinator.build_polar_visible_path(alt_series, az_series)

    @Slot(dict)
    def _update_polar_positions(self, data: dict, dynamic_only: bool = False):
        self._visibility_coordinator.update_polar_positions(data, dynamic_only=dynamic_only)

    @Slot()
    def _toggle_dark(self):
        """Toggle dark mode while preserving the base stylesheet."""
        self._set_dark_mode_enabled(not self._dark_enabled, persist=True)

    @Slot()
    def _export_plan(self):
        """Write targets JSON, plot PNG, session CSV, and calendar ICS."""
        out_dir = QFileDialog.getExistingDirectory(
            self, "Select export directory", str(Path.cwd())
        )
        if not out_dir:
            return
        out_path = Path(out_dir)
        # JSON
        with open(out_path / "plan_targets.json", "w", encoding="utf-8") as fh:
            json.dump([t.model_dump() for t in self.targets], fh, indent=2)
        # PNG
        self._ensure_visibility_plot_widgets()
        if self.plot_canvas is not None:
            self.plot_canvas.figure.savefig(out_path / "plan_plot.png", dpi=150)
        # CSV summary and ICS window schedule
        tz_name = "UTC"
        if isinstance(getattr(self, "last_payload", None), dict):
            tz_name = str(self.last_payload.get("tz", "UTC"))
        elif self.table_model.site:
            tz_name = self.table_model.site.timezone_name

        rows: list[dict[str, object]] = []
        ics_events: list[dict[str, object]] = []
        for tgt in self.targets:
            metrics = self.target_metrics.get(tgt.name)
            window = self.target_windows.get(tgt.name)
            start_txt = window[0].strftime("%Y-%m-%d %H:%M") if window else ""
            end_txt = window[1].strftime("%Y-%m-%d %H:%M") if window else ""
            rows.append({
                "name": tgt.name,
                "ra_deg": round(tgt.ra, 6),
                "dec_deg": round(tgt.dec, 6),
                "priority": tgt.priority,
                "observed": tgt.observed,
                "score": metrics.score if metrics else 0.0,
                "hours_above_limit": metrics.hours_above_limit if metrics else 0.0,
                "max_altitude_deg": metrics.max_altitude_deg if metrics else 0.0,
                "peak_moon_sep_deg": metrics.peak_moon_sep_deg if metrics else 0.0,
                "window_start_local": start_txt,
                "window_end_local": end_txt,
                "notes": tgt.notes,
            })
            if window:
                score_txt = f"{metrics.score:.1f}" if metrics else "0.0"
                ics_events.append({
                    "title": f"Observe {tgt.name}",
                    "start": window[0],
                    "end": window[1],
                    "description": f"Target score {score_txt}, priority {tgt.priority}.",
                })

        export_metrics_csv(out_path / "plan_summary.csv", rows)
        export_observation_ics(out_path / "plan_schedule.ics", tz_name=tz_name, events=ics_events)
        QMessageBox.information(self, "Export complete", f"Wrote files to {out_path}")

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _enrich_target_metadata_for_dialog(self, target: Target) -> None:
        return self._target_resolver._enrich_target_metadata_for_dialog(target)

    def _cancel_metadata_lookup(self):
        return self._target_resolver._cancel_metadata_lookup()

    @Slot(int, list)
    def _on_metadata_lookup_completed(self, request_id: int, results: list):
        return self._target_resolver._on_metadata_lookup_completed(request_id, results)

    def _fetch_missing_magnitudes_async(self) -> int:
        return self._target_resolver._fetch_missing_magnitudes_async()

    def _fetch_missing_magnitudes(self, targets: Optional[list[Target]] = None, emit_table: bool = True) -> int:
        return self._target_resolver._fetch_missing_magnitudes(targets, emit_table)

    def _resolve_target_from_coordinates(self, query: str) -> Optional[Target]:
        return self._target_resolver._resolve_target_from_coordinates(query)

    def _resolve_target_simbad(self, query: str) -> Target:
        return self._target_resolver._resolve_target_simbad(query)

    def _resolve_target_gaia_dr3(self, query: str) -> Target:
        return self._target_resolver._resolve_target_gaia_dr3(query)

    def _load_gaia_alerts_cache(self, force_refresh: bool = False):
        return self._target_resolver._load_gaia_alerts_cache(force_refresh)

    def _resolve_target_gaia_alerts(self, query: str) -> Target:
        return self._target_resolver._resolve_target_gaia_alerts(query)

    def _resolve_target_tns(self, query: str) -> Target:
        return self._target_resolver._resolve_target_tns(query)

    def _resolve_target_ned(self, query: str) -> Target:
        return self._target_resolver._resolve_target_ned(query)

    def _resolve_target_lsst(self, query: str) -> Target:
        return self._target_resolver._resolve_target_lsst(query)

    def _resolve_target(self, query: str, source: str = "simbad") -> Target:
        return self._target_resolver._resolve_target(query, source)

    def _build_ai_panel(self, parent: Optional[QWidget] = None) -> QWidget:
        return self._ai_panel_coordinator.build_panel(parent)

    def _build_ai_window(self) -> QDialog:
        return self._ai_panel_coordinator.build_window()

    def _ensure_ai_window(self) -> QDialog:
        return self._ai_panel_coordinator.ensure_window()

    @Slot(bool)
    def _toggle_ai_panel(self, checked: bool) -> None:
        self._ai_panel_coordinator.toggle_panel(checked)

    @Slot(int)
    def _on_ai_window_finished(self, _result: int) -> None:
        self._ai_panel_coordinator.on_window_finished(_result)

    def _llm_warmup_cache_key(self) -> tuple[str, str]:
        return self._ai_panel_coordinator.warmup_cache_key()

    def _llm_is_warm(self) -> bool:
        return self._ai_panel_coordinator.is_warm()

    def _start_llm_warmup(
        self,
        *,
        force: bool = False,
        user_initiated: bool = False,
        silent: bool = False,
    ) -> bool:
        return self._ai_panel_coordinator.start_warmup(
            force=force,
            user_initiated=user_initiated,
            silent=silent,
        )

    @Slot()
    def _warmup_llm_if_needed(self) -> None:
        self._ai_panel_coordinator.warmup_if_needed()

    @Slot()
    def _warmup_llm_manual(self) -> None:
        self._ai_panel_coordinator.warmup_manual()

    @Slot()
    def _warmup_llm_on_startup(self) -> None:
        self._ai_panel_coordinator.warmup_on_startup()

    def _refresh_ai_warm_indicator(self) -> None:
        self._ai_panel_coordinator.refresh_warm_indicator()

    def _set_ai_status(self, text: str, *, tone: str = "info") -> None:
        self._ai_panel_coordinator.set_status(text, tone=tone)

    @Slot()
    def _clear_ai_runtime_status(self) -> None:
        self._ai_panel_coordinator.clear_runtime_status()

    @staticmethod
    def _apply_label_glow_effect(label: Optional[QLabel], color: QColor, *, blur_radius: float) -> None:
        AIPanelCoordinator.apply_label_glow_effect(label, color, blur_radius=blur_radius)

    def _current_site_for_weather(self) -> Optional[Site]:
        site = getattr(self.table_model, "site", None) if hasattr(self, "table_model") else None
        if isinstance(site, Site):
            return site
        if hasattr(self, "obs_combo") and hasattr(self, "observatories"):
            selected = self.obs_combo.currentText()
            maybe = self.observatories.get(selected)
            if isinstance(maybe, Site):
                return maybe
        return None

    def _refresh_weather_window_context(self, *, rebuild: bool = True) -> None:
        window = getattr(self, "weather_window", None)
        if not isinstance(window, WeatherDialog):
            return
        site = self._current_site_for_weather()
        obs_name = site.name if isinstance(site, Site) else (
            self.obs_combo.currentText() if hasattr(self, "obs_combo") else "-"
        )
        date = self.date_edit.date() if hasattr(self, "date_edit") else QDate.currentDate()
        local_time_text = self.localtime_label.text() if hasattr(self, "localtime_label") else "-"
        utc_time_text = self.utctime_label.text() if hasattr(self, "utctime_label") else "-"
        sunrise_text = self.sunrise_label.text() if hasattr(self, "sunrise_label") else "-"
        sunset_text = self.sunset_label.text() if hasattr(self, "sunset_label") else "-"
        moonrise_text = self.moonrise_label.text() if hasattr(self, "moonrise_label") else "-"
        moonset_text = self.moonset_label.text() if hasattr(self, "moonset_label") else "-"
        moon_phase_pct = int(self.moonphase_bar.value()) if hasattr(self, "moonphase_bar") else 0
        window.set_context(
            site=site,
            obs_name=obs_name,
            date=date,
            sun_alt_limit=self._sun_alt_limit(),
            local_time_text=local_time_text,
            utc_time_text=utc_time_text,
            sunrise_text=sunrise_text,
            sunset_text=sunset_text,
            moonrise_text=moonrise_text,
            moonset_text=moonset_text,
            moon_phase_pct=moon_phase_pct,
            rebuild=rebuild,
        )

    @Slot()
    def _open_weather_window(self) -> None:
        window = getattr(self, "weather_window", None)
        if not isinstance(window, WeatherDialog):
            window = WeatherDialog(self)
            window.setStyleSheet(self.styleSheet())
            self.weather_window = window
        self._refresh_weather_window_context()
        window.show()
        window.raise_()
        window.activateWindow()

    def _selected_target_row_index(self) -> Optional[int]:
        return self._ai_context_coordinator._selected_target_row_index()

    def _build_llm_target_summary_line(
        self,
        row_index: int,
        target: Target,
        *,
        include_current_snapshot: bool = True,
    ) -> str:
        return self._ai_context_coordinator._build_llm_target_summary_line(
            row_index,
            target,
            include_current_snapshot=include_current_snapshot,
        )

    def _session_context_target_indices(self, *, max_items: int = 8) -> tuple[list[int], int]:
        return self._ai_context_coordinator._session_context_target_indices(
            max_items=max_items,
        )

    def _build_session_context(
        self,
        *,
        include_current_snapshot: bool = True,
        user_question: str = "",
    ) -> str:
        return self._ai_context_coordinator._build_session_context(
            include_current_snapshot=include_current_snapshot,
            user_question=user_question,
        )

    def _build_bhtom_suggestion_shortlist_context(
        self,
        *,
        user_question: str = "",
        max_items: int = 5,
    ) -> str:
        return self._ai_context_coordinator._build_bhtom_suggestion_shortlist_context(
            user_question=user_question,
            max_items=max_items,
        )

    def _target_class_family(self, target: Target) -> str:
        return self._ai_context_coordinator._target_class_family(
            target,
        )

    def _find_referenced_target_in_question(self, text: str) -> Optional[Target]:
        return self._ai_context_coordinator._find_referenced_target_in_question(
            text,
        )

    def _find_referenced_targets_in_question(self, text: str, *, max_targets: int = 6) -> list[Target]:
        return self._ai_context_coordinator._find_referenced_targets_in_question(
            text,
            max_targets=max_targets,
        )

    def _plan_row_index_for_target(self, target: Target) -> Optional[int]:
        return self._ai_context_coordinator._plan_row_index_for_target(
            target,
        )

    def _lookup_target_observing_candidate(self, target: Target) -> Optional[dict[str, object]]:
        return self._ai_context_coordinator._lookup_target_observing_candidate(
            target,
        )

    def _load_knowledge_notes(self) -> list[KnowledgeNote]:
        return self._ai_context_coordinator._load_knowledge_notes()

    def _select_knowledge_notes(
        self,
        *,
        question: str,
        target: Optional[Target] = None,
        max_notes: int = 3,
        max_chars: int = 1600,
    ) -> list[KnowledgeNote]:
        return self._ai_context_coordinator._select_knowledge_notes(
            question=question,
            target=target,
            max_notes=max_notes,
            max_chars=max_chars,
        )

    def _knowledge_request_tags(self, question: str, target: Optional[Target] = None) -> set[str]:
        return self._ai_context_coordinator._knowledge_request_tags(
            question,
            target,
        )

    def _knowledge_note_score(
        self,
        note: KnowledgeNote,
        *,
        request_tags: set[str],
        question: str,
        target: Optional[Target],
    ) -> int:
        return self._ai_context_coordinator._knowledge_note_score(
            note,
            request_tags=request_tags,
            question=question,
            target=target,
        )

    def _build_knowledge_context(
        self,
        *,
        question: str,
        target: Optional[Target] = None,
        max_notes: int = 3,
        max_chars: int = 1600,
    ) -> str:
        return self._ai_context_coordinator._build_knowledge_context(
            question=question,
            target=target,
            max_notes=max_notes,
            max_chars=max_chars,
        )

    def _build_local_object_fact_answer(self, question: str, *, target: Optional[Target] = None) -> Optional[str]:
        return self._ai_context_coordinator._build_local_object_fact_answer(
            question,
            target=target,
        )

    def _parse_class_query_spec(self, question: str) -> Optional[ClassQuerySpec]:
        return self._ai_context_coordinator._parse_class_query_spec(
            question,
        )

    def _looks_like_class_observing_query(self, text: str) -> bool:
        return self._ai_context_coordinator._looks_like_class_observing_query(
            text,
        )

    def _collect_class_observing_candidates(self, class_query: ClassQuerySpec) -> list[dict[str, object]]:
        return self._ai_context_coordinator._collect_class_observing_candidates(
            class_query,
        )

    def _recent_ai_conversation_state(self, *, max_messages: int = 8) -> dict[str, Any]:
        return self._ai_context_coordinator._recent_ai_conversation_state(
            max_messages=max_messages,
        )

    def _resolve_recent_class_marker(self, *, max_messages: int = 6) -> str:
        return self._ai_context_coordinator._resolve_recent_class_marker(
            max_messages=max_messages,
        )

    @staticmethod
    def _class_query_requested_count(question: str, *, default_count: int = 3, max_count: int = 10) -> int:
        return AIContextCoordinator._class_query_requested_count(
            question,
            default_count=default_count,
            max_count=max_count,
        )

    def _build_local_class_observing_response(self, question: str) -> Optional[dict[str, Any]]:
        return self._ai_context_coordinator._build_local_class_observing_response(
            question,
        )

    def _build_local_class_observing_answer(self, question: str) -> Optional[str]:
        return self._ai_context_coordinator._build_local_class_observing_answer(
            question,
        )

    def _build_local_object_observing_answer(self, question: str, *, target: Optional[Target] = None) -> Optional[str]:
        return self._ai_context_coordinator._build_local_object_observing_answer(
            question,
            target=target,
        )

    @staticmethod
    def _find_normalized_text_position(text: str, candidate: str) -> Optional[int]:
        return AIContextCoordinator._find_normalized_text_position(
            text,
            candidate,
        )

    def _extract_addable_bhtom_targets_from_ai_text(self, text: str, *, max_items: int = 4) -> list[Target]:
        return self._ai_context_coordinator._extract_addable_bhtom_targets_from_ai_text(
            text,
            max_items=max_items,
        )

    def _resolve_recent_chat_target_reference(self, *, max_messages: int = 6) -> Optional[Target]:
        return self._ai_context_coordinator._resolve_recent_chat_target_reference(
            max_messages=max_messages,
        )

    def _parse_object_query_spec(
        self,
        question: str,
        *,
        selected_target: Optional[Target] = None,
    ) -> Optional[ObjectQuerySpec]:
        return self._ai_context_coordinator._parse_object_query_spec(
            question,
            selected_target=selected_target,
        )

    def _parse_compare_query_spec(
        self,
        question: str,
        *,
        selected_target: Optional[Target] = None,
    ) -> Optional[CompareQuerySpec]:
        return self._ai_context_coordinator._parse_compare_query_spec(
            question,
            selected_target=selected_target,
        )

    def _build_local_compare_response(
        self,
        question: str,
        *,
        compare_query: CompareQuerySpec,
    ) -> Optional[dict[str, Any]]:
        return self._ai_context_coordinator._build_local_compare_response(
            question,
            compare_query=compare_query,
        )

    def _resolve_object_query_target(self, question: str, *, selected_target: Optional[Target]) -> Optional[Target]:
        return self._ai_context_coordinator._resolve_object_query_target(
            question,
            selected_target=selected_target,
        )

    def _should_auto_route_selected_target_query(self, text: str, target: Optional[Target]) -> bool:
        return self._ai_context_coordinator._should_auto_route_selected_target_query(
            text,
            target,
        )

    def _dispatch_selected_target_llm_question(self, target: Target, question: str, *, label: str) -> None:
        return self._ai_context_coordinator._dispatch_selected_target_llm_question(
            target,
            question,
            label=label,
        )

    def _build_deterministic_observation_order(self) -> tuple[list[dict[str, object]], list[str]]:
        return self._ai_context_coordinator._build_deterministic_observation_order()

    _SYSTEM_PROMPT = (
        "You are an expert astronomy observation assistant. "
        "Provide concise, practical, and accurate guidance for planning observations. "
        "Keep answers compact by default and avoid unnecessary background. "
        "Do not repeat the same metric, sentence, or recommendation. "
        "If the user asks about a specific class such as QSO, AGN, supernova, nova, CV, or galaxy, restrict the answer to that class only. "
        "Use only the provided session context when answering questions about tonight's plan. "
        "If local knowledge notes are included, treat them as grounded heuristics and caveats for practical advice. "
        "Do not introduce targets that are not listed in the provided session context unless the user explicitly asks for off-plan suggestions. "
        "If a cached BHTOM suggestion shortlist is included in the session context, you may use only those listed BHTOM suggestions as off-plan candidates. "
        "Use target names exactly as they appear in the provided context; do not swap in unrelated Messier, NGC, or other catalog examples. "
        "Interpret dusk as the start of the observing night window and dawn as the end of the observing night window. "
        "If the data is missing or ambiguous, say so instead of guessing."
    )

    def _format_target_coords_compact(self, target: Target) -> str:
        return self._ai_context_coordinator._format_target_coords_compact(
            target,
        )

    def _format_target_best_window_compact(self, target: Target) -> str:
        return self._ai_context_coordinator._format_target_best_window_compact(
            target,
        )

    def _build_fast_target_llm_context(self, target: Target) -> str:
        return self._ai_context_coordinator._build_fast_target_llm_context(
            target,
        )

    @staticmethod
    def _format_compact_number(value: float, *, decimals_small: int = 3) -> str:
        return AIContextCoordinator._format_compact_number(
            value,
            decimals_small=decimals_small,
        )

    @staticmethod
    def _format_signed_value(value: float, decimals: int = 2) -> str:
        return AIContextCoordinator._format_signed_value(
            value,
            decimals,
        )

    def _is_star_like_target(self, target: Target, details: dict[str, object]) -> bool:
        return self._ai_context_coordinator._is_star_like_target(
            target,
            details,
        )

    def _format_distance_text(self, details: dict[str, object]) -> str:
        return self._ai_context_coordinator._format_distance_text(
            details,
        )

    def _format_size_text(self, details: dict[str, object]) -> str:
        return self._ai_context_coordinator._format_size_text(
            details,
        )

    def _format_kinematics_text(self, details: dict[str, object]) -> str:
        return self._ai_context_coordinator._format_kinematics_text(
            details,
        )

    def _get_simbad_compact_data(self, target: Target) -> dict[str, object]:
        return self._ai_context_coordinator._get_simbad_compact_data(
            target,
        )

    def _build_compact_target_description(self, target: Target) -> str:
        return self._ai_context_coordinator._build_compact_target_description(
            target,
        )

    def _bhtom_api_base_url(self) -> str:
        return self._bhtom_coordinator._bhtom_api_base_url()

    def _bhtom_api_token_optional(self) -> str:
        return self._bhtom_coordinator._bhtom_api_token_optional()

    def _bhtom_api_token(self) -> str:
        return self._bhtom_coordinator._bhtom_api_token()

    def _current_bhtom_cache_identity(self) -> Optional[tuple[str, str]]:
        return self._bhtom_coordinator._current_bhtom_cache_identity()

    def _bhtom_should_fetch_from_network_now(self) -> bool:
        return self._bhtom_coordinator._bhtom_should_fetch_from_network_now()

    def _bhtom_token_hash(self, token: str) -> str:
        return self._bhtom_coordinator._bhtom_token_hash(token)

    def _bhtom_storage_cache_key(self, *, token: str, base_url: str) -> str:
        return self._bhtom_coordinator._bhtom_storage_cache_key(token=token, base_url=base_url)

    def _clone_bhtom_observatory_presets(self, presets: list[dict[str, object]]) -> list[dict[str, object]]:
        return self._bhtom_coordinator._clone_bhtom_observatory_presets(presets)

    def _serialize_bhtom_observatory_presets(self, presets: list[dict[str, object]]) -> list[dict[str, object]]:
        return self._bhtom_coordinator._serialize_bhtom_observatory_presets(presets)

    def _deserialize_bhtom_observatory_presets(self, payload: object) -> list[dict[str, object]]:
        return self._bhtom_coordinator._deserialize_bhtom_observatory_presets(payload)

    def _serialize_bhtom_candidates(self, candidates: list[dict[str, object]]) -> list[dict[str, object]]:
        return self._bhtom_coordinator._serialize_bhtom_candidates(candidates)

    def _deserialize_bhtom_candidates(self, payload: object) -> list[dict[str, object]]:
        return self._bhtom_coordinator._deserialize_bhtom_candidates(payload)

    def _load_bhtom_observatory_disk_cache(
        self,
        *,
        token: str,
        base_url: str,
    ) -> Optional[list[dict[str, object]]]:
        return self._bhtom_coordinator._load_bhtom_observatory_disk_cache(token=token, base_url=base_url)

    def _save_bhtom_observatory_disk_cache(
        self,
        presets: list[dict[str, object]],
        *,
        token: str,
        base_url: str,
    ) -> None:
        return self._bhtom_coordinator._save_bhtom_observatory_disk_cache(presets, token=token, base_url=base_url)

    def _fetch_bhtom_target_page(self, page: int, token: Optional[str] = None) -> object:
        return self._bhtom_coordinator._fetch_bhtom_target_page(page, token)

    def _fetch_bhtom_observatory_page(self, page: int, token: Optional[str] = None) -> object:
        return self._bhtom_coordinator._fetch_bhtom_observatory_page(page, token)

    def _bhtom_observatory_prefetch_status(self) -> tuple[bool, str]:
        return self._bhtom_coordinator._bhtom_observatory_prefetch_status()

    def _prefetch_bhtom_observatory_presets_on_startup(self) -> None:
        return self._bhtom_coordinator._prefetch_bhtom_observatory_presets_on_startup()

    def _ensure_bhtom_observatory_prefetch(self, *, force_refresh: bool = False) -> bool:
        return self._bhtom_coordinator._ensure_bhtom_observatory_prefetch(force_refresh=force_refresh)

    @Slot(int, int, str)
    def _on_bhtom_observatory_prefetch_progress(self, page: int, _total_pages: int, message: str) -> None:
        return self._bhtom_coordinator._on_bhtom_observatory_prefetch_progress(page, _total_pages, message)

    @Slot(int, list, str)
    def _on_bhtom_observatory_prefetch_completed(self, request_id: int, presets: list, err: str) -> None:
        return self._bhtom_coordinator._on_bhtom_observatory_prefetch_completed(request_id, presets, err)

    def _on_bhtom_observatory_prefetch_finished(self, worker: BhtomObservatoryPresetWorker) -> None:
        return self._bhtom_coordinator._on_bhtom_observatory_prefetch_finished(worker)

    @staticmethod
    def _extract_bhtom_items(payload: object) -> list[dict[str, Any]]:
        return BhtomCoordinator._extract_bhtom_items(payload)

    @staticmethod
    def _bhtom_payload_has_more(payload: object, page: int, item_count: int) -> bool:
        return BhtomCoordinator._bhtom_payload_has_more(payload, page, item_count)

    @staticmethod
    def _pick_first_present(sources: list[dict[str, Any]], *keys: str) -> object:
        return BhtomCoordinator._pick_first_present(sources, *keys)

    def _build_bhtom_candidate(self, item: dict[str, Any]) -> Optional[dict[str, object]]:
        return self._bhtom_coordinator._build_bhtom_candidate(item)

    @staticmethod
    def _extract_bhtom_observatory_items(payload: object) -> list[dict[str, Any]]:
        return BhtomCoordinator._extract_bhtom_observatory_items(payload)

    @staticmethod
    def _bhtom_observatory_payload_has_more(payload: object, page: int, item_count: int) -> bool:
        return BhtomCoordinator._bhtom_observatory_payload_has_more(payload, page, item_count)

    def _cached_bhtom_observatory_presets(
        self,
        *,
        token: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> Optional[list[dict[str, object]]]:
        return self._bhtom_coordinator._cached_bhtom_observatory_presets(token=token, base_url=base_url)

    def _fetch_bhtom_observatory_presets(self, *, force_refresh: bool = False) -> list[dict[str, object]]:
        return self._bhtom_coordinator._fetch_bhtom_observatory_presets(force_refresh=force_refresh)

    def _cached_bhtom_candidates(
        self,
        *,
        token: Optional[str] = None,
        base_url: Optional[str] = None,
        force_refresh: bool = False,
    ) -> Optional[list[dict[str, object]]]:
        return self._bhtom_coordinator._cached_bhtom_candidates(token=token, base_url=base_url, force_refresh=force_refresh)

    def _clear_bhtom_candidate_cache(self, *, token: Optional[str] = None, base_url: Optional[str] = None) -> None:
        return self._bhtom_coordinator._clear_bhtom_candidate_cache(token=token, base_url=base_url)

    def _fetch_bhtom_target_candidates(self, *, force_refresh: bool = False) -> list[dict[str, object]]:
        return self._bhtom_coordinator._fetch_bhtom_target_candidates(force_refresh=force_refresh)

    def _bhtom_type_for_target(self, target: Target) -> str:
        return self._bhtom_coordinator._bhtom_type_for_target(target)

    def _bhtom_importance_for_target(self, target: Target) -> Optional[float]:
        return self._bhtom_coordinator._bhtom_importance_for_target(target)

    def _ensure_known_target_type(self, target: Target) -> str:
        return self._bhtom_coordinator._ensure_known_target_type(target)

    def _reload_local_target_suggestions(self, *, force_refresh: bool = False) -> tuple[list[dict[str, object]], list[str]]:
        return self._bhtom_coordinator._reload_local_target_suggestions(force_refresh=force_refresh)

    def _build_bhtom_suggestion_context(self) -> tuple[Optional[dict[str, object]], Optional[str]]:
        return self._bhtom_coordinator._build_bhtom_suggestion_context()

    def _start_bhtom_worker(self, *, mode: str, emit_partials: bool, force_refresh: bool = False) -> bool:
        return self._bhtom_coordinator._start_bhtom_worker(mode=mode, emit_partials=emit_partials, force_refresh=force_refresh)

    def _cancel_bhtom_worker(self) -> None:
        return self._bhtom_coordinator._cancel_bhtom_worker()

    @Slot(int, list, list, int, int)
    def _on_bhtom_worker_page_ready(
        self,
        request_id: int,
        suggestions: list[dict[str, object]],
        notes: list[str],
        page: int,
        loaded_candidates: int,
    ) -> None:
        return self._bhtom_coordinator._on_bhtom_worker_page_ready(request_id, suggestions, notes, page, loaded_candidates)

    def _on_bhtom_worker_finished(self, worker: BhtomSuggestionWorker) -> None:
        return self._bhtom_coordinator._on_bhtom_worker_finished(worker)

    @Slot(int, list, list, list, str)
    def _on_bhtom_worker_completed(
        self,
        request_id: int,
        suggestions: list[dict[str, object]],
        notes: list[str],
        raw_candidates: list[dict[str, object]],
        error: str,
    ) -> None:
        return self._bhtom_coordinator._on_bhtom_worker_completed(request_id, suggestions, notes, raw_candidates, error)

    @Slot(int)
    def _on_suggest_dialog_closed(self, _result: int) -> None:
        return self._bhtom_coordinator._on_suggest_dialog_closed(_result)

    def _build_local_target_suggestions(self, *, force_refresh: bool = False) -> tuple[list[dict[str, object]], list[str]]:
        return self._bhtom_coordinator._build_local_target_suggestions(force_refresh=force_refresh)

    def _format_local_target_suggestions(self, suggestions: list[dict[str, object]], notes: list[str]) -> str:
        return self._bhtom_coordinator._format_local_target_suggestions(suggestions, notes)

    @Slot()
    def _quick_add_suggested_targets(self) -> None:
        return self._bhtom_coordinator._quick_add_suggested_targets()

    def _apply_quick_targets_from_suggestions(
        self,
        suggestions: list[dict[str, object]],
        notes: list[str],
    ) -> None:
        return self._bhtom_coordinator._apply_quick_targets_from_suggestions(suggestions, notes)

    def _resolve_ai_intent(self, question: str) -> AIIntent:
        return self._ai_context_coordinator._resolve_ai_intent(
            question,
        )

    def _execute_ai_intent(self, intent: AIIntent) -> None:
        return self._ai_context_coordinator._execute_ai_intent(
            intent,
        )

    @Slot()
    def _send_ai_query(self) -> None:
        return self._ai_context_coordinator._send_ai_query()

    def _build_selected_target_llm_prompt(self, target: Target, question: str) -> str:
        return self._ai_context_coordinator._build_selected_target_llm_prompt(
            target,
            question,
        )

    def _build_fast_general_llm_prompt(self, question: str) -> str:
        return self._ai_context_coordinator._build_fast_general_llm_prompt(
            question,
        )

    def _build_recent_chat_memory_block(self, *, max_messages: int = 4) -> str:
        return self._ai_context_coordinator._build_recent_chat_memory_block(
            max_messages=max_messages,
        )

    @Slot()
    def _send_ai_selected_target_query(self) -> None:
        return self._ai_context_coordinator._send_ai_selected_target_query()

    @Slot()
    def _ai_describe_target(self) -> None:
        return self._ai_context_coordinator._ai_describe_target()

    @Slot()
    def _ai_suggest_targets(self) -> None:
        return self._bhtom_coordinator._ai_suggest_targets()

    def _dispatch_llm(
        self,
        prompt: str,
        tag: str,
        label: str,
        *,
        requested_class: str = "",
        primary_target: Optional[Target] = None,
    ) -> None:
        self._ai_panel_coordinator.dispatch_llm(
            prompt,
            tag,
            label,
            requested_class=requested_class,
            primary_target=primary_target,
        )

    @Slot(str, str)
    def _on_ai_chunk(self, tag: str, text: str) -> None:
        self._ai_panel_coordinator.on_chunk(tag, text)

    @Slot(str, str)
    def _on_ai_response(self, tag: str, text: str) -> None:
        self._ai_panel_coordinator.on_response(tag, text)

    @Slot(str)
    def _on_ai_error(self, message: str) -> None:
        self._ai_panel_coordinator.on_error(message)

    @Slot()
    def _on_ai_worker_finished(self) -> None:
        self._ai_panel_coordinator.on_worker_finished()

    def _clear_ai_messages(self) -> None:
        self._ai_stream_render_timer.stop()
        self._ai_messages.clear()
        self._ai_message_widget_refs.clear()
        self._ai_stream_message_index = None
        if hasattr(self, "ai_output_layout"):
            self._render_ai_messages()
        self._refresh_ai_panel_action_buttons()
        self._persist_ai_messages_to_storage(allow_empty_clear=True)

    def _apply_ai_chat_font(self) -> None:
        if not hasattr(self, "ai_output"):
            return
        font = self.ai_output.font()
        font.setPointSize(int(getattr(self, "_llm_chat_font_size_pt", LLMConfig.DEFAULT_CHAT_FONT_PT)))
        self.ai_output.setFont(font)
        if hasattr(self, "ai_output_content"):
            self.ai_output_content.setFont(font)
        if getattr(self, "_ai_messages", None):
            self._render_ai_messages()

    def _last_ai_response_text(self) -> str:
        for message in reversed(self._ai_messages):
            if str(message.get("kind", "")) != "ai":
                continue
            text = str(message.get("text", "")).strip()
            if text:
                return text
        return ""

    def _build_ai_chat_transcript(self) -> str:
        role_map = {
            "user": "You",
            "ai": "LLM",
            "error": "Error",
            "info": "Info",
        }
        sections: list[str] = []
        for message in self._ai_messages:
            text = str(message.get("text", "")).strip()
            if not text:
                continue
            kind = str(message.get("kind", "info"))
            role = role_map.get(kind, kind.title() or "Info")
            sections.append(f"{role}\n{text}")
        return "\n\n".join(sections).strip()

    def _refresh_ai_panel_action_buttons(self) -> None:
        has_messages = any(str(message.get("text", "")).strip() for message in self._ai_messages)
        has_last_ai = bool(self._last_ai_response_text())
        if hasattr(self, "ai_copy_last_btn"):
            self.ai_copy_last_btn.setEnabled(has_last_ai)
        if hasattr(self, "ai_export_chat_btn"):
            self.ai_export_chat_btn.setEnabled(has_messages)

    def _visible_action_targets_for_message(self, message: dict[str, Any]) -> list[Target]:
        targets = message.get("action_targets")
        if not isinstance(targets, list):
            return []
        visible: list[Target] = []
        for target in targets:
            if not isinstance(target, Target):
                continue
            if self._plan_contains_target(target):
                continue
            visible.append(target)
        return visible

    def _add_ai_suggested_target(self, target: Target) -> None:
        added = self._append_target_to_plan(target, refresh=True, notify_duplicate=False)
        if added:
            self._set_ai_status(f"Added {target.name} to the plan.", tone="info")
        else:
            self._set_ai_status(f"{target.name} is already in the plan.", tone="warning")
        self._render_ai_messages()

    @Slot()
    def _copy_last_ai_response(self) -> None:
        text = self._last_ai_response_text()
        if not text:
            self._set_ai_status("No AI response to copy.", tone="warning")
            return
        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(text)
        self._set_ai_status("Copied last AI response.", tone="info")

    @Slot()
    def _export_ai_chat(self) -> None:
        transcript = self._build_ai_chat_transcript()
        if not transcript:
            self._set_ai_status("No AI chat to export.", tone="warning")
            return
        default_name = f"astroplanner-ai-chat-{datetime.now().strftime('%Y%m%d-%H%M%S')}.txt"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export AI Chat",
            str(Path.home() / default_name),
            "Text files (*.txt);;Markdown files (*.md);;All files (*)",
        )
        if not file_path:
            return
        try:
            Path(file_path).write_text(transcript + "\n", encoding="utf-8")
        except Exception as exc:
            QMessageBox.warning(self, "Export Failed", f"Could not export AI chat:\n{exc}")
            self._set_ai_status("AI chat export failed.", tone="warning")
            return
        self._set_ai_status(f"Exported AI chat to {Path(file_path).name}.", tone="info")

    @Slot()
    def _reset_ai_chat_appearance(self) -> None:
        self.settings.setValue("llm/chatFontSizePt", LLMConfig.DEFAULT_CHAT_FONT_PT)
        self.settings.setValue("llm/chatSpacing", LLMConfig.DEFAULT_CHAT_SPACING)
        self.settings.setValue("llm/chatTintStrength", LLMConfig.DEFAULT_CHAT_TINT_STRENGTH)
        self.settings.setValue("llm/chatMessageWidth", LLMConfig.DEFAULT_CHAT_WIDTH)
        self.settings.setValue("llm/statusErrorClearSec", LLMConfig.DEFAULT_STATUS_ERROR_CLEAR_S)
        self._llm_chat_font_size_pt = LLMConfig.DEFAULT_CHAT_FONT_PT
        self._ai_chat_spacing = LLMConfig.DEFAULT_CHAT_SPACING
        self._ai_chat_tint_strength = LLMConfig.DEFAULT_CHAT_TINT_STRENGTH
        self._ai_chat_width = LLMConfig.DEFAULT_CHAT_WIDTH
        self._ai_status_error_clear_s = LLMConfig.DEFAULT_STATUS_ERROR_CLEAR_S
        self._apply_ai_chat_font()
        self._render_ai_messages()
        self._set_ai_status("AI appearance reset to defaults.", tone="info")

    @staticmethod
    def _mix_qcolors(first: QColor, second: QColor, first_ratio: float) -> QColor:
        ratio = max(0.0, min(1.0, float(first_ratio)))
        other_ratio = 1.0 - ratio
        return QColor(
            int(round(first.red() * ratio + second.red() * other_ratio)),
            int(round(first.green() * ratio + second.green() * other_ratio)),
            int(round(first.blue() * ratio + second.blue() * other_ratio)),
        )

    @staticmethod
    def _qcolor_css(color: QColor) -> str:
        return color.name(QColor.HexRgb)

    @staticmethod
    def _qcolor_rgba_css(color: QColor, alpha: float) -> str:
        use_color = QColor(color)
        if not use_color.isValid():
            use_color = QColor("#59f3ff")
        return f"rgba({use_color.red()}, {use_color.green()}, {use_color.blue()}, {max(0.0, min(1.0, float(alpha))):.3f})"

    @staticmethod
    def _qcolor_rgba_mpl(color: QColor, alpha: float) -> tuple[float, float, float, float]:
        use_color = QColor(color)
        if not use_color.isValid():
            use_color = QColor("#59f3ff")
        return (
            float(use_color.redF()),
            float(use_color.greenF()),
            float(use_color.blueF()),
            max(0.0, min(1.0, float(alpha))),
        )

    @staticmethod
    def _format_ai_message_inline_markdown(text: str) -> str:
        rendered = str(text)
        rendered = re.sub(r"`([^`\n]+)`", r"<code>\1</code>", rendered)
        rendered = re.sub(r"\*\*([^*\n][\s\S]*?)\*\*", r"<strong>\1</strong>", rendered)
        rendered = re.sub(r"(?<!\*)\*([^*\n][\s\S]*?)\*(?!\*)", r"<em>\1</em>", rendered)
        return rendered

    def _render_ai_message_body(self, raw_text: str, *, muted_text: QColor) -> str:
        text = str(raw_text or "")
        if not text.strip():
            return f'<span style="color:{self._qcolor_css(muted_text)}"><i>...</i></span>'

        blocks: list[str] = []
        paragraph_lines: list[str] = []
        list_kind: Optional[str] = None
        list_items: list[str] = []

        def flush_paragraph() -> None:
            nonlocal paragraph_lines
            if not paragraph_lines:
                return
            joined = "<br>".join(
                self._format_ai_message_inline_markdown(html_module.escape(line))
                for line in paragraph_lines
            )
            blocks.append(f"<div>{joined}</div>")
            paragraph_lines = []

        def flush_list() -> None:
            nonlocal list_kind, list_items
            if not list_kind or not list_items:
                list_kind = None
                list_items = []
                return
            if list_kind == "ol":
                rows_html = "".join(
                    (
                        '<tr>'
                        f'<td valign="top" style="padding:0 8px 4px 0;white-space:nowrap;font-weight:600;">{idx}.</td>'
                        f'<td valign="top" style="padding:0 0 4px 0;">'
                        f'{self._format_ai_message_inline_markdown(html_module.escape(item))}'
                        "</td>"
                        "</tr>"
                    )
                    for idx, item in enumerate(list_items, start=1)
                )
                blocks.append(
                    '<table style="margin:4px 0 4px 0;border:none;border-collapse:collapse;">'
                    f"{rows_html}"
                    "</table>"
                )
            else:
                items_html = "".join(
                    f"<li>{self._format_ai_message_inline_markdown(html_module.escape(item))}</li>"
                    for item in list_items
                )
                blocks.append(
                    f'<ul style="margin:4px 0 4px 18px;padding:0;">{items_html}</ul>'
                )
            list_kind = None
            list_items = []

        for raw_line in text.splitlines():
            line = raw_line.rstrip()
            stripped = line.strip()
            if not stripped:
                flush_paragraph()
                flush_list()
                continue

            bullet_match = re.match(r"^\s*[-*]\s+(.+)$", line)
            ordered_match = re.match(r"^\s*\d+[.)]\s+(.+)$", line)
            if bullet_match:
                flush_paragraph()
                if list_kind not in (None, "ul"):
                    flush_list()
                list_kind = "ul"
                list_items.append(bullet_match.group(1).strip())
                continue
            if ordered_match:
                flush_paragraph()
                if list_kind not in (None, "ol"):
                    flush_list()
                list_kind = "ol"
                list_items.append(ordered_match.group(1).strip())
                continue

            flush_list()
            paragraph_lines.append(stripped)

        flush_paragraph()
        flush_list()
        return "".join(blocks)

    def _perform_ai_scroll_to_bottom(self) -> None:
        if not hasattr(self, "ai_output"):
            return
        if hasattr(self, "ai_output_content"):
            self.ai_output_content.adjustSize()
            layout = self.ai_output_content.layout()
            if layout is not None:
                layout.activate()
        scroll = self.ai_output.verticalScrollBar()
        if scroll is not None:
            scroll.setValue(scroll.maximum())

    def _scroll_ai_output_to_bottom(self, *, deferred: bool = True) -> None:
        if not hasattr(self, "ai_output"):
            return
        self._perform_ai_scroll_to_bottom()
        if not deferred:
            return
        if self._ai_scroll_bottom_pending:
            return
        self._ai_scroll_bottom_pending = True

        def _deferred_scroll() -> None:
            self._ai_scroll_bottom_pending = False
            self._perform_ai_scroll_to_bottom()
            QTimer.singleShot(0, self._perform_ai_scroll_to_bottom)

        QTimer.singleShot(0, _deferred_scroll)

    def _apply_ai_message_body_widget(
        self,
        body: QLabel,
        bubble: QFrame,
        *,
        body_html: str,
        font_pt: int,
        line_height: float,
        text_color: QColor,
        bubble_pad_h: int,
        bubble_max_width: int,
        stream_fixed_width: bool,
    ) -> None:
        body_rich_text = (
            f'<div style="color:{self._qcolor_css(text_color)};'
            f'font-size:{font_pt}pt;line-height:{line_height};">'
            f"{body_html}</div>"
        )
        body.setText(body_rich_text)
        max_body_width = max(150, bubble_max_width - (2 * bubble_pad_h))
        if stream_fixed_width:
            body_width = max_body_width
        else:
            doc = QTextDocument()
            doc.setDefaultFont(body.font())
            doc.setDocumentMargin(0)
            doc.setHtml(body_rich_text)
            body_width = max(150, min(max_body_width, int(math.ceil(doc.idealWidth())) + 2))
        body.setFixedWidth(body_width)
        bubble.setFixedWidth(body_width + (2 * bubble_pad_h))

    def _build_ai_message_widget(
        self,
        *,
        kind: str,
        body_html: str,
        show_headers: bool,
        font_pt: int,
        header_pt: int,
        outer_margin: int,
        header_margin: int,
        bubble_pad_h: int,
        bubble_pad_v: int,
        line_height: float,
        message_max_width_ratio: float,
        text_color: QColor,
        muted_text: QColor,
        label_color: QColor,
        border_color: QColor,
        background_color: QColor,
        header_icon_html: str,
        header_text: str,
        align_right: bool,
        streaming: bool = False,
        action_targets: Optional[list[Target]] = None,
    ) -> tuple[QWidget, dict[str, Any]]:
        viewport_width = self._ai_message_layout_width()
        bubble_max_width = max(240, int(viewport_width * message_max_width_ratio))

        row = QWidget(self.ai_output_content)
        row_l = QHBoxLayout(row)
        row_l.setContentsMargins(0, outer_margin, 0, outer_margin)
        row_l.setSpacing(0)

        stack = QWidget(row)
        stack_l = QVBoxLayout(stack)
        stack_l.setContentsMargins(0, 0, 0, 0)
        stack_l.setSpacing(header_margin)

        if show_headers:
            header = QLabel(stack)
            header.setTextFormat(Qt.RichText)
            header.setText(
                f'<span style="color:{self._qcolor_css(label_color)};'
                f'font-size:{header_pt}pt;font-weight:700;letter-spacing:0.8px;text-transform:uppercase;">'
                f'{header_icon_html}{html_module.escape(header_text)}</span>'
            )
            header.setTextInteractionFlags(Qt.NoTextInteraction)
            header.setStyleSheet("background:transparent;border:none;padding:0;margin:0;")
            stack_l.addWidget(header, 0, Qt.AlignRight if align_right else Qt.AlignLeft)

        bubble = QFrame(stack)
        bubble.setObjectName("AIChatBubble")
        bubble.setMaximumWidth(bubble_max_width)
        bubble.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        top_left = 14 if not align_right else 14
        top_right = 14 if align_right else 14
        bottom_left = 5 if not align_right else 14
        bottom_right = 5 if align_right else 14
        bubble.setStyleSheet(
            "QFrame#AIChatBubble {"
            f"background:{self._qcolor_rgba_css(background_color, background_color.alphaF())};"
            f"border:1px solid {self._qcolor_css(border_color)};"
            f"border-top-left-radius:{top_left}px;"
            f"border-top-right-radius:{top_right}px;"
            f"border-bottom-left-radius:{bottom_left}px;"
            f"border-bottom-right-radius:{bottom_right}px;"
            "}"
        )
        shadow = QGraphicsDropShadowEffect(bubble)
        shadow.setBlurRadius(18)
        shadow.setOffset(0, 6)
        shadow.setColor(QColor(border_color.red(), border_color.green(), border_color.blue(), 42))
        bubble.setGraphicsEffect(shadow)

        bubble_l = QVBoxLayout(bubble)
        bubble_l.setContentsMargins(bubble_pad_h, bubble_pad_v, bubble_pad_h, bubble_pad_v)
        bubble_l.setSpacing(0)

        body = QLabel(bubble)
        body.setWordWrap(True)
        body.setTextFormat(Qt.RichText)
        body.setTextInteractionFlags(Qt.TextSelectableByMouse)
        body.setStyleSheet(
            "background:transparent;border:none;padding:0;margin:0;"
            f"color:{self._qcolor_css(text_color)};"
        )
        self._apply_ai_message_body_widget(
            body,
            bubble,
            body_html=body_html,
            font_pt=font_pt,
            line_height=line_height,
            text_color=text_color,
            bubble_pad_h=bubble_pad_h,
            bubble_max_width=bubble_max_width,
            stream_fixed_width=streaming,
        )
        bubble_l.addWidget(body)

        if action_targets:
            action_box = QWidget(bubble)
            action_box_l = QVBoxLayout(action_box)
            action_box_l.setContentsMargins(0, 8, 0, 0)
            action_box_l.setSpacing(6)
            for target in action_targets:
                add_btn = QPushButton(f"Add {target.name}", action_box)
                _set_button_variant(add_btn, "ghost")
                _set_button_icon_kind(add_btn, "add")
                add_btn.clicked.connect(lambda _checked=False, t=target: self._add_ai_suggested_target(t))
                action_box_l.addWidget(add_btn, 0, Qt.AlignLeft)
            bubble_l.addWidget(action_box, 0, Qt.AlignLeft)

        stack_l.addWidget(bubble, 0, Qt.AlignRight if align_right else Qt.AlignLeft)

        if align_right:
            row_l.addStretch(1)
            row_l.addWidget(stack, 0, Qt.AlignRight)
        else:
            row_l.addWidget(stack, 0, Qt.AlignLeft)
            row_l.addStretch(1)
        return row, {
            "row": row,
            "bubble": bubble,
            "body": body,
            "font_pt": font_pt,
            "line_height": line_height,
            "text_color": QColor(text_color),
            "muted_text": QColor(muted_text),
            "bubble_pad_h": bubble_pad_h,
            "bubble_max_width": bubble_max_width,
            "streaming": streaming,
        }

    def _ai_message_layout_width(self) -> int:
        if not hasattr(self, "ai_output"):
            return 320
        viewport_width = 0
        try:
            viewport_width = int(self.ai_output.viewport().width())
        except Exception:
            viewport_width = int(self.ai_output.width())
        reserve = 0
        scroll = self.ai_output.verticalScrollBar() if hasattr(self.ai_output, "verticalScrollBar") else None
        if scroll is not None and not scroll.isVisible():
            try:
                reserve = max(0, int(scroll.sizeHint().width()))
            except Exception:
                reserve = 14
        return max(320, viewport_width - reserve)

    def _update_ai_message_widget(self, idx: int, *, streaming: bool) -> bool:
        if not (0 <= idx < len(self._ai_messages)) or not (0 <= idx < len(self._ai_message_widget_refs)):
            return False
        refs = self._ai_message_widget_refs[idx]
        body = refs.get("body")
        bubble = refs.get("bubble")
        if not isinstance(body, QLabel) or not isinstance(bubble, QFrame):
            return False
        message = self._ai_messages[idx]
        body_html = self._render_ai_message_body(str(message.get("text", "")), muted_text=QColor(refs["muted_text"]))
        self._apply_ai_message_body_widget(
            body,
            bubble,
            body_html=body_html,
            font_pt=int(refs["font_pt"]),
            line_height=float(refs["line_height"]),
            text_color=QColor(refs["text_color"]),
            bubble_pad_h=int(refs["bubble_pad_h"]),
            bubble_max_width=int(refs["bubble_max_width"]),
            stream_fixed_width=streaming,
        )
        refs["streaming"] = streaming
        body.updateGeometry()
        bubble.updateGeometry()
        row = refs.get("row")
        if isinstance(row, QWidget):
            row.updateGeometry()
        self.ai_output_content.updateGeometry()
        return True

    def _render_ai_messages(self) -> None:
        if not hasattr(self, "ai_output"):
            return
        font_pt = max(9, int(getattr(self, "_llm_chat_font_size_pt", LLMConfig.DEFAULT_CHAT_FONT_PT)))
        header_pt = max(8, font_pt - 2)
        spacing_mode = str(getattr(self, "_ai_chat_spacing", LLMConfig.DEFAULT_CHAT_SPACING) or LLMConfig.DEFAULT_CHAT_SPACING).strip().lower()
        tint_mode = str(getattr(self, "_ai_chat_tint_strength", LLMConfig.DEFAULT_CHAT_TINT_STRENGTH) or LLMConfig.DEFAULT_CHAT_TINT_STRENGTH).strip().lower()
        width_mode = str(getattr(self, "_ai_chat_width", LLMConfig.DEFAULT_CHAT_WIDTH) or LLMConfig.DEFAULT_CHAT_WIDTH).strip().lower()
        show_headers = True
        if spacing_mode == "compact":
            outer_margin = 4
            header_margin = 2
            bubble_pad_v = 4
            bubble_pad_h = 7
            line_height = 1.28
        else:
            outer_margin = 6
            header_margin = 3
            bubble_pad_v = 6
            bubble_pad_h = 9
            line_height = 1.35
        tint_scale_map = {
            "low": 0.75,
            "medium": 1.10,
            "high": 1.45,
        }
        tint_scale = float(tint_scale_map.get(tint_mode, 1.00))
        max_width_map = {
            "narrow": 0.34,
            "normal": 0.48,
            "wide": 0.62,
        }
        message_max_width_ratio = float(max_width_map.get(width_mode, 0.76))
        palette = self.ai_output.palette()
        base = palette.color(QPalette.Base)
        text_color = palette.color(QPalette.Text)
        highlight = palette.color(QPalette.Highlight)
        accent_red = self._theme_qcolor("state_error", "#dc2626")
        accent_primary = self._theme_qcolor("accent_primary", "#59f3ff")
        accent_secondary = self._theme_qcolor("accent_secondary", "#ff4fd8")
        accent_secondary_soft = self._theme_qcolor("accent_secondary_soft", "#d38cff")

        user_border = self._mix_qcolors(accent_primary, highlight, 0.60)
        ai_border = self._mix_qcolors(accent_secondary, accent_secondary_soft, 0.58)
        error_border = self._mix_qcolors(accent_red, base, 0.38)
        muted_text = self._mix_qcolors(text_color, base, 0.42)
        user_bg = self._mix_qcolors(accent_primary, base, 0.14)
        ai_bg = self._mix_qcolors(accent_secondary_soft, base, 0.10)
        error_bg = self._mix_qcolors(accent_red, base, 0.08)
        user_label = self._mix_qcolors(accent_primary, text_color, 0.82)
        ai_label = self._mix_qcolors(accent_secondary, text_color, 0.76)
        error_label = self._mix_qcolors(accent_red, text_color, 0.80)
        self.ai_output.setUpdatesEnabled(False)
        try:
            while self.ai_output_layout.count():
                item = self.ai_output_layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
            self._ai_message_widget_refs = []

            if not self._ai_messages:
                placeholder = QLabel(self._ai_output_placeholder_text, self.ai_output_content)
                placeholder.setObjectName("SectionHint")
                placeholder.setWordWrap(True)
                placeholder.setTextInteractionFlags(Qt.NoTextInteraction)
                self.ai_output_layout.addWidget(placeholder, 0, Qt.AlignTop)
                self.ai_output_layout.addStretch(1)
                self._refresh_ai_panel_action_buttons()
                return

            for idx, message in enumerate(self._ai_messages):
                kind = message.get("kind", "ai")
                raw_text = str(message.get("text", ""))
                body_html = self._render_ai_message_body(raw_text, muted_text=muted_text)
                is_streaming_message = bool(
                    kind == "ai"
                    and self._ai_stream_message_index is not None
                    and idx == self._ai_stream_message_index
                    and self._llm_active_tag not in {"", "warmup"}
                )
                if kind == "user":
                    widget, refs = self._build_ai_message_widget(
                        kind="user",
                        body_html=body_html,
                        show_headers=show_headers,
                        font_pt=font_pt,
                        header_pt=header_pt,
                        outer_margin=outer_margin,
                        header_margin=header_margin,
                        bubble_pad_h=bubble_pad_h,
                        bubble_pad_v=bubble_pad_v,
                        line_height=line_height,
                        message_max_width_ratio=message_max_width_ratio,
                        text_color=text_color,
                        muted_text=muted_text,
                        label_color=user_label,
                        border_color=user_border,
                        background_color=QColor.fromRgbF(user_bg.redF(), user_bg.greenF(), user_bg.blueF(), min(0.98, 0.96 * tint_scale)),
                        header_icon_html=(
                            f'<span style="display:inline-block;min-width:16px;height:16px;line-height:16px;'
                            f'padding:0 5px;margin:0 5px 0 0;text-align:center;border-radius:999px;'
                            f'background:{self._qcolor_rgba_css(user_label, 0.16)};'
                            f'border:1px solid {self._qcolor_css(user_border)};'
                            f'color:{self._qcolor_css(user_label)}">&#8250;</span>'
                        ),
                        header_text="You",
                        align_right=True,
                        streaming=False,
                    )
                elif kind == "error":
                    widget, refs = self._build_ai_message_widget(
                        kind="error",
                        body_html=body_html,
                        show_headers=show_headers,
                        font_pt=font_pt,
                        header_pt=header_pt,
                        outer_margin=outer_margin,
                        header_margin=header_margin,
                        bubble_pad_h=bubble_pad_h,
                        bubble_pad_v=bubble_pad_v,
                        line_height=line_height,
                        message_max_width_ratio=message_max_width_ratio,
                        text_color=text_color,
                        muted_text=muted_text,
                        label_color=error_label,
                        border_color=error_border,
                        background_color=QColor.fromRgbF(error_bg.redF(), error_bg.greenF(), error_bg.blueF(), min(0.98, 1.00 * tint_scale)),
                        header_icon_html=(
                            f'<span style="display:inline-block;min-width:16px;height:16px;line-height:16px;'
                            f'padding:0 5px;margin:0 5px 0 0;text-align:center;border-radius:999px;'
                            f'background:{self._qcolor_rgba_css(error_label, 0.16)};'
                            f'border:1px solid {self._qcolor_css(error_border)};'
                            f'color:{self._qcolor_css(error_label)}">!</span>'
                        ),
                        header_text="Error",
                        align_right=False,
                        streaming=False,
                    )
                else:
                    action_targets = [] if is_streaming_message else self._visible_action_targets_for_message(message)
                    widget, refs = self._build_ai_message_widget(
                        kind="ai",
                        body_html=body_html,
                        show_headers=show_headers,
                        font_pt=font_pt,
                        header_pt=header_pt,
                        outer_margin=outer_margin,
                        header_margin=header_margin,
                        bubble_pad_h=bubble_pad_h,
                        bubble_pad_v=bubble_pad_v,
                        line_height=line_height,
                        message_max_width_ratio=message_max_width_ratio,
                        text_color=text_color,
                        muted_text=muted_text,
                        label_color=ai_label,
                        border_color=ai_border,
                        background_color=QColor.fromRgbF(ai_bg.redF(), ai_bg.greenF(), ai_bg.blueF(), min(0.98, 0.98 * tint_scale)),
                        header_icon_html=(
                            f'<span style="display:inline-block;min-width:16px;height:16px;line-height:16px;'
                            f'padding:0 5px;margin:0 5px 0 0;text-align:center;border-radius:999px;'
                            f'background:{self._qcolor_rgba_css(ai_label, 0.16)};'
                            f'border:1px solid {self._qcolor_css(ai_border)};'
                            f'color:{self._qcolor_css(ai_label)}">&#10022;</span>'
                        ),
                        header_text="LLM",
                        align_right=False,
                        streaming=is_streaming_message,
                        action_targets=action_targets,
                    )
                self.ai_output_layout.addWidget(widget, 0, Qt.AlignTop)
                self._ai_message_widget_refs.append(refs)
            self.ai_output_layout.addStretch(1)
        finally:
            self.ai_output.setUpdatesEnabled(True)

        self._scroll_ai_output_to_bottom(deferred=True)
        self._refresh_ai_panel_action_buttons()

    def _flush_ai_stream_render(self) -> None:
        idx = self._ai_stream_message_index
        if idx is None:
            return
        if not self._update_ai_message_widget(idx, streaming=True):
            self._render_ai_messages()
            return
        self._scroll_ai_output_to_bottom(deferred=False)
        self._refresh_ai_panel_action_buttons()

    def _start_ai_response_message(self) -> None:
        self._ai_stream_render_timer.stop()
        self._ai_stream_message_index = self._append_ai_message("", is_ai=True)

    def _append_ai_stream_chunk(self, text: str) -> None:
        if not text:
            return
        idx = self._ai_stream_message_index
        if idx is None or not (0 <= idx < len(self._ai_messages)):
            idx = self._append_ai_message("", is_ai=True)
            self._ai_stream_message_index = idx
        self._ai_messages[idx]["text"] = self._ai_messages[idx].get("text", "") + text
        if not self._ai_stream_render_timer.isActive():
            self._ai_stream_render_timer.start(45)

    def _finalize_ai_response(self, text: str) -> None:
        self._ai_stream_render_timer.stop()
        requested_class = str(getattr(self, "_llm_active_requested_class", "") or "").strip()
        primary_target = getattr(self, "_llm_active_primary_target", None)
        idx = self._ai_stream_message_index
        if idx is not None and 0 <= idx < len(self._ai_messages):
            streamed_text = str(self._ai_messages[idx].get("text", "") or "")
            final_text = str(text or "").strip() or streamed_text
            action_targets = self._extract_addable_bhtom_targets_from_ai_text(final_text)
            if not action_targets and streamed_text and streamed_text != final_text:
                action_targets = self._extract_addable_bhtom_targets_from_ai_text(streamed_text)
            self._ai_messages[idx]["kind"] = "ai"
            self._ai_messages[idx]["text"] = final_text
            self._ai_messages[idx]["action_targets"] = action_targets
            if requested_class:
                self._ai_messages[idx]["requested_class"] = requested_class
            if isinstance(primary_target, Target):
                self._ai_messages[idx]["primary_target"] = primary_target
            if action_targets:
                self._ai_messages[idx]["suggested_targets"] = list(action_targets)
            self._ai_stream_message_index = None
            self._render_ai_messages()
        else:
            final_text = str(text or "").strip()
            self._append_ai_message(
                final_text,
                is_ai=True,
                action_targets=self._extract_addable_bhtom_targets_from_ai_text(final_text),
                requested_class=requested_class,
                primary_target=primary_target if isinstance(primary_target, Target) else None,
            )
        self._ai_stream_message_index = None
        self._persist_ai_messages_to_storage()

    def _fail_ai_response(self, message: str) -> None:
        self._ai_stream_render_timer.stop()
        idx = self._ai_stream_message_index
        if idx is not None and 0 <= idx < len(self._ai_messages):
            current_text = self._ai_messages[idx].get("text", "")
            if current_text.strip():
                self._append_ai_message(message, is_error=True)
            else:
                self._ai_messages[idx]["kind"] = "error"
                self._ai_messages[idx]["text"] = message
                self._render_ai_messages()
        else:
            self._append_ai_message(message, is_error=True)
        self._ai_stream_message_index = None
        self._persist_ai_messages_to_storage()

    def _append_ai_message(
        self,
        text: str,
        *,
        is_user: bool = False,
        is_ai: bool = False,
        is_error: bool = False,
        action_targets: Optional[list[Target]] = None,
        requested_class: str = "",
        primary_target: Optional[Target] = None,
        suggested_targets: Optional[list[Target]] = None,
    ) -> int:
        kind = "user" if is_user else "error" if is_error else "ai" if is_ai else "info"
        payload: dict[str, Any] = {
            "kind": kind,
            "text": str(text),
            "created_at": datetime.now(timezone.utc).timestamp(),
        }
        if action_targets:
            payload["action_targets"] = list(action_targets)
        requested_class_value = str(requested_class or "").strip()
        if requested_class_value:
            payload["requested_class"] = requested_class_value
        if isinstance(primary_target, Target):
            payload["primary_target"] = primary_target
        if suggested_targets:
            payload["suggested_targets"] = [target for target in suggested_targets if isinstance(target, Target)]
        self._ai_messages.append(payload)
        self._render_ai_messages()
        if not is_ai or self._llm_active_tag in {"", "warmup"}:
            self._persist_ai_messages_to_storage()
        return len(self._ai_messages) - 1

    @Slot(int)
    def _change_date(self, offset_days: int):
        """Shift the selected date by the given number of days and re-plot."""
        new_date = self.date_edit.date().addDays(offset_days)
        self.date_edit.setDate(new_date)
        self._schedule_plan_autosave()
        self._replot_timer.start()

    @Slot()
    def _change_to_today(self):
        """
        Reset date picker to the current observing night: use previous calendar date before local noon.
        This ensures pressing 'Today' after midnight and before noon still shows the previous night's date.
        """
        # Determine local timezone (site or system)
        if self.table_model.site:
            tz = pytz.timezone(self.table_model.site.timezone_name)
        else:
            tz = datetime.now().astimezone().tzinfo
        now_local = datetime.now(tz)
        # If before noon local time, use yesterday; otherwise use today
        local_qdate = QDate(now_local.year, now_local.month, now_local.day)
        if now_local.hour < 12:
            new_date = local_qdate.addDays(-1)
        else:
            new_date = local_qdate
        self.date_edit.setDate(new_date)
        self._schedule_plan_autosave()
        self._replot_timer.start()

    @Slot()
    def _update_limit(self):
        """Update limit altitude, refresh table warnings, and replot."""
        # Update table model limit so coloring reflects the new threshold
        new_limit = float(self.limit_spin.value())
        self.table_model.limit = new_limit
        self._emit_table_data_changed()
        self._schedule_plan_autosave()
        # Replot visibility with updated limit
        if self.last_payload is not None:
            self._update_plot(self.last_payload)
        else:
            self._replot_timer.start()

# --------------------------------------------------
# --- Main entry -----------------------------------
# --------------------------------------------------
def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Astronomical Observation Planner")
    parser.add_argument('--plan', '-p', help='Path to JSON plan file to load and plot on startup')
    args = parser.parse_args(argv)

    qt_argv = sys.argv if argv is None else [sys.argv[0], *argv]
    app = QApplication(qt_argv)
    win = MainWindow()

    # If a plan file is specified, import it into the workspace and immediately plot.
    if args.plan:
        try:
            win._load_plan_from_json_path(args.plan, persist_workspace=True)
        except Exception as e:
            QMessageBox.critical(None, "Startup Load Error", f"Failed to load plan '{args.plan}': {e}")

    win.show()
    return int(app.exec())


if __name__ == "__main__":
    sys.exit(main())
