from __future__ import annotations

import json
import re
from typing import Optional
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from PySide6.QtCore import QSignalBlocker, Qt, QTimer, QUrl, Slot
from PySide6.QtGui import QColor, QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from astroplanner.ai import (
    AI_CHAT_SPACING_CHOICES,
    AI_CHAT_TINT_CHOICES,
    AI_CHAT_WIDTH_CHOICES,
    LLMConfig,
    LLMModelDiscoveryWorker,
)
from astroplanner.bhtom import BHTOM_SUGGESTION_MIN_IMPORTANCE
from astroplanner.i18n import (
    SUPPORTED_UI_LANGUAGES,
    current_language,
    localize_widget_tree,
)
from astroplanner.seestar import (
    SEESTAR_ALP_DEFAULT_BASE_URL,
    SEESTAR_ALP_DEFAULT_CLIENT_ID,
    SEESTAR_ALP_DEFAULT_DEVICE_NUM,
    SEESTAR_ALP_DEFAULT_TIMEOUT_S,
    SeestarAlpClient,
    SeestarAlpConfig,
    build_alp_web_ui_url,
    render_alp_backend_status_text,
)
from astroplanner.theme import (
    DEFAULT_DARK_MODE,
    DEFAULT_UI_THEME,
    THEME_CHOICES,
    highlight_palette_for_theme,
    normalize_theme_key,
    resolve_theme_tokens,
)
from astroplanner.ui.common import _fit_dialog_to_screen
from astroplanner.ui.theme_utils import (
    _qcolor_from_token,
    _set_button_variant,
    _style_dialog_button_box,
)

TNS_ENDPOINT_CHOICES: list[tuple[str, str]] = [
    ("production", "Production (www.wis-tns.org)"),
    ("sandbox", "Sandbox (sandbox.wis-tns.org)"),
]

TNS_ENDPOINT_BASE_URLS: dict[str, str] = {
    "production": "https://www.wis-tns.org/api",
    "sandbox": "https://sandbox.wis-tns.org/api",
}

CUTOUT_SURVEY_CHOICES: list[tuple[str, str, str]] = [
    ("dss2", "DSS2", "CDS/P/DSS2/color"),
    ("panstarrs", "PanSTARRS", "CDS/P/PanSTARRS/DR1/color-z-zg-g"),
    ("2mass", "2MASS", "CDS/P/2MASS/color"),
]
CUTOUT_VIEW_CHOICES: list[tuple[str, str]] = [
    ("aladin", "Aladin"),
    ("finderchart", "Finder chart"),
]
CUTOUT_DEFAULT_SURVEY_KEY = "dss2"
CUTOUT_DEFAULT_VIEW_KEY = "aladin"
CUTOUT_DEFAULT_FOV_ARCMIN = 15
CUTOUT_DEFAULT_SIZE_PX = 280
CUTOUT_MIN_FOV_ARCMIN = 5
CUTOUT_MAX_FOV_ARCMIN = 120
CUTOUT_MIN_SIZE_PX = 128
CUTOUT_MAX_SIZE_PX = 800
QUICK_TARGETS_DEFAULT_COUNT = 10
QUICK_TARGETS_MIN_COUNT = 1
QUICK_TARGETS_MAX_COUNT = 50

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
_VALID_TABLE_COLOR_MODES = {"background", "text_glow"}
UI_FONT_SIZE_MIN = 9
UI_FONT_SIZE_MAX = 24


def _configure_tab_widget(tab_widget: QTabWidget, *, document_mode: bool = True) -> QTabWidget:
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
            auto_colors.update(
                {
                    _canonical_color_token(palette.below),
                    _canonical_color_token(palette.limit),
                    _canonical_color_token(palette.above),
                }
            )
    colorblind_palette = highlight_palette_for_theme(DEFAULT_UI_THEME, dark_enabled=False, color_blind=True)
    auto_colors.update(
        {
            _canonical_color_token(colorblind_palette.below),
            _canonical_color_token(colorblind_palette.limit),
            _canonical_color_token(colorblind_palette.above),
        }
    )
    if normalized in auto_colors:
        return str(default_color)
    return raw


def _normalized_css_color(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    color = QColor(text)
    if not color.isValid():
        return ""
    return color.name().lower()


def _swatch_text_color(value: object, *, dark: str = "#f6fbff", light: str = "#101720") -> str:
    color = _qcolor_from_token(value)
    if not color.isValid():
        return dark
    return dark if color.lightnessF() < 0.60 else light


def _normalize_table_color_mode(value: object, *, default: str = "background") -> str:
    mode = str(value or "").strip().lower()
    if mode in _VALID_TABLE_COLOR_MODES:
        return mode
    return default


def _sanitize_ui_font_size(value: object, *, default: int = 11) -> int:
    try:
        size = int(value)
    except (TypeError, ValueError):
        size = int(default)
    return max(UI_FONT_SIZE_MIN, min(UI_FONT_SIZE_MAX, size))


def _build_tns_marker(bot_id: int | str, bot_name: str) -> str:
    return f'tns_marker{{"tns_id": "{bot_id}", "type": "bot", "name": "{bot_name}"}}'


def _normalize_tns_endpoint_key(value: object) -> str:
    key = str(value or "").strip().lower()
    if key in TNS_ENDPOINT_BASE_URLS:
        return key
    if "sandbox" in key:
        return "sandbox"
    return "production"


def _tns_api_base_url(value: object) -> str:
    key = _normalize_tns_endpoint_key(value)
    return TNS_ENDPOINT_BASE_URLS[key]


def _normalize_cutout_survey_key(value: object) -> str:
    key = str(value or "").strip().lower()
    if key in {"decals", "cds/p/decals/dr5/color"}:
        return "2mass"
    for survey_key, _label, hips in CUTOUT_SURVEY_CHOICES:
        if key == survey_key or key == hips.lower():
            return survey_key
    return CUTOUT_DEFAULT_SURVEY_KEY


def _normalize_cutout_view_key(value: object) -> str:
    key = str(value or "").strip().lower()
    for view_key, _label in CUTOUT_VIEW_CHOICES:
        if key == view_key:
            return view_key
    return CUTOUT_DEFAULT_VIEW_KEY


def _sanitize_cutout_fov_arcmin(value: object) -> int:
    try:
        ivalue = int(float(value))
    except Exception:
        ivalue = CUTOUT_DEFAULT_FOV_ARCMIN
    return max(CUTOUT_MIN_FOV_ARCMIN, min(CUTOUT_MAX_FOV_ARCMIN, ivalue))


def _sanitize_cutout_size_px(value: object) -> int:
    try:
        ivalue = int(float(value))
    except Exception:
        ivalue = CUTOUT_DEFAULT_SIZE_PX
    return max(CUTOUT_MIN_SIZE_PX, min(CUTOUT_MAX_SIZE_PX, ivalue))


def _observation_columns_for_table_model(table_model: object) -> set[int]:
    return {
        int(getattr(table_model, "COL_ORDER", 0)),
        int(getattr(table_model, "COL_NAME", 1)),
        int(getattr(table_model, "COL_ALT", 5)),
        int(getattr(table_model, "COL_AZ", 6)),
        int(getattr(table_model, "COL_MOON_SEP", 7)),
        int(getattr(table_model, "COL_SCORE", 8)),
        int(getattr(table_model, "COL_HOURS", 9)),
        int(getattr(table_model, "COL_MAG", 10)),
        int(getattr(table_model, "COL_PRIORITY", 11)),
        int(getattr(table_model, "COL_OBSERVED", 12)),
    }


class TableSettingsDialog(QDialog):
    """Dialog to configure table parameters."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName(self.__class__.__name__)
        self.setWindowTitle("Table Settings")
        settings = parent.settings if parent and hasattr(parent, "settings") else None
        table_model = getattr(parent, "table_model", None)

        root_layout = QVBoxLayout(self)
        tabs = _configure_tab_widget(QTabWidget(self))
        root_layout.addWidget(tabs, 1)

        appearance_tab = QWidget(self)
        appearance_layout = QFormLayout(appearance_tab)
        columns_tab = QWidget(self)
        columns_layout = QVBoxLayout(columns_tab)
        colors_tab = QWidget(self)
        colors_layout = QFormLayout(colors_tab)

        tabs.addTab(appearance_tab, "Appearance")
        tabs.addTab(columns_tab, "Columns")
        tabs.addTab(colors_tab, "Colors")

        appearance_hint = QLabel(
            "Tune how the main targets table looks without changing the overall app theme.",
            appearance_tab,
        )
        appearance_hint.setObjectName("SectionHint")
        appearance_hint.setWordWrap(True)
        appearance_layout.addRow(appearance_hint)

        self.row_height_spin = QSpinBox(self)
        self.row_height_spin.setRange(10, 100)
        init_h = settings.value("table/rowHeight", 24, type=int) if settings is not None else 24
        self.row_height_spin.setValue(init_h)
        appearance_layout.addRow("Row height:", self.row_height_spin)

        self.first_col_width_spin = QSpinBox(self)
        self.first_col_width_spin.setRange(50, 500)
        init_w = settings.value("table/firstColumnWidth", 100, type=int) if settings is not None else 100
        self.first_col_width_spin.setValue(init_w)
        appearance_layout.addRow("Name column min width:", self.first_col_width_spin)

        self.font_spin = QSpinBox(self)
        self.font_spin.setRange(8, 16)
        init_fs = settings.value("table/fontSize", 11, type=int) if settings is not None else 11
        self.font_spin.setValue(init_fs)
        appearance_layout.addRow("Font size:", self.font_spin)

        self.color_mode_combo = QComboBox(self)
        self.color_mode_combo.addItem("Colored background", "background")
        self.color_mode_combo.addItem("Colored text + glow", "text_glow")
        init_color_mode = _normalize_table_color_mode(
            settings.value("table/colorMode", "background", type=str) if settings is not None else "background",
            default="background",
        )
        color_mode_idx = self.color_mode_combo.findData(init_color_mode)
        if color_mode_idx >= 0:
            self.color_mode_combo.setCurrentIndex(color_mode_idx)
        appearance_layout.addRow("Status color mode:", self.color_mode_combo)

        self.col_checks: dict[int, QCheckBox] = {}
        columns_hint = QLabel(
            "Choose which columns are visible in the main targets table.",
            columns_tab,
        )
        columns_hint.setObjectName("SectionHint")
        columns_hint.setWordWrap(True)
        columns_layout.addWidget(columns_hint)
        columns_grid = QGridLayout()
        columns_grid.setContentsMargins(0, 0, 0, 0)
        columns_grid.setHorizontalSpacing(16)
        columns_grid.setVerticalSpacing(8)
        headers = list(getattr(table_model, "headers", []) or [])
        for idx, lbl in enumerate(headers[:-1]):
            chk = QCheckBox(lbl, self)
            val = settings.value(f"table/col{idx}", True, type=bool) if settings is not None else True
            chk.setChecked(val)
            self.col_checks[idx] = chk
            pos = len(self.col_checks) - 1
            columns_grid.addWidget(chk, pos // 2, pos % 2)
        columns_layout.addLayout(columns_grid)
        columns_layout.addStretch(1)

        self.sort_combo = QComboBox(self)
        self.sort_combo.addItems(headers)
        default_sort = int(getattr(table_model, "COL_SCORE", 8))
        init_sort = settings.value("table/defaultSortColumn", default_sort, type=int) if settings is not None else default_sort
        if 0 <= init_sort < len(headers):
            self.sort_combo.setCurrentIndex(init_sort)
        appearance_layout.addRow("Default sort column:", self.sort_combo)

        active_theme = getattr(parent, "_theme_name", DEFAULT_UI_THEME) if parent is not None else DEFAULT_UI_THEME
        active_dark = bool(getattr(parent, "_dark_enabled", False)) if parent is not None else False
        highlight_defaults = highlight_palette_for_theme(
            active_theme,
            dark_enabled=active_dark,
            color_blind=bool(getattr(parent, "color_blind_mode", False)) if parent is not None else False,
        )
        default_colors = {
            "below": highlight_defaults.below,
            "limit": highlight_defaults.limit,
            "above": highlight_defaults.above,
        }
        self.selected_colors: dict[str, str] = {}
        colors_hint = QLabel(
            "These three colors drive object status tinting in both table color modes.",
            colors_tab,
        )
        colors_hint.setObjectName("SectionHint")
        colors_hint.setWordWrap(True)
        colors_layout.addRow(colors_hint)
        for key, label in (("below", "Below limit"), ("limit", "Near limit"), ("above", "Above limit")):
            btn = QPushButton(self)
            init = _resolve_table_highlight_color(settings, key, default_colors[key])
            self.selected_colors[key] = str(init)
            btn.clicked.connect(lambda _, k=key: self._pick_status_color(k))
            setattr(self, f"{key}_btn", btn)
            self._refresh_status_color_button(key, label)
            colors_layout.addRow(f"{label}:", btn)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.Apply, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        _style_dialog_button_box(buttons)
        apply_btn = buttons.button(QDialogButtonBox.Apply)
        if apply_btn is not None:
            apply_btn.clicked.connect(self._apply_changes)
        root_layout.addWidget(buttons)
        _fit_dialog_to_screen(
            self,
            preferred_width=860,
            preferred_height=640,
            min_width=620,
            min_height=460,
        )
        localize_widget_tree(self, current_language())

    def _refresh_status_color_button(self, key: str, label: str) -> None:
        btn = getattr(self, f"{key}_btn", None)
        if not isinstance(btn, QPushButton):
            return
        color = str(self.selected_colors.get(key, "") or "")
        text_color = _swatch_text_color(color)
        btn.setText(f"{label}  {color.upper()}")
        btn.setStyleSheet(
            f"background:{color}; color:{text_color}; border:1px solid {color}; font-weight:700; text-align:left; padding:6px 10px;"
        )

    def _pick_status_color(self, key: str) -> None:
        active_theme = getattr(self.parent(), "_theme_name", DEFAULT_UI_THEME) if self.parent() is not None else DEFAULT_UI_THEME
        active_dark = bool(getattr(self.parent(), "_dark_enabled", False)) if self.parent() is not None else False
        palette = highlight_palette_for_theme(
            active_theme,
            dark_enabled=active_dark,
            color_blind=bool(getattr(self.parent(), "color_blind_mode", False)) if self.parent() is not None else False,
        )
        defaults = {"below": palette.below, "limit": palette.limit, "above": palette.above}
        chosen = QColorDialog.getColor(
            QColor(self.selected_colors.get(key, defaults[key])),
            self,
            f"Pick {key} color",
        )
        if not chosen.isValid():
            return
        self.selected_colors[key] = chosen.name().lower()
        label = {"below": "Below limit", "limit": "Near limit", "above": "Above limit"}[key]
        self._refresh_status_color_button(key, label)

    def _apply_changes(self) -> None:
        parent = self.parent()
        if parent is None or not hasattr(parent, "settings"):
            return
        s = parent.settings
        s.setValue("table/rowHeight", self.row_height_spin.value())
        s.setValue("table/firstColumnWidth", self.first_col_width_spin.value())
        s.setValue("table/fontSize", self.font_spin.value())
        s.setValue("table/colorMode", _normalize_table_color_mode(self.color_mode_combo.currentData(), default="background"))
        s.setValue("table/colorModeExplicit", True)
        selected_columns: set[int] = set()
        for idx, chk in self.col_checks.items():
            s.setValue(f"table/col{idx}", chk.isChecked())
            if chk.isChecked():
                selected_columns.add(idx)
        table_model = getattr(parent, "table_model", None)
        observation_columns = _observation_columns_for_table_model(table_model)
        s.setValue("table/viewPreset", "observation" if selected_columns == observation_columns else "full")
        for key in ("below", "limit", "above"):
            s.setValue(f"table/color/{key}", self.selected_colors.get(key))
        s.setValue("table/defaultSortColumn", self.sort_combo.currentIndex())
        s.sync()
        parent._apply_table_settings()

    def accept(self) -> None:
        self._apply_changes()
        super().accept()


class GeneralSettingsDialog(QDialog):
    """Configure default site, date, samples & clock refresh."""

    def __init__(self, parent=None, initial_tab: Optional[str] = None):
        super().__init__(parent)
        self.setObjectName(self.__class__.__name__)
        self.setWindowTitle("General Settings")
        settings = parent.settings if parent and hasattr(parent, "settings") else None
        self._original_dark_mode = (
            settings.value("general/darkMode", DEFAULT_DARK_MODE, type=bool)
            if settings is not None
            else DEFAULT_DARK_MODE
        )
        root_layout = QVBoxLayout(self)
        tabs = _configure_tab_widget(QTabWidget(self))
        self.tabs = tabs
        self._tab_indices: dict[str, int] = {}
        root_layout.addWidget(tabs)

        def _make_tab(title: str) -> QFormLayout:
            page = QWidget(self)
            form = QFormLayout(page)
            form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
            form.setContentsMargins(12, 12, 12, 12)
            form.setSpacing(10)
            tabs.addTab(page, title)
            self._tab_indices[title.lower()] = tabs.count() - 1
            return form

        general_layout = _make_tab("General")
        cutout_layout = _make_tab("Cutout")
        quick_targets_layout = _make_tab("Quick Targets")
        seestar_layout = _make_tab("Seestar")
        ai_layout = _make_tab("AI")
        integrations_layout = _make_tab("Integrations")
        weather_layout = _make_tab("Weather")

        self.site_combo = QComboBox(self)
        observatories = getattr(parent, "observatories", {})
        self.site_combo.addItems(observatories.keys())
        current_obs_name = parent.obs_combo.currentText() if parent is not None and hasattr(parent, "obs_combo") else "Custom"
        init_site = settings.value("general/defaultSite", current_obs_name, type=str) if settings is not None else "Custom"
        self.site_combo.setCurrentText(init_site)
        general_layout.addRow("Default Observatory:", self.site_combo)

        self.ts_spin = QSpinBox(self)
        self.ts_spin.setRange(50, 1000)
        init_ts = settings.value("general/timeSamples", 240, type=int) if settings is not None else 240
        self.ts_spin.setValue(init_ts)
        general_layout.addRow("Visibility samples:", self.ts_spin)

        self.ui_font_spin = QSpinBox(self)
        self.ui_font_spin.setRange(UI_FONT_SIZE_MIN, UI_FONT_SIZE_MAX)
        self.ui_font_spin.setSuffix(" pt")
        init_ui_font = settings.value("general/uiFontSize", 11, type=int) if settings is not None else 11
        self.ui_font_spin.setValue(_sanitize_ui_font_size(init_ui_font))
        general_layout.addRow("UI font size:", self.ui_font_spin)

        self.theme_combo = QComboBox(self)
        for key, label in THEME_CHOICES:
            self.theme_combo.addItem(label, key)
        init_theme = (
            normalize_theme_key(settings.value("general/uiTheme", DEFAULT_UI_THEME, type=str))
            if settings is not None
            else DEFAULT_UI_THEME
        )
        theme_idx = self.theme_combo.findData(init_theme)
        if theme_idx >= 0:
            self.theme_combo.setCurrentIndex(theme_idx)
        general_layout.addRow("Color theme:", self.theme_combo)

        self.language_combo = QComboBox(self)
        for key, label in SUPPORTED_UI_LANGUAGES:
            self.language_combo.addItem(label, key)
        init_language = (
            str(settings.value("general/uiLanguage", "system", type=str) or "system")
            if settings is not None
            else "system"
        ).strip().lower()
        language_idx = self.language_combo.findData(init_language)
        if language_idx >= 0:
            self.language_combo.setCurrentIndex(language_idx)
        general_layout.addRow("Interface language:", self.language_combo)

        self.dark_mode_chk = QCheckBox("Enable dark mode", self)
        self.dark_mode_chk.setChecked(
            settings.value("general/darkMode", DEFAULT_DARK_MODE, type=bool)
            if settings is not None
            else DEFAULT_DARK_MODE
        )
        self.dark_mode_chk.toggled.connect(self._preview_dark_mode)
        if parent and hasattr(parent, "darkModeChanged"):
            parent.darkModeChanged.connect(self._sync_dark_mode_checkbox)
        general_layout.addRow(self.dark_mode_chk)

        self._accent_secondary_colors: dict[str, str] = {}
        for key, _label in THEME_CHOICES:
            normalized = normalize_theme_key(key)
            if parent and hasattr(parent, "_load_accent_secondary_override"):
                self._accent_secondary_colors[normalized] = str(parent._load_accent_secondary_override(normalized) or "")
            else:
                self._accent_secondary_colors[normalized] = ""
        self._accent_secondary_color = self._accent_secondary_colors.get(init_theme, "")
        accent_row = QWidget(self)
        accent_layout = QHBoxLayout(accent_row)
        accent_layout.setContentsMargins(0, 0, 0, 0)
        accent_layout.setSpacing(8)
        self.accent_secondary_btn = QPushButton(accent_row)
        self.accent_secondary_btn.clicked.connect(self._pick_accent_secondary_color)
        self.accent_secondary_reset_btn = QPushButton("Theme default", accent_row)
        _set_button_variant(self.accent_secondary_reset_btn, "ghost")
        self.accent_secondary_reset_btn.clicked.connect(self._reset_accent_secondary_color)
        accent_layout.addWidget(self.accent_secondary_btn, 1)
        accent_layout.addWidget(self.accent_secondary_reset_btn, 0)
        general_layout.addRow("Secondary accent:", accent_row)
        self.theme_combo.currentIndexChanged.connect(self._on_theme_selection_changed)
        self._refresh_accent_secondary_controls()

        self.cutout_view_combo = QComboBox(self)
        for key, label in CUTOUT_VIEW_CHOICES:
            self.cutout_view_combo.addItem(label, key)
        init_cutout_view = (
            _normalize_cutout_view_key(settings.value("general/cutoutView", CUTOUT_DEFAULT_VIEW_KEY, type=str))
            if settings is not None
            else CUTOUT_DEFAULT_VIEW_KEY
        )
        cutout_view_idx = self.cutout_view_combo.findData(init_cutout_view)
        if cutout_view_idx >= 0:
            self.cutout_view_combo.setCurrentIndex(cutout_view_idx)

        self.cutout_survey_combo = QComboBox(self)
        for key, label, _hips in CUTOUT_SURVEY_CHOICES:
            self.cutout_survey_combo.addItem(label, key)
        init_cutout_survey = (
            _normalize_cutout_survey_key(settings.value("general/cutoutSurvey", CUTOUT_DEFAULT_SURVEY_KEY, type=str))
            if settings is not None
            else CUTOUT_DEFAULT_SURVEY_KEY
        )
        cutout_survey_idx = self.cutout_survey_combo.findData(init_cutout_survey)
        if cutout_survey_idx >= 0:
            self.cutout_survey_combo.setCurrentIndex(cutout_survey_idx)

        self.cutout_fov_spin = QSpinBox(self)
        self.cutout_fov_spin.setRange(CUTOUT_MIN_FOV_ARCMIN, CUTOUT_MAX_FOV_ARCMIN)
        self.cutout_fov_spin.setSuffix(" arcmin")
        init_cutout_fov = (
            _sanitize_cutout_fov_arcmin(settings.value("general/cutoutFovArcmin", CUTOUT_DEFAULT_FOV_ARCMIN, type=int))
            if settings is not None
            else CUTOUT_DEFAULT_FOV_ARCMIN
        )
        self.cutout_fov_spin.setValue(init_cutout_fov)

        self.cutout_size_spin = QSpinBox(self)
        self.cutout_size_spin.setRange(CUTOUT_MIN_SIZE_PX, CUTOUT_MAX_SIZE_PX)
        self.cutout_size_spin.setSingleStep(16)
        self.cutout_size_spin.setSuffix(" px")
        init_cutout_size = (
            _sanitize_cutout_size_px(settings.value("general/cutoutSizePx", CUTOUT_DEFAULT_SIZE_PX, type=int))
            if settings is not None
            else CUTOUT_DEFAULT_SIZE_PX
        )
        self.cutout_size_spin.setValue(init_cutout_size)

        cutout_hdr = QLabel("Cutout defaults")
        cutout_hdr.setObjectName("SectionHint")
        cutout_layout.addRow(cutout_hdr)
        cutout_layout.addRow("Cutout view:", self.cutout_view_combo)
        cutout_layout.addRow("Cutout survey:", self.cutout_survey_combo)
        cutout_layout.addRow("Cutout FOV:", self.cutout_fov_spin)
        cutout_layout.addRow("Cutout size:", self.cutout_size_spin)

        quick_hdr = QLabel("Configure how the Quick Targets button builds and filters candidates.")
        quick_hdr.setObjectName("SectionHint")
        quick_hdr.setWordWrap(True)
        quick_targets_layout.addRow(quick_hdr)

        self.quick_targets_count_spin = QSpinBox(self)
        self.quick_targets_count_spin.setRange(QUICK_TARGETS_MIN_COUNT, QUICK_TARGETS_MAX_COUNT)
        self.quick_targets_count_spin.setValue(
            max(
                QUICK_TARGETS_MIN_COUNT,
                min(
                    QUICK_TARGETS_MAX_COUNT,
                    settings.value("general/quickTargetsCount", QUICK_TARGETS_DEFAULT_COUNT, type=int)
                    if settings is not None
                    else QUICK_TARGETS_DEFAULT_COUNT,
                ),
            )
        )
        quick_targets_layout.addRow("Targets to add:", self.quick_targets_count_spin)

        self.quick_targets_min_importance_spin = QDoubleSpinBox(self)
        self.quick_targets_min_importance_spin.setRange(0.0, 10.0)
        self.quick_targets_min_importance_spin.setDecimals(1)
        self.quick_targets_min_importance_spin.setSingleStep(0.5)
        self.quick_targets_min_importance_spin.setValue(
            settings.value("general/quickTargetsMinImportance", BHTOM_SUGGESTION_MIN_IMPORTANCE, type=float)
            if settings is not None
            else BHTOM_SUGGESTION_MIN_IMPORTANCE
        )
        quick_targets_layout.addRow("Min importance:", self.quick_targets_min_importance_spin)

        self.quick_targets_use_score_filter_chk = QCheckBox("Respect current Score ≥ filter", self)
        self.quick_targets_use_score_filter_chk.setChecked(
            settings.value("general/quickTargetsUseScoreFilter", True, type=bool)
            if settings is not None
            else True
        )
        quick_targets_layout.addRow(self.quick_targets_use_score_filter_chk)

        self.quick_targets_use_moon_filter_chk = QCheckBox("Respect current Moon Sep ≥ filter", self)
        self.quick_targets_use_moon_filter_chk.setChecked(
            settings.value("general/quickTargetsUseMoonFilter", True, type=bool)
            if settings is not None
            else True
        )
        quick_targets_layout.addRow(self.quick_targets_use_moon_filter_chk)

        self.quick_targets_use_limiting_mag_chk = QCheckBox("Apply observatory limiting magnitude", self)
        self.quick_targets_use_limiting_mag_chk.setChecked(
            settings.value("general/quickTargetsUseLimitingMag", True, type=bool)
            if settings is not None
            else True
        )
        self.quick_targets_use_limiting_mag_chk.setToolTip(
            "If enabled, targets with known magnitude above the current observatory limit are skipped."
        )
        quick_targets_layout.addRow(self.quick_targets_use_limiting_mag_chk)

        seestar_hdr = QLabel(
            "Configure only the external Seestar ALP connection here. Session templates, target plan defaults and capture settings are configured from Seestar Session…",
            self,
        )
        seestar_hdr.setObjectName("SectionHint")
        seestar_hdr.setWordWrap(True)
        seestar_layout.addRow(seestar_hdr)

        self.seestar_alp_base_url_edit = QLineEdit(self)
        self.seestar_alp_base_url_edit.setPlaceholderText(SEESTAR_ALP_DEFAULT_BASE_URL)
        self.seestar_alp_base_url_edit.setText(
            settings.value("general/seestarAlpBaseUrl", SEESTAR_ALP_DEFAULT_BASE_URL, type=str)
            if settings is not None
            else SEESTAR_ALP_DEFAULT_BASE_URL
        )
        seestar_layout.addRow("ALP base URL:", self.seestar_alp_base_url_edit)

        self.seestar_alp_device_spin = QSpinBox(self)
        self.seestar_alp_device_spin.setRange(0, 16)
        self.seestar_alp_device_spin.setValue(
            max(
                0,
                int(
                    settings.value("general/seestarAlpDeviceNum", SEESTAR_ALP_DEFAULT_DEVICE_NUM, type=int)
                    if settings is not None
                    else SEESTAR_ALP_DEFAULT_DEVICE_NUM
                ),
            )
        )
        seestar_layout.addRow("ALP device #:", self.seestar_alp_device_spin)

        self.seestar_alp_client_id_spin = QSpinBox(self)
        self.seestar_alp_client_id_spin.setRange(1, 999999)
        self.seestar_alp_client_id_spin.setValue(
            max(
                1,
                int(
                    settings.value("general/seestarAlpClientId", SEESTAR_ALP_DEFAULT_CLIENT_ID, type=int)
                    if settings is not None
                    else SEESTAR_ALP_DEFAULT_CLIENT_ID
                ),
            )
        )
        seestar_layout.addRow("ALP client ID:", self.seestar_alp_client_id_spin)

        self.seestar_alp_timeout_spin = QDoubleSpinBox(self)
        self.seestar_alp_timeout_spin.setRange(1.0, 60.0)
        self.seestar_alp_timeout_spin.setDecimals(1)
        self.seestar_alp_timeout_spin.setSingleStep(0.5)
        self.seestar_alp_timeout_spin.setSuffix(" s")
        self.seestar_alp_timeout_spin.setValue(
            max(
                1.0,
                float(
                    settings.value("general/seestarAlpTimeoutSec", SEESTAR_ALP_DEFAULT_TIMEOUT_S, type=float)
                    if settings is not None
                    else SEESTAR_ALP_DEFAULT_TIMEOUT_S
                ),
            )
        )
        seestar_layout.addRow("ALP timeout:", self.seestar_alp_timeout_spin)

        seestar_btn_row = QWidget(self)
        seestar_btn_layout = QHBoxLayout(seestar_btn_row)
        seestar_btn_layout.setContentsMargins(0, 0, 0, 0)
        seestar_btn_layout.setSpacing(8)

        self.seestar_alp_test_btn = QPushButton("Test ALP connection", seestar_btn_row)
        self.seestar_alp_test_btn.clicked.connect(self._test_seestar_alp_connection)
        seestar_btn_layout.addWidget(self.seestar_alp_test_btn, 0)

        self.seestar_alp_open_ui_btn = QPushButton("Open ALP Web UI", seestar_btn_row)
        self.seestar_alp_open_ui_btn.clicked.connect(self._open_seestar_alp_web_ui)
        seestar_btn_layout.addWidget(self.seestar_alp_open_ui_btn, 0)
        seestar_btn_layout.addStretch(1)
        seestar_layout.addRow("", seestar_btn_row)
        self.seestar_alp_status_label = QLabel(
            render_alp_backend_status_text(
                None,
                base_url=self.seestar_alp_base_url_edit.text().strip() or SEESTAR_ALP_DEFAULT_BASE_URL,
                device_num=int(self.seestar_alp_device_spin.value()),
            ),
            self,
        )
        self.seestar_alp_status_label.setObjectName("SectionHint")
        self.seestar_alp_status_label.setWordWrap(True)
        seestar_layout.addRow("ALP status:", self.seestar_alp_status_label)

        self.llm_url_edit = QLineEdit(self)
        self.llm_url_edit.setPlaceholderText(LLMConfig.DEFAULT_URL)
        self.llm_url_edit.setText(
            settings.value("llm/serverUrl", LLMConfig.DEFAULT_URL, type=str)
            if settings is not None
            else LLMConfig.DEFAULT_URL
        )
        self.llm_model_combo = QComboBox(self)
        self.llm_model_combo.setEditable(True)
        self.llm_model_combo.setInsertPolicy(QComboBox.NoInsert)
        self.llm_model_combo.setMinimumContentsLength(22)
        self.llm_model_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.llm_model_combo.lineEdit().setPlaceholderText(LLMConfig.DEFAULT_MODEL)
        self.llm_model_combo.setCurrentText(
            settings.value("llm/model", LLMConfig.DEFAULT_MODEL, type=str)
            if settings is not None
            else LLMConfig.DEFAULT_MODEL
        )
        self.llm_models_refresh_btn = QPushButton("Detect models", self)
        self.llm_models_refresh_btn.clicked.connect(self._refresh_llm_models)
        self.llm_models_status = QLabel("Model list: not loaded", self)
        self.llm_models_status.setObjectName("SectionHint")
        self.llm_models_status.setWordWrap(True)
        self._llm_models_worker: Optional[LLMModelDiscoveryWorker] = None
        self._llm_models_refresh_timer = QTimer(self)
        self._llm_models_refresh_timer.setSingleShot(True)
        self._llm_models_refresh_timer.setInterval(650)
        self._llm_models_refresh_timer.timeout.connect(self._refresh_llm_models)
        self.llm_url_edit.textChanged.connect(lambda *_: self._llm_models_refresh_timer.start())

        self.llm_timeout_spin = QSpinBox(self)
        self.llm_timeout_spin.setRange(15, 900)
        self.llm_timeout_spin.setSingleStep(15)
        self.llm_timeout_spin.setSuffix(" s")
        self.llm_timeout_spin.setValue(
            max(
                15,
                int(
                    settings.value("llm/timeoutSec", LLMConfig.DEFAULT_TIMEOUT_S, type=int)
                    if settings is not None
                    else LLMConfig.DEFAULT_TIMEOUT_S
                ),
            )
        )
        self.llm_max_tokens_spin = QSpinBox(self)
        self.llm_max_tokens_spin.setRange(32, 2048)
        self.llm_max_tokens_spin.setSingleStep(32)
        self.llm_max_tokens_spin.setSuffix(" tok")
        self.llm_max_tokens_spin.setValue(
            max(
                32,
                int(
                    settings.value("llm/maxTokens", LLMConfig.DEFAULT_MAX_TOKENS, type=int)
                    if settings is not None
                    else LLMConfig.DEFAULT_MAX_TOKENS
                ),
            )
        )
        self.llm_enable_thinking_chk = QCheckBox("Enable model thinking / reasoning", self)
        self.llm_enable_thinking_chk.setChecked(
            settings.value("llm/enableThinking", LLMConfig.DEFAULT_ENABLE_THINKING, type=bool)
            if settings is not None
            else LLMConfig.DEFAULT_ENABLE_THINKING
        )
        self.llm_enable_thinking_chk.setToolTip(
            "When enabled, the model may spend tokens on internal reasoning. "
            "This can improve some answers but usually makes responses slower and longer. "
            "Actual behavior depends on the backend and model."
        )
        self.llm_enable_chat_memory_chk = QCheckBox("Enable short chat memory (last 1-2 turns)", self)
        self.llm_enable_chat_memory_chk.setChecked(
            settings.value("llm/enableChatMemory", LLMConfig.DEFAULT_ENABLE_CHAT_MEMORY, type=bool)
            if settings is not None
            else LLMConfig.DEFAULT_ENABLE_CHAT_MEMORY
        )
        self.llm_enable_chat_memory_chk.setToolTip(
            "When enabled, the AI may reuse the last 1-2 user/LLM turns for follow-up questions. "
            "This helps short follow-ups, but requests can become slower."
        )
        self.llm_chat_font_spin = QSpinBox(self)
        self.llm_chat_font_spin.setRange(9, 24)
        self.llm_chat_font_spin.setSingleStep(1)
        self.llm_chat_font_spin.setSuffix(" pt")
        self.llm_chat_font_spin.setValue(
            max(
                9,
                int(
                    settings.value("llm/chatFontSizePt", LLMConfig.DEFAULT_CHAT_FONT_PT, type=int)
                    if settings is not None
                    else LLMConfig.DEFAULT_CHAT_FONT_PT
                ),
            )
        )
        self.llm_chat_spacing_combo = QComboBox(self)
        for key, label in AI_CHAT_SPACING_CHOICES:
            self.llm_chat_spacing_combo.addItem(label, key)
        init_chat_spacing = (
            settings.value("llm/chatSpacing", LLMConfig.DEFAULT_CHAT_SPACING, type=str)
            if settings is not None
            else LLMConfig.DEFAULT_CHAT_SPACING
        )
        idx = self.llm_chat_spacing_combo.findData(str(init_chat_spacing))
        if idx >= 0:
            self.llm_chat_spacing_combo.setCurrentIndex(idx)

        self.llm_chat_tint_combo = QComboBox(self)
        for key, label in AI_CHAT_TINT_CHOICES:
            self.llm_chat_tint_combo.addItem(label, key)
        init_chat_tint = (
            settings.value("llm/chatTintStrength", LLMConfig.DEFAULT_CHAT_TINT_STRENGTH, type=str)
            if settings is not None
            else LLMConfig.DEFAULT_CHAT_TINT_STRENGTH
        )
        idx = self.llm_chat_tint_combo.findData(str(init_chat_tint))
        if idx >= 0:
            self.llm_chat_tint_combo.setCurrentIndex(idx)

        self.llm_chat_width_combo = QComboBox(self)
        for key, label in AI_CHAT_WIDTH_CHOICES:
            self.llm_chat_width_combo.addItem(label, key)
        init_chat_width = (
            settings.value("llm/chatMessageWidth", LLMConfig.DEFAULT_CHAT_WIDTH, type=str)
            if settings is not None
            else LLMConfig.DEFAULT_CHAT_WIDTH
        )
        idx = self.llm_chat_width_combo.findData(str(init_chat_width))
        if idx >= 0:
            self.llm_chat_width_combo.setCurrentIndex(idx)

        self.llm_status_error_clear_spin = QSpinBox(self)
        self.llm_status_error_clear_spin.setRange(0, 60)
        self.llm_status_error_clear_spin.setSingleStep(1)
        self.llm_status_error_clear_spin.setSuffix(" s")
        self.llm_status_error_clear_spin.setSpecialValueText("Off")
        self.llm_status_error_clear_spin.setValue(
            max(
                0,
                int(
                    settings.value("llm/statusErrorClearSec", LLMConfig.DEFAULT_STATUS_ERROR_CLEAR_S, type=int)
                    if settings is not None
                    else LLMConfig.DEFAULT_STATUS_ERROR_CLEAR_S
                ),
            )
        )

        llm_hdr = QLabel("Local LLM assistant (optional)")
        llm_hdr.setObjectName("SectionHint")
        ai_layout.addRow(llm_hdr)
        ai_layout.addRow("LLM server URL:", self.llm_url_edit)
        ai_layout.addRow("LLM model:", self.llm_model_combo)
        llm_model_row = QWidget(self)
        llm_model_row_l = QHBoxLayout(llm_model_row)
        llm_model_row_l.setContentsMargins(0, 0, 0, 0)
        llm_model_row_l.setSpacing(8)
        llm_model_row_l.addWidget(self.llm_models_refresh_btn, 0)
        llm_model_row_l.addWidget(self.llm_models_status, 1)
        ai_layout.addRow("", llm_model_row)
        llm_endpoint_hint = QLabel(
            "Recommended: Jan local API; use Jan's server URL and active/default model.\n"
            "Ollama: URL http://localhost:11434, model gemma4:e4b\n"
            "Docker Model Runner: URL http://localhost:12434/engines, model docker.io/ai/gemma4:E4B",
            self,
        )
        llm_endpoint_hint.setObjectName("SectionHint")
        llm_endpoint_hint.setWordWrap(True)
        ai_layout.addRow("", llm_endpoint_hint)
        ai_layout.addRow("LLM timeout:", self.llm_timeout_spin)
        ai_layout.addRow("LLM max tokens:", self.llm_max_tokens_spin)
        ai_layout.addRow("", self.llm_enable_thinking_chk)
        ai_layout.addRow("", self.llm_enable_chat_memory_chk)
        ai_layout.addRow("AI chat font size:", self.llm_chat_font_spin)
        ai_layout.addRow("Chat spacing:", self.llm_chat_spacing_combo)
        ai_layout.addRow("Bubble tint strength:", self.llm_chat_tint_combo)
        ai_layout.addRow("Message width:", self.llm_chat_width_combo)
        ai_layout.addRow("Auto-clear AI warnings:", self.llm_status_error_clear_spin)

        self.bhtom_api_edit = QLineEdit(self)
        self.bhtom_api_edit.setEchoMode(QLineEdit.Password)
        self.bhtom_api_edit.setPlaceholderText("API token for BHTOM target suggestions")
        self.bhtom_api_edit.setText(
            settings.value("general/bhtomApiToken", "", type=str) if settings is not None else ""
        )

        bhtom_hdr = QLabel("BHTOM suggestions (optional)")
        bhtom_hdr.setObjectName("SectionHint")
        integrations_layout.addRow(bhtom_hdr)
        integrations_layout.addRow("BHTOM API token:", self.bhtom_api_edit)
        self.bhtom_refresh_startup_chk = QCheckBox("Always refresh BHTOM cache on startup", self)
        self.bhtom_refresh_startup_chk.setChecked(
            settings.value("general/bhtomRefreshOnStartup", True, type=bool)
            if settings is not None
            else True
        )
        self.bhtom_refresh_startup_chk.setToolTip("Fetch a fresh BHTOM candidate list after launch and overwrite cache.")
        integrations_layout.addRow(self.bhtom_refresh_startup_chk)

        self.tns_api_edit = QLineEdit(self)
        self.tns_api_edit.setEchoMode(QLineEdit.Password)
        self.tns_api_edit.setPlaceholderText("API key from TNS bot")
        self.tns_api_edit.setText(settings.value("general/tnsApiKey", "", type=str) if settings is not None else "")
        self.tns_bot_id_edit = QLineEdit(self)
        self.tns_bot_id_edit.setPlaceholderText("Numeric bot ID")
        self.tns_bot_id_edit.setText(settings.value("general/tnsBotId", "", type=str) if settings is not None else "")
        self.tns_bot_name_edit = QLineEdit(self)
        self.tns_bot_name_edit.setPlaceholderText("Exact bot name")
        self.tns_bot_name_edit.setText(settings.value("general/tnsBotName", "", type=str) if settings is not None else "")
        self.tns_endpoint_combo = QComboBox(self)
        for key, label in TNS_ENDPOINT_CHOICES:
            self.tns_endpoint_combo.addItem(label, key)
        init_tns_endpoint = (
            _normalize_tns_endpoint_key(settings.value("general/tnsEndpoint", "production", type=str))
            if settings is not None
            else "production"
        )
        endpoint_idx = self.tns_endpoint_combo.findData(init_tns_endpoint)
        if endpoint_idx >= 0:
            self.tns_endpoint_combo.setCurrentIndex(endpoint_idx)
        self.tns_test_btn = QPushButton("Test TNS credentials", self)
        self.tns_test_btn.clicked.connect(self._test_tns_credentials)

        tns_hdr = QLabel("TNS credentials (optional)")
        tns_hdr.setObjectName("SectionHint")
        integrations_layout.addRow(tns_hdr)
        integrations_layout.addRow("TNS API key:", self.tns_api_edit)
        integrations_layout.addRow("TNS Bot ID:", self.tns_bot_id_edit)
        integrations_layout.addRow("TNS Bot name:", self.tns_bot_name_edit)
        integrations_layout.addRow("TNS endpoint:", self.tns_endpoint_combo)
        integrations_layout.addRow("", self.tns_test_btn)

        weather_hdr = QLabel("Weather workspace defaults")
        weather_hdr.setObjectName("SectionHint")
        weather_layout.addRow(weather_hdr)

        self.weather_default_source_combo = QComboBox(self)
        for key, label in (
            ("average", "Average (real measurements)"),
            ("metar", "Nearest METAR"),
            ("custom", "Custom URL"),
        ):
            self.weather_default_source_combo.addItem(label, key)
        init_weather_source = (
            settings.value("weather/defaultConditionsSource", "average", type=str)
            if settings is not None
            else "average"
        )
        source_idx = self.weather_default_source_combo.findData(str(init_weather_source))
        if source_idx >= 0:
            self.weather_default_source_combo.setCurrentIndex(source_idx)
        weather_layout.addRow("Default conditions source:", self.weather_default_source_combo)

        self.weather_refresh_spin = QSpinBox(self)
        self.weather_refresh_spin.setRange(30, 1800)
        self.weather_refresh_spin.setSuffix(" s")
        self.weather_refresh_spin.setValue(
            max(
                30,
                min(
                    1800,
                    settings.value("weather/autoRefreshSec", 120, type=int)
                    if settings is not None
                    else 120,
                ),
            )
        )
        weather_layout.addRow("Auto refresh:", self.weather_refresh_spin)

        self.weather_wunderground_edit = QLineEdit(self)
        self.weather_wunderground_edit.setPlaceholderText(
            "https://www.wunderground.com/dashboard/pws/IBIAKW3 or station id"
        )
        self.weather_wunderground_edit.setText(
            settings.value("weather/wundergroundUrl", "", type=str) if settings is not None else ""
        )
        weather_layout.addRow("Wunderground station:", self.weather_wunderground_edit)

        self.weather_weathercloud_edit = QLineEdit(self)
        self.weather_weathercloud_edit.setPlaceholderText(
            "https://app.weathercloud.net/... (station public page)"
        )
        self.weather_weathercloud_edit.setText(
            settings.value("weather/weathercloudUrl", "", type=str) if settings is not None else ""
        )
        weather_layout.addRow("WeatherCloud station:", self.weather_weathercloud_edit)

        self.weather_local_links_edit = QTextEdit(self)
        self.weather_local_links_edit.setPlaceholderText(
            "Optional local sources (one per line): Label|https://example\nor just URL"
        )
        self.weather_local_links_edit.setMinimumHeight(74)
        self.weather_local_links_edit.setMaximumHeight(120)
        self.weather_local_links_edit.setPlainText(
            settings.value("weather/localConditionsLinks", "", type=str) if settings is not None else ""
        )
        weather_layout.addRow("Additional local links:", self.weather_local_links_edit)

        self.weather_cloud_source_combo = QComboBox(self)
        self.weather_cloud_source_combo.addItem("EarthEnv", "earthenv")
        init_cloud_source = (
            settings.value("weather/cloudMapSource", "earthenv", type=str)
            if settings is not None
            else "earthenv"
        )
        cloud_source_idx = self.weather_cloud_source_combo.findData(str(init_cloud_source))
        if cloud_source_idx >= 0:
            self.weather_cloud_source_combo.setCurrentIndex(cloud_source_idx)
        weather_layout.addRow("Cloud map source:", self.weather_cloud_source_combo)

        self.weather_month_mode_combo = QComboBox(self)
        self.weather_month_mode_combo.addItem("Session month", "session_month")
        self.weather_month_mode_combo.addItem("Current month", "current_month")
        init_month_mode = (
            settings.value("weather/cloudMapMonthMode", "session_month", type=str)
            if settings is not None
            else "session_month"
        )
        month_mode_idx = self.weather_month_mode_combo.findData(str(init_month_mode))
        if month_mode_idx >= 0:
            self.weather_month_mode_combo.setCurrentIndex(month_mode_idx)
        weather_layout.addRow("Cloud map month mode:", self.weather_month_mode_combo)

        custom_hint = QLabel(
            "Custom conditions URL is configured per observatory in Observatory Manager. "
            "Accepted formats: AstroPlanner JSON (temp_c/wind_ms/cloud_pct/rh_pct/pressure_hpa; missing values show as N/A) "
            "or Weather.com PWS observations/all/1day JSON.",
            self,
        )
        custom_hint.setObjectName("SectionHint")
        custom_hint.setWordWrap(True)
        weather_layout.addRow(custom_hint)

        self.sun_path_chk = QCheckBox("Plot Sun path", self)
        self.moon_path_chk = QCheckBox("Plot Moon path", self)
        self.obj_path_chk = QCheckBox("Plot object paths", self)
        self.color_blind_chk = QCheckBox("Color-blind friendly palette", self)
        self.radar_sweep_chk = QCheckBox("Radar sweep animation", self)

        self.sun_path_chk.setChecked(settings.value("general/showSunPath", True, type=bool) if settings is not None else True)
        self.moon_path_chk.setChecked(settings.value("general/showMoonPath", True, type=bool) if settings is not None else True)
        self.obj_path_chk.setChecked(settings.value("general/showObjPath", True, type=bool) if settings is not None else True)
        self.color_blind_chk.setChecked(settings.value("general/colorBlindMode", False, type=bool) if settings is not None else False)
        self.radar_sweep_chk.setChecked(
            settings.value("general/radarSweepAnimation", False, type=bool) if settings is not None else False
        )
        self.radar_sweep_chk.setToolTip(
            "Optional cyberpunk sweep line in the sky radar. A rotating scan highlights nearby targets."
        )
        self.radar_speed_slider = QSlider(Qt.Horizontal, self)
        self.radar_speed_slider.setRange(40, 260)
        self.radar_speed_slider.setSingleStep(5)
        self.radar_speed_slider.setPageStep(10)
        self.radar_speed_slider.setValue(
            max(
                40,
                min(
                    260,
                    int(settings.value("general/radarSweepSpeed", 140, type=int) if settings is not None else 140),
                ),
            )
        )
        self.radar_speed_value_label = QLabel(self)
        self.radar_speed_value_label.setObjectName("SectionHint")
        self.radar_speed_value_label.setMinimumWidth(56)
        radar_speed_row = QWidget(self)
        radar_speed_layout = QHBoxLayout(radar_speed_row)
        radar_speed_layout.setContentsMargins(0, 0, 0, 0)
        radar_speed_layout.setSpacing(8)
        radar_speed_layout.addWidget(self.radar_speed_slider, 1)
        radar_speed_layout.addWidget(self.radar_speed_value_label, 0)
        self.radar_speed_label = QLabel("Radar sweep speed:", self)
        self.radar_speed_label.setObjectName("SectionHint")
        self.radar_speed_row = radar_speed_row
        self.radar_sweep_chk.toggled.connect(self._update_radar_speed_controls)
        self.radar_speed_slider.valueChanged.connect(self._update_radar_speed_controls)

        plot_hdr = QLabel("Plot options")
        plot_hdr.setObjectName("SectionHint")
        general_layout.addRow(plot_hdr)
        general_layout.addRow(self.sun_path_chk)
        general_layout.addRow(self.moon_path_chk)
        general_layout.addRow(self.obj_path_chk)
        general_layout.addRow(self.color_blind_chk)
        general_layout.addRow(self.radar_sweep_chk)
        general_layout.addRow(self.radar_speed_label, self.radar_speed_row)
        self._update_radar_speed_controls()

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.Apply, self)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        _style_dialog_button_box(btns)
        apply_btn = btns.button(QDialogButtonBox.Apply)
        if apply_btn is not None:
            apply_btn.clicked.connect(self._apply_changes)
        root_layout.addWidget(btns)
        _fit_dialog_to_screen(
            self,
            preferred_width=1240,
            preferred_height=820,
            min_width=920,
            min_height=620,
        )
        initial_key = str(initial_tab or "").strip().lower()
        if initial_key in self._tab_indices:
            self.tabs.setCurrentIndex(self._tab_indices[initial_key])
        localize_widget_tree(self, current_language())

    @Slot(bool)
    def _preview_dark_mode(self, checked: bool):
        parent = self.parent()
        if parent and hasattr(parent, "_set_dark_mode_enabled"):
            parent._set_dark_mode_enabled(checked, persist=False)

    @Slot(bool)
    def _sync_dark_mode_checkbox(self, enabled: bool):
        blocker = QSignalBlocker(self.dark_mode_chk)
        self.dark_mode_chk.setChecked(enabled)
        del blocker

    def _effective_accent_secondary_color(self) -> str:
        if self._accent_secondary_color:
            return self._accent_secondary_color
        theme_key = normalize_theme_key(self.theme_combo.currentData())
        tokens = resolve_theme_tokens(
            theme_key,
            dark_enabled=self.dark_mode_chk.isChecked(),
            ui_font_size=int(self.ui_font_spin.value()),
        )
        return str(tokens.get("accent_secondary", "#ff4fd8"))

    def _refresh_accent_secondary_controls(self) -> None:
        color = self._effective_accent_secondary_color()
        text_color = _swatch_text_color(color)
        label = color.upper() if self._accent_secondary_color else f"Theme ({color.upper()})"
        self.accent_secondary_btn.setText(label)
        self.accent_secondary_btn.setStyleSheet(
            f"background:{color}; color:{text_color}; border:1px solid {color}; font-weight:700;"
        )
        self.accent_secondary_reset_btn.setEnabled(bool(self._accent_secondary_color))

    def _pick_accent_secondary_color(self) -> None:
        chosen = QColorDialog.getColor(QColor(self._effective_accent_secondary_color()), self, "Pick secondary accent")
        if not chosen.isValid():
            return
        self._accent_secondary_color = chosen.name().lower()
        self._accent_secondary_colors[normalize_theme_key(self.theme_combo.currentData())] = self._accent_secondary_color
        self._refresh_accent_secondary_controls()

    def _reset_accent_secondary_color(self) -> None:
        self._accent_secondary_color = ""
        self._accent_secondary_colors[normalize_theme_key(self.theme_combo.currentData())] = ""
        self._refresh_accent_secondary_controls()

    def _on_theme_selection_changed(self, _index: int) -> None:
        theme_key = normalize_theme_key(self.theme_combo.currentData())
        self._accent_secondary_color = self._accent_secondary_colors.get(theme_key, "")
        self._refresh_accent_secondary_controls()

    def _update_radar_speed_controls(self, *_args) -> None:
        visible = self.radar_sweep_chk.isChecked()
        self.radar_speed_label.setVisible(visible)
        self.radar_speed_row.setVisible(visible)
        self.radar_speed_value_label.setText(f"{self.radar_speed_slider.value() / 100.0:.2f}x")

    def _refresh_llm_models(self) -> None:
        if self._llm_models_worker is not None and self._llm_models_worker.isRunning():
            return
        base_url = str(self.llm_url_edit.text() or "").strip()
        if not base_url:
            self.llm_models_status.setText("Model list: set LLM server URL first.")
            return
        self.llm_models_status.setText("Model list: detecting…")
        self._llm_models_worker = LLMModelDiscoveryWorker(base_url, timeout_s=6, parent=self)
        self._llm_models_worker.modelsReady.connect(self._on_llm_models_ready)
        self._llm_models_worker.failed.connect(self._on_llm_models_failed)
        self._llm_models_worker.finished.connect(self._llm_models_worker.deleteLater)
        self._llm_models_worker.start()

    @Slot(list, str)
    def _on_llm_models_ready(self, models: list, backend_label: str) -> None:
        current_text = str(self.llm_model_combo.currentText() or "").strip()
        self.llm_model_combo.blockSignals(True)
        self.llm_model_combo.clear()
        for name in models:
            if isinstance(name, str) and name.strip():
                self.llm_model_combo.addItem(name.strip())
        if current_text:
            self.llm_model_combo.setCurrentText(current_text)
        elif self.llm_model_combo.count() > 0:
            self.llm_model_combo.setCurrentIndex(0)
        self.llm_model_combo.blockSignals(False)
        self.llm_models_status.setText(f"Model list: {backend_label} ({len(models)})")
        self._llm_models_worker = None

    @Slot(str)
    def _on_llm_models_failed(self, message: str) -> None:
        self.llm_models_status.setText(f"Model list: {message}")
        self._llm_models_worker = None

    def _apply_changes(self) -> None:
        parent = self.parent()
        if parent is None or not hasattr(parent, "settings"):
            return
        s = parent.settings
        rerun_plan = any(
            (
                s.value("general/defaultSite", self.site_combo.currentText(), type=str) != self.site_combo.currentText(),
                s.value("general/timeSamples", self.ts_spin.value(), type=int) != self.ts_spin.value(),
                s.value("general/showSunPath", self.sun_path_chk.isChecked(), type=bool) != self.sun_path_chk.isChecked(),
                s.value("general/showMoonPath", self.moon_path_chk.isChecked(), type=bool) != self.moon_path_chk.isChecked(),
                s.value("general/showObjPath", self.obj_path_chk.isChecked(), type=bool) != self.obj_path_chk.isChecked(),
                s.value("general/colorBlindMode", self.color_blind_chk.isChecked(), type=bool) != self.color_blind_chk.isChecked(),
            )
        )
        s.setValue("general/defaultSite", self.site_combo.currentText())
        s.setValue("general/timeSamples", self.ts_spin.value())
        s.setValue("general/uiFontSize", self.ui_font_spin.value())
        selected_theme = normalize_theme_key(self.theme_combo.currentData())
        s.setValue("general/uiTheme", selected_theme)
        s.setValue("general/uiLanguage", str(self.language_combo.currentData() or "system"))
        for theme_key, color in self._accent_secondary_colors.items():
            if parent_obj := self.parent():
                if hasattr(parent_obj, "_save_accent_secondary_override"):
                    parent_obj._save_accent_secondary_override(theme_key, color)
                    continue
            per_theme_key = f"general/accentSecondaryColorByTheme/{normalize_theme_key(theme_key)}"
            normalized_color = _normalized_css_color(color)
            if normalized_color:
                s.setValue(per_theme_key, normalized_color)
            else:
                s.remove(per_theme_key)
        s.remove("general/accentSecondaryColor")
        s.setValue("general/darkMode", self.dark_mode_chk.isChecked())
        s.setValue("general/showSunPath", self.sun_path_chk.isChecked())
        s.setValue("general/showMoonPath", self.moon_path_chk.isChecked())
        s.setValue("general/showObjPath", self.obj_path_chk.isChecked())
        s.setValue("general/colorBlindMode", self.color_blind_chk.isChecked())
        s.setValue("general/radarSweepAnimation", self.radar_sweep_chk.isChecked())
        s.setValue("general/radarSweepSpeed", int(self.radar_speed_slider.value()))
        s.setValue("general/cutoutView", _normalize_cutout_view_key(self.cutout_view_combo.currentData()))
        s.setValue("general/cutoutSurvey", _normalize_cutout_survey_key(self.cutout_survey_combo.currentData()))
        s.setValue("general/cutoutFovArcmin", _sanitize_cutout_fov_arcmin(self.cutout_fov_spin.value()))
        s.setValue("general/cutoutSizePx", _sanitize_cutout_size_px(self.cutout_size_spin.value()))
        s.setValue(
            "general/quickTargetsCount",
            max(QUICK_TARGETS_MIN_COUNT, min(QUICK_TARGETS_MAX_COUNT, self.quick_targets_count_spin.value())),
        )
        s.setValue(
            "general/quickTargetsMinImportance",
            max(0.0, float(self.quick_targets_min_importance_spin.value())),
        )
        s.setValue("general/quickTargetsUseScoreFilter", self.quick_targets_use_score_filter_chk.isChecked())
        s.setValue("general/quickTargetsUseMoonFilter", self.quick_targets_use_moon_filter_chk.isChecked())
        s.setValue("general/quickTargetsUseLimitingMag", self.quick_targets_use_limiting_mag_chk.isChecked())
        s.setValue("general/seestarAlpBaseUrl", self.seestar_alp_base_url_edit.text().strip() or SEESTAR_ALP_DEFAULT_BASE_URL)
        s.setValue("general/seestarAlpDeviceNum", max(0, int(self.seestar_alp_device_spin.value())))
        s.setValue("general/seestarAlpClientId", max(1, int(self.seestar_alp_client_id_spin.value())))
        s.setValue("general/seestarAlpTimeoutSec", max(1.0, float(self.seestar_alp_timeout_spin.value())))
        llm_url = self.llm_url_edit.text().strip() or LLMConfig.DEFAULT_URL
        llm_model = self.llm_model_combo.currentText().strip() or LLMConfig.DEFAULT_MODEL
        llm_timeout = max(15, int(self.llm_timeout_spin.value()))
        llm_max_tokens = max(32, int(self.llm_max_tokens_spin.value()))
        llm_enable_thinking = bool(self.llm_enable_thinking_chk.isChecked())
        llm_enable_chat_memory = bool(self.llm_enable_chat_memory_chk.isChecked())
        llm_chat_font_pt = max(9, int(self.llm_chat_font_spin.value()))
        llm_chat_spacing = str(self.llm_chat_spacing_combo.currentData() or LLMConfig.DEFAULT_CHAT_SPACING)
        llm_chat_tint = str(self.llm_chat_tint_combo.currentData() or LLMConfig.DEFAULT_CHAT_TINT_STRENGTH)
        llm_chat_width = str(self.llm_chat_width_combo.currentData() or LLMConfig.DEFAULT_CHAT_WIDTH)
        llm_status_error_clear_s = max(0, int(self.llm_status_error_clear_spin.value()))
        s.setValue("llm/serverUrl", llm_url)
        s.setValue("llm/model", llm_model)
        s.setValue("llm/timeoutSec", llm_timeout)
        s.setValue("llm/maxTokens", llm_max_tokens)
        s.setValue("llm/enableThinking", llm_enable_thinking)
        s.setValue("llm/enableChatMemory", llm_enable_chat_memory)
        s.setValue("llm/chatFontSizePt", llm_chat_font_pt)
        s.setValue("llm/chatSpacing", llm_chat_spacing)
        s.setValue("llm/chatTintStrength", llm_chat_tint)
        s.setValue("llm/chatMessageWidth", llm_chat_width)
        s.setValue("llm/statusErrorClearSec", llm_status_error_clear_s)
        s.setValue("general/bhtomApiToken", self.bhtom_api_edit.text().strip())
        s.setValue("general/bhtomRefreshOnStartup", bool(self.bhtom_refresh_startup_chk.isChecked()))
        s.setValue("general/tnsApiKey", self.tns_api_edit.text().strip())
        s.setValue("general/tnsBotId", self.tns_bot_id_edit.text().strip())
        s.setValue("general/tnsBotName", self.tns_bot_name_edit.text().strip())
        s.setValue("general/tnsEndpoint", _normalize_tns_endpoint_key(self.tns_endpoint_combo.currentData()))
        s.setValue("weather/defaultConditionsSource", str(self.weather_default_source_combo.currentData() or "average"))
        s.setValue("weather/autoRefreshSec", max(30, min(1800, int(self.weather_refresh_spin.value()))))
        s.setValue("weather/wundergroundUrl", self.weather_wunderground_edit.text().strip())
        s.setValue("weather/weathercloudUrl", self.weather_weathercloud_edit.text().strip())
        s.setValue("weather/localConditionsLinks", self.weather_local_links_edit.toPlainText().strip())
        s.setValue("weather/cloudMapSource", str(self.weather_cloud_source_combo.currentData() or "earthenv"))
        s.setValue("weather/cloudMapMonthMode", str(self.weather_month_mode_combo.currentData() or "session_month"))
        s.sync()
        parent._apply_general_settings()
        if rerun_plan:
            parent._run_plan()
        self._original_dark_mode = self.dark_mode_chk.isChecked()

    def accept(self) -> None:
        self._apply_changes()
        super().accept()

    def _test_seestar_alp_connection(self) -> None:
        try:
            config = SeestarAlpConfig(
                base_url=self.seestar_alp_base_url_edit.text().strip() or SEESTAR_ALP_DEFAULT_BASE_URL,
                device_num=max(0, int(self.seestar_alp_device_spin.value())),
                client_id=max(1, int(self.seestar_alp_client_id_spin.value())),
                timeout_s=max(1.0, float(self.seestar_alp_timeout_spin.value())),
            )
            result = SeestarAlpClient(config).test_connection()
        except Exception as exc:
            self.seestar_alp_status_label.setText(
                render_alp_backend_status_text(
                    None,
                    base_url=self.seestar_alp_base_url_edit.text().strip() or SEESTAR_ALP_DEFAULT_BASE_URL,
                    device_num=max(0, int(self.seestar_alp_device_spin.value())),
                    last_error=str(exc),
                )
            )
            QMessageBox.warning(self, "Seestar ALP", str(exc))
            return
        self.seestar_alp_status_label.setText(render_alp_backend_status_text(result))
        QMessageBox.information(self, "Seestar ALP", self.seestar_alp_status_label.text())

    def _open_seestar_alp_web_ui(self) -> None:
        url = build_alp_web_ui_url(self.seestar_alp_base_url_edit.text().strip() or SEESTAR_ALP_DEFAULT_BASE_URL)
        qurl = QUrl(url)
        if not qurl.isValid():
            QMessageBox.warning(self, "Seestar ALP", f"Invalid URL:\n{url}")
            return
        if not QDesktopServices.openUrl(qurl):
            QMessageBox.warning(self, "Seestar ALP", f"Unable to open URL:\n{url}")

    def reject(self) -> None:
        parent = self.parent()
        if parent and hasattr(parent, "_set_dark_mode_enabled"):
            parent._set_dark_mode_enabled(self._original_dark_mode, persist=True)
        super().reject()

    @Slot()
    def _test_tns_credentials(self) -> None:
        api_key = self.tns_api_edit.text().strip()
        bot_id_raw = self.tns_bot_id_edit.text().strip()
        bot_name = self.tns_bot_name_edit.text().strip()
        if not api_key or not bot_id_raw or not bot_name:
            QMessageBox.warning(
                self,
                "Missing TNS Fields",
                "Fill TNS API key, Bot ID and Bot name before testing.",
            )
            return
        try:
            bot_id = int(bot_id_raw)
        except ValueError:
            QMessageBox.warning(self, "Invalid Bot ID", "TNS Bot ID must be numeric.")
            return

        marker = _build_tns_marker(bot_id, bot_name)
        endpoint_key = _normalize_tns_endpoint_key(self.tns_endpoint_combo.currentData())
        api_base = _tns_api_base_url(endpoint_key).rstrip("/")
        body = urlencode(
            {
                "api_key": api_key,
                "data": json.dumps({"objname": "2023ixf"}),
            }
        ).encode("utf-8")
        req = Request(
            f"{api_base}/get/search",
            data=body,
            headers={
                "User-Agent": marker,
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            try:
                with urlopen(req, timeout=20) as resp:
                    raw = resp.read().decode("utf-8", errors="replace")
            except HTTPError as exc:
                if exc.code in {401, 403}:
                    QMessageBox.critical(
                        self,
                        "TNS Unauthorized",
                        f"Unauthorized (401/403) on {endpoint_key}. Check API key, bot ID/name and bot permissions.",
                    )
                    return
                detail = ""
                try:
                    detail = exc.read().decode("utf-8", errors="ignore").strip()
                except Exception:
                    detail = ""
                msg = f"TNS request failed ({exc.code})."
                if detail:
                    msg = f"{msg}\n\n{detail[:280]}"
                QMessageBox.critical(self, "TNS Error", msg)
                return
            except Exception as exc:
                QMessageBox.critical(self, "TNS Error", f"Request failed: {exc}")
                return

            try:
                payload = json.loads(raw)
            except Exception:
                QMessageBox.warning(
                    self,
                    "TNS Response",
                    f"Received response from {endpoint_key}, but it was not valid JSON. Credentials may still be valid.",
                )
                return

            id_code = payload.get("id_code") if isinstance(payload, dict) else None
            id_msg = str(payload.get("id_message", "")).strip() if isinstance(payload, dict) else ""
            if id_code in {401, 403} or "unauthorized" in id_msg.lower():
                QMessageBox.critical(
                    self,
                    "TNS Unauthorized",
                    f"TNS rejected credentials on {endpoint_key}: {id_msg or id_code}",
                )
                return

            QMessageBox.information(
                self,
                "TNS OK",
                f"Credentials look valid on {endpoint_key}.\n{id_msg or 'Authentication passed.'}",
            )
        finally:
            QApplication.restoreOverrideCursor()


__all__ = ["GeneralSettingsDialog", "TableSettingsDialog"]
