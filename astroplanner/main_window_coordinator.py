from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

from PySide6.QtCore import QDate, QTimer, Qt
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import QApplication, QLabel, QProgressBar

from astroplanner.ai import LLMConfig
from astroplanner.i18n import (
    current_language,
    resolve_language_choice,
    set_current_language,
    set_translated_text,
    translate_text,
)
from astroplanner.models import Site
from astroplanner.preview_coordinator import (
    CUTOUT_DEFAULT_FOV_ARCMIN,
    CUTOUT_DEFAULT_SIZE_PX,
    CUTOUT_DEFAULT_SURVEY_KEY,
    CUTOUT_DEFAULT_VIEW_KEY,
    _normalize_cutout_survey_key,
    _normalize_cutout_view_key,
    _sanitize_cutout_fov_arcmin,
    _sanitize_cutout_size_px,
)
from astroplanner.theme import DEFAULT_DARK_MODE, DEFAULT_UI_THEME, normalize_theme_key
from astroplanner.ui.targets import (
    TargetTableModel,
    _VALID_TABLE_COLOR_MODES,
    _normalize_table_color_mode,
)
from astroplanner.ui.theme_utils import _sanitize_ui_font_size
from astroplanner.weather import WeatherLiveWorker

if TYPE_CHECKING:
    from astro_planner import MainWindow


logger = logging.getLogger(__name__)


class MainWindowCoordinator:
    """Own shell-level actions, status bar, settings, and startup glue."""

    def __init__(self, planner: "MainWindow") -> None:
        object.__setattr__(self, "_planner", planner)

    def __getattr__(self, name: str):
        return getattr(self._planner, name)

    def __setattr__(self, name: str, value: object) -> None:
        if name == "_planner":
            object.__setattr__(self, name, value)
            return
        setattr(self._planner, name, value)

    def setup_status_bar_and_startup_hooks(self) -> None:
        self.status_filters_label = QLabel("Filters: 0")
        self.status_bhtom_label = QLabel("BHTOM: idle")
        self.status_bhtom_progress = QProgressBar(self._planner)
        self.status_bhtom_progress.setMinimumWidth(96)
        self.status_bhtom_progress.setMaximumWidth(140)
        self.status_bhtom_progress.setTextVisible(False)
        self.status_bhtom_progress.setRange(0, 0)
        self.status_bhtom_progress.hide()
        self.status_aladin_label = QLabel("Aladin: idle")
        self.status_aladin_progress = QProgressBar(self._planner)
        self.status_aladin_progress.setMinimumWidth(96)
        self.status_aladin_progress.setMaximumWidth(140)
        self.status_aladin_progress.setTextVisible(False)
        self.status_aladin_progress.setRange(0, 0)
        self.status_aladin_progress.hide()
        self.status_finder_label = QLabel("Finder: idle")
        self.status_finder_progress = QProgressBar(self._planner)
        self.status_finder_progress.setMinimumWidth(96)
        self.status_finder_progress.setMaximumWidth(140)
        self.status_finder_progress.setTextVisible(False)
        self.status_finder_progress.setRange(0, 0)
        self.status_finder_progress.hide()
        self.status_calc_label = QLabel("Last calc: -")
        self.statusBar().addPermanentWidget(self.status_filters_label)
        self.statusBar().addPermanentWidget(self.status_bhtom_label)
        self.statusBar().addPermanentWidget(self.status_bhtom_progress)
        self.statusBar().addPermanentWidget(self.status_aladin_label)
        self.statusBar().addPermanentWidget(self.status_aladin_progress)
        self.statusBar().addPermanentWidget(self.status_finder_label)
        self.statusBar().addPermanentWidget(self.status_finder_progress)
        self.statusBar().addPermanentWidget(self.status_calc_label)

        app = QApplication.instance()
        if app:
            app.aboutToQuit.connect(self._cleanup_threads)

        self.clock_worker = getattr(self, "clock_worker", None)
        if (
            self.table_model.site
            and self.clock_worker is None
            and not getattr(self, "_obs_change_finalize_pending", False)
            and not getattr(self, "_startup_restore_pending", False)
        ):
            self._start_clock_worker()
        self._clock_timer = QTimer(self._planner)
        self._clock_timer.timeout.connect(self._update_clock)
        self._clock_timer.start(1000)
        self._update_status_bar()
        self._update_selected_details()
        QTimer.singleShot(0, self._run_startup_sequence)

    def _build_actions(self) -> None:  # noqa: D401
        """Create shortcuts and populate the menu bar."""
        # ----- Actions -----
        self.load_act = QAction("Load targets…", self._planner, shortcut=QKeySequence("Ctrl+L"))
        self.load_act.triggered.connect(self._load_targets)

        self.open_saved_plan_act = QAction("Open saved plan…", self._planner, shortcut=QKeySequence("Ctrl+Shift+L"))
        self.open_saved_plan_act.triggered.connect(self._open_saved_plan)
        self.import_plan_json_act = QAction("Import plan JSON…", self._planner)
        self.import_plan_json_act.triggered.connect(self._import_plan_json)
        self.save_plan_as_act = QAction("Save plan as…", self._planner, shortcut=QKeySequence("Ctrl+Shift+S"))
        self.save_plan_as_act.triggered.connect(self._save_plan_as)

        self.add_act = QAction("Add target…", self._planner, shortcut=QKeySequence("Ctrl+N"))
        self.add_act.triggered.connect(self._add_target_dialog)

        self.toggle_obs_act = QAction("Toggle observed", self._planner, shortcut=QKeySequence("Ctrl+Shift+O"))
        self.toggle_obs_act.triggered.connect(self._toggle_observed_selected)

        self.exp_act = QAction("Export plan…", self._planner, shortcut=QKeySequence("Ctrl+E"))
        self.exp_act.triggered.connect(self._export_plan)
        self.seestar_session_act = QAction("Seestar Session…", self._planner)
        self.seestar_session_act.triggered.connect(self._open_seestar_session)

        self.dark_act = QAction("Toggle Dark/Light Mode", self._planner, shortcut=QKeySequence("Ctrl+D"))
        self.dark_act.setCheckable(True)
        self.dark_act.setChecked(bool(getattr(self, "_dark_enabled", False)))
        self.dark_act.setShortcutContext(Qt.ApplicationShortcut)
        self.dark_act.toggled.connect(lambda checked: self._set_dark_mode_enabled(bool(checked), persist=True))
        self.quit_act = QAction("Quit", self._planner, shortcut=QKeySequence.Quit)
        self.quit_act.setMenuRole(QAction.QuitRole)
        self.quit_act.setShortcutContext(Qt.ApplicationShortcut)
        self.quit_act.triggered.connect(self.close)
        self.ai_describe_act = QAction("Describe selected object", self._planner, shortcut=QKeySequence("Ctrl+I"))
        self.ai_describe_act.triggered.connect(self._ai_describe_target)
        self.ai_suggest_act = QAction("Suggest targets for tonight", self._planner, shortcut=QKeySequence("Ctrl+Shift+I"))
        self.ai_suggest_act.triggered.connect(self._ai_suggest_targets)
        self.ai_suggest_act.setToolTip(
            "Check cached BHTOM targets first (TTL 1h). "
            "A fresh fetch starts only when the cache is missing or expired."
        )
        self.ai_suggest_act.setStatusTip(self.ai_suggest_act.toolTip())
        self.weather_act = QAction("Weather Window…", self._planner)
        self.weather_act.triggered.connect(self._open_weather_window)
        self.ai_toggle_panel_act = QAction("Toggle AI window", self._planner)
        self.ai_toggle_panel_act.triggered.connect(
            lambda: self.ai_toggle_btn.setChecked(not self.ai_toggle_btn.isChecked())
        )
        self.ai_warmup_act = QAction("Warm up LLM", self._planner)
        self.ai_warmup_act.triggered.connect(self._warmup_llm_manual)
        self.ai_clear_chat_act = QAction("Clear AI chat", self._planner)
        self.ai_clear_chat_act.triggered.connect(self._clear_ai_messages)
        self.obs_history_act = QAction("Observation History…", self._planner)
        self.obs_history_act.triggered.connect(self._show_observation_history)
        self.ai_focus_input_act = QAction("Focus AI input", self._planner)
        self.ai_focus_input_act.triggered.connect(self._focus_ai_input)
        self.ai_settings_act = QAction("AI Settings…", self._planner)
        self.ai_settings_act.triggered.connect(self._open_ai_settings)

        self.view_obs_preset_act = QAction("Observation Columns", self._planner, checkable=True)
        self.view_obs_preset_act.triggered.connect(lambda: self._apply_column_preset("observation"))
        self.view_full_preset_act = QAction("Full Columns", self._planner, checkable=True)
        self.view_full_preset_act.triggered.connect(lambda: self._apply_column_preset("full"))
        preset_group = QActionGroup(self._planner)
        preset_group.setExclusive(True)
        preset_group.addAction(self.view_obs_preset_act)
        preset_group.addAction(self.view_full_preset_act)
        current_preset = self.settings.value("table/viewPreset", "full", type=str)
        if current_preset == "full":
            self.view_full_preset_act.setChecked(True)
        else:
            self.view_obs_preset_act.setChecked(True)

        # Make shortcuts work even if the action isn't in a visible menu
        for act in (
            self.load_act,
            self.open_saved_plan_act,
            self.import_plan_json_act,
            self.save_plan_as_act,
            self.add_act,
            self.toggle_obs_act,
            self.exp_act,
            self.seestar_session_act,
            self.dark_act,
            self.quit_act,
            self.ai_describe_act,
            self.ai_suggest_act,
            self.ai_warmup_act,
            self.ai_clear_chat_act,
            self.ai_focus_input_act,
            self.ai_settings_act,
            self.weather_act,
            self.obs_history_act,
        ):
            self.addAction(act)

        # ----- Menu bar -----
        menubar = self.menuBar()
        if sys.platform == "darwin":
            menubar.setNativeMenuBar(True)
        menubar.setVisible(True)
        file_menu = menubar.addMenu("&File")
        file_menu.addAction(self.load_act)
        file_menu.addAction(self.open_saved_plan_act)
        file_menu.addAction(self.import_plan_json_act)
        file_menu.addAction(self.save_plan_as_act)
        file_menu.addAction(self.exp_act)
        file_menu.addAction(self.seestar_session_act)
        file_menu.addSeparator()
        file_menu.addAction(self.dark_act)
        file_menu.addSeparator()
        file_menu.addAction(self.quit_act)

        target_menu = menubar.addMenu("&Targets")
        target_menu.addAction(self.add_act)
        target_menu.addAction(self.ai_suggest_act)
        target_menu.addAction(self.toggle_obs_act)

        view_menu = menubar.addMenu("&View")
        view_menu.addAction(self.view_obs_preset_act)
        view_menu.addAction(self.view_full_preset_act)
        view_menu.addSeparator()
        view_menu.addAction(self.weather_act)
        view_menu.addAction(self.obs_history_act)

        ai_menu = menubar.addMenu("&AI")
        ai_menu.addAction(self.ai_describe_act)
        ai_menu.addSeparator()
        ai_menu.addAction(self.ai_focus_input_act)
        ai_menu.addAction(self.ai_toggle_panel_act)
        ai_menu.addAction(self.ai_warmup_act)
        ai_menu.addAction(self.ai_clear_chat_act)
        ai_menu.addSeparator()
        ai_menu.addAction(self.ai_settings_act)

        # Settings menu
        settings_menu = menubar.addMenu("&Settings")
        self.gen_settings_act = QAction("General Settings…", self._planner)
        self.gen_settings_act.setMenuRole(QAction.NoRole)
        self.gen_settings_act.setShortcuts([QKeySequence("Ctrl+,"), QKeySequence("Ctrl+;")])
        self.gen_settings_act.setShortcutContext(Qt.ApplicationShortcut)
        self.gen_settings_act.triggered.connect(self.open_general_settings)
        settings_menu.addAction(self.gen_settings_act)
        self.tbl_settings_act = QAction("Table Settings…", self._planner)
        self.tbl_settings_act.setMenuRole(QAction.NoRole)
        self.tbl_settings_act.triggered.connect(self.open_table_settings)
        settings_menu.addAction(self.tbl_settings_act)
        self.obs_manager_act = QAction("Observatory Manager…", self._planner)
        self.obs_manager_act.setMenuRole(QAction.NoRole)
        self.obs_manager_act.triggered.connect(self._open_observatory_manager)
        settings_menu.addAction(self.obs_manager_act)

        self.addAction(self.gen_settings_act)
        self.addAction(self.tbl_settings_act)
        self.addAction(self.obs_manager_act)

    def _load_settings(self):
        """Load saved settings and apply both Table and General."""
        self._apply_table_settings()
        # Apply default sort column
        val = self.settings.value("table/defaultSortColumn", TargetTableModel.COL_SCORE)
        try:
            default_sort = int(val)
        except (TypeError, ValueError):
            default_sort = TargetTableModel.COL_SCORE
        # Validate index
        ncols = self.table_model.columnCount()
        if 0 <= default_sort < ncols:
            self.table_view.sortByColumn(default_sort, Qt.AscendingOrder)
        self._apply_general_settings()

    def _migrate_table_settings_schema(self) -> None:
        version = self.settings.value("table/columnSchemaVersion", 0, type=int)
        if version < 2:
            old_column_count = 12
            for idx in range(old_column_count - 1, -1, -1):
                old_key = f"table/col{idx}"
                if not self.settings.contains(old_key):
                    continue
                self.settings.setValue(f"table/col{idx + 1}", self.settings.value(old_key, type=bool))
            self.settings.setValue("table/col0", True)

            if self.settings.contains("table/defaultSortColumn"):
                try:
                    old_sort = int(self.settings.value("table/defaultSortColumn", TargetTableModel.COL_SCORE))
                except (TypeError, ValueError):
                    old_sort = TargetTableModel.COL_SCORE
                self.settings.setValue(
                    "table/defaultSortColumn",
                    min(old_sort + 1, TargetTableModel.COL_ACTIONS),
                )
            version = 2

        if version < 3:
            # Insert the new Last Mag/Mag column at index 10.
            old_column_count = 13
            inserted_col = TargetTableModel.COL_MAG
            for idx in range(old_column_count - 1, inserted_col - 1, -1):
                old_key = f"table/col{idx}"
                if not self.settings.contains(old_key):
                    continue
                self.settings.setValue(f"table/col{idx + 1}", self.settings.value(old_key, type=bool))
            self.settings.setValue(f"table/col{inserted_col}", True)

            if self.settings.contains("table/defaultSortColumn"):
                try:
                    old_sort = int(self.settings.value("table/defaultSortColumn", TargetTableModel.COL_SCORE))
                except (TypeError, ValueError):
                    old_sort = TargetTableModel.COL_SCORE
                if old_sort >= inserted_col:
                    old_sort += 1
                self.settings.setValue(
                    "table/defaultSortColumn",
                    min(old_sort, TargetTableModel.COL_ACTIONS),
                )
            version = 3

        if version < 4:
            color_mode = _normalize_table_color_mode(
                self.settings.value("table/colorMode", "", type=str),
                default="",
            )
            if color_mode not in _VALID_TABLE_COLOR_MODES:
                self.settings.setValue("table/colorMode", "background")
            version = 4

        if version < 5:
            color_mode = _normalize_table_color_mode(
                self.settings.value("table/colorMode", "", type=str),
                default="background",
            )
            if color_mode not in _VALID_TABLE_COLOR_MODES:
                self.settings.setValue("table/colorMode", "background")
            version = 5

        self.settings.setValue("table/columnSchemaVersion", version)

    def _apply_general_settings(self):
        """Apply default site."""
        s = self.settings
        prev_show_sun_path = bool(getattr(self, "show_sun_path", True))
        prev_show_moon_path = bool(getattr(self, "show_moon_path", True))
        prev_show_obj_path = bool(getattr(self, "show_obj_path", True))
        prev_dark_enabled = bool(getattr(self, "_dark_enabled", DEFAULT_DARK_MODE))
        prev_theme_name = str(getattr(self, "_theme_name", DEFAULT_UI_THEME))
        prev_accent_secondary = str(getattr(self, "_accent_secondary_override", "") or "")
        prev_ui_font_size = int(getattr(self, "_ui_font_size", 11))
        prev_color_blind_mode = bool(getattr(self, "color_blind_mode", False))
        prev_radar_sweep_enabled = bool(getattr(self, "_radar_sweep_enabled", False))
        prev_radar_sweep_speed = int(getattr(self, "_radar_sweep_speed", 140))
        prev_ui_language_choice = str(getattr(self, "_ui_language_choice", "system") or "system")
        prev_ui_language = str(getattr(self, "_ui_language", current_language()) or current_language())
        self.show_sun_path  = self.settings.value("general/showSunPath",  True, type=bool)
        self.show_moon_path = self.settings.value("general/showMoonPath", True, type=bool)
        self.show_obj_path  = self.settings.value("general/showObjPath",  True, type=bool)
        self._dark_enabled = self.settings.value("general/darkMode", DEFAULT_DARK_MODE, type=bool)
        self._theme_name = normalize_theme_key(self.settings.value("general/uiTheme", DEFAULT_UI_THEME, type=str))
        self._accent_secondary_override = self._load_accent_secondary_override(self._theme_name)
        self._ui_font_size = _sanitize_ui_font_size(self.settings.value("general/uiFontSize", 11, type=int))
        self._ui_language_choice = str(self.settings.value("general/uiLanguage", "system", type=str) or "system")
        self._ui_language = resolve_language_choice(
            self._ui_language_choice,
            system_locale=getattr(self, "_system_locale_name", ""),
        )
        set_current_language(self._ui_language, system_locale=getattr(self, "_system_locale_name", ""))
        self.color_blind_mode = self.settings.value("general/colorBlindMode", False, type=bool)
        self._radar_sweep_enabled = self.settings.value("general/radarSweepAnimation", False, type=bool)
        self._radar_sweep_speed = max(40, min(260, self.settings.value("general/radarSweepSpeed", 140, type=int)))
        new_cutout_view_key = _normalize_cutout_view_key(
            self.settings.value("general/cutoutView", CUTOUT_DEFAULT_VIEW_KEY, type=str)
        )
        new_cutout_survey_key = _normalize_cutout_survey_key(
            self.settings.value("general/cutoutSurvey", CUTOUT_DEFAULT_SURVEY_KEY, type=str)
        )
        new_cutout_fov_arcmin = _sanitize_cutout_fov_arcmin(
            self.settings.value("general/cutoutFovArcmin", CUTOUT_DEFAULT_FOV_ARCMIN, type=int)
        )
        new_cutout_size_px = _sanitize_cutout_size_px(
            self.settings.value("general/cutoutSizePx", CUTOUT_DEFAULT_SIZE_PX, type=int)
        )
        view_changed = new_cutout_view_key != self._cutout_view_key
        cutout_changed = (
            new_cutout_survey_key != self._cutout_survey_key
            or new_cutout_fov_arcmin != self._cutout_fov_arcmin
            or new_cutout_size_px != self._cutout_size_px
        )
        self._cutout_survey_key = new_cutout_survey_key
        self._cutout_fov_arcmin = new_cutout_fov_arcmin
        self._cutout_size_px = new_cutout_size_px
        self._cutout_view_key = new_cutout_view_key
        self.llm_config.url = (
            self.settings.value("llm/serverUrl", LLMConfig.DEFAULT_URL, type=str)
            or LLMConfig.DEFAULT_URL
        ).strip().rstrip("/") or LLMConfig.DEFAULT_URL
        self.llm_config.model = (
            self.settings.value("llm/model", LLMConfig.DEFAULT_MODEL, type=str)
            or LLMConfig.DEFAULT_MODEL
        ).strip() or LLMConfig.DEFAULT_MODEL
        self.llm_config.timeout_s = max(
            15,
            int(self.settings.value("llm/timeoutSec", LLMConfig.DEFAULT_TIMEOUT_S, type=int)),
        )
        self.llm_config.max_tokens = max(
            32,
            int(self.settings.value("llm/maxTokens", LLMConfig.DEFAULT_MAX_TOKENS, type=int)),
        )
        self.llm_config.enable_thinking = bool(
            self.settings.value("llm/enableThinking", LLMConfig.DEFAULT_ENABLE_THINKING, type=bool)
        )
        self.llm_config.enable_chat_memory = bool(
            self.settings.value("llm/enableChatMemory", LLMConfig.DEFAULT_ENABLE_CHAT_MEMORY, type=bool)
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
        style_changed = any(
            (
                prev_dark_enabled != self._dark_enabled,
                prev_theme_name != self._theme_name,
                prev_accent_secondary != self._accent_secondary_override,
                prev_ui_font_size != self._ui_font_size,
                prev_color_blind_mode != self.color_blind_mode,
            )
        )
        polar_visuals_changed = any(
            (
                prev_show_sun_path != self.show_sun_path,
                prev_show_moon_path != self.show_moon_path,
                prev_show_obj_path != self.show_obj_path,
                prev_radar_sweep_enabled != self._radar_sweep_enabled,
                prev_radar_sweep_speed != self._radar_sweep_speed,
            )
        )
        language_changed = any(
            (
                prev_ui_language_choice != self._ui_language_choice,
                prev_ui_language != self._ui_language,
            )
        )
        self._refresh_ai_warm_indicator()
        self._refresh_ai_context_hint()
        self._apply_ai_chat_font()
        self._render_ai_messages()

        if hasattr(self, "min_moon_sep_spin"):
            self.min_moon_sep_spin.blockSignals(True)
            self.min_moon_sep_spin.setValue(s.value("general/minMoonSep", 0, type=int))
            self.min_moon_sep_spin.blockSignals(False)
        if hasattr(self, "min_score_spin"):
            self.min_score_spin.blockSignals(True)
            self.min_score_spin.setValue(s.value("general/minScore", 0, type=int))
            self.min_score_spin.blockSignals(False)
        if hasattr(self, "hide_observed_chk"):
            self.hide_observed_chk.blockSignals(True)
            self.hide_observed_chk.setChecked(s.value("general/hideObserved", False, type=bool))
            self.hide_observed_chk.blockSignals(False)
        if hasattr(self, "sun_alt_limit_spin"):
            self.sun_alt_limit_spin.blockSignals(True)
            self.sun_alt_limit_spin.setValue(s.value("general/sunAltLimit", -10, type=int))
            self.sun_alt_limit_spin.blockSignals(False)
        if hasattr(self, "cutout_tabs"):
            tab_idx = 1 if self._cutout_view_key == "finderchart" else 0
            self.cutout_tabs.blockSignals(True)
            self.cutout_tabs.setCurrentIndex(tab_idx)
            self.cutout_tabs.blockSignals(False)
        if cutout_changed and hasattr(self, "cutout_image_label"):
            self._clear_cutout_cache()
        if (cutout_changed or view_changed) and hasattr(self, "cutout_image_label"):
            self._update_cutout_preview_for_target(self._selected_target_or_none())
        self._update_quick_targets_button_tooltip()
        self._update_seestar_button_tooltip()

        # If a payload is already cached, refresh the polar plot right away
        if getattr(self, "last_payload", None) and polar_visuals_changed:
            self._update_polar_positions(self.last_payload)
        ds = s.value("general/defaultSite", type=str)
        if ds in self.observatories and self.obs_combo.currentText() != ds:
            self.obs_combo.setCurrentText(ds)
        if style_changed:
            self._apply_table_settings()
            self._apply_styles()
        if language_changed:
            self._apply_localization()
        self._update_radar_sweep_state()
        self._update_status_bar()
        self._refresh_weather_window_context()

    def _update_status_bar(self):
        if not hasattr(self, "status_filters_label") or not hasattr(self, "status_calc_label"):
            return
        parts: list[str] = []
        if hasattr(self, "min_score_spin") and self.min_score_spin.value() > 0:
            parts.append(f"score≥{self.min_score_spin.value()}")
        if hasattr(self, "min_moon_sep_spin") and self.min_moon_sep_spin.value() > 0:
            parts.append(f"moon≥{self.min_moon_sep_spin.value()}°")
        if hasattr(self, "hide_observed_chk") and self.hide_observed_chk.isChecked():
            parts.append(translate_text("hide observed", current_language()))
        if hasattr(self, "sun_alt_limit_spin"):
            parts.append(f"sun≤{self._sun_alt_limit():.0f}°")

        total = len(self.targets)
        visible = sum(1 for flag in self.table_model.row_enabled if flag) if self.table_model.row_enabled else total
        active_filters = ", ".join(parts) if parts else translate_text("none", current_language())
        set_translated_text(
            self.status_filters_label,
            f"Filters: {active_filters} | Visible: {visible}/{total}",
            current_language(),
        )

        if self.worker is not None and self.worker.isRunning():
            set_translated_text(self.status_calc_label, "Last calc: running...", current_language())
            return
        stats = self._last_calc_stats
        if stats.total_targets > 0:
            set_translated_text(
                self.status_calc_label,
                f"Last calc: {stats.duration_s:.2f}s | {stats.visible_targets}/{stats.total_targets}",
                current_language(),
            )
        else:
            set_translated_text(self.status_calc_label, "Last calc: -", current_language())

    def _preload_cached_runtime_state_on_startup(self) -> None:
        self._prefetch_bhtom_observatory_presets_on_startup()
        self._prefetch_bhtom_candidates_on_startup()
        self._preload_cached_preview_images_on_startup()
        selected_target = self._selected_target_or_none()
        if selected_target is not None:
            self._update_cutout_preview_for_target(selected_target)
            self._prefetch_finder_charts_for_all_targets(prioritize=selected_target)
        else:
            self._prefetch_cutouts_for_all_targets()
            self._prefetch_finder_charts_for_all_targets()
        self._prefetch_weather_on_startup()

    def _preload_cached_preview_images_on_startup(self) -> None:
        selected_target = self._selected_target_or_none()
        if selected_target is None or not hasattr(self, "aladin_image_label"):
            return
        render_w, render_h = self._cutout_render_dimensions_px(getattr(self, "aladin_image_label", None))
        loaded_cutouts = 0
        loaded_finders = 0
        key = self._cutout_key_for_target(selected_target, render_w, render_h)
        cutout = self._load_pixmap_from_storage_cache("cutout_preview", key)
        if cutout is not None and not cutout.isNull():
            self._cache_cutout_pixmap(key, cutout, persist=False)
            loaded_cutouts += 1
        finder = self._load_pixmap_from_storage_cache("finder_preview", key)
        if finder is not None and not finder.isNull():
            self._cache_finder_pixmap(key, finder, persist=False)
            loaded_finders += 1
        if loaded_cutouts:
            logger.info("Preloaded %d cached Aladin preview into RAM.", loaded_cutouts)
        if loaded_finders:
            logger.info("Preloaded %d cached finder preview into RAM.", loaded_finders)

    def _run_startup_sequence(self) -> None:
        if getattr(self, "_shutting_down", False):
            return
        if hasattr(self, "_replot_timer") and self._replot_timer.isActive():
            self._replot_timer.stop()
        self._defer_startup_preview_updates = True
        restored_workspace = False
        try:
            restored_workspace = self._restore_workspace_on_startup(defer_visual_refresh=True)
        finally:
            self._defer_startup_preview_updates = False
            self._startup_restore_pending = False
        self._validate_site_inputs()
        current_site_name = self.obs_combo.currentText().strip() if hasattr(self, "obs_combo") else ""
        if current_site_name and not getattr(self, "_obs_change_finalize_pending", False):
            self._finalize_observatory_change(current_site_name)
        if restored_workspace:
            QTimer.singleShot(0, self._update_selected_details)
        if not restored_workspace:
            self._replot_timer.start()
        QTimer.singleShot(120, self._preload_cached_runtime_state_on_startup)
        QTimer.singleShot(900, self._warmup_llm_on_startup)

    def _prefetch_weather_on_startup(self) -> None:
        if getattr(self, "_shutting_down", False):
            return
        site = self._current_site_for_weather()
        if not isinstance(site, Site):
            return
        existing = getattr(self, "_startup_weather_worker", None)
        if isinstance(existing, WeatherLiveWorker) and existing.isRunning():
            return

        cloud_month_mode = str(
            self.settings.value("weather/cloudMapMonthMode", "session_month", type=str) or "session_month"
        ).strip().lower()
        cloud_month = int(QDate.currentDate().month()) if cloud_month_mode == "current_month" else int(self.date_edit.date().month())
        worker = WeatherLiveWorker(
            lat=float(site.latitude),
            lon=float(site.longitude),
            elev=float(site.elevation),
            custom_conditions_url=str(getattr(site, "custom_conditions_url", "") or "").strip(),
            cloud_map_source=str(self.settings.value("weather/cloudMapSource", "earthenv", type=str) or "earthenv").strip(),
            cloud_map_month=cloud_month,
            force_refresh=False,
            include_cloud_map=True,
            include_satellite=True,
            storage=getattr(self, "app_storage", None),
            parent=self._planner,
        )
        worker.progress.connect(self._on_startup_weather_progress)
        worker.completed.connect(self._on_startup_weather_completed)
        worker.finished.connect(lambda w=worker: self._on_startup_weather_finished(w))
        worker.finished.connect(worker.deleteLater)
        self._startup_weather_worker = worker
        worker.start()

    def _on_startup_weather_progress(self, status: str, _step: int, _total: int) -> None:
        logger.info("Startup weather refresh: %s", str(status or "").strip() or "running")

    def _on_startup_weather_completed(self, payload: dict) -> None:
        if not isinstance(payload, dict):
            return
        sections = payload.get("sections")
        if isinstance(sections, dict):
            forecast = sections.get("forecast") if isinstance(sections.get("forecast"), dict) else {}
            conditions = sections.get("conditions") if isinstance(sections.get("conditions"), dict) else {}
            logger.info(
                "Startup weather refresh ready (forecast=%s, conditions=%s).",
                str(forecast.get("status", "unknown") or "unknown"),
                str(conditions.get("status", "unknown") or "unknown"),
            )

    def _on_startup_weather_finished(self, worker: WeatherLiveWorker) -> None:
        if getattr(self, "_startup_weather_worker", None) is worker:
            self._startup_weather_worker = None
