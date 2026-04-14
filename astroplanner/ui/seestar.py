from __future__ import annotations

import logging
from typing import Callable

from astropy import units as u
from astropy.coordinates import SkyCoord
from PySide6.QtCore import QSignalBlocker, Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableView,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from astroplanner.i18n import current_language, localize_widget_tree, translate_text
from astroplanner.seestar import (
    SEESTAR_ALP_AF_MODE_OFF,
    SEESTAR_ALP_AF_MODE_PER_RUN,
    SEESTAR_ALP_AF_MODE_PER_TARGET,
    SEESTAR_ALP_DEFAULT_AUTOFOCUS_TRY_COUNT,
    SEESTAR_ALP_DEFAULT_BASE_URL,
    SEESTAR_ALP_DEFAULT_CAPTURE_FLATS,
    SEESTAR_ALP_DEFAULT_CLIENT_ID,
    SEESTAR_ALP_DEFAULT_DEVICE_NUM,
    SEESTAR_ALP_DEFAULT_DEW_HEATER_VALUE,
    SEESTAR_ALP_DEFAULT_FLATS_WAIT_S,
    SEESTAR_ALP_DEFAULT_GAIN,
    SEESTAR_ALP_DEFAULT_HONOR_QUEUE_TIMES,
    SEESTAR_ALP_DEFAULT_NUM_TRIES,
    SEESTAR_ALP_DEFAULT_PANEL_OVERLAP_PERCENT,
    SEESTAR_ALP_DEFAULT_PARK_AFTER_SESSION,
    SEESTAR_ALP_DEFAULT_RETRY_WAIT_S,
    SEESTAR_ALP_DEFAULT_SCHEDULE_AUTOFOCUS,
    SEESTAR_ALP_DEFAULT_SCHEDULE_AUTOFOCUS_MODE,
    SEESTAR_ALP_DEFAULT_SHUTDOWN_AFTER_SESSION,
    SEESTAR_ALP_DEFAULT_STACK_EXPOSURE_MS,
    SEESTAR_ALP_DEFAULT_STARTUP_AUTO_FOCUS,
    SEESTAR_ALP_DEFAULT_STARTUP_DARK_FRAMES,
    SEESTAR_ALP_DEFAULT_STARTUP_POLAR_ALIGN,
    SEESTAR_ALP_DEFAULT_STARTUP_SEQUENCE,
    SEESTAR_ALP_DEFAULT_TIMEOUT_S,
    SEESTAR_ALP_DEFAULT_USE_AUTOFOCUS,
    SEESTAR_ALP_DEFAULT_WAIT_UNTIL_LOCAL_TIME,
    SEESTAR_ALP_LP_FILTER_AUTO,
    SEESTAR_ALP_LP_FILTER_OFF,
    SEESTAR_ALP_LP_FILTER_ON,
    SEESTAR_DEFAULT_BLOCK_MINUTES,
    SEESTAR_METHOD_ALP,
    SEESTAR_METHOD_GUIDED,
    SEESTAR_TEMPLATE_SCOPE_MULTI_TARGET,
    SEESTAR_TEMPLATE_SCOPE_SINGLE_TARGET,
    SeestarAlpConfig,
    SeestarSessionTemplate,
    SeestarTargetSessionItem,
    builtin_seestar_session_templates,
    normalize_seestar_alp_schedule_autofocus_mode,
    seestar_alp_schedule_autofocus_mode_label,
)
from astroplanner.ui.theme_utils import _style_dialog_button_box


logger = logging.getLogger(__name__)


def _normalize_catalog_token(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


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


class SeestarSessionPlanDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self._planner = parent
        self._settings = parent.settings
        self._storage = getattr(parent, "app_storage", None)
        self._builtin_session_templates = builtin_seestar_session_templates()
        self._user_session_templates = self._load_user_session_templates()
        self._current_settings_template = self._load_current_settings_template()
        self._target_plan_rows: list[dict[str, object]] = []

        self.setWindowTitle("Seestar Session Settings")
        self.setModal(True)
        screen = parent.screen() if isinstance(parent, QWidget) else QApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            start_rect = available.adjusted(2, 2, -2, -2)
            self.setGeometry(start_rect)
            self.resize(start_rect.size())
            self.setMinimumSize(min(1240, start_rect.width()), min(760, start_rect.height()))
        else:
            self.resize(1560, 920)
            self.setMinimumSize(1240, 760)

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(6)

        current_alp_url = str(
            self._settings.value("general/seestarAlpBaseUrl", SEESTAR_ALP_DEFAULT_BASE_URL, type=str)
            or SEESTAR_ALP_DEFAULT_BASE_URL
        ).strip()
        session_hint = QLabel(
            f"ALP: {current_alp_url.rstrip('/')}  |  "
            "Session settings are here; base URL, device and timeout stay in General Settings.",
            self,
        )
        session_hint.setObjectName("SectionHint")
        session_hint.setWordWrap(True)
        root.addWidget(session_hint)

        body_splitter = QSplitter(Qt.Vertical, self)
        body_splitter.setChildrenCollapsible(False)
        root.addWidget(body_splitter, 1)

        settings_scroll = QScrollArea(body_splitter)
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setFrameShape(QFrame.NoFrame)
        settings_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        settings_panel = QWidget(settings_scroll)
        settings_layout = QVBoxLayout(settings_panel)
        settings_layout.setContentsMargins(0, 0, 0, 0)
        settings_layout.setSpacing(6)

        def _make_card(parent: QWidget, title: str, hint: str = "") -> tuple[QFrame, QVBoxLayout]:
            card = QFrame(parent)
            card.setObjectName("InfoCard")
            layout = QVBoxLayout(card)
            layout.setContentsMargins(8, 8, 8, 8)
            layout.setSpacing(6)
            title_lbl = QLabel(title, card)
            title_lbl.setObjectName("SectionTitle")
            layout.addWidget(title_lbl)
            if hint:
                hint_lbl = QLabel(hint, card)
                hint_lbl.setObjectName("SectionHint")
                hint_lbl.setWordWrap(True)
                layout.addWidget(hint_lbl)
            return card, layout

        def _make_field_row(
            parent: QWidget,
            pairs: list[tuple[str, QWidget]],
            *,
            stretch_last: bool = True,
            label_width: int = 84,
        ) -> QWidget:
            row = QWidget(parent)
            layout = QHBoxLayout(row)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(8)
            for idx, (label_text, widget) in enumerate(pairs):
                lbl = QLabel(label_text, row)
                lbl.setMinimumWidth(label_width)
                layout.addWidget(lbl, 0)
                layout.addWidget(widget, 1 if stretch_last or idx < len(pairs) - 1 else 0)
            layout.addStretch(1)
            return row

        tabs = _configure_tab_widget(QTabWidget(settings_panel))
        settings_layout.addWidget(tabs, 1)
        self._session_tabs = tabs

        defaults_tab = QWidget(tabs)
        defaults_tab_layout = QVBoxLayout(defaults_tab)
        defaults_tab_layout.setContentsMargins(6, 6, 6, 6)
        defaults_tab_layout.setSpacing(8)

        self.seestar_method_combo = QComboBox(self)
        self.seestar_method_combo.addItem("App handoff", SEESTAR_METHOD_GUIDED)
        self.seestar_method_combo.addItem("ALP service", SEESTAR_METHOD_ALP)
        init_method = str(
            self._settings.value("general/seestarMethod", SEESTAR_METHOD_GUIDED, type=str) or SEESTAR_METHOD_GUIDED
        ).strip().lower()
        method_idx = self.seestar_method_combo.findData(init_method or SEESTAR_METHOD_GUIDED)
        if method_idx >= 0:
            self.seestar_method_combo.setCurrentIndex(method_idx)

        self.seestar_session_template_combo = QComboBox(self)
        self._populate_session_template_combo()
        template_row = QWidget(self)
        template_row_l = QHBoxLayout(template_row)
        template_row_l.setContentsMargins(0, 0, 0, 0)
        template_row_l.setSpacing(8)
        template_row_l.addWidget(self.seestar_session_template_combo, 1)
        self.seestar_session_template_save_btn = QPushButton("Save As…", template_row)
        self.seestar_session_template_save_btn.clicked.connect(self._save_session_template)
        template_row_l.addWidget(self.seestar_session_template_save_btn, 0)
        self.seestar_session_template_delete_btn = QPushButton("Delete", template_row)
        self.seestar_session_template_delete_btn.clicked.connect(self._delete_session_template)
        template_row_l.addWidget(self.seestar_session_template_delete_btn, 0)

        self.seestar_session_repeat_spin = QSpinBox(self)
        self.seestar_session_repeat_spin.setRange(1, 50)

        self.seestar_session_minutes_spin = QSpinBox(self)
        self.seestar_session_minutes_spin.setRange(1, 240)
        self.seestar_session_minutes_spin.setSingleStep(5)
        self.seestar_session_minutes_spin.setSuffix(" min")

        self.seestar_session_gap_spin = QSpinBox(self)
        self.seestar_session_gap_spin.setRange(0, 3600)
        self.seestar_session_gap_spin.setSingleStep(5)
        self.seestar_session_gap_spin.setSuffix(" s")

        self.seestar_alp_gain_spin = QSpinBox(self)
        self.seestar_alp_gain_spin.setRange(0, 500)

        self.seestar_alp_panel_overlap_spin = QSpinBox(self)
        self.seestar_alp_panel_overlap_spin.setRange(0, 95)
        self.seestar_alp_panel_overlap_spin.setSuffix(" %")
        self.seestar_alp_panel_overlap_spin.hide()

        self.seestar_alp_use_autofocus_chk = QCheckBox("Capture-job AF", self)
        self.seestar_alp_use_autofocus_chk.setToolTip(
            "Passes is_use_autofocus into each ALP capture target item. "
            "Can duplicate Schedule AF if both are enabled."
        )

        self.seestar_alp_num_tries_spin = QSpinBox(self)
        self.seestar_alp_num_tries_spin.setRange(1, 10)

        self.seestar_alp_retry_wait_spin = QSpinBox(self)
        self.seestar_alp_retry_wait_spin.setRange(0, 3600)
        self.seestar_alp_retry_wait_spin.setSingleStep(5)
        self.seestar_alp_retry_wait_spin.setSuffix(" s")

        self.seestar_alp_stack_exposure_spin = QSpinBox(self)
        self.seestar_alp_stack_exposure_spin.setRange(0, 600000)
        self.seestar_alp_stack_exposure_spin.setSingleStep(500)
        self.seestar_alp_stack_exposure_spin.setSuffix(" ms")
        self.seestar_alp_stack_exposure_spin.setSpecialValueText("Use ALP default")

        self.seestar_alp_lp_filter_combo = QComboBox(self)
        self.seestar_alp_lp_filter_combo.addItem("Auto", SEESTAR_ALP_LP_FILTER_AUTO)
        self.seestar_alp_lp_filter_combo.addItem("Force OFF", SEESTAR_ALP_LP_FILTER_OFF)
        self.seestar_alp_lp_filter_combo.addItem("Force ON", SEESTAR_ALP_LP_FILTER_ON)

        defaults_card, defaults_card_l = _make_card(defaults_tab, "Session")
        defaults_card_l.addWidget(_make_field_row(defaults_card, [("Backend", self.seestar_method_combo)], label_width=72))
        defaults_card_l.addWidget(_make_field_row(defaults_card, [("Template", template_row)], label_width=72))
        self.seestar_defaults_summary_label = QLabel(defaults_card)
        self.seestar_defaults_summary_label.setObjectName("SectionHint")
        self.seestar_defaults_summary_label.setWordWrap(True)
        defaults_card_l.addWidget(self.seestar_defaults_summary_label)
        defaults_tab_layout.addWidget(defaults_card)

        defaults_row = QWidget(defaults_tab)
        defaults_row_l = QHBoxLayout(defaults_row)
        defaults_row_l.setContentsMargins(0, 0, 0, 0)
        defaults_row_l.setSpacing(8)

        run_defaults_card, run_defaults_card_l = _make_card(defaults_row, "Runs")
        run_defaults_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        run_defaults_card_l.addWidget(
            _make_field_row(
                run_defaults_card,
                [
                    ("Runs", self.seestar_session_repeat_spin),
                    ("Min/run", self.seestar_session_minutes_spin),
                    ("Gap", self.seestar_session_gap_spin),
                ],
                label_width=60,
            )
        )
        defaults_row_l.addWidget(run_defaults_card, 1)

        capture_defaults_card, capture_defaults_card_l = _make_card(defaults_row, "Capture")
        capture_defaults_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        capture_defaults_card_l.addWidget(
            _make_field_row(
                capture_defaults_card,
                [
                    ("Exposure", self.seestar_alp_stack_exposure_spin),
                    ("LP", self.seestar_alp_lp_filter_combo),
                ],
                label_width=64,
            )
        )
        capture_defaults_card_l.addWidget(
            _make_field_row(
                capture_defaults_card,
                [
                    ("Gain", self.seestar_alp_gain_spin),
                    ("Retries", self.seestar_alp_num_tries_spin),
                    ("Wait", self.seestar_alp_retry_wait_spin),
                ],
                label_width=64,
                stretch_last=False,
            )
        )
        capture_defaults_card_l.addWidget(
            _make_field_row(
                capture_defaults_card,
                [("Focus", self.seestar_alp_use_autofocus_chk)],
                label_width=64,
                stretch_last=False,
            )
        )
        defaults_row_l.addWidget(capture_defaults_card, 1)
        defaults_tab_layout.addWidget(defaults_row)

        defaults_tab_layout.addStretch(1)
        tabs.addTab(defaults_tab, "Session")

        checklist_tab = QWidget(tabs)
        checklist_tab_layout = QVBoxLayout(checklist_tab)
        checklist_tab_layout.setContentsMargins(6, 6, 6, 6)
        checklist_tab_layout.setSpacing(8)

        checklist_splitter = QSplitter(Qt.Horizontal, checklist_tab)
        checklist_splitter.setChildrenCollapsible(False)
        checklist_tab_layout.addWidget(checklist_splitter, 1)

        notes_card, notes_card_l = _make_card(
            checklist_splitter,
            "Session Notes",
            "Shown later in the ALP dialog before upload or start.",
        )

        self.seestar_session_require_checklist_chk = QCheckBox(
            "Require checklist before push/start",
            notes_card,
        )
        notes_card_l.addWidget(self.seestar_session_require_checklist_chk)

        notes_label = QLabel("Notes", notes_card)
        notes_label.setObjectName("SectionHint")
        notes_card_l.addWidget(notes_label)

        self.seestar_session_notes_edit = QTextEdit(notes_card)
        self.seestar_session_notes_edit.setMinimumHeight(130)
        self.seestar_session_notes_edit.setPlaceholderText("Notes shown in the ALP dialog")
        notes_card_l.addWidget(self.seestar_session_notes_edit, 1)
        checklist_splitter.addWidget(notes_card)

        checklist_card, checklist_card_l = _make_card(
            checklist_splitter,
            "Science Checklist",
            "One item per line. Only shown when confirmation is required.",
        )

        checklist_label = QLabel("Checklist", checklist_card)
        checklist_label.setObjectName("SectionHint")
        checklist_card_l.addWidget(checklist_label)

        self.seestar_session_checklist_edit = QTextEdit(checklist_card)
        self.seestar_session_checklist_edit.setMinimumHeight(130)
        self.seestar_session_checklist_edit.setPlaceholderText("One checklist item per line")
        checklist_card_l.addWidget(self.seestar_session_checklist_edit, 1)
        checklist_splitter.addWidget(checklist_card)
        checklist_splitter.setStretchFactor(0, 1)
        checklist_splitter.setStretchFactor(1, 1)

        tabs.addTab(checklist_tab, "Notes")

        automation_tab = QWidget(tabs)
        automation_tab_layout = QVBoxLayout(automation_tab)
        automation_tab_layout.setContentsMargins(6, 6, 6, 6)
        automation_tab_layout.setSpacing(8)

        automation_row = QWidget(automation_tab)
        automation_row_l = QHBoxLayout(automation_row)
        automation_row_l.setContentsMargins(0, 0, 0, 0)
        automation_row_l.setSpacing(8)

        automation_left_card, automation_left_card_l = _make_card(automation_row, "Timing And Startup")
        automation_left_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        automation_left_form_widget = QWidget(automation_left_card)
        automation_left_form = QFormLayout(automation_left_form_widget)
        automation_left_form.setContentsMargins(0, 0, 0, 0)
        automation_left_form.setSpacing(6)
        automation_left_card_l.addWidget(automation_left_form_widget)

        self.seestar_alp_honor_queue_times_chk = QCheckBox("Use planned start times", automation_left_card)

        self.seestar_alp_wait_until_edit = QLineEdit(automation_left_card)
        self.seestar_alp_wait_until_edit.setPlaceholderText("HH:MM (optional)")

        self.seestar_alp_startup_enabled_chk = QCheckBox("Run startup first", automation_left_card)

        startup_row = QWidget(automation_left_card)
        startup_row_l = QHBoxLayout(startup_row)
        startup_row_l.setContentsMargins(0, 0, 0, 0)
        startup_row_l.setSpacing(8)
        startup_row_l.addWidget(self.seestar_alp_startup_enabled_chk, 0)
        self.seestar_alp_startup_polar_align_chk = QCheckBox("Polar align / 3PPA", startup_row)
        startup_row_l.addWidget(self.seestar_alp_startup_polar_align_chk, 0)
        self.seestar_alp_startup_autofocus_chk = QCheckBox("Startup autofocus", startup_row)
        startup_row_l.addWidget(self.seestar_alp_startup_autofocus_chk, 0)
        self.seestar_alp_startup_dark_frames_chk = QCheckBox("Dark frames", startup_row)
        startup_row_l.addWidget(self.seestar_alp_startup_dark_frames_chk, 0)
        startup_row_l.addStretch(1)

        self.seestar_alp_capture_flats_chk = QCheckBox("Blind flats before session", automation_left_card)
        self.seestar_alp_capture_flats_chk.setToolTip(
            "Only sends start_create_calib_frame to ALP. It does not measure ADU or tune flat exposure."
        )

        self.seestar_alp_flats_wait_spin = QSpinBox(automation_left_card)
        self.seestar_alp_flats_wait_spin.setRange(0, 3600)
        self.seestar_alp_flats_wait_spin.setSingleStep(10)
        self.seestar_alp_flats_wait_spin.setSuffix(" s")
        self.seestar_alp_flats_wait_spin.setToolTip(
            "Wait time after the blind ALP flat trigger. Use this only if you understand the flat routine timing."
        )

        timing_row = QWidget(automation_left_card)
        timing_row_l = QHBoxLayout(timing_row)
        timing_row_l.setContentsMargins(0, 0, 0, 0)
        timing_row_l.setSpacing(8)
        timing_row_l.addWidget(self.seestar_alp_honor_queue_times_chk, 0)
        timing_row_l.addWidget(QLabel("Not before", timing_row), 0)
        timing_row_l.addWidget(self.seestar_alp_wait_until_edit, 0)
        timing_row_l.addStretch(1)

        flats_row = QWidget(automation_left_card)
        flats_row_l = QHBoxLayout(flats_row)
        flats_row_l.setContentsMargins(0, 0, 0, 0)
        flats_row_l.setSpacing(8)
        flats_row_l.addWidget(self.seestar_alp_capture_flats_chk, 0)
        flats_row_l.addWidget(QLabel("Wait", flats_row), 0)
        flats_row_l.addWidget(self.seestar_alp_flats_wait_spin, 0)
        flats_row_l.addStretch(1)

        automation_left_form.addRow("Timing", timing_row)
        automation_left_form.addRow("Startup", startup_row)
        automation_left_form.addRow("Flats", flats_row)
        automation_row_l.addWidget(automation_left_card, 1)

        automation_right_card, automation_right_card_l = _make_card(automation_row, "Focus And Finish")
        automation_right_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        automation_right_form_widget = QWidget(automation_right_card)
        automation_right_form = QFormLayout(automation_right_form_widget)
        automation_right_form.setContentsMargins(0, 0, 0, 0)
        automation_right_form.setSpacing(6)
        automation_right_card_l.addWidget(automation_right_form_widget)

        focus_row = QWidget(automation_right_card)
        focus_row_l = QHBoxLayout(focus_row)
        focus_row_l.setContentsMargins(0, 0, 0, 0)
        focus_row_l.setSpacing(8)
        self.seestar_alp_schedule_af_mode_combo = QComboBox(focus_row)
        self.seestar_alp_schedule_af_mode_combo.addItem("Off", SEESTAR_ALP_AF_MODE_OFF)
        self.seestar_alp_schedule_af_mode_combo.addItem("Before each run", SEESTAR_ALP_AF_MODE_PER_RUN)
        self.seestar_alp_schedule_af_mode_combo.addItem("Once per target", SEESTAR_ALP_AF_MODE_PER_TARGET)
        focus_row_l.addWidget(self.seestar_alp_schedule_af_mode_combo, 0)
        self.seestar_alp_schedule_af_try_spin = QSpinBox(focus_row)
        self.seestar_alp_schedule_af_try_spin.setRange(1, 10)
        focus_row_l.addWidget(QLabel("Rounds", focus_row), 0)
        focus_row_l.addWidget(self.seestar_alp_schedule_af_try_spin, 0)
        focus_row_l.addStretch(1)

        dew_row = QWidget(automation_right_card)
        dew_row_l = QHBoxLayout(dew_row)
        dew_row_l.setContentsMargins(0, 0, 0, 0)
        dew_row_l.setSpacing(8)
        self.seestar_alp_dew_heater_chk = QCheckBox("Set dew heater", dew_row)
        dew_row_l.addWidget(self.seestar_alp_dew_heater_chk, 0)
        self.seestar_alp_dew_heater_spin = QSpinBox(dew_row)
        self.seestar_alp_dew_heater_spin.setRange(0, 100)
        self.seestar_alp_dew_heater_spin.setSuffix(" %")
        dew_row_l.addWidget(self.seestar_alp_dew_heater_spin, 0)
        dew_row_l.addStretch(1)

        end_row = QWidget(automation_right_card)
        end_row_l = QHBoxLayout(end_row)
        end_row_l.setContentsMargins(0, 0, 0, 0)
        end_row_l.setSpacing(8)
        self.seestar_alp_park_after_chk = QCheckBox("Park after session", end_row)
        end_row_l.addWidget(self.seestar_alp_park_after_chk, 0)
        self.seestar_alp_shutdown_after_chk = QCheckBox("Shutdown after session", end_row)
        end_row_l.addWidget(self.seestar_alp_shutdown_after_chk, 0)
        end_row_l.addStretch(1)
        automation_right_form.addRow("Schedule AF", focus_row)
        automation_right_form.addRow("Dew heater", dew_row)
        automation_right_form.addRow("Finish", end_row)
        automation_row_l.addWidget(automation_right_card, 1)

        automation_tab_layout.addWidget(automation_row, 0)
        automation_tab_layout.addStretch(1)
        tabs.addTab(automation_tab, "Automation")

        targets_panel = QWidget(body_splitter)
        targets_panel.setMinimumHeight(180)
        targets_layout = QVBoxLayout(targets_panel)
        targets_layout.setContentsMargins(0, 0, 0, 0)
        targets_layout.setSpacing(8)

        target_actions = QWidget(targets_panel)
        target_actions_l = QHBoxLayout(target_actions)
        target_actions_l.setContentsMargins(0, 0, 0, 0)
        target_actions_l.setSpacing(8)
        targets_hdr = QLabel(
            "Target Plan. Current Targets order; checked box = use defaults above.",
            target_actions,
        )
        targets_hdr.setObjectName("SectionHint")
        targets_hdr.setWordWrap(True)
        target_actions_l.addWidget(targets_hdr, 1)
        self.seestar_targets_enable_all_btn = QPushButton("Enable All", target_actions)
        self.seestar_targets_enable_all_btn.clicked.connect(lambda: self._set_all_target_rows_enabled(True))
        target_actions_l.addWidget(self.seestar_targets_enable_all_btn, 0)
        self.seestar_targets_disable_all_btn = QPushButton("Disable All", target_actions)
        self.seestar_targets_disable_all_btn.clicked.connect(lambda: self._set_all_target_rows_enabled(False))
        target_actions_l.addWidget(self.seestar_targets_disable_all_btn, 0)
        self.seestar_targets_defaults_btn = QPushButton("Defaults For All", target_actions)
        self.seestar_targets_defaults_btn.clicked.connect(self._reset_all_target_row_defaults)
        target_actions_l.addWidget(self.seestar_targets_defaults_btn, 0)
        target_actions_l.addStretch(1)
        targets_layout.addWidget(target_actions)

        self.seestar_targets_table = QTableWidget(targets_panel)
        self.seestar_targets_table.setColumnCount(12)
        self.seestar_targets_table.setHorizontalHeaderLabels(
            [translate_text(label, current_language()) for label in [
                "On",
                "#",
                "Target",
                "RA",
                "Dec",
                "Type",
                "Runs",
                "Run min",
                "Gap",
                "Exp ms",
                "LP",
                "Focus",
            ]]
        )
        self.seestar_targets_table.verticalHeader().setVisible(False)
        self.seestar_targets_table.setAlternatingRowColors(True)
        self.seestar_targets_table.setSelectionMode(QTableView.NoSelection)
        self.seestar_targets_table.setEditTriggers(QTableView.NoEditTriggers)
        header = self.seestar_targets_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        for column in (3, 4, 5, 6, 7, 8, 9, 10, 11):
            header.setSectionResizeMode(column, QHeaderView.Interactive)
        targets_layout.addWidget(self.seestar_targets_table, 1)
        self._populate_targets_table()
        settings_scroll.setWidget(settings_panel)
        body_splitter.addWidget(settings_scroll)
        body_splitter.addWidget(targets_panel)
        self._body_splitter = body_splitter
        self._settings_scroll = settings_scroll
        self._settings_panel = settings_panel
        self._targets_panel = targets_panel
        body_splitter.setStretchFactor(0, 0)
        body_splitter.setStretchFactor(1, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        _style_dialog_button_box(buttons)
        root.addWidget(buttons)

        self.seestar_session_template_combo.currentIndexChanged.connect(self._on_session_template_changed)
        self.seestar_session_repeat_spin.valueChanged.connect(self._refresh_target_override_defaults)
        self.seestar_session_minutes_spin.valueChanged.connect(self._refresh_target_override_defaults)
        self.seestar_session_gap_spin.valueChanged.connect(self._refresh_target_override_defaults)
        self.seestar_alp_stack_exposure_spin.valueChanged.connect(self._refresh_target_override_defaults)
        self.seestar_alp_lp_filter_combo.currentIndexChanged.connect(self._refresh_target_override_defaults)
        self.seestar_alp_use_autofocus_chk.toggled.connect(self._refresh_target_override_defaults)
        self.seestar_alp_schedule_af_mode_combo.currentIndexChanged.connect(self._refresh_target_override_defaults)
        self.seestar_alp_startup_enabled_chk.toggled.connect(self._update_automation_fields_enabled)
        self.seestar_alp_capture_flats_chk.toggled.connect(self._update_automation_fields_enabled)
        self.seestar_alp_schedule_af_mode_combo.currentIndexChanged.connect(self._update_automation_fields_enabled)
        self.seestar_alp_dew_heater_chk.toggled.connect(self._update_automation_fields_enabled)
        self.seestar_alp_shutdown_after_chk.toggled.connect(self._update_automation_fields_enabled)
        self._session_tabs.currentChanged.connect(lambda _: QTimer.singleShot(0, self._adjust_session_splitter_sizes))

        self._apply_template_to_fields(self._current_settings_template)
        self._select_session_template(
            self._settings.value(
                "general/seestarSessionTemplateKey",
                "",
                type=str,
            )
        )
        self._on_session_template_changed()
        self._update_automation_fields_enabled()
        self._refresh_target_override_defaults()
        QTimer.singleShot(0, self._adjust_session_splitter_sizes)
        localize_widget_tree(self, current_language())

    def _load_user_session_templates(self) -> dict[str, SeestarSessionTemplate]:
        storage = getattr(self, "_storage", None)
        if storage is not None:
            try:
                stored_items = storage.session_templates.list_all()
                if stored_items:
                    templates: dict[str, SeestarSessionTemplate] = {}
                    for item in stored_items:
                        try:
                            template = SeestarSessionTemplate.model_validate(item)
                        except Exception:
                            continue
                        templates[str(template.key)] = template
                    return templates
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to load session templates from storage: %s", exc)
        return {}

    def _load_current_settings_template(self) -> SeestarSessionTemplate:
        checklist_text = self._settings.value(
            "general/seestarSessionChecklistText",
            "",
            type=str,
        )
        legacy_schedule_autofocus = self._settings.value(
            "general/seestarAlpScheduleAutofocusBeforeEachTarget",
            SEESTAR_ALP_DEFAULT_SCHEDULE_AUTOFOCUS,
            type=bool,
        )
        schedule_autofocus_mode = normalize_seestar_alp_schedule_autofocus_mode(
            self._settings.value(
                "general/seestarAlpScheduleAutofocusMode",
                SEESTAR_ALP_DEFAULT_SCHEDULE_AUTOFOCUS_MODE,
                type=str,
            ),
            legacy_enabled=bool(legacy_schedule_autofocus),
        )
        return SeestarSessionTemplate(
            key="",
            name="",
            scope=SEESTAR_TEMPLATE_SCOPE_MULTI_TARGET,
            repeat_count=max(
                1,
                int(
                    self._settings.value(
                        "general/seestarSessionRepeatCount",
                        1,
                        type=int,
                    )
                ),
            ),
            minutes_per_run=max(
                1,
                int(
                    self._settings.value(
                        "general/seestarSessionMinutesPerRun",
                        SEESTAR_DEFAULT_BLOCK_MINUTES,
                        type=int,
                    )
                ),
            ),
            gap_seconds=max(
                0,
                int(
                    self._settings.value(
                        "general/seestarSessionGapSeconds",
                        0,
                        type=int,
                    )
                ),
            ),
            require_science_checklist=self._settings.value(
                "general/seestarSessionRequireChecklist",
                False,
                type=bool,
            ),
            science_checklist_items=[
                line.strip()
                for line in str(checklist_text or "").splitlines()
                if line.strip()
            ],
            template_notes=str(self._settings.value("general/seestarSessionTemplateNotes", "", type=str) or "").strip(),
            lp_filter_mode=str(
                self._settings.value("general/seestarAlpLpFilterMode", SEESTAR_ALP_LP_FILTER_AUTO, type=str)
                or SEESTAR_ALP_LP_FILTER_AUTO
            ).strip().lower(),
            gain=max(0, int(self._settings.value("general/seestarAlpGain", SEESTAR_ALP_DEFAULT_GAIN, type=int))),
            panel_overlap_percent=max(
                0,
                int(
                    self._settings.value(
                        "general/seestarAlpPanelOverlapPercent",
                        SEESTAR_ALP_DEFAULT_PANEL_OVERLAP_PERCENT,
                        type=int,
                    )
                ),
            ),
            use_autofocus=self._settings.value("general/seestarAlpUseAutofocus", SEESTAR_ALP_DEFAULT_USE_AUTOFOCUS, type=bool),
            num_tries=max(1, int(self._settings.value("general/seestarAlpNumTries", SEESTAR_ALP_DEFAULT_NUM_TRIES, type=int))),
            retry_wait_s=max(0, int(self._settings.value("general/seestarAlpRetryWaitSec", SEESTAR_ALP_DEFAULT_RETRY_WAIT_S, type=int))),
            target_integration_override_min=0,
            stack_exposure_ms=max(0, int(self._settings.value("general/seestarAlpStackExposureMs", SEESTAR_ALP_DEFAULT_STACK_EXPOSURE_MS, type=int))),
            honor_queue_times=self._settings.value("general/seestarAlpHonorQueueTimes", SEESTAR_ALP_DEFAULT_HONOR_QUEUE_TIMES, type=bool),
            wait_until_local_time=str(
                self._settings.value(
                    "general/seestarAlpWaitUntilLocalTime",
                    SEESTAR_ALP_DEFAULT_WAIT_UNTIL_LOCAL_TIME,
                    type=str,
                )
                or ""
            ).strip(),
            startup_enabled=self._settings.value("general/seestarAlpStartupEnabled", SEESTAR_ALP_DEFAULT_STARTUP_SEQUENCE, type=bool),
            startup_polar_align=self._settings.value("general/seestarAlpStartupPolarAlign", SEESTAR_ALP_DEFAULT_STARTUP_POLAR_ALIGN, type=bool),
            startup_auto_focus=self._settings.value("general/seestarAlpStartupAutoFocus", SEESTAR_ALP_DEFAULT_STARTUP_AUTO_FOCUS, type=bool),
            startup_dark_frames=self._settings.value("general/seestarAlpStartupDarkFrames", SEESTAR_ALP_DEFAULT_STARTUP_DARK_FRAMES, type=bool),
            capture_flats_before_session=self._settings.value("general/seestarAlpCaptureFlatsBeforeSession", SEESTAR_ALP_DEFAULT_CAPTURE_FLATS, type=bool),
            flats_wait_s=max(0, int(self._settings.value("general/seestarAlpFlatsWaitSec", SEESTAR_ALP_DEFAULT_FLATS_WAIT_S, type=int))),
            schedule_autofocus_mode=schedule_autofocus_mode,
            schedule_autofocus_before_each_target=(schedule_autofocus_mode == SEESTAR_ALP_AF_MODE_PER_RUN),
            schedule_autofocus_try_count=max(
                1,
                int(self._settings.value("general/seestarAlpScheduleAutofocusTryCount", SEESTAR_ALP_DEFAULT_AUTOFOCUS_TRY_COUNT, type=int)),
            ),
            dew_heater_value=int(self._settings.value("general/seestarAlpDewHeaterValue", SEESTAR_ALP_DEFAULT_DEW_HEATER_VALUE, type=int)),
            park_after_session=self._settings.value("general/seestarAlpParkAfterSession", SEESTAR_ALP_DEFAULT_PARK_AFTER_SESSION, type=bool),
            shutdown_after_session=self._settings.value("general/seestarAlpShutdownAfterSession", SEESTAR_ALP_DEFAULT_SHUTDOWN_AFTER_SESSION, type=bool),
        )

    def _populate_session_template_combo(self) -> None:
        blocker = QSignalBlocker(self.seestar_session_template_combo)
        self.seestar_session_template_combo.clear()
        self.seestar_session_template_combo.addItem("Current settings", "")
        for key, template in self._builtin_session_templates.items():
            self.seestar_session_template_combo.addItem(f"Built-in: {template.name}", f"builtin:{key}")
        for key in sorted(self._user_session_templates.keys()):
            template = self._user_session_templates[key]
            self.seestar_session_template_combo.addItem(f"User: {template.name}", f"user:{key}")
        del blocker

    def _current_science_checklist_items(self) -> list[str]:
        return [line.strip() for line in self.seestar_session_checklist_edit.toPlainText().splitlines() if line.strip()]

    def _current_template_name_for_selection(self) -> str:
        data = str(self.seestar_session_template_combo.currentData() or "")
        if data.startswith("builtin:"):
            template = self._builtin_session_templates.get(data.split(":", 1)[1])
            return str(getattr(template, "name", "") or "").strip()
        if data.startswith("user:"):
            template = self._user_session_templates.get(data.split(":", 1)[1])
            return str(getattr(template, "name", "") or "").strip()
        return ""

    def _current_template_scope_for_selection(self) -> str:
        data = str(self.seestar_session_template_combo.currentData() or "")
        if data.startswith("builtin:"):
            template = self._builtin_session_templates.get(data.split(":", 1)[1])
            scope = str(getattr(template, "scope", SEESTAR_TEMPLATE_SCOPE_MULTI_TARGET) or "").strip()
            return scope or SEESTAR_TEMPLATE_SCOPE_MULTI_TARGET
        if data.startswith("user:"):
            template = self._user_session_templates.get(data.split(":", 1)[1])
            scope = str(getattr(template, "scope", SEESTAR_TEMPLATE_SCOPE_MULTI_TARGET) or "").strip()
            return scope or SEESTAR_TEMPLATE_SCOPE_MULTI_TARGET
        return SEESTAR_TEMPLATE_SCOPE_MULTI_TARGET

    def _current_template_from_fields(
        self,
        *,
        key: str = "",
        name_override: str = "",
        scope_override: str = "",
    ) -> SeestarSessionTemplate:
        name = str(name_override or self._current_template_name_for_selection() or "").strip()
        scope = str(scope_override or self._current_template_scope_for_selection() or SEESTAR_TEMPLATE_SCOPE_MULTI_TARGET).strip()
        schedule_autofocus_mode = normalize_seestar_alp_schedule_autofocus_mode(
            self.seestar_alp_schedule_af_mode_combo.currentData(),
        )
        return SeestarSessionTemplate(
            key=_normalize_catalog_token(key or name).replace(" ", "_") or "template",
            name=name,
            scope=scope or SEESTAR_TEMPLATE_SCOPE_MULTI_TARGET,
            repeat_count=max(1, int(self.seestar_session_repeat_spin.value())),
            minutes_per_run=max(1, int(self.seestar_session_minutes_spin.value())),
            gap_seconds=max(0, int(self.seestar_session_gap_spin.value())),
            require_science_checklist=self.seestar_session_require_checklist_chk.isChecked(),
            science_checklist_items=self._current_science_checklist_items(),
            template_notes=self.seestar_session_notes_edit.toPlainText().strip(),
            lp_filter_mode=str(self.seestar_alp_lp_filter_combo.currentData() or SEESTAR_ALP_LP_FILTER_AUTO),
            gain=max(0, int(self.seestar_alp_gain_spin.value())),
            panel_overlap_percent=max(0, int(self.seestar_alp_panel_overlap_spin.value())),
            use_autofocus=self.seestar_alp_use_autofocus_chk.isChecked(),
            num_tries=max(1, int(self.seestar_alp_num_tries_spin.value())),
            retry_wait_s=max(0, int(self.seestar_alp_retry_wait_spin.value())),
            target_integration_override_min=0,
            stack_exposure_ms=max(0, int(self.seestar_alp_stack_exposure_spin.value())),
            honor_queue_times=self.seestar_alp_honor_queue_times_chk.isChecked(),
            wait_until_local_time=self.seestar_alp_wait_until_edit.text().strip(),
            startup_enabled=self.seestar_alp_startup_enabled_chk.isChecked(),
            startup_polar_align=self.seestar_alp_startup_polar_align_chk.isChecked(),
            startup_auto_focus=self.seestar_alp_startup_autofocus_chk.isChecked(),
            startup_dark_frames=self.seestar_alp_startup_dark_frames_chk.isChecked(),
            capture_flats_before_session=self.seestar_alp_capture_flats_chk.isChecked(),
            flats_wait_s=max(0, int(self.seestar_alp_flats_wait_spin.value())),
            schedule_autofocus_mode=schedule_autofocus_mode,
            schedule_autofocus_before_each_target=(schedule_autofocus_mode == SEESTAR_ALP_AF_MODE_PER_RUN),
            schedule_autofocus_try_count=max(1, int(self.seestar_alp_schedule_af_try_spin.value())),
            dew_heater_value=(
                max(0, int(self.seestar_alp_dew_heater_spin.value()))
                if self.seestar_alp_dew_heater_chk.isChecked()
                else SEESTAR_ALP_DEFAULT_DEW_HEATER_VALUE
            ),
            park_after_session=self.seestar_alp_park_after_chk.isChecked(),
            shutdown_after_session=self.seestar_alp_shutdown_after_chk.isChecked(),
        )

    def _apply_template_to_fields(self, template: SeestarSessionTemplate) -> None:
        self.seestar_session_repeat_spin.setValue(max(1, int(template.repeat_count)))
        self.seestar_session_minutes_spin.setValue(max(1, int(template.minutes_per_run)))
        self.seestar_session_gap_spin.setValue(max(0, int(template.gap_seconds)))
        self.seestar_session_require_checklist_chk.setChecked(bool(template.require_science_checklist))
        self.seestar_session_notes_edit.setPlainText(str(template.template_notes or ""))
        self.seestar_session_checklist_edit.setPlainText("\n".join(template.science_checklist_items))
        lp_idx = self.seestar_alp_lp_filter_combo.findData(str(template.lp_filter_mode or SEESTAR_ALP_LP_FILTER_AUTO))
        if lp_idx >= 0:
            self.seestar_alp_lp_filter_combo.setCurrentIndex(lp_idx)
        self.seestar_alp_gain_spin.setValue(max(0, int(template.gain)))
        self.seestar_alp_panel_overlap_spin.setValue(max(0, int(template.panel_overlap_percent)))
        self.seestar_alp_use_autofocus_chk.setChecked(bool(template.use_autofocus))
        self.seestar_alp_num_tries_spin.setValue(max(1, int(template.num_tries)))
        self.seestar_alp_retry_wait_spin.setValue(max(0, int(template.retry_wait_s)))
        self.seestar_alp_stack_exposure_spin.setValue(max(0, int(template.stack_exposure_ms)))
        self.seestar_alp_honor_queue_times_chk.setChecked(bool(template.honor_queue_times))
        self.seestar_alp_wait_until_edit.setText(str(template.wait_until_local_time or ""))
        self.seestar_alp_startup_enabled_chk.setChecked(bool(template.startup_enabled))
        self.seestar_alp_startup_polar_align_chk.setChecked(bool(template.startup_polar_align))
        self.seestar_alp_startup_autofocus_chk.setChecked(bool(template.startup_auto_focus))
        self.seestar_alp_startup_dark_frames_chk.setChecked(bool(template.startup_dark_frames))
        self.seestar_alp_capture_flats_chk.setChecked(bool(template.capture_flats_before_session))
        self.seestar_alp_flats_wait_spin.setValue(max(0, int(template.flats_wait_s)))
        schedule_af_mode = normalize_seestar_alp_schedule_autofocus_mode(
            getattr(template, "schedule_autofocus_mode", ""),
            legacy_enabled=bool(getattr(template, "schedule_autofocus_before_each_target", False)),
        )
        schedule_af_idx = self.seestar_alp_schedule_af_mode_combo.findData(schedule_af_mode)
        if schedule_af_idx < 0:
            schedule_af_idx = 0
        self.seestar_alp_schedule_af_mode_combo.setCurrentIndex(schedule_af_idx)
        self.seestar_alp_schedule_af_try_spin.setValue(max(1, int(template.schedule_autofocus_try_count)))
        self.seestar_alp_dew_heater_chk.setChecked(int(template.dew_heater_value) >= 0)
        self.seestar_alp_dew_heater_spin.setValue(max(0, int(template.dew_heater_value) if int(template.dew_heater_value) >= 0 else 10))
        self.seestar_alp_park_after_chk.setChecked(bool(template.park_after_session))
        self.seestar_alp_shutdown_after_chk.setChecked(bool(template.shutdown_after_session))
        self._update_automation_fields_enabled()
        self._update_template_state()
        self._refresh_target_override_defaults()

    def _update_template_state(self) -> None:
        scope = self._current_template_scope_for_selection()
        if scope == SEESTAR_TEMPLATE_SCOPE_SINGLE_TARGET:
            self._seestar_template_scope_text = "Single target."
        else:
            self._seestar_template_scope_text = "Multi-target."
        self.seestar_session_template_delete_btn.setEnabled(
            str(self.seestar_session_template_combo.currentData() or "").startswith("user:")
        )
        self._refresh_target_override_defaults()
        QTimer.singleShot(0, self._adjust_session_splitter_sizes)

    def _update_automation_fields_enabled(self) -> None:
        startup_enabled = self.seestar_alp_startup_enabled_chk.isChecked()
        self.seestar_alp_startup_polar_align_chk.setEnabled(startup_enabled)
        self.seestar_alp_startup_autofocus_chk.setEnabled(startup_enabled)
        self.seestar_alp_startup_dark_frames_chk.setEnabled(startup_enabled)

        flats_enabled = self.seestar_alp_capture_flats_chk.isChecked()
        self.seestar_alp_flats_wait_spin.setEnabled(flats_enabled)

        schedule_af_enabled = (
            str(self.seestar_alp_schedule_af_mode_combo.currentData() or SEESTAR_ALP_AF_MODE_OFF)
            != SEESTAR_ALP_AF_MODE_OFF
        )
        self.seestar_alp_schedule_af_try_spin.setEnabled(schedule_af_enabled)

        dew_enabled = self.seestar_alp_dew_heater_chk.isChecked()
        self.seestar_alp_dew_heater_spin.setEnabled(dew_enabled)

        shutdown_enabled = self.seestar_alp_shutdown_after_chk.isChecked()
        if shutdown_enabled:
            self.seestar_alp_park_after_chk.setChecked(True)
        self.seestar_alp_park_after_chk.setEnabled(not shutdown_enabled)

    def _select_session_template(self, preset_key: str) -> None:
        normalized = str(preset_key or "").strip()
        if normalized and not normalized.startswith(("builtin:", "user:")):
            if normalized in self._builtin_session_templates:
                normalized = f"builtin:{normalized}"
            elif normalized in self._user_session_templates:
                normalized = f"user:{normalized}"
            else:
                normalized = ""
        idx = self.seestar_session_template_combo.findData(normalized)
        if idx < 0:
            idx = 0
        blocker = QSignalBlocker(self.seestar_session_template_combo)
        self.seestar_session_template_combo.setCurrentIndex(idx)
        del blocker

    def _on_session_template_changed(self) -> None:
        data = str(self.seestar_session_template_combo.currentData() or "")
        if data.startswith("builtin:"):
            template = self._builtin_session_templates.get(data.split(":", 1)[1])
        elif data.startswith("user:"):
            template = self._user_session_templates.get(data.split(":", 1)[1])
        else:
            template = self._current_settings_template
        if template is not None:
            self._apply_template_to_fields(template)

    def _save_session_template(self) -> None:
        name, ok = QInputDialog.getText(
            self,
            "Save Seestar Session Template",
            "Template name:",
            text=self._current_template_name_for_selection() or "Session template",
        )
        if not ok:
            return
        template_name = str(name or "").strip()
        if not template_name:
            QMessageBox.warning(self, "Seestar Template", "Template name cannot be empty.")
            return
        key = _normalize_catalog_token(template_name).replace(" ", "_") or "template"
        template = self._current_template_from_fields(
            key=key,
            name_override=template_name,
            scope_override=self._current_template_scope_for_selection(),
        )
        self._user_session_templates[key] = template
        self._populate_session_template_combo()
        idx = self.seestar_session_template_combo.findData(f"user:{key}")
        if idx >= 0:
            self.seestar_session_template_combo.setCurrentIndex(idx)
        QMessageBox.information(self, "Seestar Template", f"Saved template '{template_name}'.")

    def _delete_session_template(self) -> None:
        data = str(self.seestar_session_template_combo.currentData() or "")
        if not data.startswith("user:"):
            QMessageBox.information(self, "Seestar Template", "Only user templates can be deleted.")
            return
        key = data.split(":", 1)[1]
        template = self._user_session_templates.get(key)
        if template is None:
            return
        reply = QMessageBox.question(
            self,
            "Delete Seestar Template",
            f"Delete template '{template.name}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self._user_session_templates.pop(key, None)
        self._populate_session_template_combo()
        self.seestar_session_template_combo.setCurrentIndex(0)
        self._update_template_state()

    def _set_combo_data(self, combo: QComboBox, value: object) -> None:
        idx = combo.findData(value)
        if idx < 0:
            idx = 0
        combo.setCurrentIndex(idx)

    def _make_override_spin_cell(
        self,
        *,
        minimum: int,
        maximum: int,
        default_provider: Callable[[], int],
        suffix: str = "",
        single_step: int | None = None,
        min_width: int = 92,
    ) -> dict[str, object]:
        cell = QWidget(self.seestar_targets_table)
        layout = QHBoxLayout(cell)
        layout.setContentsMargins(4, 0, 4, 0)
        layout.setSpacing(4)

        use_default_chk = QCheckBox(cell)
        use_default_chk.setChecked(True)
        use_default_chk.setToolTip("Checked = use the Plan default for this column")
        layout.addWidget(use_default_chk, 0)

        editor = QSpinBox(cell)
        editor.setRange(minimum, maximum)
        editor.setSingleStep(single_step if single_step is not None else (5 if maximum > 100 else 1))
        if suffix:
            editor.setSuffix(suffix)
        editor.setMinimumWidth(min_width)
        editor.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(editor, 1)

        def _refresh() -> None:
            default_value = max(minimum, min(maximum, int(default_provider())))
            if use_default_chk.isChecked():
                blocked = editor.blockSignals(True)
                editor.setValue(default_value)
                editor.blockSignals(blocked)
                editor.setEnabled(False)
                editor.setToolTip(f"Using Plan default: {default_value}{suffix}")
            else:
                editor.setEnabled(True)
                editor.setToolTip("Row override")

        def _on_toggle(checked: bool) -> None:
            if checked:
                editor.setProperty("_override_value", int(editor.value()))
            else:
                stored = editor.property("_override_value")
                if stored is not None:
                    blocked = editor.blockSignals(True)
                    editor.setValue(max(minimum, min(maximum, int(stored))))
                    editor.blockSignals(blocked)
            _refresh()

        def _on_value_changed(value: int) -> None:
            if not use_default_chk.isChecked():
                editor.setProperty("_override_value", int(value))

        use_default_chk.toggled.connect(_on_toggle)
        editor.valueChanged.connect(_on_value_changed)
        _refresh()
        return {
            "widget": cell,
            "default": use_default_chk,
            "editor": editor,
            "refresh": _refresh,
            "value": lambda: None if use_default_chk.isChecked() else int(editor.value()),
        }

    def _make_override_combo_cell(
        self,
        items: list[tuple[str, object]],
        *,
        default_provider: Callable[[], object],
        min_width: int = 96,
    ) -> dict[str, object]:
        cell = QWidget(self.seestar_targets_table)
        layout = QHBoxLayout(cell)
        layout.setContentsMargins(4, 0, 4, 0)
        layout.setSpacing(4)

        use_default_chk = QCheckBox(cell)
        use_default_chk.setChecked(True)
        use_default_chk.setToolTip("Checked = use the Plan default for this column")
        layout.addWidget(use_default_chk, 0)

        combo = QComboBox(cell)
        combo.setMinimumWidth(min_width)
        combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        for label, data in items:
            combo.addItem(label, data)
        layout.addWidget(combo, 1)

        def _refresh() -> None:
            if use_default_chk.isChecked():
                blocked = combo.blockSignals(True)
                self._set_combo_data(combo, default_provider())
                combo.blockSignals(blocked)
                combo.setEnabled(False)
                combo.setToolTip("Using Plan default")
            else:
                combo.setEnabled(True)
                combo.setToolTip("Row override")

        def _on_toggle(checked: bool) -> None:
            if checked:
                combo.setProperty("_override_data", combo.currentData())
            else:
                stored = combo.property("_override_data")
                if stored is not None:
                    blocked = combo.blockSignals(True)
                    self._set_combo_data(combo, stored)
                    combo.blockSignals(blocked)
            _refresh()

        def _on_index_changed(_: int) -> None:
            if not use_default_chk.isChecked():
                combo.setProperty("_override_data", combo.currentData())

        use_default_chk.toggled.connect(_on_toggle)
        combo.currentIndexChanged.connect(_on_index_changed)
        _refresh()
        return {
            "widget": cell,
            "default": use_default_chk,
            "editor": combo,
            "refresh": _refresh,
            "value": lambda: None if use_default_chk.isChecked() else combo.currentData(),
        }

    def _refresh_target_override_defaults(self) -> None:
        for row_state in self._target_plan_rows:
            for key in ("repeat", "minutes", "gap", "exposure", "lp", "af"):
                cell = row_state.get(key)
                if isinstance(cell, dict):
                    refresh = cell.get("refresh")
                    if callable(refresh):
                        refresh()
            self._update_target_row_enabled_state(row_state)
        if hasattr(self, "seestar_defaults_summary_label"):
            lp_label = self.seestar_alp_lp_filter_combo.currentText() if hasattr(self, "seestar_alp_lp_filter_combo") else "Auto"
            af_label = "on" if getattr(self, "seestar_alp_use_autofocus_chk", None) and self.seestar_alp_use_autofocus_chk.isChecked() else "off"
            schedule_af_label = seestar_alp_schedule_autofocus_mode_label(
                self.seestar_alp_schedule_af_mode_combo.currentData()
                if hasattr(self, "seestar_alp_schedule_af_mode_combo")
                else SEESTAR_ALP_AF_MODE_OFF,
                short=True,
            )
            scope_text = str(getattr(self, "_seestar_template_scope_text", "Multi-target.") or "Multi-target.").strip()
            self.seestar_defaults_summary_label.setText(
                f"{scope_text} Defaults: "
                f"{int(self.seestar_session_repeat_spin.value())} x "
                f"{int(self.seestar_session_minutes_spin.value())} min, "
                f"gap {int(self.seestar_session_gap_spin.value())} s, "
                f"LP {lp_label}, exp {int(self.seestar_alp_stack_exposure_spin.value()) or 'ALP default'}, "
                f"capture-job AF {af_label}, scheduled AF {schedule_af_label}."
            )
        QTimer.singleShot(0, self._adjust_session_splitter_sizes)

    def _adjust_session_splitter_sizes(self) -> None:
        splitter = getattr(self, "_body_splitter", None)
        settings_panel = getattr(self, "_settings_panel", None)
        targets_panel = getattr(self, "_targets_panel", None)
        if splitter is None or settings_panel is None or targets_panel is None:
            return
        body_total = splitter.height()
        if body_total <= 0:
            return

        current_tab = self._session_tabs.currentWidget() if hasattr(self, "_session_tabs") else None
        tab_height = current_tab.sizeHint().height() if current_tab is not None else settings_panel.sizeHint().height()
        tabs_height = self._session_tabs.tabBar().sizeHint().height() if hasattr(self, "_session_tabs") else 0
        top_desired = max(300, tab_height + tabs_height + 28)

        row_count = self.seestar_targets_table.rowCount() if hasattr(self, "seestar_targets_table") else 0
        current_tab_name = self._session_tabs.tabText(self._session_tabs.currentIndex()) if hasattr(self, "_session_tabs") else ""
        max_visible_rows = 6 if current_tab_name == "Session" else 4
        visible_rows = min(max(row_count, 1), max_visible_rows)
        header_height = self.seestar_targets_table.horizontalHeader().height() if hasattr(self, "seestar_targets_table") else 28
        row_height = (
            self.seestar_targets_table.rowHeight(0)
            if hasattr(self, "seestar_targets_table") and row_count > 0
            else max(30, self.fontMetrics().height() + 10)
        )
        bottom_base = 170 if current_tab_name == "Session" else 125
        bottom_desired = bottom_base + header_height + (visible_rows * row_height)
        bottom_desired = max(220, min(bottom_desired, 420))

        if top_desired + bottom_desired > body_total:
            bottom_desired = max(220, min(bottom_desired, body_total - 260))
            top_desired = max(260, body_total - bottom_desired)
        else:
            top_desired = min(top_desired, body_total - 220)

        splitter.setSizes([top_desired, max(220, body_total - top_desired)])

    def _set_all_target_rows_enabled(self, enabled: bool) -> None:
        for row_state in self._target_plan_rows:
            checkbox = row_state.get("enabled")
            if hasattr(checkbox, "setChecked"):
                checkbox.setChecked(bool(enabled))

    def _reset_all_target_row_defaults(self) -> None:
        for row_state in self._target_plan_rows:
            for key in ("repeat", "minutes", "gap", "exposure", "lp", "af"):
                cell = row_state.get(key)
                if not isinstance(cell, dict):
                    continue
                default_chk = cell.get("default")
                if hasattr(default_chk, "setChecked"):
                    default_chk.setChecked(True)
        self._refresh_target_override_defaults()

    def _update_target_row_enabled_state(self, row_state: dict[str, object]) -> None:
        enabled = bool(row_state.get("enabled") and row_state["enabled"].isChecked())
        for key in ("repeat", "minutes", "gap", "exposure", "lp", "af"):
            cell = row_state.get(key)
            if not isinstance(cell, dict):
                continue
            default_chk = cell.get("default")
            editor = cell.get("editor")
            if hasattr(default_chk, "setEnabled"):
                default_chk.setEnabled(enabled)
            if hasattr(editor, "setEnabled"):
                if enabled:
                    refresh = cell.get("refresh")
                    if callable(refresh):
                        refresh()
                else:
                    editor.setEnabled(False)

    def _format_target_ra(self, ra_deg: float) -> str:
        try:
            coord = SkyCoord(ra=float(ra_deg) * u.deg, dec=0.0 * u.deg)
            return coord.ra.to_string(unit=u.hour, sep=":", precision=0, pad=True)
        except Exception:
            return f"{float(ra_deg):.4f}"

    def _format_target_dec(self, dec_deg: float) -> str:
        try:
            coord = SkyCoord(ra=0.0 * u.deg, dec=float(dec_deg) * u.deg)
            return coord.dec.to_string(unit=u.deg, sep=":", precision=0, pad=True, alwayssign=True)
        except Exception:
            return f"{float(dec_deg):+.4f}"

    def _populate_targets_table(self) -> None:
        self.seestar_targets_table.setRowCount(0)
        self._target_plan_rows = []
        self.seestar_targets_table.setWordWrap(False)
        self.seestar_targets_table.verticalHeader().setDefaultSectionSize(42)
        for row, target in enumerate(self._planner.targets):
            self.seestar_targets_table.insertRow(row)

            enabled_chk = QCheckBox(self.seestar_targets_table)
            enabled_chk.setChecked(True)
            enabled_wrap = QWidget(self.seestar_targets_table)
            enabled_layout = QHBoxLayout(enabled_wrap)
            enabled_layout.setContentsMargins(4, 0, 4, 0)
            enabled_layout.setAlignment(Qt.AlignCenter)
            enabled_layout.addWidget(enabled_chk)
            self.seestar_targets_table.setCellWidget(row, 0, enabled_wrap)

            order_item = QTableWidgetItem(str(row + 1))
            order_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.seestar_targets_table.setItem(row, 1, order_item)

            for column, value in (
                (2, str(target.name or "")),
                (3, self._format_target_ra(float(target.ra))),
                (4, self._format_target_dec(float(target.dec))),
                (5, str(target.object_type or "")),
            ):
                item = QTableWidgetItem(value)
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                item.setToolTip(value)
                self.seestar_targets_table.setItem(row, column, item)

            repeat_cell = self._make_override_spin_cell(
                minimum=1,
                maximum=50,
                default_provider=lambda: int(self.seestar_session_repeat_spin.value()),
                min_width=74,
            )
            minutes_cell = self._make_override_spin_cell(
                minimum=1,
                maximum=240,
                default_provider=lambda: int(self.seestar_session_minutes_spin.value()),
                suffix=" min",
                single_step=5,
                min_width=88,
            )
            gap_cell = self._make_override_spin_cell(
                minimum=0,
                maximum=3600,
                default_provider=lambda: int(self.seestar_session_gap_spin.value()),
                suffix=" s",
                single_step=5,
                min_width=82,
            )
            exposure_cell = self._make_override_spin_cell(
                minimum=0,
                maximum=600000,
                default_provider=lambda: int(self.seestar_alp_stack_exposure_spin.value()),
                suffix=" ms",
                single_step=500,
                min_width=108,
            )
            lp_cell = self._make_override_combo_cell(
                [
                    ("Auto", SEESTAR_ALP_LP_FILTER_AUTO),
                    ("OFF", SEESTAR_ALP_LP_FILTER_OFF),
                    ("ON", SEESTAR_ALP_LP_FILTER_ON),
                ],
                default_provider=lambda: str(self.seestar_alp_lp_filter_combo.currentData() or SEESTAR_ALP_LP_FILTER_AUTO),
                min_width=92,
            )
            af_cell = self._make_override_combo_cell(
                [
                    ("On", True),
                    ("Off", False),
                ],
                default_provider=lambda: bool(self.seestar_alp_use_autofocus_chk.isChecked()),
                min_width=78,
            )

            self.seestar_targets_table.setCellWidget(row, 6, repeat_cell["widget"])
            self.seestar_targets_table.setCellWidget(row, 7, minutes_cell["widget"])
            self.seestar_targets_table.setCellWidget(row, 8, gap_cell["widget"])
            self.seestar_targets_table.setCellWidget(row, 9, exposure_cell["widget"])
            self.seestar_targets_table.setCellWidget(row, 10, lp_cell["widget"])
            self.seestar_targets_table.setCellWidget(row, 11, af_cell["widget"])

            row_state = {
                "target": target,
                "enabled": enabled_chk,
                "repeat": repeat_cell,
                "minutes": minutes_cell,
                "gap": gap_cell,
                "exposure": exposure_cell,
                "lp": lp_cell,
                "af": af_cell,
            }
            enabled_chk.toggled.connect(lambda _checked, state=row_state: self._update_target_row_enabled_state(state))
            self._target_plan_rows.append(row_state)
            self._update_target_row_enabled_state(row_state)

        self.seestar_targets_table.setColumnWidth(0, 54)
        self.seestar_targets_table.setColumnWidth(1, 40)
        self.seestar_targets_table.setColumnWidth(3, 100)
        self.seestar_targets_table.setColumnWidth(4, 112)
        self.seestar_targets_table.setColumnWidth(5, 74)
        self.seestar_targets_table.setColumnWidth(6, 98)
        self.seestar_targets_table.setColumnWidth(7, 114)
        self.seestar_targets_table.setColumnWidth(8, 100)
        self.seestar_targets_table.setColumnWidth(9, 132)
        self.seestar_targets_table.setColumnWidth(10, 102)
        self.seestar_targets_table.setColumnWidth(11, 92)
        self.seestar_targets_table.setMinimumHeight(max(220, min(420, 42 + 30 * max(1, len(self._target_plan_rows)))))
        self._refresh_target_override_defaults()

    def session_template(self) -> SeestarSessionTemplate:
        data = str(self.seestar_session_template_combo.currentData() or "")
        name = self._current_template_name_for_selection() if data else ""
        scope = self._current_template_scope_for_selection()
        return self._current_template_from_fields(name_override=name, scope_override=scope)

    def method(self) -> str:
        return str(self.seestar_method_combo.currentData() or SEESTAR_METHOD_GUIDED).strip().lower()

    def alp_config(self) -> SeestarAlpConfig:
        template = self.session_template()
        schedule_autofocus_mode = normalize_seestar_alp_schedule_autofocus_mode(
            getattr(template, "schedule_autofocus_mode", ""),
            legacy_enabled=bool(getattr(template, "schedule_autofocus_before_each_target", False)),
        )
        return SeestarAlpConfig(
            base_url=str(
                self._settings.value("general/seestarAlpBaseUrl", SEESTAR_ALP_DEFAULT_BASE_URL, type=str)
                or SEESTAR_ALP_DEFAULT_BASE_URL
            ).strip(),
            device_num=max(
                0,
                int(
                    self._settings.value(
                        "general/seestarAlpDeviceNum",
                        SEESTAR_ALP_DEFAULT_DEVICE_NUM,
                        type=int,
                    )
                ),
            ),
            client_id=max(
                1,
                int(
                    self._settings.value(
                        "general/seestarAlpClientId",
                        SEESTAR_ALP_DEFAULT_CLIENT_ID,
                        type=int,
                    )
                ),
            ),
            timeout_s=max(
                1.0,
                float(
                    self._settings.value(
                        "general/seestarAlpTimeoutSec",
                        SEESTAR_ALP_DEFAULT_TIMEOUT_S,
                        type=float,
                    )
                ),
            ),
            gain=max(0, int(template.gain)),
            panel_overlap_percent=max(0, int(template.panel_overlap_percent)),
            use_autofocus=bool(template.use_autofocus),
            num_tries=max(1, int(template.num_tries)),
            retry_wait_s=max(0, int(template.retry_wait_s)),
            target_integration_override_min=max(0, int(template.target_integration_override_min)),
            stack_exposure_ms=max(0, int(template.stack_exposure_ms)),
            lp_filter_mode=str(template.lp_filter_mode or SEESTAR_ALP_LP_FILTER_AUTO).strip().lower(),
            honor_queue_times=bool(template.honor_queue_times),
            wait_until_local_time=str(template.wait_until_local_time or "").strip(),
            startup_enabled=bool(template.startup_enabled),
            startup_polar_align=bool(template.startup_polar_align),
            startup_auto_focus=bool(template.startup_auto_focus),
            startup_dark_frames=bool(template.startup_dark_frames),
            capture_flats_before_session=bool(template.capture_flats_before_session),
            flats_wait_s=max(0, int(template.flats_wait_s)),
            schedule_autofocus_mode=schedule_autofocus_mode,
            schedule_autofocus_before_each_target=(schedule_autofocus_mode == SEESTAR_ALP_AF_MODE_PER_RUN),
            schedule_autofocus_try_count=max(1, int(template.schedule_autofocus_try_count)),
            dew_heater_value=int(template.dew_heater_value),
            park_after_session=bool(template.park_after_session),
            shutdown_after_session=bool(template.shutdown_after_session),
        )

    def session_items(self) -> list[SeestarTargetSessionItem]:
        items: list[SeestarTargetSessionItem] = []
        for row, row_state in enumerate(self._target_plan_rows, start=1):
            target = row_state["target"]
            metrics = self._planner.target_metrics.get(target.name)
            window = self._planner.target_windows.get(target.name)
            lp_value = row_state["lp"]["value"]()
            af_value = row_state["af"]["value"]()
            items.append(
                SeestarTargetSessionItem(
                    enabled=bool(row_state["enabled"].isChecked()),
                    order=row,
                    target_name=str(target.name or "").strip(),
                    ra_deg=float(target.ra),
                    dec_deg=float(target.dec),
                    object_type=str(target.object_type or ""),
                    notes=str(target.notes or "").strip(),
                    score=float(getattr(metrics, "score", 0.0) or 0.0),
                    hours_above_limit=float(getattr(metrics, "hours_above_limit", 0.0) or 0.0),
                    max_altitude_deg=float(getattr(metrics, "max_altitude_deg", 0.0) or 0.0),
                    window_start_local=window[0] if isinstance(window, tuple) and len(window) == 2 else None,
                    window_end_local=window[1] if isinstance(window, tuple) and len(window) == 2 else None,
                    repeat_count=row_state["repeat"]["value"](),
                    segment_minutes=row_state["minutes"]["value"](),
                    gap_seconds=row_state["gap"]["value"](),
                    stack_exposure_ms=row_state["exposure"]["value"](),
                    lp_filter_mode=str(lp_value).strip().lower() or None,
                    autofocus=af_value if isinstance(af_value, bool) else None,
                )
            )
        return items

    def _apply_changes(self) -> None:
        s = self._settings
        selected_template_key = str(self.seestar_session_template_combo.currentData() or "")
        current_template = self._current_template_from_fields()
        s.setValue("general/seestarMethod", self.method())
        s.setValue("general/seestarSessionTemplateKey", selected_template_key)
        storage = getattr(self, "_storage", None)
        if storage is not None:
            try:
                storage.session_templates.replace_all(
                    [template.model_dump(mode="json") for template in self._user_session_templates.values()]
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to save session templates to storage: %s", exc)
        if not selected_template_key:
            s.setValue("general/seestarSessionRepeatCount", max(1, int(self.seestar_session_repeat_spin.value())))
            s.setValue("general/seestarSessionMinutesPerRun", max(1, int(self.seestar_session_minutes_spin.value())))
            s.setValue("general/seestarSessionGapSeconds", max(0, int(self.seestar_session_gap_spin.value())))
            s.setValue("general/seestarSessionRequireChecklist", self.seestar_session_require_checklist_chk.isChecked())
            s.setValue("general/seestarSessionChecklistText", self.seestar_session_checklist_edit.toPlainText().strip())
            s.setValue("general/seestarSessionTemplateNotes", self.seestar_session_notes_edit.toPlainText().strip())
            s.setValue("general/seestarAlpGain", max(0, int(self.seestar_alp_gain_spin.value())))
            s.setValue("general/seestarAlpPanelOverlapPercent", max(0, int(self.seestar_alp_panel_overlap_spin.value())))
            s.setValue("general/seestarAlpUseAutofocus", self.seestar_alp_use_autofocus_chk.isChecked())
            s.setValue("general/seestarAlpNumTries", max(1, int(self.seestar_alp_num_tries_spin.value())))
            s.setValue("general/seestarAlpRetryWaitSec", max(0, int(self.seestar_alp_retry_wait_spin.value())))
            s.setValue("general/seestarAlpStackExposureMs", max(0, int(self.seestar_alp_stack_exposure_spin.value())))
            s.setValue("general/seestarAlpLpFilterMode", str(self.seestar_alp_lp_filter_combo.currentData() or SEESTAR_ALP_LP_FILTER_AUTO).strip().lower())
            s.setValue("general/seestarAlpHonorQueueTimes", self.seestar_alp_honor_queue_times_chk.isChecked())
            s.setValue("general/seestarAlpWaitUntilLocalTime", self.seestar_alp_wait_until_edit.text().strip())
            s.setValue("general/seestarAlpStartupEnabled", self.seestar_alp_startup_enabled_chk.isChecked())
            s.setValue("general/seestarAlpStartupPolarAlign", self.seestar_alp_startup_polar_align_chk.isChecked())
            s.setValue("general/seestarAlpStartupAutoFocus", self.seestar_alp_startup_autofocus_chk.isChecked())
            s.setValue("general/seestarAlpStartupDarkFrames", self.seestar_alp_startup_dark_frames_chk.isChecked())
            s.setValue("general/seestarAlpCaptureFlatsBeforeSession", self.seestar_alp_capture_flats_chk.isChecked())
            s.setValue("general/seestarAlpFlatsWaitSec", max(0, int(self.seestar_alp_flats_wait_spin.value())))
            schedule_autofocus_mode = normalize_seestar_alp_schedule_autofocus_mode(
                self.seestar_alp_schedule_af_mode_combo.currentData(),
            )
            s.setValue("general/seestarAlpScheduleAutofocusMode", schedule_autofocus_mode)
            s.setValue(
                "general/seestarAlpScheduleAutofocusBeforeEachTarget",
                schedule_autofocus_mode == SEESTAR_ALP_AF_MODE_PER_RUN,
            )
            s.setValue("general/seestarAlpScheduleAutofocusTryCount", max(1, int(self.seestar_alp_schedule_af_try_spin.value())))
            s.setValue(
                "general/seestarAlpDewHeaterValue",
                max(0, int(self.seestar_alp_dew_heater_spin.value()))
                if self.seestar_alp_dew_heater_chk.isChecked()
                else SEESTAR_ALP_DEFAULT_DEW_HEATER_VALUE,
            )
            s.setValue("general/seestarAlpParkAfterSession", self.seestar_alp_park_after_chk.isChecked())
            s.setValue("general/seestarAlpShutdownAfterSession", self.seestar_alp_shutdown_after_chk.isChecked())
            self._current_settings_template = current_template

    def accept(self) -> None:
        enabled_items = [item for item in self.session_items() if item.enabled]
        if not enabled_items:
            QMessageBox.warning(self, "Seestar Session", "Enable at least one target in the session table.")
            return
        template = self.session_template()
        if str(template.scope or SEESTAR_TEMPLATE_SCOPE_MULTI_TARGET) == SEESTAR_TEMPLATE_SCOPE_SINGLE_TARGET and len(enabled_items) != 1:
            QMessageBox.warning(
                self,
                "Seestar Session",
                "The selected template is single-target. Enable exactly one target in the session table.",
            )
            return
        self._apply_changes()
        super().accept()


__all__ = ["SeestarSessionPlanDialog"]
