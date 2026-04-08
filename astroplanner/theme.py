from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
from urllib.parse import quote


@dataclass(frozen=True)
class UiTheme:
    key: str
    label: str
    light: Mapping[str, str]
    dark: Mapping[str, str]


def _merge(base: Mapping[str, str], **updates: str) -> dict[str, str]:
    merged = dict(base)
    merged.update(updates)
    return merged


_LIGHT_TEMPLATE = """
    QWidget {
        font-family: %(font_family)s;
        font-size: %(font_size_pt)spt;
        color: %(base_text)s;
        background: transparent;
    }
    QWidget#RootContainer {
        background: %(root_background)s;
    }
    QDialog {
        background: %(panel_bg)s;
        color: %(base_text)s;
    }
    QFrame#SidebarPanel,
    QFrame#TopControlsBar,
    QFrame#ActionsBar,
    QFrame#PlotCard,
    QFrame#TableCard,
    QFrame#PolarCard,
    QFrame#InfoCard,
    QFrame#CutoutFrame {
        background: %(panel_bg)s;
        border: 1px solid %(panel_border)s;
        border-radius: 16px;
    }
    QFrame#TopControlsBar,
    QFrame#ActionsBar {
        background: %(top_bg)s;
        border: 1px solid %(top_border)s;
    }
    QFrame[accented="true"] {
        border: 1px solid %(glow_edge)s;
    }
    QGroupBox {
        font-weight: 600;
        color: %(group_text)s;
        border: 1px solid %(group_border)s;
        border-radius: 12px;
        margin-top: 12px;
        padding-top: 10px;
        background: %(group_bg)s;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 4px;
    }
    QTabWidget::pane {
        background: %(tab_pane_bg)s;
        border: 1px solid %(group_border)s;
        border-radius: 10px;
        top: -1px;
    }
    QTabBar {
        background: transparent;
    }
    QTabBar::tab {
        background: %(tab_inactive_bg)s;
        color: %(strip_label)s;
        border: 1px solid %(panel_border)s;
        border-bottom: none;
        border-top-left-radius: 10px;
        border-top-right-radius: 10px;
        padding: 7px 16px;
        min-width: 58px;
        font-family: %(display_font_family)s;
        font-weight: 600;
        letter-spacing: 0.6px;
    }
    QTabBar::tab:selected {
        background: %(tab_selected_bg)s;
        color: %(section_title)s;
        border: 1px solid %(glow_edge)s;
        border-bottom: 1px solid %(tab_selected_bg)s;
    }
    QTabBar::tab:!selected:hover {
        background: %(tab_hover_bg)s;
        color: %(section_title)s;
    }
    QLabel#SectionTitle {
        font-family: %(display_font_family)s;
        color: %(section_title)s;
        font-size: %(section_font_size_pt)spt;
        font-weight: 700;
        letter-spacing: 0.7px;
        padding: 0 0 4px 2px;
    }
    QLabel#SectionHint {
        color: %(section_hint)s;
        font-size: %(strip_font_size_pt)spt;
        font-weight: 500;
        padding: 0 2px 4px 2px;
    }
    QLabel[tone="info"] {
        color: %(state_info)s;
    }
    QLabel[tone="success"] {
        color: %(state_success)s;
    }
    QLabel[tone="warning"] {
        color: %(state_warning)s;
    }
    QLabel[tone="error"] {
        color: %(state_error)s;
    }
    QLabel[tone="muted"] {
        color: %(section_hint)s;
    }
    QLabel[weather_chip="true"] {
        background: %(chip_bg)s;
        border: 1px solid %(chip_border)s;
        border-radius: 9px;
        padding: 4px 8px;
        font-weight: 600;
        color: %(chip_text)s;
    }
    QLabel[weather_chip="true"][weather_chip_role="weather"] {
        background: %(chip_weather_bg)s;
        border-color: %(chip_weather_border)s;
    }
    QLabel[weather_chip="true"][weather_chip_role="context"] {
        background: %(chip_context_bg)s;
        border-color: %(chip_context_border)s;
    }
    QLabel[weather_chip="true"][weather_chip_role="clock"] {
        background: %(chip_clock_bg)s;
        border-color: %(chip_clock_border)s;
    }
    QLabel[weather_chip="true"][weather_chip_role="solar"] {
        background: %(chip_solar_bg)s;
        border-color: %(chip_solar_border)s;
    }
    QLabel[weather_chip="true"][weather_chip_role="lunar"] {
        background: %(chip_lunar_bg)s;
        border-color: %(chip_lunar_border)s;
    }
    QLabel[weather_chip="true"][weather_chip_series="temp"] {
        background: %(chip_series_temp_bg)s;
        border-color: %(chip_series_temp_border)s;
        color: %(chip_series_temp_text)s;
    }
    QLabel[weather_chip="true"][weather_chip_series="wind"] {
        background: %(chip_series_wind_bg)s;
        border-color: %(chip_series_wind_border)s;
        color: %(chip_series_wind_text)s;
    }
    QLabel[weather_chip="true"][weather_chip_series="cloud"] {
        background: %(chip_series_cloud_bg)s;
        border-color: %(chip_series_cloud_border)s;
        color: %(chip_series_cloud_text)s;
    }
    QLabel[weather_chip="true"][weather_chip_series="humidity"] {
        background: %(chip_series_humidity_bg)s;
        border-color: %(chip_series_humidity_border)s;
        color: %(chip_series_humidity_text)s;
    }
    QLabel[weather_chip="true"][weather_chip_series="pressure"] {
        background: %(chip_series_pressure_bg)s;
        border-color: %(chip_series_pressure_border)s;
        color: %(chip_series_pressure_text)s;
    }
    *[cutout_page="true"] {
        background: %(tab_pane_bg)s;
    }
    *[cutout_tool_col="true"] {
        background: %(tab_pane_bg)s;
        border-left: 1px solid %(group_border)s;
        border-top-right-radius: 10px;
        border-bottom-right-radius: 10px;
    }
    *[cutout_image="true"] {
        background: %(input_bg)s;
        border: 1px solid %(input_border)s;
        border-radius: 10px;
        color: %(strip_label)s;
        font-size: %(strip_font_size_pt)spt;
        font-weight: 500;
        padding: 4px;
    }
    QFrame#VisibilityLoadingCard {
        background: %(loading_bg)s;
        border: 1px solid %(loading_border)s;
        border-radius: 16px;
    }
    QWidget#SessionStrip QLabel,
    QWidget#FiltersStrip QLabel {
        color: %(strip_label)s;
        font-size: %(strip_font_size_pt)spt;
        font-weight: 600;
    }
    QFrame#TopControlsBar {
        border-radius: 10px;
    }
    QFrame#ActionsBar {
        border-radius: 10px;
    }
    QPushButton {
        background: qlineargradient(
            x1: 0, y1: 0, x2: 1, y2: 0,
            stop: 0 %(btn0)s,
            stop: 0.55 %(btn1)s,
            stop: 1 %(btn2)s
        );
        color: %(btn_text)s;
        border: 1px solid %(glow_edge)s;
        border-radius: 12px;
        padding: 4px 16px 5px 12px;
        min-height: 24px;
        font-weight: 600;
        font-family: %(display_font_family)s;
        letter-spacing: 0.6px;
    }
    QPushButton:hover {
        background: qlineargradient(
            x1: 0, y1: 0, x2: 1, y2: 0,
            stop: 0 %(btnh0)s,
            stop: 0.55 %(btnh1)s,
            stop: 1 %(btnh2)s
        );
        border-color: %(button_hover_glow)s;
    }
    QPushButton:pressed {
        background: %(btn_pressed)s;
    }
    QPushButton:checked {
        background: qlineargradient(
            x1: 0, y1: 0, x2: 1, y2: 0,
            stop: 0 %(btnh0)s,
            stop: 0.55 %(btnh1)s,
            stop: 1 %(btnh2)s
        );
        border: 1px solid %(accent_secondary)s;
    }
    QPushButton:disabled {
        background: %(btn_disabled_bg)s;
        color: %(btn_disabled_text)s;
        border: 1px solid %(btn_disabled_border)s;
    }
    QPushButton[variant="primary"] {
        border: 1px solid %(glow_edge)s;
    }
    QPushButton[variant="secondary"] {
        background: qlineargradient(
            x1: 0, y1: 0, x2: 1, y2: 0,
            stop: 0 %(tab_inactive_bg)s,
            stop: 0.5 %(tab_hover_bg)s,
            stop: 1 %(chip_bg)s
        );
        color: %(section_title)s;
        border: 1px solid %(glow_edge)s;
    }
    QPushButton[variant="secondary"]:hover {
        background: qlineargradient(
            x1: 0, y1: 0, x2: 1, y2: 0,
            stop: 0 %(tab_hover_bg)s,
            stop: 1 %(chip_bg)s
        );
        border-color: %(button_hover_glow)s;
    }
    QPushButton[variant="neutral"] {
        background: qlineargradient(
            x1: 0, y1: 0, x2: 1, y2: 0,
            stop: 0 %(tab_inactive_bg)s,
            stop: 1 %(tab_pane_bg)s
        );
        color: %(section_title)s;
        border: 1px solid %(panel_border)s;
    }
    QPushButton[variant="neutral"]:hover {
        background: %(tab_hover_bg)s;
        border-color: %(button_hover_glow)s;
    }
    QPushButton[variant="ghost"] {
        background: transparent;
        color: %(strip_label)s;
        border: 1px solid %(panel_border)s;
    }
    QPushButton[variant="ghost"]:hover {
        background: %(tool_hover)s;
        color: %(section_title)s;
        border-color: %(button_hover_glow)s;
    }
    QPushButton[weather_link="true"] {
        padding: 2px 10px 2px 8px;
        min-height: 22px;
        max-height: 22px;
        border-radius: 8px;
        font-weight: 500;
        font-size: 10pt;
    }
    QToolButton {
        background: transparent;
        border-radius: 6px;
        padding: 3px;
    }
    QToolButton:hover {
        background: %(tool_hover)s;
    }
    QToolButton#DateNavButton {
        background: %(input_bg)s;
        border: 1px solid %(input_border)s;
        border-radius: 8px;
        padding: 0px;
    }
    QToolButton#DateNavButton:hover {
        background: %(tool_hover)s;
        border: 1px solid %(input_focus)s;
    }
    QToolButton#DateNavButton:pressed {
        background: %(tab_selected_bg)s;
        border: 1px solid %(input_focus)s;
    }
    QComboBox, QSpinBox, QDoubleSpinBox, QDateEdit, QLineEdit, QTextEdit {
        background: %(input_bg)s;
        border: 1px solid %(input_border)s;
        border-radius: 7px;
        padding: 4px 6px;
        min-height: 20px;
        color: %(input_text)s;
    }
    QComboBox {
        padding-right: 24px;
    }
    QComboBox::drop-down, QDateEdit::drop-down {
        subcontrol-origin: border;
        subcontrol-position: top right;
        width: 20px;
        border-left: 1px solid %(panel_border)s;
        border-top-right-radius: 6px;
        border-bottom-right-radius: 6px;
        background: %(tab_inactive_bg)s;
    }
    QComboBox::drop-down:hover, QDateEdit::drop-down:hover {
        background: %(tab_hover_bg)s;
    }
    QComboBox::down-arrow, QDateEdit::down-arrow {
        image: %(combo_arrow_icon)s;
        width: 10px;
        height: 6px;
        margin-right: 7px;
    }
    QComboBox::down-arrow:disabled, QDateEdit::down-arrow:disabled {
        image: %(combo_arrow_disabled_icon)s;
    }
    QComboBox QAbstractItemView {
        background: %(combo_popup_bg)s;
        color: %(combo_popup_text)s;
        border: 1px solid %(combo_popup_border)s;
        border-radius: 8px;
        padding: 4px;
        outline: 0;
        selection-background-color: %(combo_popup_sel_bg)s;
        selection-color: %(combo_popup_sel_text)s;
    }
    QComboBox QAbstractItemView::item {
        min-height: 22px;
        padding: 4px 8px;
        border-radius: 6px;
    }
    QComboBox QAbstractItemView::item:hover {
        background: %(combo_popup_hover)s;
    }
    QSpinBox, QDoubleSpinBox, QDateEdit {
        padding-right: 22px;
    }
    QSpinBox::up-button, QDoubleSpinBox::up-button, QDateEdit::up-button {
        subcontrol-origin: border;
        subcontrol-position: top right;
        width: 18px;
        border-left: 1px solid %(panel_border)s;
        border-bottom: 1px solid %(panel_border)s;
        border-top-right-radius: 6px;
        background: %(spin_btn_bg)s;
    }
    QSpinBox::down-button, QDoubleSpinBox::down-button, QDateEdit::down-button {
        subcontrol-origin: border;
        subcontrol-position: bottom right;
        width: 18px;
        border-left: 1px solid %(panel_border)s;
        border-top: 1px solid %(panel_border)s;
        border-bottom-right-radius: 6px;
        background: %(spin_btn_bg)s;
    }
    QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover, QDateEdit::up-button:hover,
    QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover, QDateEdit::down-button:hover {
        background: %(spin_btn_hover)s;
    }
    QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed, QDateEdit::up-button:pressed,
    QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed, QDateEdit::down-button:pressed {
        background: %(spin_btn_pressed)s;
    }
    QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
        image: %(spin_up_icon)s;
        width: 10px;
        height: 6px;
    }
    QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
        image: %(spin_down_icon)s;
        width: 10px;
        height: 6px;
    }
    QSpinBox::up-arrow:disabled, QDoubleSpinBox::up-arrow:disabled {
        image: %(spin_up_disabled_icon)s;
    }
    QSpinBox::down-arrow:disabled, QDoubleSpinBox::down-arrow:disabled {
        image: %(spin_down_disabled_icon)s;
    }
    QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QDateEdit:focus, QLineEdit:focus, QTextEdit:focus {
        border: 1px solid %(input_focus)s;
    }
    QComboBox[invalid="true"], QSpinBox[invalid="true"], QDoubleSpinBox[invalid="true"],
    QDateEdit[invalid="true"], QLineEdit[invalid="true"], QTextEdit[invalid="true"] {
        border: 1px solid %(state_error)s;
        background: %(state_error_bg)s;
    }
    QCalendarWidget {
        background: %(input_bg)s;
        border: 1px solid %(input_border)s;
        border-radius: 8px;
    }
    QCalendarWidget QWidget {
        background: %(input_bg)s;
        color: %(input_text)s;
    }
    QCalendarWidget QWidget#qt_calendar_navigationbar {
        background: %(panel_bg)s;
        border-bottom: 1px solid %(input_border)s;
    }
    QCalendarWidget QToolButton {
        color: %(input_text)s;
        background: %(tab_inactive_bg)s;
        border: 1px solid %(input_border)s;
        border-radius: 6px;
        padding: 2px 6px;
    }
    QCalendarWidget QToolButton:hover {
        background: %(tab_hover_bg)s;
        border: 1px solid %(input_focus)s;
    }
    QCalendarWidget QSpinBox {
        background: %(input_bg)s;
        color: %(input_text)s;
        border: 1px solid %(input_border)s;
        border-radius: 6px;
        padding: 2px 4px;
    }
    QCalendarWidget QDoubleSpinBox {
        background: %(input_bg)s;
        color: %(input_text)s;
        border: 1px solid %(input_border)s;
        border-radius: 6px;
        padding: 2px 4px;
    }
    QCalendarWidget QAbstractItemView {
        background: %(input_bg)s;
        color: %(input_text)s;
        selection-background-color: %(table_sel_bg)s;
        selection-color: %(table_sel_text)s;
        border: 1px solid %(input_border)s;
        outline: 0;
    }
    QTableView, QTableWidget {
        background: %(table_bg)s;
        alternate-background-color: %(table_alt)s;
        border: 1px solid %(table_border)s;
        border-radius: 12px;
        gridline-color: %(table_grid)s;
        color: %(base_text)s;
        selection-color: %(table_sel_text)s;
        selection-background-color: %(table_sel_bg)s;
        padding: 2px;
    }
    QTableView::item, QTableWidget::item {
        color: %(base_text)s;
        padding: 4px 8px;
        border-bottom: 1px solid %(table_grid)s;
    }
    QTableView::item:selected, QTableWidget::item:selected {
        background: %(table_sel_item)s;
        color: %(table_sel_text)s;
    }
    QTableCornerButton::section {
        background: %(corner_bg)s;
        border: none;
    }
    QHeaderView::section {
        font-family: %(display_font_family)s;
        background: qlineargradient(
            x1: 0, y1: 0, x2: 0, y2: 1,
            stop: 0 %(tab_hover_bg)s,
            stop: 1 %(tab_inactive_bg)s
        );
        color: %(header_text)s;
        font-weight: 600;
        letter-spacing: 0.55px;
        padding: 7px 8px;
        border: 0;
        border-bottom: 1px solid %(glow_edge)s;
        border-right: 1px solid %(table_grid)s;
    }
    QCheckBox {
        spacing: 5px;
        color: %(check_text)s;
    }
    QCheckBox::indicator {
        width: 15px;
        height: 15px;
        border-radius: 4px;
        border: 1px solid %(check_border)s;
        background: %(check_bg)s;
    }
    QCheckBox::indicator:checked {
        background: qlineargradient(
            x1: 0, y1: 0, x2: 1, y2: 1,
            stop: 0 %(check_checked0)s,
            stop: 1 %(check_checked1)s
        );
        border: 1px solid %(check_checked_border)s;
    }
    QProgressBar {
        border: 1px solid %(progress_border)s;
        border-radius: 7px;
        background: %(progress_bg)s;
        color: %(progress_text)s;
        text-align: center;
        min-height: 18px;
    }
    QProgressBar::chunk {
        border-radius: 6px;
        margin: 1px;
        background: qlineargradient(
            x1: 0, y1: 0, x2: 1, y2: 0,
            stop: 0 %(progress_chunk0)s,
            stop: 1 %(progress_chunk1)s
        );
    }
    QProgressBar[weather_phase_chip="true"] {
        background: %(chip_lunar_bg)s;
        border: 1px solid %(chip_lunar_border)s;
        border-radius: 9px;
        color: %(chip_text)s;
        text-align: center;
        min-height: 26px;
        max-height: 26px;
        font-weight: 700;
    }
    QProgressBar[weather_phase_chip="true"]::chunk {
        border-radius: 8px;
        margin: 1px;
        background: qlineargradient(
            x1: 0, y1: 0, x2: 1, y2: 0,
            stop: 0 %(progress_chunk0)s,
            stop: 1 %(progress_chunk1)s
        );
    }
    QSlider#PlotModeSwitch::groove:horizontal {
        height: 18px;
        border-radius: 9px;
        background: %(tab_inactive_bg)s;
        border: 1px solid %(panel_border)s;
    }
    QSlider#PlotModeSwitch::sub-page:horizontal {
        background: %(btn1)s;
        border-radius: 9px;
    }
    QSlider#PlotModeSwitch::add-page:horizontal {
        background: transparent;
        border-radius: 9px;
    }
    QSlider#PlotModeSwitch::handle:horizontal {
        width: 14px;
        margin: 1px;
        border-radius: 7px;
        background: %(btn_text)s;
        border: 1px solid %(input_border)s;
    }
    QToolBar {
        background: transparent;
        border: none;
    }
    QMenu {
        font-family: %(font_family)s;
        background: %(menu_bg)s;
        color: %(menu_text)s;
        border: 1px solid %(menu_border)s;
        padding: 6px;
    }
    QMenu::item {
        padding: 8px 18px;
        border-radius: 6px;
        background: transparent;
    }
    QMenu::item:selected {
        background: %(menu_hover)s;
        color: %(menu_text_selected)s;
    }
    QMenu::separator {
        height: 1px;
        background: %(menu_separator)s;
        margin: 6px 8px;
    }
    QStatusBar {
        background: %(status_bg)s;
        border-top: 1px solid %(status_border)s;
        color: %(status_text)s;
    }
    QSplitter::handle {
        background: transparent;
    }
    QSplitter::handle:hover {
        background: %(splitter)s;
    }
    QSplitter::handle:horizontal {
        width: 1px;
        margin: 0;
    }
    QSplitter::handle:vertical {
        height: 1px;
        margin: 0;
    }
    QScrollArea {
        border: none;
    }
    QScrollBar:vertical {
        background: %(scroll_bg)s;
        width: 10px;
        margin: 2px;
        border-radius: 5px;
    }
    QScrollBar::handle:vertical {
        background: %(scroll_handle)s;
        min-height: 24px;
        border-radius: 5px;
    }
    QScrollBar::handle:vertical:hover {
        background: %(scroll_handle_hover)s;
    }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        height: 0px;
    }
    QScrollBar:horizontal {
        background: %(scroll_bg)s;
        height: 10px;
        margin: 2px;
        border-radius: 5px;
    }
    QScrollBar::handle:horizontal {
        background: %(scroll_handle)s;
        min-width: 24px;
        border-radius: 5px;
    }
    QScrollBar::handle:horizontal:hover {
        background: %(scroll_handle_hover)s;
    }
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
        width: 0px;
    }
"""


_DARK_TEMPLATE = """
    QWidget#RootContainer {
        background: qradialgradient(
            cx: 0.18, cy: 0.0, radius: 1.45,
            fx: 0.18, fy: 0.0,
            stop: 0 %(root0)s,
            stop: 0.34 %(root1)s,
            stop: 0.74 %(root2)s,
            stop: 1 %(root3)s
        );
    }
    QDialog {
        background: %(panel_bg)s;
    }
    QWidget {
        color: %(text)s;
    }
    QFrame#SidebarPanel,
    QFrame#TopControlsBar,
    QFrame#ActionsBar,
    QFrame#PlotCard,
    QFrame#TableCard,
    QFrame#PolarCard,
    QFrame#InfoCard {
        background: %(panel_bg)s;
        border: 1px solid %(panel_border)s;
    }
    QFrame#TopControlsBar,
    QFrame#ActionsBar {
        background: %(top_bg)s;
        border: 1px solid %(top_border)s;
    }
    QGroupBox {
        color: %(group_text)s;
        border: 1px solid %(group_border)s;
        background: %(group_bg)s;
    }
    QTabWidget::pane {
        background: %(tab_pane_bg)s;
        border: 1px solid %(group_border)s;
        border-radius: 10px;
        top: -1px;
    }
    QTabBar {
        background: transparent;
    }
    QTabBar::tab {
        background: %(tab_inactive_bg)s;
        color: %(strip_label)s;
        border: 1px solid %(panel_border)s;
        border-bottom: none;
        border-top-left-radius: 8px;
        border-top-right-radius: 8px;
        padding: 6px 14px;
        min-width: 48px;
    }
    QTabBar::tab:selected {
        background: %(tab_selected_bg)s;
        color: %(section_title)s;
        border: 1px solid %(group_border)s;
        border-bottom: 1px solid %(tab_selected_bg)s;
    }
    QTabBar::tab:!selected:hover {
        background: %(tab_hover_bg)s;
        color: %(section_title)s;
    }
    QLabel#SectionTitle {
        color: %(section_title)s;
        font-weight: 700;
        letter-spacing: 0.35px;
    }
    QLabel#SectionHint {
        color: %(strip_label)s;
        font-weight: 500;
    }
    QLabel[weather_chip="true"] {
        background: %(tab_inactive_bg)s;
        border: 1px solid %(panel_border)s;
        border-radius: 8px;
        padding: 4px 8px;
        font-weight: 600;
        color: %(section_title)s;
    }
    QLabel[weather_chip="true"][weather_chip_role="weather"] {
        background: %(chip_weather_bg)s;
        border-color: %(chip_weather_border)s;
    }
    QLabel[weather_chip="true"][weather_chip_role="context"] {
        background: %(chip_context_bg)s;
        border-color: %(chip_context_border)s;
    }
    QLabel[weather_chip="true"][weather_chip_role="clock"] {
        background: %(chip_clock_bg)s;
        border-color: %(chip_clock_border)s;
    }
    QLabel[weather_chip="true"][weather_chip_role="solar"] {
        background: %(chip_solar_bg)s;
        border-color: %(chip_solar_border)s;
    }
    QLabel[weather_chip="true"][weather_chip_role="lunar"] {
        background: %(chip_lunar_bg)s;
        border-color: %(chip_lunar_border)s;
    }
    QLabel[weather_chip="true"][weather_chip_series="temp"] {
        background: %(chip_series_temp_bg)s;
        border-color: %(chip_series_temp_border)s;
        color: %(chip_series_temp_text)s;
    }
    QLabel[weather_chip="true"][weather_chip_series="wind"] {
        background: %(chip_series_wind_bg)s;
        border-color: %(chip_series_wind_border)s;
        color: %(chip_series_wind_text)s;
    }
    QLabel[weather_chip="true"][weather_chip_series="cloud"] {
        background: %(chip_series_cloud_bg)s;
        border-color: %(chip_series_cloud_border)s;
        color: %(chip_series_cloud_text)s;
    }
    QLabel[weather_chip="true"][weather_chip_series="humidity"] {
        background: %(chip_series_humidity_bg)s;
        border-color: %(chip_series_humidity_border)s;
        color: %(chip_series_humidity_text)s;
    }
    QLabel[weather_chip="true"][weather_chip_series="pressure"] {
        background: %(chip_series_pressure_bg)s;
        border-color: %(chip_series_pressure_border)s;
        color: %(chip_series_pressure_text)s;
    }
    *[cutout_page="true"] {
        background: %(tab_pane_bg)s;
    }
    *[cutout_tool_col="true"] {
        background: %(tab_pane_bg)s;
        border-left: 1px solid %(group_border)s;
        border-top-right-radius: 10px;
        border-bottom-right-radius: 10px;
    }
    *[cutout_image="true"] {
        background: %(input_bg)s;
        border: 1px solid %(input_border)s;
        border-radius: 10px;
        color: %(strip_label)s;
        font-size: %(strip_font_size_pt)spt;
        font-weight: 500;
        padding: 4px;
    }
    QWidget#SessionStrip QLabel,
    QWidget#FiltersStrip QLabel {
        color: %(strip_label)s;
    }
    QHeaderView::section {
        font-family: %(display_font_family)s;
        background: qlineargradient(
            x1: 0, y1: 0, x2: 0, y2: 1,
            stop: 0 %(header0)s,
            stop: 0.52 %(header1)s,
            stop: 1 %(header2)s
        );
        color: %(header_text)s;
        font-weight: 600;
    }
    QTableCornerButton::section {
        background: %(corner_bg)s;
        border: none;
    }
    QLineEdit, QComboBox, QDateEdit, QSpinBox, QDoubleSpinBox, QTextEdit {
        background-color: %(input_bg)s;
        border: 1px solid %(input_border)s;
        color: %(input_text)s;
        border-radius: 7px;
        min-height: 20px;
    }
    QComboBox {
        padding-right: 24px;
    }
    QComboBox::drop-down, QDateEdit::drop-down {
        subcontrol-origin: border;
        subcontrol-position: top right;
        width: 20px;
        border-left: 1px solid %(panel_border)s;
        border-top-right-radius: 6px;
        border-bottom-right-radius: 6px;
        background: %(tab_inactive_bg)s;
    }
    QComboBox::drop-down:hover, QDateEdit::drop-down:hover {
        background: %(tab_hover_bg)s;
    }
    QComboBox::down-arrow, QDateEdit::down-arrow {
        image: %(combo_arrow_icon)s;
        width: 10px;
        height: 6px;
        margin-right: 7px;
    }
    QComboBox::down-arrow:disabled, QDateEdit::down-arrow:disabled {
        image: %(combo_arrow_disabled_icon)s;
    }
    QComboBox QAbstractItemView {
        background: %(tab_pane_bg)s;
        color: %(input_text)s;
        border: 1px solid %(group_border)s;
        border-radius: 8px;
        padding: 4px;
        outline: 0;
        selection-background-color: %(table_sel_bg)s;
        selection-color: %(table_sel_text)s;
    }
    QComboBox QAbstractItemView::item {
        min-height: 22px;
        padding: 4px 8px;
        border-radius: 6px;
    }
    QComboBox QAbstractItemView::item:hover {
        background: %(tool_hover)s;
    }
    QSpinBox, QDoubleSpinBox, QDateEdit {
        padding-right: 22px;
    }
    QSpinBox::up-button, QDoubleSpinBox::up-button, QDateEdit::up-button {
        subcontrol-origin: border;
        subcontrol-position: top right;
        width: 18px;
        border-left: 1px solid %(panel_border)s;
        border-bottom: 1px solid %(panel_border)s;
        border-top-right-radius: 6px;
        background: %(tab_inactive_bg)s;
    }
    QSpinBox::down-button, QDoubleSpinBox::down-button, QDateEdit::down-button {
        subcontrol-origin: border;
        subcontrol-position: bottom right;
        width: 18px;
        border-left: 1px solid %(panel_border)s;
        border-top: 1px solid %(panel_border)s;
        border-bottom-right-radius: 6px;
        background: %(tab_inactive_bg)s;
    }
    QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover, QDateEdit::up-button:hover,
    QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover, QDateEdit::down-button:hover {
        background: %(tab_hover_bg)s;
    }
    QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed, QDateEdit::up-button:pressed,
    QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed, QDateEdit::down-button:pressed {
        background: %(tab_selected_bg)s;
    }
    QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
        image: %(spin_up_icon)s;
        width: 10px;
        height: 6px;
    }
    QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
        image: %(spin_down_icon)s;
        width: 10px;
        height: 6px;
    }
    QSpinBox::up-arrow:disabled, QDoubleSpinBox::up-arrow:disabled {
        image: %(spin_up_disabled_icon)s;
    }
    QSpinBox::down-arrow:disabled, QDoubleSpinBox::down-arrow:disabled {
        image: %(spin_down_disabled_icon)s;
    }
    QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QDateEdit:focus, QLineEdit:focus, QTextEdit:focus {
        border: 1px solid %(input_focus)s;
    }
    QCalendarWidget {
        background: %(input_bg)s;
        border: 1px solid %(input_border)s;
        border-radius: 8px;
    }
    QCalendarWidget QWidget {
        background: %(input_bg)s;
        color: %(input_text)s;
    }
    QCalendarWidget QWidget#qt_calendar_navigationbar {
        background: %(panel_bg)s;
        border-bottom: 1px solid %(input_border)s;
    }
    QCalendarWidget QToolButton {
        color: %(input_text)s;
        background: %(tab_inactive_bg)s;
        border: 1px solid %(input_border)s;
        border-radius: 6px;
        padding: 2px 6px;
    }
    QCalendarWidget QToolButton:hover {
        background: %(tab_hover_bg)s;
        border: 1px solid %(input_focus)s;
    }
    QCalendarWidget QSpinBox {
        background: %(input_bg)s;
        color: %(input_text)s;
        border: 1px solid %(input_border)s;
        border-radius: 6px;
        padding: 2px 4px;
    }
    QCalendarWidget QDoubleSpinBox {
        background: %(input_bg)s;
        color: %(input_text)s;
        border: 1px solid %(input_border)s;
        border-radius: 6px;
        padding: 2px 4px;
    }
    QCalendarWidget QAbstractItemView {
        background: %(input_bg)s;
        color: %(input_text)s;
        selection-background-color: %(table_sel_bg)s;
        selection-color: %(table_sel_text)s;
        border: 1px solid %(input_border)s;
        outline: 0;
    }
    QTableView {
        background: %(table_bg)s;
        alternate-background-color: %(table_alt)s;
        border: 1px solid %(table_border)s;
        gridline-color: %(table_grid)s;
        color: %(base_text)s;
        selection-color: %(table_sel_text)s;
        selection-background-color: %(table_sel_bg)s;
        border-radius: 10px;
    }
    QPushButton {
        background: qlineargradient(
            x1: 0, y1: 0, x2: 1, y2: 0,
            stop: 0 %(btn0)s,
            stop: 0.5 %(btn1)s,
            stop: 1 %(btn2)s
        );
        color: %(btn_text)s;
        border: 1px solid %(btn_border)s;
        padding: 4px 16px 5px 12px;
        min-height: 24px;
        font-weight: 600;
    }
    QPushButton:hover {
        background: qlineargradient(
            x1: 0, y1: 0, x2: 1, y2: 0,
            stop: 0 %(btnh0)s,
            stop: 0.5 %(btnh1)s,
            stop: 1 %(btnh2)s
        );
        border-color: %(button_hover_glow)s;
    }
    QPushButton:pressed {
        background: %(btn_pressed)s;
    }
    QPushButton:disabled {
        background: %(tab_inactive_bg)s;
        color: %(strip_label)s;
        border: 1px solid %(panel_border)s;
    }
    QPushButton[weather_link="true"] {
        padding: 2px 8px;
        min-height: 22px;
        max-height: 22px;
        border-radius: 6px;
        font-weight: 500;
        font-size: 10pt;
    }
    QCheckBox {
        color: %(check_text)s;
    }
    QCheckBox::indicator {
        width: 15px;
        height: 15px;
        border-radius: 4px;
        border: 1px solid %(check_border)s;
        background: %(check_bg)s;
    }
    QCheckBox::indicator:checked {
        background: qlineargradient(
            x1: 0, y1: 0, x2: 1, y2: 1,
            stop: 0 %(check_checked0)s,
            stop: 1 %(check_checked1)s
        );
        border: 1px solid %(check_checked_border)s;
    }
    QProgressBar {
        border: 1px solid %(input_border)s;
        border-radius: 7px;
        background: %(input_bg)s;
        color: %(input_text)s;
        text-align: center;
        min-height: 18px;
    }
    QProgressBar::chunk {
        border-radius: 6px;
        margin: 1px;
        background: qlineargradient(
            x1: 0, y1: 0, x2: 1, y2: 0,
            stop: 0 %(btn1)s,
            stop: 1 %(btn2)s
        );
    }
    QProgressBar[weather_phase_chip="true"] {
        background: %(chip_lunar_bg)s;
        border: 1px solid %(chip_lunar_border)s;
        border-radius: 9px;
        color: %(chip_text)s;
        text-align: center;
        min-height: 26px;
        max-height: 26px;
        font-weight: 700;
    }
    QProgressBar[weather_phase_chip="true"]::chunk {
        border-radius: 8px;
        margin: 1px;
        background: qlineargradient(
            x1: 0, y1: 0, x2: 1, y2: 0,
            stop: 0 %(accent_secondary)s,
            stop: 1 %(plot_moon)s
        );
    }
    QSlider#PlotModeSwitch::groove:horizontal {
        height: 18px;
        border-radius: 9px;
        background: %(tab_inactive_bg)s;
        border: 1px solid %(panel_border)s;
    }
    QSlider#PlotModeSwitch::sub-page:horizontal {
        background: %(btn1)s;
        border-radius: 9px;
    }
    QSlider#PlotModeSwitch::add-page:horizontal {
        background: transparent;
        border-radius: 9px;
    }
    QSlider#PlotModeSwitch::handle:horizontal {
        width: 14px;
        margin: 1px;
        border-radius: 7px;
        background: %(btn_text)s;
        border: 1px solid %(input_border)s;
    }
    QToolButton:hover {
        background: %(tool_hover)s;
    }
    QToolButton#DateNavButton {
        background: %(input_bg)s;
        border: 1px solid %(input_border)s;
        border-radius: 8px;
        padding: 0px;
    }
    QToolButton#DateNavButton:hover {
        background: %(tool_hover)s;
        border: 1px solid %(input_focus)s;
    }
    QToolButton#DateNavButton:pressed {
        background: %(tab_selected_bg)s;
        border: 1px solid %(input_focus)s;
    }
    QToolBar {
        background: transparent;
        border: none;
    }
    QMenu {
        font-family: %(font_family)s;
        background: %(menu_bg)s;
        color: %(menu_text)s;
        border: 1px solid %(menu_border)s;
        padding: 6px;
    }
    QMenu::item {
        padding: 8px 18px;
        border-radius: 6px;
        background: transparent;
    }
    QMenu::item:selected {
        background: %(menu_hover)s;
        color: %(menu_text_selected)s;
    }
    QMenu::separator {
        height: 1px;
        background: %(menu_separator)s;
        margin: 6px 8px;
    }
    QStatusBar {
        background: %(status_bg)s;
        border-top: 1px solid %(status_border)s;
        color: %(status_text)s;
    }
    QSplitter::handle {
        background: transparent;
    }
    QSplitter::handle:hover {
        background: %(splitter)s;
    }
    QSplitter::handle:horizontal {
        width: 1px;
        margin: 0;
    }
    QSplitter::handle:vertical {
        height: 1px;
        margin: 0;
    }
    QScrollArea {
        border: none;
    }
    QScrollBar:vertical {
        background: %(scroll_bg)s;
        width: 10px;
        margin: 2px;
        border-radius: 5px;
    }
    QScrollBar::handle:vertical {
        background: %(scroll_handle)s;
        min-height: 24px;
        border-radius: 5px;
    }
    QScrollBar::handle:vertical:hover {
        background: %(scroll_handle_hover)s;
    }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        height: 0px;
    }
    QScrollBar:horizontal {
        background: %(scroll_bg)s;
        height: 10px;
        margin: 2px;
        border-radius: 5px;
    }
    QScrollBar::handle:horizontal {
        background: %(scroll_handle)s;
        min-width: 24px;
        border-radius: 5px;
    }
    QScrollBar::handle:horizontal:hover {
        background: %(scroll_handle_hover)s;
    }
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
        width: 0px;
    }
"""


BASE_LIGHT = {
    "font_family": '"SF Pro Text", "Avenir Next", "Inter", "Segoe UI", "Noto Sans", "Helvetica Neue", "Arial"',
    "display_font_family": '"SF Pro Display", "Avenir Next", "Inter", "Segoe UI Semibold", "Helvetica Neue", "Arial"',
    "base_text": "#1f2d3d",
    "root0": "#eef3fb",
    "root1": "#f6f9fd",
    "root2": "#edf5f7",
    "panel_bg": "rgba(255, 255, 255, 0.92)",
    "panel_border": "#c9d6e4",
    "top_bg": "rgba(246, 250, 255, 0.94)",
    "top_border": "#b6c8dc",
    "group_text": "#2f4a66",
    "group_border": "#d3deea",
    "group_bg": "rgba(247, 250, 254, 0.96)",
    "tab_pane_bg": "#f7fbff",
    "tab_selected_bg": "#f7fbff",
    "tab_inactive_bg": "#edf3fa",
    "tab_hover_bg": "#f1f6fb",
    "section_title": "#2e4f70",
    "section_hint": "#6f8399",
    "strip_label": "#3b5878",
    "btn0": "#5f7e9a",
    "btn1": "#6f96ad",
    "btn2": "#7baea9",
    "btn_text": "#f7fbff",
    "btn_border": "rgba(38, 57, 77, 0.22)",
    "btnh0": "#55718c",
    "btnh1": "#64899f",
    "btnh2": "#6f9c97",
    "btn_pressed": "#48627a",
    "btn_disabled_bg": "#dce5ef",
    "btn_disabled_text": "#8195aa",
    "btn_disabled_border": "#c3d1df",
    "tool_hover": "rgba(83, 120, 160, 0.17)",
    "input_bg": "#ffffff",
    "input_border": "#bdcddd",
    "input_focus": "#6d8fb1",
    "input_text": "#1f2d3d",
    "combo_drop_bg": "#edf3fa",
    "combo_drop_hover": "#e0ebf7",
    "combo_drop_border": "#afc2d4",
    "combo_popup_bg": "#f7fafd",
    "combo_popup_text": "#213548",
    "combo_popup_border": "#b9cada",
    "combo_popup_sel_bg": "rgba(111, 150, 173, 0.30)",
    "combo_popup_sel_text": "#1f2d3d",
    "combo_popup_hover": "rgba(111, 150, 173, 0.18)",
    "spin_btn_bg": "#e7edf5",
    "spin_btn_hover": "#dbe6f1",
    "spin_btn_pressed": "#cfdeec",
    "spin_btn_border": "#afc2d4",
    "table_bg": "rgba(255, 255, 255, 0.98)",
    "table_alt": "rgba(237, 243, 249, 0.75)",
    "table_border": "#c7d3e0",
    "table_grid": "#e1e8f0",
    "table_sel_text": "#1f2d3d",
    "table_sel_bg": "rgba(111, 150, 173, 0.22)",
    "table_sel_item": "rgba(111, 150, 173, 0.28)",
    "header0": "#425d78",
    "header1": "#344c64",
    "header_text": "#f4f8fc",
    "check_text": "#344f6d",
    "check_bg": "#ffffff",
    "check_border": "#a8bfd6",
    "check_checked0": "#6d8fad",
    "check_checked1": "#7ea7a5",
    "check_checked_border": "#6b879f",
    "progress_bg": "#e9f0f7",
    "progress_border": "#b7c8d9",
    "progress_text": "#1f2d3d",
    "progress_chunk0": "#6d8fad",
    "progress_chunk1": "#7ea7a5",
    "menu_bg": "#f7fafc",
    "menu_text": "#22364b",
    "menu_border": "#bccddd",
    "menu_hover": "rgba(111, 150, 173, 0.28)",
    "menu_text_selected": "#1f2d3d",
    "menu_separator": "#cdd9e6",
    "status_bg": "rgba(255, 255, 255, 0.92)",
    "status_border": "#cdd9e6",
    "status_text": "#22364b",
    "splitter": "rgba(124, 145, 168, 0.32)",
    "scroll_bg": "rgba(236, 242, 249, 0.90)",
    "scroll_handle": "rgba(126, 153, 180, 0.62)",
    "scroll_handle_hover": "rgba(110, 136, 162, 0.78)",
    "corner_bg": "#344c64",
}

BASE_DARK = {
    "font_family": '"SF Pro Text", "Avenir Next", "Inter", "Segoe UI", "Noto Sans", "Helvetica Neue", "Arial"',
    "display_font_family": '"SF Pro Display", "Avenir Next", "Inter", "Segoe UI Semibold", "Helvetica Neue", "Arial"',
    "root0": "#22344f",
    "root1": "#17263d",
    "root2": "#0f192b",
    "root3": "#0a111d",
    "text": "#d7e4f0",
    "panel_bg": "rgba(18, 27, 41, 0.88)",
    "panel_border": "rgba(118, 147, 179, 0.34)",
    "top_bg": "rgba(15, 23, 36, 0.92)",
    "top_border": "rgba(118, 147, 179, 0.46)",
    "group_text": "#a7bed5",
    "group_border": "rgba(118, 147, 179, 0.40)",
    "group_bg": "rgba(14, 22, 34, 0.95)",
    "tab_pane_bg": "#162131",
    "tab_selected_bg": "#162131",
    "tab_inactive_bg": "#1d2a3d",
    "tab_hover_bg": "#223246",
    "section_title": "#dbe9f7",
    "section_hint": "#8ea3b9",
    "strip_label": "#9db4cb",
    "header0": "#354b63",
    "header1": "#2a3c52",
    "header2": "#223244",
    "header_text": "#ebf2fa",
    "corner_bg": "#2a3c52",
    "input_bg": "rgba(17, 27, 40, 0.96)",
    "input_border": "rgba(118, 147, 179, 0.44)",
    "input_text": "#d7e4f0",
    "input_focus": "#8aa9c7",
    "combo_drop_bg": "rgba(40, 57, 76, 0.96)",
    "combo_drop_hover": "rgba(52, 72, 94, 0.98)",
    "combo_drop_border": "rgba(118, 147, 179, 0.64)",
    "combo_popup_bg": "#1c2938",
    "combo_popup_text": "#e5eef7",
    "combo_popup_border": "#5b7692",
    "combo_popup_sel_bg": "rgba(120, 152, 183, 0.40)",
    "combo_popup_sel_text": "#f5fbff",
    "combo_popup_hover": "rgba(120, 152, 183, 0.24)",
    "spin_btn_bg": "rgba(41, 57, 75, 0.96)",
    "spin_btn_hover": "rgba(52, 70, 90, 0.98)",
    "spin_btn_pressed": "rgba(34, 48, 65, 0.98)",
    "spin_btn_border": "rgba(118, 147, 179, 0.64)",
    "table_bg": "rgba(14, 22, 34, 0.96)",
    "table_alt": "rgba(18, 28, 42, 0.94)",
    "table_border": "rgba(118, 147, 179, 0.44)",
    "table_grid": "rgba(118, 147, 179, 0.22)",
    "table_sel_text": "#f2f7fb",
    "table_sel_bg": "rgba(120, 152, 183, 0.28)",
    "btn0": "#5e7895",
    "btn1": "#6f8fa9",
    "btn2": "#7aa3a4",
    "btn_text": "#f6fbff",
    "btn_border": "rgba(215, 228, 240, 0.24)",
    "btnh0": "#526a85",
    "btnh1": "#63829a",
    "btnh2": "#6f9496",
    "btn_pressed": "#43576f",
    "btn_disabled_bg": "rgba(58, 72, 90, 0.70)",
    "btn_disabled_text": "rgba(201, 214, 227, 0.62)",
    "btn_disabled_border": "rgba(132, 156, 182, 0.38)",
    "check_text": "#d7e4f0",
    "check_bg": "rgba(16, 25, 37, 0.96)",
    "check_border": "rgba(118, 147, 179, 0.56)",
    "check_checked0": "#7f9eb9",
    "check_checked1": "#88aca7",
    "check_checked_border": "rgba(215, 228, 240, 0.72)",
    "progress_bg": "rgba(20, 31, 45, 0.94)",
    "progress_border": "rgba(118, 147, 179, 0.56)",
    "progress_text": "#e4eef8",
    "progress_chunk0": "#7f9eb9",
    "progress_chunk1": "#88aca7",
    "menu_bg": "#1a2533",
    "menu_text": "#e4edf6",
    "menu_border": "#5f7894",
    "menu_hover": "rgba(120, 152, 183, 0.34)",
    "menu_text_selected": "#f6fbff",
    "menu_separator": "#405469",
    "tool_hover": "rgba(120, 152, 183, 0.22)",
    "status_bg": "rgba(13, 20, 31, 0.95)",
    "status_border": "rgba(118, 147, 179, 0.38)",
    "status_text": "#d7e4f0",
    "splitter": "rgba(120, 152, 183, 0.24)",
    "scroll_bg": "rgba(12, 20, 31, 0.86)",
    "scroll_handle": "rgba(120, 152, 183, 0.56)",
    "scroll_handle_hover": "rgba(100, 128, 153, 0.78)",
}


THEMES: dict[str, UiTheme] = {
    "graphite": UiTheme(
        key="graphite",
        label="Graphite Grid",
        light=_merge(
            BASE_LIGHT,
            base_text="#2b3138",
            root0="#f1f3f6",
            root1="#f8f9fb",
            root2="#efefee",
            panel_border="#ced3da",
            top_border="#bec5cf",
            group_text="#404a56",
            section_title="#3b4654",
            tab_pane_bg="#f4f6f8",
            tab_selected_bg="#f4f6f8",
            tab_inactive_bg="#eceff3",
            tab_hover_bg="#f6f8fb",
            strip_label="#4b5664",
            btn0="#6b7482",
            btn1="#7a8592",
            btn2="#8b949f",
            btnh0="#5f6875",
            btnh1="#707a86",
            btnh2="#818b94",
            btn_pressed="#525a66",
            tool_hover="rgba(101, 112, 128, 0.18)",
            input_border="#c1c8d2",
            input_focus="#7b8794",
            table_alt="rgba(238, 241, 244, 0.76)",
            table_border="#c8ced6",
            table_grid="#e3e7ec",
            table_sel_bg="rgba(139, 152, 166, 0.22)",
            table_sel_item="rgba(139, 152, 166, 0.28)",
            header0="#4a5463",
            header1="#3d4654",
            check_text="#505b68",
            check_border="#aeb7c2",
            check_checked0="#7c8793",
            check_checked1="#8b949e",
            check_checked_border="#76818d",
            splitter="rgba(127, 136, 148, 0.34)",
            scroll_handle="rgba(132, 142, 154, 0.62)",
            scroll_handle_hover="rgba(113, 122, 134, 0.78)",
        ),
        dark=_merge(
            BASE_DARK,
            root0="#353c47",
            root1="#2a3039",
            root2="#1d222a",
            root3="#12161d",
            text="#e3e7ec",
            panel_bg="rgba(27, 31, 38, 0.90)",
            panel_border="rgba(147, 159, 174, 0.34)",
            top_bg="rgba(24, 28, 34, 0.94)",
            top_border="rgba(147, 159, 174, 0.46)",
            group_text="#c2cbd5",
            group_border="rgba(147, 159, 174, 0.40)",
            group_bg="rgba(23, 27, 33, 0.95)",
            tab_pane_bg="#1b1f26",
            tab_selected_bg="#1b1f26",
            tab_inactive_bg="#21262e",
            tab_hover_bg="#282d36",
            section_title="#edf1f6",
            strip_label="#b7c0cb",
            header0="#4b5564",
            header1="#3d4653",
            header2="#333b46",
            header_text="#f1f4f8",
            corner_bg="#3d4653",
            input_bg="rgba(26, 31, 37, 0.96)",
            input_border="rgba(147, 159, 174, 0.44)",
            input_text="#e3e7ec",
            input_focus="#9aa6b3",
            table_bg="rgba(22, 26, 33, 0.96)",
            table_alt="rgba(27, 32, 39, 0.94)",
            table_border="rgba(147, 159, 174, 0.44)",
            table_grid="rgba(147, 159, 174, 0.22)",
            table_sel_bg="rgba(139, 152, 166, 0.28)",
            btn0="#6e7885",
            btn1="#7b8692",
            btn2="#8b949d",
            btnh0="#626b78",
            btnh1="#717b86",
            btnh2="#7f8992",
            btn_pressed="#555d68",
            check_text="#e3e7ec",
            check_bg="rgba(24, 28, 34, 0.96)",
            check_border="rgba(147, 159, 174, 0.56)",
            check_checked0="#8994a0",
            check_checked1="#949ea8",
            tool_hover="rgba(139, 152, 166, 0.22)",
            status_bg="rgba(19, 23, 29, 0.95)",
            status_border="rgba(147, 159, 174, 0.38)",
            status_text="#e3e7ec",
            splitter="rgba(139, 152, 166, 0.24)",
            scroll_bg="rgba(18, 22, 28, 0.86)",
            scroll_handle="rgba(139, 152, 166, 0.56)",
            scroll_handle_hover="rgba(115, 126, 138, 0.78)",
        ),
    ),
    "laguna": UiTheme(
        key="laguna",
        label="Neon Harbor",
        light=_merge(
            BASE_LIGHT,
            base_text="#173740",
            root0="#ecfbff",
            root1="#f7fdff",
            root2="#eef8fb",
            panel_border="#bddde6",
            top_border="#a7ced9",
            group_text="#1d5663",
            group_border="#cce6ed",
            tab_pane_bg="#f4fcff",
            tab_selected_bg="#f4fcff",
            tab_inactive_bg="#e3f3f7",
            tab_hover_bg="#f7feff",
            section_title="#165f70",
            strip_label="#2b6e7a",
            btn0="#0f7892",
            btn1="#24a0b5",
            btn2="#ff4fd8",
            btnh0="#0d6980",
            btnh1="#1f8d9f",
            btnh2="#ea3fc2",
            btn_pressed="#0b596c",
            tool_hover="rgba(36, 160, 181, 0.18)",
            input_border="#aacfd8",
            input_focus="#20a2bb",
            table_alt="rgba(227, 246, 251, 0.80)",
            table_border="#c2dee7",
            table_grid="#d7ebf1",
            table_sel_bg="rgba(36, 160, 181, 0.18)",
            table_sel_item="rgba(255, 138, 91, 0.18)",
            header0="#1c7084",
            header1="#145d6d",
            check_text="#255d6c",
            check_border="#98c1cb",
            check_checked0="#1994ad",
            check_checked1="#ff63e0",
            check_checked_border="#167d90",
            menu_hover="rgba(36, 160, 181, 0.20)",
            splitter="rgba(70, 149, 166, 0.30)",
            scroll_handle="rgba(70, 149, 166, 0.56)",
            scroll_handle_hover="rgba(49, 123, 138, 0.76)",
        ),
        dark=_merge(
            BASE_DARK,
            root0="#17404f",
            root1="#0d2731",
            root2="#07171d",
            root3="#040f13",
            text="#def8ff",
            panel_bg="rgba(8, 24, 31, 0.90)",
            panel_border="rgba(93, 183, 202, 0.34)",
            top_bg="rgba(7, 20, 25, 0.94)",
            top_border="rgba(93, 183, 202, 0.46)",
            group_text="#a4e2ef",
            group_border="rgba(93, 183, 202, 0.40)",
            group_bg="rgba(8, 19, 24, 0.95)",
            tab_pane_bg="#0d2028",
            tab_selected_bg="#0d2028",
            tab_inactive_bg="#14313b",
            tab_hover_bg="#1a3d49",
            section_title="#eefcff",
            strip_label="#8bd0de",
            header0="#1a7892",
            header1="#155f74",
            header2="#114c5c",
            header_text="#f1fdff",
            corner_bg="#155f74",
            input_bg="rgba(9, 24, 29, 0.96)",
            input_border="rgba(93, 183, 202, 0.44)",
            input_text="#def8ff",
            input_focus="#31c5dd",
            table_bg="rgba(8, 18, 23, 0.96)",
            table_alt="rgba(10, 26, 33, 0.94)",
            table_border="rgba(93, 183, 202, 0.44)",
            table_grid="rgba(93, 183, 202, 0.18)",
            table_sel_bg="rgba(49, 197, 221, 0.20)",
            btn0="#1590aa",
            btn1="#23aec0",
            btn2="#ff4fd8",
            btnh0="#117d94",
            btnh1="#1d98a8",
            btnh2="#e842c2",
            btn_pressed="#0d687b",
            check_text="#def8ff",
            check_bg="rgba(9, 22, 27, 0.96)",
            check_border="rgba(93, 183, 202, 0.56)",
            check_checked0="#22b1c6",
            check_checked1="#ff63e0",
            tool_hover="rgba(49, 197, 221, 0.18)",
            status_bg="rgba(6, 16, 20, 0.95)",
            status_border="rgba(93, 183, 202, 0.38)",
            status_text="#def8ff",
            splitter="rgba(49, 197, 221, 0.22)",
            scroll_bg="rgba(7, 17, 21, 0.86)",
            scroll_handle="rgba(49, 197, 221, 0.44)",
            scroll_handle_hover="rgba(35, 155, 176, 0.72)",
        ),
    ),
    "terracotta": UiTheme(
        key="terracotta",
        label="Ember Circuit",
        light=_merge(
            BASE_LIGHT,
            base_text="#4a3028",
            root0="#fff3eb",
            root1="#fffaf5",
            root2="#fdf0e7",
            panel_border="#f0d1c1",
            top_border="#e5bba6",
            group_text="#7c4738",
            group_border="#f3ddd0",
            tab_pane_bg="#fff8f2",
            tab_selected_bg="#fff8f2",
            tab_inactive_bg="#fde8dc",
            tab_hover_bg="#fff1e7",
            section_title="#a8543c",
            strip_label="#8f6147",
            btn0="#c96a43",
            btn1="#d98d46",
            btn2="#e3b75d",
            btnh0="#b85d39",
            btnh1="#c77d3f",
            btnh2="#d5a84f",
            btn_pressed="#a24f31",
            tool_hover="rgba(201, 106, 67, 0.18)",
            input_border="#e7c0ae",
            input_focus="#d97649",
            table_alt="rgba(255, 238, 226, 0.80)",
            table_border="#eed1c4",
            table_grid="#f6e3da",
            table_sel_bg="rgba(217, 118, 73, 0.18)",
            table_sel_item="rgba(227, 183, 93, 0.20)",
            header0="#b45d3e",
            header1="#944d35",
            check_text="#884d3e",
            check_border="#d8ae9a",
            check_checked0="#d97346",
            check_checked1="#dfb25d",
            check_checked_border="#b5603d",
            menu_hover="rgba(217, 118, 73, 0.22)",
            splitter="rgba(193, 119, 89, 0.30)",
            scroll_handle="rgba(207, 137, 99, 0.56)",
            scroll_handle_hover="rgba(177, 97, 63, 0.76)",
        ),
        dark=_merge(
            BASE_DARK,
            root0="#563126",
            root1="#381d18",
            root2="#24120f",
            root3="#160b09",
            text="#ffe9df",
            panel_bg="rgba(33, 17, 14, 0.90)",
            panel_border="rgba(226, 136, 90, 0.34)",
            top_bg="rgba(29, 14, 12, 0.94)",
            top_border="rgba(226, 136, 90, 0.46)",
            group_text="#ffc4a7",
            group_border="rgba(226, 136, 90, 0.40)",
            group_bg="rgba(27, 13, 11, 0.95)",
            tab_pane_bg="#271310",
            tab_selected_bg="#271310",
            tab_inactive_bg="#341a15",
            tab_hover_bg="#40211c",
            section_title="#fff2ea",
            strip_label="#f2b797",
            header0="#bf6544",
            header1="#9a4c33",
            header2="#7c3b28",
            header_text="#fff7f2",
            corner_bg="#9a4c33",
            input_bg="rgba(32, 16, 13, 0.96)",
            input_border="rgba(226, 136, 90, 0.44)",
            input_text="#ffe9df",
            input_focus="#ff9a68",
            table_bg="rgba(27, 13, 11, 0.96)",
            table_alt="rgba(34, 17, 14, 0.94)",
            table_border="rgba(226, 136, 90, 0.44)",
            table_grid="rgba(226, 136, 90, 0.18)",
            table_sel_bg="rgba(255, 154, 104, 0.22)",
            btn0="#d36e48",
            btn1="#e08f45",
            btn2="#e6b65d",
            btnh0="#bc5e3d",
            btnh1="#cb7d3d",
            btnh2="#d8a54f",
            btn_pressed="#9a492f",
            check_text="#ffe9df",
            check_bg="rgba(28, 13, 11, 0.96)",
            check_border="rgba(226, 136, 90, 0.56)",
            check_checked0="#e57d4d",
            check_checked1="#e5b765",
            tool_hover="rgba(255, 154, 104, 0.20)",
            status_bg="rgba(21, 10, 8, 0.95)",
            status_border="rgba(226, 136, 90, 0.38)",
            status_text="#ffe9df",
            splitter="rgba(255, 154, 104, 0.22)",
            scroll_bg="rgba(19, 9, 8, 0.86)",
            scroll_handle="rgba(255, 154, 104, 0.46)",
            scroll_handle_hover="rgba(215, 110, 69, 0.72)",
        ),
    ),
    "violet": UiTheme(
        key="violet",
        label="Violet Pulse",
        light=_merge(
            BASE_LIGHT,
            base_text="#2d2750",
            root0="#f5f0ff",
            root1="#fcf9ff",
            root2="#f7f2ff",
            panel_border="#d8cef0",
            top_border="#c7bbea",
            group_text="#4e477d",
            group_border="#e2daf6",
            tab_pane_bg="#faf7ff",
            tab_selected_bg="#faf7ff",
            tab_inactive_bg="#eee6ff",
            tab_hover_bg="#f6f1ff",
            section_title="#5b43bf",
            strip_label="#6c55ce",
            btn0="#4b63ff",
            btn1="#8a45d8",
            btn2="#11b8d5",
            btnh0="#4056e3",
            btnh1="#793bc2",
            btnh2="#0fa3be",
            btn_pressed="#3548b7",
            tool_hover="rgba(138, 69, 216, 0.18)",
            input_border="#cac0e8",
            input_focus="#7467ff",
            table_alt="rgba(241, 235, 255, 0.80)",
            table_border="#dbd2f1",
            table_grid="#e8e1f7",
            table_sel_bg="rgba(116, 103, 255, 0.18)",
            table_sel_item="rgba(17, 184, 213, 0.20)",
            header0="#4c59d2",
            header1="#6a40b0",
            check_text="#554d89",
            check_border="#b7add8",
            check_checked0="#626cff",
            check_checked1="#18b5d2",
            check_checked_border="#5158cf",
            menu_hover="rgba(138, 69, 216, 0.20)",
            splitter="rgba(105, 95, 209, 0.30)",
            scroll_handle="rgba(105, 95, 209, 0.56)",
            scroll_handle_hover="rgba(84, 72, 184, 0.76)",
        ),
        dark=_merge(
            BASE_DARK,
            root0="#2b2058",
            root1="#19113a",
            root2="#100a23",
            root3="#08050f",
            text="#f1ebff",
            panel_bg="rgba(18, 14, 36, 0.90)",
            panel_border="rgba(125, 112, 232, 0.34)",
            top_bg="rgba(14, 11, 28, 0.94)",
            top_border="rgba(125, 112, 232, 0.46)",
            group_text="#d2caff",
            group_border="rgba(125, 112, 232, 0.40)",
            group_bg="rgba(16, 12, 31, 0.95)",
            tab_pane_bg="#191430",
            tab_selected_bg="#191430",
            tab_inactive_bg="#241c45",
            tab_hover_bg="#2d2455",
            section_title="#f5f1ff",
            strip_label="#bfb5ff",
            header0="#5662e7",
            header1="#7b42c8",
            header2="#2f8ab2",
            header_text="#f8f5ff",
            corner_bg="#7b42c8",
            input_bg="rgba(18, 14, 34, 0.96)",
            input_border="rgba(125, 112, 232, 0.44)",
            input_text="#f1ebff",
            input_focus="#9183ff",
            table_bg="rgba(15, 11, 29, 0.96)",
            table_alt="rgba(20, 15, 38, 0.94)",
            table_border="rgba(125, 112, 232, 0.44)",
            table_grid="rgba(125, 112, 232, 0.18)",
            table_sel_bg="rgba(145, 131, 255, 0.22)",
            btn0="#4f66ff",
            btn1="#8a4bda",
            btn2="#15b8d5",
            btnh0="#4358e3",
            btnh1="#7a3fc4",
            btnh2="#10a2be",
            btn_pressed="#3343a8",
            check_text="#f1ebff",
            check_bg="rgba(17, 13, 31, 0.96)",
            check_border="rgba(125, 112, 232, 0.56)",
            check_checked0="#6f78ff",
            check_checked1="#1bbad6",
            tool_hover="rgba(145, 131, 255, 0.20)",
            status_bg="rgba(12, 9, 24, 0.95)",
            status_border="rgba(125, 112, 232, 0.38)",
            status_text="#f1ebff",
            splitter="rgba(145, 131, 255, 0.22)",
            scroll_bg="rgba(11, 8, 22, 0.86)",
            scroll_handle="rgba(145, 131, 255, 0.44)",
            scroll_handle_hover="rgba(112, 96, 228, 0.72)",
        ),
    ),
    "rosewood": UiTheme(
        key="rosewood",
        label="Rose Noir",
        light=_merge(
            BASE_LIGHT,
            base_text="#412b36",
            root0="#fff2f6",
            root1="#fffafd",
            root2="#fceff4",
            panel_border="#ecccd8",
            top_border="#dfb2c3",
            group_text="#7a4458",
            group_border="#f2d8e2",
            tab_pane_bg="#fff8fb",
            tab_selected_bg="#fff8fb",
            tab_inactive_bg="#f9e8ef",
            tab_hover_bg="#fff2f7",
            section_title="#9f4f6c",
            strip_label="#8a5969",
            btn0="#8f355c",
            btn1="#c36f8c",
            btn2="#d6a45a",
            btnh0="#7b2d4f",
            btnh1="#ae5d79",
            btnh2="#c3934d",
            btn_pressed="#692541",
            tool_hover="rgba(195, 111, 140, 0.18)",
            input_border="#e3bfd0",
            input_focus="#bf5d87",
            table_alt="rgba(251, 234, 241, 0.80)",
            table_border="#ecd0dc",
            table_grid="#f5e2ea",
            table_sel_bg="rgba(191, 93, 135, 0.18)",
            table_sel_item="rgba(214, 164, 90, 0.18)",
            header0="#a04c70",
            header1="#7d3653",
            check_text="#864d61",
            check_border="#d6aebf",
            check_checked0="#c46089",
            check_checked1="#cf9f57",
            check_checked_border="#a65173",
            menu_hover="rgba(195, 111, 140, 0.20)",
            splitter="rgba(167, 106, 128, 0.30)",
            scroll_handle="rgba(182, 123, 145, 0.56)",
            scroll_handle_hover="rgba(151, 85, 110, 0.76)",
        ),
        dark=_merge(
            BASE_DARK,
            root0="#4c2336",
            root1="#311523",
            root2="#1f0d17",
            root3="#12070d",
            text="#ffeaf1",
            panel_bg="rgba(31, 14, 22, 0.90)",
            panel_border="rgba(196, 111, 141, 0.34)",
            top_bg="rgba(26, 11, 18, 0.94)",
            top_border="rgba(196, 111, 141, 0.46)",
            group_text="#ffc7d8",
            group_border="rgba(196, 111, 141, 0.40)",
            group_bg="rgba(24, 10, 17, 0.95)",
            tab_pane_bg="#22101a",
            tab_selected_bg="#22101a",
            tab_inactive_bg="#311823",
            tab_hover_bg="#3c1f2c",
            section_title="#fff1f6",
            strip_label="#f0b3c8",
            header0="#a74f72",
            header1="#7f3652",
            header2="#8f6a2c",
            header_text="#fff7fa",
            corner_bg="#7f3652",
            input_bg="rgba(29, 13, 20, 0.96)",
            input_border="rgba(196, 111, 141, 0.44)",
            input_text="#ffeaf1",
            input_focus="#d7759f",
            table_bg="rgba(24, 10, 17, 0.96)",
            table_alt="rgba(31, 14, 22, 0.94)",
            table_border="rgba(196, 111, 141, 0.44)",
            table_grid="rgba(196, 111, 141, 0.18)",
            table_sel_bg="rgba(215, 117, 159, 0.22)",
            btn0="#94365e",
            btn1="#c46d8b",
            btn2="#d6a35a",
            btnh0="#802c50",
            btnh1="#af5b77",
            btnh2="#c3924c",
            btn_pressed="#6d2543",
            check_text="#ffeaf1",
            check_bg="rgba(25, 10, 18, 0.96)",
            check_border="rgba(196, 111, 141, 0.56)",
            check_checked0="#cf6f98",
            check_checked1="#d8a55a",
            tool_hover="rgba(215, 117, 159, 0.20)",
            status_bg="rgba(18, 8, 13, 0.95)",
            status_border="rgba(196, 111, 141, 0.38)",
            status_text="#ffeaf1",
            splitter="rgba(215, 117, 159, 0.22)",
            scroll_bg="rgba(16, 7, 12, 0.86)",
            scroll_handle="rgba(215, 117, 159, 0.46)",
            scroll_handle_hover="rgba(176, 88, 124, 0.72)",
        ),
    ),
}

DEFAULT_UI_THEME = "violet"
DEFAULT_DARK_MODE = True
_DEFAULT_THEME_TOKEN_OVERRIDES: dict[str, dict[str, str]] = {
    "violet": {
        "btn2": "#f74dff",
    },
}
THEME_CHOICES: tuple[tuple[str, str], ...] = tuple((key, theme.label) for key, theme in THEMES.items())


def normalize_theme_key(theme_key: str | None) -> str:
    if theme_key and theme_key in THEMES:
        return theme_key
    return DEFAULT_UI_THEME


def _format_pt(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.1f}"


def _font_tokens(ui_font_size: int | float) -> dict[str, str]:
    base_size = max(9.0, min(16.0, float(ui_font_size)))
    strip_size = max(9.5, base_size - 0.5)
    section_size = max(11.0, base_size + 1.0)
    return {
        "font_size_pt": _format_pt(base_size),
        "strip_font_size_pt": _format_pt(strip_size),
        "section_font_size_pt": _format_pt(section_size),
    }


def _quote_qss_font_family(family: str) -> str:
    escaped = family.replace('"', '\\"')
    return f'"{escaped}"'


def _svg_data_url(svg: str) -> str:
    return f'url("data:image/svg+xml;utf8,{quote(svg)}")'


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    text = str(value or "").strip()
    if text.startswith("#"):
        text = text[1:]
    if len(text) == 3:
        text = "".join(ch * 2 for ch in text)
    if len(text) != 6:
        return (127, 127, 127)
    try:
        return tuple(int(text[idx : idx + 2], 16) for idx in (0, 2, 4))  # type: ignore[return-value]
    except ValueError:
        return (127, 127, 127)


def _rgba(value: str, alpha: float) -> str:
    r, g, b = _hex_to_rgb(value)
    return f"rgba({r}, {g}, {b}, {max(0.0, min(1.0, float(alpha))):.3f})"


def _mix_hex(first: str, second: str, first_ratio: float) -> str:
    ratio = max(0.0, min(1.0, float(first_ratio)))
    other_ratio = 1.0 - ratio
    r1, g1, b1 = _hex_to_rgb(first)
    r2, g2, b2 = _hex_to_rgb(second)
    return "#{:02x}{:02x}{:02x}".format(
        int(round(r1 * ratio + r2 * other_ratio)),
        int(round(g1 * ratio + g2 * other_ratio)),
        int(round(b1 * ratio + b2 * other_ratio)),
    )


def _root_overlay_icon(line_color: str, scan_color: str) -> str:
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="84" height="84" viewBox="0 0 84 84">'
        '<rect width="84" height="84" fill="none"/>'
        f'<path d="M0 0.5H84M0 28.5H84M0 56.5H84M0.5 0V84M28.5 0V84M56.5 0V84" '
        f'stroke="{line_color}" stroke-width="1" opacity="0.22"/>'
        f'<path d="M0 14.5H84M0 42.5H84M0 70.5H84" stroke="{scan_color}" stroke-width="0.8" opacity="0.14"/>'
        "</svg>"
    )
    return _svg_data_url(svg)


_THEME_LINE_PALETTES: dict[str, tuple[str, ...]] = {
    "graphite": ("#59f3ff", "#ff9a5f", "#8cff84", "#ff4da6", "#9f7cff", "#f6d365", "#3ac7d8", "#d9e4f2", "#ff6d5d", "#5ca8ff"),
    "laguna": ("#22f0ff", "#ff4fd8", "#a0ff8f", "#ffe45e", "#7b8cff", "#ff5f87", "#5ff7d2", "#c9f6ff", "#ff7de7", "#76d1ff"),
    "terracotta": ("#ff8b57", "#ffd166", "#58f0d3", "#ff5f8e", "#ffb86a", "#d3ff7a", "#8c7cff", "#ffe2bd", "#ff6a3d", "#80e3ff"),
    "violet": ("#9a6bff", "#ff4fd8", "#1be7ff", "#ffd166", "#7affd2", "#ff7aa2", "#5f89ff", "#d4c2ff", "#ff9c52", "#9bf6ff"),
    "rosewood": ("#ff6fae", "#f6c667", "#7df9ff", "#cf8bff", "#ff8c69", "#ffe580", "#84ffd6", "#ffd7ea", "#ff5e86", "#8fc6ff"),
}

_CYBERPUNK_COLORBLIND_PALETTE: tuple[str, ...] = (
    "#00c8ff",
    "#ff9f1c",
    "#2dd4bf",
    "#ff5d8f",
    "#a78bfa",
    "#ffe066",
    "#38bdf8",
    "#95f985",
    "#f97316",
    "#e879f9",
)


def _root_background(theme_tokens: Mapping[str, str], dark_enabled: bool) -> str:
    if dark_enabled:
        return (
            "qradialgradient("
            "cx: 0.16, cy: 0.0, radius: 1.48, fx: 0.16, fy: 0.0,"
            f"stop: 0 {theme_tokens.get('root0', '#26194c')},"
            f"stop: 0.34 {theme_tokens.get('root1', '#170f32')},"
            f"stop: 0.74 {theme_tokens.get('root2', '#0d0b1b')},"
            f"stop: 1 {theme_tokens.get('root3', theme_tokens.get('root2', '#080710'))}"
            ")"
        )
    return (
        "qlineargradient("
        "x1: 0, y1: 0, x2: 1, y2: 1,"
        f"stop: 0 {theme_tokens.get('root0', '#f5f2ff')},"
        f"stop: 0.52 {theme_tokens.get('root1', '#fbf8ff')},"
        f"stop: 1 {theme_tokens.get('root2', '#f5f8ff')}"
        ")"
    )


def _semantic_tokens(theme_key: str, theme_tokens: Mapping[str, str], dark_enabled: bool) -> dict[str, str]:
    accent_primary = str(theme_tokens.get("btn1", theme_tokens.get("input_focus", "#55d7ff")))
    accent_secondary = str(theme_tokens.get("btn2", theme_tokens.get("header0", "#ff7ecf")))
    accent_tertiary = str(theme_tokens.get("header0", theme_tokens.get("btn0", "#8b5cf6")))
    accent_secondary_soft = _mix_hex(accent_secondary, surface := str(theme_tokens.get("root2" if dark_enabled else "root1", "#161d2a" if dark_enabled else "#f2f6fb")), 0.36 if dark_enabled else 0.24)
    state_info = str(theme_tokens.get("input_focus", accent_primary))
    state_success = "#67ff9a" if dark_enabled else "#1ea85f"
    state_warning = "#ffe45e" if dark_enabled else "#c89a00"
    state_error = "#ff5f87" if dark_enabled else "#d84674"
    line_palette = _THEME_LINE_PALETTES.get(theme_key, _THEME_LINE_PALETTES["graphite"])
    surface_base = surface
    surface_alt = str(theme_tokens.get("root3" if dark_enabled else "root0", "#0b1017" if dark_enabled else "#fbfdff"))
    table_positive = _rgba(state_success, 0.56 if dark_enabled else 0.30)
    table_warning = _rgba(state_warning, 0.50 if dark_enabled else 0.28)
    table_negative = _rgba(state_error, 0.56 if dark_enabled else 0.30)
    table_hover_bg = _mix_hex(accent_primary, surface_alt, 0.44 if dark_enabled else 0.26)
    table_action_bg = _mix_hex(accent_primary, surface_base, 0.15 if dark_enabled else 0.08)
    table_action_hover_bg = _mix_hex(accent_secondary, surface_base, 0.18 if dark_enabled else 0.10)
    table_action_done_bg = _mix_hex(line_palette[2], surface_base, 0.23 if dark_enabled else 0.13)
    table_disabled_bg = _mix_hex(str(theme_tokens.get("section_hint", "#8390a1")), surface_base, 0.12 if dark_enabled else 0.08)
    plot_sun = "#ffe45e" if theme_key == "laguna" else "#ffb224"
    plot_moon = "#d7e2ff"
    polar_sun = "#ffe45e" if theme_key == "laguna" else "#ffb224"
    return {
        "base_text": str(theme_tokens.get("base_text", theme_tokens.get("text", "#d7e4f0"))),
        "accent_primary": accent_primary,
        "accent_secondary": accent_secondary,
        "accent_secondary_soft": accent_secondary_soft,
        "accent_tertiary": accent_tertiary,
        "state_info": state_info,
        "state_success": state_success,
        "state_warning": state_warning,
        "state_error": state_error,
        "state_disabled": str(theme_tokens.get("btn_disabled_text", theme_tokens.get("section_hint", "#8ea3b9"))),
        "state_error_bg": _rgba(state_error, 0.12 if dark_enabled else 0.08),
        "chip_bg": _rgba(accent_primary, 0.14 if dark_enabled else 0.08),
        "chip_border": _rgba(accent_primary, 0.34 if dark_enabled else 0.24),
        "chip_text": str(theme_tokens.get("section_title", theme_tokens.get("text", "#f4f8fc"))),
        "chip_weather_bg": _rgba(accent_primary, 0.14 if dark_enabled else 0.08),
        "chip_weather_border": _rgba(accent_primary, 0.34 if dark_enabled else 0.24),
        "chip_context_bg": _rgba(accent_tertiary, 0.15 if dark_enabled else 0.10),
        "chip_context_border": _rgba(accent_tertiary, 0.36 if dark_enabled else 0.24),
        "chip_clock_bg": _rgba(accent_secondary, 0.15 if dark_enabled else 0.10),
        "chip_clock_border": _rgba(accent_secondary, 0.36 if dark_enabled else 0.24),
        "chip_solar_bg": _rgba(plot_sun, 0.18 if dark_enabled else 0.11),
        "chip_solar_border": _rgba(plot_sun, 0.42 if dark_enabled else 0.28),
        "chip_lunar_bg": _rgba(plot_moon, 0.14 if dark_enabled else 0.10),
        "chip_lunar_border": _rgba(plot_moon, 0.34 if dark_enabled else 0.24),
        "chip_series_temp_bg": _rgba(line_palette[0], 0.15 if dark_enabled else 0.10),
        "chip_series_temp_border": _rgba(line_palette[0], 0.46 if dark_enabled else 0.28),
        "chip_series_temp_text": line_palette[0],
        "chip_series_wind_bg": _rgba(line_palette[1], 0.15 if dark_enabled else 0.10),
        "chip_series_wind_border": _rgba(line_palette[1], 0.46 if dark_enabled else 0.28),
        "chip_series_wind_text": line_palette[1],
        "chip_series_cloud_bg": _rgba(line_palette[2], 0.15 if dark_enabled else 0.10),
        "chip_series_cloud_border": _rgba(line_palette[2], 0.46 if dark_enabled else 0.28),
        "chip_series_cloud_text": line_palette[2],
        "chip_series_humidity_bg": _rgba(line_palette[3], 0.15 if dark_enabled else 0.10),
        "chip_series_humidity_border": _rgba(line_palette[3], 0.46 if dark_enabled else 0.28),
        "chip_series_humidity_text": line_palette[3],
        "chip_series_pressure_bg": _rgba(line_palette[4], 0.15 if dark_enabled else 0.10),
        "chip_series_pressure_border": _rgba(line_palette[4], 0.46 if dark_enabled else 0.28),
        "chip_series_pressure_text": line_palette[4],
        "glow_edge": _rgba(accent_primary, 0.72 if dark_enabled else 0.38),
        "loading_bg": str(theme_tokens.get("plot_panel_bg", theme_tokens.get("tab_pane_bg", surface_alt))),
        "loading_border": _rgba(accent_primary, 0.24 if dark_enabled else 0.16),
        "loading_shimmer": _rgba(accent_secondary, 0.22 if dark_enabled else 0.15),
        "loading_text": str(theme_tokens.get("section_hint", theme_tokens.get("base_text", "#d7e4f0"))),
        "button_hover_glow": _rgba(accent_secondary, 0.98 if dark_enabled else 0.62),
        "button_hover_aura": _rgba(accent_secondary_soft, 0.92 if dark_enabled else 0.58),
        "progress_bg": _mix_hex(surface_alt, surface_base, 0.58 if dark_enabled else 0.68),
        "progress_border": _rgba(accent_primary, 0.34 if dark_enabled else 0.24),
        "progress_text": str(theme_tokens.get("section_title", theme_tokens.get("text", "#f4f8fc"))),
        "progress_chunk0": accent_primary,
        "progress_chunk1": accent_secondary,
        "root_background": _root_background(theme_tokens, dark_enabled),
        "root_overlay_image": _root_overlay_icon(accent_primary, accent_secondary),
        "plot_bg": str(theme_tokens.get("root2", "#121b29") if dark_enabled else theme_tokens.get("root0", "#f5f8ff")),
        "plot_panel_bg": str(theme_tokens.get("tab_pane_bg", "#121b29")),
        "plot_grid": accent_primary,
        "plot_text": str(theme_tokens.get("section_title", theme_tokens.get("base_text", "#d7e4f0"))),
        "plot_guide": str(theme_tokens.get("section_hint", accent_primary)),
        "plot_limit": state_error,
        "plot_now": "#ff3df0" if dark_enabled else "#b30086",
        "plot_sun": plot_sun,
        "plot_moon": plot_moon,
        "plot_twilight_civil": "#ffd166",
        "plot_twilight_naut": "#5ab6ff",
        "plot_twilight_astro": "#8094c8",
        "plot_series_temp": line_palette[0],
        "plot_series_wind": line_palette[1],
        "plot_series_cloud": line_palette[2],
        "plot_series_humidity": line_palette[3],
        "plot_series_pressure": line_palette[4],
        "polar_target": line_palette[0],
        "polar_selected": accent_secondary,
        "polar_selected_path": line_palette[2],
        "polar_sun": polar_sun,
        "polar_moon": "#dbe7ff",
        "polar_pole": accent_tertiary,
        "polar_limit": state_error,
        "overlay_crosshair": accent_secondary,
        "overlay_fov": accent_primary,
        "overlay_strip_bg": _rgba("#08111d", 0.78 if dark_enabled else 0.66),
        "overlay_strip_text": str(theme_tokens.get("section_title", theme_tokens.get("base_text", "#f4f8fc"))),
        "table_surface": surface_base,
        "table_surface_alt": surface_alt,
        "table_status_green": table_positive,
        "table_status_yellow": table_warning,
        "table_status_red": table_negative,
        "table_hover_bg": table_hover_bg,
        "table_hover_text": "#ffffff" if dark_enabled else "#161219",
        "table_warning_bg": table_warning,
        "table_warning_text": str(theme_tokens.get("section_title", "#392c00" if not dark_enabled else "#f4f8fc")),
        "table_action_bg": table_action_bg,
        "table_action_text": str(theme_tokens.get("strip_label", "#c7d7e9")),
        "table_action_hover_bg": table_action_hover_bg,
        "table_action_hover_text": str(theme_tokens.get("section_title", "#f4f8fc")),
        "table_action_done_bg": table_action_done_bg,
        "table_action_done_text": str(theme_tokens.get("section_title", "#9af1c3" if dark_enabled else "#24533c")),
        "table_row_positive": table_positive,
        "table_row_warning": table_warning,
        "table_row_negative": table_negative,
        "table_disabled_bg": table_disabled_bg,
        "table_disabled_text": str(theme_tokens.get("section_hint", "#8390a1" if dark_enabled else "#6d7684")),
    }


def resolve_theme_tokens(
    theme_key: str | None,
    dark_enabled: bool = False,
    *,
    ui_font_size: int | float = 11,
    font_family: str | None = None,
    display_font_family: str | None = None,
    overrides: Mapping[str, str] | None = None,
) -> dict[str, str]:
    normalized_key = normalize_theme_key(theme_key)
    theme = THEMES[normalized_key]
    base_tokens = dict(theme.dark if dark_enabled else theme.light)
    theme_defaults = _DEFAULT_THEME_TOKEN_OVERRIDES.get(normalized_key, {})
    if theme_defaults:
        for key, value in theme_defaults.items():
            if value:
                base_tokens[str(key)] = str(value)
    if overrides:
        for key, value in overrides.items():
            if value:
                base_tokens[str(key)] = str(value)
    tokens = {
        **base_tokens,
        **_semantic_tokens(normalized_key, base_tokens, dark_enabled),
        **_font_tokens(ui_font_size),
    }
    tokens.setdefault("table_sel_item", str(tokens.get("table_sel_bg", "#000000")))
    tokens.setdefault("menu_text_selected", str(tokens.get("base_text", "#f4f8fc")))
    tokens.setdefault("progress_chunk0", str(tokens.get("accent_primary", tokens.get("btn1", "#5fd7ff"))))
    tokens.setdefault("progress_chunk1", str(tokens.get("accent_secondary", tokens.get("btn2", "#ff7ecf"))))
    tokens.setdefault("loading_bg", str(tokens.get("plot_panel_bg", tokens.get("panel_bg", "#162334"))))
    tokens.setdefault("loading_border", str(tokens.get("panel_border", "#20506a")))
    tokens.setdefault("loading_shimmer", _rgba(str(tokens.get("accent_secondary", "#ff7ecf")), 0.22))
    tokens.setdefault("loading_text", str(tokens.get("section_hint", tokens.get("base_text", "#d7e4f0"))))
    tokens.setdefault("button_hover_glow", _rgba(str(tokens.get("accent_secondary", "#ff7ecf")), 0.62))
    tokens.setdefault("button_hover_aura", _rgba(str(tokens.get("accent_secondary_soft", tokens.get("accent_secondary", "#ff7ecf"))), 0.52))
    tokens.setdefault("table_surface", str(tokens.get("table_bg", "#0e1622")))
    tokens.setdefault("table_surface_alt", str(tokens.get("table_alt", "#101a28")))
    tokens.setdefault("table_status_green", str(tokens.get("table_row_positive", "#9af1c3")))
    tokens.setdefault("table_status_yellow", str(tokens.get("table_row_warning", "#ffe45e")))
    tokens.setdefault("table_status_red", str(tokens.get("table_row_negative", "#ff5f87")))
    tokens.setdefault("corner_bg", str(tokens.get("header1", tokens.get("header0", "#344c64"))))
    if font_family:
        tokens["font_family"] = _quote_qss_font_family(font_family)
    if display_font_family:
        tokens["display_font_family"] = _quote_qss_font_family(display_font_family)
    tokens.update(_icon_tokens(tokens))
    return tokens


def line_palette_for_theme(theme_key: str | None, dark_enabled: bool = False, color_blind: bool = False) -> list[str]:
    _ = dark_enabled
    normalized_key = normalize_theme_key(theme_key)
    if color_blind:
        return list(_CYBERPUNK_COLORBLIND_PALETTE)
    return list(_THEME_LINE_PALETTES.get(normalized_key, _THEME_LINE_PALETTES[DEFAULT_UI_THEME]))



def _icon_tokens(theme_tokens: Mapping[str, str]) -> dict[str, str]:
    arrow_color = theme_tokens.get("strip_label", theme_tokens.get("section_hint", "#6f8399"))
    disabled_arrow_color = theme_tokens.get("panel_border", arrow_color)
    combo_arrow = _svg_data_url(
        (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 6">'
            f'<path d="M1 1l4 4 4-4" fill="none" stroke="{arrow_color}" '
            'stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/>'
            "</svg>"
        )
    )
    combo_arrow_disabled = _svg_data_url(
        (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 6">'
            f'<path d="M1 1l4 4 4-4" fill="none" stroke="{disabled_arrow_color}" '
            'stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/>'
            "</svg>"
        )
    )
    spin_up = _svg_data_url(
        (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 6">'
            f'<path d="M1 5l4-4 4 4" fill="none" stroke="{arrow_color}" '
            'stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/>'
            "</svg>"
        )
    )
    spin_down = _svg_data_url(
        (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 6">'
            f'<path d="M1 1l4 4 4-4" fill="none" stroke="{arrow_color}" '
            'stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/>'
            "</svg>"
        )
    )
    spin_up_disabled = _svg_data_url(
        (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 6">'
            f'<path d="M1 5l4-4 4 4" fill="none" stroke="{disabled_arrow_color}" '
            'stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/>'
            "</svg>"
        )
    )
    spin_down_disabled = _svg_data_url(
        (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 6">'
            f'<path d="M1 1l4 4 4-4" fill="none" stroke="{disabled_arrow_color}" '
            'stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/>'
            "</svg>"
        )
    )
    return {
        "combo_arrow_icon": combo_arrow,
        "combo_arrow_disabled_icon": combo_arrow_disabled,
        "spin_up_icon": spin_up,
        "spin_down_icon": spin_down,
        "spin_up_disabled_icon": spin_up_disabled,
        "spin_down_disabled_icon": spin_down_disabled,
    }


def build_stylesheet(
    theme_key: str | None,
    dark_enabled: bool = False,
    ui_font_size: int | float = 11,
    font_family: str | None = None,
    display_font_family: str | None = None,
    overrides: Mapping[str, str] | None = None,
) -> str:
    tokens = resolve_theme_tokens(
        theme_key,
        dark_enabled=dark_enabled,
        ui_font_size=ui_font_size,
        font_family=font_family,
        display_font_family=display_font_family,
        overrides=overrides,
    )
    return _LIGHT_TEMPLATE % tokens


# Backward-compatible exports used by older call sites.
LIGHT_STYLESHEET = _LIGHT_TEMPLATE % resolve_theme_tokens(DEFAULT_UI_THEME, dark_enabled=False)
DARK_OVERRIDES = _LIGHT_TEMPLATE % resolve_theme_tokens(DEFAULT_UI_THEME, dark_enabled=True)


@dataclass(frozen=True)
class HighlightPalette:
    below: str
    limit: str
    above: str


DEFAULT_HIGHLIGHT = HighlightPalette(
    below="#ffe0ea",
    limit="#fff2cf",
    above="#d9ffe9",
)

COLORBLIND_HIGHLIGHT = HighlightPalette(
    below="#ffb18e",
    limit="#ffe680",
    above="#9bf985",
)

DEFAULT_LINE_COLORS = list(_THEME_LINE_PALETTES[DEFAULT_UI_THEME])
COLORBLIND_LINE_COLORS = list(_CYBERPUNK_COLORBLIND_PALETTE)


def highlight_palette_for_theme(
    theme_key: str | None,
    dark_enabled: bool = False,
    color_blind: bool = False,
) -> HighlightPalette:
    if color_blind:
        return COLORBLIND_HIGHLIGHT
    tokens = resolve_theme_tokens(theme_key, dark_enabled=dark_enabled)
    return HighlightPalette(
        below=str(tokens.get("table_row_negative", DEFAULT_HIGHLIGHT.below)),
        limit=str(tokens.get("table_row_warning", DEFAULT_HIGHLIGHT.limit)),
        above=str(tokens.get("table_row_positive", DEFAULT_HIGHLIGHT.above)),
    )
