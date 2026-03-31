from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


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
        background: qlineargradient(
            x1: 0, y1: 0, x2: 1, y2: 1,
            stop: 0 %(root0)s,
            stop: 0.52 %(root1)s,
            stop: 1 %(root2)s
        );
    }
    QDialog {
        background: %(panel_bg)s;
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
        border-radius: 14px;
    }
    QFrame#TopControlsBar,
    QFrame#ActionsBar {
        background: %(top_bg)s;
        border: 1px solid %(top_border)s;
    }
    QGroupBox {
        font-weight: 600;
        color: %(group_text)s;
        border: 1px solid %(group_border)s;
        border-radius: 10px;
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
        background: %(group_bg)s;
        border: 1px solid %(group_border)s;
        border-radius: 10px;
        top: -1px;
    }
    QTabBar::tab {
        background: %(top_bg)s;
        color: %(section_hint)s;
        border: 1px solid %(panel_border)s;
        border-bottom: none;
        border-top-left-radius: 8px;
        border-top-right-radius: 8px;
        padding: 6px 14px;
        min-width: 48px;
    }
    QTabBar::tab:selected {
        background: %(group_bg)s;
        color: %(base_text)s;
        border: 1px solid %(group_border)s;
        border-bottom: 1px solid %(group_bg)s;
    }
    QTabBar::tab:!selected:hover {
        background: %(panel_bg)s;
        color: %(base_text)s;
    }
    QLabel#SectionTitle {
        font-family: %(display_font_family)s;
        color: %(section_title)s;
        font-size: %(section_font_size_pt)spt;
        font-weight: 700;
        letter-spacing: 0.3px;
        padding: 0 0 4px 2px;
    }
    QLabel#SectionHint {
        color: %(section_hint)s;
        font-size: %(strip_font_size_pt)spt;
        font-weight: 500;
        padding: 0 2px 4px 2px;
    }
    QLabel#CutoutImage {
        background: %(input_bg)s;
        border: 1px solid %(input_border)s;
        border-radius: 10px;
        color: %(section_hint)s;
        font-size: %(strip_font_size_pt)spt;
        font-weight: 500;
        padding: 4px;
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
        border: 1px solid %(btn_border)s;
        border-radius: 8px;
        padding: 6px 12px;
        min-height: 26px;
        font-weight: 600;
    }
    QPushButton:hover {
        background: qlineargradient(
            x1: 0, y1: 0, x2: 1, y2: 0,
            stop: 0 %(btnh0)s,
            stop: 0.55 %(btnh1)s,
            stop: 1 %(btnh2)s
        );
    }
    QPushButton:pressed {
        background: %(btn_pressed)s;
    }
    QPushButton:disabled {
        background: %(btn_disabled_bg)s;
        color: %(btn_disabled_text)s;
        border: 1px solid %(btn_disabled_border)s;
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
        background: %(spin_btn_pressed)s;
        border: 1px solid %(input_focus)s;
    }
    QComboBox, QSpinBox, QDateEdit, QLineEdit, QTextEdit {
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
    QComboBox::drop-down {
        subcontrol-origin: border;
        subcontrol-position: top right;
        width: 20px;
        border-left: 1px solid %(combo_drop_border)s;
        border-top-right-radius: 6px;
        border-bottom-right-radius: 6px;
        background: %(combo_drop_bg)s;
    }
    QComboBox::drop-down:hover {
        background: %(combo_drop_hover)s;
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
    QSpinBox, QDateEdit {
        padding-right: 22px;
    }
    QSpinBox::up-button, QDateEdit::up-button {
        subcontrol-origin: border;
        subcontrol-position: top right;
        width: 18px;
        border-left: 1px solid %(spin_btn_border)s;
        border-bottom: 1px solid %(spin_btn_border)s;
        border-top-right-radius: 6px;
        background: %(spin_btn_bg)s;
    }
    QSpinBox::down-button, QDateEdit::down-button {
        subcontrol-origin: border;
        subcontrol-position: bottom right;
        width: 18px;
        border-left: 1px solid %(spin_btn_border)s;
        border-top: 1px solid %(spin_btn_border)s;
        border-bottom-right-radius: 6px;
        background: %(spin_btn_bg)s;
    }
    QSpinBox::up-button:hover, QDateEdit::up-button:hover,
    QSpinBox::down-button:hover, QDateEdit::down-button:hover {
        background: %(spin_btn_hover)s;
    }
    QSpinBox::up-button:pressed, QDateEdit::up-button:pressed,
    QSpinBox::down-button:pressed, QDateEdit::down-button:pressed {
        background: %(spin_btn_pressed)s;
    }
    QComboBox:focus, QSpinBox:focus, QDateEdit:focus, QLineEdit:focus, QTextEdit:focus {
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
        background: %(combo_drop_bg)s;
        border: 1px solid %(input_border)s;
        border-radius: 6px;
        padding: 2px 6px;
    }
    QCalendarWidget QToolButton:hover {
        background: %(combo_drop_hover)s;
        border: 1px solid %(input_focus)s;
    }
    QCalendarWidget QSpinBox {
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
        border-radius: 10px;
        gridline-color: %(table_grid)s;
        selection-color: %(table_sel_text)s;
        selection-background-color: %(table_sel_bg)s;
        padding: 2px;
    }
    QTableView::item:selected {
        background: %(table_sel_item)s;
    }
    QHeaderView::section {
        font-family: %(display_font_family)s;
        background: qlineargradient(
            x1: 0, y1: 0, x2: 0, y2: 1,
            stop: 0 %(header0)s,
            stop: 1 %(header1)s
        );
        color: %(header_text)s;
        font-weight: 600;
        padding: 6px;
        border: 0;
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
        background: %(group_bg)s;
        border: 1px solid %(group_border)s;
        border-radius: 10px;
        top: -1px;
    }
    QTabBar::tab {
        background: %(top_bg)s;
        color: %(section_hint)s;
        border: 1px solid %(panel_border)s;
        border-bottom: none;
        border-top-left-radius: 8px;
        border-top-right-radius: 8px;
        padding: 6px 14px;
        min-width: 48px;
    }
    QTabBar::tab:selected {
        background: %(group_bg)s;
        color: %(text)s;
        border: 1px solid %(group_border)s;
        border-bottom: 1px solid %(group_bg)s;
    }
    QTabBar::tab:!selected:hover {
        background: %(panel_bg)s;
        color: %(text)s;
    }
    QLabel#SectionTitle {
        color: %(section_title)s;
        font-weight: 700;
        letter-spacing: 0.35px;
    }
    QLabel#SectionHint {
        color: %(section_hint)s;
        font-weight: 500;
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
    QLineEdit, QComboBox, QDateEdit, QSpinBox, QTextEdit {
        background-color: %(input_bg)s;
        border: 1px solid %(input_border)s;
        color: %(input_text)s;
        border-radius: 7px;
        min-height: 20px;
    }
    QComboBox {
        padding-right: 24px;
    }
    QComboBox::drop-down {
        subcontrol-origin: border;
        subcontrol-position: top right;
        width: 20px;
        border-left: 1px solid %(combo_drop_border)s;
        border-top-right-radius: 6px;
        border-bottom-right-radius: 6px;
        background: %(combo_drop_bg)s;
    }
    QComboBox::drop-down:hover {
        background: %(combo_drop_hover)s;
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
    QSpinBox, QDateEdit {
        padding-right: 22px;
    }
    QSpinBox::up-button, QDateEdit::up-button {
        subcontrol-origin: border;
        subcontrol-position: top right;
        width: 18px;
        border-left: 1px solid %(spin_btn_border)s;
        border-bottom: 1px solid %(spin_btn_border)s;
        border-top-right-radius: 6px;
        background: %(spin_btn_bg)s;
    }
    QSpinBox::down-button, QDateEdit::down-button {
        subcontrol-origin: border;
        subcontrol-position: bottom right;
        width: 18px;
        border-left: 1px solid %(spin_btn_border)s;
        border-top: 1px solid %(spin_btn_border)s;
        border-bottom-right-radius: 6px;
        background: %(spin_btn_bg)s;
    }
    QSpinBox::up-button:hover, QDateEdit::up-button:hover,
    QSpinBox::down-button:hover, QDateEdit::down-button:hover {
        background: %(spin_btn_hover)s;
    }
    QSpinBox::up-button:pressed, QDateEdit::up-button:pressed,
    QSpinBox::down-button:pressed, QDateEdit::down-button:pressed {
        background: %(spin_btn_pressed)s;
    }
    QComboBox:focus, QSpinBox:focus, QDateEdit:focus, QLineEdit:focus, QTextEdit:focus {
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
        background: %(combo_drop_bg)s;
        border: 1px solid %(input_border)s;
        border-radius: 6px;
        padding: 2px 6px;
    }
    QCalendarWidget QToolButton:hover {
        background: %(combo_drop_hover)s;
        border: 1px solid %(input_focus)s;
    }
    QCalendarWidget QSpinBox {
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
        font-weight: 600;
    }
    QPushButton:hover {
        background: qlineargradient(
            x1: 0, y1: 0, x2: 1, y2: 0,
            stop: 0 %(btnh0)s,
            stop: 0.5 %(btnh1)s,
            stop: 1 %(btnh2)s
        );
    }
    QPushButton:pressed {
        background: %(btn_pressed)s;
    }
    QPushButton:disabled {
        background: %(btn_disabled_bg)s;
        color: %(btn_disabled_text)s;
        border: 1px solid %(btn_disabled_border)s;
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
        background: %(spin_btn_pressed)s;
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
    "splitter": "rgba(124, 145, 168, 0.32)",
    "scroll_bg": "rgba(236, 242, 249, 0.90)",
    "scroll_handle": "rgba(126, 153, 180, 0.62)",
    "scroll_handle_hover": "rgba(110, 136, 162, 0.78)",
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
    "ocean": UiTheme(
        key="ocean",
        label="Ocean Mist",
        light=BASE_LIGHT,
        dark=BASE_DARK,
    ),
    "graphite": UiTheme(
        key="graphite",
        label="Graphite",
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
    "sage": UiTheme(
        key="sage",
        label="Sage",
        light=_merge(
            BASE_LIGHT,
            base_text="#25362f",
            root0="#eef3ef",
            root1="#f6f9f6",
            root2="#edf5f0",
            panel_border="#cad8cf",
            top_border="#b9cdbf",
            group_text="#335245",
            group_border="#cddfd3",
            section_title="#315347",
            strip_label="#3e6253",
            btn0="#5f8675",
            btn1="#6e9684",
            btn2="#7ea896",
            btnh0="#567969",
            btnh1="#648978",
            btnh2="#749886",
            btn_pressed="#4b6a5c",
            tool_hover="rgba(86, 127, 107, 0.18)",
            input_border="#b9cfbf",
            input_focus="#6f9a85",
            table_alt="rgba(233, 243, 236, 0.74)",
            table_border="#c4d6cb",
            table_grid="#dbe9e1",
            table_sel_bg="rgba(115, 154, 135, 0.23)",
            table_sel_item="rgba(115, 154, 135, 0.29)",
            header0="#3f6557",
            header1="#315245",
            check_text="#3f6455",
            check_border="#9fc2b0",
            check_checked0="#6f9a85",
            check_checked1="#7aa994",
            check_checked_border="#638b79",
            splitter="rgba(116, 146, 130, 0.34)",
            scroll_handle="rgba(116, 146, 130, 0.62)",
            scroll_handle_hover="rgba(95, 122, 108, 0.78)",
        ),
        dark=_merge(
            BASE_DARK,
            root0="#2a3f36",
            root1="#1d2f28",
            root2="#14211d",
            root3="#0d1512",
            text="#d7e7de",
            panel_bg="rgba(18, 31, 26, 0.88)",
            panel_border="rgba(124, 170, 147, 0.34)",
            top_bg="rgba(15, 27, 22, 0.92)",
            top_border="rgba(124, 170, 147, 0.46)",
            group_text="#a9d0bc",
            group_border="rgba(124, 170, 147, 0.40)",
            group_bg="rgba(13, 24, 20, 0.95)",
            section_title="#e2f3ea",
            strip_label="#9dc7b2",
            header0="#396250",
            header1="#2f5142",
            header2="#274437",
            header_text="#edf7f2",
            corner_bg="#2f5142",
            input_bg="rgba(16, 29, 24, 0.96)",
            input_border="rgba(124, 170, 147, 0.44)",
            input_text="#d7e7de",
            input_focus="#86b59e",
            table_bg="rgba(14, 25, 21, 0.96)",
            table_alt="rgba(17, 31, 26, 0.94)",
            table_border="rgba(124, 170, 147, 0.44)",
            table_grid="rgba(124, 170, 147, 0.22)",
            table_sel_bg="rgba(126, 175, 151, 0.29)",
            btn0="#5d8675",
            btn1="#6e9684",
            btn2="#7ca696",
            btnh0="#527667",
            btnh1="#638679",
            btnh2="#739589",
            btn_pressed="#456359",
            check_text="#d7e7de",
            check_bg="rgba(15, 27, 22, 0.96)",
            check_border="rgba(124, 170, 147, 0.56)",
            check_checked0="#7aa995",
            check_checked1="#87b5a2",
            tool_hover="rgba(126, 175, 151, 0.22)",
            status_bg="rgba(12, 22, 18, 0.95)",
            status_border="rgba(124, 170, 147, 0.38)",
            status_text="#d7e7de",
            splitter="rgba(126, 175, 151, 0.24)",
            scroll_bg="rgba(11, 21, 17, 0.86)",
            scroll_handle="rgba(126, 175, 151, 0.56)",
            scroll_handle_hover="rgba(101, 142, 122, 0.78)",
        ),
    ),
    "sand": UiTheme(
        key="sand",
        label="Soft Sand",
        light=_merge(
            BASE_LIGHT,
            base_text="#3a2f27",
            root0="#f6f1eb",
            root1="#fbf8f4",
            root2="#f2ede6",
            panel_border="#dfd1c2",
            top_border="#d2c3b3",
            group_text="#654f3d",
            group_border="#e4d6c8",
            section_title="#6a503d",
            strip_label="#715a46",
            btn0="#907566",
            btn1="#a18474",
            btn2="#b19180",
            btnh0="#836a5c",
            btnh1="#947969",
            btnh2="#a68674",
            btn_pressed="#735d50",
            tool_hover="rgba(143, 113, 92, 0.18)",
            input_border="#d4c3b3",
            input_focus="#9f7e67",
            table_alt="rgba(247, 239, 231, 0.74)",
            table_border="#dfd1c2",
            table_grid="#ebdfd3",
            table_sel_bg="rgba(161, 132, 112, 0.23)",
            table_sel_item="rgba(161, 132, 112, 0.29)",
            header0="#7a5f4b",
            header1="#694f3d",
            check_text="#6e5642",
            check_border="#c9b29f",
            check_checked0="#a3846c",
            check_checked1="#b6937b",
            check_checked_border="#92745e",
            splitter="rgba(152, 128, 109, 0.34)",
            scroll_handle="rgba(162, 137, 117, 0.62)",
            scroll_handle_hover="rgba(137, 113, 96, 0.78)",
        ),
        dark=_merge(
            BASE_DARK,
            root0="#4a3a2f",
            root1="#342820",
            root2="#241b16",
            root3="#17110d",
            text="#f0e6dc",
            panel_bg="rgba(37, 28, 23, 0.90)",
            panel_border="rgba(195, 161, 131, 0.34)",
            top_bg="rgba(32, 24, 20, 0.94)",
            top_border="rgba(195, 161, 131, 0.46)",
            group_text="#e0cdbb",
            group_border="rgba(195, 161, 131, 0.40)",
            group_bg="rgba(30, 22, 18, 0.95)",
            section_title="#f4eadf",
            strip_label="#d8c1ad",
            header0="#7a614e",
            header1="#674f3f",
            header2="#574234",
            header_text="#f8efe5",
            corner_bg="#674f3f",
            input_bg="rgba(33, 25, 20, 0.96)",
            input_border="rgba(195, 161, 131, 0.44)",
            input_text="#f0e6dc",
            input_focus="#c09574",
            table_bg="rgba(30, 23, 18, 0.96)",
            table_alt="rgba(35, 27, 21, 0.94)",
            table_border="rgba(195, 161, 131, 0.44)",
            table_grid="rgba(195, 161, 131, 0.22)",
            table_sel_bg="rgba(178, 142, 115, 0.30)",
            btn0="#9e7d67",
            btn1="#b18b74",
            btn2="#be9a82",
            btnh0="#8f715d",
            btnh1="#a38069",
            btnh2="#b08e77",
            btn_pressed="#7d6351",
            check_text="#f0e6dc",
            check_bg="rgba(30, 22, 18, 0.96)",
            check_border="rgba(195, 161, 131, 0.56)",
            check_checked0="#b08a72",
            check_checked1="#c39a81",
            tool_hover="rgba(178, 142, 115, 0.22)",
            status_bg="rgba(26, 19, 15, 0.95)",
            status_border="rgba(195, 161, 131, 0.38)",
            status_text="#f0e6dc",
            splitter="rgba(178, 142, 115, 0.24)",
            scroll_bg="rgba(24, 18, 14, 0.86)",
            scroll_handle="rgba(178, 142, 115, 0.56)",
            scroll_handle_hover="rgba(151, 119, 96, 0.78)",
        ),
    ),
}

DEFAULT_UI_THEME = "ocean"
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


def build_stylesheet(
    theme_key: str | None,
    dark_enabled: bool = False,
    ui_font_size: int | float = 11,
    font_family: str | None = None,
    display_font_family: str | None = None,
) -> str:
    theme = THEMES[normalize_theme_key(theme_key)]
    tokens = _font_tokens(ui_font_size)
    if font_family:
        tokens["font_family"] = _quote_qss_font_family(font_family)
    if display_font_family:
        tokens["display_font_family"] = _quote_qss_font_family(display_font_family)
    light_qss = _LIGHT_TEMPLATE % {**theme.light, **tokens}
    if dark_enabled:
        return light_qss + (_DARK_TEMPLATE % {**theme.dark, **tokens})
    return light_qss


# Backward-compatible exports used by older call sites.
_DEFAULT_FONT_TOKENS = _font_tokens(11)
LIGHT_STYLESHEET = _LIGHT_TEMPLATE % {**THEMES[DEFAULT_UI_THEME].light, **_DEFAULT_FONT_TOKENS}
DARK_OVERRIDES = _DARK_TEMPLATE % {**THEMES[DEFAULT_UI_THEME].dark, **_DEFAULT_FONT_TOKENS}


@dataclass(frozen=True)
class HighlightPalette:
    below: str
    limit: str
    above: str


DEFAULT_HIGHLIGHT = HighlightPalette(
    below="#ff8080",
    limit="#ffff80",
    above="#b3ffb3",
)

COLORBLIND_HIGHLIGHT = HighlightPalette(
    below="#f28e2b",
    limit="#edc948",
    above="#59a14f",
)


DEFAULT_LINE_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

COLORBLIND_LINE_COLORS = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#56B4E9",
    "#F0E442",
    "#000000",
    "#999999",
    "#1B9E77",
]
