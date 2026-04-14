from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QFont, QFontMetrics
from PySide6.QtWidgets import QHeaderView, QTableView

from astroplanner.theme import DEFAULT_UI_THEME, highlight_palette_for_theme
from astroplanner.ui.common import _distribute_extra_table_width
from astroplanner.ui.targets import (
    _normalize_table_color_mode,
    TargetTableGlowDelegate,
    TargetTableModel,
)
from astroplanner.ui.theme_utils import _qcolor_from_token

if TYPE_CHECKING:
    from astro_planner import MainWindow


def _resolve_table_highlight_color(settings: object, key: str, default_color: str) -> str:
    getter = getattr(settings, "value", None)
    if not callable(getter):
        return str(default_color)
    direct = str(getter(f"table/highlight/{key}", "", type=str) or "").strip()
    if direct:
        return direct
    legacy = str(getter(f"table/{key}Color", "", type=str) or "").strip()
    if legacy:
        return legacy
    return str(default_color)


class TargetTableCoordinator:
    def __init__(self, planner: "MainWindow") -> None:
        self._planner = planner
        self._width_reset_requested = False
        self._width_timer = QTimer(planner)
        self._width_timer.setSingleShot(True)
        self._width_timer.setInterval(0)
        self._width_timer.timeout.connect(self.refresh_table_column_widths)

    def bind(self) -> None:
        planner = self._planner
        planner.table_view.setObjectName("MainTargetsTable")
        planner.table_view.setSelectionBehavior(QTableView.SelectRows)
        planner.table_view.deleteRequested.connect(planner._delete_selected_targets)
        planner.table_view.hoverRowChanged.connect(planner.table_model.set_hover_row)
        planner.table_view.setItemDelegate(TargetTableGlowDelegate(planner.table_view))

        header = planner.table_view.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        header.setStretchLastSection(True)
        planner.table_view.verticalHeader().setVisible(False)
        planner.table_view.setShowGrid(False)
        planner.table_view.setModel(planner.table_model)

        planner.table_model.layoutChanged.connect(self.apply_table_row_visibility)
        planner.table_model.modelReset.connect(self.apply_table_row_visibility)
        planner.table_model.layoutChanged.connect(lambda: self.schedule_table_column_width_refresh(reset_widths=True))
        planner.table_model.modelReset.connect(lambda: self.schedule_table_column_width_refresh(reset_widths=True))
        planner.table_model.dataChanged.connect(lambda *_: self.schedule_table_column_width_refresh(reset_widths=True))
        planner.table_model.rowsInserted.connect(lambda *_: self.schedule_table_column_width_refresh(reset_widths=True))
        planner.table_model.rowsRemoved.connect(lambda *_: self.schedule_table_column_width_refresh(reset_widths=True))
        planner.table_model.modelReset.connect(self.schedule_primary_target_selection)
        planner.table_model.rowsInserted.connect(lambda *_: self.schedule_primary_target_selection())
        planner.table_model.layoutChanged.connect(planner._schedule_plan_autosave)
        planner.table_model.modelReset.connect(planner._schedule_plan_autosave)
        planner.table_model.rowsInserted.connect(lambda *_: planner._schedule_plan_autosave())
        planner.table_model.rowsRemoved.connect(lambda *_: planner._schedule_plan_autosave())
        planner.table_view.layoutWidthChanged.connect(lambda: self.schedule_table_column_width_refresh(reset_widths=True))

        planner.table_view.setSortingEnabled(True)
        planner.table_view.horizontalHeader().setSectionsClickable(True)
        planner.table_view.horizontalHeader().sortIndicatorChanged.connect(lambda *_: planner._schedule_plan_autosave())
        planner.table_view.setDragDropMode(QTableView.InternalMove)
        planner.table_view.setContextMenuPolicy(Qt.CustomContextMenu)
        planner.table_view.customContextMenuRequested.connect(planner._open_table_context_menu)
        planner.table_view.doubleClicked.connect(planner._on_table_double_click)

    def recompute_recommended_order_cache(self) -> None:
        planner = self._planner
        n_targets = len(planner.targets)
        previous = list(planner.table_model.order_values)
        order_values = [0] * n_targets
        ordered, _ = planner._build_deterministic_observation_order()
        for rank, item in enumerate(ordered, start=1):
            row_index = int(item.get("row_index", -1))
            if 0 <= row_index < len(order_values):
                order_values[row_index] = rank
        next_rank = max(order_values, default=0) + 1
        if next_rank <= n_targets:
            remaining_rows = [idx for idx, value in enumerate(order_values) if value <= 0]
            remaining_rows.sort(
                key=lambda idx: (
                    0 if idx < len(previous) and previous[idx] > 0 else 1,
                    int(previous[idx]) if idx < len(previous) and previous[idx] > 0 else 10**9,
                    idx,
                )
            )
            for idx in remaining_rows:
                order_values[idx] = next_rank
                next_rank += 1
        planner.table_model.order_values = order_values

    def apply_table_settings(self) -> None:
        planner = self._planner
        row_h = planner.settings.value("table/rowHeight", 24, type=int)
        planner.table_view.set_forced_row_height(row_h)
        header = planner.table_view.horizontalHeader()
        header.setStretchLastSection(False)
        header.setMinimumSectionSize(36)
        for col in range(planner.table_model.columnCount()):
            header.setSectionResizeMode(col, QHeaderView.Interactive)

        font_size = planner.settings.value("table/fontSize", 11, type=int)
        font = planner.table_view.font()
        font.setPointSize(font_size)
        planner.table_view.setUpdatesEnabled(False)
        planner.table_view.setFont(font)
        planner.table_view.viewport().setFont(font)
        header_font = QFont(font)
        header_font.setWeight(QFont.Weight.Bold)
        planner.table_view.horizontalHeader().setFont(header_font)
        planner.table_view.verticalHeader().setFont(font)

        for col in range(planner.table_model.columnCount()):
            show = planner.settings.value(f"table/col{col}", True, type=bool)
            planner.table_view.setColumnHidden(col, not show)

        palette = highlight_palette_for_theme(
            getattr(planner, "_theme_name", DEFAULT_UI_THEME),
            dark_enabled=bool(getattr(planner, "_dark_enabled", False)),
            color_blind=bool(planner.color_blind_mode),
        )
        default_colors = {"below": palette.below, "limit": palette.limit, "above": palette.above}
        planner.table_model.highlight_colors = {
            key: _qcolor_from_token(
                _resolve_table_highlight_color(planner.settings, key, default_colors[key]),
                default_colors[key],
            )
            for key in default_colors
        }
        table_color_mode = _normalize_table_color_mode(
            planner.settings.value("table/colorMode", "background", type=str),
            default="background",
        )
        planner.table_model.color_mode = table_color_mode
        planner.table_view.setProperty("table_color_mode", table_color_mode)
        self.apply_column_preset(planner.settings.value("table/viewPreset", "full", type=str), save=False)
        planner.table_view.doItemsLayout()
        planner.table_view.viewport().update()
        planner.table_view.horizontalHeader().viewport().update()
        planner.table_view.setUpdatesEnabled(True)
        planner.table_view.viewport().update()
        self.schedule_table_column_width_refresh(reset_widths=True)

    @staticmethod
    def _observation_visible_columns() -> set[int]:
        return {
            TargetTableModel.COL_ORDER,
            TargetTableModel.COL_NAME,
            TargetTableModel.COL_ALT,
            TargetTableModel.COL_AZ,
            TargetTableModel.COL_MOON_SEP,
            TargetTableModel.COL_SCORE,
            TargetTableModel.COL_HOURS,
            TargetTableModel.COL_MAG,
            TargetTableModel.COL_PRIORITY,
            TargetTableModel.COL_OBSERVED,
        }

    def table_matches_observation_preset(self) -> bool:
        planner = self._planner
        obs_visible = self._observation_visible_columns()
        for col in range(planner.table_model.columnCount()):
            hidden = bool(planner.table_view.isColumnHidden(col))
            if col == TargetTableModel.COL_ACTIONS:
                if not hidden:
                    return False
                continue
            should_be_hidden = col not in obs_visible
            if hidden != should_be_hidden:
                return False
        return True

    def apply_column_preset(self, preset: str, save: bool = True) -> None:
        planner = self._planner
        if preset not in {"observation", "full"}:
            preset = "full"
        obs_visible = self._observation_visible_columns()
        for col in range(planner.table_model.columnCount()):
            if col == TargetTableModel.COL_ACTIONS:
                planner.table_view.setColumnHidden(col, True)
                continue
            if preset == "observation":
                planner.table_view.setColumnHidden(col, col not in obs_visible)
            else:
                show = planner.settings.value(f"table/col{col}", True, type=bool)
                planner.table_view.setColumnHidden(col, not show)
        self.schedule_table_column_width_refresh(reset_widths=True)
        if save:
            planner.settings.setValue("table/viewPreset", preset)
        if hasattr(planner, "view_obs_preset_act"):
            planner.view_obs_preset_act.setChecked(preset == "observation")
        if hasattr(planner, "view_full_preset_act"):
            planner.view_full_preset_act.setChecked(preset == "full")
        planner._schedule_plan_autosave()

    def schedule_table_column_width_refresh(self, reset_widths: bool = False) -> None:
        self._width_reset_requested = self._width_reset_requested or reset_widths
        self._width_timer.start()

    @staticmethod
    def _main_table_stretch_weights() -> dict[int, float]:
        return {
            TargetTableModel.COL_NAME: 4.0,
            TargetTableModel.COL_RA: 1.0,
            TargetTableModel.COL_HA: 1.0,
            TargetTableModel.COL_DEC: 1.0,
            TargetTableModel.COL_ALT: 1.0,
            TargetTableModel.COL_AZ: 1.0,
            TargetTableModel.COL_MOON_SEP: 1.2,
            TargetTableModel.COL_SCORE: 1.0,
            TargetTableModel.COL_HOURS: 1.2,
            TargetTableModel.COL_MAG: 1.1,
        }

    def refresh_table_column_widths(self) -> None:
        planner = self._planner
        reset_widths = self._width_reset_requested
        self._width_reset_requested = False

        header = planner.table_view.horizontalHeader()
        header_font_metrics = QFontMetrics(header.font())
        table_font_metrics = QFontMetrics(planner.table_view.font())
        name_font = QFont(planner.table_view.font())
        name_font.setWeight(QFont.Weight.DemiBold)
        name_font_metrics = QFontMetrics(name_font)
        baseline_name_width = planner.settings.value("table/firstColumnWidth", 100, type=int)
        col_padding = 20
        sort_indicator_padding = 18
        baseline_widths = {
            TargetTableModel.COL_NAME: baseline_name_width,
        }
        row_count = planner.table_model.rowCount()
        required_widths: dict[int, int] = {}

        for col in range(planner.table_model.columnCount()):
            if planner.table_view.isColumnHidden(col):
                continue
            if col == TargetTableModel.COL_ACTIONS:
                continue

            header_text = planner.table_model.headerData(col, Qt.Horizontal, Qt.DisplayRole) or ""
            header_width = (
                header_font_metrics.horizontalAdvance(str(header_text)) + col_padding + sort_indicator_padding
            )
            content_width = 0
            for row in range(row_count):
                idx = planner.table_model.index(row, col)
                if not idx.isValid():
                    continue
                cell = planner.table_model.data(idx, Qt.DisplayRole)
                if cell is None:
                    continue
                text = str(cell)
                if not text:
                    continue
                metrics = name_font_metrics if col == TargetTableModel.COL_NAME else table_font_metrics
                content_width = max(content_width, metrics.horizontalAdvance(text))
            required_width = max(
                baseline_widths.get(col, 0),
                header_width,
                content_width + col_padding,
            )
            if reset_widths:
                new_width = required_width
            else:
                new_width = max(planner.table_view.columnWidth(col), required_width)
            required_widths[col] = int(new_width)

        if not required_widths:
            return
        viewport_width = max(0, int(planner.table_view.viewport().width()) - 2)
        fitted_widths = _distribute_extra_table_width(
            required_widths,
            available_width=viewport_width,
            stretch_weights=self._main_table_stretch_weights(),
        )
        for col, width in fitted_widths.items():
            planner.table_view.setColumnWidth(col, int(width))

    def emit_table_data_changed(self) -> None:
        planner = self._planner
        rows = planner.table_model.rowCount()
        cols = planner.table_model.columnCount()
        if rows <= 0 or cols <= 0:
            return
        top_left = planner.table_model.index(0, 0)
        bottom_right = planner.table_model.index(rows - 1, cols - 1)
        planner.table_model.dataChanged.emit(
            top_left,
            bottom_right,
            [Qt.DisplayRole, Qt.BackgroundRole, Qt.ForegroundRole, Qt.ToolTipRole, Qt.CheckStateRole],
        )
        self.schedule_table_column_width_refresh(reset_widths=True)

    def apply_table_row_visibility(self) -> None:
        planner = self._planner
        rows = planner.table_model.rowCount()
        for row in range(rows):
            planner.table_view.setRowHidden(row, False)
        self.schedule_primary_target_selection()

    def clear_table_dynamic_cache(self) -> None:
        planner = self._planner
        planner.table_model.order_values = []
        planner.table_model.current_alts = []
        planner.table_model.current_azs = []
        planner.table_model.current_seps = []
        planner.table_model.scores = []
        planner.table_model.hours_above_limit = []
        planner.table_model.row_enabled = []
        self.apply_table_row_visibility()

    def schedule_primary_target_selection(self) -> None:
        QTimer.singleShot(0, self._planner._ensure_primary_target_selected)


__all__ = ["TargetTableCoordinator"]
