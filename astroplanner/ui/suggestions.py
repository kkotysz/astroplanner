from __future__ import annotations

import math
import os
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import quote

from PySide6.QtCore import QAbstractTableModel, QCoreApplication, QEvent, QModelIndex, QSignalBlocker, Qt, QStandardPaths, Slot, QUrl
from PySide6.QtGui import QBrush, QColor, QDesktopServices, QFont, QFontMetrics, QPainter
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFrame,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QStackedLayout,
    QStyledItemDelegate,
    QStyle,
    QStyleOptionViewItem,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from astroplanner.bhtom import BHTOM_SUGGESTION_MIN_IMPORTANCE, bhtom_suggestion_source_message
from astroplanner.i18n import current_language, localize_widget_tree, translate_text
from astroplanner.models import DEFAULT_LIMITING_MAGNITUDE, Target, targets_match
from astroplanner.scoring import TargetNightMetrics
from astroplanner.storage import AppStorage, SettingsAdapter
from astroplanner.ui.common import SkeletonShimmerWidget, _distribute_extra_table_width, _fit_dialog_to_screen
from astroplanner.ui.theme_utils import (
    _set_button_icon_kind,
    _set_button_variant,
    _set_label_tone,
    _style_dialog_button_box,
    _theme_color_from_widget,
    _theme_qcolor_from_widget,
)


_SETTINGS_ORG = "krzkot"
_SETTINGS_APP = "AstroPlanner"
_SETTINGS_ENV_KEY = "ASTROPLANNER_CONFIG_DIR"


def _safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _config_root_dir() -> Path:
    generic_cfg = str(QStandardPaths.writableLocation(QStandardPaths.GenericConfigLocation) or "").strip()
    if generic_cfg:
        return Path(generic_cfg).expanduser()
    xdg_cfg = str(os.getenv("XDG_CONFIG_HOME", "") or "").strip()
    if xdg_cfg:
        return Path(xdg_cfg).expanduser()
    return Path.home() / ".config"


def _resolve_settings_dir() -> Path:
    try:
        QCoreApplication.setOrganizationName(_SETTINGS_ORG)
        QCoreApplication.setApplicationName(_SETTINGS_APP)
    except Exception:
        pass
    env_override = str(os.getenv(_SETTINGS_ENV_KEY, "") or "").strip()
    if env_override:
        return Path(env_override).expanduser()
    return _config_root_dir() / _SETTINGS_ORG / _SETTINGS_APP


def _create_app_settings() -> SettingsAdapter:
    settings_dir = _resolve_settings_dir()
    try:
        settings_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    storage = AppStorage(settings_dir)
    settings = SettingsAdapter(storage)
    try:
        settings.setValue("__meta/storageBackend", "sqlite")
    except Exception:
        pass
    return settings


class SuggestedTargetsDelegate(QStyledItemDelegate):
    """Custom paint for Suggested Targets so row hover and Add action stay visually obvious."""

    @staticmethod
    def _brush_to_color(value: object) -> QColor:
        if isinstance(value, QBrush):
            return QColor(value.color())
        if isinstance(value, QColor):
            return QColor(value)
        return QColor()

    def paint(self, painter, option, index):
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        widget = opt.widget
        style = widget.style() if widget is not None else QApplication.style()
        model = index.model()
        row = index.row()

        cell_rect = opt.rect.adjusted(0, 0, -1, -1)
        bg_color = self._brush_to_color(index.data(Qt.BackgroundRole))
        fg_color = self._brush_to_color(index.data(Qt.ForegroundRole))
        if not bg_color.isValid():
            if index.row() % 2 == 1:
                bg_color = _theme_qcolor_from_widget(widget, "table_alt", "#131a28")
            else:
                bg_color = _theme_qcolor_from_widget(widget, "table_bg", "#0f1722")
        if not fg_color.isValid():
            fg_color = _theme_qcolor_from_widget(widget, "base_text", "#f6fbff")

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(cell_rect, bg_color)

        if opt.state & QStyle.State_Selected:
            overlay = _theme_qcolor_from_widget(widget, "table_sel_bg", "#163455")
            overlay.setAlpha(56 if bg_color.alpha() >= 140 else 96)
            painter.fillRect(cell_rect, overlay)

        draw_opt = QStyleOptionViewItem(opt)
        draw_opt.text = ""
        if draw_opt.state & QStyle.State_Selected:
            draw_opt.state &= ~QStyle.State_Selected
        style.drawControl(QStyle.CE_ItemViewItem, draw_opt, painter, widget)

        text = str(index.data(Qt.DisplayRole) or "")
        if text:
            draw_font = QFont(index.data(Qt.FontRole) or draw_opt.font)
            painter.setFont(draw_font)
            text_rect = style.subElementRect(QStyle.SE_ItemViewItemText, draw_opt, widget)
            if not text_rect.isValid():
                text_rect = cell_rect.adjusted(8, 0, -8, 0)
            text_rect = text_rect.adjusted(2, 0, -2, 0)
            alignment = index.data(Qt.TextAlignmentRole)
            flags = int(alignment) if alignment is not None else int(Qt.AlignLeft | Qt.AlignVCenter)
            flags |= int(Qt.TextSingleLine)
            elided = QFontMetrics(draw_font).elidedText(text, Qt.ElideRight, max(10, text_rect.width()))
            hover_glow_color = _theme_qcolor_from_widget(widget, "button_hover_glow", "#ff7ecf")
            row_hover_glow = row == getattr(model, "_hover_row", None)

            if row_hover_glow:
                glow = QColor(hover_glow_color)
                glow.setAlpha(70)
                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    painter.setPen(glow)
                    painter.drawText(text_rect.translated(dx, dy), flags, elided)

            painter.setPen(fg_color)
            painter.drawText(text_rect, flags, elided)

        painter.restore()


class SuggestionTableModel(QAbstractTableModel):
    COL_NAME = 0
    COL_TYPE = 1
    COL_MAG = 2
    COL_PRIORITY = 3
    COL_IMPORTANCE = 4
    COL_SCORE = 5
    COL_AIRMASS = 6
    COL_HOURS = 7
    COL_WINDOW = 8
    COL_MOON_SEP = 9
    COL_ACTION = 10

    headers = [
        "Name",
        "Type",
        "Last Mag",
        "Pri",
        "Importance",
        "Score",
        "Min Airmass",
        "Over Lim (h)",
        "Best Window",
        "Moon Sep",
        "Add",
    ]

    def __init__(
        self,
        suggestions: list[dict[str, object]],
        moon_sep_threshold: float,
        mag_warning_threshold: float = DEFAULT_LIMITING_MAGNITUDE,
        parent=None,
    ):
        super().__init__(parent)
        self._all_rows = suggestions
        self._visible_rows: list[dict[str, object]] = []
        self._min_importance = BHTOM_SUGGESTION_MIN_IMPORTANCE
        self._min_score = 0.0
        self._min_hours = 0.0
        self._min_moon_sep = 0.0
        self._max_airmass = 99.0
        self._max_magnitude = 99.0
        self._moon_sep_threshold = float(moon_sep_threshold)
        self._mag_warning_threshold = float(mag_warning_threshold)
        self._sort_column = self.COL_SCORE
        self._sort_order = Qt.DescendingOrder
        self._hover_row: Optional[int] = None
        self._hover_name_row: Optional[int] = None
        self._hover_action_row: Optional[int] = None
        self._rebuild_rows()

    @staticmethod
    def _target(item: dict[str, object]) -> Target:
        target = item["target"]
        assert isinstance(target, Target)
        return target

    @staticmethod
    def _metrics(item: dict[str, object]) -> TargetNightMetrics:
        metrics = item["metrics"]
        assert isinstance(metrics, TargetNightMetrics)
        return metrics

    @staticmethod
    def _window_start(item: dict[str, object]) -> datetime:
        value = item["window_start"]
        assert isinstance(value, datetime)
        return value

    @staticmethod
    def _window_end(item: dict[str, object]) -> datetime:
        value = item["window_end"]
        assert isinstance(value, datetime)
        return value

    @staticmethod
    def _optional_float_key(value: object, descending: bool) -> tuple[int, float]:
        number = _safe_float(value)
        if number is None or not math.isfinite(number):
            return (1, 0.0)
        return (0, -float(number) if descending else float(number))

    @staticmethod
    def _numeric_key(value: object, descending: bool) -> float:
        number = _safe_float(value)
        if number is None or not math.isfinite(number):
            number = 0.0
        return -float(number) if descending else float(number)

    def _passes_filters(self, item: dict[str, object]) -> bool:
        target = self._target(item)
        metrics = self._metrics(item)
        importance = float(item.get("importance", 0.0) or 0.0)
        if importance < self._min_importance:
            return False
        if metrics.score < self._min_score:
            return False
        if metrics.hours_above_limit < self._min_hours:
            return False
        min_window_moon_sep = _safe_float(item.get("min_window_moon_sep"))
        if self._min_moon_sep > 0.0:
            if min_window_moon_sep is None or not math.isfinite(min_window_moon_sep) or min_window_moon_sep < self._min_moon_sep:
                return False
        best_airmass = _safe_float(item.get("best_airmass"))
        if self._max_airmass < 99.0:
            if best_airmass is None or not math.isfinite(best_airmass) or best_airmass > self._max_airmass:
                return False
        if self._max_magnitude < 99.0:
            if target.magnitude is None or target.magnitude > self._max_magnitude:
                return False
        return True

    def _sort_rows(self, rows: list[dict[str, object]]) -> None:
        descending = self._sort_order == Qt.DescendingOrder
        col = self._sort_column
        if col == self.COL_NAME:
            rows.sort(key=lambda item: self._target(item).name.lower(), reverse=descending)
        elif col == self.COL_TYPE:
            rows.sort(key=lambda item: (self._target(item).object_type or "").lower(), reverse=descending)
        elif col == self.COL_MAG:
            rows.sort(key=lambda item: self._optional_float_key(self._target(item).magnitude, descending))
        elif col == self.COL_PRIORITY:
            rows.sort(key=lambda item: self._numeric_key(self._target(item).priority, descending))
        elif col == self.COL_IMPORTANCE:
            rows.sort(key=lambda item: self._numeric_key(item.get("importance"), descending))
        elif col == self.COL_SCORE:
            rows.sort(key=lambda item: self._numeric_key(self._metrics(item).score, descending))
        elif col == self.COL_AIRMASS:
            rows.sort(key=lambda item: self._optional_float_key(item.get("best_airmass"), descending))
        elif col == self.COL_HOURS:
            rows.sort(key=lambda item: self._numeric_key(self._metrics(item).hours_above_limit, descending))
        elif col == self.COL_WINDOW:
            rows.sort(
                key=lambda item: self._window_start(item).timestamp() * (-1.0 if descending else 1.0)
            )
        elif col == self.COL_MOON_SEP:
            rows.sort(key=lambda item: self._optional_float_key(item.get("min_window_moon_sep"), descending))
        elif col == self.COL_ACTION:
            rows.sort(
                key=lambda item: (
                    bool(item.get("added_to_plan")),
                    self._target(item).name.lower(),
                )
            )
            if descending:
                rows.reverse()
        else:
            rows.sort(
                key=lambda item: (
                    -float(item.get("importance", 0.0) or 0.0),
                    -float(self._metrics(item).score),
                    self._target(item).name.lower(),
                )
            )

    def _rebuild_rows(self) -> None:
        self.beginResetModel()
        filtered = [item for item in self._all_rows if self._passes_filters(item)]
        self._sort_rows(filtered)
        self._visible_rows = filtered
        self._hover_row = None
        self._hover_name_row = None
        self._hover_action_row = None
        self.endResetModel()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return len(self._visible_rows)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return len(self.headers)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):  # noqa: N802
        if not index.isValid() or not (0 <= index.row() < len(self._visible_rows)):
            return None

        item = self._visible_rows[index.row()]
        target = self._target(item)
        metrics = self._metrics(item)
        row = index.row()
        col = index.column()

        if role == Qt.TextAlignmentRole:
            if col in {self.COL_NAME, self.COL_TYPE, self.COL_WINDOW}:
                return Qt.AlignLeft | Qt.AlignVCenter
            return Qt.AlignCenter | Qt.AlignVCenter

        if role == Qt.FontRole and col in {self.COL_NAME, self.COL_ACTION}:
            font = QFont()
            font.setBold(True)
            if col == self.COL_NAME and row == self._hover_name_row:
                font.setUnderline(True)
            return font

        if role == Qt.ToolTipRole:
            if col == self.COL_NAME:
                return f"Open in BHTOM: {target.name}"
            if col == self.COL_WINDOW:
                return (
                    f"{self._window_start(item).strftime('%Y-%m-%d %H:%M')} -> "
                    f"{self._window_end(item).strftime('%Y-%m-%d %H:%M')}"
                )
            if col == self.COL_MOON_SEP and item.get("moon_sep_warning"):
                min_sep = _safe_float(item.get("min_window_moon_sep"))
                if min_sep is not None:
                    return (
                        f"Moon separation drops to {min_sep:.1f} deg in the best window "
                        f"(threshold {self._moon_sep_threshold:.0f} deg)."
                    )
            if col == self.COL_ACTION and item.get("added_to_plan"):
                return "This target has already been added to the current plan."

        if role == Qt.BackgroundRole:
            hover_bg = _theme_color_from_widget(self, "table_hover_bg", "#ff3d78")
            warning_bg = _theme_color_from_widget(self, "table_warning_bg", "#fff1a8")
            action_done_bg = _theme_color_from_widget(self, "table_action_done_bg", "#e6f4ea")
            action_hover_bg = _theme_color_from_widget(self, "table_action_hover_bg", "#d8e2ef")
            action_bg = _theme_color_from_widget(self, "table_action_bg", "#c7d1dc")
            if row == self._hover_row:
                return QBrush(QColor(hover_bg))
            if col == self.COL_MOON_SEP and item.get("moon_sep_warning"):
                return QBrush(QColor(warning_bg))
            if col == self.COL_ACTION:
                if item.get("added_to_plan"):
                    return QBrush(QColor(action_done_bg))
                if row == self._hover_action_row:
                    return QBrush(QColor(action_hover_bg))
                return QBrush(QColor(action_bg))

        if role == Qt.ForegroundRole:
            warning_text = _theme_color_from_widget(self, "table_warning_text", "#3f3200")
            action_done_text = _theme_color_from_widget(self, "table_action_done_text", "#4f6f52")
            action_hover_text = _theme_color_from_widget(self, "table_action_hover_text", "#17212b")
            action_text = _theme_color_from_widget(self, "table_action_text", "#243241")
            hover_name_text = _theme_color_from_widget(self, "accent_primary", "#59f3ff")
            state_warning = _theme_color_from_widget(self, "state_warning", "#c9a227")
            state_error = _theme_color_from_widget(self, "state_error", "#b85a5a")
            if col == self.COL_NAME and row == self._hover_name_row:
                return QBrush(QColor(hover_name_text))
            if col == self.COL_MAG and target.magnitude is not None and math.isfinite(float(target.magnitude)):
                mag_value = float(target.magnitude)
                delta = abs(mag_value - self._mag_warning_threshold)
                if delta <= 1.0:
                    return QBrush(QColor(state_warning))
                if mag_value > self._mag_warning_threshold:
                    return QBrush(QColor(state_error))
            if col == self.COL_MOON_SEP and item.get("moon_sep_warning"):
                return QBrush(QColor(warning_text))
            if col == self.COL_ACTION:
                if item.get("added_to_plan"):
                    return QBrush(QColor(action_done_text))
                if row == self._hover_action_row:
                    return QBrush(QColor(action_hover_text))
                return QBrush(QColor(action_text))

        if role not in (Qt.DisplayRole, Qt.EditRole):
            return None

        if col == self.COL_NAME:
            return target.name
        if col == self.COL_TYPE:
            return target.object_type or "-"
        if col == self.COL_MAG:
            return f"{target.magnitude:.2f}" if target.magnitude is not None else "-"
        if col == self.COL_PRIORITY:
            return str(target.priority)
        if col == self.COL_IMPORTANCE:
            return f"{float(item.get('importance', 0.0) or 0.0):.1f}"
        if col == self.COL_SCORE:
            return f"{metrics.score:.1f}"
        if col == self.COL_AIRMASS:
            best_airmass = _safe_float(item.get("best_airmass"))
            return f"{best_airmass:.2f}" if best_airmass is not None else "-"
        if col == self.COL_HOURS:
            return f"{metrics.hours_above_limit:.1f}"
        if col == self.COL_WINDOW:
            return f"{self._window_start(item).strftime('%H:%M')} - {self._window_end(item).strftime('%H:%M')}"
        if col == self.COL_MOON_SEP:
            min_sep = _safe_float(item.get("min_window_moon_sep"))
            return f"{min_sep:.1f}" if min_sep is not None else "-"
        if col == self.COL_ACTION:
            return translate_text("Added" if item.get("added_to_plan") else "Add", current_language())
        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole):  # noqa: N802
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return translate_text(self.headers[section], current_language())
        return None

    def flags(self, index: QModelIndex):  # noqa: N802
        if not index.isValid():
            return Qt.ItemIsEnabled
        return Qt.ItemIsSelectable | Qt.ItemIsEnabled

    def sort(self, column: int, order: Qt.SortOrder = Qt.AscendingOrder) -> None:
        self._sort_column = int(column)
        self._sort_order = order
        self._rebuild_rows()

    def set_filters(
        self,
        min_importance: float,
        min_score: float,
        min_hours: float,
        min_moon_sep: float,
        max_airmass: float,
        max_magnitude: float,
    ) -> None:
        self._min_importance = float(min_importance)
        self._min_score = float(min_score)
        self._min_hours = float(min_hours)
        self._min_moon_sep = float(min_moon_sep)
        self._max_airmass = float(max_airmass)
        self._max_magnitude = float(max_magnitude)
        self._rebuild_rows()

    def filtered_count(self) -> int:
        return sum(1 for item in self._all_rows if self._passes_filters(item))

    def total_count(self) -> int:
        return len(self._all_rows)

    def suggestion_at(self, row: int) -> dict[str, object]:
        return self._visible_rows[row]

    def mark_added(self, target: Target) -> None:
        changed = False
        for item in self._all_rows:
            item_target = item.get("target")
            if isinstance(item_target, Target) and targets_match(item_target, target):
                item["added_to_plan"] = True
                changed = True
        if changed:
            self._rebuild_rows()

    def replace_suggestions(self, suggestions: list[dict[str, object]]) -> None:
        self.beginResetModel()
        self._all_rows = suggestions
        self._visible_rows = []
        self._hover_row = None
        self._hover_name_row = None
        self._hover_action_row = None
        self.endResetModel()
        self._rebuild_rows()

    def set_action_hover_row(self, row: Optional[int]) -> None:
        new_row = row if row is not None and 0 <= row < len(self._visible_rows) else None
        if new_row == self._hover_action_row:
            return
        old_row = self._hover_action_row
        self._hover_action_row = new_row
        for changed_row in (old_row, new_row):
            if changed_row is None:
                continue
            idx = self.index(changed_row, self.COL_ACTION)
            self.dataChanged.emit(idx, idx, [Qt.BackgroundRole, Qt.ForegroundRole])

    def set_hover_row(self, row: Optional[int]) -> None:
        new_row = row if row is not None and 0 <= row < len(self._visible_rows) else None
        if new_row == self._hover_row:
            return
        old_row = self._hover_row
        self._hover_row = new_row
        for changed_row in (old_row, new_row):
            if changed_row is None:
                continue
            left = self.index(changed_row, 0)
            right = self.index(changed_row, self.columnCount() - 1)
            self.dataChanged.emit(
                left,
                right,
                [Qt.BackgroundRole, Qt.ForegroundRole, Qt.FontRole],
            )

    def set_name_hover_row(self, row: Optional[int]) -> None:
        new_row = row if row is not None and 0 <= row < len(self._visible_rows) else None
        if new_row == self._hover_name_row:
            return
        old_row = self._hover_name_row
        self._hover_name_row = new_row
        for changed_row in (old_row, new_row):
            if changed_row is None:
                continue
            idx = self.index(changed_row, self.COL_NAME)
            self.dataChanged.emit(idx, idx, [Qt.BackgroundRole, Qt.ForegroundRole, Qt.FontRole])


class SuggestedTargetsDialog(QDialog):
    def __init__(
        self,
        suggestions: list[dict[str, object]],
        notes: list[str],
        moon_sep_threshold: float,
        mag_warning_threshold: float,
        initial_score_filter: float,
        bhtom_base_url: str,
        add_callback: Callable[[Target], bool],
        reload_callback: Optional[Callable[[], tuple[list[dict[str, object]], list[str]]]] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setObjectName(self.__class__.__name__)
        self.setWindowTitle("Suggested Targets")
        _fit_dialog_to_screen(
            self,
            preferred_width=1460,
            preferred_height=860,
            min_width=1180,
            min_height=720,
        )
        self._add_callback = add_callback
        self._reload_callback = reload_callback
        self._bhtom_base_url = bhtom_base_url.rstrip("/")
        self._notes = notes
        self._settings = (
            parent.settings
            if parent is not None and hasattr(parent, "settings")
            else _create_app_settings()
        )
        self._filter_defaults = {
            "importance": float(BHTOM_SUGGESTION_MIN_IMPORTANCE),
            "score": float(initial_score_filter),
            "hours": 0.0,
            "moon_sep": float(moon_sep_threshold),
            "airmass": 99.0,
            "magnitude": 99.0,
        }

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self.summary_label = QLabel(self)
        self.summary_label.setWordWrap(True)
        _set_label_tone(self.summary_label, "muted")
        layout.addWidget(self.summary_label)

        self.source_label = QLabel(self)
        self.source_label.setWordWrap(True)
        self.source_label.setText(bhtom_suggestion_source_message("loading"))
        _set_label_tone(self.source_label, "muted")
        layout.addWidget(self.source_label)

        filters_row = QHBoxLayout()
        filters_row.setSpacing(8)
        filters_row.addWidget(QLabel("Importance >=", self))
        self.importance_spin = QDoubleSpinBox(self)
        self.importance_spin.setRange(0.0, 10.0)
        self.importance_spin.setDecimals(1)
        self.importance_spin.setSingleStep(0.5)
        self.importance_spin.setValue(BHTOM_SUGGESTION_MIN_IMPORTANCE)
        self.importance_spin.setToolTip("Minimum BHTOM importance.")
        filters_row.addWidget(self.importance_spin)

        filters_row.addWidget(QLabel("Score >=", self))
        self.score_spin = QDoubleSpinBox(self)
        self.score_spin.setRange(0.0, 100.0)
        self.score_spin.setDecimals(1)
        self.score_spin.setSingleStep(1.0)
        self.score_spin.setValue(float(initial_score_filter))
        filters_row.addWidget(self.score_spin)

        filters_row.addWidget(QLabel("Over Lim >=", self))
        self.hours_spin = QDoubleSpinBox(self)
        self.hours_spin.setRange(0.0, 24.0)
        self.hours_spin.setDecimals(1)
        self.hours_spin.setSingleStep(0.5)
        self.hours_spin.setValue(0.0)
        self.hours_spin.setToolTip("Minimum hours above the altitude limit in the observing window.")
        filters_row.addWidget(self.hours_spin)

        filters_row.addWidget(QLabel("Moon Sep >=", self))
        self.moon_sep_spin = QDoubleSpinBox(self)
        self.moon_sep_spin.setRange(0.0, 180.0)
        self.moon_sep_spin.setDecimals(1)
        self.moon_sep_spin.setSingleStep(1.0)
        self.moon_sep_spin.setValue(float(moon_sep_threshold))
        self.moon_sep_spin.setToolTip("Minimum Moon separation in the best observing window.")
        filters_row.addWidget(self.moon_sep_spin)

        filters_row.addWidget(QLabel("Min Airmass <=", self))
        self.airmass_spin = QDoubleSpinBox(self)
        self.airmass_spin.setRange(1.0, 99.0)
        self.airmass_spin.setDecimals(2)
        self.airmass_spin.setSingleStep(0.1)
        self.airmass_spin.setValue(99.0)
        self.airmass_spin.setToolTip(
            "Maximum minimum airmass reached at any point of the observing night "
            "(within Sun-alt and altitude-limit masks). Leave at 99 to disable."
        )
        filters_row.addWidget(self.airmass_spin)

        filters_row.addWidget(QLabel("Mag <=", self))
        self.magnitude_spin = QDoubleSpinBox(self)
        self.magnitude_spin.setRange(-5.0, 99.0)
        self.magnitude_spin.setDecimals(1)
        self.magnitude_spin.setSingleStep(0.5)
        self.magnitude_spin.setValue(99.0)
        self.magnitude_spin.setToolTip("Maximum magnitude. Leave at 99 to disable this filter.")
        filters_row.addWidget(self.magnitude_spin)

        filters_row.addStretch(1)
        self.reset_filters_btn = QPushButton("Reset", self)
        self.reset_filters_btn.setToolTip("Restore the default suggested-target filters.")
        _set_button_variant(self.reset_filters_btn, "ghost")
        _set_button_icon_kind(self.reset_filters_btn, "clear")
        filters_row.addWidget(self.reset_filters_btn)
        self.reload_btn = QPushButton("Reload", self)
        self.reload_btn.setToolTip("Fetch a fresh BHTOM target list and rebuild suggestions.")
        self.reload_btn.setEnabled(self._reload_callback is not None)
        _set_button_variant(self.reload_btn, "secondary")
        _set_button_icon_kind(self.reload_btn, "refresh")
        filters_row.addWidget(self.reload_btn)
        layout.addLayout(filters_row)

        self._restore_filter_settings()

        self.table_model = SuggestionTableModel(
            suggestions,
            moon_sep_threshold,
            mag_warning_threshold=mag_warning_threshold,
            parent=self,
        )
        self.table_view = QTableView(self)
        self.table_view.setObjectName("SuggestionTable")
        self.table_view.setModel(self.table_model)
        self.table_view.setItemDelegate(SuggestedTargetsDelegate(self.table_view))
        self.table_view.setSelectionBehavior(QTableView.SelectRows)
        self.table_view.setSelectionMode(QTableView.SingleSelection)
        self.table_view.setSortingEnabled(True)
        self.table_view.setEditTriggers(QTableView.NoEditTriggers)
        self.table_view.setHorizontalScrollMode(QTableView.ScrollPerPixel)
        self.table_view.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.table_view.setMouseTracking(True)
        self.table_view.setAttribute(Qt.WA_Hover, True)
        self.table_view.viewport().setMouseTracking(True)
        self.table_view.viewport().setAttribute(Qt.WA_Hover, True)
        self.table_view.verticalHeader().setVisible(False)
        self.table_view.setAlternatingRowColors(True)
        self.table_view.setShowGrid(False)
        self.table_view.viewport().installEventFilter(self)
        header = self.table_view.horizontalHeader()
        header.setSectionsClickable(True)
        header.setStretchLastSection(True)
        header.setSectionResizeMode(QHeaderView.Interactive)
        self.table_view.setColumnWidth(
            SuggestionTableModel.COL_NAME,
            self._default_name_column_width(suggestions),
        )
        self.table_view.setColumnWidth(SuggestionTableModel.COL_TYPE, 128)
        self.table_view.setColumnWidth(SuggestionTableModel.COL_MAG, 64)
        self.table_view.setColumnWidth(SuggestionTableModel.COL_PRIORITY, 42)
        self.table_view.setColumnWidth(SuggestionTableModel.COL_IMPORTANCE, 88)
        self.table_view.setColumnWidth(SuggestionTableModel.COL_SCORE, 76)
        self.table_view.setColumnWidth(SuggestionTableModel.COL_AIRMASS, 96)
        self.table_view.setColumnWidth(SuggestionTableModel.COL_HOURS, 92)
        self.table_view.setColumnWidth(SuggestionTableModel.COL_WINDOW, 140)
        self.table_view.setColumnWidth(SuggestionTableModel.COL_MOON_SEP, 90)
        self.table_view.setColumnWidth(SuggestionTableModel.COL_ACTION, 74)
        self.table_loading_widget = self._build_table_loading_placeholder("Loading BHTOM targets...")
        self.table_stack_host = QWidget(self)
        self.table_stack = QStackedLayout(self.table_stack_host)
        self.table_stack.setContentsMargins(0, 0, 0, 0)
        self.table_stack.addWidget(self.table_loading_widget)
        self.table_stack.addWidget(self.table_view)
        self.table_stack.setCurrentWidget(self.table_view)
        layout.addWidget(self.table_stack_host, 1)

        self.notes_label = QLabel(self)
        self.notes_label.setWordWrap(True)
        self.notes_label.setVisible(bool(notes))
        layout.addWidget(self.notes_label)

        buttons = QDialogButtonBox(QDialogButtonBox.Close, self)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        _style_dialog_button_box(buttons)
        layout.addWidget(buttons)

        self.importance_spin.valueChanged.connect(self._apply_filters)
        self.score_spin.valueChanged.connect(self._apply_filters)
        self.hours_spin.valueChanged.connect(self._apply_filters)
        self.moon_sep_spin.valueChanged.connect(self._apply_filters)
        self.airmass_spin.valueChanged.connect(self._apply_filters)
        self.magnitude_spin.valueChanged.connect(self._apply_filters)
        self.reset_filters_btn.clicked.connect(self._reset_filters_to_defaults)
        self.reload_btn.clicked.connect(self._reload_suggestions)
        self.table_model.modelReset.connect(self._refresh_dialog_state)
        self.table_view.clicked.connect(self._on_table_clicked)
        self.table_view.entered.connect(self._on_table_entered)

        self.table_view.sortByColumn(SuggestionTableModel.COL_SCORE, Qt.DescendingOrder)
        self._apply_filters()
        self._fit_table_columns_to_width()
        localize_widget_tree(self, current_language())

    def set_source_message(self, message: str) -> None:
        self.source_label.setText(str(message or "").strip())
        self.source_label.setVisible(bool(str(message or "").strip()))
        localize_widget_tree(self, current_language())

    def _build_table_loading_placeholder(self, message: str) -> QWidget:
        widget = QWidget(self)
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        card = QFrame(widget)
        card.setObjectName("VisibilityLoadingCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(20, 18, 20, 18)
        card_layout.setSpacing(14)
        title = QLabel("Suggested Targets", card)
        title.setObjectName("SectionTitle")
        title.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        skeleton = SkeletonShimmerWidget("table", card)
        skeleton.setMinimumHeight(380)
        hint_label = QLabel(message, card)
        hint_label.setObjectName("SectionHint")
        hint_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        hint_label.setWordWrap(True)
        card_layout.addWidget(title)
        card_layout.addWidget(skeleton, 1)
        card_layout.addWidget(hint_label)
        layout.addWidget(card, 1)
        widget._loading_hint_label = hint_label  # type: ignore[attr-defined]
        return widget

    def set_loading_state(self, loading: bool, message: str = "") -> None:
        is_loading = bool(loading)
        hint_label = getattr(self.table_loading_widget, "_loading_hint_label", None)
        if message and isinstance(hint_label, QLabel):
            hint_label.setText(message)
        self.table_view.setEnabled(not is_loading)
        if is_loading:
            self.table_model.set_hover_row(None)
            self.table_model.set_name_hover_row(None)
            self.table_model.set_action_hover_row(None)
            self.table_view.viewport().setCursor(Qt.CursorShape.ArrowCursor)
            self.table_stack.setCurrentWidget(self.table_loading_widget)
            localize_widget_tree(self, current_language())
            return
        self.table_stack.setCurrentWidget(self.table_view)
        localize_widget_tree(self, current_language())

    def update_suggestions(self, suggestions: list[dict[str, object]], notes: list[str]) -> None:
        self._notes = list(notes)
        self.table_model.replace_suggestions(suggestions)
        self.table_view.setColumnWidth(
            SuggestionTableModel.COL_NAME,
            self._default_name_column_width(suggestions),
        )
        self._fit_table_columns_to_width()
        self._refresh_dialog_state()

    @staticmethod
    def _table_stretch_weights() -> dict[int, float]:
        return {
            SuggestionTableModel.COL_NAME: 4.0,
            SuggestionTableModel.COL_TYPE: 1.6,
            SuggestionTableModel.COL_IMPORTANCE: 1.0,
            SuggestionTableModel.COL_SCORE: 1.0,
            SuggestionTableModel.COL_AIRMASS: 1.1,
            SuggestionTableModel.COL_HOURS: 1.1,
            SuggestionTableModel.COL_WINDOW: 2.0,
            SuggestionTableModel.COL_MOON_SEP: 1.1,
        }

    def _fit_table_columns_to_width(self) -> None:
        if not hasattr(self, "table_view"):
            return
        widths = {
            col: int(self.table_view.columnWidth(col))
            for col in range(self.table_model.columnCount())
            if not self.table_view.isColumnHidden(col)
        }
        if not widths:
            return
        viewport_width = max(0, int(self.table_view.viewport().width()) - 2)
        fitted = _distribute_extra_table_width(
            widths,
            available_width=viewport_width,
            stretch_weights=self._table_stretch_weights(),
        )
        for col, width in fitted.items():
            self.table_view.setColumnWidth(col, int(width))

    def _default_name_column_width(self, suggestions: list[dict[str, object]]) -> int:
        name_font = QFont(self.table_view.font())
        name_font.setBold(True)
        name_metrics = QFontMetrics(name_font)
        header_metrics = QFontMetrics(self.table_view.horizontalHeader().font())
        max_width = header_metrics.horizontalAdvance(
            translate_text(SuggestionTableModel.headers[SuggestionTableModel.COL_NAME], current_language())
        )
        for item in suggestions:
            target = item.get("target")
            if isinstance(target, Target):
                max_width = max(max_width, name_metrics.horizontalAdvance(target.name))
        return max_width + 24

    @Slot()
    def _apply_filters(self) -> None:
        self._save_filter_settings()
        self.table_model.set_filters(
            self.importance_spin.value(),
            self.score_spin.value(),
            self.hours_spin.value(),
            self.moon_sep_spin.value(),
            self.airmass_spin.value(),
            self.magnitude_spin.value(),
        )

    def _restore_filter_settings(self) -> None:
        self.importance_spin.setValue(
            self._settings.value(
                "suggestions/minImportance",
                self._filter_defaults["importance"],
                type=float,
            )
        )
        self.score_spin.setValue(
            self._settings.value(
                "suggestions/minScore",
                self._filter_defaults["score"],
                type=float,
            )
        )
        self.hours_spin.setValue(
            self._settings.value(
                "suggestions/minHours",
                self._filter_defaults["hours"],
                type=float,
            )
        )
        self.moon_sep_spin.setValue(
            self._settings.value(
                "suggestions/minMoonSep",
                self._filter_defaults["moon_sep"],
                type=float,
            )
        )
        self.airmass_spin.setValue(
            self._settings.value(
                "suggestions/maxAirmass",
                self._filter_defaults["airmass"],
                type=float,
            )
        )
        self.magnitude_spin.setValue(
            self._settings.value(
                "suggestions/maxMagnitude",
                self._filter_defaults["magnitude"],
                type=float,
            )
        )

    def _save_filter_settings(self) -> None:
        self._settings.setValue("suggestions/minImportance", self.importance_spin.value())
        self._settings.setValue("suggestions/minScore", self.score_spin.value())
        self._settings.setValue("suggestions/minHours", self.hours_spin.value())
        self._settings.setValue("suggestions/minMoonSep", self.moon_sep_spin.value())
        self._settings.setValue("suggestions/maxAirmass", self.airmass_spin.value())
        self._settings.setValue("suggestions/maxMagnitude", self.magnitude_spin.value())

    @Slot()
    def _reset_filters_to_defaults(self) -> None:
        blockers = [
            QSignalBlocker(self.importance_spin),
            QSignalBlocker(self.score_spin),
            QSignalBlocker(self.hours_spin),
            QSignalBlocker(self.moon_sep_spin),
            QSignalBlocker(self.airmass_spin),
            QSignalBlocker(self.magnitude_spin),
        ]
        self.importance_spin.setValue(self._filter_defaults["importance"])
        self.score_spin.setValue(self._filter_defaults["score"])
        self.hours_spin.setValue(self._filter_defaults["hours"])
        self.moon_sep_spin.setValue(self._filter_defaults["moon_sep"])
        self.airmass_spin.setValue(self._filter_defaults["airmass"])
        self.magnitude_spin.setValue(self._filter_defaults["magnitude"])
        del blockers
        self._apply_filters()

    @Slot()
    def _reload_suggestions(self) -> None:
        if self._reload_callback is None:
            return
        self.reload_btn.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            suggestions, notes = self._reload_callback()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Reload Suggestions", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()
            self.reload_btn.setEnabled(True)

        self.update_suggestions(suggestions, notes)

    @Slot()
    def _refresh_dialog_state(self) -> None:
        filtered = self.table_model.filtered_count()
        total = self.table_model.total_count()
        self.summary_label.setText(
            f"Showing {filtered} matching BHTOM targets "
            f"(loaded {total}, base importance >= {BHTOM_SUGGESTION_MIN_IMPORTANCE:.1f})."
        )
        if self._notes:
            self.notes_label.setText("Notes: " + " | ".join(self._notes))
            self.notes_label.setVisible(True)
        else:
            self.notes_label.clear()
            self.notes_label.setVisible(False)
        localize_widget_tree(self, current_language())

    @Slot(QModelIndex)
    def _on_table_clicked(self, index: QModelIndex) -> None:
        if not index.isValid():
            return
        if index.column() == SuggestionTableModel.COL_NAME:
            self._open_bhtom_target(index.row())
            return
        if index.column() != SuggestionTableModel.COL_ACTION:
            return
        item = self.table_model.suggestion_at(index.row())
        if bool(item.get("added_to_plan")):
            return
        self._add_row_to_plan(index.row())

    @Slot(QModelIndex)
    def _on_table_entered(self, index: QModelIndex) -> None:
        hover_row: Optional[int] = index.row() if index.isValid() else None
        name_hover_row: Optional[int] = None
        action_hover_row: Optional[int] = None
        if index.isValid():
            if index.column() == SuggestionTableModel.COL_NAME:
                name_hover_row = index.row()
            elif index.column() == SuggestionTableModel.COL_ACTION:
                item = self.table_model.suggestion_at(index.row())
                if not bool(item.get("added_to_plan")):
                    action_hover_row = index.row()
        self.table_model.set_hover_row(hover_row)
        self.table_model.set_name_hover_row(name_hover_row)
        self.table_model.set_action_hover_row(action_hover_row)
        self.table_view.viewport().setCursor(
            Qt.CursorShape.PointingHandCursor
            if (action_hover_row is not None or name_hover_row is not None)
            else Qt.CursorShape.ArrowCursor
        )

    def eventFilter(self, watched, event):  # noqa: D401
        if watched is self.table_view.viewport():
            if event.type() in (QEvent.Type.MouseMove, QEvent.Type.HoverMove):
                pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
                self._on_table_entered(self.table_view.indexAt(pos))
            elif event.type() == QEvent.Type.Leave:
                self.table_model.set_hover_row(None)
                self.table_model.set_name_hover_row(None)
                self.table_model.set_action_hover_row(None)
                self.table_view.viewport().setCursor(Qt.CursorShape.ArrowCursor)
        return super().eventFilter(watched, event)

    def resizeEvent(self, event):  # noqa: D401
        super().resizeEvent(event)
        self._fit_table_columns_to_width()

    def _open_bhtom_target(self, row: int) -> None:
        item = self.table_model.suggestion_at(row)
        target = item.get("target")
        if not isinstance(target, Target):
            return
        slug = quote(target.name.strip(), safe="")
        if not slug:
            return
        QDesktopServices.openUrl(QUrl(f"{self._bhtom_base_url}/targets/{slug}/"))

    def _add_row_to_plan(self, row: int) -> None:
        item = self.table_model.suggestion_at(row)
        target = item.get("target")
        if not isinstance(target, Target):
            return
        if self._add_callback(target):
            self.table_model.mark_added(target)


__all__ = [
    "SuggestedTargetsDelegate",
    "SuggestedTargetsDialog",
    "SuggestionTableModel",
]
