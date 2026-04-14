from __future__ import annotations

import json
import math
from typing import Optional

import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import Angle
from astropy.time import Time
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QFontMetrics, QIcon, QPainter, QPen
from PySide6.QtWidgets import QApplication, QStyle, QStyledItemDelegate, QStyleOptionViewItem

from astroplanner.i18n import current_language, translate_text
from astroplanner.models import Site, Target
from astroplanner.ui.theme_utils import _qcolor_from_token, _theme_color_from_widget, _theme_qcolor_from_widget


def _normalize_catalog_token(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _target_magnitude_label(target: Target) -> str:
    return "Last Mag" if _normalize_catalog_token(target.source_catalog) == "bhtom" else "Mag"


def _object_type_is_unknown(value: object) -> bool:
    token = _normalize_catalog_token(value)
    return token in {"", "-", "unknown", "unk", "n/a", "na", "none"}


def _normalized_css_color(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    color = QColor(text)
    if not color.isValid():
        return ""
    return color.name().lower()


def _safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
        0.2126 * float(bg.red()) + 0.7152 * float(bg.green()) + 0.0722 * float(bg.blue())
    ) / 255.0
    return QColor("#101720") if luminance >= 0.62 else QColor("#f6fbff")


_VALID_TABLE_COLOR_MODES = {"background", "text_glow"}


def _normalize_table_color_mode(value: object, *, default: str = "background") -> str:
    mode = str(value or "").strip().lower()
    if mode in _VALID_TABLE_COLOR_MODES:
        return mode
    return default


class TargetTableGlowDelegate(QStyledItemDelegate):
    """Custom paint for the main targets table with visible status tint and subtle neon text glow."""

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
        color_mode = _normalize_table_color_mode(getattr(model, "color_mode", "background"), default="background")
        row_hover = index.row() == getattr(model, "_hover_row", None)

        cell_rect = opt.rect.adjusted(0, 0, -1, -1)
        bg_color = self._brush_to_color(index.data(Qt.BackgroundRole))
        fg_color = self._brush_to_color(index.data(Qt.ForegroundRole))
        if not fg_color.isValid():
            fg_color = _theme_qcolor_from_widget(widget, "base_text", "#f6fbff")
        if not bg_color.isValid():
            if color_mode == "text_glow" and index.row() % 2 == 1:
                bg_color = _theme_qcolor_from_widget(widget, "table_surface_alt", "#101a28")
            else:
                bg_color = _theme_qcolor_from_widget(widget, "table_surface", "#0e1622")

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        painter.fillRect(cell_rect, bg_color)

        if row_hover:
            hover_overlay = _theme_qcolor_from_widget(widget, "table_action_hover_bg", "#d8e2ef")
            if not hover_overlay.isValid():
                hover_overlay = _theme_qcolor_from_widget(widget, "accent_secondary", "#ff4fd8")
            if hover_overlay.isValid():
                overlay = QColor(hover_overlay)
                if overlay.alpha() <= 0:
                    overlay.setAlpha(56 if bg_color.alpha() >= 140 else 88)
                painter.fillRect(cell_rect, overlay)

        if opt.state & QStyle.State_Selected:
            overlay = _theme_qcolor_from_widget(widget, "table_sel_bg", "#163455")
            overlay.setAlpha(52 if bg_color.alpha() >= 140 else 96)
            painter.fillRect(cell_rect, overlay)
            border = _theme_qcolor_from_widget(widget, "accent_primary", "#59f3ff")
            border.setAlpha(132)
            painter.setPen(QPen(border, 1))
            painter.drawRect(cell_rect)
        elif row_hover:
            border = _theme_qcolor_from_widget(widget, "accent_secondary", "#ff4fd8")
            border.setAlpha(108)
            painter.setPen(QPen(border, 1))
            painter.drawRect(cell_rect)

        draw_opt = QStyleOptionViewItem(opt)
        draw_opt.text = ""
        draw_opt.icon = QIcon()
        if draw_opt.state & QStyle.State_Selected:
            draw_opt.state &= ~QStyle.State_Selected
        style.drawControl(QStyle.CE_ItemViewItem, draw_opt, painter, widget)

        text = str(index.data(Qt.DisplayRole) or "")
        if text:
            draw_font = QFont(draw_opt.font)
            if color_mode == "text_glow":
                draw_font.setWeight(
                    QFont.Weight.DemiBold if not (opt.state & QStyle.State_Selected) else QFont.Weight.Bold
                )
            painter.setFont(draw_font)
            draw_opt.font = draw_font
            text_rect = style.subElementRect(QStyle.SE_ItemViewItemText, draw_opt, widget)
            if not text_rect.isValid():
                text_rect = cell_rect.adjusted(8, 0, -8, 0)
            text_rect = text_rect.adjusted(2, 0, -2, 0)
            alignment = index.data(Qt.TextAlignmentRole)
            flags = int(alignment) if alignment is not None else int(Qt.AlignLeft | Qt.AlignVCenter)
            flags |= int(Qt.TextSingleLine)
            elided = QFontMetrics(draw_font).elidedText(text, Qt.ElideRight, max(10, text_rect.width()))

            if color_mode == "text_glow":
                glow = QColor(fg_color)
                if not glow.isValid():
                    glow = _theme_qcolor_from_widget(widget, "accent_primary", "#59f3ff")
                glow.setAlpha(92 if (opt.state & QStyle.State_Selected) else 70)
                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    painter.setPen(glow)
                    painter.drawText(text_rect.translated(dx, dy), flags, elided)
            else:
                glow = QColor(bg_color)
                if not glow.isValid() or glow.alpha() <= 0:
                    glow = _theme_qcolor_from_widget(widget, "accent_primary", "#59f3ff")
                glow.setAlpha(118 if (opt.state & QStyle.State_Selected) else 82)
                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    painter.setPen(glow)
                    painter.drawText(text_rect.translated(dx, dy), flags, elided)

            painter.setPen(fg_color)
            painter.drawText(text_rect, flags, elided)

        painter.restore()


class TargetTableModel(QAbstractTableModel):
    COL_ORDER = 0
    COL_NAME = 1
    COL_RA = 2
    COL_HA = 3
    COL_DEC = 4
    COL_ALT = 5
    COL_AZ = 6
    COL_MOON_SEP = 7
    COL_SCORE = 8
    COL_HOURS = 9
    COL_MAG = 10
    COL_PRIORITY = 11
    COL_OBSERVED = 12
    COL_ACTIONS = 13
    headers = [
        "Order",
        "Name",
        "RA (°)",
        "HA (h)",
        "Dec (°)",
        "Alt (°)",
        "Az (°)",
        "Moon Sep (°)",
        "Score",
        "Over Lim (h)",
        "Last Mag/Mag",
        "Pri",
        "Obs",
        "Actions",
    ]

    def __init__(self, targets: list[Target], site: Optional[Site] = None, parent=None):
        super().__init__(parent)
        self.setObjectName(self.__class__.__name__)
        self._targets = targets
        self.site = site
        self.limit: float | None = None
        self.color_mode = "background"
        self._hover_row: Optional[int] = None
        self.order_values: list[int] = []
        self.current_alts: list[float] = []
        self.current_azs: list[float] = []
        self.current_seps: list[float] = []
        self.scores: list[float] = []
        self.hours_above_limit: list[float] = []
        self.row_enabled: list[bool] = []
        self.color_map: dict[str, QColor] = {}

    def _ensure_cache_lengths(self) -> None:
        n = len(self._targets)
        if len(self.order_values) < n:
            self.order_values.extend([0] * (n - len(self.order_values)))
        if len(self.current_alts) < n:
            self.current_alts.extend([float("nan")] * (n - len(self.current_alts)))
        if len(self.current_azs) < n:
            self.current_azs.extend([float("nan")] * (n - len(self.current_azs)))
        if len(self.current_seps) < n:
            self.current_seps.extend([float("nan")] * (n - len(self.current_seps)))
        if len(self.scores) < n:
            self.scores.extend([0.0] * (n - len(self.scores)))
        if len(self.hours_above_limit) < n:
            self.hours_above_limit.extend([0.0] * (n - len(self.hours_above_limit)))
        if len(self.row_enabled) < n:
            self.row_enabled.extend([True] * (n - len(self.row_enabled)))

    @staticmethod
    def _optional_float_key(value: object, descending: bool) -> tuple[int, float]:
        number = _safe_float(value)
        if number is None or not math.isfinite(number):
            return (1, 0.0)
        return (0, -float(number) if descending else float(number))

    def _table_surface_color(self) -> QColor:
        return _theme_qcolor_from_widget(self, "table_surface", "#0e1622")

    def _table_alt_surface_color(self) -> QColor:
        return _theme_qcolor_from_widget(self, "table_surface_alt", "#101a28")

    def _table_color_mode(self) -> str:
        return _normalize_table_color_mode(getattr(self, "color_mode", "background"), default="background")

    def _boost_table_tint(self, color: QColor, *, strong: bool = False) -> QColor:
        out = QColor(color)
        if not out.isValid():
            return out
        dark_surface = self._table_surface_color().lightness() < 128
        min_alpha = 214 if strong else (160 if dark_surface else 108)
        if out.alpha() < min_alpha:
            out.setAlpha(min_alpha)
        return out

    def _status_color(self, row: int) -> Optional[QColor]:
        if not (self.site and self.limit is not None):
            return None
        alt = self.current_alts[row] if row < len(self.current_alts) else float("nan")
        if math.isnan(alt):
            return None
        hc = getattr(self, "highlight_colors", {})
        if alt < 0:
            color = _qcolor_from_token(hc.get("below"), _theme_color_from_widget(self, "table_status_red", "#ff8080"))
        elif alt < self.limit:
            color = _qcolor_from_token(hc.get("limit"), _theme_color_from_widget(self, "table_status_yellow", "#ffff80"))
        else:
            color = _qcolor_from_token(hc.get("above"), _theme_color_from_widget(self, "table_status_green", "#b3ffb3"))
        return color if color.isValid() else None

    def _cell_background_color(self, row: int, col: int, tgt: Target, row_is_enabled: bool) -> Optional[QColor]:
        if not row_is_enabled:
            return self._boost_table_tint(_theme_qcolor_from_widget(self, "table_disabled_bg", "#ececec"))

        if not (self.site and self.limit is not None):
            return None

        alt = self.current_alts[row] if row < len(self.current_alts) else float("nan")
        if math.isnan(alt):
            return None

        if self._table_color_mode() == "text_glow":
            return None

        colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        fallback_css = colors[row % len(colors)] if colors else "#4da3ff"
        plot_css = _normalized_css_color(tgt.plot_color) or fallback_css
        plot_color = QColor(plot_css)

        if col == self.COL_ALT:
            status_color = self._status_color(row)
            if status_color is not None:
                return self._boost_table_tint(status_color, strong=True)

        if col == self.COL_NAME:
            brush_color = self.color_map.get(tgt.name)
            if brush_color and brush_color.isValid():
                return brush_color
            return plot_color if plot_color.isValid() else None

        status_color = self._status_color(row)
        if status_color is not None:
            return self._boost_table_tint(status_color)
        return None

    def reset_targets(self, targets: list[Target]) -> None:
        self.beginResetModel()
        self._targets[:] = targets
        self._hover_row = None
        self.order_values = []
        self.current_alts = []
        self.current_azs = []
        self.current_seps = []
        self.scores = []
        self.hours_above_limit = []
        self.row_enabled = []
        self.color_map.clear()
        self.endResetModel()

    def append_target(self, target: Target) -> None:
        row = len(self._targets)
        self.beginInsertRows(QModelIndex(), row, row)
        self._targets.append(target)
        self.order_values.append(0)
        self.endInsertRows()

    def set_hover_row(self, row: Optional[int]) -> None:
        new_row = row if row is not None and 0 <= row < len(self._targets) else None
        if new_row == self._hover_row:
            return
        old_row = self._hover_row
        self._hover_row = new_row
        for changed_row in (old_row, new_row):
            if changed_row is None:
                continue
            left = self.index(changed_row, 0)
            right = self.index(changed_row, self.columnCount() - 1)
            self.dataChanged.emit(left, right, [Qt.BackgroundRole, Qt.ForegroundRole, Qt.FontRole])

    def remove_rows(self, rows: list[int]) -> list[Target]:
        removed: list[Target] = []
        for row in sorted(rows, reverse=True):
            if not (0 <= row < len(self._targets)):
                continue
            self.beginRemoveRows(QModelIndex(), row, row)
            removed_target = self._targets.pop(row)
            removed.append(removed_target)
            self.color_map.pop(removed_target.name, None)
            if row < len(self.order_values):
                self.order_values.pop(row)
            if row < len(self.current_alts):
                self.current_alts.pop(row)
            if row < len(self.current_azs):
                self.current_azs.pop(row)
            if row < len(self.current_seps):
                self.current_seps.pop(row)
            if row < len(self.scores):
                self.scores.pop(row)
            if row < len(self.hours_above_limit):
                self.hours_above_limit.pop(row)
            if row < len(self.row_enabled):
                self.row_enabled.pop(row)
            self.endRemoveRows()
        return removed

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return len(self._targets)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return len(self.headers)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):  # noqa: N802
        if not index.isValid():
            return None
        self._ensure_cache_lengths()
        tgt = self._targets[index.row()]
        col = index.column()
        row = index.row()
        row_is_enabled = self.row_enabled[row] if row < len(self.row_enabled) else True
        cell_bg = self._cell_background_color(row, col, tgt, row_is_enabled)

        if role == Qt.TextAlignmentRole:
            if col == self.COL_NAME:
                return Qt.AlignLeft | Qt.AlignVCenter
            return Qt.AlignCenter | Qt.AlignVCenter

        if role in (Qt.DisplayRole, Qt.EditRole) and col == self.COL_HA and self.site:
            now = Time.now()
            loc = self.site.to_earthlocation()
            lst = now.sidereal_time("apparent", loc.lon).hour
            ra_h = self._targets[row].ra / 15.0
            ha = (lst - ra_h + 24) % 24
            ha_angle = Angle(ha, u.hour)
            return ha_angle.to_string(unit=u.hour, sep=":", pad=True, precision=0)

        if role == Qt.BackgroundRole and cell_bg is not None and cell_bg.isValid():
            return QBrush(cell_bg)

        if role == Qt.ForegroundRole:
            if not row_is_enabled:
                return QBrush(QColor(_theme_color_from_widget(self, "table_disabled_text", "#777777")))
            if self._table_color_mode() == "text_glow":
                if col == self.COL_NAME:
                    name_color = QColor(self.color_map.get(tgt.name, QColor()))
                    if not name_color.isValid():
                        custom_css = _normalized_css_color(tgt.plot_color)
                        if custom_css:
                            name_color = QColor(custom_css)
                    if name_color.isValid():
                        return QBrush(name_color)
                status_color = self._status_color(row)
                if status_color is not None and status_color.isValid():
                    status_fg = QColor(status_color)
                    status_fg = status_fg.lighter(118)
                    status_fg.setAlpha(255)
                    return QBrush(status_fg)
            if cell_bg is not None and cell_bg.isValid():
                return QBrush(_contrast_text_for_background(cell_bg, table_bg=self._table_surface_color()))
            return QBrush(QColor(_theme_color_from_widget(self, "base_text", "#000000")))

        if role == Qt.ToolTipRole and col == self.COL_NAME:
            display_type = tgt.object_type
            parent_widget = self.parent()
            if _object_type_is_unknown(display_type) and parent_widget is not None and hasattr(parent_widget, "_bhtom_type_for_target"):
                try:
                    fallback_type = parent_widget._bhtom_type_for_target(tgt)
                except Exception:
                    fallback_type = ""
                if fallback_type:
                    display_type = fallback_type
            extras: list[str] = []
            if display_type and not _object_type_is_unknown(display_type):
                extras.append(f"Type: {display_type}")
            if tgt.magnitude is not None:
                extras.append(f"{_target_magnitude_label(tgt)}: {tgt.magnitude:.1f}")
            if tgt.size_arcmin is not None:
                extras.append(f"Size: {tgt.size_arcmin:.1f}'")
            extras.append(f"Priority: {tgt.priority}")
            extras.append(f"Observed: {'Yes' if tgt.observed else 'No'}")
            if tgt.notes:
                extras.append(f"Notes: {tgt.notes}")
            return "\n".join([tgt.name, *extras])

        if role == Qt.ToolTipRole and col == self.COL_ORDER:
            order_value = self.order_values[row] if row < len(self.order_values) else 0
            if order_value > 0:
                return f"Recommended observing order: {order_value}"
            return "No deterministic observing order is available for this row yet."

        if role == Qt.ToolTipRole and col == self.COL_ALT and self.site:
            alt = self.current_alts[row] if row < len(self.current_alts) else None
            limit = self.limit or 0
            if alt is None or math.isnan(alt):
                return ""
            if alt < 0:
                return "Below horizon"
            if alt < limit:
                return "Below limit altitude"
            return "Above limit altitude"

        if role not in (Qt.DisplayRole, Qt.EditRole):
            return None
        if col == self.COL_ORDER:
            order_value = self.order_values[row] if row < len(self.order_values) else 0
            return str(order_value) if order_value > 0 else ""
        if col == self.COL_NAME:
            return tgt.name
        if col == self.COL_RA:
            ra_angle = Angle(tgt.ra, u.degree)
            return ra_angle.to_string(unit="hourangle", sep=":", pad=True, precision=0)
        if col == self.COL_DEC:
            dec_angle = Angle(tgt.dec, u.degree)
            return dec_angle.to_string(unit="deg", sep=":", alwayssign=True, pad=True, precision=0)
        if col == self.COL_ALT:
            return f"{self.current_alts[row]:.1f}" if row < len(self.current_alts) and not math.isnan(self.current_alts[row]) else ""
        if col == self.COL_AZ:
            return f"{self.current_azs[row]:.1f}" if row < len(self.current_azs) and not math.isnan(self.current_azs[row]) else ""
        if col == self.COL_MOON_SEP:
            return f"{self.current_seps[row]:.1f}" if row < len(self.current_seps) and not math.isnan(self.current_seps[row]) else ""
        if col == self.COL_SCORE:
            return f"{self.scores[row]:.1f}" if row < len(self.scores) else "0.0"
        if col == self.COL_HOURS:
            return f"{self.hours_above_limit[row]:.2f}" if row < len(self.hours_above_limit) else "0.00"
        if col == self.COL_MAG:
            return f"{tgt.magnitude:.2f}" if tgt.magnitude is not None else "-"
        if col == self.COL_PRIORITY:
            return str(tgt.priority)
        if col == self.COL_OBSERVED:
            return "Yes" if tgt.observed else "No"
        if col == self.COL_ACTIONS:
            return ""
        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole):  # noqa: N802
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return translate_text(self.headers[section], current_language())
        return None

    def sort(self, column: int, order: Qt.SortOrder = Qt.AscendingOrder) -> None:
        self.layoutAboutToBeChanged.emit()
        reverse = order == Qt.DescendingOrder
        self._ensure_cache_lengths()
        n = len(self._targets)
        rows = []
        for idx, tgt in enumerate(self._targets):
            rows.append(
                {
                    "target": tgt,
                    "order": self.order_values[idx] if idx < len(self.order_values) else 0,
                    "alt": self.current_alts[idx] if idx < len(self.current_alts) else float("-inf"),
                    "az": self.current_azs[idx] if idx < len(self.current_azs) else float("-inf"),
                    "sep": self.current_seps[idx] if idx < len(self.current_seps) else float("-inf"),
                    "score": self.scores[idx] if idx < len(self.scores) else 0.0,
                    "hours": self.hours_above_limit[idx] if idx < len(self.hours_above_limit) else 0.0,
                    "enabled": self.row_enabled[idx] if idx < len(self.row_enabled) else True,
                }
            )

        if column == self.COL_ORDER:
            rows.sort(
                key=lambda r: (r["order"] <= 0, int(r["order"]) if r["order"] > 0 else 10**9),
                reverse=reverse,
            )
        elif column == self.COL_NAME:
            rows.sort(key=lambda r: r["target"].name.lower(), reverse=reverse)
        elif column == self.COL_RA:
            rows.sort(key=lambda r: r["target"].ra, reverse=reverse)
        elif column == self.COL_HA and self.site:
            now = Time.now()
            lon = self.site.to_earthlocation().lon
            lst = now.sidereal_time("apparent", lon).hour
            rows.sort(
                key=lambda r: (float(lst) - (r["target"].ra / 15.0) + 24.0) % 24.0,  # type: ignore[arg-type]
                reverse=reverse,
            )
        elif column == self.COL_DEC:
            rows.sort(key=lambda r: r["target"].dec, reverse=reverse)
        elif column == self.COL_ALT:
            rows.sort(key=lambda r: r["alt"], reverse=reverse)
        elif column == self.COL_AZ:
            rows.sort(key=lambda r: r["az"], reverse=reverse)
        elif column == self.COL_MOON_SEP:
            rows.sort(key=lambda r: r["sep"], reverse=reverse)
        elif column == self.COL_SCORE:
            rows.sort(key=lambda r: r["score"], reverse=reverse)
        elif column == self.COL_HOURS:
            rows.sort(key=lambda r: r["hours"], reverse=reverse)
        elif column == self.COL_MAG:
            rows.sort(key=lambda r: self._optional_float_key(r["target"].magnitude, reverse))
        elif column == self.COL_PRIORITY:
            rows.sort(key=lambda r: r["target"].priority, reverse=reverse)
        elif column == self.COL_OBSERVED:
            rows.sort(key=lambda r: r["target"].observed, reverse=reverse)

        if rows:
            self._targets[:] = [row["target"] for row in rows]
            self.order_values = [int(row["order"]) for row in rows]
            self.current_alts = [float(row["alt"]) for row in rows]
            self.current_azs = [float(row["az"]) for row in rows]
            self.current_seps = [float(row["sep"]) for row in rows]
            self.scores = [float(row["score"]) for row in rows]
            self.hours_above_limit = [float(row["hours"]) for row in rows]
            self.row_enabled = [bool(row["enabled"]) for row in rows]
        elif n == 0:
            self.order_values = []
            self.current_alts = []
            self.current_azs = []
            self.current_seps = []
            self.scores = []
            self.hours_above_limit = []
            self.row_enabled = []

        self.layoutChanged.emit()

    def flags(self, index: QModelIndex):  # noqa: N802
        if not index.isValid():
            return Qt.ItemIsEnabled
        return Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsDragEnabled | Qt.ItemIsDropEnabled

    def supportedDropActions(self):  # noqa: N802
        return Qt.MoveAction

    def mimeTypes(self):  # noqa: N802
        return ["application/x-target-row"]

    def mimeData(self, indexes):  # noqa: N802
        mimedata = super().mimeData(indexes)
        rows = sorted({idx.row() for idx in indexes})
        mimedata.setData("application/x-target-row", json.dumps(rows).encode())
        return mimedata

    def dropMimeData(self, mimedata, action, row, column, parent):  # noqa: N802
        if action == Qt.IgnoreAction or not mimedata.hasFormat("application/x-target-row"):
            return False
        self._ensure_cache_lengths()
        rows = json.loads(bytes(mimedata.data("application/x-target-row")).decode())
        rows = sorted(set(int(r) for r in rows))
        if not rows:
            return False

        insert_row = row if row >= 0 else (parent.row() if parent.isValid() else len(self._targets))
        insert_row = min(max(insert_row, 0), len(self._targets))
        for r in rows:
            if r < insert_row:
                insert_row -= 1

        bundles = []
        for r in rows:
            bundles.append(
                (
                    self._targets[r],
                    self.order_values[r] if r < len(self.order_values) else 0,
                    self.current_alts[r] if r < len(self.current_alts) else float("nan"),
                    self.current_azs[r] if r < len(self.current_azs) else float("nan"),
                    self.current_seps[r] if r < len(self.current_seps) else float("nan"),
                    self.scores[r] if r < len(self.scores) else 0.0,
                    self.hours_above_limit[r] if r < len(self.hours_above_limit) else 0.0,
                    self.row_enabled[r] if r < len(self.row_enabled) else True,
                )
            )

        for r in reversed(rows):
            self._targets.pop(r)
            if r < len(self.order_values):
                self.order_values.pop(r)
            if r < len(self.current_alts):
                self.current_alts.pop(r)
            if r < len(self.current_azs):
                self.current_azs.pop(r)
            if r < len(self.current_seps):
                self.current_seps.pop(r)
            if r < len(self.scores):
                self.scores.pop(r)
            if r < len(self.hours_above_limit):
                self.hours_above_limit.pop(r)
            if r < len(self.row_enabled):
                self.row_enabled.pop(r)

        for offset, bundle in enumerate(bundles):
            pos = insert_row + offset
            tgt, order_value, alt, az, sep, score, hours, enabled = bundle
            self._targets.insert(pos, tgt)
            self.order_values.insert(pos, int(order_value))
            self.current_alts.insert(pos, alt)
            self.current_azs.insert(pos, az)
            self.current_seps.insert(pos, sep)
            self.scores.insert(pos, score)
            self.hours_above_limit.insert(pos, hours)
            self.row_enabled.insert(pos, enabled)
        self.layoutChanged.emit()
        return True


__all__ = [
    "_VALID_TABLE_COLOR_MODES",
    "_normalize_table_color_mode",
    "TargetTableGlowDelegate",
    "TargetTableModel",
]
