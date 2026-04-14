from __future__ import annotations

import math
import sys
from typing import Optional

from PySide6.QtCore import QEvent, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QLinearGradient, QPainter
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QHeaderView,
    QSizePolicy,
    QTableView,
    QWidget,
)

from .theme_utils import _theme_qcolor_from_widget


def _fit_dialog_to_screen(
    dialog: QDialog,
    *,
    preferred_width: int,
    preferred_height: int,
    min_width: int,
    min_height: int,
) -> None:
    screen = dialog.screen()
    if screen is None and isinstance(dialog.parentWidget(), QWidget):
        screen = dialog.parentWidget().screen()
    if screen is None:
        screen = QApplication.primaryScreen()

    hint = dialog.sizeHint()
    target_w = max(int(min_width), int(preferred_width), int(hint.width()))
    target_h = max(int(min_height), int(preferred_height), int(hint.height()))

    if screen is not None:
        available = screen.availableGeometry()
        max_w = max(420, available.width() - 24)
        max_h = max(320, available.height() - 24)
        target_w = min(max_w, target_w)
        target_h = min(max_h, target_h)
        min_width = min(max_w, max(int(min_width), min(int(hint.width()), target_w)))
        min_height = min(max_h, max(int(min_height), min(int(hint.height()), target_h)))

    dialog.setMinimumSize(int(min_width), int(min_height))
    dialog.resize(int(target_w), int(target_h))


def distribute_extra_table_width(
    widths: dict[int, int],
    *,
    available_width: int,
    stretch_weights: dict[int, float],
) -> dict[int, int]:
    normalized = {
        int(col): max(0, int(width))
        for col, width in widths.items()
        if int(width) > 0
    }
    if not normalized:
        return {}
    extra = int(available_width) - sum(normalized.values())
    if extra <= 0:
        return dict(normalized)

    candidates = [
        (int(col), float(stretch_weights.get(col, 0.0)))
        for col in normalized
        if float(stretch_weights.get(col, 0.0)) > 0.0
    ]
    if not candidates:
        return dict(normalized)

    total_weight = sum(weight for _col, weight in candidates)
    if total_weight <= 0.0:
        return dict(normalized)

    distributed = dict(normalized)
    allocations: list[tuple[float, int]] = []
    consumed = 0
    for col, weight in candidates:
        raw_share = (float(extra) * float(weight)) / total_weight
        share = int(math.floor(raw_share))
        distributed[col] += share
        consumed += share
        allocations.append((raw_share - share, col))

    remainder = max(0, int(extra - consumed))
    allocations.sort(key=lambda item: (-item[0], item[1]))
    for _fraction, col in allocations[:remainder]:
        distributed[col] += 1
    return distributed


_distribute_extra_table_width = distribute_extra_table_width


class TargetTableView(QTableView):
    """
    Emit `deleteRequested` when the user presses an erase key while the
    targets table is focused.
    """

    deleteRequested = Signal()
    layoutWidthChanged = Signal()
    hoverRowChanged = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._forced_row_height: Optional[int] = None
        self._hover_row: Optional[int] = None
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)

    def _set_hover_row(self, row: Optional[int]) -> None:
        new_row = row if row is not None and row >= 0 else None
        if new_row == self._hover_row:
            return
        self._hover_row = new_row
        self.hoverRowChanged.emit(new_row)

    def set_forced_row_height(self, height: int) -> None:
        value = max(10, int(height))
        self._forced_row_height = value
        header = self.verticalHeader()
        header.setSectionResizeMode(QHeaderView.Fixed)
        header.setDefaultSectionSize(value)
        header.setMinimumSectionSize(value)
        header.setMaximumSectionSize(value)
        for row in range(self.model().rowCount() if self.model() is not None else 0):
            self.setRowHeight(row, value)
        self.updateGeometries()
        self.viewport().update()

    def setModel(self, model) -> None:  # type: ignore[override]
        super().setModel(model)
        if self._forced_row_height is not None:
            self.set_forced_row_height(self._forced_row_height)

    def sizeHintForRow(self, row: int) -> int:  # noqa: D401
        if self._forced_row_height is not None:
            return int(self._forced_row_height)
        return super().sizeHintForRow(row)

    def resizeEvent(self, event):  # noqa: D401
        super().resizeEvent(event)
        self.layoutWidthChanged.emit()

    def mouseMoveEvent(self, event):  # noqa: D401
        pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
        index = self.indexAt(pos)
        self._set_hover_row(index.row() if index.isValid() else None)
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):  # noqa: D401
        self._set_hover_row(None)
        super().leaveEvent(event)

    def viewportEvent(self, event):  # noqa: D401
        if event.type() in (QEvent.Type.Leave, QEvent.Type.HoverLeave):
            self._set_hover_row(None)
        return super().viewportEvent(event)

    def keyPressEvent(self, event):  # noqa: D401
        key = event.key()
        if key in (Qt.Key_Backspace, Qt.Key_Delete):
            mods = event.modifiers() & (
                Qt.ControlModifier | Qt.MetaModifier | Qt.AltModifier | Qt.ShiftModifier
            )
            if mods in (
                Qt.NoModifier,
                Qt.ControlModifier,
                Qt.MetaModifier if sys.platform == "darwin" else Qt.NoModifier,
            ):
                self.deleteRequested.emit()
                event.accept()
                return
        super().keyPressEvent(event)


class SkeletonShimmerWidget(QWidget):
    """Animated skeleton placeholder used while charts or images are loading."""

    def __init__(self, variant: str = "plot", parent=None):
        super().__init__(parent)
        self._variant = str(variant or "plot").strip().lower()
        self._phase = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        try:
            self._timer.setTimerType(Qt.PreciseTimer)
        except Exception:
            pass
        self._timer.timeout.connect(self._advance)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._timer.start()

    def showEvent(self, event) -> None:  # noqa: D401
        if not self._timer.isActive():
            self._timer.start()
        super().showEvent(event)

    def hideEvent(self, event) -> None:  # noqa: D401
        self._timer.stop()
        super().hideEvent(event)

    def _advance(self) -> None:
        self._phase = (self._phase + 0.028) % 1.35
        self.update()

    @staticmethod
    def _mix_colors(first: QColor, second: QColor, ratio: float) -> QColor:
        ratio = max(0.0, min(1.0, float(ratio)))
        other = 1.0 - ratio
        return QColor(
            int(round(first.red() * ratio + second.red() * other)),
            int(round(first.green() * ratio + second.green() * other)),
            int(round(first.blue() * ratio + second.blue() * other)),
            int(round(first.alpha() * ratio + second.alpha() * other)),
        )

    def _spec(self) -> list[tuple[float, float, float, float, float]]:
        if self._variant == "image":
            return [
                (0.00, 0.00, 1.00, 1.00, 16.0),
                (0.05, 0.07, 0.34, 0.06, 8.0),
                (0.05, 0.16, 0.52, 0.04, 7.0),
                (0.05, 0.23, 0.42, 0.04, 7.0),
                (0.73, 0.09, 0.17, 0.17, 999.0),
            ]
        if self._variant == "inline":
            return [
                (0.00, 0.16, 0.26, 0.56, 999.0),
                (0.30, 0.16, 0.18, 0.56, 999.0),
                (0.52, 0.16, 0.14, 0.56, 999.0),
            ]
        if self._variant == "table":
            return [
                (0.00, 0.00, 1.00, 0.12, 12.0),
                (0.03, 0.03, 0.18, 0.04, 8.0),
                (0.24, 0.03, 0.10, 0.04, 8.0),
                (0.38, 0.03, 0.08, 0.04, 8.0),
                (0.50, 0.03, 0.10, 0.04, 8.0),
                (0.65, 0.03, 0.16, 0.04, 8.0),
                (0.85, 0.03, 0.10, 0.04, 8.0),
                (0.02, 0.18, 0.96, 0.10, 10.0),
                (0.03, 0.205, 0.28, 0.04, 8.0),
                (0.37, 0.205, 0.10, 0.04, 8.0),
                (0.52, 0.205, 0.16, 0.04, 8.0),
                (0.76, 0.205, 0.16, 0.04, 8.0),
                (0.02, 0.32, 0.96, 0.10, 10.0),
                (0.03, 0.345, 0.34, 0.04, 8.0),
                (0.43, 0.345, 0.12, 0.04, 8.0),
                (0.60, 0.345, 0.11, 0.04, 8.0),
                (0.77, 0.345, 0.15, 0.04, 8.0),
                (0.02, 0.46, 0.96, 0.10, 10.0),
                (0.03, 0.485, 0.25, 0.04, 8.0),
                (0.35, 0.485, 0.14, 0.04, 8.0),
                (0.55, 0.485, 0.18, 0.04, 8.0),
                (0.79, 0.485, 0.13, 0.04, 8.0),
                (0.02, 0.60, 0.96, 0.10, 10.0),
                (0.03, 0.625, 0.31, 0.04, 8.0),
                (0.40, 0.625, 0.10, 0.04, 8.0),
                (0.57, 0.625, 0.15, 0.04, 8.0),
                (0.79, 0.625, 0.14, 0.04, 8.0),
                (0.02, 0.74, 0.96, 0.10, 10.0),
                (0.03, 0.765, 0.22, 0.04, 8.0),
                (0.31, 0.765, 0.18, 0.04, 8.0),
                (0.56, 0.765, 0.13, 0.04, 8.0),
                (0.75, 0.765, 0.17, 0.04, 8.0),
            ]
        return [
            (0.00, 0.00, 0.60, 0.08, 10.0),
            (0.00, 0.13, 0.82, 0.05, 8.0),
            (0.00, 0.22, 0.74, 0.05, 8.0),
            (0.00, 0.34, 1.00, 0.50, 16.0),
            (0.05, 0.40, 0.88, 0.04, 7.0),
            (0.05, 0.50, 0.76, 0.04, 7.0),
            (0.05, 0.60, 0.82, 0.04, 7.0),
            (0.05, 0.70, 0.64, 0.04, 7.0),
            (0.00, 0.89, 0.28, 0.06, 999.0),
            (0.32, 0.89, 0.22, 0.06, 999.0),
        ]

    def paintEvent(self, event) -> None:  # noqa: D401
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = QRectF(self.rect()).adjusted(1.0, 1.0, -1.0, -1.0)
        if rect.width() <= 2 or rect.height() <= 2:
            return

        loading_bg = _theme_qcolor_from_widget(self, "loading_bg", "#162334")
        loading_border = _theme_qcolor_from_widget(self, "loading_border", "#20506a")
        loading_shimmer = _theme_qcolor_from_widget(self, "loading_shimmer", "#4fd8ff")
        accent_primary = _theme_qcolor_from_widget(self, "accent_primary", "#59f3ff")
        base_fill = self._mix_colors(loading_bg, loading_border, 0.76)
        block_fill = self._mix_colors(loading_bg, loading_border, 0.66)
        highlight = self._mix_colors(loading_shimmer, accent_primary, 0.60)

        painter.setPen(Qt.NoPen)
        for x_ratio, y_ratio, w_ratio, h_ratio, radius in self._spec():
            block = QRectF(
                rect.left() + rect.width() * x_ratio,
                rect.top() + rect.height() * y_ratio,
                rect.width() * w_ratio,
                rect.height() * h_ratio,
            )
            if block.width() <= 1 or block.height() <= 1:
                continue
            rad = min(float(radius), block.width() / 2.0, block.height() / 2.0)
            painter.setBrush(block_fill)
            painter.drawRoundedRect(block, rad, rad)

            shimmer = QLinearGradient(
                block.left() + (block.width() * (self._phase - 0.48)),
                block.top(),
                block.left() + (block.width() * (self._phase + 0.22)),
                block.bottom(),
            )
            left = QColor(base_fill)
            left.setAlpha(0)
            mid = QColor(highlight)
            mid.setAlpha(110 if self._variant == "plot" else 92)
            right = QColor(base_fill)
            right.setAlpha(0)
            shimmer.setColorAt(0.0, left)
            shimmer.setColorAt(0.50, mid)
            shimmer.setColorAt(1.0, right)
            painter.setBrush(shimmer)
            painter.drawRoundedRect(block, rad, rad)
        painter.end()


__all__ = [
    "SkeletonShimmerWidget",
    "TargetTableView",
    "_distribute_extra_table_width",
    "_fit_dialog_to_screen",
    "distribute_extra_table_width",
]
