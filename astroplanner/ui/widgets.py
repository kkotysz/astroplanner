from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import Property, QEasingCurve, QPoint, QPropertyAnimation, QRectF, Qt, Signal, QSize
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QAbstractButton, QLabel, QSizePolicy, QStyle, QStyledItemDelegate

from astroplanner.ui.theme_utils import _theme_qcolor_from_widget


class NeonToggleSwitch(QAbstractButton):
    """Compact animated neon switch used for the Altitude/Airmass toggle."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("NeonToggleSwitch")
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self._offset = 0.0
        self._switch_anim = QPropertyAnimation(self, b"offset", self)
        self._switch_anim.setDuration(170)
        self._switch_anim.setEasingCurve(QEasingCurve.OutCubic)
        self.toggled.connect(self._animate_to_checked_state)
        self.setFixedSize(70, 34)

    def sizeHint(self) -> QSize:  # noqa: D401
        return QSize(70, 34)

    def minimumSizeHint(self) -> QSize:  # noqa: D401
        return self.sizeHint()

    def _get_offset(self) -> float:
        return float(self._offset)

    def _set_offset(self, value: float) -> None:
        self._offset = max(0.0, min(1.0, float(value)))
        self.update()

    offset = Property(float, _get_offset, _set_offset)

    def _animate_to_checked_state(self, checked: bool) -> None:
        try:
            self._switch_anim.stop()
        except Exception:
            pass
        self._switch_anim.setStartValue(self._offset)
        self._switch_anim.setEndValue(1.0 if checked else 0.0)
        self._switch_anim.start()

    def setChecked(self, checked: bool) -> None:  # type: ignore[override]
        changed = bool(checked) != self.isChecked()
        super().setChecked(bool(checked))
        if not changed:
            self._set_offset(1.0 if self.isChecked() else 0.0)

    def mouseReleaseEvent(self, event) -> None:  # noqa: D401
        if event.button() == Qt.LeftButton and self.rect().contains(event.position().toPoint()):
            self.toggle()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:  # noqa: D401
        if event.key() in (Qt.Key_Space, Qt.Key_Return, Qt.Key_Enter):
            self.toggle()
            event.accept()
            return
        super().keyPressEvent(event)

    def paintEvent(self, event) -> None:  # noqa: D401
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        track_rect = QRectF(1.0, 3.0, float(self.width() - 2), float(self.height() - 6))
        radius = track_rect.height() / 2.0
        panel_color = _theme_qcolor_from_widget(self, "plot_panel_bg", "#162334")
        primary_color = _theme_qcolor_from_widget(self, "accent_primary", "#18d8f2")
        secondary_color = _theme_qcolor_from_widget(self, "accent_secondary", "#ff4fd8")
        state_color = QColor(secondary_color if self.isChecked() else primary_color)
        border_color = _theme_qcolor_from_widget(self, "panel_border", "#20506a")
        knob_color = _theme_qcolor_from_widget(self, "btn_text", "#f6fbff")
        if self.isChecked():
            off_color = QColor(panel_color)
            mix_ratio = 0.26 + max(0.0, min(1.0, self._offset * 0.72))
        else:
            off_color = QColor(panel_color)
            mix_ratio = 0.74
        groove_mix = QColor(
            round(off_color.red() + (state_color.red() - off_color.red()) * mix_ratio),
            round(off_color.green() + (state_color.green() - off_color.green()) * mix_ratio),
            round(off_color.blue() + (state_color.blue() - off_color.blue()) * mix_ratio),
            round(off_color.alpha() + (state_color.alpha() - off_color.alpha()) * mix_ratio),
        )

        if self.isEnabled():
            glow_color = QColor(state_color)
            glow_color.setAlpha(int(76 + (72 * self._offset if self.isChecked() else 28)))
            painter.setPen(QPen(glow_color, 3.0))
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(track_rect.adjusted(0.5, 0.5, -0.5, -0.5), radius, radius)

        painter.setPen(QPen(border_color, 1.2))
        painter.setBrush(groove_mix)
        painter.drawRoundedRect(track_rect, radius, radius)

        highlight_rect = QRectF(track_rect.left() + 2.0, track_rect.top() + 2.0, track_rect.width() * 0.42, max(2.0, track_rect.height() * 0.28))
        highlight = QColor(state_color)
        highlight.setAlpha(46 if self.isEnabled() else 18)
        painter.setPen(Qt.NoPen)
        painter.setBrush(highlight)
        painter.drawRoundedRect(highlight_rect, highlight_rect.height() / 2.0, highlight_rect.height() / 2.0)

        scan_color = QColor(knob_color)
        scan_color.setAlpha(18 if self.isEnabled() else 8)
        painter.setPen(QPen(scan_color, 1.0))
        for idx in range(3):
            y = track_rect.top() + 5.0 + idx * 5.0
            painter.drawLine(track_rect.left() + 8.0, y, track_rect.right() - 8.0, y)

        handle_d = track_rect.height() - 4.0
        min_x = track_rect.left() + 2.0
        max_x = track_rect.right() - handle_d - 2.0
        handle_x = min_x + (max_x - min_x) * self._offset
        handle_rect = QRectF(handle_x, track_rect.top() + 2.0, handle_d, handle_d)

        handle_glow = QColor(state_color if self.isEnabled() else border_color)
        handle_glow.setAlpha(96 if self.isEnabled() else 28)
        painter.setPen(QPen(handle_glow, 2.0))
        painter.setBrush(knob_color)
        painter.drawEllipse(handle_rect)

        inner_dot = QColor(state_color if self.isChecked() else border_color)
        inner_dot.setAlpha(200 if self.isEnabled() else 90)
        painter.setPen(Qt.NoPen)
        painter.setBrush(inner_dot)
        painter.drawEllipse(handle_rect.center(), 2.1, 2.1)


class CoverImageLabel(QLabel):
    """Render pixmap preserving aspect ratio, centered in available space."""

    resized = Signal(int, int)
    zoomOutLimitReached = Signal()

    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self._source_pixmap = QPixmap()
        self._overlay_painter: Optional[Callable[..., None]] = None
        self._zoom_enabled = False
        self._zoom_factor = 1.0
        self._zoom_min = 1.0
        self._zoom_max = 18.0
        self._pan_offset = QPoint(0, 0)
        self._drag_active = False
        self._drag_start = QPoint(0, 0)
        self._drag_origin = QPoint(0, 0)

    def setPixmap(self, pixmap):  # type: ignore[override]
        if pixmap is None or pixmap.isNull():
            self._source_pixmap = QPixmap()
            self._drag_active = False
            super().setPixmap(QPixmap())
            self.reset_zoom()
            return
        self._source_pixmap = pixmap.copy()
        self._apply_cover_pixmap()

    def setText(self, text: str):  # type: ignore[override]
        self._source_pixmap = QPixmap()
        self._drag_active = False
        super().setText(text)
        self.reset_zoom()

    def resizeEvent(self, event):  # noqa: D401
        super().resizeEvent(event)
        if not self._source_pixmap.isNull():
            self._apply_cover_pixmap()
        self.resized.emit(self.width(), self.height())

    def wheelEvent(self, event):  # noqa: D401
        if not self._zoom_enabled or self._source_pixmap.isNull():
            super().wheelEvent(event)
            return
        delta = event.angleDelta().y()
        if delta == 0:
            return
        if delta < 0 and self._zoom_factor <= (self._zoom_min + 1e-3):
            self.zoomOutLimitReached.emit()
            event.accept()
            return
        step = float(delta) / 120.0
        factor = 1.17 ** step
        self.set_zoom(self._zoom_factor * factor)
        event.accept()

    def mousePressEvent(self, event):  # noqa: D401
        if (
            self._zoom_enabled
            and not self._source_pixmap.isNull()
            and event.button() == Qt.MouseButton.LeftButton
            and self._can_pan_current_view()
        ):
            self._drag_active = True
            self._drag_start = event.position().toPoint()
            self._drag_origin = QPoint(self._pan_offset)
            self._update_cursor()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):  # noqa: D401
        if self._drag_active and self._zoom_enabled:
            delta = event.position().toPoint() - self._drag_start
            self._pan_offset = self._drag_origin + delta
            self._apply_cover_pixmap()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):  # noqa: D401
        if self._drag_active and event.button() == Qt.MouseButton.LeftButton:
            self._drag_active = False
            self._update_cursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):  # noqa: D401
        if self._zoom_enabled and event.button() == Qt.MouseButton.LeftButton:
            self.reset_zoom()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def set_zoom_enabled(self, enabled: bool) -> None:
        self._zoom_enabled = bool(enabled)
        self._update_cursor()

    def set_overlay_painter(self, callback: Optional[Callable[..., None]]) -> None:
        self._overlay_painter = callback
        if not self._source_pixmap.isNull():
            self._apply_cover_pixmap()

    def set_zoom(self, factor: float) -> None:
        try:
            value = float(factor)
        except (TypeError, ValueError):
            value = 1.0
        value = max(self._zoom_min, min(self._zoom_max, value))
        if abs(value - self._zoom_factor) < 1e-4:
            return
        self._zoom_factor = value
        if abs(self._zoom_factor - 1.0) <= 1e-3:
            self._pan_offset = QPoint(0, 0)
        self._apply_cover_pixmap()
        self._update_cursor()

    def zoom_factor(self) -> float:
        return float(self._zoom_factor)

    def reset_zoom(self) -> None:
        self._zoom_factor = 1.0
        self._pan_offset = QPoint(0, 0)
        if not self._source_pixmap.isNull():
            self._apply_cover_pixmap()
        self._update_cursor()

    def _update_cursor(self) -> None:
        if not self._zoom_enabled or self._source_pixmap.isNull() or not self._can_pan_current_view():
            self.setCursor(Qt.CursorShape.ArrowCursor)
            return
        self.setCursor(Qt.CursorShape.ClosedHandCursor if self._drag_active else Qt.CursorShape.OpenHandCursor)

    def _can_pan_current_view(self) -> bool:
        if not self._zoom_enabled or self._source_pixmap.isNull():
            return False
        w = max(1, self.width())
        h = max(1, self.height())
        scale_target_w = max(1, int(round(w * self._zoom_factor)))
        scale_target_h = max(1, int(round(h * self._zoom_factor)))
        scaled = self._source_pixmap.scaled(
            scale_target_w,
            scale_target_h,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        return abs(int(scaled.width()) - int(w)) > 1 or abs(int(scaled.height()) - int(h)) > 1

    def _apply_cover_pixmap(self):
        if self._source_pixmap.isNull():
            super().setPixmap(QPixmap())
            self._update_cursor()
            return
        w = max(1, self.width())
        h = max(1, self.height())
        scale_target_w = max(1, int(round(w * self._zoom_factor)))
        scale_target_h = max(1, int(round(h * self._zoom_factor)))
        scaled = self._source_pixmap.scaled(
            scale_target_w,
            scale_target_h,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        max_dx = max(0, abs(scaled.width() - w) // 2)
        max_dy = max(0, abs(scaled.height() - h) // 2)
        self._pan_offset.setX(max(-max_dx, min(max_dx, self._pan_offset.x())))
        self._pan_offset.setY(max(-max_dy, min(max_dy, self._pan_offset.y())))
        canvas = QPixmap(w, h)
        canvas.fill(Qt.GlobalColor.transparent)
        painter = QPainter(canvas)
        x = ((w - scaled.width()) // 2) + self._pan_offset.x()
        y = ((h - scaled.height()) // 2) + self._pan_offset.y()
        painter.drawPixmap(x, y, scaled)
        if self._overlay_painter is not None:
            try:
                self._overlay_painter(painter, w, h, x, y, scaled.width(), scaled.height())
            except TypeError:
                try:
                    self._overlay_painter(painter, w, h)
                except Exception:
                    pass
            except Exception:
                pass
        painter.end()
        super().setPixmap(canvas)
        self._update_cursor()


class NoSelectBackgroundDelegate(QStyledItemDelegate):
    """Delegate that preserves model background for the target-name column when selected."""

    def paint(self, painter, option, index):
        if option.state & QStyle.State_Selected:
            option.state &= ~QStyle.State_Selected
        super().paint(painter, option, index)


__all__ = [
    "CoverImageLabel",
    "NeonToggleSwitch",
    "NoSelectBackgroundDelegate",
]
