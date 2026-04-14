from __future__ import annotations

import re
from typing import Optional

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QDialogButtonBox, QLabel, QWidget


def _set_dynamic_property(widget: Optional[QWidget], name: str, value: object) -> None:
    if widget is None:
        return
    widget.setProperty(name, value)


def _set_button_variant(button: Optional[QWidget], variant: str) -> None:
    _set_dynamic_property(button, "variant", str(variant))


def _set_label_tone(label: Optional[QWidget], tone: str) -> None:
    _set_dynamic_property(label, "tone", str(tone))


def _theme_tokens_from_widget(widget: object) -> dict[str, str]:
    current = widget
    for _ in range(6):
        tokens = getattr(current, "_theme_tokens", None)
        if isinstance(tokens, dict):
            return tokens
        parent = getattr(current, "parent", None)
        if not callable(parent):
            break
        current = parent()
        if current is None:
            break
    return {}


def _theme_color_from_widget(widget: object, key: str, fallback: str) -> str:
    tokens = _theme_tokens_from_widget(widget)
    value = tokens.get(key) if isinstance(tokens, dict) else None
    return str(value or fallback)


_CSS_RGBA_RE = re.compile(
    r"^rgba?\(\s*([0-9]{1,3})\s*,\s*([0-9]{1,3})\s*,\s*([0-9]{1,3})(?:\s*,\s*([0-9]*\.?[0-9]+))?\s*\)$",
    re.IGNORECASE,
)


def _parse_qcolor_token(value: object) -> QColor:
    if isinstance(value, QColor):
        color = QColor(value)
        return color if color.isValid() else QColor()

    text = str(value or "").strip()
    if not text:
        return QColor()

    color = QColor(text)
    if color.isValid():
        return color

    match = _CSS_RGBA_RE.match(text)
    if not match:
        return QColor()

    red, green, blue = (max(0, min(255, int(part))) for part in match.groups()[:3])
    alpha_token = match.group(4)
    if alpha_token is None or alpha_token == "":
        alpha = 255
    else:
        alpha_value = float(alpha_token)
        if alpha_value <= 1.0:
            alpha = int(round(max(0.0, min(1.0, alpha_value)) * 255.0))
        else:
            alpha = int(round(max(0.0, min(255.0, alpha_value))))
    return QColor(red, green, blue, alpha)


def _qcolor_from_token(value: object, fallback: object | None = None) -> QColor:
    color = _parse_qcolor_token(value)
    if color.isValid():
        return color
    if fallback is not None:
        fallback_color = _parse_qcolor_token(fallback)
        if fallback_color.isValid():
            return fallback_color
    return QColor()


def _theme_qcolor_from_widget(widget: object, key: str, fallback: str) -> QColor:
    color = _qcolor_from_token(_theme_color_from_widget(widget, key, fallback), fallback)
    if color.isValid():
        return color
    return _qcolor_from_token(fallback)


def _icon_pen(color: QColor, width: float) -> QPen:
    pen = QPen(color)
    pen.setWidthF(width)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    return pen


def _button_icon_palette(widget: object) -> tuple[QColor, QColor, QColor]:
    variant = ""
    if hasattr(widget, "property"):
        try:
            variant = str(widget.property("variant") or "")
        except Exception:
            variant = ""
    if variant == "primary":
        fg = _theme_qcolor_from_widget(widget, "btn_text", "#f7fbff")
    elif variant == "ghost":
        fg = _theme_qcolor_from_widget(widget, "strip_label", "#a4b4c8")
    else:
        fg = _theme_qcolor_from_widget(widget, "section_title", "#eef4fc")
    accent = _theme_qcolor_from_widget(widget, "accent_primary", "#59f3ff")
    accent_secondary = _theme_qcolor_from_widget(widget, "accent_secondary", "#ff4da6")
    return fg, accent, accent_secondary


def _build_button_icon_pixmap(kind: str, fg: QColor, accent: QColor, accent_secondary: QColor, size: int = 18) -> QPixmap:
    px = QPixmap(size, size)
    px.fill(Qt.GlobalColor.transparent)
    painter = QPainter(px)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    stroke = max(1.5, float(size) * 0.11)
    painter.setPen(_icon_pen(fg, stroke))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    kind = str(kind or "").strip().lower()
    s = float(size)
    m = s * 0.2

    def line(x1: float, y1: float, x2: float, y2: float) -> None:
        painter.drawLine(int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))

    def ellipse(x: float, y: float, w: float, h: float) -> None:
        painter.drawEllipse(int(round(x)), int(round(y)), int(round(w)), int(round(h)))

    def arc(x: float, y: float, w: float, h: float, start_deg: float, span_deg: float) -> None:
        painter.drawArc(
            int(round(x)),
            int(round(y)),
            int(round(w)),
            int(round(h)),
            int(round(start_deg * 16)),
            int(round(span_deg * 16)),
        )

    def dot(x: float, y: float, r: float = 1.7) -> None:
        painter.save()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(accent_secondary)
        painter.drawEllipse(int(round(x - r)), int(round(y - r)), int(round(r * 2)), int(round(r * 2)))
        painter.restore()

    if kind in {"add", "new"}:
        line(s * 0.5, m, s * 0.5, s - m)
        line(m, s * 0.5, s - m, s * 0.5)
        dot(s * 0.77, s * 0.23)
    elif kind in {"toggle", "observed", "ok", "apply"}:
        line(s * 0.23, s * 0.56, s * 0.42, s * 0.74)
        line(s * 0.42, s * 0.74, s * 0.78, s * 0.30)
        dot(s * 0.76, s * 0.24)
    elif kind in {"cancel", "clear"}:
        line(m, m, s - m, s - m)
        line(s - m, m, m, s - m)
    elif kind in {"remove", "delete"}:
        line(m, s * 0.5, s - m, s * 0.5)
    elif kind in {"lookup", "resolve"}:
        ellipse(s * 0.22, s * 0.22, s * 0.36, s * 0.36)
        line(s * 0.52, s * 0.52, s * 0.74, s * 0.74)
        line(s * 0.40, s * 0.28, s * 0.40, s * 0.52)
        line(s * 0.28, s * 0.40, s * 0.52, s * 0.40)
    elif kind in {"refresh"}:
        painter.drawArc(int(round(m)), int(round(m)), int(round(s - 2 * m)), int(round(s - 2 * m)), int(25 * 16), int(280 * 16))
        line(s * 0.72, s * 0.18, s * 0.82, s * 0.18)
        line(s * 0.82, s * 0.18, s * 0.82, s * 0.30)
    elif kind in {"open", "load"}:
        line(m, s * 0.72, s - m, s * 0.72)
        line(m, s * 0.72, m, s * 0.58)
        line(s - m, s * 0.72, s - m, s * 0.58)
        line(s * 0.5, m, s * 0.5, s * 0.58)
        line(s * 0.5, s * 0.58, s * 0.34, s * 0.42)
        line(s * 0.5, s * 0.58, s * 0.66, s * 0.42)
    elif kind in {"link", "open-link"}:
        line(s * 0.28, s * 0.72, s * 0.72, s * 0.28)
        line(s * 0.50, s * 0.28, s * 0.72, s * 0.28)
        line(s * 0.72, s * 0.28, s * 0.72, s * 0.50)
        line(s * 0.30, s * 0.34, s * 0.30, s * 0.70)
        line(s * 0.30, s * 0.70, s * 0.66, s * 0.70)
    elif kind in {"notes", "note"}:
        painter.drawRoundedRect(int(round(s * 0.24)), int(round(s * 0.18)), int(round(s * 0.50)), int(round(s * 0.62)), 3, 3)
        line(s * 0.34, s * 0.36, s * 0.64, s * 0.36)
        line(s * 0.34, s * 0.50, s * 0.64, s * 0.50)
        line(s * 0.34, s * 0.64, s * 0.56, s * 0.64)
    elif kind in {"window", "best_window"}:
        line(s * 0.24, s * 0.28, s * 0.24, s * 0.72)
        line(s * 0.76, s * 0.28, s * 0.76, s * 0.72)
        line(s * 0.24, s * 0.28, s * 0.76, s * 0.28)
        line(s * 0.24, s * 0.72, s * 0.76, s * 0.72)
        line(s * 0.50, s * 0.34, s * 0.50, s * 0.50)
        line(s * 0.50, s * 0.50, s * 0.62, s * 0.58)
    else:
        ellipse(m, m, s - 2 * m, s - 2 * m)
        dot(s * 0.72, s * 0.28)

    painter.end()
    return px


def _build_button_icon(button: object, kind: str, size: int = 18) -> QIcon:
    fg, accent, accent_secondary = _button_icon_palette(button)
    disabled = _theme_qcolor_from_widget(button, "state_disabled", "#7f8ca3")
    icon = QIcon()
    icon.addPixmap(_build_button_icon_pixmap(kind, fg, accent, accent_secondary, size), QIcon.Mode.Normal, QIcon.State.Off)
    icon.addPixmap(_build_button_icon_pixmap(kind, fg, accent_secondary, accent, size), QIcon.Mode.Active, QIcon.State.Off)
    icon.addPixmap(_build_button_icon_pixmap(kind, disabled, disabled, disabled, size), QIcon.Mode.Disabled, QIcon.State.Off)
    return icon


def _set_button_icon_kind(button: Optional[QWidget], kind: str, size: int = 16) -> None:
    if button is None or not hasattr(button, "setIcon"):
        return
    button.setProperty("icon_kind", str(kind))
    button.setProperty("icon_size", int(size))
    button.setIcon(_build_button_icon(button, kind, size))
    if hasattr(button, "setIconSize"):
        button.setIconSize(QSize(int(size), int(size)))


def _style_dialog_button_box(
    button_box: Optional[QDialogButtonBox],
    *,
    ok: str = "secondary",
    cancel: str = "ghost",
    apply: str = "secondary",
    close: str = "secondary",
) -> None:
    if button_box is None:
        return
    mapping = {
        QDialogButtonBox.Ok: ok,
        QDialogButtonBox.Cancel: cancel,
        QDialogButtonBox.Apply: apply,
        QDialogButtonBox.Close: close,
    }
    icon_mapping = {
        QDialogButtonBox.Ok: "ok",
        QDialogButtonBox.Cancel: "cancel",
        QDialogButtonBox.Apply: "apply",
        QDialogButtonBox.Close: "cancel",
    }
    for role, variant in mapping.items():
        btn = button_box.button(role)
        if btn is not None:
            _set_button_variant(btn, variant)
            _set_button_icon_kind(btn, icon_mapping.get(role, "window"), 14)


__all__ = [
    "_qcolor_from_token",
    "_set_button_icon_kind",
    "_set_button_variant",
    "_set_label_tone",
    "_style_dialog_button_box",
    "_theme_color_from_widget",
    "_theme_qcolor_from_widget",
]
