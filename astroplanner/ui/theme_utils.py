from __future__ import annotations

import base64
import re
import sys
from pathlib import Path
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


_DISPLAY_FONT_PATH = Path(__file__).resolve().parents[2] / "assets" / "fonts" / "Rajdhani-SemiBold.ttf"
_DISPLAY_FONT_LOADED = False
_DISPLAY_FONT_CSS_CACHE: Optional[str] = None
_DISPLAY_WEB_FONT_FAMILY = "Rajdhani Web"
_DISPLAY_FONT_QT_FAMILY: Optional[str] = None
_QT_FALLBACK_FONT_FAMILY: Optional[str] = None
_MPL_AVAILABLE_FONT_NAMES: Optional[set[str]] = None


def _preferred_display_font_family() -> str:
    """Return enforced display font family used across the app and plots."""
    return str(_DISPLAY_FONT_QT_FAMILY or "Rajdhani").strip() or "Rajdhani"


def _split_font_families(value: object) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    families: list[str] = []
    for chunk in text.split(","):
        name = str(chunk or "").strip().strip('"').strip("'").strip()
        if name:
            families.append(name)
    return families


def _platform_ui_font_candidates() -> list[str]:
    if sys.platform == "darwin":
        return [".AppleSystemUIFont", "SF Pro Text", "Avenir Next", "Helvetica Neue", "Arial"]
    if sys.platform.startswith("win"):
        return ["Segoe UI", "Arial"]
    return ["Noto Sans", "DejaVu Sans", "Liberation Sans", "Arial"]


def _resolved_platform_ui_font_family() -> str:
    global _QT_FALLBACK_FONT_FAMILY
    if _QT_FALLBACK_FONT_FAMILY:
        return _QT_FALLBACK_FONT_FAMILY
    for family in _platform_ui_font_candidates():
        if family:
            _QT_FALLBACK_FONT_FAMILY = family
            return family
    if _DISPLAY_FONT_QT_FAMILY:
        _QT_FALLBACK_FONT_FAMILY = _DISPLAY_FONT_QT_FAMILY
        return _QT_FALLBACK_FONT_FAMILY
    _QT_FALLBACK_FONT_FAMILY = "Arial"
    return _QT_FALLBACK_FONT_FAMILY


def _available_mpl_font_names() -> set[str]:
    global _MPL_AVAILABLE_FONT_NAMES
    if _MPL_AVAILABLE_FONT_NAMES is None:
        from matplotlib import font_manager as mpl_font_manager

        _MPL_AVAILABLE_FONT_NAMES = {str(font.name) for font in mpl_font_manager.fontManager.ttflist}
    return _MPL_AVAILABLE_FONT_NAMES


def _ensure_display_font_loaded() -> None:
    global _DISPLAY_FONT_LOADED, _DISPLAY_FONT_QT_FAMILY
    if _DISPLAY_FONT_LOADED:
        return
    if not _DISPLAY_FONT_PATH.exists():
        return
    try:
        from matplotlib import font_manager as mpl_font_manager

        mpl_font_manager.fontManager.addfont(str(_DISPLAY_FONT_PATH))
    except Exception:
        pass
    try:
        from PySide6.QtGui import QFontDatabase

        font_id = QFontDatabase.addApplicationFont(str(_DISPLAY_FONT_PATH))
    except Exception:
        font_id = -1
    if font_id >= 0:
        try:
            from PySide6.QtGui import QFontDatabase

            families = QFontDatabase.applicationFontFamilies(font_id)
        except Exception:
            families = []
        if families:
            _DISPLAY_FONT_QT_FAMILY = str(families[0] or "").strip() or None
            if _DISPLAY_FONT_QT_FAMILY and _MPL_AVAILABLE_FONT_NAMES is not None:
                _MPL_AVAILABLE_FONT_NAMES.add(_DISPLAY_FONT_QT_FAMILY)
        _DISPLAY_FONT_LOADED = True


def _pick_font_family(candidates: list[str]) -> str:
    _ensure_display_font_loaded()
    if _DISPLAY_FONT_QT_FAMILY and _DISPLAY_FONT_QT_FAMILY in candidates:
        return _DISPLAY_FONT_QT_FAMILY
    for family in _platform_ui_font_candidates():
        if family in candidates:
            return family
    for family in candidates:
        if family and family != "Sans Serif":
            return family
    return _resolved_platform_ui_font_family()


def _pick_matplotlib_font_family(candidates: list[str]) -> str:
    available = _available_mpl_font_names()
    for family in candidates:
        if family in available:
            return family
    return "DejaVu Sans"


def _embedded_display_font_css() -> str:
    global _DISPLAY_FONT_CSS_CACHE
    if _DISPLAY_FONT_CSS_CACHE is not None:
        return _DISPLAY_FONT_CSS_CACHE
    if not _DISPLAY_FONT_PATH.exists():
        _DISPLAY_FONT_CSS_CACHE = ""
        return _DISPLAY_FONT_CSS_CACHE
    try:
        encoded = base64.b64encode(_DISPLAY_FONT_PATH.read_bytes()).decode("ascii")
    except Exception:
        _DISPLAY_FONT_CSS_CACHE = ""
        return _DISPLAY_FONT_CSS_CACHE
    _DISPLAY_FONT_CSS_CACHE = (
        "@font-face{"
        f"font-family:'{_DISPLAY_WEB_FONT_FAMILY}';"
        f"src:url(data:font/ttf;base64,{encoded}) format('truetype');"
        "font-style:normal;"
        "font-weight:600;"
        "font-display:swap;"
        "}"
    )
    return _DISPLAY_FONT_CSS_CACHE


def _plot_font_css_stack(theme_tokens: Optional[dict[str, str]] = None) -> str:
    fallback_families: list[str] = [_preferred_display_font_family(), "Rajdhani"]
    if theme_tokens:
        for body_font in _split_font_families(theme_tokens.get("font_family", "")):
            if body_font and body_font not in fallback_families:
                fallback_families.append(body_font)
    fallback_families.extend(_platform_ui_font_candidates())
    fallback_families.extend(["Arial", "Helvetica"])
    deduped: list[str] = []
    for family in fallback_families:
        normalized = str(family or "").strip()
        if normalized and normalized not in deduped and normalized != "Sans Serif":
            deduped.append(normalized)
    fallback = ", ".join(f'"{family}"' for family in deduped) or '"Arial"'
    if _DISPLAY_FONT_PATH.exists():
        return f'"{_DISPLAY_WEB_FONT_FAMILY}", {fallback}'
    return fallback


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


def _normalized_css_color(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    color = QColor(text)
    if not color.isValid():
        return ""
    return color.name().lower()


def _swatch_text_color(value: object, *, dark: str = "#f6fbff", light: str = "#101720") -> str:
    color = _qcolor_from_token(value)
    if not color.isValid():
        return dark
    return dark if color.lightnessF() < 0.60 else light


UI_FONT_SIZE_MIN = 9
UI_FONT_SIZE_MAX = 24


def _sanitize_ui_font_size(value: object, *, default: int = 11) -> int:
    try:
        size = int(value)
    except (TypeError, ValueError):
        size = int(default)
    return max(UI_FONT_SIZE_MIN, min(UI_FONT_SIZE_MAX, size))


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
    elif kind in {"describe", "info"}:
        ellipse(s * 0.20, s * 0.18, s * 0.60, s * 0.64)
        line(s * 0.50, s * 0.42, s * 0.50, s * 0.66)
        dot(s * 0.50, s * 0.31, 1.5)
    elif kind in {"send"}:
        line(m, s * 0.54, s - m, m)
        line(s - m, m, s * 0.62, s - m)
        line(s * 0.62, s - m, s * 0.50, s * 0.56)
        line(s * 0.50, s * 0.56, m, s * 0.54)
    elif kind in {"save"}:
        painter.drawRoundedRect(int(round(m)), int(round(m)), int(round(s - 2 * m)), int(round(s - 2 * m)), 3, 3)
        line(s * 0.32, m + 1, s * 0.32, s * 0.45)
        line(s * 0.32, s * 0.45, s * 0.68, s * 0.45)
        line(s * 0.28, s * 0.67, s * 0.72, s * 0.67)
    elif kind in {"edit"}:
        line(s * 0.28, s * 0.70, s * 0.70, s * 0.28)
        line(s * 0.64, s * 0.22, s * 0.78, s * 0.36)
        line(s * 0.24, s * 0.74, s * 0.36, s * 0.70)
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
    "UI_FONT_SIZE_MAX",
    "UI_FONT_SIZE_MIN",
    "_embedded_display_font_css",
    "_ensure_display_font_loaded",
    "_normalized_css_color",
    "_pick_font_family",
    "_pick_matplotlib_font_family",
    "_platform_ui_font_candidates",
    "_plot_font_css_stack",
    "_preferred_display_font_family",
    "_qcolor_from_token",
    "_resolved_platform_ui_font_family",
    "_sanitize_ui_font_size",
    "_set_button_icon_kind",
    "_set_button_variant",
    "_set_label_tone",
    "_style_dialog_button_box",
    "_swatch_text_color",
    "_theme_color_from_widget",
    "_theme_qcolor_from_widget",
]
