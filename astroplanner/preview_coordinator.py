from __future__ import annotations

import logging
import math
from time import perf_counter
from typing import Optional

import shiboken6 as shb
from PySide6.QtCore import QBuffer, QByteArray, QIODevice, Qt, QUrl, QUrlQuery, Slot
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtNetwork import QNetworkReply, QNetworkRequest
from PySide6.QtWidgets import QLabel

from astroplanner.models import Site, Target
from astroplanner.ui.add_target import FinderChartWorker


logger = logging.getLogger(__name__)


def _safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


CUTOUT_SURVEY_CHOICES: list[tuple[str, str, str]] = [
    ("dss2", "DSS2", "CDS/P/DSS2/color"),
    ("panstarrs", "PanSTARRS", "CDS/P/PanSTARRS/DR1/color-z-zg-g"),
    ("2mass", "2MASS", "CDS/P/2MASS/color"),
]
CUTOUT_VIEW_CHOICES: list[tuple[str, str]] = [
    ("aladin", "Aladin"),
    ("finderchart", "Finder chart"),
]
CUTOUT_DEFAULT_SURVEY_KEY = "dss2"
CUTOUT_DEFAULT_VIEW_KEY = "aladin"
CUTOUT_DEFAULT_FOV_ARCMIN = 15
CUTOUT_DEFAULT_SIZE_PX = 280
CUTOUT_MIN_FOV_ARCMIN = 5
CUTOUT_MAX_FOV_ARCMIN = 120
CUTOUT_MIN_SIZE_PX = 128
CUTOUT_MAX_SIZE_PX = 800
CUTOUT_CACHE_MAX = 24
# Fetch a slightly wider field for Aladin so zoom-out/pan has context around target.
CUTOUT_ALADIN_FETCH_MARGIN = 1.28
# Keep at least ~2 deg on the shorter axis for panning context without
# making telescope FOV overlays too small at max zoom.
CUTOUT_ALADIN_FETCH_MIN_ARCMIN = 120.0
# Prefer dynamic fetch size based on telescope FOV when available.
CUTOUT_ALADIN_FETCH_TELESCOPE_MARGIN = 5.0
CUTOUT_ALADIN_FETCH_TELESCOPE_MAX_ARCMIN = 480.0
# How much to widen fetched Aladin context when user zooms out past 1x.
CUTOUT_ALADIN_CONTEXT_STEP = 1.6
# Slight initial zoom so panning works immediately after load.
CUTOUT_ALADIN_INITIAL_PAN_ZOOM = 1.50
# Request higher native cutout resolution so zoom quality stays acceptable.
CUTOUT_ALADIN_FETCH_RES_MULT = 2.2
CUTOUT_ALADIN_FETCH_MIN_SHORT_PX = 900
CUTOUT_ALADIN_FETCH_MAX_EDGE_PX = 1440
FINDER_WORKER_TIMEOUT_MS = 10000
FINDER_RETRY_COOLDOWN_S = 45.0

def _normalize_cutout_survey_key(value: object) -> str:
    key = str(value or "").strip().lower()
    if key in {"decals", "cds/p/decals/dr5/color"}:
        # Legacy migration: DECaLS was removed from options.
        return "2mass"
    for survey_key, _, hips in CUTOUT_SURVEY_CHOICES:
        if key == survey_key or key == hips.lower():
            return survey_key
    return CUTOUT_DEFAULT_SURVEY_KEY


def _normalize_cutout_view_key(value: object) -> str:
    key = str(value or "").strip().lower()
    for view_key, _ in CUTOUT_VIEW_CHOICES:
        if key == view_key:
            return view_key
    return CUTOUT_DEFAULT_VIEW_KEY


def _cutout_survey_label(key: object) -> str:
    norm = _normalize_cutout_survey_key(key)
    for survey_key, label, _ in CUTOUT_SURVEY_CHOICES:
        if survey_key == norm:
            return label
    return "DSS2"


def _cutout_survey_hips(key: object) -> str:
    norm = _normalize_cutout_survey_key(key)
    for survey_key, _, hips in CUTOUT_SURVEY_CHOICES:
        if survey_key == norm:
            return hips
    return CUTOUT_SURVEY_CHOICES[0][2]


def _sanitize_cutout_fov_arcmin(value: object) -> int:
    try:
        ivalue = int(float(value))
    except Exception:
        ivalue = CUTOUT_DEFAULT_FOV_ARCMIN
    return max(CUTOUT_MIN_FOV_ARCMIN, min(CUTOUT_MAX_FOV_ARCMIN, ivalue))


def _sanitize_cutout_size_px(value: object) -> int:
    try:
        ivalue = int(float(value))
    except Exception:
        ivalue = CUTOUT_DEFAULT_SIZE_PX
    return max(CUTOUT_MIN_SIZE_PX, min(CUTOUT_MAX_SIZE_PX, ivalue))


class PreviewCoordinator:
    """Own Aladin/finder preview rendering, cache, and worker lifecycle."""

    def __init__(self, planner: object) -> None:
        object.__setattr__(self, "_planner", planner)

    def __getattr__(self, name: str):
        return getattr(self._planner, name)

    def __setattr__(self, name: str, value: object) -> None:
        if name == "_planner":
            object.__setattr__(self, name, value)
            return
        setattr(self._planner, name, value)

    def _cutout_render_dimensions_px(self, label: Optional[QLabel] = None) -> tuple[int, int]:
        base = _sanitize_cutout_size_px(getattr(self, "_cutout_size_px", CUTOUT_DEFAULT_SIZE_PX))
        probe = label if label is not None else getattr(self, "aladin_image_label", None)
        w = int(getattr(probe, "width", lambda: 0)()) if probe is not None else 0
        h = int(getattr(probe, "height", lambda: 0)()) if probe is not None else 0
        if (w < 32 or h < 32) and hasattr(self, "cutout_tabs"):
            tw = int(self.cutout_tabs.width())
            th = int(self.cutout_tabs.height() - self.cutout_tabs.tabBar().height())
            if tw > 0 and th > 0:
                w = max(w, tw)
                h = max(h, th)
        if w < 32 or h < 32:
            return base, base

        ratio = max(0.4, min(2.5, float(w) / float(h)))
        if ratio >= 1.0:
            out_h = base
            out_w = int(round(base * ratio))
        else:
            out_w = base
            out_h = int(round(base / ratio))

        max_side = 1400
        largest = max(out_w, out_h)
        if largest > max_side:
            scale = max_side / float(largest)
            out_w = int(round(out_w * scale))
            out_h = int(round(out_h * scale))
        step = 8
        out_w = max(128, int(round(out_w / step) * step))
        out_h = max(128, int(round(out_h / step) * step))
        return out_w, out_h

    def _cutout_fov_axes_arcmin(self, width_px: int, height_px: int) -> tuple[float, float]:
        base = max(1.0, float(self._cutout_fov_arcmin))
        if width_px <= 0 or height_px <= 0:
            return base, base
        ratio = float(width_px) / float(height_px)
        if ratio >= 1.0:
            fov_y = base
            fov_x = base * ratio
        else:
            fov_x = base
            fov_y = base / ratio
        return fov_x, fov_y

    def _aladin_fetch_fov_axes_arcmin(self, width_px: int, height_px: int) -> tuple[float, float]:
        fov_x, fov_y = self._cutout_fov_axes_arcmin(width_px, height_px)
        margin = max(1.0, float(CUTOUT_ALADIN_FETCH_MARGIN))
        fetch_x = float(fov_x) * margin
        fetch_y = float(fov_y) * margin
        min_short_axis = self._aladin_fetch_min_short_axis_arcmin()
        if width_px > 0 and height_px > 0:
            ratio = max(0.25, min(4.0, float(width_px) / float(height_px)))
            if ratio >= 1.0:
                # Height is the shorter axis in landscape.
                fetch_y = max(fetch_y, min_short_axis)
                fetch_x = max(fetch_x, fetch_y * ratio)
            else:
                # Width is the shorter axis in portrait.
                fetch_x = max(fetch_x, min_short_axis)
                fetch_y = max(fetch_y, fetch_x / ratio)
        else:
            fetch_x = max(fetch_x, min_short_axis)
            fetch_y = max(fetch_y, min_short_axis)
        context_factor = max(1.0, float(getattr(self, "_aladin_context_factor", 1.0)))
        if context_factor > 1.0:
            fetch_x *= context_factor
            fetch_y *= context_factor
        return fetch_x, fetch_y

    def _aladin_fetch_min_short_axis_arcmin(self) -> float:
        fallback = max(1.0, float(CUTOUT_ALADIN_FETCH_MIN_ARCMIN))
        tel_fov = self._site_telescope_fov_arcmin()
        if tel_fov is None:
            return fallback
        tel_short = min(float(tel_fov[0]), float(tel_fov[1]))
        if not math.isfinite(tel_short) or tel_short <= 0.0:
            return fallback
        dynamic = tel_short * max(1.0, float(CUTOUT_ALADIN_FETCH_TELESCOPE_MARGIN))
        dynamic = max(float(CUTOUT_MIN_FOV_ARCMIN), dynamic)
        dynamic = min(float(CUTOUT_ALADIN_FETCH_TELESCOPE_MAX_ARCMIN), dynamic)
        return dynamic

    def _aladin_fetch_dimensions_px(self, width_px: int, height_px: int) -> tuple[int, int]:
        w = max(1, int(width_px))
        h = max(1, int(height_px))
        short_axis = max(1, min(w, h))
        min_short_px = max(256, int(CUTOUT_ALADIN_FETCH_MIN_SHORT_PX))
        scale = max(
            1.0,
            float(CUTOUT_ALADIN_FETCH_RES_MULT),
            float(min_short_px) / float(short_axis),
        )
        out_w = max(64, int(round(float(w) * scale)))
        out_h = max(64, int(round(float(h) * scale)))
        max_edge = max(min_short_px, int(CUTOUT_ALADIN_FETCH_MAX_EDGE_PX))
        largest = max(out_w, out_h)
        if largest > max_edge:
            down = float(max_edge) / float(largest)
            out_w = max(64, int(round(float(out_w) * down)))
            out_h = max(64, int(round(float(out_h) * down)))
        step = 8
        out_w = max(64, int(round(out_w / step) * step))
        out_h = max(64, int(round(out_h / step) * step))
        return out_w, out_h

    def _site_telescope_fov_arcmin(self, site: Optional[Site] = None) -> Optional[tuple[float, float]]:
        site_obj = site
        if site_obj is None:
            if hasattr(self, "obs_combo") and hasattr(self, "observatories"):
                site_obj = self.observatories.get(self.obs_combo.currentText())
            if site_obj is None and hasattr(self, "table_model"):
                site_obj = self.table_model.site
        if site_obj is None:
            return None
        fov = site_obj.fov_arcmin
        if fov is None:
            return None
        fov_x = _safe_float(fov[0])
        fov_y = _safe_float(fov[1])
        if fov_x is None or fov_y is None:
            return None
        if not math.isfinite(fov_x) or not math.isfinite(fov_y) or fov_x <= 0.0 or fov_y <= 0.0:
            return None
        return float(fov_x), float(fov_y)

    def _telescope_overlay_signature(self, site: Optional[Site] = None) -> str:
        fov = self._site_telescope_fov_arcmin(site)
        if fov is None:
            return "none"
        return f"{fov[0]:.3f}x{fov[1]:.3f}"

    def _fit_cutout_base_fov_to_telescope(self, tel_fov_x: float, tel_fov_y: float, width_px: int, height_px: int) -> int:
        ratio = 1.0
        if width_px > 0 and height_px > 0:
            ratio = max(0.25, min(4.0, float(width_px) / float(height_px)))
        if ratio >= 1.0:
            required_base = max(float(tel_fov_y), float(tel_fov_x) / ratio)
        else:
            required_base = max(float(tel_fov_x), float(tel_fov_y) * ratio)
        return _sanitize_cutout_fov_arcmin(int(round(required_base)))

    def _sync_cutout_fov_to_site(self, site: Optional[Site] = None, persist: bool = False) -> bool:
        tel_fov = self._site_telescope_fov_arcmin(site)
        if tel_fov is None:
            return False
        render_w, render_h = self._cutout_render_dimensions_px(getattr(self, "aladin_image_label", None))
        fitted = self._fit_cutout_base_fov_to_telescope(tel_fov[0], tel_fov[1], render_w, render_h)
        if fitted == int(self._cutout_fov_arcmin):
            return False
        self._cutout_fov_arcmin = int(fitted)
        if persist:
            self.settings.setValue("general/cutoutFovArcmin", int(self._cutout_fov_arcmin))
        return True

    def _telescope_overlay_rect(
        self,
        width_px: int,
        height_px: int,
        margin_px: int = 0,
        fov_axes: Optional[tuple[float, float]] = None,
    ) -> Optional[tuple[int, int, int, int]]:
        tel_fov = self._site_telescope_fov_arcmin()
        if tel_fov is None:
            return None
        if fov_axes is None:
            cutout_fov_x, cutout_fov_y = self._cutout_fov_axes_arcmin(width_px, height_px)
        else:
            cutout_fov_x, cutout_fov_y = float(fov_axes[0]), float(fov_axes[1])
        if cutout_fov_x <= 0.0 or cutout_fov_y <= 0.0:
            return None
        rel_w = max(0.0, min(1.0, float(tel_fov[0]) / float(cutout_fov_x)))
        rel_h = max(0.0, min(1.0, float(tel_fov[1]) / float(cutout_fov_y)))
        if rel_w <= 0.0 or rel_h <= 0.0:
            return None
        safe_margin = max(0, int(margin_px))
        avail_w = max(8, int(width_px) - 2 * safe_margin)
        avail_h = max(8, int(height_px) - 2 * safe_margin)
        rect_w = max(8, int(round(float(avail_w) * rel_w)))
        rect_h = max(8, int(round(float(avail_h) * rel_h)))
        rect_w = min(rect_w, avail_w)
        rect_h = min(rect_h, avail_h)
        x0 = safe_margin + max(0, (avail_w - rect_w) // 2)
        y0 = safe_margin + max(0, (avail_h - rect_h) // 2)
        return x0, y0, rect_w, rect_h

    def _paint_telescope_fov_overlay(
        self,
        painter: QPainter,
        w: int,
        h: int,
        fill: bool = True,
        color: Optional[QColor] = None,
        offset_x: int = 0,
        offset_y: int = 0,
        fov_axes: Optional[tuple[float, float]] = None,
        min_margin_px: int = 4,
    ) -> None:
        pen_width = max(1, int(min(w, h) * 0.005))
        margin = max(int(min_margin_px), int(math.ceil(pen_width * 0.8)))
        overlay = self._telescope_overlay_rect(w, h, margin_px=margin, fov_axes=fov_axes)
        if overlay is None:
            return
        x0, y0, rw, rh = overlay
        overlay_color = color or self._theme_qcolor("overlay_fov", "#59f3ff")
        overlay_color.setAlpha(180 if color is None else overlay_color.alpha())
        pen = QPen(overlay_color)
        pen.setWidth(pen_width)
        pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        if fill:
            fill_color = QColor(overlay_color)
            fill_color.setAlpha(24)
            painter.setBrush(fill_color)
        else:
            painter.setBrush(Qt.BrushStyle.NoBrush)
        # Qt drawRect includes the right/bottom border pixels; shrink by 1 px
        # so dashed edges are not clipped when overlay touches panel bounds.
        draw_w = max(2, int(rw) - 1)
        draw_h = max(2, int(rh) - 1)
        painter.drawRect(int(x0 + offset_x), int(y0 + offset_y), draw_w, draw_h)

    def _cutout_fov_text(
        self,
        width_px: int,
        height_px: int,
        fetch_margin: bool = False,
        fov_axes: Optional[tuple[float, float]] = None,
    ) -> str:
        if fov_axes is not None:
            fov_x, fov_y = float(fov_axes[0]), float(fov_axes[1])
        elif fetch_margin:
            fov_x, fov_y = self._aladin_fetch_fov_axes_arcmin(width_px, height_px)
        else:
            fov_x, fov_y = self._cutout_fov_axes_arcmin(width_px, height_px)
        return f"{fov_x:.1f}x{fov_y:.1f} arcmin"

    def _cutout_key_for_target(self, target: Target, width_px: int, height_px: int) -> str:
        fetch_w, fetch_h = self._aladin_fetch_dimensions_px(width_px, height_px)
        return (
            f"{self._cutout_survey_key}:{self._cutout_fov_arcmin}:{width_px}x{height_px}:"
            f"fetch{fetch_w}x{fetch_h}:"
            f"ctx{float(getattr(self, '_aladin_context_factor', 1.0)):.2f}:"
            f"fetchm{CUTOUT_ALADIN_FETCH_MARGIN:.2f}:minfov{CUTOUT_ALADIN_FETCH_MIN_ARCMIN:.1f}:"
            f"telfetchm{CUTOUT_ALADIN_FETCH_TELESCOPE_MARGIN:.2f}:"
            f"telfetchmax{CUTOUT_ALADIN_FETCH_TELESCOPE_MAX_ARCMIN:.1f}:"
            f"res{CUTOUT_ALADIN_FETCH_RES_MULT:.2f}:minpx{CUTOUT_ALADIN_FETCH_MIN_SHORT_PX}:maxpx{CUTOUT_ALADIN_FETCH_MAX_EDGE_PX}:"
            f"{self._telescope_overlay_signature()}:"
            f"{target.ra:.6f},{target.dec:.6f}"
        )

    def _aladin_visible_image_rect(
        self,
        widget_w: int,
        widget_h: int,
        image_x: int,
        image_y: int,
        image_w: int,
        image_h: int,
    ) -> tuple[int, int, int, int]:
        if widget_w <= 0 or widget_h <= 0:
            return 0, 0, 1, 1
        if image_w <= 0 or image_h <= 0:
            return 0, 0, int(widget_w), int(widget_h)
        left = max(0, int(image_x))
        top = max(0, int(image_y))
        right = min(int(widget_w), int(image_x + image_w))
        bottom = min(int(widget_h), int(image_y + image_h))
        if right <= left or bottom <= top:
            return 0, 0, int(widget_w), int(widget_h)
        return left, top, int(right - left), int(bottom - top)

    def _aladin_visible_fov_axes_arcmin(
        self,
        widget_w: int,
        widget_h: int,
        image_x: int,
        image_y: int,
        image_w: int,
        image_h: int,
    ) -> tuple[float, float]:
        base_x, base_y = self._aladin_fetch_fov_axes_arcmin(image_w, image_h)
        _ = (image_x, image_y)
        # FOV should react to zoom level; panning must not change FOV value.
        frac_x = max(0.02, min(1.0, float(max(1, widget_w)) / float(max(1, image_w))))
        frac_y = max(0.02, min(1.0, float(max(1, widget_h)) / float(max(1, image_h))))
        return float(base_x) * frac_x, float(base_y) * frac_y

    def _cutout_resize_signature_for_target(self, target: Optional[Target]) -> Optional[tuple]:
        if target is None:
            return None
        aw, ah = self._cutout_render_dimensions_px(getattr(self, "aladin_image_label", None))
        fw, fh = self._cutout_render_dimensions_px(getattr(self, "finder_image_label", None))
        return (
            target.name.strip().lower(),
            self._cutout_view_key,
            self._cutout_survey_key,
            int(self._cutout_fov_arcmin),
            round(float(getattr(self, "_aladin_context_factor", 1.0)), 3),
            self._telescope_overlay_signature(),
            int(aw),
            int(ah),
            int(fw),
            int(fh),
        )

    @Slot(int, int)
    def _schedule_cutout_resize_refresh(self, *_):
        if getattr(self, "_shutting_down", False):
            return
        timer = getattr(self, "_cutout_resize_timer", None)
        if timer is not None:
            timer.start()

    @Slot()
    def _on_cutout_resize_timeout(self):
        if getattr(self, "_shutting_down", False):
            return
        target = self._selected_target_or_none()
        if target is None:
            return
        fov_changed = self._sync_cutout_fov_to_site()
        sig = self._cutout_resize_signature_for_target(target)
        if sig is None:
            return
        if (not fov_changed) and sig == self._cutout_last_resize_signature:
            return
        self._cutout_last_resize_signature = sig
        self._update_cutout_preview_for_target(target)

    def _set_cutout_placeholder(self, text: str):
        if not hasattr(self, "aladin_image_label"):
            return
        for kind, label in (
            ("aladin", self.aladin_image_label),
            ("finder", getattr(self, "finder_image_label", None)),
        ):
            if label is None:
                continue
            label.setPixmap(QPixmap())
            label.setText(text)
            self._set_cutout_image_loading(kind, text, visible=False)

    def _paint_aladin_static_overlay(
        self,
        painter: QPainter,
        w: int,
        h: int,
        image_x: Optional[int] = None,
        image_y: Optional[int] = None,
        image_w: Optional[int] = None,
        image_h: Optional[int] = None,
    ) -> None:
        painter.setRenderHint(QPainter.Antialiasing, True)
        iw = int(image_w if image_w is not None else w)
        ih = int(image_h if image_h is not None else h)
        ix = int(image_x if image_x is not None else 0)
        iy = int(image_y if image_y is not None else 0)
        _ = (ix, iy)

        # Keep crosshair + telescope FOV fixed in viewport center.
        cx = w // 2
        cy = h // 2
        radius = max(10, int(min(w, h) * 0.14))

        # Crosshair and center ring for quick visual centering.
        crosshair = self._theme_qcolor("overlay_crosshair", "#ff5d8f")
        crosshair.setAlpha(225)
        pen_cross = QPen(crosshair)
        pen_cross.setWidth(max(1, int(min(w, h) * 0.01)))
        painter.setPen(pen_cross)
        span = max(14, int(min(w, h) * 0.18))
        painter.drawLine(cx - span, cy, cx + span, cy)
        painter.drawLine(cx, cy - span, cx, cy + span)
        ring_color = self._theme_qcolor("overlay_strip_text", "#eef4fc")
        ring_color.setAlpha(210)
        pen_ring = QPen(ring_color)
        pen_ring.setWidth(max(1, int(min(w, h) * 0.008)))
        painter.setPen(pen_ring)
        painter.drawEllipse(cx - radius, cy - radius, radius * 2, radius * 2)
        overlay_fov_axes = self._aladin_visible_fov_axes_arcmin(
            w,
            h,
            ix,
            iy,
            iw,
            ih,
        )
        self._paint_telescope_fov_overlay(
            painter,
            w,
            h,
            fov_axes=overlay_fov_axes,
        )

        # Keep survey/FOV visible directly on Aladin image.
        strip_h = max(16, int(h * 0.10))
        strip_y = h - strip_h
        strip_bg = self._theme_qcolor("overlay_strip_bg", "#08111d")
        strip_bg.setAlpha(165)
        painter.fillRect(0, strip_y, w, strip_h, strip_bg)
        strip_text = self._theme_qcolor("overlay_strip_text", "#eef4fc")
        strip_text.setAlpha(240)
        painter.setPen(strip_text)
        meta_txt = f"{_cutout_survey_label(self._cutout_survey_key)} | {self._cutout_fov_text(w, h, fov_axes=overlay_fov_axes)}"
        painter.drawText(8, strip_y, max(8, w - 16), strip_h, Qt.AlignVCenter | Qt.AlignLeft, meta_txt)

    def _build_aladin_overlay_pixmap(self, source: QPixmap) -> QPixmap:
        if source.isNull():
            return source
        view = source.copy()
        painter = QPainter(view)
        self._paint_aladin_static_overlay(painter, view.width(), view.height())
        painter.end()
        return view

    def _build_finder_overlay_pixmap(self, source: QPixmap) -> QPixmap:
        if source.isNull():
            return source
        view = source.copy()
        painter = QPainter(view)
        painter.setRenderHint(QPainter.Antialiasing, True)
        w = view.width()
        h = view.height()
        cx = w // 2
        cy = h // 2

        # Centered finder reticle (manual, to avoid astroplan offset on non-square frames).
        crosshair = self._theme_qcolor("overlay_crosshair", "#ff5d8f")
        crosshair.setAlpha(225)
        pen_cross = QPen(crosshair)
        pen_cross.setWidth(max(1, int(min(w, h) * 0.010)))
        painter.setPen(pen_cross)
        span = max(16, int(min(w, h) * 0.10))
        gap = max(6, int(span * 0.34))
        painter.drawLine(cx - span, cy, cx - gap, cy)
        painter.drawLine(cx + gap, cy, cx + span, cy)
        painter.drawLine(cx, cy - span, cx, cy - gap)
        painter.drawLine(cx, cy + gap, cx, cy + span)
        self._paint_telescope_fov_overlay(
            painter,
            w,
            h,
            fill=False,
            color=self._theme_qcolor("overlay_crosshair", "#ff5d8f"),
            min_margin_px=8,
        )

        strip_h = max(16, int(h * 0.10))
        strip_bg = self._theme_qcolor("overlay_strip_bg", "#08111d")
        strip_bg.setAlpha(165)
        painter.fillRect(0, h - strip_h, w, strip_h, strip_bg)
        strip_text = self._theme_qcolor("overlay_strip_text", "#eef4fc")
        strip_text.setAlpha(240)
        painter.setPen(strip_text)
        painter.drawText(
            8,
            h - strip_h,
            w - 16,
            strip_h,
            Qt.AlignVCenter | Qt.AlignLeft,
            f"FOV: {self._cutout_fov_text(w, h)}",
        )
        painter.end()
        return view

    def _pixmap_to_png_bytes(self, pixmap: QPixmap) -> bytes:
        if pixmap.isNull():
            return b""
        payload = QByteArray()
        buffer = QBuffer(payload)
        if not buffer.open(QIODevice.WriteOnly):
            return b""
        try:
            ok = pixmap.save(buffer, "PNG")
        finally:
            buffer.close()
        return bytes(payload) if ok else b""

    def _load_pixmap_from_storage_cache(self, namespace: str, key: str) -> Optional[QPixmap]:
        storage = getattr(self, "app_storage", None)
        if storage is None or not key:
            return None
        try:
            payload = storage.cache.get_json(namespace, key)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to read %s image cache '%s': %s", namespace, key, exc)
            return None
        if not isinstance(payload, dict):
            return None
        image_bytes = payload.get("image_bytes")
        if not isinstance(image_bytes, (bytes, bytearray)):
            return None
        pixmap = QPixmap()
        if not pixmap.loadFromData(bytes(image_bytes), "PNG") or pixmap.isNull():
            return None
        return pixmap

    def _persist_pixmap_to_storage_cache(self, namespace: str, key: str, pixmap: QPixmap) -> None:
        storage = getattr(self, "app_storage", None)
        if storage is None or not key or pixmap.isNull():
            return
        payload = self._pixmap_to_png_bytes(pixmap)
        if not payload:
            return
        try:
            storage.cache.set_json(namespace, key, {"image_bytes": payload}, ttl_s=14 * 24 * 60 * 60)
            storage.cache.prune_namespace(namespace, max_entries=160)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to persist %s image cache '%s': %s", namespace, key, exc)

    def _cache_cutout_pixmap(self, key: str, pixmap: QPixmap, *, persist: bool = True):
        self._cutout_cache[key] = pixmap
        if key in self._cutout_cache_order:
            self._cutout_cache_order.remove(key)
        self._cutout_cache_order.append(key)
        targets_n = len(getattr(self, "targets", []))
        cache_limit = max(CUTOUT_CACHE_MAX, min(160, int(targets_n) + 24))
        while len(self._cutout_cache_order) > cache_limit:
            stale = self._cutout_cache_order.pop(0)
            self._cutout_cache.pop(stale, None)
        if persist:
            self._persist_pixmap_to_storage_cache("cutout_preview", key, pixmap)

    def _cache_finder_pixmap(self, key: str, pixmap: QPixmap, *, persist: bool = True):
        self._finder_cache[key] = pixmap
        if key in self._finder_cache_order:
            self._finder_cache_order.remove(key)
        self._finder_cache_order.append(key)
        targets_n = len(getattr(self, "targets", []))
        cache_limit = max(CUTOUT_CACHE_MAX, min(160, int(targets_n) + 24))
        while len(self._finder_cache_order) > cache_limit:
            stale = self._finder_cache_order.pop(0)
            self._finder_cache.pop(stale, None)
        if persist:
            self._persist_pixmap_to_storage_cache("finder_preview", key, pixmap)

    def _find_cutout_cache_variant(self, target: Target) -> Optional[tuple[str, QPixmap]]:
        coord_suffix = f"{target.ra:.6f},{target.dec:.6f}"
        prefix = f"{self._cutout_survey_key}:{self._cutout_fov_arcmin}:"
        ctx_token = f"ctx{float(getattr(self, '_aladin_context_factor', 1.0)):.2f}:"
        overlay_token = f"{self._telescope_overlay_signature()}:"
        for key in reversed(self._cutout_cache_order):
            if not key.endswith(coord_suffix):
                continue
            if not key.startswith(prefix):
                continue
            if ctx_token not in key:
                continue
            if overlay_token not in key:
                continue
            pix = self._cutout_cache.get(key)
            if pix is None or pix.isNull():
                continue
            return key, pix
        return None

    def _find_finder_cache_variant(self, target: Target) -> Optional[tuple[str, QPixmap]]:
        coord_suffix = f"{target.ra:.6f},{target.dec:.6f}"
        prefix = f"{self._cutout_survey_key}:{self._cutout_fov_arcmin}:"
        overlay_token = f"{self._telescope_overlay_signature()}:"
        # Finder chart is independent from Aladin context zoom factor,
        # so allow reusing cached variants across ctx changes.
        for key in reversed(self._finder_cache_order):
            if not key.endswith(coord_suffix):
                continue
            if not key.startswith(prefix):
                continue
            if overlay_token not in key:
                continue
            pix = self._finder_cache.get(key)
            if pix is None or pix.isNull():
                continue
            return key, pix
        return None

    def _show_finder_aladin_fallback(self, key: str, text_if_missing: str) -> bool:
        if not hasattr(self, "finder_image_label"):
            return False
        # Keep Finder tab dedicated to finder output (no Aladin substitution).
        fallback = self._finder_cache.get(key)
        if fallback is not None and not fallback.isNull():
            self.finder_image_label.setText("")
            self.finder_image_label.setPixmap(fallback)
            self._set_cutout_image_loading("finder", "", visible=False)
            self._finder_displayed_key = key
            return True
        self.finder_image_label.setPixmap(QPixmap())
        self.finder_image_label.setText(text_if_missing)
        self._set_cutout_image_loading("finder", text_if_missing, visible=False)
        self._finder_displayed_key = ""
        return False

    def _ensure_aladin_pan_ready(self) -> None:
        if not hasattr(self, "aladin_image_label"):
            return
        label = self.aladin_image_label
        try:
            zoom_now = float(label.zoom_factor())
        except Exception:
            zoom_now = 1.0
        if abs(zoom_now - 1.0) > 1e-3:
            return
        label.set_zoom(float(CUTOUT_ALADIN_INITIAL_PAN_ZOOM))

    def _set_finder_status(self, text: str, busy: bool = False):
        if not hasattr(self, "status_finder_label") or not hasattr(self, "status_finder_progress"):
            return
        self.status_finder_label.setText(text)
        if busy:
            self.status_finder_progress.setRange(0, 0)
            self.status_finder_progress.show()
            return
        self.status_finder_progress.hide()
        self.status_finder_progress.setRange(0, 1)
        self.status_finder_progress.setValue(0)

    def _set_aladin_status(self, text: str, busy: bool = False):
        if not hasattr(self, "status_aladin_label") or not hasattr(self, "status_aladin_progress"):
            return
        self.status_aladin_label.setText(text)
        if busy:
            self.status_aladin_progress.setRange(0, 0)
            self.status_aladin_progress.show()
            return
        self.status_aladin_progress.hide()
        self.status_aladin_progress.setRange(0, 1)
        self.status_aladin_progress.setValue(0)

    def _finder_prefetch_done_status(self) -> None:
        total = max(int(getattr(self, "_finder_prefetch_total", 0)), int(getattr(self, "_finder_prefetch_completed", 0)))
        cached = max(0, int(getattr(self, "_finder_prefetch_cached", 0)))
        self._set_finder_status(
            f"Finder: prefetch done ({int(self._finder_prefetch_completed)}/{total}, {cached} cached)",
            busy=False,
        )

    def _aladin_prefetch_done_status(self) -> None:
        total = max(int(getattr(self, "_cutout_prefetch_total", 0)), int(getattr(self, "_cutout_prefetch_completed", 0)))
        cached = max(0, int(getattr(self, "_cutout_prefetch_cached", 0)))
        self._set_aladin_status(
            f"Aladin: prefetch done ({int(self._cutout_prefetch_completed)}/{total}, {cached} cached)",
            busy=False,
        )

    def _stop_finder_workers(self, aggressive: bool = False):
        workers = list(getattr(self, "_finder_workers", []))
        alive: list[FinderChartWorker] = []
        for worker in workers:
            try:
                if not shb.isValid(worker):
                    continue
            except Exception:
                continue

            running = False
            try:
                running = worker.isRunning()
            except Exception:
                running = False

            if running:
                try:
                    worker.requestInterruption()
                    worker.quit()
                except Exception:
                    pass
                if aggressive:
                    stopped = False
                    try:
                        stopped = worker.wait(5500)
                    except Exception:
                        stopped = False

            still_running = False
            try:
                still_running = shb.isValid(worker) and worker.isRunning()
            except Exception:
                still_running = False
            if still_running:
                alive.append(worker)
        self._finder_workers = alive
        if self._finder_worker is not None and self._finder_worker not in alive:
            self._finder_worker = None

    def _cancel_finder_chart_worker(self):
        had_pending = bool(self._finder_pending_key)
        if hasattr(self, "_finder_timeout_timer"):
            self._finder_timeout_timer.stop()
        self._stop_finder_workers(aggressive=False)
        self._finder_pending_key = ""
        self._finder_pending_name = ""
        self._finder_pending_background = False
        self._finder_prefetch_queue.clear()
        self._finder_prefetch_enqueued_keys.clear()
        self._finder_prefetch_total = 0
        self._finder_prefetch_completed = 0
        self._finder_prefetch_cached = 0
        self._finder_prefetch_active = False
        self._finder_request_id += 1
        self._set_finder_status("Finder: cancelled" if had_pending else "Finder: idle", busy=False)

    @Slot(int, str, bytes, str)
    def _on_finder_chart_completed(self, request_id: int, key: str, payload: bytes, err: str):
        if hasattr(self, "_finder_timeout_timer"):
            self._finder_timeout_timer.stop()
        if request_id != self._finder_request_id:
            return
        if key != self._finder_pending_key:
            return
        pending_name = str(getattr(self, "_finder_pending_name", "") or "")
        self._finder_pending_key = ""
        self._finder_pending_name = ""
        was_background = bool(getattr(self, "_finder_pending_background", False))
        self._finder_pending_background = False

        if err or not payload:
            status_text = "Finder: unavailable"
            if err and err.lower() == "cancelled":
                status_text = "Finder: cancelled"
            if not was_background:
                self._set_finder_status(status_text, busy=False)
            else:
                self._finder_prefetch_completed += 1
            if err and err.lower() != "cancelled":
                logger.warning("Finder chart generation failed for key '%s': %s", key, err)
                self._finder_retry_after[key] = perf_counter() + FINDER_RETRY_COOLDOWN_S
            if not was_background:
                self._show_finder_aladin_fallback(key, "Finder chart unavailable")
            self._drain_finder_prefetch_queue()
            if was_background and not self._finder_pending_key and not self._finder_prefetch_queue:
                self._finder_prefetch_done_status()
                self._finder_prefetch_active = False
            return

        image = QImage.fromData(payload, "PNG")
        if image.isNull():
            if not was_background:
                self._set_finder_status("Finder: decode failed", busy=False)
            else:
                self._finder_prefetch_completed += 1
            self._finder_retry_after[key] = perf_counter() + FINDER_RETRY_COOLDOWN_S
            if not was_background:
                self._show_finder_aladin_fallback(key, "Finder chart decode failed")
            self._drain_finder_prefetch_queue()
            if was_background and not self._finder_pending_key and not self._finder_prefetch_queue:
                self._finder_prefetch_done_status()
                self._finder_prefetch_active = False
            return
        pix = QPixmap.fromImage(image)
        if pix.isNull():
            if not was_background:
                self._set_finder_status("Finder: decode failed", busy=False)
            else:
                self._finder_prefetch_completed += 1
            self._finder_retry_after[key] = perf_counter() + FINDER_RETRY_COOLDOWN_S
            if not was_background:
                self._show_finder_aladin_fallback(key, "Finder chart decode failed")
            self._drain_finder_prefetch_queue()
            if was_background and not self._finder_pending_key and not self._finder_prefetch_queue:
                self._finder_prefetch_done_status()
                self._finder_prefetch_active = False
            return
        pix_with_overlay = self._build_finder_overlay_pixmap(pix)
        self._cache_finder_pixmap(key, pix_with_overlay)
        self._finder_retry_after.pop(key, None)
        if not was_background:
            name_hint = pending_name.strip()
            if name_hint:
                self._set_finder_status(f"Finder: ready ({name_hint})", busy=False)
            else:
                self._set_finder_status("Finder: ready", busy=False)
        if (not was_background) and hasattr(self, "finder_image_label"):
            self.finder_image_label.setText("")
            self.finder_image_label.setPixmap(pix_with_overlay)
            self._set_cutout_image_loading("finder", "", visible=False)
            self._finder_displayed_key = key
        if was_background:
            self._finder_prefetch_completed += 1
        self._drain_finder_prefetch_queue()
        if was_background and not self._finder_pending_key and not self._finder_prefetch_queue:
            self._finder_prefetch_done_status()
            self._finder_prefetch_active = False

    def _on_finder_chart_worker_finished(self, worker: FinderChartWorker):
        workers = getattr(self, "_finder_workers", None)
        if isinstance(workers, list) and worker in workers:
            workers.remove(worker)
        if self._finder_worker is worker:
            self._finder_worker = None
        if self._finder_pending_key:
            return
        if self._finder_prefetch_queue:
            self._drain_finder_prefetch_queue()
            return
        if self._finder_prefetch_active and self._finder_prefetch_completed >= self._finder_prefetch_total:
            self._finder_prefetch_done_status()
            self._finder_prefetch_active = False

    @Slot()
    def _on_finder_chart_timeout(self):
        key = self._finder_pending_key
        if not key:
            return
        pending_name = str(getattr(self, "_finder_pending_name", "") or "")
        was_background = bool(getattr(self, "_finder_pending_background", False))
        self._finder_pending_key = ""
        self._finder_pending_name = ""
        self._finder_pending_background = False
        self._finder_request_id += 1
        self._finder_retry_after[key] = perf_counter() + FINDER_RETRY_COOLDOWN_S
        self._stop_finder_workers(aggressive=True)
        if not was_background:
            if pending_name.strip():
                self._set_finder_status(f"Finder: timeout ({pending_name})", busy=False)
            else:
                self._set_finder_status("Finder: timeout", busy=False)
            self._show_finder_aladin_fallback(key, "Finder chart timeout")
        else:
            self._finder_prefetch_completed += 1
        logger.warning("Finder chart timed out for key '%s'", key)
        if not self._finder_workers:
            self._drain_finder_prefetch_queue()
        if was_background and not self._finder_pending_key and not self._finder_prefetch_queue and not self._finder_workers:
            self._finder_prefetch_done_status()
            self._finder_prefetch_active = False

    def _update_finder_chart_for_target(
        self,
        target: Target,
        key: str,
        *,
        background: bool = False,
        cache_only: bool = False,
    ):
        if not hasattr(self, "finder_image_label"):
            return
        if (
            not background
            and key == getattr(self, "_finder_displayed_key", "")
            and not self._finder_pending_key
        ):
            return
        cached = self._finder_cache.get(key)
        if cached is not None and not cached.isNull():
            if (not background) and hasattr(self, "_finder_timeout_timer"):
                self._finder_timeout_timer.stop()
            if not background:
                self.finder_image_label.setText("")
                self.finder_image_label.setPixmap(cached)
                self._set_cutout_image_loading("finder", "", visible=False)
                self._set_finder_status(f"Finder: cached ({target.name})", busy=False)
                self._finder_displayed_key = key
            return
        persisted = self._load_pixmap_from_storage_cache("finder_preview", key)
        if persisted is not None and not persisted.isNull():
            self._cache_finder_pixmap(key, persisted, persist=False)
            if (not background) and hasattr(self, "_finder_timeout_timer"):
                self._finder_timeout_timer.stop()
            if not background:
                self.finder_image_label.setText("")
                self.finder_image_label.setPixmap(persisted)
                self._set_cutout_image_loading("finder", "", visible=False)
                self._set_finder_status(f"Finder: cached ({target.name})", busy=False)
                self._finder_displayed_key = key
            return
        cached_variant = self._find_finder_cache_variant(target)
        if cached_variant is not None:
            _variant_key, variant_pix = cached_variant
            self._cache_finder_pixmap(key, variant_pix)
            if (not background) and hasattr(self, "_finder_timeout_timer"):
                self._finder_timeout_timer.stop()
            if not background:
                self.finder_image_label.setText("")
                self.finder_image_label.setPixmap(variant_pix)
                self._set_cutout_image_loading("finder", "", visible=False)
                self._set_finder_status(f"Finder: cached ({target.name})", busy=False)
                self._finder_displayed_key = key
            return
        if cache_only:
            if not background:
                self._set_finder_status(f"Finder: cache miss ({target.name})", busy=False)
            return
        if self._finder_pending_key == key:
            if not background:
                pending_name = str(getattr(self, "_finder_pending_name", "") or "").strip()
                if pending_name:
                    self._set_finder_status(f"Finder: loading {pending_name}...", busy=True)
                    self._set_cutout_image_loading("finder", f"Loading finder chart for {pending_name}…", visible=True)
                else:
                    self._set_finder_status("Finder: loading...", busy=True)
                    self._set_cutout_image_loading("finder", "Loading finder chart…", visible=True)
            return
        if background and self._finder_pending_key:
            self._enqueue_finder_prefetch(target, key)
            return

        retry_after = float(self._finder_retry_after.get(key, 0.0))
        now = perf_counter()
        if retry_after > now:
            secs = max(1, int(round(retry_after - now)))
            if not background:
                self._set_finder_status(f"Finder: retry in {secs}s", busy=False)
                self._show_finder_aladin_fallback(key, f"Finder chart unavailable ({secs}s)")
            return

        if not background:
            self.finder_image_label.setPixmap(QPixmap())
            self.finder_image_label.setText("Loading finder chart…")
            self._set_cutout_image_loading("finder", f"Loading finder chart for {target.name}…", visible=True)

        self._finder_request_id += 1
        req_id = self._finder_request_id
        self._finder_pending_key = key
        self._finder_pending_name = target.name
        self._finder_pending_background = background
        render_w, render_h = self._cutout_render_dimensions_px(getattr(self, "finder_image_label", None))
        old_worker = self._finder_worker
        if (not background) and old_worker is not None:
            try:
                if old_worker.isRunning():
                    old_worker.requestInterruption()
            except Exception:
                pass
        worker = FinderChartWorker(
            request_id=req_id,
            key=key,
            name=target.name,
            ra_deg=target.ra,
            dec_deg=target.dec,
            survey_key=self._cutout_survey_key,
            fov_arcmin=self._cutout_fov_arcmin,
            width_px=render_w,
            height_px=render_h,
            parent=self._planner,
        )
        worker.completed.connect(self._on_finder_chart_completed)
        worker.finished.connect(lambda w=worker: self._on_finder_chart_worker_finished(w))
        worker.finished.connect(worker.deleteLater)
        self._finder_workers.append(worker)
        self._finder_worker = worker
        if not background:
            self._set_finder_status(f"Finder: loading {target.name}...", busy=True)
        else:
            total = max(int(getattr(self, "_finder_prefetch_total", 0)), 1)
            done = int(getattr(self, "_finder_prefetch_completed", 0))
            cached = int(getattr(self, "_finder_prefetch_cached", 0))
            current_idx = min(total, max(1, done + 1))
            queued_left = len(getattr(self, "_finder_prefetch_queue", []))
            self._set_finder_status(
                f"Finder: prefetch {current_idx}/{total} ({cached} cached) {target.name} (queue {queued_left})",
                busy=True,
            )
        worker.start()
        if hasattr(self, "_finder_timeout_timer"):
            self._finder_timeout_timer.start()

    def _enqueue_finder_prefetch(self, target: Optional[Target], key: str) -> str:
        if target is None or not key:
            return "skipped"
        cached = self._finder_cache.get(key)
        if cached is not None and not cached.isNull():
            return "cached"
        cached_variant = self._find_finder_cache_variant(target)
        if cached_variant is not None:
            _variant_key, variant_pix = cached_variant
            # Alias current exact key to cached variant to prevent duplicate fetches.
            self._cache_finder_pixmap(key, variant_pix)
            return "cached"
        if key == self._finder_pending_key:
            return "skipped"
        if key in self._finder_prefetch_enqueued_keys:
            return "skipped"
        retry_after = float(self._finder_retry_after.get(key, 0.0))
        if retry_after > perf_counter():
            return "skipped"
        self._finder_prefetch_queue.append((target, key))
        self._finder_prefetch_enqueued_keys.add(key)
        return "queued"

    def _drain_finder_prefetch_queue(self) -> None:
        if self._finder_pending_key:
            return
        while self._finder_prefetch_queue and not self._finder_pending_key:
            target, key = self._finder_prefetch_queue.pop(0)
            self._finder_prefetch_enqueued_keys.discard(key)
            if target not in self.targets:
                continue
            self._update_finder_chart_for_target(target, key, background=True)

    def _prefetch_finder_charts_for_all_targets(self, prioritize: Optional[Target] = None) -> None:
        if not self.targets:
            return
        # Keep one stable batch at a time to avoid progress inflation.
        if self._finder_prefetch_active and (self._finder_pending_key or self._finder_prefetch_queue):
            return
        if self._finder_pending_key or self._finder_prefetch_queue:
            return
        candidates: list[Target] = []
        if prioritize is not None:
            candidates.append(prioritize)
        for candidate in self.targets:
            if prioritize is not None and candidate is prioritize:
                continue
            candidates.append(candidate)
        if not candidates:
            return

        self._finder_prefetch_total = len(candidates)
        self._finder_prefetch_completed = 0
        self._finder_prefetch_cached = 0
        self._finder_prefetch_active = True

        render_w, render_h = self._cutout_render_dimensions_px(getattr(self, "finder_image_label", None))
        for candidate in candidates:
            key = self._cutout_key_for_target(candidate, render_w, render_h)
            enqueue_state = self._enqueue_finder_prefetch(candidate, key)
            if enqueue_state == "queued":
                pass
            else:
                self._finder_prefetch_completed += 1
                if enqueue_state == "cached":
                    self._finder_prefetch_cached += 1

        if self._finder_prefetch_completed >= self._finder_prefetch_total:
            self._finder_prefetch_done_status()
            self._finder_prefetch_active = False
            return
        self._drain_finder_prefetch_queue()

    def _start_cutout_request(
        self,
        target: Target,
        key: str,
        *,
        render_w: int,
        render_h: int,
        fetch_w: int,
        fetch_h: int,
        background: bool = False,
    ) -> None:
        self._cutout_request_id += 1
        self._cutout_pending_key = key
        self._cutout_pending_name = target.name
        self._cutout_pending_background = background
        if not background:
            self.aladin_image_label.setPixmap(QPixmap())
            self.aladin_image_label.setText("Loading…")
            self._set_cutout_image_loading("aladin", f"Loading Aladin preview for {target.name}…", visible=True)
            self._set_aladin_status(f"Aladin: loading {target.name}...", busy=True)
        else:
            total = max(int(getattr(self, "_cutout_prefetch_total", 0)), 1)
            done = int(getattr(self, "_cutout_prefetch_completed", 0))
            cached = int(getattr(self, "_cutout_prefetch_cached", 0))
            current_idx = min(total, max(1, done + 1))
            queued_left = len(getattr(self, "_cutout_prefetch_queue", []))
            self._set_aladin_status(
                f"Aladin: prefetch {current_idx}/{total} ({cached} cached) {target.name} (queue {queued_left})",
                busy=True,
            )

        query = QUrlQuery()
        query.addQueryItem("hips", _cutout_survey_hips(self._cutout_survey_key))
        query.addQueryItem("ra", f"{target.ra:.8f}")
        query.addQueryItem("dec", f"{target.dec:.8f}")
        fov_x_arcmin, _ = self._aladin_fetch_fov_axes_arcmin(fetch_w, fetch_h)
        query.addQueryItem("fov", f"{(fov_x_arcmin / 60.0):.6f}")
        query.addQueryItem("width", str(fetch_w))
        query.addQueryItem("height", str(fetch_h))
        query.addQueryItem("projection", "TAN")
        query.addQueryItem("coordsys", "icrs")
        query.addQueryItem("format", "png")

        url = QUrl("https://alasky.cds.unistra.fr/hips-image-services/hips2fits")
        url.setQuery(query)
        req = QNetworkRequest(url)
        req.setAttribute(QNetworkRequest.Http2AllowedAttribute, False)
        if hasattr(req, "setTransferTimeout"):
            req.setTransferTimeout(12000)
        req.setRawHeader(b"User-Agent", b"AstroPlanner/1.0 (cutout)")
        req.setRawHeader(b"Accept", b"image/png,image/*;q=0.8,*/*;q=0.2")
        reply = self._cutout_manager.get(req)
        reply.setProperty("cutout_request_id", self._cutout_request_id)
        reply.setProperty("cutout_key", key)
        reply.setProperty("cutout_name", target.name)
        reply.setProperty("cutout_background", 1 if background else 0)
        self._cutout_reply = reply

    def _enqueue_cutout_prefetch(
        self,
        target: Optional[Target],
        key: str,
        *,
        render_w: int,
        render_h: int,
        fetch_w: int,
        fetch_h: int,
    ) -> str:
        if target is None or not key:
            return "skipped"
        cached = self._cutout_cache.get(key)
        if cached is not None and not cached.isNull():
            return "cached"
        cached_variant = self._find_cutout_cache_variant(target)
        if cached_variant is not None:
            _variant_key, variant_pix = cached_variant
            # Alias current exact key to cached variant to prevent duplicate fetches.
            self._cache_cutout_pixmap(key, variant_pix)
            return "cached"
        if key == self._cutout_pending_key:
            return "skipped"
        if key in self._cutout_prefetch_enqueued_keys:
            return "skipped"
        self._cutout_prefetch_queue.append((target, key, int(render_w), int(render_h), int(fetch_w), int(fetch_h)))
        self._cutout_prefetch_enqueued_keys.add(key)
        return "queued"

    def _drain_cutout_prefetch_queue(self) -> None:
        if self._cutout_pending_key:
            return
        while self._cutout_prefetch_queue and not self._cutout_pending_key:
            target, key, render_w, render_h, fetch_w, fetch_h = self._cutout_prefetch_queue.pop(0)
            self._cutout_prefetch_enqueued_keys.discard(key)
            if target not in self.targets:
                if self._cutout_prefetch_active:
                    self._cutout_prefetch_completed += 1
                continue
            self._start_cutout_request(
                target,
                key,
                render_w=render_w,
                render_h=render_h,
                fetch_w=fetch_w,
                fetch_h=fetch_h,
                background=True,
            )

    def _prefetch_cutouts_for_all_targets(self, prioritize: Optional[Target] = None) -> None:
        if not self.targets:
            return
        # Keep one stable batch at a time to avoid progress inflation.
        if self._cutout_prefetch_active and (self._cutout_pending_key or self._cutout_prefetch_queue):
            return
        if self._cutout_pending_key or self._cutout_prefetch_queue:
            return
        candidates: list[Target] = []
        if prioritize is not None:
            candidates.append(prioritize)
        for candidate in self.targets:
            if prioritize is not None and candidate is prioritize:
                continue
            candidates.append(candidate)
        if not candidates:
            return

        self._cutout_prefetch_total = len(candidates)
        self._cutout_prefetch_completed = 0
        self._cutout_prefetch_cached = 0
        self._cutout_prefetch_active = True

        render_w, render_h = self._cutout_render_dimensions_px(getattr(self, "aladin_image_label", None))
        fetch_w, fetch_h = self._aladin_fetch_dimensions_px(render_w, render_h)
        for candidate in candidates:
            key = self._cutout_key_for_target(candidate, render_w, render_h)
            enqueue_state = self._enqueue_cutout_prefetch(
                candidate,
                key,
                render_w=render_w,
                render_h=render_h,
                fetch_w=fetch_w,
                fetch_h=fetch_h,
            )
            if enqueue_state == "queued":
                pass
            else:
                self._cutout_prefetch_completed += 1
                if enqueue_state == "cached":
                    self._cutout_prefetch_cached += 1

        if self._cutout_prefetch_completed >= self._cutout_prefetch_total:
            self._aladin_prefetch_done_status()
            self._cutout_prefetch_active = False
            return
        self._drain_cutout_prefetch_queue()

    def _update_cutout_preview_for_target(self, target: Optional[Target], *, cache_only: bool = False):
        if not hasattr(self, "cutout_image_label"):
            return
        use_cache_only = bool(cache_only)

        if target is None:
            if self._cutout_reply is not None and not self._cutout_reply.isFinished():
                self._cutout_reply.abort()
            self._cutout_reply = None
            self._cutout_pending_key = ""
            self._cutout_pending_name = ""
            self._cutout_pending_background = False
            self._cutout_displayed_key = ""
            self._finder_displayed_key = ""
            self._set_cutout_placeholder("Select a target")
            self._set_aladin_status("Aladin: idle", busy=False)
            self._set_finder_status("Finder: idle", busy=False)
            return

        self._sync_cutout_fov_to_site()
        render_w, render_h = self._cutout_render_dimensions_px(getattr(self, "aladin_image_label", None))
        fetch_w, fetch_h = self._aladin_fetch_dimensions_px(render_w, render_h)
        key = self._cutout_key_for_target(target, render_w, render_h)
        self._cutout_last_resize_signature = self._cutout_resize_signature_for_target(target)
        show_finder = getattr(self, "_cutout_view_key", "aladin") == "finderchart"
        finder_already_displayed = (not show_finder) or (
            key == getattr(self, "_finder_displayed_key", "")
            and not self._finder_pending_key
        )
        if (
            key == getattr(self, "_cutout_displayed_key", "")
            and not self._cutout_pending_key
            and finder_already_displayed
        ):
            return
        if show_finder:
            self._update_finder_chart_for_target(target, key, cache_only=use_cache_only)

        if key in self._cutout_cache:
            aladin_pix = self._cutout_cache[key]
            self.aladin_image_label.setText("")
            self.aladin_image_label.setPixmap(aladin_pix)
            self._set_cutout_image_loading("aladin", "", visible=False)
            self._ensure_aladin_pan_ready()
            self._set_aladin_status(f"Aladin: cached ({target.name})", busy=False)
            self._prefetch_cutouts_for_all_targets(prioritize=target)
            if show_finder:
                self._update_finder_chart_for_target(target, key, cache_only=use_cache_only)
            else:
                self._prefetch_finder_charts_for_all_targets(prioritize=target)
            self._cutout_displayed_key = key
            return

        persisted_cutout = self._load_pixmap_from_storage_cache("cutout_preview", key)
        if persisted_cutout is not None and not persisted_cutout.isNull():
            self._cache_cutout_pixmap(key, persisted_cutout, persist=False)
            self.aladin_image_label.setText("")
            self.aladin_image_label.setPixmap(persisted_cutout)
            self._set_cutout_image_loading("aladin", "", visible=False)
            self._ensure_aladin_pan_ready()
            self._set_aladin_status(f"Aladin: cached ({target.name})", busy=False)
            self._prefetch_cutouts_for_all_targets(prioritize=target)
            if show_finder:
                self._update_finder_chart_for_target(target, key, cache_only=use_cache_only)
            else:
                self._prefetch_finder_charts_for_all_targets(prioritize=target)
            self._cutout_displayed_key = key
            return

        cached_variant = self._find_cutout_cache_variant(target)
        if cached_variant is not None:
            cached_key, cached_pix = cached_variant
            self.aladin_image_label.setText("")
            self.aladin_image_label.setPixmap(cached_pix)
            self._set_cutout_image_loading("aladin", "", visible=False)
            self._ensure_aladin_pan_ready()
            self._set_aladin_status(f"Aladin: cached ({target.name})", busy=False)
            self._prefetch_cutouts_for_all_targets(prioritize=target)
            if show_finder:
                self._update_finder_chart_for_target(target, key, cache_only=use_cache_only)
            else:
                self._prefetch_finder_charts_for_all_targets(prioritize=target)
            self._cutout_displayed_key = cached_key
            return
        if use_cache_only:
            self._set_aladin_status(f"Aladin: cache miss ({target.name})", busy=False)
            if show_finder:
                self._update_finder_chart_for_target(target, key, cache_only=True)
            if not getattr(self, "_cutout_displayed_key", ""):
                self._set_cutout_placeholder("Preview not cached yet")
            return

        if key == self._cutout_pending_key and self._cutout_reply is not None and not self._cutout_reply.isFinished():
            pending_name = str(getattr(self, "_cutout_pending_name", "") or "").strip()
            if pending_name:
                self._set_aladin_status(f"Aladin: loading {pending_name}...", busy=True)
                self._set_cutout_image_loading("aladin", f"Loading Aladin preview for {pending_name}…", visible=True)
            else:
                self._set_aladin_status("Aladin: loading...", busy=True)
                self._set_cutout_image_loading("aladin", "Loading Aladin preview…", visible=True)
            return

        if self._cutout_reply is not None and not self._cutout_reply.isFinished():
            if self._cutout_pending_background and self._cutout_prefetch_active:
                self._cutout_prefetch_completed += 1
            self._cutout_reply.abort()
        self._cutout_reply = None
        self._cutout_pending_key = ""
        self._cutout_pending_name = ""
        self._cutout_pending_background = False

        self._start_cutout_request(
            target,
            key,
            render_w=render_w,
            render_h=render_h,
            fetch_w=fetch_w,
            fetch_h=fetch_h,
            background=False,
        )

    @Slot(QNetworkReply)
    def _on_cutout_reply(self, reply: QNetworkReply):
        req_id = int(reply.property("cutout_request_id") or 0)
        key = str(reply.property("cutout_key") or "")
        target_name = str(reply.property("cutout_name") or "").strip()
        is_background = bool(int(reply.property("cutout_background") or 0))

        # Ignore stale or unrelated replies.
        if req_id != self._cutout_request_id or key != self._cutout_pending_key:
            reply.deleteLater()
            return

        self._cutout_reply = None
        self._cutout_pending_key = ""
        self._cutout_pending_name = ""
        self._cutout_pending_background = False

        if reply.error() != QNetworkReply.NoError:
            err = reply.error()
            if err != QNetworkReply.OperationCanceledError:
                logger.warning("Cutout fetch failed for '%s': %s", target_name or key, reply.errorString())
                if not is_background:
                    self._set_cutout_placeholder("Preview unavailable")
                    label = target_name or "target"
                    self._set_aladin_status(f"Aladin: unavailable ({label})", busy=False)
            if is_background:
                self._cutout_prefetch_completed += 1
            reply.deleteLater()
            self._drain_cutout_prefetch_queue()
            if is_background and not self._cutout_pending_key and not self._cutout_prefetch_queue:
                self._aladin_prefetch_done_status()
                self._cutout_prefetch_active = False
            return

        payload = bytes(reply.readAll())
        reply.deleteLater()
        pixmap = QPixmap()
        if not payload or not pixmap.loadFromData(payload):
            if not is_background:
                self._set_cutout_placeholder("Preview decode failed")
                label = target_name or "target"
                self._set_aladin_status(f"Aladin: decode failed ({label})", busy=False)
            else:
                self._cutout_prefetch_completed += 1
                self._drain_cutout_prefetch_queue()
                if not self._cutout_pending_key and not self._cutout_prefetch_queue:
                    self._aladin_prefetch_done_status()
                    self._cutout_prefetch_active = False
            return

        self._cache_cutout_pixmap(key, pixmap)
        target_lc = target_name.strip().lower()
        target = next((t for t in self.targets if t.name.strip().lower() == target_lc), None)
        if not is_background:
            self.aladin_image_label.setText("")
            self.aladin_image_label.setPixmap(pixmap)
            self._set_cutout_image_loading("aladin", "", visible=False)
            self._ensure_aladin_pan_ready()
            label = target_name or "target"
            self._set_aladin_status(f"Aladin: ready ({label})", busy=False)
            self._cutout_displayed_key = key
            if target is not None and hasattr(self, "finder_image_label"):
                should_render_finder = getattr(self, "_cutout_view_key", "aladin") == "finderchart"
                if should_render_finder:
                    self._update_finder_chart_for_target(target, key)
                else:
                    self._prefetch_finder_charts_for_all_targets(prioritize=target)
            self._prefetch_cutouts_for_all_targets(prioritize=target)
        else:
            self._cutout_prefetch_completed += 1
        self._drain_cutout_prefetch_queue()
        if is_background and not self._cutout_pending_key and not self._cutout_prefetch_queue:
            self._aladin_prefetch_done_status()
            self._cutout_prefetch_active = False

    def _clear_cutout_cache(self):
        if self._cutout_reply is not None and not self._cutout_reply.isFinished():
            self._cutout_reply.abort()
        self._cancel_finder_chart_worker()
        self._cutout_reply = None
        self._cutout_prefetch_queue.clear()
        self._cutout_prefetch_enqueued_keys.clear()
        self._cutout_prefetch_total = 0
        self._cutout_prefetch_completed = 0
        self._cutout_prefetch_cached = 0
        self._cutout_prefetch_active = False
        self._cutout_cache.clear()
        self._cutout_cache_order.clear()
        self._finder_cache.clear()
        self._finder_cache_order.clear()
        self._finder_retry_after.clear()
        self._cutout_last_resize_signature = None
        self._cutout_displayed_key = ""
        self._finder_displayed_key = ""
        self._cutout_pending_key = ""
        self._cutout_pending_name = ""
        self._cutout_pending_background = False
        self._set_aladin_status("Aladin: idle", busy=False)

    def _selected_target_or_none(self) -> Optional[Target]:
        rows = self._selected_rows() if hasattr(self, "table_view") else []
        if rows and 0 <= rows[0] < len(self.targets):
            return self.targets[rows[0]]
        return None

    @Slot(int)
    def _on_cutout_tab_changed(self, index: int):
        self._cutout_view_key = "finderchart" if index == 1 else "aladin"
        self.settings.setValue("general/cutoutView", self._cutout_view_key)
        if self._cutout_view_key != "finderchart":
            self._cancel_finder_chart_worker()
        self._update_cutout_preview_for_target(self._selected_target_or_none())

    @Slot()
    def _aladin_zoom_in(self):
        if not hasattr(self, "aladin_image_label"):
            return
        self.aladin_image_label.set_zoom(self.aladin_image_label.zoom_factor() * 1.15)

    def _aladin_expand_context(self) -> None:
        context_now = max(1.0, float(getattr(self, "_aladin_context_factor", 1.0)))
        step = max(1.05, float(CUTOUT_ALADIN_CONTEXT_STEP))
        context_next = min(8.0, context_now * step)
        if context_next <= context_now + 1e-4:
            return
        self._aladin_context_factor = context_next
        self._update_cutout_preview_for_target(self._selected_target_or_none())

    @Slot()
    def _aladin_zoom_out(self):
        if not hasattr(self, "aladin_image_label"):
            return
        label = self.aladin_image_label
        current = float(label.zoom_factor())
        if current > 1.02:
            label.set_zoom(current / 1.15)
            return
        # Already at base scale: widen fetched Aladin context.
        self._aladin_expand_context()

    @Slot()
    def _aladin_zoom_reset(self):
        if not hasattr(self, "aladin_image_label"):
            return
        # Reset only viewport state (zoom/pan) in current image.
        # Do not change context factor and do not refetch cutout.
        self.aladin_image_label.reset_zoom()
        self.aladin_image_label.set_zoom(float(CUTOUT_ALADIN_INITIAL_PAN_ZOOM))


__all__ = [
    "CUTOUT_ALADIN_CONTEXT_STEP",
    "CUTOUT_ALADIN_FETCH_MARGIN",
    "CUTOUT_ALADIN_FETCH_MAX_EDGE_PX",
    "CUTOUT_ALADIN_FETCH_MIN_ARCMIN",
    "CUTOUT_ALADIN_FETCH_MIN_SHORT_PX",
    "CUTOUT_ALADIN_FETCH_RES_MULT",
    "CUTOUT_ALADIN_FETCH_TELESCOPE_MARGIN",
    "CUTOUT_ALADIN_FETCH_TELESCOPE_MAX_ARCMIN",
    "CUTOUT_CACHE_MAX",
    "CUTOUT_DEFAULT_FOV_ARCMIN",
    "CUTOUT_DEFAULT_SIZE_PX",
    "CUTOUT_DEFAULT_SURVEY_KEY",
    "CUTOUT_DEFAULT_VIEW_KEY",
    "CUTOUT_MAX_FOV_ARCMIN",
    "CUTOUT_MAX_SIZE_PX",
    "CUTOUT_MIN_FOV_ARCMIN",
    "CUTOUT_MIN_SIZE_PX",
    "CUTOUT_SURVEY_CHOICES",
    "CUTOUT_VIEW_CHOICES",
    "FINDER_RETRY_COOLDOWN_S",
    "FINDER_WORKER_TIMEOUT_MS",
    "PreviewCoordinator",
    "_cutout_survey_hips",
    "_cutout_survey_label",
    "_normalize_cutout_survey_key",
    "_normalize_cutout_view_key",
    "_sanitize_cutout_fov_arcmin",
    "_sanitize_cutout_size_px",
]
