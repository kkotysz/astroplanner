from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
from time import perf_counter
from typing import TYPE_CHECKING, Any, Optional

import ephem
import matplotlib.dates as mdates
import numpy as np
import pytz
from astropy import units as u
from astropy.coordinates import Angle, SkyCoord
from astropy.time import Time
from astroplan import FixedTarget, Observer
from matplotlib import patheffects as mpl_patheffects
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from PySide6.QtCore import QEasingCurve, QSignalBlocker, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QFontMetrics

from astroplanner.models import CalcRunStats, Site, Target
from astroplanner.resolvers import _normalize_catalog_token, _safe_float
from astroplanner.scoring import compute_target_metrics
from astroplanner.theme import (
    COLORBLIND_LINE_COLORS,
    DEFAULT_LINE_COLORS,
    DEFAULT_UI_THEME,
    line_palette_for_theme,
)
from astroplanner.ui.theme_utils import _normalized_css_color, _set_label_tone
from astroplanner.visibility_plotly import (
    VISIBILITY_AIRMASS_Y_MAX,
    VISIBILITY_AIRMASS_Y_MIN,
    VISIBILITY_AIRMASS_Y_TICKS,
)

if TYPE_CHECKING:
    from astro_planner import MainWindow


logger = logging.getLogger(__name__)


class VisibilityMatplotlibCoordinator:
    """Own Matplotlib visibility rendering and radar/polar plot helpers."""

    def __init__(self, planner: "MainWindow") -> None:
        object.__setattr__(self, "_planner", planner)

    def __getattr__(self, name: str):
        return getattr(self._planner, name)

    def __setattr__(self, name: str, value: object) -> None:
        if name == "_planner":
            object.__setattr__(self, name, value)
            return
        setattr(self._planner, name, value)

    def _refresh_mpl_theme(self) -> None:
        plot_bg = self._theme_color("plot_bg", "#0f1825")
        plot_panel_bg = self._theme_color("plot_panel_bg", "#162334")
        plot_text = self._theme_color("plot_text", "#d7e4f0")
        plot_grid = self._theme_color("plot_grid", "#2f4666")
        plot_canvas = getattr(self, "plot_canvas", None)
        ax_alt = getattr(self, "ax_alt", None)
        if plot_canvas is not None and ax_alt is not None:
            plot_canvas.figure.patch.set_facecolor(plot_bg)
            ax_alt.set_facecolor(plot_panel_bg)
            ax_alt.tick_params(axis="x", colors=plot_text)
            ax_alt.tick_params(axis="y", colors=plot_text)
            for spine in ax_alt.spines.values():
                spine.set_color(plot_grid)
            ax_alt.xaxis.label.set_color(plot_text)
            ax_alt.yaxis.label.set_color(plot_text)
            ax_alt.title.set_color(plot_text)
        if hasattr(self, "polar_canvas") and hasattr(self, "polar_ax"):
            self.polar_canvas.figure.patch.set_facecolor(plot_bg)
            self.polar_ax.set_facecolor(plot_panel_bg)
            self.polar_ax.tick_params(axis="x", colors=plot_text, pad=0)
            self.polar_ax.tick_params(axis="y", colors=plot_text, pad=1)
            self.polar_ax.grid(True, color=plot_grid, alpha=0.36, linestyle="--", linewidth=0.7)
            for spine in self.polar_ax.spines.values():
                spine.set_color(plot_grid)
            if hasattr(self, "polar_scatter"):
                self.polar_scatter.set_color(self._theme_color("polar_target", "#59f3ff"))
            if hasattr(self, "selected_scatter"):
                self.selected_scatter.set_color(self._theme_color("polar_selected", "#ff4fd8"))
            if hasattr(self, "sun_marker"):
                self.sun_marker.set_color(self._theme_color("polar_sun", "#ffb224"))
            if hasattr(self, "moon_marker"):
                self.moon_marker.set_color(self._theme_color("polar_moon", "#dbe7ff"))
            if hasattr(self, "radar_sweep_line") and self.radar_sweep_line is not None:
                self.radar_sweep_line.set_color(self._qcolor_rgba_mpl(self._theme_qcolor("accent_secondary", "#ff4fd8"), 0.0))
            if hasattr(self, "radar_sweep_glow_line") and self.radar_sweep_glow_line is not None:
                self.radar_sweep_glow_line.set_color(self._qcolor_rgba_mpl(self._theme_qcolor("accent_secondary_soft", "#d38cff"), 0.48))
            if hasattr(self, "radar_sweep_core") and self.radar_sweep_core is not None:
                self.radar_sweep_core.set_visible(False)
            if hasattr(self, "radar_sweep_mesh") and self.radar_sweep_mesh is not None:
                try:
                    self.radar_sweep_mesh.set_cmap(self._build_radar_sweep_cmap())
                except Exception:
                    pass
            if hasattr(self, "_radar_echo_artists") or hasattr(self, "radar_sweep_mesh") or (hasattr(self, "radar_echo_scatter") and self.radar_echo_scatter is not None):
                self._refresh_radar_sweep_artists(redraw=False)
        if plot_canvas is not None:
            plot_canvas.draw_idle()
        if hasattr(self, "polar_canvas"):
            self.polar_canvas.draw_idle()


    def _line_palette(self) -> list[str]:
        palette = line_palette_for_theme(
            getattr(self, "_theme_name", DEFAULT_UI_THEME),
            dark_enabled=getattr(self, "_dark_enabled", False),
            color_blind=bool(getattr(self, "color_blind_mode", False)),
        )
        return palette or (COLORBLIND_LINE_COLORS if self.color_blind_mode else DEFAULT_LINE_COLORS)


    @staticmethod
    def _target_color_key(target: Target) -> str:
        source_key = _normalize_catalog_token(getattr(target, "source_object_id", ""))
        if source_key:
            return f"id:{source_key}"
        name_key = _normalize_catalog_token(getattr(target, "name", ""))
        if name_key:
            return f"name:{name_key}"
        ra = _safe_float(getattr(target, "ra", None))
        dec = _safe_float(getattr(target, "dec", None))
        if ra is not None and dec is not None and math.isfinite(ra) and math.isfinite(dec):
            return f"coord:{ra:.6f},{dec:.6f}"
        return f"obj:{id(target)}"


    def _ensure_auto_target_color_palette(self, palette: Optional[list[str]] = None) -> list[str]:
        use_palette = palette if palette is not None else self._line_palette()
        signature = tuple(str(color) for color in use_palette)
        prev_signature = tuple(getattr(self, "_auto_target_color_palette_signature", ()))
        if signature != prev_signature:
            self._auto_target_color_palette_signature = signature
            self._auto_target_color_map = {}
        return use_palette


    def _target_plot_color_css(
        self,
        target: Target,
        index: int,
        palette: Optional[list[str]] = None,
    ) -> str:
        custom_css = _normalized_css_color(target.plot_color)
        if custom_css:
            return custom_css
        use_palette = self._ensure_auto_target_color_palette(palette)
        if not use_palette:
            return self._theme_color("accent_primary", "#4da3ff")
        key = self._target_color_key(target)
        auto_map = getattr(self, "_auto_target_color_map", {})
        cached = _normalized_css_color(auto_map.get(key, ""))
        if cached:
            return cached
        color_css = str(use_palette[len(auto_map) % len(use_palette)])
        normalized = _normalized_css_color(color_css) or color_css
        auto_map[key] = str(normalized)
        self._auto_target_color_map = auto_map
        return str(normalized)


    def _airmass_from_altitude(self, altitude_deg: object) -> np.ndarray:
        altitude = np.asarray(altitude_deg, dtype=float)
        airmass = np.full_like(altitude, np.nan, dtype=float)
        valid = np.isfinite(altitude) & (altitude > 0.0)
        if np.any(valid):
            alt_valid = altitude[valid]
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                denom = np.sin(np.radians(alt_valid)) + 0.50572 * np.power(alt_valid + 6.07995, -1.6364)
                airmass[valid] = 1.0 / denom
        return airmass


    def _plot_y_values(self, altitude_deg: object) -> np.ndarray:
        altitude = np.asarray(altitude_deg, dtype=float)
        if not self._plot_airmass:
            return altitude
        return self._airmass_from_altitude(altitude)


    def _plot_limit_value(self) -> float:
        limit_alt = float(self.limit_spin.value())
        if not self._plot_airmass:
            return limit_alt
        limit_airmass = self._airmass_from_altitude(np.array([limit_alt], dtype=float))
        value = _safe_float(limit_airmass[0])
        return value if value is not None else 1.0


    def _visibility_time_window(
        self,
        data: dict,
        tz,
    ) -> tuple[datetime, datetime, dict[str, datetime]]:
        event_map: dict[str, datetime] = {}
        for key in (
            "sunset",
            "dusk_civ",
            "dusk_naut",
            "dusk",
            "dawn",
            "dawn_naut",
            "dawn_civ",
            "sunrise",
            "moonrise",
            "moonset",
        ):
            try:
                event_map[key] = mdates.num2date(data[key]).astimezone(tz)
            except Exception:
                continue

        sunset_dt = event_map.get("sunset")
        sunrise_dt = event_map.get("sunrise")
        if isinstance(sunset_dt, datetime) and isinstance(sunrise_dt, datetime):
            start_dt = sunset_dt - timedelta(hours=1)
            end_dt = sunrise_dt + timedelta(hours=1)
            if end_dt > start_dt:
                return start_dt, end_dt, event_map

        obs_date = self.date_edit.date()
        start_noon_naive = datetime(obs_date.year(), obs_date.month(), obs_date.day(), 12, 0)
        next_date = obs_date.addDays(1)
        end_noon_naive = datetime(next_date.year(), next_date.month(), next_date.day(), 12, 0)
        try:
            start_dt = tz.localize(start_noon_naive)
            end_dt = tz.localize(end_noon_naive)
        except Exception:
            center_dt = mdates.num2date(data["midnight"]).astimezone(tz)
            start_dt = center_dt - timedelta(hours=12)
            end_dt = center_dt + timedelta(hours=12)
        return start_dt, end_dt, event_map


    def _visibility_grid_color(self, *, alpha: float = 1.0) -> str:
        base = self._theme_qcolor("plot_grid", "#2f4666")
        panel = self._theme_qcolor("plot_panel_bg", "#162334")
        softened = self._mix_qcolors(base, panel, 0.34)
        return self._qcolor_rgba_css(softened, alpha)


    def _visibility_grid_rgba(self, *, alpha: float = 1.0) -> tuple[float, float, float, float]:
        base = self._theme_qcolor("plot_grid", "#2f4666")
        panel = self._theme_qcolor("plot_panel_bg", "#162334")
        softened = self._mix_qcolors(base, panel, 0.34)
        return self._qcolor_rgba_mpl(softened, alpha)


    def _visibility_context_key_from_parts(
        self,
        *,
        site_name: str,
        latitude: float,
        longitude: float,
        elevation: float,
        obs_date,
        time_samples: int,
        limit_altitude: float,
    ) -> str:
        return (
            f"{site_name}|{latitude:.6f}|{longitude:.6f}|{elevation:.1f}|"
            f"{obs_date.toString('yyyy-MM-dd')}|{int(time_samples)}|{float(limit_altitude):.1f}"
        )


    def _current_visibility_context_key(self) -> str:
        try:
            return self._visibility_context_key_from_parts(
                site_name=str(self.obs_combo.currentText()),
                latitude=self._read_site_float(self.lat_edit),
                longitude=self._read_site_float(self.lon_edit),
                elevation=self._read_site_float(self.elev_edit),
                obs_date=self.date_edit.date(),
                time_samples=self.settings.value("general/timeSamples", 240, type=int),
                limit_altitude=float(self.limit_spin.value()),
            )
        except Exception:
            return ""


    def _show_visibility_matplotlib_placeholder(self, title: str, message: str) -> None:
        if getattr(self, "_use_visibility_web", False):
            return
        if getattr(self, "ax_alt", None) is None or getattr(self, "plot_canvas", None) is None:
            return
        plot_bg = self._theme_color("plot_bg", "#0f1825")
        plot_panel_bg = self._theme_color("plot_panel_bg", "#162334")
        plot_text = self._theme_color("plot_text", "#d7e4f0")
        self.ax_alt.clear()
        self.plot_canvas.figure.patch.set_facecolor(plot_bg)
        self.ax_alt.set_facecolor(plot_panel_bg)
        self.ax_alt.set_xticks([])
        self.ax_alt.set_yticks([])
        for spine in self.ax_alt.spines.values():
            spine.set_visible(False)
        self.ax_alt.text(
            0.5,
            0.56,
            title,
            transform=self.ax_alt.transAxes,
            ha="center",
            va="center",
            color=plot_text,
            fontsize=15,
            fontweight="bold",
        )
        self.ax_alt.text(
            0.5,
            0.46,
            message,
            transform=self.ax_alt.transAxes,
            ha="center",
            va="center",
            color=self._theme_color("section_hint", plot_text),
            fontsize=10,
            wrap=True,
        )
        self.plot_canvas.draw_idle()


    def _begin_visibility_refresh(self, message: str) -> None:
        self._visibility_web_has_content = False
        if not bool(getattr(self, "_visibility_plot_widgets_ready", False)):
            self._ensure_visibility_plot_placeholder_message(message)
        if getattr(self, "visibility_web", None) is not None:
            self.visibility_web.setEnabled(False)
            self._set_visibility_loading_state(message, visible=True)
        self._show_visibility_matplotlib_placeholder("Visibility Plot", message)


    def _configure_main_plot_y_axis(self) -> None:
        if not self._plot_airmass:
            self.ax_alt.set_ylabel("Altitude (°)")
            self.ax_alt.set_ylim(0, 90)
            self.ax_alt.set_yticks([0, 15, 30, 45, 60, 75, 90])
            return

        self.ax_alt.set_ylabel("Airmass")
        self.ax_alt.set_ylim(VISIBILITY_AIRMASS_Y_MAX, VISIBILITY_AIRMASS_Y_MIN)
        self.ax_alt.set_yticks(VISIBILITY_AIRMASS_Y_TICKS)
        self.ax_alt.set_yticklabels([f"{tick:.1f}" for tick in VISIBILITY_AIRMASS_Y_TICKS])


    def _update_plot_mode_label_metrics(self) -> None:
        if not hasattr(self, "plot_mode_alt_label") or not hasattr(self, "plot_mode_airmass_label"):
            return
        for label, text, align in (
            (self.plot_mode_alt_label, "Altitude", Qt.AlignRight | Qt.AlignVCenter),
            (self.plot_mode_airmass_label, "Airmass", Qt.AlignLeft | Qt.AlignVCenter),
        ):
            metrics_font = QFont(label.font())
            metrics_font.setWeight(QFont.Weight.Bold)
            width = QFontMetrics(metrics_font).horizontalAdvance(text) + 28
            label.setMinimumWidth(max(width, 102))
            label.setAlignment(align)
        if hasattr(self, "plot_mode_widget"):
            switch_width = 70
            if hasattr(self, "airmass_toggle_btn") and self.airmass_toggle_btn is not None:
                switch_width = max(switch_width, self.airmass_toggle_btn.minimumSizeHint().width())
            self.plot_mode_widget.setMinimumWidth(
                self.plot_mode_alt_label.minimumWidth() + self.plot_mode_airmass_label.minimumWidth() + switch_width + 52
            )
            self.plot_mode_widget.setMinimumHeight(44)


    def _animate_plot_mode_switch(self) -> None:
        labels = (
            (getattr(self, "plot_mode_alt_label", None), not self._plot_airmass),
            (getattr(self, "plot_mode_airmass_label", None), self._plot_airmass),
        )
        for anim in self._plot_mode_animations:
            try:
                anim.stop()
            except Exception:
                pass
        self._plot_mode_animations.clear()
        for label, active in labels:
            effect = getattr(label, "_opacity_effect", None)
            if label is None or effect is None:
                continue
            target = 1.0 if active else 0.68
            anim = QPropertyAnimation(effect, b"opacity", self._planner)
            anim.setDuration(180)
            anim.setStartValue(effect.opacity())
            anim.setEndValue(target)
            anim.setEasingCurve(QEasingCurve.OutCubic)
            anim.finished.connect(lambda a=anim: self._plot_mode_animations.remove(a) if a in self._plot_mode_animations else None)
            self._plot_mode_animations.append(anim)
            anim.start()


    def _refresh_plot_mode_switch(self) -> None:
        if not hasattr(self, "plot_mode_alt_label") or not hasattr(self, "plot_mode_airmass_label"):
            return
        if hasattr(self, "airmass_toggle_btn") and self.airmass_toggle_btn is not None:
            blocker = QSignalBlocker(self.airmass_toggle_btn)
            self.airmass_toggle_btn.setChecked(bool(self._plot_airmass))
            del blocker
        for label, active in (
            (self.plot_mode_alt_label, not self._plot_airmass),
            (self.plot_mode_airmass_label, self._plot_airmass),
        ):
            font = QFont(label.font())
            font.setWeight(QFont.Weight.Bold if active else QFont.Weight.Medium)
            label.setFont(font)
            _set_label_tone(label, "info" if active else "muted")
            effect = getattr(label, "_opacity_effect", None)
            if effect is not None:
                effect.setOpacity(1.0 if active else 0.68)
        self._update_plot_mode_label_metrics()


    @staticmethod
    def _polar_rgba_array(color: QColor, alphas: np.ndarray) -> np.ndarray:
        if not color.isValid():
            color = QColor("#59f3ff")
        red = color.redF()
        green = color.greenF()
        blue = color.blueF()
        rows = []
        for alpha in alphas:
            rows.append((red, green, blue, float(max(0.0, min(1.0, alpha)))))
        return np.array(rows, dtype=float) if rows else np.empty((0, 4), dtype=float)


    def _ensure_radar_echo_artists(self, count: int) -> None:
        artists = list(getattr(self, "_radar_echo_artists", []))
        while len(artists) < count:
            artist, = self.polar_ax.plot(
                [],
                [],
                linestyle="",
                marker="o",
                markersize=1.5,
                markeredgewidth=1.35,
                alpha=0.0,
                zorder=4,
            )
            artists.append(artist)
        while len(artists) > count:
            artist = artists.pop()
            try:
                artist.remove()
            except Exception:
                pass
        self._radar_echo_artists = artists


    def _build_radar_sweep_cmap(self) -> LinearSegmentedColormap:
        base = self._theme_qcolor("accent_secondary_soft", "#d38cff")
        r = float(base.redF())
        g = float(base.greenF())
        b = float(base.blueF())
        return LinearSegmentedColormap.from_list(
            "astroplanner_radar_sweep",
            [
                (0.0, (r, g, b, 0.0)),
                (0.28, (r, g, b, 0.04)),
                (0.68, (r, g, b, 0.20)),
                (1.0, (r, g, b, 0.54)),
            ],
            N=512,
        )


    @staticmethod
    def _radar_sector_vertices(theta_start: float, theta_end: float, outer_radius: float = 90.0, samples: int = 12) -> np.ndarray:
        arc_thetas = np.linspace(theta_start, theta_end, max(4, int(samples)))
        vertices: list[tuple[float, float]] = [(theta_start, 0.0)]
        vertices.extend((float(theta), float(outer_radius)) for theta in arc_thetas)
        vertices.append((theta_end, 0.0))
        vertices.append((theta_start, 0.0))
        return np.array(vertices, dtype=float)


    def _refresh_radar_sweep_artists(self, *, redraw: bool = True, delta_s: Optional[float] = None) -> None:
        if (
            not hasattr(self, "radar_sweep_line")
            or not hasattr(self, "polar_ax")
        ):
            return
        enabled = bool(getattr(self, "_radar_sweep_enabled", False))
        if not enabled:
            self.radar_sweep_line.set_data([], [])
            self.radar_sweep_glow_line.set_data([], [])
            if hasattr(self, "radar_sweep_core"):
                self.radar_sweep_core.set_offsets(np.empty((0, 2)))
            if hasattr(self, "radar_sweep_mesh") and self.radar_sweep_mesh is not None:
                self.radar_sweep_mesh.set_array(np.zeros_like(self._radar_sweep_mesh_values).ravel())
                self.radar_sweep_mesh.set_visible(False)
            self._ensure_radar_echo_artists(0)
            self._radar_echo_strengths = np.zeros(0, dtype=float)
            self.polar_scatter.set_alpha(0.52)
            if hasattr(self, "selected_scatter"):
                self.selected_scatter.set_alpha(1.0)
            if redraw and hasattr(self, "polar_canvas"):
                self.polar_canvas.draw_idle()
            return

        theta = float(getattr(self, "_radar_sweep_angle", 0.0)) % (2.0 * math.pi)
        self.radar_sweep_line.set_data([theta, theta], [0.0, 90.0])
        self.radar_sweep_glow_line.set_data([theta, theta], [0.0, 90.0])
        if hasattr(self, "radar_sweep_core"):
            self.radar_sweep_core.set_offsets(np.empty((0, 2)))
        self.polar_scatter.set_alpha(0.0)
        if hasattr(self, "selected_scatter"):
            self.selected_scatter.set_alpha(1.0)
        if hasattr(self, "radar_sweep_mesh") and self.radar_sweep_mesh is not None:
            centers = getattr(self, "_radar_sweep_theta_centers", np.empty(0, dtype=float))
            if isinstance(centers, np.ndarray) and centers.size:
                trail_extent = math.pi / 2.05
                deltas = (theta - centers) % (2.0 * math.pi)
                strengths_1d = np.zeros_like(centers)
                mask = deltas <= trail_extent
                if np.any(mask):
                    normalized = 1.0 - (deltas[mask] / trail_extent)
                    strengths_1d[mask] = np.power(normalized, 1.28)
                radial_rows = len(self._radar_sweep_radius_edges) - 1
                mesh_values = np.repeat(strengths_1d[np.newaxis, :], radial_rows, axis=0)
                self._radar_sweep_mesh_values = mesh_values
                self.radar_sweep_mesh.set_array(mesh_values.ravel())
                self.radar_sweep_mesh.set_visible(bool(np.any(mesh_values > 0.0005)))
        coords = getattr(self, "_radar_target_coords", np.empty((0, 2)))
        strengths = getattr(self, "_radar_echo_strengths", np.zeros(0, dtype=float))
        if not isinstance(strengths, np.ndarray) or strengths.shape[0] != (coords.shape[0] if isinstance(coords, np.ndarray) else 0):
            strengths = np.zeros(coords.shape[0] if isinstance(coords, np.ndarray) else 0, dtype=float)
        if delta_s is not None and strengths.size:
            speed_multiplier = max(0.4, min(2.6, float(getattr(self, "_radar_sweep_speed", 140)) / 100.0))
            revolution_s = 1.0 / speed_multiplier
            decay_tau = max(0.06, revolution_s / 6.0)
            strengths *= math.exp(-max(0.0, float(delta_s)) / decay_tau)
        if isinstance(coords, np.ndarray) and coords.size:
            deltas = np.abs(np.angle(np.exp(1j * (coords[:, 0] - theta))))
            sweep_width = np.deg2rad(8.5)
            mask = deltas <= sweep_width
            if np.any(mask):
                strengths[mask] = np.maximum(strengths[mask], 1.0)
            visible = strengths > 0.0002
            if np.any(visible):
                offsets = coords[visible]
                vis_strength = strengths[visible]
                main_sizes = 1.8 + (np.power(vis_strength, 0.78) * 3.6)
                main_alphas = np.clip(np.power(vis_strength, 1.55) * 0.52, 0.0, 0.52)
                echo_color = self._theme_qcolor("polar_target", "#59f3ff")
                self._ensure_radar_echo_artists(len(offsets))
                for idx, ((theta_value, radius_value), size_value, alpha_value) in enumerate(
                    zip(offsets, main_sizes, main_alphas)
                ):
                    artist = self._radar_echo_artists[idx]
                    artist.set_data([float(theta_value)], [float(radius_value)])
                    artist.set_markersize(float(size_value))
                    artist.set_markeredgewidth(1.05 + (float(alpha_value) * 1.35))
                    artist.set_color(self._qcolor_rgba_mpl(echo_color, float(alpha_value)))
                    artist.set_alpha(float(alpha_value))
                for artist in self._radar_echo_artists[len(offsets):]:
                    artist.set_data([], [])
                    artist.set_alpha(0.0)
            else:
                for artist in getattr(self, "_radar_echo_artists", []):
                    artist.set_data([], [])
                    artist.set_alpha(0.0)
        else:
            self._ensure_radar_echo_artists(0)
        self._radar_echo_strengths = strengths
        if redraw and hasattr(self, "polar_canvas"):
            self.polar_canvas.draw_idle()


    def _update_radar_sweep_state(self) -> None:
        enabled = bool(getattr(self, "_radar_sweep_enabled", False))
        timer = getattr(self, "_radar_sweep_timer", None)
        if timer is None:
            return
        if enabled:
            if hasattr(self, "_radar_sweep_clock"):
                try:
                    self._radar_sweep_clock.start()
                except Exception:
                    pass
            if not timer.isActive():
                timer.start()
        else:
            timer.stop()
            self._radar_sweep_clock.invalidate()
        self._refresh_radar_sweep_artists(redraw=True)


    def _advance_radar_sweep(self) -> None:
        elapsed_ms = 16.0
        if hasattr(self, "_radar_sweep_clock"):
            if not self._radar_sweep_clock.isValid():
                self._radar_sweep_clock.start()
            elapsed_ms = float(self._radar_sweep_clock.restart())
        delta_s = max(1.0 / 240.0, min(0.05, elapsed_ms / 1000.0))
        speed_multiplier = max(0.4, min(2.6, float(getattr(self, "_radar_sweep_speed", 140)) / 100.0))
        degrees_per_second = 360.0 * speed_multiplier
        self._radar_sweep_angle = (
            float(getattr(self, "_radar_sweep_angle", 0.0)) + math.radians(degrees_per_second * delta_s)
        ) % (2.0 * math.pi)
        self._refresh_radar_sweep_artists(redraw=True, delta_s=delta_s)


    def _reset_plot_navigation_home(self) -> None:
        toolbar = getattr(self, "plot_toolbar", None)
        if toolbar is None:
            return
        try:
            toolbar.update()
            toolbar.push_current()
        except Exception:
            pass


    def _refresh_visibility_matplotlib_mode_only(self, data: Optional[dict] = None) -> None:
        payload = data if isinstance(data, dict) else getattr(self, "last_payload", None)
        if not isinstance(payload, dict):
            return
        self._ensure_visibility_plot_widgets()
        if getattr(self, "ax_alt", None) is None or getattr(self, "plot_canvas", None) is None:
            return
        times = payload.get("times")
        if not isinstance(times, np.ndarray) and not isinstance(times, list):
            return
        sample_count = len(times)
        limit = float(self.limit_spin.value())
        sun_alt_series = np.array(payload.get("sun_alt", np.full(sample_count, np.nan)), dtype=float)
        sun_alt_limit = self._sun_alt_limit()
        obs_sun_mask = np.isfinite(sun_alt_series) & (sun_alt_series <= sun_alt_limit)

        base_lines: dict[str, Any] = {}
        high_lines: dict[str, Any] = {}
        for name, line, is_over in list(getattr(self, "vis_lines", [])):
            if is_over:
                high_lines[name] = line
            else:
                base_lines[name] = line

        for tgt in self.targets:
            row = payload.get(tgt.name)
            if not isinstance(row, dict):
                continue
            alt = np.array(row.get("altitude", np.full(sample_count, np.nan)), dtype=float)
            if alt.shape[0] != sample_count:
                continue
            base_line = base_lines.get(tgt.name)
            if base_line is not None:
                alt_vis = np.array(alt, copy=True)
                alt_vis[~(np.isfinite(alt) & (alt > 0.0))] = np.nan
                base_line.set_ydata(self._plot_y_values(alt_vis))
            high_line = high_lines.get(tgt.name)
            if high_line is not None:
                alt_high = np.array(alt, copy=True)
                alt_high[~(np.isfinite(alt) & (alt >= limit) & obs_sun_mask)] = np.nan
                high_line.set_ydata(self._plot_y_values(alt_high))

        if getattr(self, "sun_line", None) is not None and "sun_alt" in payload:
            self.sun_line.set_ydata(self._plot_y_values(payload["sun_alt"]))
        if getattr(self, "moon_line", None) is not None and "moon_alt" in payload:
            self.moon_line.set_ydata(self._plot_y_values(payload["moon_alt"]))
        if getattr(self, "limit_line", None) is not None:
            limit_value = self._plot_limit_value()
            self.limit_line.set_ydata([limit_value, limit_value])
            self.limit_line.set_label("Limit Airmass" if self._plot_airmass else "Limit Altitude")

        self._configure_main_plot_y_axis()
        plot_text = self._theme_color("plot_text", "#d7e4f0")
        self.ax_alt.xaxis.label.set_color(plot_text)
        self.ax_alt.yaxis.label.set_color(plot_text)
        self.ax_alt.tick_params(axis="x", colors=plot_text)
        self.ax_alt.tick_params(axis="y", colors=plot_text)
        self.plot_canvas.draw_idle()


    def _apply_visibility_line_style(self, line: object, *, is_over: bool, is_selected: bool) -> None:
        if line is None:
            return
        try:
            highlight = bool(is_over and is_selected)
            line.set_solid_capstyle("round")
            line.set_solid_joinstyle("round")
            line.set_linewidth(2.8 if highlight else 1.4)
            line.set_alpha(1.0 if highlight else (0.7 if is_over else (0.42 if is_selected else 0.3)))
            line.set_zorder(80 if highlight else (2 if is_over or is_selected else 1))
            if highlight:
                line_color = getattr(line, "get_color", lambda: "#ff4fd8")()

                def glow_color(alpha: float) -> object:
                    try:
                        return to_rgba(line_color, alpha=alpha)
                    except (TypeError, ValueError):
                        return self._qcolor_rgba_mpl(self._theme_qcolor("accent_secondary", "#ff4fd8"), alpha)

                line.set_path_effects(
                    [
                        mpl_patheffects.Stroke(
                            linewidth=38.0,
                            foreground=glow_color(0.160),
                        ),
                        mpl_patheffects.Stroke(
                            linewidth=26.0,
                            foreground=glow_color(0.260),
                        ),
                        mpl_patheffects.Stroke(
                            linewidth=16.0,
                            foreground=glow_color(0.460),
                        ),
                        mpl_patheffects.Stroke(
                            linewidth=8.0,
                            foreground=glow_color(0.900),
                        ),
                        mpl_patheffects.Normal(),
                    ]
                )
            else:
                line.set_path_effects([])
        except Exception:
            return


    def _refresh_target_color_map(self, palette: Optional[list[str]] = None):
        use_palette = palette if palette is not None else self._line_palette()
        self.table_model.color_map.clear()
        for idx, tgt in enumerate(self.targets):
            self.table_model.color_map[tgt.name] = QColor(
                self._target_plot_color_css(tgt, idx, use_palette)
            )


    def _update_plot(self, data: dict, *, sender: object | None = None):
        """Redraw the altitude plot with new data from the worker."""
        logger.info("Altitude plot refresh (%d targets)", len(self.targets))
        self._ensure_visibility_plot_widgets()
        if self.plot_canvas is None or self.ax_alt is None:
            return
        if sender is None:
            sender_method = getattr(self._planner, "sender", None)
            sender = sender_method() if callable(sender_method) else None
        if sender is getattr(self, "worker", None):
            self.worker = None
        payload_key = str(data.get("site_key", "") or "")
        current_key = self._current_visibility_context_key()
        if payload_key and current_key and payload_key != current_key:
            if self._queued_plan_run:
                self._queued_plan_run = False
                QTimer.singleShot(0, self._run_plan)
            return
        self.last_payload = data
        # Keep full visibility data around for polar path plotting
        self.full_payload = data
        # Reset stored visibility lines for this redraw
        self.vis_lines.clear()
        self.ax_alt.clear()
        plot_bg = self._theme_color("plot_bg", "#0f1825")
        plot_panel_bg = self._theme_color("plot_panel_bg", "#162334")
        plot_text = self._theme_color("plot_text", "#d7e4f0")
        plot_grid = self._theme_color("plot_grid", "#2f4666")
        plot_guide = self._theme_color("plot_guide", "#62748a")
        soft_grid_color = self._mix_qcolors(self._theme_qcolor("plot_grid", plot_grid), self._theme_qcolor("plot_panel_bg", plot_panel_bg), 0.34)
        self.plot_canvas.figure.patch.set_facecolor(plot_bg)
        self.ax_alt.set_facecolor(plot_panel_bg)

        # Localise the timezone
        tz = pytz.timezone(data.get("tz", "UTC"))

        # Convert times array to that timezone
        times = [t.astimezone(tz) for t in mdates.num2date(data["times"])]
        times_nums = mdates.date2num(times)
        # Generate line colors from palette, overridden by optional per-target color.
        line_palette = self._line_palette()
        target_colors = [
            self._target_plot_color_css(tgt, idx, line_palette)
            for idx, tgt in enumerate(self.targets)
        ]
        limit = float(self.limit_spin.value())
        sample_hours = 24.0 / max(len(times) - 1, 1)

        self.target_metrics.clear()
        self.target_windows.clear()
        score_vals: list[float] = []
        hour_vals: list[float] = []
        row_enabled: list[bool] = []
        sun_alt_series = np.array(data.get("sun_alt", np.full(len(times), np.nan)), dtype=float)
        sun_alt_limit = self._sun_alt_limit()
        obs_sun_mask = np.isfinite(sun_alt_series) & (sun_alt_series <= sun_alt_limit)
        tz_name = data.get("tz", "UTC")
        site = Site(
            name="",
            latitude=self._read_site_float(self.lat_edit),
            longitude=self._read_site_float(self.lon_edit),
            elevation=self._read_site_float(self.elev_edit),
            limiting_magnitude=self._current_limiting_magnitude(),
        )
        observer_now = Observer(location=site.to_earthlocation(), timezone=tz_name)
        now_dt = datetime.now(pytz.timezone(tz_name))
        # Prepare ephem observer once and use it for both filtering and labels.
        eph_obs = ephem.Observer()
        eph_obs.lat = str(site.latitude)
        eph_obs.lon = str(site.longitude)
        eph_obs.elevation = site.elevation
        eph_obs.date = now_dt
        moon = ephem.Moon(eph_obs)
        moon_coord = SkyCoord(ra=Angle(moon.ra, u.rad), dec=Angle(moon.dec, u.rad))

        # Keep table Name cell colors aligned with plot line colors.
        self._refresh_target_color_map(line_palette)

        for idx, tgt in enumerate(self.targets):
            alt = np.array(data[tgt.name]["altitude"], dtype=float)
            moon_sep_series = np.array(data[tgt.name].get("moon_sep", np.full_like(alt, np.nan)), dtype=float)
            color = target_colors[idx] if idx < len(target_colors) else self._target_plot_color_css(tgt, idx, line_palette)
            metrics = compute_target_metrics(
                altitude_deg=alt,
                moon_sep_deg=moon_sep_series,
                limit_altitude=limit,
                sample_hours=sample_hours,
                priority=tgt.priority,
                observed=tgt.observed,
                valid_mask=obs_sun_mask,
            )
            self.target_metrics[tgt.name] = metrics
            score_vals.append(metrics.score)
            hour_vals.append(metrics.hours_above_limit)

            limit_mask = (alt >= limit) & obs_sun_mask
            if limit_mask.any():
                vis_idx = np.where(limit_mask)[0]
                runs = np.split(vis_idx, np.where(np.diff(vis_idx) != 1)[0] + 1)
                best_run = max(runs, key=len)
                start_idx = int(best_run[0])
                end_idx = min(int(best_run[-1]) + 1, len(times) - 1)
                self.target_windows[tgt.name] = (times[start_idx], times[end_idx])

            moon_sep_now = float(tgt.skycoord.separation(moon_coord).deg)
            passes_filters = self._passes_active_filters(tgt, metrics.score, moon_sep_now)
            row_enabled.append(passes_filters)
            if not passes_filters:
                continue

            # Points above horizon
            vis_mask = np.isfinite(alt) & (alt > 0)
            if not vis_mask.any():
                continue
            # Dashed base path for full visible range.
            # Use NaN outside mask so matplotlib breaks segments instead of drawing
            # straight connector lines through non-visible intervals.
            alt_vis = np.array(alt, copy=True)
            alt_vis[~vis_mask] = np.nan
            plot_alt_vis = self._plot_y_values(alt_vis)
            base_line, = self.ax_alt.plot(
                times_nums, plot_alt_vis,
                color=color, linewidth=1.4,
                linestyle="--", alpha=0.3, zorder=1
            )
            self.vis_lines.append((tgt.name, base_line, False))
            # Solid overlay for portions above limit
            high_mask = np.isfinite(alt) & (alt >= limit) & obs_sun_mask
            if high_mask.any():
                alt_high = np.array(alt, copy=True)
                alt_high[~high_mask] = np.nan
                plot_alt_high = self._plot_y_values(alt_high)
                solid_line, = self.ax_alt.plot(
                    times_nums, plot_alt_high,
                    color=color, linewidth=1.4,
                    linestyle="-", alpha=1.0, zorder=2
                )
                self.vis_lines.append((tgt.name, solid_line, True))

        # ------------------------------------------------------------------
        # Compute and cache current alt, az, sep for each target for the table

        current_alts: list[float] = []
        current_azs: list[float] = []
        current_seps: list[float] = []
        if self.targets:
            try:
                coords_now = SkyCoord(
                    ra=np.array([float(t.ra) for t in self.targets], dtype=float) * u.deg,
                    dec=np.array([float(t.dec) for t in self.targets], dtype=float) * u.deg,
                )
                altaz_now_all = observer_now.altaz(Time(now_dt), coords_now)
                alt_vals = np.array(altaz_now_all.alt.deg, dtype=float)  # type: ignore[arg-type]
                az_vals = np.array(altaz_now_all.az.deg, dtype=float)  # type: ignore[arg-type]
                sep_vals = np.array(coords_now.separation(moon_coord).deg, dtype=float)
                current_alts = [float(value) for value in np.ravel(alt_vals)]
                current_azs = [float(value) for value in np.ravel(az_vals)]
                current_seps = [float(value) for value in np.ravel(sep_vals)]
                if len(current_alts) != len(self.targets):
                    raise ValueError("Unexpected alt/az vector size.")
            except Exception:
                current_alts = []
                current_azs = []
                current_seps = []
                for tgt in self.targets:
                    fixed = FixedTarget(name=tgt.name, coord=tgt.skycoord)
                    altaz_now = observer_now.altaz(Time(now_dt), fixed)
                    current_alts.append(float(altaz_now.alt.deg))  # type: ignore[arg-type]
                    current_azs.append(float(altaz_now.az.deg))  # type: ignore[arg-type]
                    current_seps.append(float(tgt.skycoord.separation(moon_coord).deg))  # type: ignore[arg-type]

        # Assign to model and refresh table
        self.table_model.current_alts = current_alts
        self.table_model.current_azs = current_azs
        self.table_model.current_seps = current_seps
        self.table_model.scores = score_vals
        self.table_model.hours_above_limit = hour_vals
        self.table_model.row_enabled = row_enabled
        self._recompute_recommended_order_cache()
        self._reapply_current_table_sort()
        self._apply_table_row_visibility()
        self._emit_table_data_changed()

        # Compute and display current sun and moon altitudes
        sun_obs = ephem.Sun(eph_obs)
        moon_obs = ephem.Moon(eph_obs)
        sun_alt_curr = sun_obs.alt * 180.0 / math.pi
        moon_alt_curr = moon_obs.alt * 180.0 / math.pi
        self.sun_alt_label.setText(f"{sun_alt_curr:.1f}°")
        self.moon_alt_label.setText(f"{moon_alt_curr:.1f}°")

        # Compute and display sidereal time
        sidereal = Time(now_dt).sidereal_time('apparent', site.to_earthlocation().lon)
        self.sidereal_label.setText(sidereal.to_string(unit=u.hour, sep=":", pad=True, precision=0))

        # ------------------------------------------------------------------
        # Twilight shading (civil, nautical, astronomical), only when valid
        # ------------------------------------------------------------------
        civil_col = self._theme_color("plot_twilight_civil", "#FFF2CC")
        naut_col = self._theme_color("plot_twilight_naut", "#CCE5FF")
        astro_col = self._theme_color("plot_twilight_astro", "#D9D9D9")
        start_dt, end_dt, ev = self._visibility_time_window(data, tz)
        self.ax_alt.set_xlim(start_dt, end_dt)
        xmin, xmax = self.ax_alt.get_xlim()

        # Segments to shade, only if both endpoints exist and start < end, and within window
        segments = [
            ("sunset", "dusk_civ", civil_col),
            ("dusk_civ", "dusk_naut", naut_col),
            ("dusk_naut", "dusk", astro_col),
            ("dawn", "dawn_naut", astro_col),
            ("dawn_naut", "dawn_civ", naut_col),
            ("dawn_civ", "sunrise", civil_col),
        ]
        for start_key, end_key, col in segments:
            if start_key in ev and end_key in ev:
                s_num = mdates.date2num(ev[start_key])
                e_num = mdates.date2num(ev[end_key])
                # Only shade if the segment is within the visible window
                if s_num < e_num and s_num < xmax and e_num > xmin:
                    # Clip to window
                    s_dt = max(float(s_num), float(xmin))
                    e_dt = min(float(e_num), float(xmax))
                    self.ax_alt.axvspan(mdates.num2date(s_dt), mdates.num2date(e_dt),
                                        color=col, alpha=0.4, zorder=0)

        # Guide lines at each valid boundary
        for key, dt in ev.items():
            num = mdates.date2num(dt)
            if xmin <= num <= xmax:
                self.ax_alt.axvline(dt, color=self._qcolor_rgba_mpl(soft_grid_color, 0.24), linestyle="--", alpha=1.0, linewidth=0.9)

        # ------------------------------------------------------------------
        # Red limiting‑altitude line
        # ------------------------------------------------------------------
        limit_line_value = self._plot_limit_value()
        limit_line_label = "Limit Airmass" if self._plot_airmass else "Limit Altitude"
        self.limit_line = self.ax_alt.axhline(
            limit_line_value,
            color=self._theme_color("plot_limit", "#ff5d8f"),
            linestyle="-",
            linewidth=0.5,
            alpha=0.4,
            label=limit_line_label,
        )

        # Reset line references
        self.sun_line = None
        self.moon_line = None

        # Sun altitude curve (always plot, visibility controlled)
        if "sun_alt" in data:
            sun_plot_values = self._plot_y_values(data["sun_alt"])
            self.sun_line, = self.ax_alt.plot(
                times, sun_plot_values,
                color=self._theme_color("plot_sun", "orange"), linewidth=1.2, linestyle='-',
                alpha=0.8, label="Sun"
            )
            self.sun_line.set_visible(self.sun_check.isChecked())

        # Moon altitude curve (always plot, visibility controlled)
        if "moon_alt" in data:
            moon_plot_values = self._plot_y_values(data["moon_alt"])
            self.moon_line, = self.ax_alt.plot(
                times, moon_plot_values,
                color=self._theme_color("plot_moon", "silver"), linewidth=1.2, linestyle='-',
                alpha=0.8, label="Moon"
            )
            self.moon_line.set_visible(self.moon_check.isChecked())

        # Update info panel labels in local time
        fmt = "%Y-%m-%d %H:%M"
        if "sunrise" in ev:
            self.sunrise_label.setText(ev["sunrise"].strftime(fmt))
        else:
            self.sunrise_label.setText("-")
        if "sunset" in ev:
            self.sunset_label.setText(ev["sunset"].strftime(fmt))
        else:
            self.sunset_label.setText("-")
        if "moonrise" in ev:
            self.moonrise_label.setText(ev["moonrise"].strftime(fmt))
        else:
            self.moonrise_label.setText("-")
        if "moonset" in ev:
            self.moonset_label.setText(ev["moonset"].strftime(fmt))
        else:
            self.moonset_label.setText("-")
        # Use cached moon_phase percent
        phase_pct = float(data.get("moon_phase", 0.0))
        phase_value = int(max(0, min(100, round(phase_pct))))
        self.moonphase_bar.setValue(phase_value)
        self.moonphase_bar.setFormat(f"{phase_value}%")
        self._configure_main_plot_y_axis()
        self.ax_alt.set_xlabel("Time (local)")
        self.ax_alt.xaxis.label.set_color(plot_text)
        self.ax_alt.yaxis.label.set_color(plot_text)
        self.ax_alt.tick_params(axis="x", colors=plot_text)
        self.ax_alt.tick_params(axis="y", colors=plot_text)
        for spine in self.ax_alt.spines.values():
            spine.set_color(self._qcolor_css(soft_grid_color))
        self.ax_alt.grid(True, color=self._qcolor_rgba_mpl(soft_grid_color, 0.16), alpha=1.0, linestyle="--", linewidth=0.6)
        # self.ax_alt.legend(loc="upper right")
        # Hour labels in the observer's local timezone
        self.ax_alt.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=tz))
        # Display selected observation date
        date_str = self.date_edit.date().toString("yyyy-MM-dd")
        self.ax_alt.set_title(f"Date: {date_str}", color=plot_text)
        # Current time indicator
        now = datetime.now(tz)
        data["now_local"] = now
        self.now_line = self.ax_alt.axvline(
            float(mdates.date2num(now)),
            color=self._theme_color("plot_now", "magenta"),
            linestyle=":",
            linewidth=1.2,
            label="Now",
        )

        # Update time labels
        # Local time
        self.localtime_label.setText(now.strftime("%Y-%m-%d %H:%M:%S"))
        # UTC time (with globe icon)
        now_utc = datetime.now(timezone.utc)
        self.utctime_label.setText(f"{now_utc.strftime('%Y-%m-%d %H:%M:%S')}")
        self._refresh_weather_window_context()

        # Apply default alpha and width based on altitude limit
        for name, line, is_over in self.vis_lines:
            self._apply_visibility_line_style(line, is_over=is_over, is_selected=False)
        # Highlight selected targets over limit
        sel_rows = [i.row() for i in self.table_view.selectionModel().selectedRows()]
        sel_names = [self.targets[i].name for i in sel_rows]
        for name, line, is_over in self.vis_lines:
            self._apply_visibility_line_style(line, is_over=is_over, is_selected=(name in sel_names))
        visible_targets = sum(1 for flag in row_enabled if flag)
        if self._calc_started_at > 0:
            self._last_calc_stats = CalcRunStats(
                duration_s=max(0.0, perf_counter() - self._calc_started_at),
                visible_targets=visible_targets,
                total_targets=len(self.targets),
            )
            self._calc_started_at = 0.0
        self._refresh_cached_bhtom_suggestions()
        self._update_selected_details()
        self._update_status_bar()
        self._reset_plot_navigation_home()
        self.plot_canvas.draw_idle()
        self._render_visibility_web_plot(data)
        self._update_polar_positions(data)
        # Force one selected-path refresh after a full recompute.
        # Selection signatures are debounced in live selection updates.
        self._last_polar_selection_signature = ()
        self._update_polar_selection(None, None)
        if self._queued_plan_run:
            self._queued_plan_run = False
            QTimer.singleShot(0, self._run_plan)


__all__ = ["VisibilityMatplotlibCoordinator"]
