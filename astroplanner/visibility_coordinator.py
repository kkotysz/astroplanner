from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from typing import TYPE_CHECKING, Optional

import ephem
import numpy as np
from PySide6.QtCore import QItemSelectionModel, QObject, QTimer, QUrl
from PySide6.QtGui import QColor
from matplotlib.lines import Line2D

from astroplanner.theme import DEFAULT_UI_THEME
from astroplanner.visibility_plotly import PLOTLY_JS_BASE_DIR

if TYPE_CHECKING:
    from astro_planner import MainWindow
    from astroplanner.models import Target


def _normalized_css_color(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    color = QColor(text)
    if not color.isValid():
        return ""
    return color.name().lower()


class VisibilityCoordinator(QObject):
    SUN_PATH_LINESTYLE = "--"
    MOON_PATH_LINESTYLE = "--"
    POLAR_AXES_HORIZON_RADIUS = 0.494

    def __init__(self, planner: "MainWindow") -> None:
        super().__init__(planner)
        self._planner = planner

    def bind(self) -> None:
        planner = self._planner
        planner._visibility_plot_refresh_timer = QTimer(planner)
        planner._visibility_plot_refresh_timer.setSingleShot(True)
        planner._visibility_plot_refresh_timer.setInterval(0)
        planner._visibility_plot_refresh_timer.timeout.connect(self.flush_visibility_plot_refresh)

        planner._selected_cutout_update_timer = QTimer(planner)
        planner._selected_cutout_update_timer.setSingleShot(True)
        planner._selected_cutout_update_timer.setInterval(90)
        planner._selected_cutout_update_timer.timeout.connect(self.flush_selected_cutout_update)

        planner._pending_selected_cutout_target = None
        planner._last_vis_selected_names = set()
        planner._last_polar_selection_signature = ()
        planner._visibility_web_sync_selection = True
        planner._visibility_web_html_cache = {}

    def selected_target_names(self) -> list[str]:
        planner = self._planner
        selection_model = planner.table_view.selectionModel()
        if selection_model is None:
            return []
        names: list[str] = []
        for index in selection_model.selectedRows():
            row = index.row()
            if 0 <= row < len(planner.targets):
                names.append(str(planner.targets[row].name))
        return names

    def schedule_selected_cutout_update(self, target: Optional["Target"]) -> None:
        planner = self._planner
        planner._pending_selected_cutout_target = target
        if getattr(planner, "_defer_startup_preview_updates", False):
            return
        if hasattr(planner, "_selected_cutout_update_timer"):
            planner._selected_cutout_update_timer.start()
            return
        self.flush_selected_cutout_update()

    def flush_selected_cutout_update(self) -> None:
        planner = self._planner
        if getattr(planner, "_defer_startup_preview_updates", False):
            return
        target = planner._pending_selected_cutout_target
        planner._pending_selected_cutout_target = None
        planner._update_cutout_preview_for_target(target)

    def schedule_visibility_plot_refresh(self, *, delay_ms: int = 0) -> None:
        planner = self._planner
        if not isinstance(getattr(planner, "last_payload", None), dict):
            return
        if getattr(planner, "_use_visibility_web", False):
            planner._ensure_visibility_plot_widgets()
        planner._visibility_plot_refresh_timer.start(max(0, int(delay_ms)))

    def flush_visibility_plot_refresh(self) -> None:
        planner = self._planner
        if isinstance(getattr(planner, "last_payload", None), dict):
            self.render_visibility_web_plot(planner.last_payload)

    def visibility_web_render_signature(
        self,
        data: dict,
        *,
        now_override: Optional[datetime] = None,
    ) -> str:
        planner = self._planner
        current_now = now_override if isinstance(now_override, datetime) else data.get("now_local")
        now_key = (
            current_now.strftime("%Y-%m-%d %H:%M")
            if isinstance(current_now, datetime)
            else ""
        )
        target_signature = [
            (
                str(tgt.name),
                _normalized_css_color(getattr(tgt, "plot_color", "")),
                bool(planner.table_model.row_enabled[idx]) if idx < len(planner.table_model.row_enabled) else True,
            )
            for idx, tgt in enumerate(planner.targets)
        ]
        signature_payload = {
            "context": str(data.get("site_key", "") or planner._current_visibility_context_key()),
            "mode": "airmass" if planner._plot_airmass else "altitude",
            "theme": str(getattr(planner, "_theme_name", DEFAULT_UI_THEME)),
            "dark": bool(getattr(planner, "_dark_enabled", False)),
            "date": planner.date_edit.date().toString("yyyy-MM-dd") if hasattr(planner, "date_edit") else "",
            "sun": bool(planner.sun_check.isChecked()) if hasattr(planner, "sun_check") else True,
            "moon": bool(planner.moon_check.isChecked()) if hasattr(planner, "moon_check") else True,
            "targets": target_signature,
            "now": now_key,
        }
        serialized = json.dumps(signature_payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(serialized.encode("utf-8")).hexdigest()

    def store_visibility_web_html_cache(self, cache_key: str, html: str) -> None:
        planner = self._planner
        cache = getattr(planner, "_visibility_web_html_cache", {})
        cache[str(cache_key)] = str(html)
        while len(cache) > 8:
            cache.pop(next(iter(cache)))
        planner._visibility_web_html_cache = cache

    def render_visibility_web_plot(
        self,
        data: Optional[dict] = None,
        *,
        now_override: Optional[datetime] = None,
    ) -> None:
        planner = self._planner
        payload = data if isinstance(data, dict) else getattr(planner, "last_payload", None)
        if not isinstance(payload, dict):
            return
        if getattr(planner, "_use_visibility_web", False):
            planner._ensure_visibility_plot_widgets()
        web_view = getattr(planner, "visibility_web", None)
        if web_view is None:
            return
        cache = getattr(planner, "_visibility_web_html_cache", {})
        cache_key = self.visibility_web_render_signature(payload, now_override=now_override)
        html = cache.get(cache_key)
        if not html:
            html = planner._build_visibility_plotly_html(payload, now_override=now_override)
            if html:
                self.store_visibility_web_html_cache(cache_key, html)
        if not html:
            planner._visibility_web_has_content = False
            planner._set_visibility_loading_state("Unable to render interactive chart.", visible=True)
            return
        if not planner._visibility_web_has_content:
            planner._set_visibility_loading_state("Rendering interactive chart…", visible=True)
        if PLOTLY_JS_BASE_DIR:
            web_view.setHtml(html, QUrl.fromLocalFile(str(PLOTLY_JS_BASE_DIR) + "/"))
            return
        web_view.setHtml(html)

    def apply_visibility_web_selection_style(self, selected_names: Optional[set[str]] = None) -> None:
        planner = self._planner
        if not bool(getattr(planner, "_use_visibility_web", False)):
            return
        if not bool(getattr(planner, "_visibility_web_has_content", False)):
            return
        web_view = getattr(planner, "visibility_web", None)
        if web_view is None:
            return
        page = web_view.page() if hasattr(web_view, "page") else None
        if page is None:
            return
        if selected_names is None:
            selected_names = set(self.selected_target_names())
        selected_list = sorted(str(name) for name in selected_names if str(name).strip())
        selected_json = json.dumps(selected_list, ensure_ascii=False)
        script = (
            "(function(){"
            f"const selected = new Set({selected_json});"
            "const gd=document.querySelector('.plotly-graph-div');"
            "if(!gd||!window.Plotly||!Array.isArray(gd.data)){return 0;}"
            "const indices=[];const widths=[];const opacities=[];const selectedHighIndices=[];"
            "for(let i=0;i<gd.data.length;i++){"
            "const tr=gd.data[i]||{};const meta=tr.meta;"
            "if(!meta||typeof meta!=='object'||meta.kind!=='target'){continue;}"
            "const name=String(meta.target||'');if(!name){continue;}"
            "const seg=String(meta.segment||'');"
            "const isSelected=selected.has(name);"
            "if(seg==='base'){indices.push(i);widths.push(1.4);opacities.push(isSelected?0.42:0.28);}"
            "else if(seg==='high'){if(isSelected){selectedHighIndices.push(i);}"
            "indices.push(i);widths.push(isSelected?4.1:1.9);opacities.push(isSelected?1.00:0.92);}"
            "}"
            "if(!indices.length){return 0;}"
            "const applyGlow=()=>{"
            "const nodes=Array.from(gd.querySelectorAll('.scatterlayer .trace'));"
            "for(let i=0;i<nodes.length;i++){"
            "const tr=gd.data[i]||{};const meta=tr.meta||{};"
            "const on=meta.kind==='target'&&meta.segment==='high'&&selected.has(String(meta.target||''));"
            "const paths=nodes[i].querySelectorAll('path.js-line');"
            "for(let j=0;j<paths.length;j++){"
            "const path=paths[j];"
            "if(on){"
            "const stroke=path.getAttribute('stroke')||((tr.line||{}).color)||'#59f3ff';"
            "path.style.filter=`drop-shadow(0 0 2px ${stroke}) drop-shadow(0 0 7px ${stroke}) drop-shadow(0 0 15px ${stroke})`;"
            "path.style.strokeLinecap='round';path.style.strokeLinejoin='round';"
            "}else{path.style.filter='';}"
            "}"
            "}"
            "};"
            "let op=Plotly.restyle(gd, {'line.width': widths, 'opacity': opacities}, indices);"
            "if(selectedHighIndices.length&&typeof Plotly.moveTraces==='function'){"
            "op=Promise.resolve(op).then(()=>{"
            "const targetIndices=selectedHighIndices.map((_,j)=>gd.data.length-selectedHighIndices.length+j);"
            "return Plotly.moveTraces(gd, selectedHighIndices, targetIndices);"
            "});"
            "}"
            "Promise.resolve(op).then(()=>requestAnimationFrame(applyGlow)).catch(()=>requestAnimationFrame(applyGlow));"
            "return indices.length;"
            "})();"
        )
        try:
            page.runJavaScript(script)
        except Exception:
            return

    def toggle_visibility(self) -> None:
        planner = self._planner
        if hasattr(planner, "sun_line") and planner.sun_line:
            planner.sun_line.set_visible(planner.sun_check.isChecked())
        if hasattr(planner, "moon_line") and planner.moon_line:
            planner.moon_line.set_visible(planner.moon_check.isChecked())
        plot_canvas = getattr(planner, "plot_canvas", None)
        if plot_canvas is not None:
            plot_canvas.draw_idle()
        if isinstance(getattr(planner, "last_payload", None), dict):
            self.render_visibility_web_plot(planner.last_payload)

    def on_plot_mode_switch_changed(self, checked: bool) -> None:
        planner = self._planner
        if planner._plot_airmass == checked:
            planner._refresh_plot_mode_switch()
            return
        planner._plot_airmass = checked
        planner._refresh_plot_mode_switch()
        planner._animate_plot_mode_switch()
        planner.settings.setValue("general/plotAirmass", planner._plot_airmass)
        if isinstance(planner.last_payload, dict):
            if getattr(planner, "_use_visibility_web", False):
                self.schedule_visibility_plot_refresh()
            else:
                planner._refresh_visibility_matplotlib_mode_only(planner.last_payload)

    def update_polar_selection(self, _selected, _deselected) -> None:
        planner = self._planner
        sel_model = planner.table_view.selectionModel()
        if sel_model is None:
            return
        sel_rows = sorted(idx.row() for idx in sel_model.selectedRows())
        selection_signature = tuple(
            planner.targets[row].name
            for row in sel_rows
            if 0 <= row < len(planner.targets)
        )
        if selection_signature == getattr(planner, "_last_polar_selection_signature", ()):
            return
        planner._last_polar_selection_signature = selection_signature

        sel_coords = []
        for row in sel_rows:
            if row < len(planner.table_model.row_enabled) and not planner.table_model.row_enabled[row]:
                continue
            alt = planner.table_model.current_alts[row] if row < len(planner.table_model.current_alts) else None
            az = planner.table_model.current_azs[row] if row < len(planner.table_model.current_azs) else None
            if alt is None or az is None or alt <= 0:
                continue
            theta = np.deg2rad(az)
            r = 90 - alt
            sel_coords.append((theta, r))
        if sel_coords:
            arr = np.array(sel_coords)
            planner.selected_scatter.set_offsets(arr)
        else:
            planner.selected_scatter.set_offsets(np.empty((0, 2)))

        if not planner.show_obj_path:
            if planner.selected_trace_line:
                try:
                    planner.selected_trace_line.remove()
                except Exception:
                    pass
                planner.selected_trace_line = None
            planner.polar_canvas.draw_idle()
            return

        if planner.selected_trace_line:
            try:
                planner.selected_trace_line.remove()
            except Exception:
                pass
        planner.selected_trace_line = None

        if sel_rows:
            idx0 = sel_rows[0]
            name = planner.targets[idx0].name
            if not hasattr(planner, "full_payload") or not isinstance(planner.full_payload, dict) or name not in planner.full_payload:
                planner.selected_trace_line = None
                planner.polar_canvas.draw_idle()
                return
            alt_arr = np.array(planner.full_payload[name]["altitude"], dtype=float)
            az_arr = np.array(planner.full_payload[name]["azimuth"], dtype=float)
            path_x, path_y = self.build_polar_axes_path(alt_arr, az_arr)
            if path_x.size == 0 or path_y.size == 0:
                planner.selected_trace_line = None
                planner.polar_canvas.draw_idle()
                return
            trace = self.add_polar_axes_path_line(
                path_x,
                path_y,
                color=planner._theme_color("polar_selected_path", "#8cff84"),
                linewidth=0.8,
                linestyle=":",
                alpha=0.7,
                zorder=1,
            )
            planner.selected_trace_line = trace
        planner.polar_canvas.draw_idle()

    def on_polar_pick(self, event) -> None:
        planner = self._planner
        if event.artist is not planner.polar_scatter:
            return
        inds = event.ind
        if not len(inds):
            return
        ptr = inds[0]
        i = planner.polar_indices[ptr]
        sel_model = planner.table_view.selectionModel()
        if sel_model is None:
            return
        sel_model.clearSelection()
        idx = planner.table_model.index(i, 0)
        sel_model.select(idx, QItemSelectionModel.Select | QItemSelectionModel.Rows)
        alt = planner.table_model.current_alts[i]
        az = planner.table_model.current_azs[i]
        theta = np.deg2rad(az)
        r = 90 - alt
        planner.selected_scatter.set_offsets(np.array([[theta, r]]))
        planner.polar_canvas.draw_idle()

    def update_vis_selection(self, _selected, _deselected) -> None:
        planner = self._planner
        sel_model = planner.table_view.selectionModel()
        if sel_model is None:
            return
        sel_rows = [idx.row() for idx in sel_model.selectedRows()]
        sel_names = {
            planner.targets[i].name
            for i in sel_rows
            if 0 <= i < len(planner.targets)
        }
        previous_sel_names = set(getattr(planner, "_last_vis_selected_names", set()))
        if sel_names == previous_sel_names:
            return
        changed_names = sel_names.symmetric_difference(previous_sel_names)
        planner._last_vis_selected_names = set(sel_names)

        for name, line, is_over in planner.vis_lines:
            if name not in changed_names:
                continue
            planner._apply_visibility_line_style(line, is_over=is_over, is_selected=(name in sel_names))
        plot_canvas = getattr(planner, "plot_canvas", None)
        if plot_canvas is not None:
            plot_canvas.draw_idle()
        if (
            bool(getattr(planner, "_visibility_web_sync_selection", False))
            and isinstance(getattr(planner, "last_payload", None), dict)
        ):
            self.apply_visibility_web_selection_style(sel_names)

    @staticmethod
    def _build_visible_polar_segments(
        alt_series: object,
        az_series: object,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        alt_arr = np.array(alt_series, dtype=float).ravel()
        az_arr = np.array(az_series, dtype=float).ravel()
        n = min(alt_arr.size, az_arr.size)
        if n <= 0:
            return []
        alt_arr = alt_arr[:n]
        az_arr = np.mod(az_arr[:n], 360.0)
        finite_idx = np.where(np.isfinite(alt_arr) & np.isfinite(az_arr))[0]
        if finite_idx.size == 0:
            return []

        segments: list[tuple[np.ndarray, np.ndarray]] = []
        runs = np.split(finite_idx, np.where(np.diff(finite_idx) != 1)[0] + 1)
        for run in runs:
            if run.size == 0:
                continue
            theta_run = np.unwrap(np.deg2rad(az_arr[run]))
            alt_run = alt_arr[run]
            current_theta: list[float] = []
            current_alt: list[float] = []

            def append_point(theta: float, alt: float) -> None:
                if current_theta and math.isclose(current_theta[-1], theta) and math.isclose(current_alt[-1], alt):
                    return
                current_theta.append(float(theta))
                current_alt.append(float(np.clip(alt, 0.0, 90.0)))

            def append_crossing(t0: float, a0: float, t1: float, a1: float) -> None:
                denom = a1 - a0
                if math.isclose(float(denom), 0.0):
                    return
                frac = (0.0 - a0) / denom
                if 0.0 <= frac <= 1.0:
                    append_point(t0 + frac * (t1 - t0), 0.0)

            def flush_segment() -> None:
                if len(current_theta) >= 2:
                    segments.append(
                        (
                            np.array(current_theta, dtype=float),
                            np.array(current_alt, dtype=float),
                        )
                    )
                current_theta.clear()
                current_alt.clear()

            if alt_run.size == 1:
                if alt_run[0] > 0.0:
                    append_point(float(theta_run[0]), float(alt_run[0]))
                    flush_segment()
                continue

            for idx in range(1, alt_run.size):
                a0 = float(alt_run[idx - 1])
                a1 = float(alt_run[idx])
                t0 = float(theta_run[idx - 1])
                t1 = float(theta_run[idx])
                v0 = a0 > 0.0
                v1 = a1 > 0.0
                if v0 and v1:
                    if not current_theta:
                        append_point(t0, a0)
                    append_point(t1, a1)
                elif v0 and not v1:
                    if not current_theta:
                        append_point(t0, a0)
                    append_crossing(t0, a0, t1, a1)
                    flush_segment()
                elif not v0 and v1:
                    append_crossing(t0, a0, t1, a1)
                    append_point(t1, a1)
                else:
                    flush_segment()
            flush_segment()

        return segments

    @staticmethod
    def build_polar_visible_path(
        alt_series: object,
        az_series: object,
        *,
        clip_to_horizon: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not clip_to_horizon:
            alt_arr = np.array(alt_series, dtype=float).ravel()
            az_arr = np.array(az_series, dtype=float).ravel()
            n = min(alt_arr.size, az_arr.size)
            if n <= 0:
                return np.array([], dtype=float), np.array([], dtype=float)
            alt_arr = alt_arr[:n]
            az_arr = np.mod(az_arr[:n], 360.0)
            finite_idx = np.where(np.isfinite(alt_arr) & np.isfinite(az_arr))[0]
            if finite_idx.size == 0:
                return np.array([], dtype=float), np.array([], dtype=float)
            theta_segments: list[np.ndarray] = []
            r_segments: list[np.ndarray] = []
            runs = np.split(finite_idx, np.where(np.diff(finite_idx) != 1)[0] + 1)
            for run in runs:
                if run.size == 0:
                    continue
                theta_segments.append(np.unwrap(np.deg2rad(az_arr[run])))
                r_segments.append(90.0 - alt_arr[run])
                theta_segments.append(np.array([np.nan], dtype=float))
                r_segments.append(np.array([np.nan], dtype=float))
            if not theta_segments:
                return np.array([], dtype=float), np.array([], dtype=float)
            return np.concatenate(theta_segments), np.concatenate(r_segments)

        visible_segments = VisibilityCoordinator._build_visible_polar_segments(alt_series, az_series)
        if not visible_segments:
            return np.array([], dtype=float), np.array([], dtype=float)
        theta_segments: list[np.ndarray] = []
        r_segments: list[np.ndarray] = []
        for theta_seg, alt_seg in visible_segments:
            theta_segments.append(theta_seg)
            r_segments.append(90.0 - alt_seg)
            theta_segments.append(np.array([np.nan], dtype=float))
            r_segments.append(np.array([np.nan], dtype=float))
        return np.concatenate(theta_segments), np.concatenate(r_segments)

    @staticmethod
    def build_polar_axes_path(
        alt_series: object,
        az_series: object,
    ) -> tuple[np.ndarray, np.ndarray]:
        visible_segments = VisibilityCoordinator._build_visible_polar_segments(alt_series, az_series)
        if not visible_segments:
            return np.array([], dtype=float), np.array([], dtype=float)
        x_segments: list[np.ndarray] = []
        y_segments: list[np.ndarray] = []
        for theta_seg, alt_seg in visible_segments:
            radius = np.minimum((90.0 - alt_seg) / 180.0, VisibilityCoordinator.POLAR_AXES_HORIZON_RADIUS)
            x_segments.append(0.5 + radius * np.sin(theta_seg))
            y_segments.append(0.5 + radius * np.cos(theta_seg))
            x_segments.append(np.array([np.nan], dtype=float))
            y_segments.append(np.array([np.nan], dtype=float))

        if not x_segments:
            return np.array([], dtype=float), np.array([], dtype=float)
        return np.concatenate(x_segments), np.concatenate(y_segments)

    def add_polar_axes_path_line(
        self,
        x_values: np.ndarray,
        y_values: np.ndarray,
        *,
        color: str,
        linewidth: float,
        linestyle: str,
        alpha: float,
        zorder: float,
        dash_pattern: Optional[tuple[float, ...]] = None,
        gap_alpha: float = 0.0,
    ) -> Line2D:
        planner = self._planner
        line = Line2D(
            x_values,
            y_values,
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=alpha,
            zorder=zorder,
            transform=planner.polar_ax.transAxes,
            clip_on=True,
        )
        line.set_solid_capstyle("round")
        line.set_solid_joinstyle("round")
        line.set_dash_capstyle("round")
        line.set_dash_joinstyle("round")
        if dash_pattern:
            line.set_dashes(dash_pattern)
        if gap_alpha > 0 and hasattr(line, "set_gapcolor"):
            gap_color = QColor(str(color))
            if gap_color.isValid():
                line.set_gapcolor((gap_color.redF(), gap_color.greenF(), gap_color.blueF(), float(gap_alpha)))
        line.set_clip_path(planner.polar_ax.patch)
        planner.polar_ax.add_line(line)
        return line

    def update_polar_positions(self, data: dict, dynamic_only: bool = False) -> None:
        planner = self._planner
        has_sun_path = "sun_alt" in data and "sun_az" in data
        has_moon_path = "moon_alt" in data and "moon_az" in data
        full_refresh = not dynamic_only and (has_sun_path or has_moon_path)

        if "alts" in data and "azs" in data:
            alt_list = data["alts"]
            az_list = data["azs"]
        else:
            alt_list = planner.table_model.current_alts
            az_list = planner.table_model.current_azs

        tgt_coords = []
        planner.polar_indices = []
        for i in range(len(planner.targets)):
            if i >= len(alt_list) or i >= len(az_list):
                continue
            if i < len(planner.table_model.row_enabled) and not planner.table_model.row_enabled[i]:
                continue
            alt = alt_list[i]
            az = az_list[i]
            if alt is None or alt <= 0:
                continue
            tgt_coords.append((np.deg2rad(az), 90 - alt))
            planner.polar_indices.append(i)
        planner._radar_target_coords = np.array(tgt_coords, dtype=float) if tgt_coords else np.empty((0, 2), dtype=float)
        planner.polar_scatter.set_offsets(planner._radar_target_coords if tgt_coords else np.empty((0, 2)))

        sel_coords = []
        for row in planner._selected_rows():
            if row >= len(alt_list) or row >= len(az_list):
                continue
            alt = alt_list[row]
            az = az_list[row]
            if alt is None or alt <= 0:
                continue
            sel_coords.append((np.deg2rad(az), 90 - alt))
        planner.selected_scatter.set_offsets(np.array(sel_coords) if sel_coords else np.empty((0, 2)))

        site = planner.table_model.site
        if site is None:
            planner.sun_marker.set_offsets(np.empty((0, 2)))
            planner.moon_marker.set_offsets(np.empty((0, 2)))
            planner._refresh_radar_sweep_artists(redraw=False)
            planner.polar_canvas.draw_idle()
            return

        now_local = data.get("now_local")
        if now_local is not None:
            eph_obs = ephem.Observer()
            eph_obs.lat = str(site.latitude)
            eph_obs.lon = str(site.longitude)
            eph_obs.elevation = site.elevation
            eph_obs.date = now_local

            sun = ephem.Sun(eph_obs)
            sun_alt = sun.alt * 180.0 / math.pi  # type: ignore[arg-type]
            sun_az = sun.az * 180.0 / math.pi  # type: ignore[arg-type]
            if sun_alt > 0:
                planner.sun_marker.set_offsets(np.array([[np.deg2rad(sun_az), 90 - sun_alt]]))
            else:
                planner.sun_marker.set_offsets(np.empty((0, 2)))

            moon = ephem.Moon(eph_obs)
            moon_alt = moon.alt * 180.0 / math.pi  # type: ignore[arg-type]
            moon_az = moon.az * 180.0 / math.pi  # type: ignore[arg-type]
            if moon_alt > 0:
                planner.moon_marker.set_offsets(np.array([[np.deg2rad(moon_az), 90 - moon_alt]]))
            else:
                planner.moon_marker.set_offsets(np.empty((0, 2)))

        if full_refresh:
            for line_attr in ("sun_path_line", "moon_path_line"):
                line = getattr(planner, line_attr, None)
                if line is None:
                    continue
                try:
                    line.remove()
                except Exception:
                    pass
                setattr(planner, line_attr, None)

            if planner.show_sun_path and has_sun_path:
                path_x, path_y = self.build_polar_axes_path(data["sun_alt"], data["sun_az"])
                if path_x.size > 0 and path_y.size > 0:
                    planner.sun_path_line = self.add_polar_axes_path_line(
                        path_x,
                        path_y,
                        color=planner._theme_color("polar_sun", "gold"),
                        linewidth=1.0,
                        linestyle=self.SUN_PATH_LINESTYLE,
                        alpha=0.75,
                        zorder=2.2,
                    )

            if planner.show_moon_path and has_moon_path:
                path_x, path_y = self.build_polar_axes_path(data["moon_alt"], data["moon_az"])
                if path_x.size > 0 and path_y.size > 0:
                    planner.moon_path_line = self.add_polar_axes_path_line(
                        path_x,
                        path_y,
                        color=planner._theme_color("polar_moon", "silver"),
                        linewidth=1.15,
                        linestyle=self.MOON_PATH_LINESTYLE,
                        alpha=0.88,
                        zorder=2.4,
                    )

            signature = (round(site.latitude, 6), int(planner.limit_spin.value()))
            if getattr(planner, "_polar_static_signature", None) != signature:
                if planner.pole_marker:
                    try:
                        if isinstance(planner.pole_marker, (list, tuple)):
                            for art in planner.pole_marker:
                                art.remove()
                        else:
                            planner.pole_marker.remove()
                    except Exception:
                        pass
                pole_alt = site.latitude if site.latitude >= 0 else -site.latitude
                pole_az = 0.0 if site.latitude >= 0 else 180.0
                r_pol = 90 - pole_alt
                theta_pol = np.deg2rad(pole_az)
                circle = planner.polar_ax.scatter(
                    [theta_pol],
                    [r_pol],
                    facecolors="none",
                    edgecolors=planner._theme_color("polar_pole", "purple"),
                    marker="o",
                    s=80,
                    linewidths=1.5,
                    zorder=3,
                    alpha=0.3,
                )
                dot = planner.polar_ax.scatter(
                    [theta_pol],
                    [r_pol],
                    c=planner._theme_color("polar_pole", "purple"),
                    marker=".",
                    s=30,
                    zorder=4,
                    alpha=0.3,
                )
                planner.pole_marker = (circle, dot)

                if planner.limit_circle:
                    try:
                        planner.limit_circle.remove()
                    except Exception:
                        pass
                lim = planner.limit_spin.value()
                theta_full = np.linspace(0, 2 * math.pi, 200)
                r_full = np.full_like(theta_full, 90 - lim)
                planner.limit_circle, = planner.polar_ax.plot(
                    theta_full,
                    r_full,
                    color=planner._theme_color("polar_limit", "#ff5d8f"),
                    linestyle="-",
                    linewidth=0.5,
                    alpha=0.4,
                )
                planner._polar_static_signature = signature

        planner._clock_polar_tick += 1
        planner._refresh_radar_sweep_artists(redraw=False)
        planner.polar_canvas.draw_idle()


__all__ = ["VisibilityCoordinator"]
