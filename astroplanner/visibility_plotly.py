from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import matplotlib.dates as mdates
import numpy as np
import pytz

from astroplanner.models import Target

try:
    import plotly
    from plotly.io import to_html as plotly_to_html
    import plotly.graph_objects as go

    _plotly_base_dir = Path(plotly.__file__).resolve().parent / "package_data"
    PLOTLY_JS_BASE_DIR = str(_plotly_base_dir) if (_plotly_base_dir / "plotly.min.js").exists() else ""
    HAS_PLOTLY = True
except Exception:  # pragma: no cover - optional runtime dependency
    plotly_to_html = None  # type: ignore[assignment]
    go = None  # type: ignore[assignment]
    PLOTLY_JS_BASE_DIR = ""
    HAS_PLOTLY = False


VISIBILITY_AIRMASS_Y_MIN = 0.9
VISIBILITY_AIRMASS_Y_MAX = 2.1
VISIBILITY_AIRMASS_Y_TICKS = [0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.7, 1.9, 2.1]
PLOTLY_CDN_URL = "https://cdn.plot.ly/plotly-3.4.0.min.js"


@dataclass(frozen=True)
class VisibilityPlotlyRequest:
    data: Mapping[str, Any]
    targets: Sequence[Target]
    row_enabled: Sequence[bool]
    target_colors: Sequence[str]
    limit_altitude: float
    sun_alt_limit: float
    plot_airmass: bool
    show_sun: bool
    show_moon: bool
    date_label: str
    theme_tokens: Mapping[str, object]
    dark_enabled: bool
    start_dt: datetime
    end_dt: datetime
    event_map: Mapping[str, datetime]
    grid_css: str
    guide_css: str
    plot_font: str
    font_face_css: str = ""
    use_local_plotly_js: bool = bool(PLOTLY_JS_BASE_DIR)
    plotly_cdn_url: str = PLOTLY_CDN_URL


def airmass_from_altitude(altitude_deg: object) -> np.ndarray:
    altitude = np.asarray(altitude_deg, dtype=float)
    airmass = np.full_like(altitude, np.nan, dtype=float)
    valid = np.isfinite(altitude) & (altitude > 0.0)
    if np.any(valid):
        alt_valid = altitude[valid]
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            denom = np.sin(np.radians(alt_valid)) + 0.50572 * np.power(alt_valid + 6.07995, -1.6364)
            airmass[valid] = 1.0 / denom
    return airmass


def _plot_y_values(altitude_deg: object, *, plot_airmass: bool) -> np.ndarray:
    altitude = np.asarray(altitude_deg, dtype=float)
    if not plot_airmass:
        return altitude
    return airmass_from_altitude(altitude)


def _plot_limit_value(limit_altitude: float, *, plot_airmass: bool) -> float:
    if not plot_airmass:
        return float(limit_altitude)
    limit_airmass = airmass_from_altitude(np.array([float(limit_altitude)], dtype=float))
    value = limit_airmass[0]
    return float(value) if np.isfinite(value) else 1.0


def build_visibility_plotly_html(
    request: VisibilityPlotlyRequest,
    *,
    now_override: Optional[datetime] = None,
) -> Optional[str]:
    if not (HAS_PLOTLY and plotly_to_html is not None and go is not None):
        return None
    data = request.data
    try:
        tz = pytz.timezone(str(data.get("tz", "UTC") or "UTC"))
    except Exception:
        tz = pytz.UTC
    try:
        times = [t.astimezone(tz) for t in mdates.num2date(data["times"])]
    except Exception:
        return None
    if len(times) < 2:
        return None

    dark = bool(request.dark_enabled)
    use_tokens = request.theme_tokens
    plot_bg = str(use_tokens.get("plot_bg", "#121b29"))
    plot_panel_bg = str(use_tokens.get("plot_panel_bg", plot_bg))
    plot_text = str(use_tokens.get("plot_text", "#d7e4f0" if dark else "#253347"))
    plot_grid = str(use_tokens.get("plot_grid", "#2f4666" if dark else "#d6deea"))
    plot_guide = str(use_tokens.get("plot_guide", plot_grid))
    plot_limit = str(use_tokens.get("plot_limit", "#ff5d8f"))
    plot_now = str(use_tokens.get("plot_now", "#ff3df0"))
    plot_sun = str(use_tokens.get("plot_sun", "#ffb224"))
    plot_moon = str(use_tokens.get("plot_moon", "#d7e2ff"))
    civil_col = str(use_tokens.get("plot_twilight_civil", "#ffd166"))
    naut_col = str(use_tokens.get("plot_twilight_naut", "#5ab6ff"))
    astro_col = str(use_tokens.get("plot_twilight_astro", "#8094c8"))

    limit = float(request.limit_altitude)
    sun_alt_series = np.array(data.get("sun_alt", np.full(len(times), np.nan)), dtype=float)
    obs_sun_mask = np.isfinite(sun_alt_series) & (sun_alt_series <= float(request.sun_alt_limit))
    row_enabled = list(request.row_enabled)
    if len(row_enabled) != len(request.targets):
        row_enabled = [True] * len(request.targets)

    fig = go.Figure()
    y_title = "Airmass" if request.plot_airmass else "Altitude (°)"
    y_precision = ".2f" if request.plot_airmass else ".1f"

    def _visible_series(values: np.ndarray, mask: np.ndarray) -> list[float]:
        series = np.array(values, copy=True, dtype=float)
        series[~mask] = np.nan
        return [float(v) if np.isfinite(v) else np.nan for v in series]

    for idx, tgt in enumerate(request.targets):
        if idx >= len(row_enabled) or not row_enabled[idx]:
            continue
        row = data.get(tgt.name)
        if not isinstance(row, Mapping):
            continue
        alt = np.array(row.get("altitude", np.full(len(times), np.nan)), dtype=float)
        if alt.shape[0] != len(times):
            continue
        color = request.target_colors[idx] if idx < len(request.target_colors) else "#7dd3fc"
        vis_mask = np.isfinite(alt) & (alt > 0.0)
        if not vis_mask.any():
            continue
        base_y = _visible_series(_plot_y_values(alt, plot_airmass=request.plot_airmass), vis_mask)
        fig.add_trace(
            go.Scatter(
                x=times,
                y=base_y,
                mode="lines",
                name=tgt.name,
                showlegend=False,
                hoverinfo="skip",
                line={"color": color, "width": 1.4, "dash": "dash"},
                opacity=0.28,
                meta={"kind": "target", "target": tgt.name, "segment": "base"},
            )
        )

        high_mask = np.isfinite(alt) & (alt >= limit) & obs_sun_mask
        if high_mask.any():
            high_y = _visible_series(_plot_y_values(alt, plot_airmass=request.plot_airmass), high_mask)
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=high_y,
                    mode="lines",
                    name=tgt.name,
                    showlegend=False,
                    line={"color": color, "width": 1.9},
                    opacity=0.92,
                    hovertemplate=f"{tgt.name}<br>%{{x|%H:%M}}<br>{y_title}: %{{y:{y_precision}}}<extra></extra>",
                    meta={"kind": "target", "target": tgt.name, "segment": "high"},
                )
            )

    for start_key, end_key, color in (
        ("sunset", "dusk_civ", civil_col),
        ("dusk_civ", "dusk_naut", naut_col),
        ("dusk_naut", "dusk", astro_col),
        ("dawn", "dawn_naut", astro_col),
        ("dawn_naut", "dawn_civ", naut_col),
        ("dawn_civ", "sunrise", civil_col),
    ):
        start_evt = request.event_map.get(start_key)
        end_evt = request.event_map.get(end_key)
        if start_evt is None or end_evt is None or start_evt >= end_evt:
            continue
        x0 = max(start_evt, request.start_dt)
        x1 = min(end_evt, request.end_dt)
        if x0 >= x1:
            continue
        fig.add_vrect(x0=x0, x1=x1, fillcolor=color, opacity=0.22, line_width=0, layer="below")

    for _key, evt in request.event_map.items():
        if request.start_dt <= evt <= request.end_dt:
            fig.add_vline(x=evt, line={"color": request.guide_css, "width": 1, "dash": "dash"}, opacity=1.0)

    limit_value = _plot_limit_value(limit, plot_airmass=request.plot_airmass)
    fig.add_hline(y=limit_value, line={"color": plot_limit, "width": 1.0}, opacity=0.55)

    if request.show_sun and "sun_alt" in data:
        sun_y = [
            float(v) if np.isfinite(v) else np.nan
            for v in _plot_y_values(data["sun_alt"], plot_airmass=request.plot_airmass)
        ]
        fig.add_trace(
            go.Scatter(
                x=times,
                y=sun_y,
                mode="lines",
                name="Sun",
                showlegend=False,
                line={"color": plot_sun, "width": 1.2},
                opacity=0.82,
                hovertemplate=f"Sun<br>%{{x|%H:%M}}<br>{y_title}: %{{y:{y_precision}}}<extra></extra>",
            )
        )
    if request.show_moon and "moon_alt" in data:
        moon_y = [
            float(v) if np.isfinite(v) else np.nan
            for v in _plot_y_values(data["moon_alt"], plot_airmass=request.plot_airmass)
        ]
        fig.add_trace(
            go.Scatter(
                x=times,
                y=moon_y,
                mode="lines",
                name="Moon",
                showlegend=False,
                line={"color": plot_moon, "width": 1.2},
                opacity=0.84,
                hovertemplate=f"Moon<br>%{{x|%H:%M}}<br>{y_title}: %{{y:{y_precision}}}<extra></extra>",
            )
        )

    now_dt = now_override
    if not isinstance(now_dt, datetime):
        payload_now = data.get("now_local")
        now_dt = payload_now if isinstance(payload_now, datetime) else datetime.now(tz)
    if now_dt.tzinfo is None:
        now_dt = tz.localize(now_dt)
    else:
        now_dt = now_dt.astimezone(tz)
    fig.add_vline(x=now_dt, line={"color": plot_now, "width": 1.4, "dash": "dot"})

    fig.update_layout(
        template="plotly_dark" if dark else "plotly_white",
        title={"text": f"Date: {request.date_label}", "x": 0.5, "xanchor": "center"},
        margin={"l": 56, "r": 20, "t": 48, "b": 52},
        hovermode="x unified",
        paper_bgcolor=plot_bg,
        plot_bgcolor=plot_panel_bg,
        font={"color": plot_text, "family": request.plot_font},
        showlegend=False,
        dragmode="pan",
    )
    fig.update_xaxes(
        range=[request.start_dt, request.end_dt],
        showgrid=True,
        gridcolor=request.grid_css,
        zerolinecolor=request.grid_css,
        color=plot_text,
        tickformat="%H:%M",
        title_text="Time (local)",
    )
    if not request.plot_airmass:
        fig.update_yaxes(
            title_text="Altitude (°)",
            range=[0, 90],
            tickvals=[0, 15, 30, 45, 60, 75, 90],
            showgrid=True,
            gridcolor=request.grid_css,
            zerolinecolor=request.grid_css,
            color=plot_text,
        )
    else:
        fig.update_yaxes(
            title_text="Airmass",
            range=[VISIBILITY_AIRMASS_Y_MAX, VISIBILITY_AIRMASS_Y_MIN],
            tickvals=VISIBILITY_AIRMASS_Y_TICKS,
            ticktext=[f"{tick:.1f}" for tick in VISIBILITY_AIRMASS_Y_TICKS],
            showgrid=True,
            gridcolor=request.grid_css,
            zerolinecolor=request.grid_css,
            color=plot_text,
        )

    html_fragment = plotly_to_html(
        fig,
        include_plotlyjs=False,
        full_html=False,
        config={
            "displaylogo": False,
            "responsive": True,
            "scrollZoom": True,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        },
    )
    html_fragment = html_fragment.replace("<div>", "<div id='plot-host' style='width:100%;height:100%;min-height:100%;'>", 1)
    graph_id_match = re.search(r'<div id="([^"]+)" class="plotly-graph-div"', html_fragment)
    graph_id = graph_id_match.group(1) if graph_id_match else ""
    resize_script = ""
    if graph_id:
        resize_script = (
            "<script>"
            "(function(){"
            f"const gd=document.getElementById('{graph_id}');"
            "if(!gd||!window.Plotly){return;}"
            "const resize=()=>{"
            "gd.style.width='100%';"
            "gd.style.height='100%';"
            "if(window.Plotly&&Plotly.Plots){Plotly.Plots.resize(gd);}"
            "};"
            "window.addEventListener('resize', resize);"
            "if(window.ResizeObserver){new ResizeObserver(resize).observe(document.body);}"
            "setTimeout(resize,0);"
            "setTimeout(resize,120);"
            "})();"
            "</script>"
        )
    html_head = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<style>"
        f"{request.font_face_css}"
        f"html,body,#plot-host{{margin:0;width:100%;height:100%;overflow:hidden;background:{plot_bg};}}"
        f"html,body,#plot-host,.plotly-graph-div{{font-family:{request.plot_font};}}"
        ".plotly-graph-div{width:100%!important;height:100%!important;}"
        "</style>"
    )
    if request.use_local_plotly_js:
        return (
            f"{html_head}<script src='plotly.min.js'></script></head><body>"
            f"{html_fragment}{resize_script}</body></html>"
        )
    return (
        f"{html_head}<script src='{request.plotly_cdn_url}'></script></head><body>"
        f"{html_fragment}{resize_script}</body></html>"
    )


__all__ = [
    "HAS_PLOTLY",
    "PLOTLY_CDN_URL",
    "PLOTLY_JS_BASE_DIR",
    "VISIBILITY_AIRMASS_Y_MAX",
    "VISIBILITY_AIRMASS_Y_MIN",
    "VISIBILITY_AIRMASS_Y_TICKS",
    "VisibilityPlotlyRequest",
    "airmass_from_altitude",
    "build_visibility_plotly_html",
]
