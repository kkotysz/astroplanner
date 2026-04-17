from __future__ import annotations

from datetime import datetime, timedelta, timezone

import matplotlib.dates as mdates
import numpy as np
import pytest

from astroplanner.models import Target
from astroplanner.visibility_plotly import (
    HAS_PLOTLY,
    VisibilityPlotlyRequest,
    build_visibility_plotly_html,
)


def _request() -> VisibilityPlotlyRequest:
    start = datetime(2026, 1, 1, 18, 0, tzinfo=timezone.utc)
    times = [start + timedelta(hours=idx) for idx in range(4)]
    event_map = {
        "sunset": times[0],
        "dusk_civ": times[0] + timedelta(minutes=20),
        "dusk_naut": times[0] + timedelta(minutes=40),
        "dusk": times[1],
        "dawn": times[2],
        "dawn_naut": times[2] + timedelta(minutes=20),
        "dawn_civ": times[2] + timedelta(minutes=40),
        "sunrise": times[3],
    }
    data = {
        "tz": "UTC",
        "times": np.array([mdates.date2num(value) for value in times], dtype=float),
        "sun_alt": np.array([-18.0, -25.0, -22.0, -8.0], dtype=float),
        "moon_alt": np.array([10.0, 12.0, 8.0, 5.0], dtype=float),
        "M31": {"altitude": np.array([12.0, 45.0, 58.0, 20.0], dtype=float)},
        "M42": {"altitude": np.array([5.0, 36.0, 62.0, 35.0], dtype=float)},
    }
    return VisibilityPlotlyRequest(
        data=data,
        targets=[
            Target(name="M31", ra=10.684, dec=41.269),
            Target(name="M42", ra=83.822, dec=-5.391),
        ],
        row_enabled=[True, True],
        target_colors=["#59f3ff", "#ff5d8f"],
        limit_altitude=30.0,
        sun_alt_limit=-10.0,
        plot_airmass=False,
        show_sun=True,
        show_moon=True,
        date_label="2026-01-01",
        theme_tokens={
            "plot_bg": "#101820",
            "plot_panel_bg": "#162334",
            "plot_text": "#d7e4f0",
            "plot_grid": "#2f4666",
            "plot_limit": "#ff5d8f",
            "plot_now": "#ff3df0",
            "plot_sun": "#ffb224",
            "plot_moon": "#d7e2ff",
        },
        dark_enabled=True,
        start_dt=times[0],
        end_dt=times[-1],
        event_map=event_map,
        grid_css="rgba(47,70,102,0.42)",
        guide_css="rgba(47,70,102,0.24)",
        plot_font='"Rajdhani", "Arial"',
        font_face_css="",
        use_local_plotly_js=False,
        plotly_cdn_url="plotly-test.js",
    )


@pytest.mark.skipif(not HAS_PLOTLY, reason="Plotly is optional")
def test_build_visibility_plotly_html_contains_targets_and_resize_script() -> None:
    html = build_visibility_plotly_html(
        _request(),
        now_override=datetime(2026, 1, 1, 19, 30, tzinfo=timezone.utc),
    )

    assert html is not None
    assert "id='plot-host'" in html
    assert "M31" in html
    assert "M42" in html
    assert "Plotly.Plots.resize" in html
    assert "plotly-test.js" in html


def test_build_visibility_plotly_html_rejects_too_short_payload() -> None:
    request = _request()
    data = dict(request.data)
    data["times"] = np.array([data["times"][0]], dtype=float)
    short_request = VisibilityPlotlyRequest(
        **{
            **request.__dict__,
            "data": data,
        }
    )

    assert build_visibility_plotly_html(short_request) is None
