from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TargetNightMetrics:
    hours_above_limit: float
    max_altitude_deg: float
    peak_moon_sep_deg: float
    score: float


def _clamp(val: float, low: float, high: float) -> float:
    return max(low, min(high, val))


def compute_target_metrics(
    altitude_deg: np.ndarray,
    moon_sep_deg: np.ndarray,
    limit_altitude: float,
    sample_hours: float,
    priority: int,
    observed: bool,
    valid_mask: np.ndarray | None = None,
) -> TargetNightMetrics:
    finite_alt = np.isfinite(altitude_deg)
    finite_sep = np.isfinite(moon_sep_deg)
    if valid_mask is None:
        eff_alt = finite_alt
        eff_sep = finite_sep
    else:
        safe_mask = np.asarray(valid_mask, dtype=bool)
        if safe_mask.shape != altitude_deg.shape:
            safe_mask = np.broadcast_to(safe_mask, altitude_deg.shape)
        eff_alt = finite_alt & safe_mask
        eff_sep = finite_sep & safe_mask

    valid_alt = altitude_deg[eff_alt]
    valid_sep = moon_sep_deg[eff_sep]

    above_limit = eff_alt & (altitude_deg >= limit_altitude)
    hours_above_limit = float(np.count_nonzero(above_limit) * sample_hours)
    max_altitude = float(np.max(valid_alt)) if valid_alt.size else 0.0
    peak_moon_sep = float(np.max(valid_sep)) if valid_sep.size else 0.0

    vis_component = _clamp(hours_above_limit / 6.0, 0.0, 1.0) * 50.0
    alt_component = _clamp((max_altitude - 20.0) / 60.0, 0.0, 1.0) * 30.0
    moon_component = _clamp(peak_moon_sep / 180.0, 0.0, 1.0) * 20.0

    base_score = vis_component + alt_component + moon_component
    priority_mult = 0.7 + (0.3 * _clamp(priority / 5.0, 0.0, 1.0))
    observed_mult = 0.5 if observed else 1.0
    score = round(base_score * priority_mult * observed_mult, 1)

    return TargetNightMetrics(
        hours_above_limit=round(hours_above_limit, 2),
        max_altitude_deg=round(max_altitude, 1),
        peak_moon_sep_deg=round(peak_moon_sep, 1),
        score=score,
    )
