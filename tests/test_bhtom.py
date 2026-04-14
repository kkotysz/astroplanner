from __future__ import annotations

from datetime import datetime, timedelta, timezone

import matplotlib.dates as mdates
import numpy as np
from astropy.time import Time
from astroplan import Observer
from astropy.utils import iers

from astroplanner.bhtom import (
    _extract_bhtom_items,
    _rank_local_target_suggestions_from_candidates,
    dedupe_bhtom_candidates,
)
from astroplanner.models import Site, Target


iers.conf.auto_download = False


def test_extract_bhtom_items_reads_nested_results_payload() -> None:
    payload = {
        "data": {
            "results": [
                {"name": "SN 2026abc", "id": 1},
                {"name": "AT 2026xyz", "id": 2},
            ]
        }
    }

    items = _extract_bhtom_items(payload)

    assert [item["name"] for item in items] == ["SN 2026abc", "AT 2026xyz"]


def test_dedupe_bhtom_candidates_uses_source_object_id_when_present() -> None:
    candidates = [
        {
            "target": Target(
                name="SN 2026abc",
                ra=10.0,
                dec=20.0,
                source_catalog="bhtom",
                source_object_id="bhtom-123",
            )
        },
        {
            "target": Target(
                name="SN 2026abc duplicate",
                ra=10.1,
                dec=20.1,
                source_catalog="bhtom",
                source_object_id="bhtom-123",
            )
        },
        {
            "target": Target(
                name="AT 2026xyz",
                ra=30.0,
                dec=-5.0,
                source_catalog="bhtom",
                source_object_id="bhtom-456",
            )
        },
    ]

    deduped = dedupe_bhtom_candidates(candidates)

    assert [item["target"].name for item in deduped] == ["SN 2026abc", "AT 2026xyz"]


def test_rank_local_target_suggestions_from_candidates_keeps_visible_target() -> None:
    site = Site(name="Equator", latitude=0.0, longitude=0.0, elevation=0.0)
    observer = Observer(location=site.to_earthlocation(), timezone="UTC")

    start = datetime(2026, 4, 13, 0, 0, tzinfo=timezone.utc)
    samples = [start + timedelta(minutes=30 * idx) for idx in range(5)]
    transit_ra = float(observer.local_sidereal_time(Time(samples[2])).deg)
    target = Target(
        name="Visible SN",
        ra=transit_ra,
        dec=0.0,
        source_catalog="bhtom",
        source_object_id="visible-sn-1",
        magnitude=12.4,
        object_type="supernova",
        priority=4,
    )
    payload = {
        "tz": "UTC",
        "times": mdates.date2num(samples),
        "sun_alt": np.full(len(samples), -25.0, dtype=float),
        "moon_ra": np.full(len(samples), (transit_ra + 180.0) % 360.0, dtype=float),
        "moon_dec": np.zeros(len(samples), dtype=float),
    }

    ranked, notes = _rank_local_target_suggestions_from_candidates(
        payload=payload,
        site=site,
        targets=[],
        limit_altitude=20.0,
        sun_alt_limit=-18.0,
        min_moon_sep=30.0,
        candidates=[
            {
                "target": target,
                "importance": 5.0,
                "bhtom_priority": 4,
                "sun_separation": 135.0,
            }
        ],
    )

    assert notes == []
    assert len(ranked) == 1
    result = ranked[0]
    assert result["target"].name == "Visible SN"
    assert result["metrics"].hours_above_limit > 0.0
    assert result["metrics"].score > 0.0
    assert result["best_airmass"] is not None
    assert float(result["best_airmass"]) <= 1.2
    assert result["min_window_moon_sep"] is not None
    assert float(result["min_window_moon_sep"]) >= 150.0
    assert result["moon_sep_warning"] is False
