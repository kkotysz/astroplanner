from __future__ import annotations

from types import SimpleNamespace

import pytest

from astroplanner.observatory_coordinator import ObservatoryCoordinator


def test_observatory_coordinator_parses_legacy_payload_aliases() -> None:
    coordinator = ObservatoryCoordinator(SimpleNamespace())

    sites, preset_keys = coordinator.parse_custom_observatories_payload(
        {
            "observatories": [
                {
                    "name": "Backyard",
                    "latitude": "50.061",
                    "longitude": "19.938",
                    "elevation": "220",
                    "limitingMagnitude": "20.1",
                    "telescopeDiameterMm": "80",
                    "focalLengthMm": "400",
                    "pixelSizeUm": "2.9",
                    "detectorWidthPx": "1920",
                    "detectorHeightPx": "1080",
                    "customConditionsUrl": " https://example.com/station.json ",
                    "presetKey": "custom-1",
                }
            ]
        }
    )

    site = sites["Backyard"]
    assert site.latitude == pytest.approx(50.061)
    assert site.longitude == pytest.approx(19.938)
    assert site.elevation == pytest.approx(220.0)
    assert site.limiting_magnitude == pytest.approx(20.1)
    assert site.telescope_diameter_mm == pytest.approx(80.0)
    assert site.focal_length_mm == pytest.approx(400.0)
    assert site.pixel_size_um == pytest.approx(2.9)
    assert site.detector_width_px == 1920
    assert site.detector_height_px == 1080
    assert site.custom_conditions_url == "https://example.com/station.json"
    assert preset_keys["Backyard"] == "custom-1"
