from __future__ import annotations

import time
from pathlib import Path

from astroplanner.storage import AppStorage, SettingsAdapter


def test_settings_adapter_roundtrips_values(tmp_path: Path) -> None:
    storage = AppStorage(tmp_path)
    settings = SettingsAdapter(storage)

    settings.setValue("general/defaultSite", "WRO")
    settings.setValue("general/showMoonPath", True)
    settings.setValue("general/timeSamples", 240)
    settings.setValue("weather/autoRefreshSec", 180)

    assert settings.value("general/defaultSite", type=str) == "WRO"
    assert settings.value("general/showMoonPath", type=bool) is True
    assert settings.value("general/timeSamples", type=int) == 240
    assert settings.value("weather/autoRefreshSec", type=int) == 180
    assert settings.contains("general/defaultSite") is True
    assert "general/defaultSite" in settings.allKeys()

    settings.remove("general/defaultSite")
    assert settings.contains("general/defaultSite") is False


def test_settings_import_from_mapping_preserves_existing_values(tmp_path: Path) -> None:
    storage = AppStorage(tmp_path)
    settings = SettingsAdapter(storage)
    settings.setValue("general/defaultSite", "Current")

    copied = settings.import_from_source(
        {
            "general/defaultSite": "Legacy",
            "general/uiTheme": "neon",
        }
    )

    assert copied == 1
    assert settings.value("general/defaultSite", type=str) == "Current"
    assert settings.value("general/uiTheme", type=str) == "neon"


def test_cache_repository_persists_assets_and_respects_ttl(tmp_path: Path) -> None:
    storage = AppStorage(tmp_path)
    payload = {"image_bytes": b"abc123", "title": "cloud"}

    storage.cache.set_json("weather", "satellite:test", payload, ttl_s=0.2)
    cached = storage.cache.get_json("weather", "satellite:test")

    assert isinstance(cached, dict)
    assert cached["image_bytes"] == b"abc123"
    assert cached["title"] == "cloud"
    assert any(storage.assets_dir.rglob("*.bin"))

    time.sleep(0.25)
    assert storage.cache.get_json("weather", "satellite:test") is None


def test_observatories_and_session_templates_roundtrip(tmp_path: Path) -> None:
    storage = AppStorage(tmp_path)

    storage.observatories.replace_all(
        [
            {
                "name": "WRO",
                "latitude": 51.1,
                "longitude": 17.0,
                "elevation": 120.0,
                "limiting_magnitude": 18.5,
                "telescope_diameter_mm": 200.0,
                "focal_length_mm": 1000.0,
                "pixel_size_um": 3.76,
                "detector_width_px": 3000,
                "detector_height_px": 2000,
                "custom_conditions_url": "https://example.test/weather",
                "preset_key": "custom",
            }
        ]
    )
    storage.session_templates.replace_all(
        [
            {
                "key": "template-1",
                "name": "Session template",
                "scope": "multi_target",
                "repeat_count": 2,
                "minutes_per_run": 45,
            }
        ]
    )

    observatories = storage.observatories.list_all()
    templates = storage.session_templates.list_all()

    assert len(observatories) == 1
    assert observatories[0]["name"] == "WRO"
    assert observatories[0]["detector_width_px"] == 3000
    assert len(templates) == 1
    assert templates[0]["key"] == "template-1"
    assert templates[0]["repeat_count"] == 2


def test_corrupt_database_is_rotated_and_recreated(tmp_path: Path) -> None:
    db_path = tmp_path / "app.db"
    db_path.write_text("this is not sqlite", encoding="utf-8")

    storage = AppStorage(tmp_path)

    assert storage.db_path.exists()
    assert any(tmp_path.glob("app.db.corrupt.*"))
    settings = SettingsAdapter(storage)
    settings.setValue("general/uiTheme", "retro")
    assert settings.value("general/uiTheme", type=str) == "retro"
