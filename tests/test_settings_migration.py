from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import astro_planner
from astroplanner.storage import AppStorage, SettingsAdapter


def test_cleanup_obsolete_settings_removes_legacy_keys(tmp_path: Path) -> None:
    storage = AppStorage(tmp_path / "settings")
    settings = SettingsAdapter(storage)
    settings.setValue("general/customObservatories", "{\"observatories\":[]}")
    settings.setValue("__meta/legacyMigrated", True)
    settings.setValue("__meta/seestarSessionTemplatesMigrated", True)
    storage.state.set("legacy/settings/imported", True)

    astro_planner._cleanup_obsolete_settings(storage, settings)

    assert settings.contains("general/customObservatories") is False
    assert settings.contains("__meta/legacyMigrated") is False
    assert settings.contains("__meta/seestarSessionTemplatesMigrated") is False
    assert storage.state.get("legacy/settings/imported", False, type_hint=bool) is False


def test_workspace_site_snapshot_does_not_override_existing_observatory() -> None:
    existing = astro_planner.Site(
        name="WRO",
        latitude=51.1000,
        longitude=17.0000,
        elevation=120.0,
        limiting_magnitude=19.5,
        telescope_diameter_mm=250.0,
        focal_length_mm=800.0,
        pixel_size_um=3.76,
        detector_width_px=3000,
        detector_height_px=2000,
        custom_conditions_url="https://example.test/current",
    )
    save_calls: list[object] = []
    refresh_calls: list[object] = []
    dummy = SimpleNamespace(
        observatories={"WRO": existing},
        _observatory_preset_keys={"WRO": "custom"},
        _save_custom_observatories=lambda *args, **kwargs: save_calls.append((args, kwargs)),
        _refresh_observatory_combo=lambda *args, **kwargs: refresh_calls.append((args, kwargs)),
    )

    astro_planner.MainWindow._ensure_named_site_available(
        dummy,
        {
            "name": "WRO",
            "latitude": 10.0,
            "longitude": 20.0,
            "elevation": 30.0,
            "limiting_magnitude": 18.0,
            "telescope_diameter_mm": 50.0,
            "focal_length_mm": 250.0,
            "pixel_size_um": 2.4,
            "detector_width_px": 1000,
            "detector_height_px": 800,
            "custom_conditions_url": "https://example.test/stale",
        },
        preferred_name="WRO",
    )

    restored = dummy.observatories["WRO"]
    assert restored.latitude == existing.latitude
    assert restored.longitude == existing.longitude
    assert restored.elevation == existing.elevation
    assert restored.custom_conditions_url == existing.custom_conditions_url
    assert save_calls == []
    assert refresh_calls == []


def test_table_color_mode_migration_preserves_legacy_text_glow_mode(tmp_path: Path) -> None:
    storage = AppStorage(tmp_path / "settings")
    settings = SettingsAdapter(storage)
    settings.setValue("table/columnSchemaVersion", 4)
    settings.setValue("table/colorMode", "text_glow")

    dummy = SimpleNamespace(settings=settings)

    astro_planner.MainWindow._migrate_table_settings_schema(dummy)

    assert settings.value("table/colorMode", type=str) == "text_glow"
    assert settings.value("table/columnSchemaVersion", type=int) == 5


def test_table_color_mode_migration_preserves_explicit_choice(tmp_path: Path) -> None:
    storage = AppStorage(tmp_path / "settings")
    settings = SettingsAdapter(storage)
    settings.setValue("table/columnSchemaVersion", 4)
    settings.setValue("table/colorMode", "text_glow")
    settings.setValue("table/colorModeExplicit", True)

    dummy = SimpleNamespace(settings=settings)

    astro_planner.MainWindow._migrate_table_settings_schema(dummy)

    assert settings.value("table/colorMode", type=str) == "text_glow"
    assert settings.value("table/columnSchemaVersion", type=int) == 5


def test_persist_ai_messages_keeps_existing_history_until_explicit_clear(tmp_path: Path) -> None:
    storage = AppStorage(tmp_path / "settings")
    workspace = storage.plans.save_workspace({"site_name": "WRO"}, [])
    storage.chat_history.replace_messages(
        workspace["id"],
        [
            {
                "kind": "user",
                "text": "Keep this chat",
                "created_at": 123.0,
            }
        ],
    )

    dummy = SimpleNamespace(
        app_storage=storage,
        _serialize_ai_messages_for_storage=lambda: [],
        _active_plan_storage_id=lambda: workspace["id"],
    )

    astro_planner.MainWindow._persist_ai_messages_to_storage(dummy)
    kept = storage.chat_history.list_messages(workspace["id"])
    assert len(kept) == 1
    assert kept[0]["text"] == "Keep this chat"

    astro_planner.MainWindow._persist_ai_messages_to_storage(dummy, allow_empty_clear=True)
    assert storage.chat_history.list_messages(workspace["id"]) == []


def test_sanitize_ui_font_size_preserves_larger_saved_values() -> None:
    assert astro_planner._sanitize_ui_font_size(20) == 20
    assert astro_planner._sanitize_ui_font_size("24") == 24
    assert astro_planner._sanitize_ui_font_size(99) == astro_planner.UI_FONT_SIZE_MAX
