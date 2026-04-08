from __future__ import annotations

import sqlite3
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


def test_cache_repository_replaces_old_assets_for_same_key(tmp_path: Path) -> None:
    storage = AppStorage(tmp_path)

    storage.cache.set_json("weather", "satellite:test", {"image_bytes": b"first"})
    first_asset = next(storage.assets_dir.rglob("*.bin"))

    storage.cache.set_json("weather", "satellite:test", {"image_bytes": b"second"})
    assets = list(storage.assets_dir.rglob("*.bin"))

    assert len(assets) == 1
    assert assets[0] != first_asset
    assert first_asset.exists() is False
    cached = storage.cache.get_json("weather", "satellite:test")
    assert isinstance(cached, dict)
    assert cached["image_bytes"] == b"second"


def test_cache_repository_treats_missing_asset_as_cache_miss(tmp_path: Path) -> None:
    storage = AppStorage(tmp_path)
    storage.cache.set_json("weather", "satellite:test", {"image_bytes": b"abc123"})

    asset = next(storage.assets_dir.rglob("*.bin"))
    asset.unlink()

    assert storage.cache.get_json("weather", "satellite:test") is None
    with storage.connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM cache_entries WHERE namespace = ? AND cache_key = ?",
            ("weather", "satellite:test"),
        ).fetchone()
    assert row is None


def test_cache_repository_prunes_namespace_entries(tmp_path: Path) -> None:
    storage = AppStorage(tmp_path)
    storage.cache.set_json("finder_preview", "a", {"image_bytes": b"a"})
    time.sleep(0.01)
    storage.cache.set_json("finder_preview", "b", {"image_bytes": b"b"})
    time.sleep(0.01)
    storage.cache.set_json("finder_preview", "c", {"image_bytes": b"c"})

    removed = storage.cache.prune_namespace("finder_preview", max_entries=2)

    assert removed == 1
    assert storage.cache.get_json("finder_preview", "a") is None
    assert storage.cache.get_json("finder_preview", "b") is not None
    assert storage.cache.get_json("finder_preview", "c") is not None


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


def test_plans_repository_roundtrips_workspace_and_saved_plans(tmp_path: Path) -> None:
    storage = AppStorage(tmp_path)
    snapshot = {
        "date": "2026-04-08",
        "site_name": "WRO",
        "limit_altitude": 28.0,
        "selected_target_name": "M31",
        "view_preset": "observation",
        "default_sort_column": 7,
    }
    targets = [
        {
            "name": "M31",
            "ra": 10.684,
            "dec": 41.269,
            "priority": 4,
            "observed": False,
            "notes": "Galaxy",
        },
        {
            "name": "M42",
            "ra": 83.822,
            "dec": -5.391,
            "priority": 3,
            "observed": True,
            "plot_color": "#ff5500",
        },
    ]

    workspace = storage.plans.save_workspace(snapshot, targets)
    saved = storage.plans.save_named("Spring session", snapshot, targets)

    loaded_workspace = storage.plans.load_workspace()
    listed = storage.plans.list_saved()
    loaded_saved = storage.plans.load_plan(saved["id"])

    assert workspace["plan_kind"] == "workspace"
    assert loaded_workspace is not None
    assert loaded_workspace["snapshot"]["site_name"] == "WRO"
    assert [target["name"] for target in loaded_workspace["targets"]] == ["M31", "M42"]
    assert len(listed) == 1
    assert listed[0]["name"] == "Spring session"
    assert loaded_saved is not None
    assert loaded_saved["plan_kind"] == "saved"
    assert loaded_saved["snapshot"]["selected_target_name"] == "M31"
    assert loaded_saved["targets"][1]["plot_color"] == "#ff5500"
    assert storage.plans.delete_plan(saved["id"]) is True
    assert storage.plans.list_saved() == []


def test_observation_log_and_chat_history_roundtrip(tmp_path: Path) -> None:
    storage = AppStorage(tmp_path)
    workspace = storage.plans.save_workspace({"site_name": "WRO"}, [])

    storage.observation_log.append(
        target_name="M31",
        target_key="m31",
        target_payload={"name": "M31", "ra": 10.684, "dec": 41.269},
        site_name="WRO",
        site_payload={"name": "WRO", "latitude": 51.1, "longitude": 17.0, "elevation": 120.0},
        notes="Good transparency",
        source="toggle_row",
        plan_id=workspace["id"],
    )
    storage.chat_history.replace_messages(
        workspace["id"],
        [
            {
                "kind": "user",
                "text": "What should I image tonight?",
                "action_targets": [{"name": "M31", "ra": 10.684, "dec": 41.269}],
                "created_at": 123.0,
            },
            {
                "kind": "ai",
                "text": "Try M31 first.",
                "created_at": 124.0,
            },
        ],
    )

    history = storage.observation_log.list_entries()
    messages = storage.chat_history.list_messages(workspace["id"])

    assert len(history) == 1
    assert history[0]["target_name"] == "M31"
    assert history[0]["site_name"] == "WRO"
    assert history[0]["source"] == "toggle_row"
    assert len(messages) == 2
    assert messages[0]["kind"] == "user"
    assert messages[0]["action_targets"][0]["name"] == "M31"
    assert messages[1]["text"] == "Try M31 first."

    storage.chat_history.clear(workspace["id"])
    assert storage.chat_history.list_messages(workspace["id"]) == []


def test_storage_migrates_v1_database_to_latest_schema(tmp_path: Path) -> None:
    db_path = tmp_path / "app.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE schema_migrations (
                version INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                applied_at REAL NOT NULL
            )
            """
        )
        conn.executescript(AppStorage._MIGRATIONS[0][2])
        conn.execute(
            "INSERT INTO schema_migrations (version, name, applied_at) VALUES (1, 'initial', 0)"
        )
        conn.commit()

    storage = AppStorage(tmp_path)

    with storage.connect() as conn:
        plan_columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(plans)").fetchall()
        }
        table_names = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }

    assert "plan_kind" in plan_columns
    assert "last_opened_at" in plan_columns
    assert {"observation_log", "chat_threads", "chat_messages"} <= table_names


def test_corrupt_database_is_rotated_and_recreated(tmp_path: Path) -> None:
    db_path = tmp_path / "app.db"
    db_path.write_text("this is not sqlite", encoding="utf-8")

    storage = AppStorage(tmp_path)

    assert storage.db_path.exists()
    assert any(tmp_path.glob("app.db.corrupt.*"))
    settings = SettingsAdapter(storage)
    settings.setValue("general/uiTheme", "retro")
    assert settings.value("general/uiTheme", type=str) == "retro"
