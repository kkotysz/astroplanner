from __future__ import annotations

from pathlib import Path

import astro_planner
from astroplanner.storage import AppStorage, SettingsAdapter


def test_legacy_storage_dir_is_moved_to_krzkot_root(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv(astro_planner.APP_SETTINGS_ENV_KEY, raising=False)
    monkeypatch.setattr(astro_planner, "_config_root_dir", lambda: tmp_path)

    legacy_dir = astro_planner._settings_dir_for_org(astro_planner.LEGACY_APP_SETTINGS_ORGS[0])
    legacy_storage = AppStorage(legacy_dir)
    legacy_settings = SettingsAdapter(legacy_storage)
    legacy_settings.setValue("general/uiTheme", "retro")
    legacy_storage.cache.set_json("weather", "satellite:test", {"image_bytes": b"abc123"})

    target_dir = astro_planner._settings_dir_for_org(astro_planner.APP_SETTINGS_ORG)
    migrated_from = astro_planner._migrate_legacy_storage_dir(target_dir)

    assert migrated_from == legacy_dir
    assert legacy_dir.exists() is False

    migrated_storage = AppStorage(target_dir)
    migrated_settings = SettingsAdapter(migrated_storage)
    assert migrated_settings.value("general/uiTheme", type=str) == "retro"
    cached = migrated_storage.cache.get_json("weather", "satellite:test")
    assert isinstance(cached, dict)
    assert cached["image_bytes"] == b"abc123"


def test_legacy_storage_dir_migration_does_not_override_existing_target(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv(astro_planner.APP_SETTINGS_ENV_KEY, raising=False)
    monkeypatch.setattr(astro_planner, "_config_root_dir", lambda: tmp_path)

    legacy_dir = astro_planner._settings_dir_for_org(astro_planner.LEGACY_APP_SETTINGS_ORGS[0])
    legacy_storage = AppStorage(legacy_dir)
    SettingsAdapter(legacy_storage).setValue("general/uiTheme", "legacy")

    target_dir = astro_planner._settings_dir_for_org(astro_planner.APP_SETTINGS_ORG)
    target_storage = AppStorage(target_dir)
    SettingsAdapter(target_storage).setValue("general/uiTheme", "current")

    migrated_from = astro_planner._migrate_legacy_storage_dir(target_dir)

    assert migrated_from is None
    assert legacy_dir.exists() is True

    reloaded_target = AppStorage(target_dir)
    assert SettingsAdapter(reloaded_target).value("general/uiTheme", type=str) == "current"


def test_legacy_ini_candidates_include_previous_org_path(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(astro_planner, "_config_root_dir", lambda: tmp_path)

    target_ini = astro_planner._settings_dir_for_org(astro_planner.APP_SETTINGS_ORG) / astro_planner.APP_SETTINGS_FILE_NAME
    candidates = astro_planner._legacy_settings_ini_candidates(target_ini)

    assert (
        tmp_path
        / astro_planner.LEGACY_APP_SETTINGS_ORGS[0]
        / astro_planner.APP_SETTINGS_APP
        / astro_planner.APP_SETTINGS_FILE_NAME
    ) in candidates
