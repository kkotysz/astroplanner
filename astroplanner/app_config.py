from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QCoreApplication, QStandardPaths

from astroplanner.storage import AppStorage, SettingsAdapter


APP_SETTINGS_ORG = "krzkot"
APP_SETTINGS_APP = "AstroPlanner"
APP_SETTINGS_ENV_KEY = "ASTROPLANNER_CONFIG_DIR"
OBSOLETE_APP_SETTINGS_KEYS = (
    "general/customObservatories",
    "general/seestarSessionUserTemplates",
    "general/seestarCampaignUserPresets",
    "general/accentSecondaryColor",
    "__meta/customObservatoriesMigrated",
    "__meta/seestarSessionTemplatesMigrated",
    "__meta/legacyMigrated",
    "__meta/legacyMigratedFromOrg",
    "__meta/legacyIniMigrated",
    "__meta/legacyIniMigratedCount",
    "__meta/legacyIniMigratedFrom",
    "__meta/storageDirMigrated",
    "__meta/storageDirMigratedFrom",
)
OBSOLETE_APP_STATE_KEYS = ("legacy/settings/imported",)


def _config_root_dir() -> Path:
    """Return a stable config root independent from current executable name."""
    generic_cfg = str(QStandardPaths.writableLocation(QStandardPaths.GenericConfigLocation) or "").strip()
    if generic_cfg:
        return Path(generic_cfg).expanduser()
    xdg_cfg = str(os.getenv("XDG_CONFIG_HOME", "") or "").strip()
    if xdg_cfg:
        return Path(xdg_cfg).expanduser()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Preferences"
    return Path.home() / ".config"


def _settings_dir_env_override() -> Optional[Path]:
    env_override = str(os.getenv(APP_SETTINGS_ENV_KEY, "") or "").strip()
    if not env_override:
        return None
    return Path(env_override).expanduser()


def _settings_dir_for_org(org_name: str, *, root_dir: Optional[Path] = None) -> Path:
    root = root_dir if root_dir is not None else _config_root_dir()
    return Path(root).expanduser() / str(org_name).strip() / APP_SETTINGS_APP


def _resolve_settings_dir() -> Path:
    """Return a stable writable directory for app settings."""
    try:
        QCoreApplication.setOrganizationName(APP_SETTINGS_ORG)
        QCoreApplication.setApplicationName(APP_SETTINGS_APP)
    except Exception:
        pass

    env_override = _settings_dir_env_override()
    if env_override is not None:
        return env_override
    return _settings_dir_for_org(APP_SETTINGS_ORG)


def _cleanup_obsolete_settings(storage: AppStorage, settings: SettingsAdapter) -> None:
    for key in OBSOLETE_APP_SETTINGS_KEYS:
        if settings.contains(key):
            settings.remove(key)
    for key in OBSOLETE_APP_STATE_KEYS:
        storage.state.remove(key)


def _create_app_settings() -> SettingsAdapter:
    """Create SQLite-backed settings storage."""
    settings_dir = _resolve_settings_dir()
    try:
        settings_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    storage = AppStorage(settings_dir)
    settings = SettingsAdapter(storage)
    _cleanup_obsolete_settings(storage, settings)
    try:
        settings.setValue("__meta/storageBackend", "sqlite")
    except Exception:
        pass
    return settings


def create_app_settings() -> SettingsAdapter:
    """Public alias for the app settings factory."""
    return _create_app_settings()


def cleanup_obsolete_settings(storage: AppStorage, settings: SettingsAdapter) -> None:
    """Public alias for settings cleanup."""
    _cleanup_obsolete_settings(storage, settings)


__all__ = [
    "APP_SETTINGS_APP",
    "APP_SETTINGS_ENV_KEY",
    "APP_SETTINGS_ORG",
    "OBSOLETE_APP_SETTINGS_KEYS",
    "OBSOLETE_APP_STATE_KEYS",
    "_cleanup_obsolete_settings",
    "_config_root_dir",
    "_create_app_settings",
    "_resolve_settings_dir",
    "_settings_dir_env_override",
    "_settings_dir_for_org",
    "cleanup_obsolete_settings",
    "create_app_settings",
]
