from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping
from uuid import uuid4


def _utc_timestamp() -> float:
    return datetime.now(timezone.utc).timestamp()


def _serialize_value(value: object) -> tuple[str, str]:
    if isinstance(value, bool):
        value_type = "bool"
    elif isinstance(value, int):
        value_type = "int"
    elif isinstance(value, float):
        value_type = "float"
    elif isinstance(value, str):
        value_type = "str"
    elif value is None:
        value_type = "none"
    elif isinstance(value, list):
        value_type = "list"
    elif isinstance(value, dict):
        value_type = "dict"
    else:
        value_type = type(value).__name__.lower()
    return json.dumps(value, ensure_ascii=False), value_type


def _deserialize_value(value_json: str, value_type: str) -> object:
    try:
        return json.loads(value_json)
    except Exception:
        if value_type == "str":
            return value_json
        return None


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off", "", "none", "null"}:
        return False
    return bool(text)


def _coerce_typed_value(value: object, type_hint: type | None) -> object:
    if type_hint is None:
        return value
    if value is None:
        if type_hint is bool:
            return False
        if type_hint is int:
            return 0
        if type_hint is float:
            return 0.0
        if type_hint is str:
            return ""
        try:
            return type_hint()
        except Exception:
            return None
    try:
        if type_hint is bool:
            return _coerce_bool(value)
        if type_hint is int:
            if isinstance(value, bool):
                return int(value)
            return int(value)
        if type_hint is float:
            if isinstance(value, bool):
                return float(int(value))
            return float(value)
        if type_hint is str:
            return str(value)
        return type_hint(value)
    except Exception:
        return value


def _is_secret_key(key: str) -> bool:
    normalized = str(key or "").strip().lower()
    if not normalized:
        return False
    return any(
        token in normalized
        for token in (
            "token",
            "secret",
            "password",
            "apikey",
            "api_key",
            "accesskey",
            "access_key",
        )
    )


def _normalize_cache_asset_path(namespace: str, cache_key: str, data: bytes) -> str:
    ns = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in namespace.strip().lower()) or "cache"
    digest = hashlib.sha256(
        f"{namespace}\0{cache_key}".encode("utf-8") + bytes(data)
    ).hexdigest()
    return f"{ns}/{digest}.bin"


def _stash_cache_payload(value: object, *, namespace: str, cache_key: str, assets_dir: Path) -> tuple[object, str | None]:
    asset_path: str | None = None

    def _walk(item: object) -> object:
        nonlocal asset_path
        if isinstance(item, dict):
            return {str(key): _walk(val) for key, val in item.items()}
        if isinstance(item, list):
            return [_walk(val) for val in item]
        if isinstance(item, tuple):
            return [_walk(val) for val in item]
        if isinstance(item, (bytes, bytearray)):
            raw = bytes(item)
            relative_path = _normalize_cache_asset_path(namespace, cache_key, raw)
            full_path = assets_dir / relative_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            if not full_path.exists():
                full_path.write_bytes(raw)
            if asset_path is None:
                asset_path = relative_path
            return {"__cache_asset__": relative_path}
        return item

    return _walk(value), asset_path


def _restore_cache_payload(value: object, *, assets_dir: Path) -> object:
    if isinstance(value, dict):
        relative_path = value.get("__cache_asset__")
        if isinstance(relative_path, str) and relative_path.strip():
            try:
                return (assets_dir / relative_path).read_bytes()
            except Exception:
                return b""
        return {str(key): _restore_cache_payload(val, assets_dir=assets_dir) for key, val in value.items()}
    if isinstance(value, list):
        return [_restore_cache_payload(val, assets_dir=assets_dir) for val in value]
    return value


@contextmanager
def _sqlite_connection(db_path: Path) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(str(db_path), timeout=5.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=5000")
        yield conn
    finally:
        conn.close()


class SettingsRepository:
    def __init__(self, storage: "AppStorage") -> None:
        self.storage = storage

    def get(self, key: str, default: object = None, *, type_hint: type | None = None) -> object:
        with self.storage.connect() as conn:
            row = conn.execute(
                "SELECT value_json, value_type FROM app_settings WHERE key = ?",
                (str(key),),
            ).fetchone()
        if row is None:
            return _coerce_typed_value(default, type_hint)
        value = _deserialize_value(str(row["value_json"]), str(row["value_type"]))
        return _coerce_typed_value(value, type_hint)

    def set(self, key: str, value: object, *, is_secret: bool | None = None) -> None:
        value_json, value_type = _serialize_value(value)
        secret = _is_secret_key(str(key)) if is_secret is None else bool(is_secret)
        now = _utc_timestamp()
        with self.storage.connect() as conn, conn:
            conn.execute(
                """
                INSERT INTO app_settings (key, value_json, value_type, is_secret, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value_json = excluded.value_json,
                    value_type = excluded.value_type,
                    is_secret = excluded.is_secret,
                    updated_at = excluded.updated_at
                """,
                (str(key), value_json, value_type, int(secret), now),
            )

    def remove(self, key: str) -> None:
        with self.storage.connect() as conn, conn:
            conn.execute("DELETE FROM app_settings WHERE key = ?", (str(key),))

    def contains(self, key: str) -> bool:
        with self.storage.connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM app_settings WHERE key = ? LIMIT 1",
                (str(key),),
            ).fetchone()
        return row is not None

    def keys(self) -> list[str]:
        with self.storage.connect() as conn:
            rows = conn.execute("SELECT key FROM app_settings ORDER BY key").fetchall()
        return [str(row["key"]) for row in rows]

    def import_from_source(self, source: object, *, overwrite: bool = False) -> int:
        if hasattr(source, "allKeys") and callable(getattr(source, "allKeys")):
            keys = [str(key) for key in getattr(source, "allKeys")()]
            value_getter = getattr(source, "value")
        elif isinstance(source, Mapping):
            keys = [str(key) for key in source.keys()]
            value_getter = source.get
        else:
            return 0
        copied = 0
        for key in keys:
            if not overwrite and self.contains(key):
                continue
            try:
                value = value_getter(key)
            except Exception:
                continue
            self.set(key, value)
            copied += 1
        return copied


class StateRepository:
    def __init__(self, storage: "AppStorage") -> None:
        self.storage = storage

    def get(self, key: str, default: object = None, *, type_hint: type | None = None) -> object:
        with self.storage.connect() as conn:
            row = conn.execute(
                "SELECT value_json, value_type FROM app_state WHERE key = ?",
                (str(key),),
            ).fetchone()
        if row is None:
            return _coerce_typed_value(default, type_hint)
        value = _deserialize_value(str(row["value_json"]), str(row["value_type"]))
        return _coerce_typed_value(value, type_hint)

    def set(self, key: str, value: object) -> None:
        value_json, value_type = _serialize_value(value)
        now = _utc_timestamp()
        with self.storage.connect() as conn, conn:
            conn.execute(
                """
                INSERT INTO app_state (key, value_json, value_type, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value_json = excluded.value_json,
                    value_type = excluded.value_type,
                    updated_at = excluded.updated_at
                """,
                (str(key), value_json, value_type, now),
            )


class CacheRepository:
    def __init__(self, storage: "AppStorage") -> None:
        self.storage = storage

    def get_json(self, namespace: str, cache_key: str) -> object | None:
        now = _utc_timestamp()
        with self.storage.connect() as conn:
            row = conn.execute(
                """
                SELECT payload_json, asset_path, expires_at
                FROM cache_entries
                WHERE namespace = ? AND cache_key = ?
                """,
                (str(namespace), str(cache_key)),
            ).fetchone()
        if row is None:
            return None
        expires_at = row["expires_at"]
        if expires_at is not None and float(expires_at) < now:
            self.delete(namespace, cache_key)
            return None
        payload_json = str(row["payload_json"] or "null")
        try:
            payload = json.loads(payload_json)
        except Exception:
            return None
        return _restore_cache_payload(payload, assets_dir=self.storage.assets_dir)

    def set_json(
        self,
        namespace: str,
        cache_key: str,
        payload: object,
        *,
        ttl_s: float | None = None,
        expires_at: float | None = None,
        etag: str = "",
        version: int = 1,
        last_error: str = "",
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        now = _utc_timestamp()
        effective_expires_at = expires_at
        if effective_expires_at is None and ttl_s is not None:
            effective_expires_at = now + max(0.0, float(ttl_s))
        prepared_payload, asset_path = _stash_cache_payload(
            payload,
            namespace=str(namespace),
            cache_key=str(cache_key),
            assets_dir=self.storage.assets_dir,
        )
        payload_json = json.dumps(prepared_payload, ensure_ascii=False)
        metadata_json = json.dumps(dict(metadata or {}), ensure_ascii=False)
        with self.storage.connect() as conn, conn:
            conn.execute(
                """
                INSERT INTO cache_entries (
                    namespace, cache_key, payload_json, asset_path, fetched_at,
                    expires_at, etag, version, last_error, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(namespace, cache_key) DO UPDATE SET
                    payload_json = excluded.payload_json,
                    asset_path = excluded.asset_path,
                    fetched_at = excluded.fetched_at,
                    expires_at = excluded.expires_at,
                    etag = excluded.etag,
                    version = excluded.version,
                    last_error = excluded.last_error,
                    metadata_json = excluded.metadata_json
                """,
                (
                    str(namespace),
                    str(cache_key),
                    payload_json,
                    asset_path or "",
                    now,
                    effective_expires_at,
                    str(etag or ""),
                    int(version),
                    str(last_error or ""),
                    metadata_json,
                ),
            )

    def delete(self, namespace: str, cache_key: str) -> None:
        with self.storage.connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM cache_entries WHERE namespace = ? AND cache_key = ?",
                (str(namespace), str(cache_key)),
            ).fetchone()
        self._delete_assets_from_row(row["payload_json"] if row is not None else None)
        with self.storage.connect() as conn, conn:
            conn.execute(
                "DELETE FROM cache_entries WHERE namespace = ? AND cache_key = ?",
                (str(namespace), str(cache_key)),
            )

    def purge_expired(self) -> int:
        now = _utc_timestamp()
        with self.storage.connect() as conn:
            rows = conn.execute(
                "SELECT payload_json FROM cache_entries WHERE expires_at IS NOT NULL AND expires_at < ?",
                (now,),
            ).fetchall()
        for row in rows:
            self._delete_assets_from_row(row["payload_json"])
        with self.storage.connect() as conn, conn:
            cur = conn.execute(
                "DELETE FROM cache_entries WHERE expires_at IS NOT NULL AND expires_at < ?",
                (now,),
            )
            return int(cur.rowcount or 0)

    def _delete_assets_from_row(self, payload_json: object) -> None:
        if not isinstance(payload_json, str) or not payload_json:
            return
        try:
            payload = json.loads(payload_json)
        except Exception:
            return
        for relative_path in self._iter_asset_paths(payload):
            try:
                (self.storage.assets_dir / relative_path).unlink(missing_ok=True)
            except Exception:
                continue

    def _iter_asset_paths(self, payload: object) -> Iterator[str]:
        if isinstance(payload, dict):
            marker = payload.get("__cache_asset__")
            if isinstance(marker, str) and marker.strip():
                yield marker
            for value in payload.values():
                yield from self._iter_asset_paths(value)
        elif isinstance(payload, list):
            for value in payload:
                yield from self._iter_asset_paths(value)


class ObservatoriesRepository:
    def __init__(self, storage: "AppStorage") -> None:
        self.storage = storage

    def list_all(self) -> list[dict[str, object]]:
        with self.storage.connect() as conn:
            rows = conn.execute(
                """
                SELECT id, name, latitude, longitude, elevation, limiting_magnitude,
                       telescope_diameter_mm, focal_length_mm, pixel_size_um,
                       detector_width_px, detector_height_px, custom_conditions_url,
                       preset_key, payload_json
                FROM observatories
                WHERE deleted_at IS NULL
                ORDER BY lower(name)
                """
            ).fetchall()
        records: list[dict[str, object]] = []
        for row in rows:
            payload = self._decode_payload(row["payload_json"])
            if not isinstance(payload, dict):
                payload = {}
            payload.update(
                {
                    "id": str(row["id"]),
                    "name": str(row["name"]),
                    "latitude": float(row["latitude"]),
                    "longitude": float(row["longitude"]),
                    "elevation": float(row["elevation"]),
                    "limiting_magnitude": float(row["limiting_magnitude"]),
                    "telescope_diameter_mm": float(row["telescope_diameter_mm"]),
                    "focal_length_mm": float(row["focal_length_mm"]),
                    "pixel_size_um": float(row["pixel_size_um"]),
                    "detector_width_px": int(row["detector_width_px"]),
                    "detector_height_px": int(row["detector_height_px"]),
                    "custom_conditions_url": str(row["custom_conditions_url"]),
                    "preset_key": str(row["preset_key"]),
                }
            )
            records.append(payload)
        return records

    def replace_all(self, records: Iterable[Mapping[str, object]]) -> None:
        now = _utc_timestamp()
        normalized = [self._normalize_record(record) for record in records]
        with self.storage.connect() as conn, conn:
            conn.execute("UPDATE observatories SET deleted_at = ? WHERE deleted_at IS NULL", (now,))
            existing_ids = {
                str(row["name"]).lower(): str(row["id"])
                for row in conn.execute("SELECT id, name FROM observatories")
            }
            for record in normalized:
                obs_id = existing_ids.get(str(record["name"]).lower()) or str(record.get("id") or uuid4())
                payload_json = json.dumps(record, ensure_ascii=False)
                conn.execute(
                    """
                    INSERT INTO observatories (
                        id, name, latitude, longitude, elevation, limiting_magnitude,
                        telescope_diameter_mm, focal_length_mm, pixel_size_um,
                        detector_width_px, detector_height_px, custom_conditions_url,
                        preset_key, payload_json, created_at, updated_at, deleted_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                    ON CONFLICT(id) DO UPDATE SET
                        name = excluded.name,
                        latitude = excluded.latitude,
                        longitude = excluded.longitude,
                        elevation = excluded.elevation,
                        limiting_magnitude = excluded.limiting_magnitude,
                        telescope_diameter_mm = excluded.telescope_diameter_mm,
                        focal_length_mm = excluded.focal_length_mm,
                        pixel_size_um = excluded.pixel_size_um,
                        detector_width_px = excluded.detector_width_px,
                        detector_height_px = excluded.detector_height_px,
                        custom_conditions_url = excluded.custom_conditions_url,
                        preset_key = excluded.preset_key,
                        payload_json = excluded.payload_json,
                        updated_at = excluded.updated_at,
                        deleted_at = NULL
                    """,
                    (
                        obs_id,
                        str(record["name"]),
                        float(record["latitude"]),
                        float(record["longitude"]),
                        float(record["elevation"]),
                        float(record["limiting_magnitude"]),
                        float(record["telescope_diameter_mm"]),
                        float(record["focal_length_mm"]),
                        float(record["pixel_size_um"]),
                        int(record["detector_width_px"]),
                        int(record["detector_height_px"]),
                        str(record["custom_conditions_url"]),
                        str(record["preset_key"]),
                        payload_json,
                        now,
                        now,
                    ),
                )

    def _normalize_record(self, record: Mapping[str, object]) -> dict[str, object]:
        return {
            "id": str(record.get("id") or ""),
            "name": str(record.get("name") or "").strip(),
            "latitude": float(record.get("latitude", 0.0) or 0.0),
            "longitude": float(record.get("longitude", 0.0) or 0.0),
            "elevation": float(record.get("elevation", 0.0) or 0.0),
            "limiting_magnitude": float(record.get("limiting_magnitude", 0.0) or 0.0),
            "telescope_diameter_mm": float(record.get("telescope_diameter_mm", 0.0) or 0.0),
            "focal_length_mm": float(record.get("focal_length_mm", 0.0) or 0.0),
            "pixel_size_um": float(record.get("pixel_size_um", 0.0) or 0.0),
            "detector_width_px": int(record.get("detector_width_px", 0) or 0),
            "detector_height_px": int(record.get("detector_height_px", 0) or 0),
            "custom_conditions_url": str(record.get("custom_conditions_url") or "").strip(),
            "preset_key": str(record.get("preset_key") or "custom").strip() or "custom",
        }

    @staticmethod
    def _decode_payload(value: object) -> object:
        try:
            return json.loads(str(value))
        except Exception:
            return None


class SessionTemplatesRepository:
    def __init__(self, storage: "AppStorage") -> None:
        self.storage = storage

    def list_all(self) -> list[dict[str, object]]:
        with self.storage.connect() as conn:
            rows = conn.execute(
                """
                SELECT id, template_key, name, scope, payload_json
                FROM session_templates
                WHERE deleted_at IS NULL
                ORDER BY lower(name), lower(template_key)
                """
            ).fetchall()
        records: list[dict[str, object]] = []
        for row in rows:
            payload = self._decode_payload(row["payload_json"])
            if not isinstance(payload, dict):
                payload = {}
            payload.update(
                {
                    "id": str(row["id"]),
                    "key": str(row["template_key"]),
                    "name": str(row["name"]),
                    "scope": str(row["scope"]),
                }
            )
            records.append(payload)
        return records

    def replace_all(self, records: Iterable[Mapping[str, object]]) -> None:
        now = _utc_timestamp()
        normalized = [self._normalize_record(record) for record in records if str(record.get("key") or "").strip()]
        with self.storage.connect() as conn, conn:
            conn.execute("UPDATE session_templates SET deleted_at = ? WHERE deleted_at IS NULL", (now,))
            existing_ids = {
                str(row["template_key"]).lower(): str(row["id"])
                for row in conn.execute("SELECT id, template_key FROM session_templates")
            }
            for record in normalized:
                template_id = existing_ids.get(str(record["key"]).lower()) or str(record.get("id") or uuid4())
                payload_json = json.dumps(record, ensure_ascii=False)
                conn.execute(
                    """
                    INSERT INTO session_templates (
                        id, template_key, name, scope, payload_json,
                        created_at, updated_at, deleted_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, NULL)
                    ON CONFLICT(id) DO UPDATE SET
                        template_key = excluded.template_key,
                        name = excluded.name,
                        scope = excluded.scope,
                        payload_json = excluded.payload_json,
                        updated_at = excluded.updated_at,
                        deleted_at = NULL
                    """,
                    (
                        template_id,
                        str(record["key"]),
                        str(record["name"]),
                        str(record["scope"]),
                        payload_json,
                        now,
                        now,
                    ),
                )

    @staticmethod
    def _normalize_record(record: Mapping[str, object]) -> dict[str, object]:
        data = dict(record)
        data["key"] = str(record.get("key") or "").strip()
        data["name"] = str(record.get("name") or "").strip()
        data["scope"] = str(record.get("scope") or "").strip()
        return data

    @staticmethod
    def _decode_payload(value: object) -> object:
        try:
            return json.loads(str(value))
        except Exception:
            return None


class PlansRepository:
    def __init__(self, storage: "AppStorage") -> None:
        self.storage = storage


class AppStorage:
    _MIGRATIONS: list[tuple[int, str, str]] = [
        (
            1,
            "initial",
            """
            CREATE TABLE IF NOT EXISTS app_settings (
                key TEXT PRIMARY KEY,
                value_json TEXT NOT NULL,
                value_type TEXT NOT NULL,
                is_secret INTEGER NOT NULL DEFAULT 0,
                updated_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS app_state (
                key TEXT PRIMARY KEY,
                value_json TEXT NOT NULL,
                value_type TEXT NOT NULL,
                updated_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS cache_entries (
                namespace TEXT NOT NULL,
                cache_key TEXT NOT NULL,
                payload_json TEXT NOT NULL DEFAULT 'null',
                asset_path TEXT NOT NULL DEFAULT '',
                fetched_at REAL NOT NULL,
                expires_at REAL,
                etag TEXT NOT NULL DEFAULT '',
                version INTEGER NOT NULL DEFAULT 1,
                last_error TEXT NOT NULL DEFAULT '',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                PRIMARY KEY (namespace, cache_key)
            );

            CREATE INDEX IF NOT EXISTS idx_cache_entries_expires_at
            ON cache_entries (expires_at);

            CREATE TABLE IF NOT EXISTS observatories (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                elevation REAL NOT NULL DEFAULT 0,
                limiting_magnitude REAL NOT NULL DEFAULT 0,
                telescope_diameter_mm REAL NOT NULL DEFAULT 0,
                focal_length_mm REAL NOT NULL DEFAULT 0,
                pixel_size_um REAL NOT NULL DEFAULT 0,
                detector_width_px INTEGER NOT NULL DEFAULT 0,
                detector_height_px INTEGER NOT NULL DEFAULT 0,
                custom_conditions_url TEXT NOT NULL DEFAULT '',
                preset_key TEXT NOT NULL DEFAULT 'custom',
                payload_json TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                deleted_at REAL
            );

            CREATE INDEX IF NOT EXISTS idx_observatories_deleted
            ON observatories (deleted_at, name);

            CREATE TABLE IF NOT EXISTS session_templates (
                id TEXT PRIMARY KEY,
                template_key TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                scope TEXT NOT NULL DEFAULT '',
                payload_json TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                deleted_at REAL
            );

            CREATE INDEX IF NOT EXISTS idx_session_templates_deleted
            ON session_templates (deleted_at, template_key);

            CREATE TABLE IF NOT EXISTS plans (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                payload_json TEXT NOT NULL DEFAULT '{}',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                deleted_at REAL
            );

            CREATE TABLE IF NOT EXISTS plan_targets (
                id TEXT PRIMARY KEY,
                plan_id TEXT NOT NULL REFERENCES plans(id) ON DELETE CASCADE,
                target_key TEXT NOT NULL,
                sort_order INTEGER NOT NULL DEFAULT 0,
                payload_json TEXT NOT NULL DEFAULT '{}',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                deleted_at REAL
            );
            """,
        ),
    ]

    def __init__(self, base_dir: Path, *, db_name: str = "app.db") -> None:
        self.base_dir = Path(base_dir).expanduser()
        self.db_path = self.base_dir / db_name
        self.assets_dir = self.base_dir / "assets-cache"
        self.settings = SettingsRepository(self)
        self.state = StateRepository(self)
        self.cache = CacheRepository(self)
        self.observatories = ObservatoriesRepository(self)
        self.session_templates = SessionTemplatesRepository(self)
        self.plans = PlansRepository(self)
        self._initialize()

    def _initialize(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._initialize_database()
        except sqlite3.DatabaseError:
            self._recover_from_corruption()
            self._initialize_database()
        self.cache.purge_expired()

    def _initialize_database(self) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    applied_at REAL NOT NULL
                )
                """
            )
            rows = conn.execute("SELECT version FROM schema_migrations").fetchall()
            applied_versions = {int(row["version"]) for row in rows}
            for version, name, script in self._MIGRATIONS:
                if version in applied_versions:
                    continue
                with conn:
                    conn.executescript(script)
                    conn.execute(
                        "INSERT INTO schema_migrations (version, name, applied_at) VALUES (?, ?, ?)",
                        (int(version), str(name), _utc_timestamp()),
                    )

    def _recover_from_corruption(self) -> None:
        if not self.db_path.exists():
            return
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        corrupt_path = self.base_dir / f"{self.db_path.name}.corrupt.{stamp}"
        shutil.move(str(self.db_path), str(corrupt_path))
        for suffix in ("-wal", "-shm"):
            sidecar = self.base_dir / f"{self.db_path.name}{suffix}"
            try:
                sidecar.unlink(missing_ok=True)
            except Exception:
                continue

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        with _sqlite_connection(self.db_path) as conn:
            yield conn


class SettingsAdapter:
    def __init__(self, storage: AppStorage) -> None:
        self.storage = storage

    def value(self, key: str, defaultValue: object = None, *, type: type | None = None) -> object:
        return self.storage.settings.get(key, defaultValue, type_hint=type)

    def setValue(self, key: str, value: object) -> None:
        self.storage.settings.set(key, value)

    def contains(self, key: str) -> bool:
        return self.storage.settings.contains(key)

    def remove(self, key: str) -> None:
        self.storage.settings.remove(key)

    def allKeys(self) -> list[str]:
        return self.storage.settings.keys()

    def sync(self) -> None:
        return None

    def fileName(self) -> str:
        return str(self.storage.db_path)

    def status(self) -> int:
        return 0

    def import_from_source(self, source: object, *, overwrite: bool = False) -> int:
        return self.storage.settings.import_from_source(source, overwrite=overwrite)
