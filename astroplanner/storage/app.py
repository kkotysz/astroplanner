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


_SETTINGS_MISSING = object()


class _CacheAssetMissingError(FileNotFoundError):
    """Raised when a cached payload references an asset that is no longer available."""


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


def _decode_json_object(value: object, default: object) -> object:
    if isinstance(value, str) and value:
        try:
            return json.loads(value)
        except Exception:
            return default
    return default


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
            except Exception as exc:
                raise _CacheAssetMissingError(relative_path) from exc
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
        self._cache: dict[str, object] = {}
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self.storage.connect() as conn:
            rows = conn.execute(
                "SELECT key, value_json, value_type FROM app_settings ORDER BY key"
            ).fetchall()
        self._cache = {
            str(row["key"]): _deserialize_value(str(row["value_json"]), str(row["value_type"]))
            for row in rows
        }
        self._loaded = True

    def get(self, key: str, default: object = None, *, type_hint: type | None = None) -> object:
        self._ensure_loaded()
        value = self._cache.get(str(key), _SETTINGS_MISSING)
        if value is _SETTINGS_MISSING:
            return _coerce_typed_value(default, type_hint)
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
        self._cache[str(key)] = value
        self._loaded = True

    def remove(self, key: str) -> None:
        with self.storage.connect() as conn, conn:
            conn.execute("DELETE FROM app_settings WHERE key = ?", (str(key),))
        if self._loaded:
            self._cache.pop(str(key), None)

    def contains(self, key: str) -> bool:
        self._ensure_loaded()
        return str(key) in self._cache

    def keys(self) -> list[str]:
        self._ensure_loaded()
        return sorted(self._cache.keys())

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

    def remove(self, key: str) -> None:
        with self.storage.connect() as conn, conn:
            conn.execute("DELETE FROM app_state WHERE key = ?", (str(key),))


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
        try:
            return _restore_cache_payload(payload, assets_dir=self.storage.assets_dir)
        except _CacheAssetMissingError:
            self.delete(namespace, cache_key)
            return None

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
        new_asset_paths = set(self._iter_asset_paths(prepared_payload))
        old_asset_paths: set[str] = set()
        with self.storage.connect() as conn, conn:
            row = conn.execute(
                "SELECT payload_json FROM cache_entries WHERE namespace = ? AND cache_key = ?",
                (str(namespace), str(cache_key)),
            ).fetchone()
            old_asset_paths = self._asset_paths_from_payload_json(row["payload_json"] if row is not None else None)
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
        for relative_path in old_asset_paths - new_asset_paths:
            try:
                (self.storage.assets_dir / relative_path).unlink(missing_ok=True)
            except Exception:
                continue

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

    def delete_namespace(self, namespace: str) -> int:
        with self.storage.connect() as conn:
            rows = conn.execute(
                "SELECT payload_json FROM cache_entries WHERE namespace = ?",
                (str(namespace),),
            ).fetchall()
        for row in rows:
            self._delete_assets_from_row(row["payload_json"])
        with self.storage.connect() as conn, conn:
            cur = conn.execute("DELETE FROM cache_entries WHERE namespace = ?", (str(namespace),))
            return int(cur.rowcount or 0)

    def prune_namespace(self, namespace: str, max_entries: int) -> int:
        limit = max(0, int(max_entries))
        with self.storage.connect() as conn:
            rows = conn.execute(
                """
                SELECT cache_key, payload_json
                FROM cache_entries
                WHERE namespace = ?
                ORDER BY fetched_at DESC, cache_key DESC
                """,
                (str(namespace),),
            ).fetchall()
        if len(rows) <= limit:
            return 0
        stale_rows = rows[limit:]
        for row in stale_rows:
            self._delete_assets_from_row(row["payload_json"])
        with self.storage.connect() as conn, conn:
            cur = conn.executemany(
                "DELETE FROM cache_entries WHERE namespace = ? AND cache_key = ?",
                [(str(namespace), str(row["cache_key"])) for row in stale_rows],
            )
            return int(cur.rowcount or 0)

    def _delete_assets_from_row(self, payload_json: object) -> None:
        for relative_path in self._asset_paths_from_payload_json(payload_json):
            try:
                (self.storage.assets_dir / relative_path).unlink(missing_ok=True)
            except Exception:
                continue

    def _asset_paths_from_payload_json(self, payload_json: object) -> set[str]:
        if not isinstance(payload_json, str) or not payload_json:
            return set()
        try:
            payload = json.loads(payload_json)
        except Exception:
            return set()
        return set(self._iter_asset_paths(payload))

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

    def load_workspace(self) -> dict[str, object] | None:
        return self._load_one("plan_kind = 'workspace'")

    def save_workspace(
        self,
        snapshot: Mapping[str, object] | None,
        targets: Iterable[Mapping[str, object]],
    ) -> dict[str, object]:
        now = _utc_timestamp()
        payload_json = json.dumps(dict(snapshot or {}), ensure_ascii=False)
        existing_id = self._find_workspace_id()
        with self.storage.connect() as conn, conn:
            if existing_id is None:
                plan_id = str(uuid4())
                conn.execute(
                    """
                    INSERT INTO plans (
                        id, name, plan_kind, payload_json, created_at, updated_at, last_opened_at, deleted_at
                    )
                    VALUES (?, ?, 'workspace', ?, ?, ?, ?, NULL)
                    """,
                    (plan_id, "Workspace", payload_json, now, now, now),
                )
            else:
                plan_id = existing_id
                conn.execute(
                    """
                    UPDATE plans
                    SET name = ?, payload_json = ?, updated_at = ?, last_opened_at = ?, deleted_at = NULL
                    WHERE id = ?
                    """,
                    ("Workspace", payload_json, now, now, plan_id),
                )
            self._replace_targets(conn, plan_id, targets, now=now)
        return self.load_plan(plan_id) or {}

    def list_saved(self) -> list[dict[str, object]]:
        with self.storage.connect() as conn:
            rows = conn.execute(
                """
                SELECT plans.id, plans.name, plans.payload_json, plans.created_at,
                       plans.updated_at, plans.last_opened_at, COUNT(plan_targets.id) AS target_count
                FROM plans
                LEFT JOIN plan_targets
                    ON plan_targets.plan_id = plans.id AND plan_targets.deleted_at IS NULL
                WHERE plans.deleted_at IS NULL AND plans.plan_kind = 'saved'
                GROUP BY plans.id, plans.name, plans.payload_json, plans.created_at, plans.updated_at, plans.last_opened_at
                ORDER BY plans.last_opened_at DESC, lower(plans.name), plans.created_at DESC
                """
            ).fetchall()
        items: list[dict[str, object]] = []
        for row in rows:
            items.append(
                {
                    "id": str(row["id"]),
                    "name": str(row["name"]),
                    "snapshot": _decode_json_object(row["payload_json"], {}),
                    "created_at": float(row["created_at"] or 0.0),
                    "updated_at": float(row["updated_at"] or 0.0),
                    "last_opened_at": float(row["last_opened_at"] or 0.0),
                    "target_count": int(row["target_count"] or 0),
                }
            )
        return items

    def save_named(
        self,
        name: str,
        snapshot: Mapping[str, object] | None,
        targets: Iterable[Mapping[str, object]],
        plan_id: str | None = None,
    ) -> dict[str, object]:
        normalized_name = str(name or "").strip() or "Saved Plan"
        now = _utc_timestamp()
        payload_json = json.dumps(dict(snapshot or {}), ensure_ascii=False)
        existing_id: str | None = None
        if plan_id:
            with self.storage.connect() as conn:
                row = conn.execute(
                    """
                    SELECT id
                    FROM plans
                    WHERE id = ? AND deleted_at IS NULL AND plan_kind = 'saved'
                    """,
                    (str(plan_id),),
                ).fetchone()
            if row is not None:
                existing_id = str(row["id"])
        with self.storage.connect() as conn, conn:
            if existing_id is None:
                existing_id = str(uuid4())
                conn.execute(
                    """
                    INSERT INTO plans (
                        id, name, plan_kind, payload_json, created_at, updated_at, last_opened_at, deleted_at
                    )
                    VALUES (?, ?, 'saved', ?, ?, ?, ?, NULL)
                    """,
                    (existing_id, normalized_name, payload_json, now, now, now),
                )
            else:
                conn.execute(
                    """
                    UPDATE plans
                    SET name = ?, payload_json = ?, updated_at = ?, last_opened_at = ?, deleted_at = NULL
                    WHERE id = ?
                    """,
                    (normalized_name, payload_json, now, now, existing_id),
                )
            self._replace_targets(conn, existing_id, targets, now=now)
        return self.load_plan(existing_id) or {}

    def load_plan(self, plan_id: str) -> dict[str, object] | None:
        return self._load_one("id = ?", (str(plan_id),), touch=True)

    def delete_plan(self, plan_id: str) -> bool:
        with self.storage.connect() as conn, conn:
            row = conn.execute(
                "SELECT 1 FROM plans WHERE id = ? AND deleted_at IS NULL AND plan_kind = 'saved'",
                (str(plan_id),),
            ).fetchone()
            if row is None:
                return False
            conn.execute("DELETE FROM plans WHERE id = ?", (str(plan_id),))
        return True

    def _find_workspace_id(self) -> str | None:
        with self.storage.connect() as conn:
            row = conn.execute(
                """
                SELECT id
                FROM plans
                WHERE deleted_at IS NULL AND plan_kind = 'workspace'
                ORDER BY updated_at DESC
                LIMIT 1
                """
            ).fetchone()
        if row is None:
            return None
        return str(row["id"])

    def _load_one(
        self,
        where_clause: str,
        params: tuple[object, ...] = (),
        *,
        touch: bool = False,
    ) -> dict[str, object] | None:
        with self.storage.connect() as conn:
            row = conn.execute(
                f"""
                SELECT id, name, plan_kind, payload_json, created_at, updated_at, last_opened_at
                FROM plans
                WHERE deleted_at IS NULL AND {where_clause}
                LIMIT 1
                """,
                params,
            ).fetchone()
            if row is None:
                return None
            targets = self._load_targets(conn, str(row["id"]))
        if touch:
            now = _utc_timestamp()
            with self.storage.connect() as conn, conn:
                conn.execute("UPDATE plans SET last_opened_at = ? WHERE id = ?", (now, str(row["id"])))
            last_opened_at = now
        else:
            last_opened_at = float(row["last_opened_at"] or 0.0)
        return {
            "id": str(row["id"]),
            "name": str(row["name"]),
            "plan_kind": str(row["plan_kind"]),
            "snapshot": _decode_json_object(row["payload_json"], {}),
            "targets": targets,
            "created_at": float(row["created_at"] or 0.0),
            "updated_at": float(row["updated_at"] or 0.0),
            "last_opened_at": last_opened_at,
        }

    def _load_targets(self, conn: sqlite3.Connection, plan_id: str) -> list[dict[str, object]]:
        rows = conn.execute(
            """
            SELECT payload_json
            FROM plan_targets
            WHERE plan_id = ? AND deleted_at IS NULL
            ORDER BY sort_order ASC, created_at ASC
            """,
            (str(plan_id),),
        ).fetchall()
        targets: list[dict[str, object]] = []
        for row in rows:
            payload = _decode_json_object(row["payload_json"], {})
            if isinstance(payload, dict):
                targets.append(payload)
        return targets

    def _replace_targets(
        self,
        conn: sqlite3.Connection,
        plan_id: str,
        targets: Iterable[Mapping[str, object]],
        *,
        now: float,
    ) -> None:
        conn.execute("DELETE FROM plan_targets WHERE plan_id = ?", (str(plan_id),))
        for sort_order, target in enumerate(list(targets)):
            payload = dict(target)
            target_name = str(payload.get("name") or "").strip()
            target_key = self._target_key(payload, fallback=target_name or str(sort_order))
            conn.execute(
                """
                INSERT INTO plan_targets (
                    id, plan_id, target_key, sort_order, payload_json, created_at, updated_at, deleted_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, NULL)
                """,
                (
                    str(uuid4()),
                    str(plan_id),
                    target_key,
                    int(sort_order),
                    json.dumps(payload, ensure_ascii=False),
                    now,
                    now,
                ),
            )

    @staticmethod
    def _target_key(payload: Mapping[str, object], *, fallback: str) -> str:
        source_catalog = str(payload.get("source_catalog") or "").strip().lower()
        source_object_id = str(payload.get("source_object_id") or "").strip().lower()
        if source_catalog and source_object_id:
            return f"{source_catalog}:{source_object_id}"
        name = str(payload.get("name") or "").strip().lower()
        ra = payload.get("ra")
        dec = payload.get("dec")
        if name:
            return name
        return f"{fallback}:{ra}:{dec}"


class ObservationLogRepository:
    def __init__(self, storage: "AppStorage") -> None:
        self.storage = storage

    def append(
        self,
        *,
        target_name: str,
        target_key: str,
        target_payload: Mapping[str, object],
        site_name: str,
        site_payload: Mapping[str, object],
        notes: str = "",
        source: str = "",
        plan_id: str | None = None,
        observed_at: float | None = None,
    ) -> str:
        entry_id = str(uuid4())
        stamp = float(observed_at or _utc_timestamp())
        with self.storage.connect() as conn, conn:
            conn.execute(
                """
                INSERT INTO observation_log (
                    id, observed_at, target_name, target_key, target_payload_json,
                    site_name, site_payload_json, notes, source, plan_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry_id,
                    stamp,
                    str(target_name).strip(),
                    str(target_key).strip(),
                    json.dumps(dict(target_payload), ensure_ascii=False),
                    str(site_name).strip(),
                    json.dumps(dict(site_payload), ensure_ascii=False),
                    str(notes or ""),
                    str(source or ""),
                    str(plan_id or "") or None,
                ),
            )
        return entry_id

    def list_entries(self, *, search: str = "", limit: int | None = None) -> list[dict[str, object]]:
        clauses = ["1 = 1"]
        params: list[object] = []
        search_text = str(search or "").strip().lower()
        if search_text:
            clauses.append("(lower(target_name) LIKE ? OR lower(site_name) LIKE ?)")
            pattern = f"%{search_text}%"
            params.extend([pattern, pattern])
        sql = f"""
            SELECT id, observed_at, target_name, target_key, target_payload_json,
                   site_name, site_payload_json, notes, source, plan_id
            FROM observation_log
            WHERE {' AND '.join(clauses)}
            ORDER BY observed_at DESC, target_name ASC
        """
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        with self.storage.connect() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        entries: list[dict[str, object]] = []
        for row in rows:
            entries.append(
                {
                    "id": str(row["id"]),
                    "observed_at": float(row["observed_at"] or 0.0),
                    "target_name": str(row["target_name"]),
                    "target_key": str(row["target_key"]),
                    "target_payload": _decode_json_object(row["target_payload_json"], {}),
                    "site_name": str(row["site_name"]),
                    "site_payload": _decode_json_object(row["site_payload_json"], {}),
                    "notes": str(row["notes"] or ""),
                    "source": str(row["source"] or ""),
                    "plan_id": str(row["plan_id"] or ""),
                }
            )
        return entries


class ChatHistoryRepository:
    def __init__(self, storage: "AppStorage") -> None:
        self.storage = storage

    def list_messages(self, plan_id: str) -> list[dict[str, object]]:
        with self.storage.connect() as conn:
            thread = conn.execute(
                "SELECT id, plan_id FROM chat_threads WHERE plan_id = ?",
                (str(plan_id),),
            ).fetchone()
            if thread is None:
                return []
            rows = conn.execute(
                """
                SELECT id, kind, text, sort_order, payload_json, created_at
                FROM chat_messages
                WHERE thread_id = ?
                ORDER BY sort_order ASC, created_at ASC
                """,
                (str(thread["id"]),),
            ).fetchall()
        messages: list[dict[str, object]] = []
        for row in rows:
            message = {
                "id": str(row["id"]),
                "plan_id": str(thread["plan_id"]),
                "kind": str(row["kind"] or "info"),
                "text": str(row["text"] or ""),
                "sort_order": int(row["sort_order"] or 0),
                "created_at": float(row["created_at"] or 0.0),
            }
            payload = _decode_json_object(row["payload_json"], {})
            if isinstance(payload, dict):
                message.update(payload)
            messages.append(message)
        return messages

    def replace_messages(self, plan_id: str, messages: Iterable[Mapping[str, object]]) -> None:
        thread_id = self._ensure_thread(plan_id)
        now = _utc_timestamp()
        with self.storage.connect() as conn, conn:
            conn.execute(
                "UPDATE chat_threads SET updated_at = ? WHERE id = ?",
                (now, thread_id),
            )
            conn.execute("DELETE FROM chat_messages WHERE thread_id = ?", (thread_id,))
            for sort_order, message in enumerate(list(messages)):
                payload = dict(message)
                kind = str(payload.pop("kind", "info") or "info")
                text = str(payload.pop("text", "") or "")
                created_at = float(payload.pop("created_at", now) or now)
                conn.execute(
                    """
                    INSERT INTO chat_messages (
                        id, thread_id, kind, text, sort_order, payload_json, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(uuid4()),
                        thread_id,
                        kind,
                        text,
                        int(sort_order),
                        json.dumps(payload, ensure_ascii=False),
                        created_at,
                    ),
                )

    def clear(self, plan_id: str) -> None:
        with self.storage.connect() as conn, conn:
            row = conn.execute(
                "SELECT id FROM chat_threads WHERE plan_id = ?",
                (str(plan_id),),
            ).fetchone()
            if row is None:
                return
            conn.execute("DELETE FROM chat_messages WHERE thread_id = ?", (str(row["id"]),))
            conn.execute("UPDATE chat_threads SET updated_at = ? WHERE id = ?", (_utc_timestamp(), str(row["id"])))

    def _ensure_thread(self, plan_id: str) -> str:
        with self.storage.connect() as conn, conn:
            row = conn.execute(
                "SELECT id FROM chat_threads WHERE plan_id = ?",
                (str(plan_id),),
            ).fetchone()
            if row is not None:
                return str(row["id"])
            thread_id = str(uuid4())
            now = _utc_timestamp()
            conn.execute(
                """
                INSERT INTO chat_threads (id, plan_id, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (thread_id, str(plan_id), now, now),
            )
            return thread_id


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
        (
            2,
            "plans-chat-observation-upgrade",
            """
            ALTER TABLE plans ADD COLUMN plan_kind TEXT NOT NULL DEFAULT 'saved';
            ALTER TABLE plans ADD COLUMN last_opened_at REAL NOT NULL DEFAULT 0;

            CREATE UNIQUE INDEX IF NOT EXISTS idx_plans_workspace_unique
            ON plans (plan_kind)
            WHERE deleted_at IS NULL AND plan_kind = 'workspace';

            CREATE INDEX IF NOT EXISTS idx_plans_kind_updated
            ON plans (plan_kind, deleted_at, last_opened_at, updated_at);

            CREATE INDEX IF NOT EXISTS idx_plan_targets_plan_sort
            ON plan_targets (plan_id, deleted_at, sort_order);

            CREATE INDEX IF NOT EXISTS idx_cache_entries_namespace_fetched_at
            ON cache_entries (namespace, fetched_at);

            CREATE TABLE IF NOT EXISTS observation_log (
                id TEXT PRIMARY KEY,
                observed_at REAL NOT NULL,
                target_name TEXT NOT NULL,
                target_key TEXT NOT NULL,
                target_payload_json TEXT NOT NULL DEFAULT '{}',
                site_name TEXT NOT NULL DEFAULT '',
                site_payload_json TEXT NOT NULL DEFAULT '{}',
                notes TEXT NOT NULL DEFAULT '',
                source TEXT NOT NULL DEFAULT '',
                plan_id TEXT REFERENCES plans(id) ON DELETE SET NULL
            );

            CREATE INDEX IF NOT EXISTS idx_observation_log_observed_at
            ON observation_log (observed_at DESC, target_name);

            CREATE INDEX IF NOT EXISTS idx_observation_log_target_key
            ON observation_log (target_key, observed_at DESC);

            CREATE TABLE IF NOT EXISTS chat_threads (
                id TEXT PRIMARY KEY,
                plan_id TEXT NOT NULL UNIQUE REFERENCES plans(id) ON DELETE CASCADE,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_chat_threads_plan
            ON chat_threads (plan_id);

            CREATE TABLE IF NOT EXISTS chat_messages (
                id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL REFERENCES chat_threads(id) ON DELETE CASCADE,
                kind TEXT NOT NULL DEFAULT 'info',
                text TEXT NOT NULL DEFAULT '',
                sort_order INTEGER NOT NULL DEFAULT 0,
                payload_json TEXT NOT NULL DEFAULT '{}',
                created_at REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_chat_messages_thread_sort
            ON chat_messages (thread_id, sort_order, created_at);
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
        self.observation_log = ObservationLogRepository(self)
        self.chat_history = ChatHistoryRepository(self)
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
