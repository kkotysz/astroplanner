from __future__ import annotations

import hashlib
import logging
import math
import os
from datetime import datetime
from time import perf_counter
from typing import TYPE_CHECKING, Any, Optional

import shiboken6 as shb
from PySide6.QtWidgets import QMessageBox

from astroplanner.bhtom import (
    BHTOM_API_BASE_URL,
    BHTOM_MAX_OBSERVATORY_PAGES,
    BHTOM_MAX_SUGGESTION_PAGES,
    BHTOM_OBSERVATORY_CACHE_TTL_S,
    BHTOM_SUGGESTION_CACHE_TTL_S,
    BhtomCandidatePrefetchWorker,
    BhtomObservatoryPresetWorker,
    BhtomSuggestionWorker,
    _bhtom_observatory_payload_has_more,
    _bhtom_payload_has_more,
    _bhtom_suggestion_source_message,
    _build_bhtom_candidate_from_item,
    _build_bhtom_observatory_presets,
    _extract_bhtom_items,
    _extract_bhtom_observatory_items,
    _fetch_bhtom_observatory_page_payload,
    _fetch_bhtom_target_page_payload,
    _pick_first_present,
    _rank_local_target_suggestions_from_candidates,
)
from astroplanner.models import Site, Target, targets_match as _targets_match
from astroplanner.resolvers import (
    _normalize_catalog_token,
    _object_type_is_unknown,
    _safe_float,
    _target_magnitude_label,
)
from astroplanner.scoring import TargetNightMetrics
from astroplanner.ui.suggestions import SuggestedTargetsDialog
from astroplanner.ui.targets import TargetTableModel

if TYPE_CHECKING:
    from astro_planner import MainWindow


logger = logging.getLogger(__name__)


class BhtomCoordinator:
    """Own BHTOM cache, worker lifecycle, quick targets, and suggestions glue."""

    def __init__(self, planner: "MainWindow") -> None:
        object.__setattr__(self, "_planner", planner)

    def __getattr__(self, name: str):
        return getattr(self._planner, name)

    def __setattr__(self, name: str, value: object) -> None:
        if name == "_planner":
            object.__setattr__(self, name, value)
            return
        setattr(self._planner, name, value)

    def _prefetch_bhtom_candidates_on_startup(self) -> None:
        token = self._bhtom_api_token_optional()
        if not token:
            return
        refresh_on_startup = bool(self.settings.value("general/bhtomRefreshOnStartup", True, type=bool))
        if refresh_on_startup:
            self._set_bhtom_status("BHTOM: refreshing cache...", busy=True)
            self._start_bhtom_candidate_prefetch(force_refresh=True)
            return
        base_url = self._bhtom_api_base_url()
        cached_candidates = self._cached_bhtom_candidates(token=token, base_url=base_url)
        if not cached_candidates:
            return
        self._bhtom_candidate_cache_key = (base_url, token)
        self._bhtom_candidate_cache = list(cached_candidates)
        self._bhtom_candidate_cache_loaded_at = perf_counter()
        self._refresh_cached_bhtom_suggestions()
        ranked_count = len(self._bhtom_ranked_suggestions_cache or [])
        if ranked_count > 0:
            self._set_bhtom_status(
                f"BHTOM: cache ({ranked_count} ranked / {len(cached_candidates)} cached)",
                busy=False,
            )
        else:
            self._set_bhtom_status(f"BHTOM: cache ({len(cached_candidates)} cached)", busy=False)


    def _start_bhtom_candidate_prefetch(self, *, force_refresh: bool = False) -> bool:
        token = self._bhtom_api_token_optional()
        if not token:
            return False
        worker = getattr(self, "_bhtom_candidate_prefetch_worker", None)
        if worker is not None and worker.isRunning():
            return False
        base_url = self._bhtom_api_base_url()
        self._bhtom_candidate_prefetch_request_id += 1
        req_id = self._bhtom_candidate_prefetch_request_id
        prefetch_worker = BhtomCandidatePrefetchWorker(
            request_id=req_id,
            base_url=base_url,
            token=token,
            parent=self._planner,
        )
        prefetch_worker.completed.connect(self._on_bhtom_candidate_prefetch_completed)
        prefetch_worker.finished.connect(lambda w=prefetch_worker: self._on_bhtom_candidate_prefetch_finished(w))
        prefetch_worker.finished.connect(prefetch_worker.deleteLater)
        self._bhtom_candidate_prefetch_worker = prefetch_worker
        self._set_bhtom_status("BHTOM: refreshing cache...", busy=True)
        prefetch_worker.start()
        return True


    def _on_bhtom_candidate_prefetch_completed(self, request_id: int, candidates: list, err: str) -> None:
        if request_id != self._bhtom_candidate_prefetch_request_id:
            return
        if err:
            # Fallback to cached data if refresh failed.
            token = self._bhtom_api_token_optional()
            base_url = self._bhtom_api_base_url()
            cached = self._cached_bhtom_candidates(token=token, base_url=base_url, force_refresh=False)
            if cached:
                self._bhtom_candidate_cache_key = (base_url, token)
                self._bhtom_candidate_cache = list(cached)
                self._bhtom_candidate_cache_loaded_at = perf_counter()
                self._refresh_cached_bhtom_suggestions()
                ranked_count = len(self._bhtom_ranked_suggestions_cache or [])
                if ranked_count > 0:
                    self._set_bhtom_status(
                        f"BHTOM: stale cache ({ranked_count} ranked / {len(cached)} cached), refresh failed",
                        busy=False,
                    )
                else:
                    self._set_bhtom_status(f"BHTOM: stale cache ({len(cached)} cached), refresh failed", busy=False)
            else:
                self._set_bhtom_status("BHTOM: cache refresh failed", busy=False)
            logger.warning("BHTOM candidate prefetch failed: %s", err)
            return
        token = self._bhtom_api_token_optional()
        base_url = self._bhtom_api_base_url()
        cache_key = (base_url, token)
        self._bhtom_candidate_cache_key = cache_key
        self._bhtom_candidate_cache = list(candidates)
        self._bhtom_candidate_cache_loaded_at = perf_counter()
        self._bhtom_last_network_fetch_key = cache_key
        self._refresh_cached_bhtom_suggestions()
        storage = getattr(self, "app_storage", None)
        if storage is not None:
            try:
                storage.cache.set_json(
                    "bhtom_candidates",
                    self._bhtom_storage_cache_key(token=token, base_url=base_url),
                    self._serialize_bhtom_candidates(candidates),
                    ttl_s=BHTOM_SUGGESTION_CACHE_TTL_S,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to persist BHTOM candidates after refresh: %s", exc)
        ranked_count = len(self._bhtom_ranked_suggestions_cache or [])
        if ranked_count > 0:
            self._set_bhtom_status(
                f"BHTOM: cache refreshed ({ranked_count} ranked / {len(candidates)} cached)",
                busy=False,
            )
        else:
            self._set_bhtom_status(f"BHTOM: cache refreshed ({len(candidates)} cached)", busy=False)


    def _on_bhtom_candidate_prefetch_finished(self, worker: BhtomCandidatePrefetchWorker) -> None:
        if self._bhtom_candidate_prefetch_worker is worker:
            self._bhtom_candidate_prefetch_worker = None


    def _refresh_cached_bhtom_suggestions(self) -> None:
        self._bhtom_ranked_suggestions_cache = []
        candidates = list(self._bhtom_candidate_cache or [])
        if not candidates:
            return
        context, error = self._build_bhtom_suggestion_context()
        if context is None:
            if error:
                logger.info("Skipping cached BHTOM suggestion ranking: %s", error)
            return
        try:
            suggestions, _notes = _rank_local_target_suggestions_from_candidates(
                payload=context["payload"],  # type: ignore[index]
                site=context["site"],  # type: ignore[index]
                targets=context["targets"],  # type: ignore[index]
                limit_altitude=float(context["limit_altitude"]),
                sun_alt_limit=float(context["sun_alt_limit"]),
                min_moon_sep=float(context["min_moon_sep"]),
                candidates=candidates,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to rebuild cached BHTOM suggestions: %s", exc)
            return
        self._bhtom_ranked_suggestions_cache = list(suggestions)


    def _bhtom_api_base_url(self) -> str:
        return (os.getenv("BHTOM_API_BASE_URL", "") or BHTOM_API_BASE_URL).strip().rstrip("/")


    def _bhtom_api_token_optional(self) -> str:
        return (
            self.settings.value("general/bhtomApiToken", "", type=str)
            or os.getenv("BHTOM_API_TOKEN", "")
        ).strip()


    def _bhtom_api_token(self) -> str:
        token = self._bhtom_api_token_optional()
        if not token:
            raise RuntimeError("BHTOM features require an API token in Settings -> General Settings.")
        return token


    def _current_bhtom_cache_identity(self) -> Optional[tuple[str, str]]:
        token = self._bhtom_api_token_optional()
        if not token:
            return None
        return (self._bhtom_api_base_url(), token)


    def _bhtom_should_fetch_from_network_now(self) -> bool:
        """Return True when Suggest/Quick should force one network refresh in this session."""
        identity = self._current_bhtom_cache_identity()
        if identity is None:
            return False
        return self._bhtom_last_network_fetch_key != identity


    def _bhtom_token_hash(self, token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()


    def _bhtom_storage_cache_key(self, *, token: str, base_url: str) -> str:
        return f"{str(base_url).strip().rstrip('/')}::{self._bhtom_token_hash(token)}"


    def _clone_bhtom_observatory_presets(self, presets: list[dict[str, object]]) -> list[dict[str, object]]:
        cloned: list[dict[str, object]] = []
        for item in presets:
            if not isinstance(item, dict):
                continue
            site = item.get("site")
            if not isinstance(site, Site):
                continue
            cloned.append(
                {
                    "key": str(item.get("key", "")),
                    "label": str(item.get("label", "")),
                    "source": str(item.get("source", "bhtom") or "bhtom"),
                    "site": Site(**site.model_dump()),
                }
            )
        return cloned


    def _serialize_bhtom_observatory_presets(self, presets: list[dict[str, object]]) -> list[dict[str, object]]:
        serializable: list[dict[str, object]] = []
        for item in presets:
            if not isinstance(item, dict):
                continue
            site = item.get("site")
            if not isinstance(site, Site):
                continue
            serializable.append(
                {
                    "key": str(item.get("key", "")),
                    "label": str(item.get("label", "")),
                    "source": str(item.get("source", "bhtom") or "bhtom"),
                    "site": site.model_dump(mode="json"),
                }
            )
        return serializable


    def _deserialize_bhtom_observatory_presets(self, payload: object) -> list[dict[str, object]]:
        if not isinstance(payload, list):
            return []
        parsed: list[dict[str, object]] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            key = str(item.get("key", "")).strip()
            label = str(item.get("label", "")).strip()
            site_payload = item.get("site")
            if not key or not label or not isinstance(site_payload, dict):
                continue
            try:
                site = Site(**site_payload)
            except Exception:
                continue
            parsed.append({"key": key, "label": label, "source": str(item.get("source", "bhtom") or "bhtom"), "site": site})
        return parsed


    def _serialize_bhtom_candidates(self, candidates: list[dict[str, object]]) -> list[dict[str, object]]:
        serializable: list[dict[str, object]] = []
        for item in candidates:
            if not isinstance(item, dict):
                continue
            target = item.get("target")
            if not isinstance(target, Target):
                continue
            serializable.append(
                {
                    "target": target.model_dump(mode="json"),
                    "importance": float(item.get("importance", 0.0) or 0.0),
                    "bhtom_priority": int(item.get("bhtom_priority", 0) or 0),
                    "sun_separation": _safe_float(item.get("sun_separation")),
                }
            )
        return serializable


    def _deserialize_bhtom_candidates(self, payload: object) -> list[dict[str, object]]:
        if not isinstance(payload, list):
            return []
        candidates: list[dict[str, object]] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            target_payload = item.get("target")
            if not isinstance(target_payload, dict):
                continue
            try:
                target = Target(**target_payload)
            except Exception:
                continue
            candidates.append(
                {
                    "target": target,
                    "importance": float(item.get("importance", 0.0) or 0.0),
                    "bhtom_priority": int(item.get("bhtom_priority", 0) or 0),
                    "sun_separation": _safe_float(item.get("sun_separation")),
                }
            )
        return candidates


    def _load_bhtom_observatory_disk_cache(
        self,
        *,
        token: str,
        base_url: str,
    ) -> Optional[list[dict[str, object]]]:
        storage = getattr(self, "app_storage", None)
        if storage is None:
            return None
        cache_key = self._bhtom_storage_cache_key(token=token, base_url=base_url)
        try:
            cached = storage.cache.get_json("bhtom_observatory_presets", cache_key)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to read BHTOM observatory cache from storage: %s", exc)
            return None
        presets = self._deserialize_bhtom_observatory_presets(cached)
        return presets or None


    def _save_bhtom_observatory_disk_cache(
        self,
        presets: list[dict[str, object]],
        *,
        token: str,
        base_url: str,
    ) -> None:
        serializable = self._serialize_bhtom_observatory_presets(presets)
        if not serializable:
            return
        storage = getattr(self, "app_storage", None)
        if storage is None:
            return
        cache_key = self._bhtom_storage_cache_key(token=token, base_url=base_url)
        storage.cache.set_json(
            "bhtom_observatory_presets",
            cache_key,
            serializable,
            ttl_s=BHTOM_OBSERVATORY_CACHE_TTL_S,
        )


    def _fetch_bhtom_target_page(self, page: int, token: Optional[str] = None) -> object:
        return _fetch_bhtom_target_page_payload(
            endpoint_base_url=self._bhtom_api_base_url(),
            token=(token or self._bhtom_api_token()),
            page=int(page),
        )


    def _fetch_bhtom_observatory_page(self, page: int, token: Optional[str] = None) -> object:
        return _fetch_bhtom_observatory_page_payload(
            endpoint_base_url=self._bhtom_api_base_url(),
            token=(token or self._bhtom_api_token()),
            page=int(page),
        )


    def _bhtom_observatory_prefetch_status(self) -> tuple[bool, str]:
        worker = getattr(self, "_bhtom_observatory_worker", None)
        if worker is not None and worker.isRunning():
            return True, str(self._bhtom_observatory_loading_message or "Loading BHTOM presets...")
        cache = self._cached_bhtom_observatory_presets()
        if cache:
            return False, f"Loaded {len(cache)} cached BHTOM presets."
        if not self._bhtom_api_token_optional():
            return False, "BHTOM token is not configured."
        return False, "BHTOM presets load in background."


    def _prefetch_bhtom_observatory_presets_on_startup(self) -> None:
        token = self._bhtom_api_token_optional()
        if not token:
            return
        base_url = self._bhtom_api_base_url()
        cached = self._cached_bhtom_observatory_presets(token=token, base_url=base_url)
        if cached:
            self.bhtom_observatory_presets_changed.emit(cached, f"Loaded {len(cached)} cached BHTOM presets.")
            self._set_bhtom_status(f"BHTOM: cache ({len(cached)})", busy=False)


    def _ensure_bhtom_observatory_prefetch(self, *, force_refresh: bool = False) -> bool:
        token = self._bhtom_api_token_optional()
        if not token:
            self._bhtom_observatory_loading_message = "BHTOM token is not configured."
            self.bhtom_observatory_presets_loading.emit(False, self._bhtom_observatory_loading_message)
            return False
        base_url = self._bhtom_api_base_url()
        worker = self._bhtom_observatory_worker
        if worker is not None and worker.isRunning():
            return False
        if not force_refresh:
            cached = self._cached_bhtom_observatory_presets(token=token, base_url=base_url)
            if cached:
                self.bhtom_observatory_presets_changed.emit(cached, f"Loaded {len(cached)} cached BHTOM presets.")
                return False
        self._bhtom_observatory_worker_request_id += 1
        req_id = self._bhtom_observatory_worker_request_id
        self._bhtom_observatory_loading_message = "Loading BHTOM presets..."
        self.bhtom_observatory_presets_loading.emit(True, self._bhtom_observatory_loading_message)
        prefetch_worker = BhtomObservatoryPresetWorker(req_id, base_url=base_url, token=token, parent=self._planner)
        prefetch_worker.progress.connect(self._on_bhtom_observatory_prefetch_progress)
        prefetch_worker.completed.connect(self._on_bhtom_observatory_prefetch_completed)
        prefetch_worker.finished.connect(lambda w=prefetch_worker: self._on_bhtom_observatory_prefetch_finished(w))
        prefetch_worker.finished.connect(prefetch_worker.deleteLater)
        self._bhtom_observatory_worker = prefetch_worker
        prefetch_worker.start()
        return True


    def _on_bhtom_observatory_prefetch_progress(self, page: int, _total_pages: int, message: str) -> None:
        if self._bhtom_observatory_worker is None:
            return
        txt = str(message or "").strip() or f"Loading BHTOM presets... page {int(page)}"
        self._bhtom_observatory_loading_message = txt
        self.bhtom_observatory_presets_loading.emit(True, txt)


    def _on_bhtom_observatory_prefetch_completed(self, request_id: int, presets: list, err: str) -> None:
        if request_id != self._bhtom_observatory_worker_request_id:
            return
        token = self._bhtom_api_token_optional()
        base_url = self._bhtom_api_base_url()
        if err:
            msg = "BHTOM preset refresh failed." if str(err).strip().lower() == "cancelled" else f"BHTOM preset refresh failed: {err}"
            cached = self._cached_bhtom_observatory_presets(token=token, base_url=base_url)
            self._bhtom_observatory_loading_message = msg
            self.bhtom_observatory_presets_loading.emit(False, msg)
            if cached:
                self.bhtom_observatory_presets_changed.emit(cached, f"{msg} Using cached presets.")
            return
        safe_presets = self._clone_bhtom_observatory_presets(presets if isinstance(presets, list) else [])
        if not safe_presets:
            self._bhtom_observatory_loading_message = "BHTOM returned no usable presets."
            self.bhtom_observatory_presets_loading.emit(False, self._bhtom_observatory_loading_message)
            return
        self._bhtom_observatory_cache_key = (base_url, token)
        self._bhtom_observatory_cache = safe_presets
        self._bhtom_observatory_cache_loaded_at = perf_counter()
        try:
            self._save_bhtom_observatory_disk_cache(safe_presets, token=token, base_url=base_url)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to persist BHTOM observatory cache: %s", exc)
        loaded_msg = f"Loaded {len(safe_presets)} BHTOM presets."
        self._bhtom_observatory_loading_message = loaded_msg
        self.bhtom_observatory_presets_loading.emit(False, loaded_msg)
        self.bhtom_observatory_presets_changed.emit(self._clone_bhtom_observatory_presets(safe_presets), loaded_msg)


    def _on_bhtom_observatory_prefetch_finished(self, worker: BhtomObservatoryPresetWorker) -> None:
        if self._bhtom_observatory_worker is worker:
            self._bhtom_observatory_worker = None


    @staticmethod
    def _extract_bhtom_items(payload: object) -> list[dict[str, Any]]:
        return _extract_bhtom_items(payload)


    @staticmethod
    def _bhtom_payload_has_more(payload: object, page: int, item_count: int) -> bool:
        return _bhtom_payload_has_more(payload, page, item_count)


    @staticmethod
    def _pick_first_present(sources: list[dict[str, Any]], *keys: str) -> object:
        return _pick_first_present(sources, *keys)


    def _build_bhtom_candidate(self, item: dict[str, Any]) -> Optional[dict[str, object]]:
        return _build_bhtom_candidate_from_item(item)


    @staticmethod
    def _extract_bhtom_observatory_items(payload: object) -> list[dict[str, Any]]:
        return _extract_bhtom_observatory_items(payload)


    @staticmethod
    def _bhtom_observatory_payload_has_more(payload: object, page: int, item_count: int) -> bool:
        return _bhtom_observatory_payload_has_more(payload, page, item_count)


    def _cached_bhtom_observatory_presets(
        self,
        *,
        token: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> Optional[list[dict[str, object]]]:
        resolved_token = (token or "").strip()
        if not resolved_token:
            try:
                resolved_token = self._bhtom_api_token()
            except Exception:
                return None
        resolved_base_url = (base_url or self._bhtom_api_base_url()).strip().rstrip("/")
        cache_key = (resolved_base_url, resolved_token)
        cache_age = perf_counter() - self._bhtom_observatory_cache_loaded_at
        if (
            self._bhtom_observatory_cache_key == cache_key
            and self._bhtom_observatory_cache is not None
            and cache_age < BHTOM_OBSERVATORY_CACHE_TTL_S
        ):
            return self._clone_bhtom_observatory_presets(self._bhtom_observatory_cache)
        disk_cached = self._load_bhtom_observatory_disk_cache(token=resolved_token, base_url=resolved_base_url)
        if disk_cached:
            self._bhtom_observatory_cache_key = cache_key
            self._bhtom_observatory_cache = self._clone_bhtom_observatory_presets(disk_cached)
            self._bhtom_observatory_cache_loaded_at = perf_counter()
            return self._clone_bhtom_observatory_presets(self._bhtom_observatory_cache)
        return None


    def _fetch_bhtom_observatory_presets(self, *, force_refresh: bool = False) -> list[dict[str, object]]:
        token = self._bhtom_api_token()
        base_url = self._bhtom_api_base_url()
        cache_key = (base_url, token)
        if not force_refresh:
            cached = self._cached_bhtom_observatory_presets(token=token, base_url=base_url)
            if cached is not None:
                return cached

        items: list[dict[str, Any]] = []
        for page in range(1, BHTOM_MAX_OBSERVATORY_PAGES + 1):
            payload = self._fetch_bhtom_observatory_page(page, token=token)
            page_items = self._extract_bhtom_observatory_items(payload)
            if not page_items:
                if page == 1:
                    if isinstance(payload, dict):
                        keys = ", ".join(sorted(str(key) for key in payload.keys()))
                        raise RuntimeError(
                            f"BHTOM observatory endpoint returned an unexpected payload shape (keys: {keys or 'none'})."
                        )
                    raise RuntimeError("BHTOM observatory endpoint returned an unexpected payload shape.")
                break
            items.extend(page_items)
            if not self._bhtom_observatory_payload_has_more(payload, page, len(page_items)):
                break

        presets = _build_bhtom_observatory_presets(items)
        if not presets:
            raise RuntimeError("BHTOM returned no usable observatory/camera presets.")

        self._bhtom_observatory_cache_key = cache_key
        self._bhtom_observatory_cache = self._clone_bhtom_observatory_presets(
            presets if isinstance(presets, list) else []
        )
        self._bhtom_observatory_cache_loaded_at = perf_counter()
        try:
            self._save_bhtom_observatory_disk_cache(
                self._bhtom_observatory_cache,
                token=token,
                base_url=base_url,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to persist BHTOM observatory cache: %s", exc)
        return self._clone_bhtom_observatory_presets(self._bhtom_observatory_cache)


    def _cached_bhtom_candidates(
        self,
        *,
        token: Optional[str] = None,
        base_url: Optional[str] = None,
        force_refresh: bool = False,
    ) -> Optional[list[dict[str, object]]]:
        resolved_token = (token or "").strip()
        if not resolved_token:
            try:
                resolved_token = self._bhtom_api_token()
            except Exception:
                return None
        resolved_base_url = (base_url or self._bhtom_api_base_url()).strip().rstrip("/")
        cache_key = (resolved_base_url, resolved_token)
        cache_age = perf_counter() - self._bhtom_candidate_cache_loaded_at
        if not force_refresh and (
            self._bhtom_candidate_cache_key == cache_key
            and self._bhtom_candidate_cache is not None
            and cache_age < BHTOM_SUGGESTION_CACHE_TTL_S
        ):
            return list(self._bhtom_candidate_cache)
        storage = getattr(self, "app_storage", None)
        if storage is not None and not force_refresh:
            try:
                persisted = storage.cache.get_json(
                    "bhtom_candidates",
                    self._bhtom_storage_cache_key(token=resolved_token, base_url=resolved_base_url),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to read BHTOM candidate cache from storage: %s", exc)
            else:
                candidates = self._deserialize_bhtom_candidates(persisted)
                if candidates:
                    self._bhtom_candidate_cache_key = cache_key
                    self._bhtom_candidate_cache = list(candidates)
                    self._bhtom_candidate_cache_loaded_at = perf_counter()
                    return list(candidates)
        return None


    def _clear_bhtom_candidate_cache(self, *, token: Optional[str] = None, base_url: Optional[str] = None) -> None:
        resolved_token = (token or "").strip()
        if not resolved_token:
            try:
                resolved_token = self._bhtom_api_token()
            except Exception:
                resolved_token = ""
        resolved_base_url = (base_url or self._bhtom_api_base_url()).strip().rstrip("/")
        self._bhtom_candidate_cache_key = None
        self._bhtom_candidate_cache = None
        self._bhtom_candidate_cache_loaded_at = 0.0
        storage = getattr(self, "app_storage", None)
        if storage is not None:
            try:
                storage.cache.delete(
                    "bhtom_candidates",
                    self._bhtom_storage_cache_key(token=resolved_token, base_url=resolved_base_url),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to clear BHTOM candidate cache: %s", exc)


    def _fetch_bhtom_target_candidates(self, *, force_refresh: bool = False) -> list[dict[str, object]]:
        token = self._bhtom_api_token()
        base_url = self._bhtom_api_base_url()
        cache_key = (base_url, token)
        cached = self._cached_bhtom_candidates(token=token, base_url=base_url, force_refresh=force_refresh)
        if cached is not None:
            return cached

        candidates: list[dict[str, object]] = []
        seen_keys: set[str] = set()

        for page in range(1, BHTOM_MAX_SUGGESTION_PAGES + 1):
            payload = self._fetch_bhtom_target_page(page, token=token)
            items = self._extract_bhtom_items(payload)
            if not items:
                if page == 1:
                    if isinstance(payload, dict):
                        keys = ", ".join(sorted(str(key) for key in payload.keys()))
                        raise RuntimeError(f"BHTOM returned an unexpected payload shape (keys: {keys or 'none'}).")
                    raise RuntimeError("BHTOM returned an unexpected payload shape.")
                break

            for item in items:
                candidate = self._build_bhtom_candidate(item)
                if candidate is None:
                    continue
                target = candidate["target"]
                assert isinstance(target, Target)
                dedupe_key = _normalize_catalog_token(target.source_object_id or target.name)
                if not dedupe_key or dedupe_key in seen_keys:
                    continue
                seen_keys.add(dedupe_key)
                candidates.append(candidate)

            if not self._bhtom_payload_has_more(payload, page, len(items)):
                break

        if not candidates:
            raise RuntimeError("BHTOM returned no usable target candidates.")
        self._bhtom_candidate_cache_key = cache_key
        self._bhtom_candidate_cache = list(candidates)
        self._bhtom_candidate_cache_loaded_at = perf_counter()
        self._bhtom_last_network_fetch_key = cache_key
        storage = getattr(self, "app_storage", None)
        if storage is not None:
            try:
                storage.cache.set_json(
                    "bhtom_candidates",
                    self._bhtom_storage_cache_key(token=token, base_url=base_url),
                    self._serialize_bhtom_candidates(candidates),
                    ttl_s=BHTOM_SUGGESTION_CACHE_TTL_S,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to persist BHTOM candidates: %s", exc)
        return list(candidates)


    def _bhtom_type_for_target(self, target: Target) -> str:
        candidates = self._bhtom_candidate_cache or []
        if not candidates:
            return ""
        target_name = _normalize_catalog_token(target.name)
        target_source_id = _normalize_catalog_token(target.source_object_id)
        for candidate in candidates:
            candidate_target = candidate.get("target")
            if not isinstance(candidate_target, Target):
                continue
            candidate_name = _normalize_catalog_token(candidate_target.name)
            candidate_source_id = _normalize_catalog_token(candidate_target.source_object_id)
            if target_source_id and candidate_source_id == target_source_id:
                candidate_type = candidate_target.object_type.strip()
                if not _object_type_is_unknown(candidate_type):
                    return candidate_type
            if target_name and candidate_name == target_name:
                candidate_type = candidate_target.object_type.strip()
                if not _object_type_is_unknown(candidate_type):
                    return candidate_type
        return ""


    def _bhtom_importance_for_target(self, target: Target) -> Optional[float]:
        candidates = self._bhtom_candidate_cache or []
        if not candidates:
            return None
        target_name = _normalize_catalog_token(target.name)
        target_source_id = _normalize_catalog_token(target.source_object_id)
        for candidate in candidates:
            candidate_target = candidate.get("target")
            if not isinstance(candidate_target, Target):
                continue
            candidate_name = _normalize_catalog_token(candidate_target.name)
            candidate_source_id = _normalize_catalog_token(candidate_target.source_object_id)
            if (target_source_id and candidate_source_id == target_source_id) or (target_name and candidate_name == target_name):
                importance = _safe_float(candidate.get("importance"))
                if importance is not None and math.isfinite(importance):
                    return float(importance)
        return None


    def _ensure_known_target_type(self, target: Target) -> str:
        if not _object_type_is_unknown(target.object_type):
            return target.object_type
        bhtom_type = self._bhtom_type_for_target(target)
        if bhtom_type:
            target.object_type = bhtom_type
            return bhtom_type
        return target.object_type


    def _reload_local_target_suggestions(self, *, force_refresh: bool = False) -> tuple[list[dict[str, object]], list[str]]:
        if force_refresh:
            self._clear_bhtom_candidate_cache()
        else:
            self._bhtom_candidate_cache_key = None
            self._bhtom_candidate_cache = None
            self._bhtom_candidate_cache_loaded_at = 0.0
        self._bhtom_ranked_suggestions_cache = []
        return self._build_local_target_suggestions(force_refresh=force_refresh)


    def _build_bhtom_suggestion_context(self) -> tuple[Optional[dict[str, object]], Optional[str]]:
        payload = self.full_payload if isinstance(getattr(self, "full_payload", None), dict) else None
        if not payload:
            return None, "Run a visibility calculation first so suggestions use tonight's sky."
        if not self.table_model.site:
            return None, "Set a valid observing site before requesting suggestions."
        try:
            token = self._bhtom_api_token()
        except Exception as exc:  # noqa: BLE001
            return None, str(exc)
        try:
            site = Site(
                name=self.table_model.site.name,
                latitude=self._read_site_float(self.lat_edit),
                longitude=self._read_site_float(self.lon_edit),
                elevation=self._read_site_float(self.elev_edit),
                limiting_magnitude=self._current_limiting_magnitude(),
            )
        except Exception as exc:  # noqa: BLE001
            return None, f"Invalid observing site values: {exc}"

        min_moon_sep = float(self.min_moon_sep_spin.value()) if hasattr(self, "min_moon_sep_spin") else 0.0
        context = {
            "payload": payload,
            "site": site,
            "targets": [Target(**target.model_dump()) for target in self.targets],
            "limit_altitude": float(self.limit_spin.value()),
            "sun_alt_limit": self._sun_alt_limit(),
            "min_moon_sep": min_moon_sep,
            "bhtom_base_url": self._bhtom_api_base_url(),
            "bhtom_token": token,
        }
        return context, None


    def _start_bhtom_worker(self, *, mode: str, emit_partials: bool, force_refresh: bool = False) -> bool:
        existing = self._bhtom_worker
        if existing is not None and existing.isRunning():
            title = "Quick Targets" if mode == "quick" else "Suggest Targets"
            QMessageBox.information(self._planner, title, "A BHTOM request is already in progress.")
            return False

        context, error = self._build_bhtom_suggestion_context()
        if context is None:
            title = "Quick Targets" if mode == "quick" else "Suggest Targets"
            QMessageBox.warning(self._planner, title, error or "Unable to prepare BHTOM request.")
            return False

        self._bhtom_worker_request_id += 1
        req_id = self._bhtom_worker_request_id
        self._bhtom_worker_mode = mode
        base_url = str(context["bhtom_base_url"])
        token = str(context["bhtom_token"])
        self._bhtom_worker_cache_key = (base_url, token)
        cached_candidates = self._cached_bhtom_candidates(
            token=token,
            base_url=base_url,
            force_refresh=force_refresh,
        )
        if cached_candidates is not None:
            logger.info("Using cached BHTOM candidates (%d entries).", len(cached_candidates))
        self._bhtom_worker_source = "cache" if cached_candidates is not None else "network"

        worker = BhtomSuggestionWorker(
            request_id=req_id,
            payload=dict(context["payload"]),  # type: ignore[arg-type]
            site=Site(**context["site"].model_dump()),  # type: ignore[index]
            targets=[Target(**t.model_dump()) for t in context["targets"]],  # type: ignore[index]
            limit_altitude=float(context["limit_altitude"]),
            sun_alt_limit=float(context["sun_alt_limit"]),
            min_moon_sep=float(context["min_moon_sep"]),
            bhtom_base_url=base_url,
            bhtom_token=token,
            cached_candidates=cached_candidates,
            emit_partials=emit_partials,
            parent=self._planner,
        )
        worker.pageReady.connect(self._on_bhtom_worker_page_ready)
        worker.completed.connect(self._on_bhtom_worker_completed)
        worker.finished.connect(lambda w=worker: self._on_bhtom_worker_finished(w))
        worker.finished.connect(worker.deleteLater)
        self._bhtom_worker = worker
        worker.start()
        return True


    def _cancel_bhtom_worker(self) -> None:
        worker = self._bhtom_worker
        if worker is None:
            return
        try:
            if worker.isRunning():
                worker.requestInterruption()
                worker.quit()
        except Exception:
            pass


    def _on_bhtom_worker_page_ready(
        self,
        request_id: int,
        suggestions: list[dict[str, object]],
        notes: list[str],
        page: int,
        loaded_candidates: int,
    ) -> None:
        if request_id != self._bhtom_worker_request_id:
            return
        if page < 0:
            self._set_bhtom_status(f"BHTOM: cache ({loaded_candidates})", busy=True)
        else:
            self._set_bhtom_status(f"BHTOM: page {page} ({loaded_candidates})", busy=True)
        if self._bhtom_worker_mode != "suggest":
            return
        dlg = self._bhtom_dialog
        if dlg is None or not shb.isValid(dlg):
            return
        dlg.update_suggestions(suggestions, notes)
        if page < 0:
            dlg.set_source_message(_bhtom_suggestion_source_message("cache"))
            dlg.set_loading_state(True, "Loading BHTOM targets from cache...")
        else:
            dlg.set_source_message(_bhtom_suggestion_source_message("network"))
            dlg.set_loading_state(True, f"Loading BHTOM targets... page {page}")


    def _on_bhtom_worker_finished(self, worker: BhtomSuggestionWorker) -> None:
        if self._bhtom_worker is worker:
            self._bhtom_worker = None
        if not self._bhtom_worker_mode:
            self._bhtom_worker_cache_key = None


    def _on_bhtom_worker_completed(
        self,
        request_id: int,
        suggestions: list[dict[str, object]],
        notes: list[str],
        raw_candidates: list[dict[str, object]],
        error: str,
    ) -> None:
        if request_id != self._bhtom_worker_request_id:
            return
        mode = self._bhtom_worker_mode
        self._bhtom_worker_mode = ""
        cache_key = self._bhtom_worker_cache_key
        self._bhtom_worker_cache_key = None
        source = str(self._bhtom_worker_source or "").strip().lower()
        self._bhtom_worker_source = ""

        if error:
            self._set_bhtom_status("BHTOM: cancelled" if error == "cancelled" else "BHTOM: error", busy=False)
            if error != "cancelled":
                logger.warning("BHTOM suggestion worker failed: %s", error)
                if mode == "suggest":
                    dlg = self._bhtom_dialog
                    if dlg is not None and shb.isValid(dlg):
                        dlg.set_loading_state(False, "Loading failed")
                        dlg.set_source_message(_bhtom_suggestion_source_message(source or "loading"))
                        if not dlg.table_model.total_count():
                            dlg.notes_label.setText(f"Notes: {error}")
                            dlg.notes_label.setVisible(True)
                    else:
                        QMessageBox.warning(self._planner, "Suggest Targets", error)
                elif mode == "quick":
                    QMessageBox.warning(self._planner, "Quick Targets", error)
            self._set_ai_status("Ready", tone="info")
            return

        cached_candidate_count = 0
        if cache_key is not None and raw_candidates:
            self._bhtom_candidate_cache_key = cache_key
            self._bhtom_candidate_cache = list(raw_candidates)
            self._bhtom_candidate_cache_loaded_at = perf_counter()
            cached_candidate_count = len(raw_candidates)
            storage = getattr(self, "app_storage", None)
            if storage is not None:
                try:
                    storage.cache.set_json(
                        "bhtom_candidates",
                        self._bhtom_storage_cache_key(token=cache_key[1], base_url=cache_key[0]),
                        self._serialize_bhtom_candidates(raw_candidates),
                        ttl_s=BHTOM_SUGGESTION_CACHE_TTL_S,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to persist BHTOM candidates from worker: %s", exc)
        elif cache_key is not None and self._bhtom_candidate_cache_key == cache_key and self._bhtom_candidate_cache:
            cached_candidate_count = len(self._bhtom_candidate_cache)
        if source == "network" and cache_key is not None:
            self._bhtom_last_network_fetch_key = cache_key
        self._bhtom_ranked_suggestions_cache = list(suggestions)

        if mode == "suggest":
            dlg = self._bhtom_dialog
            if dlg is not None and shb.isValid(dlg):
                dlg.update_suggestions(suggestions, notes)
                dlg.set_source_message(_bhtom_suggestion_source_message(source or "network"))
                dlg.set_loading_state(False, "Loaded")
            if cached_candidate_count > 0:
                self._set_bhtom_status(
                    f"BHTOM: ready ({len(suggestions)} ranked / {cached_candidate_count} cached)",
                    busy=False,
                )
            else:
                self._set_bhtom_status(f"BHTOM: ready ({len(suggestions)} ranked)", busy=False)
        elif mode == "quick":
            self._apply_quick_targets_from_suggestions(suggestions, notes)
            if cached_candidate_count > 0:
                self._set_bhtom_status(
                    f"BHTOM: ready ({len(suggestions)} ranked / {cached_candidate_count} cached)",
                    busy=False,
                )
            else:
                self._set_bhtom_status(f"BHTOM: ready ({len(suggestions)} ranked)", busy=False)
        else:
            self._set_bhtom_status("BHTOM: idle", busy=False)

        self._set_ai_status("Ready", tone="info")


    def _on_suggest_dialog_closed(self, _result: int) -> None:
        if self._bhtom_worker_mode == "suggest":
            self._cancel_bhtom_worker()
            self._set_bhtom_status("BHTOM: cancelled", busy=False)
        self._bhtom_dialog = None
        self._set_ai_status("Ready", tone="info")


    def _build_local_target_suggestions(self, *, force_refresh: bool = False) -> tuple[list[dict[str, object]], list[str]]:
        context, error = self._build_bhtom_suggestion_context()
        if context is None:
            return [], [error or "Unable to prepare BHTOM context."]
        candidates = self._fetch_bhtom_target_candidates(force_refresh=force_refresh)
        suggestions, notes = _rank_local_target_suggestions_from_candidates(
            payload=context["payload"],  # type: ignore[index]
            site=context["site"],  # type: ignore[index]
            targets=context["targets"],  # type: ignore[index]
            limit_altitude=float(context["limit_altitude"]),
            sun_alt_limit=float(context["sun_alt_limit"]),
            min_moon_sep=float(context["min_moon_sep"]),
            candidates=candidates,
        )
        self._bhtom_ranked_suggestions_cache = list(suggestions)
        return suggestions, notes


    def _format_local_target_suggestions(self, suggestions: list[dict[str, object]], notes: list[str]) -> str:
        if not suggestions:
            return "\n".join(notes) if notes else "No suggestions are available."

        min_moon_sep = float(self.min_moon_sep_spin.value()) if hasattr(self, "min_moon_sep_spin") else 0.0
        lines = ["Here are suitable additional targets from BHTOM for tonight:"]
        for idx, item in enumerate(suggestions, start=1):
            target = item["target"]
            metrics = item["metrics"]
            window_start = item["window_start"]
            window_end = item["window_end"]
            assert isinstance(target, Target)
            assert isinstance(metrics, TargetNightMetrics)
            assert isinstance(window_start, datetime)
            assert isinstance(window_end, datetime)
            start_txt = window_start.strftime("%H:%M")
            end_txt = window_end.strftime("%H:%M")
            target_type = target.object_type or "Object"
            reason_parts = [
                f"score {metrics.score:.1f}",
                f"above limit {metrics.hours_above_limit:.1f} h",
                f"max alt {metrics.max_altitude_deg:.0f} deg",
            ]
            importance = item.get("importance")
            if isinstance(importance, (int, float)) and float(importance) > 0:
                reason_parts.insert(0, f"BHTOM importance {float(importance):.1f}")
            bhtom_priority = item.get("bhtom_priority")
            if isinstance(bhtom_priority, int) and bhtom_priority > 0:
                reason_parts.insert(1 if reason_parts else 0, f"priority {bhtom_priority}")
            reason = ", ".join(reason_parts) + "."
            warning_line = None
            min_window_moon_sep = item.get("min_window_moon_sep")
            moon_sep_warning = bool(item.get("moon_sep_warning"))
            if moon_sep_warning and isinstance(min_window_moon_sep, (int, float)) and math.isfinite(float(min_window_moon_sep)):
                warning_line = (
                    f"Warning: Moon separation in the best window drops to {float(min_window_moon_sep):.1f} deg "
                    f"(< {min_moon_sep:.0f} deg)."
                )
            lines.extend(
                [
                    f"{idx}. {target.name}",
                    f"Type: {target_type}",
                    f"{_target_magnitude_label(target)}: {target.magnitude:.2f}" if target.magnitude is not None else f"{_target_magnitude_label(target)}: -",
                    f"Best window: {start_txt}-{end_txt}",
                    f"Reason: {reason}",
                    *( [warning_line] if warning_line else [] ),
                    "",
                ]
            )

        if notes:
            lines.append("Notes: " + " | ".join(notes))
        return "\n".join(lines).strip()


    def _quick_add_suggested_targets(self) -> None:
        self._set_ai_status("Loading suggestions...", tone="info")
        self._set_bhtom_status("BHTOM: loading quick targets...", busy=True)
        force_refresh = self._bhtom_should_fetch_from_network_now()
        if not self._start_bhtom_worker(mode="quick", emit_partials=True, force_refresh=force_refresh):
            self._set_bhtom_status("BHTOM: idle", busy=False)
            self._set_ai_status("Ready", tone="info")
            return


    def _apply_quick_targets_from_suggestions(
        self,
        suggestions: list[dict[str, object]],
        notes: list[str],
    ) -> None:
        if not suggestions:
            QMessageBox.information(
                self,
                "Quick Targets",
                "\n".join(notes) if notes else "No BHTOM targets matched the current night window.",
            )
            return

        cfg = self._quick_targets_config()
        quick_count = int(cfg["count"])
        min_importance = float(cfg["min_importance"])
        min_score = (
            float(self.min_score_spin.value())
            if bool(cfg["use_score_filter"]) and hasattr(self, "min_score_spin")
            else 0.0
        )
        min_moon_sep = (
            float(self.min_moon_sep_spin.value())
            if bool(cfg["use_moon_filter"]) and hasattr(self, "min_moon_sep_spin")
            else 0.0
        )
        use_limiting_mag = bool(cfg["use_limiting_mag"])
        limiting_mag = float(self._current_limiting_magnitude())

        filtered: list[dict[str, object]] = []
        for item in suggestions:
            target = item.get("target")
            metrics = item.get("metrics")
            if not isinstance(target, Target) or not isinstance(metrics, TargetNightMetrics):
                continue
            if float(item.get("importance", 0.0) or 0.0) < min_importance:
                continue
            if float(metrics.score) < min_score:
                continue
            min_sep = _safe_float(item.get("min_window_moon_sep"))
            if min_moon_sep > 0.0:
                if min_sep is None or not math.isfinite(min_sep) or min_sep < min_moon_sep:
                    continue
            if use_limiting_mag and target.magnitude is not None and math.isfinite(float(target.magnitude)):
                if float(target.magnitude) > limiting_mag:
                    continue
            filtered.append(item)

        filtered.sort(
            key=lambda item: (
                -float(item["metrics"].score),  # type: ignore[index]
                -float(item["metrics"].hours_above_limit),  # type: ignore[index]
                -float(item.get("importance", 0.0) or 0.0),
                str(item["target"].name).lower(),  # type: ignore[index]
            )
        )

        quick_rows = filtered[:quick_count]
        if not quick_rows:
            details: list[str] = []
            if min_score > 0.0:
                details.append(f"score≥{min_score:.0f}")
            if min_moon_sep > 0.0:
                details.append(f"moon≥{min_moon_sep:.0f}°")
            if use_limiting_mag:
                details.append(f"mag≤{limiting_mag:.1f}")
            details_txt = f" ({', '.join(details)})" if details else ""
            QMessageBox.information(
                self,
                "Quick Targets",
                f"No suggested targets matched the configured Quick Targets filters{details_txt}.",
            )
            return

        added_count = 0
        skipped_count = 0
        first_added: Optional[Target] = None
        for item in quick_rows:
            target = item.get("target")
            if not isinstance(target, Target):
                continue
            if self._append_target_to_plan(target, refresh=False, notify_duplicate=False):
                added_count += 1
                if first_added is None:
                    first_added = target
            else:
                skipped_count += 1

        if added_count > 0:
            self._recompute_recommended_order_cache()
            self._apply_table_settings()
            self._apply_default_sort()
            self._refresh_target_color_map()
            self._emit_table_data_changed()
            self._fetch_missing_magnitudes_async()
            self._replot_timer.start()
            if first_added is not None:
                for row_idx, existing in enumerate(self.targets):
                    if _targets_match(existing, first_added):
                        self.table_view.selectRow(row_idx)
                        self.table_view.scrollTo(self.table_model.index(row_idx, TargetTableModel.COL_NAME))
                        break
            self._update_selected_details()
            self._prefetch_cutouts_for_all_targets(prioritize=self._selected_target_or_none())
            self._prefetch_finder_charts_for_all_targets(prioritize=self._selected_target_or_none())
        summary = (
            f"Quick Targets: added {added_count}/{len(quick_rows)}"
            + (f", skipped {skipped_count} duplicates" if skipped_count > 0 else "")
        )
        self._set_bhtom_status("BHTOM: quick targets ready", busy=False)
        self._set_ai_status(summary, tone="info")
        status_bar = self.statusBar() if hasattr(self, "statusBar") else None
        if status_bar is not None:
            status_bar.showMessage(summary, 5000)


    def _ai_suggest_targets(self) -> None:
        self._set_ai_status("Loading suggestions...", tone="info")
        if self._bhtom_worker is not None and self._bhtom_worker.isRunning():
            QMessageBox.information(self._planner, "Suggest Targets", "A BHTOM request is already in progress.")
            self._set_ai_status("Ready", tone="info")
            return

        dlg = SuggestedTargetsDialog(
            suggestions=[],
            notes=[],
            moon_sep_threshold=float(self.min_moon_sep_spin.value()) if hasattr(self, "min_moon_sep_spin") else 0.0,
            mag_warning_threshold=self._current_limiting_magnitude(),
            initial_score_filter=float(self.min_score_spin.value()) if hasattr(self, "min_score_spin") else 0.0,
            bhtom_base_url=self._bhtom_api_base_url(),
            add_callback=self._append_target_to_plan,
            reload_callback=lambda: self._reload_local_target_suggestions(force_refresh=True),
            parent=self._planner,
        )
        dlg.set_source_message(_bhtom_suggestion_source_message("loading"))
        dlg.set_loading_state(True, "Loading BHTOM targets...")
        self._bhtom_dialog = dlg
        dlg.finished.connect(self._on_suggest_dialog_closed)
        self._set_bhtom_status("BHTOM: loading suggestions...", busy=True)
        force_refresh = self._bhtom_should_fetch_from_network_now()
        if not self._start_bhtom_worker(mode="suggest", emit_partials=True, force_refresh=force_refresh):
            self._bhtom_dialog = None
            dlg.set_loading_state(False, "Loading failed")
            self._set_bhtom_status("BHTOM: idle", busy=False)
            self._set_ai_status("Ready", tone="info")
            return
        dlg.exec()

        worker = self._llm_worker
        if worker is None or not worker.isRunning():
            self._set_ai_status("Ready", tone="info")


__all__ = ["BhtomCoordinator"]
