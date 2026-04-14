from __future__ import annotations

import json
import math
import re
from datetime import datetime
from typing import Any, Optional
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import matplotlib.dates as mdates
import numpy as np
import pytz
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astroplan import FixedTarget, Observer
from PySide6.QtCore import QThread, Signal

from astroplanner.models import DEFAULT_LIMITING_MAGNITUDE, Site, Target
from astroplanner.scoring import compute_target_metrics


BHTOM_API_BASE_URL = "https://bh-tom2.astrouw.edu.pl"
BHTOM_TARGET_LIST_PATH = "/targets/getTargetList/"
BHTOM_OBSERVATORY_LIST_PATH = "/observatory/getObservatoryList/"
BHTOM_MAX_SUGGESTION_PAGES = 5
BHTOM_MAX_OBSERVATORY_PAGES = 20
BHTOM_PAGE_SIZE = 200
BHTOM_SUGGESTION_MIN_IMPORTANCE = 2.0
BHTOM_SUGGESTION_CACHE_TTL_S = 60 * 60
BHTOM_OBSERVATORY_CACHE_TTL_S = 60 * 60


def _safe_float(value: object) -> Optional[float]:
    if value is None or np.ma.is_masked(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: object) -> Optional[int]:
    if value is None or np.ma.is_masked(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None


def _normalize_catalog_token(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _normalize_catalog_display_name(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return " ".join(text.split())


def _airmass_from_altitude_values(altitude_deg: object) -> np.ndarray:
    altitude = np.asarray(altitude_deg, dtype=float)
    airmass = np.full_like(altitude, np.nan, dtype=float)
    valid = np.isfinite(altitude) & (altitude > 0.0)
    if np.any(valid):
        alt_valid = altitude[valid]
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            denom = np.sin(np.radians(alt_valid)) + 0.50572 * np.power(alt_valid + 6.07995, -1.6364)
            airmass[valid] = 1.0 / denom
    return airmass


def _pick_first_present(sources: list[dict[str, Any]], *keys: str) -> object:
    for source in sources:
        for key in keys:
            if key in source and source[key] not in (None, ""):
                return source[key]
    return None


def _fetch_bhtom_target_page_payload(endpoint_base_url: str, token: str, page: int) -> object:
    endpoint = f"{endpoint_base_url.rstrip('/')}{BHTOM_TARGET_LIST_PATH}"
    body = json.dumps(
        {
            "page": int(page),
            "type": "SIDEREAL",
            "importanceMin": BHTOM_SUGGESTION_MIN_IMPORTANCE,
        }
    ).encode("utf-8")
    req = Request(
        endpoint,
        data=body,
        headers={
            "Authorization": f"Token {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "AstroPlanner/1.0 (desktop app)",
        },
    )
    try:
        with urlopen(req, timeout=20) as resp:
            raw = resp.read().decode("utf-8")
    except HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="ignore").strip()
        except Exception:
            detail = ""
        if exc.code in {401, 403}:
            raise RuntimeError("BHTOM unauthorized (401/403). Check BHTOM account/token.") from exc
        if detail:
            raise RuntimeError(f"BHTOM request failed ({exc.code}): {detail[:240]}") from exc
        raise RuntimeError(f"BHTOM request failed ({exc.code}).") from exc
    except Exception as exc:
        raise RuntimeError(f"BHTOM lookup failed: {exc}") from exc

    try:
        return json.loads(raw)
    except Exception as exc:
        raise RuntimeError("BHTOM returned non-JSON response.") from exc


def _extract_bhtom_items(payload: object) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []

    for key in ("results", "targets", "items", "data", "objects"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if isinstance(value, dict):
            for nested_key in ("results", "targets", "items", "data", "objects"):
                nested_value = value.get(nested_key)
                if isinstance(nested_value, list):
                    return [item for item in nested_value if isinstance(item, dict)]
    return []


def _bhtom_payload_has_more(payload: object, page: int, item_count: int) -> bool:
    if isinstance(payload, dict):
        next_value = payload.get("next")
        if next_value not in (None, "", False):
            return True
        for key in ("has_next", "hasNext"):
            if payload.get(key) is True:
                return True
        total_pages = None
        for key in ("totalPages", "total_pages", "numPages", "num_pages", "pages", "page_count"):
            total_pages = _safe_int(payload.get(key))
            if total_pages is not None:
                break
        if total_pages is not None:
            return page < total_pages
        total_count = None
        for key in ("count", "total", "totalCount", "recordsTotal"):
            total_count = _safe_int(payload.get(key))
            if total_count is not None:
                break
        if total_count is not None:
            return page * BHTOM_PAGE_SIZE < total_count
    return item_count >= BHTOM_PAGE_SIZE and page < BHTOM_MAX_SUGGESTION_PAGES


def _fetch_bhtom_observatory_page_payload(endpoint_base_url: str, token: str, page: int) -> object:
    endpoint = f"{endpoint_base_url.rstrip('/')}{BHTOM_OBSERVATORY_LIST_PATH}"
    body = urlencode({"page": int(page)}).encode("utf-8")
    req = Request(
        endpoint,
        data=body,
        headers={
            "Authorization": f"Token {token}",
            "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
            "Accept": "application/json",
            "User-Agent": "AstroPlanner/1.0 (desktop app)",
        },
    )
    try:
        with urlopen(req, timeout=20) as resp:
            raw = resp.read().decode("utf-8")
    except HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="ignore").strip()
        except Exception:
            detail = ""
        if exc.code in {401, 403}:
            raise RuntimeError("BHTOM unauthorized (401/403). Check BHTOM account/token.") from exc
        if detail:
            raise RuntimeError(f"BHTOM observatory request failed ({exc.code}): {detail[:240]}") from exc
        raise RuntimeError(f"BHTOM observatory request failed ({exc.code}).") from exc
    except Exception as exc:
        raise RuntimeError(f"BHTOM observatory lookup failed: {exc}") from exc

    try:
        return json.loads(raw)
    except Exception as exc:
        raise RuntimeError("BHTOM observatory endpoint returned non-JSON response.") from exc


def _extract_bhtom_observatory_items(payload: object) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []

    for key in ("data", "results", "observatories", "items"):
        value = payload.get(key)
        if isinstance(value, list):
            if len(value) == 1 and isinstance(value[0], list):
                return [item for item in value[0] if isinstance(item, dict)]
            rows = [item for item in value if isinstance(item, dict)]
            if rows:
                return rows
        if isinstance(value, dict):
            for nested_key in ("data", "results", "observatories", "items"):
                nested_value = value.get(nested_key)
                if isinstance(nested_value, list):
                    return [item for item in nested_value if isinstance(item, dict)]

    if "name" in payload and ("lat" in payload or "latitude" in payload):
        return [payload]
    return []


def _bhtom_observatory_payload_has_more(payload: object, page: int, item_count: int) -> bool:
    if isinstance(payload, dict):
        next_value = payload.get("next")
        if next_value not in (None, "", False):
            return True
        for key in ("has_next", "hasNext"):
            if payload.get(key) is True:
                return True

        total_pages = None
        for key in ("num_pages", "numPages", "total_pages", "totalPages", "pages", "page_count"):
            total_pages = _safe_int(payload.get(key))
            if total_pages is not None:
                break
        current_page = None
        for key in ("page", "current_page", "currentPage"):
            current_page = _safe_int(payload.get(key))
            if current_page is not None:
                break
        if total_pages is not None:
            current = current_page if current_page is not None else int(page)
            return current < total_pages
    return item_count >= BHTOM_PAGE_SIZE and page < BHTOM_MAX_OBSERVATORY_PAGES


def _bhtom_camera_detector_shape(camera: dict[str, Any]) -> tuple[int, int]:
    def _dimension(*keys: str) -> int:
        value = _safe_int(_pick_first_present([camera], *keys))
        if value is None or value <= 0:
            return 0
        return int(value)

    width = _dimension(
        "detector_width_px",
        "detectorWidthPx",
        "width_px",
        "width",
        "x_size",
        "xSize",
        "nx",
        "res_x",
        "resolution_x",
    )
    height = _dimension(
        "detector_height_px",
        "detectorHeightPx",
        "height_px",
        "height",
        "y_size",
        "ySize",
        "ny",
        "res_y",
        "resolution_y",
    )
    if width > 0 and height > 0:
        return width, height

    shape_raw = _pick_first_present([camera], "detector_shape", "detectorShape", "resolution")
    shape_txt = str(shape_raw or "").strip().lower().replace(" ", "")
    for sep in ("x", "×", "*"):
        if sep in shape_txt:
            parts = shape_txt.split(sep)
            if len(parts) >= 2:
                try:
                    width = int(float(parts[0]))
                    height = int(float(parts[1]))
                except Exception:
                    continue
                if width > 0 and height > 0:
                    return width, height
    return 0, 0


def _build_bhtom_observatory_presets(items: list[dict[str, Any]]) -> list[dict[str, object]]:
    presets: list[dict[str, object]] = []
    used_keys: set[str] = set()
    for obs in items:
        obs_name = _normalize_catalog_display_name(
            _pick_first_present([obs], "name", "observatory_name", "title", "site_name")
        )
        if not obs_name:
            continue

        lat = _safe_float(_pick_first_present([obs], "lat", "latitude"))
        lon = _safe_float(_pick_first_present([obs], "lon", "longitude"))
        if lat is None or lon is None:
            continue
        elev = _safe_float(_pick_first_present([obs], "altitude", "elevation", "alt")) or 0.0
        lim_mag = _safe_float(
            _pick_first_present([obs], "approx_lim_mag", "limiting_magnitude", "limitingMagnitude")
        )
        aperture_raw = _safe_float(
            _pick_first_present([obs], "aperture", "telescope_diameter_mm", "diameter_mm")
        )
        aperture_mm: Optional[float] = None
        if aperture_raw is not None and math.isfinite(aperture_raw) and aperture_raw > 0:
            aperture_mm = float(aperture_raw * 1000.0 if aperture_raw <= 30.0 else aperture_raw)
        focal_mm = _safe_float(_pick_first_present([obs], "focal_length", "focal_length_mm", "focalLengthMm"))
        obs_id_raw = _pick_first_present([obs], "id", "pk", "observatory_id")
        obs_id = str(obs_id_raw).strip() if obs_id_raw is not None else obs_name

        camera_rows: list[dict[str, Any]] = []
        cameras_payload = obs.get("cameras")
        if isinstance(cameras_payload, list):
            camera_rows = [cam for cam in cameras_payload if isinstance(cam, dict)]

        if not camera_rows:
            key = f"obs:{obs_id}|cam:none"
            while key in used_keys:
                key = f"{key}-"
            used_keys.add(key)
            site = Site(
                name=obs_name,
                latitude=float(lat),
                longitude=float(lon),
                elevation=float(elev),
                limiting_magnitude=float(lim_mag if lim_mag is not None else DEFAULT_LIMITING_MAGNITUDE),
                telescope_diameter_mm=float(aperture_mm if aperture_mm is not None else 0.0),
                focal_length_mm=float(focal_mm if focal_mm is not None else 0.0),
                pixel_size_um=0.0,
                detector_width_px=0,
                detector_height_px=0,
            )
            presets.append(
                {
                    "key": key,
                    "label": obs_name,
                    "source": "bhtom",
                    "site": site,
                    "observatory_id": obs_id,
                    "camera_id": "",
                }
            )
            continue

        for cam_idx, cam in enumerate(camera_rows):
            cam_name = _normalize_catalog_display_name(
                _pick_first_present([cam], "camera_name", "name", "prefix", "model", "code")
            )
            prefix = _normalize_catalog_display_name(_pick_first_present([cam], "prefix"))
            cam_id_raw = _pick_first_present([cam], "id", "pk", "camera_id")
            cam_id = str(cam_id_raw).strip() if cam_id_raw is not None else (prefix or cam_name or f"cam{cam_idx + 1}")
            label_suffix = prefix or cam_name or f"camera {cam_idx + 1}"
            key = f"obs:{obs_id}|cam:{cam_id}"
            while key in used_keys:
                key = f"{key}-"
            used_keys.add(key)

            pixel_um = _safe_float(_pick_first_present([cam], "pixel_size_um", "pixel_size", "pixelSize"))
            det_w, det_h = _bhtom_camera_detector_shape(cam)
            site = Site(
                name=f"{obs_name} [{label_suffix}]",
                latitude=float(lat),
                longitude=float(lon),
                elevation=float(elev),
                limiting_magnitude=float(lim_mag if lim_mag is not None else DEFAULT_LIMITING_MAGNITUDE),
                telescope_diameter_mm=float(aperture_mm if aperture_mm is not None else 0.0),
                focal_length_mm=float(focal_mm if focal_mm is not None else 0.0),
                pixel_size_um=float(pixel_um if pixel_um is not None else 0.0),
                detector_width_px=int(det_w),
                detector_height_px=int(det_h),
            )
            presets.append(
                {
                    "key": key,
                    "label": f"{obs_name} | {label_suffix}",
                    "source": "bhtom",
                    "site": site,
                    "observatory_id": obs_id,
                    "camera_id": cam_id,
                }
            )

    presets.sort(key=lambda item: str(item.get("label", "")).lower())
    return presets


def _build_bhtom_candidate_from_item(item: dict[str, Any]) -> Optional[dict[str, object]]:
    nested_sources = [item]
    for key in ("target", "target_data", "data", "object", "coordinates"):
        nested = item.get(key)
        if isinstance(nested, dict):
            nested_sources.append(nested)

    name_raw = _pick_first_present(nested_sources, "name", "target_name", "display_name", "identifier")
    name = _normalize_catalog_display_name(name_raw)
    if not name:
        return None

    ra_deg = _safe_float(_pick_first_present(nested_sources, "ra", "raDeg", "ra_deg", "rightAscension"))
    dec_deg = _safe_float(_pick_first_present(nested_sources, "dec", "decDeg", "dec_deg", "declination"))
    if ra_deg is None or dec_deg is None:
        return None

    classification = str(
        _pick_first_present(
            nested_sources,
            "classification",
            "object_type",
            "objectType",
            "targetClass",
            "class",
        )
        or ""
    ).strip()
    magnitude = _safe_float(
        _pick_first_present(
            nested_sources,
            "mag_last",
            "last_mag",
            "lastMagnitude",
            "magnitude",
            "mag",
        )
    )
    importance = _safe_float(_pick_first_present(nested_sources, "importance", "importance_value")) or 0.0
    bhtom_priority = _safe_int(_pick_first_present(nested_sources, "priority", "observing_priority")) or 0
    target_priority = max(1, min(5, int(bhtom_priority))) if bhtom_priority > 0 else 3
    sun_separation = _safe_float(
        _pick_first_present(nested_sources, "sun_separation", "sunSeparation", "sun")
    )
    source_id_raw = _pick_first_present(nested_sources, "id", "pk", "target_id", "targetId", "name")
    source_id = str(source_id_raw).strip() if source_id_raw is not None else name

    return {
        "target": Target(
            name=name,
            ra=float(ra_deg),
            dec=float(dec_deg),
            source_catalog="bhtom",
            source_object_id=source_id or name,
            magnitude=magnitude,
            object_type=classification,
            priority=target_priority,
        ),
        "importance": float(importance),
        "bhtom_priority": int(bhtom_priority),
        "sun_separation": float(sun_separation) if sun_separation is not None else None,
    }


def dedupe_bhtom_candidates(candidates: list[dict[str, object]]) -> list[dict[str, object]]:
    deduped: list[dict[str, object]] = []
    seen_keys: set[str] = set()
    for candidate in candidates:
        target = candidate.get("target")
        if not isinstance(target, Target):
            continue
        dedupe_key = _normalize_catalog_token(target.source_object_id or target.name)
        if not dedupe_key or dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        deduped.append(candidate)
    return deduped


def _rank_local_target_suggestions_from_candidates(
    payload: dict[str, object],
    site: Site,
    targets: list[Target],
    limit_altitude: float,
    sun_alt_limit: float,
    min_moon_sep: float,
    candidates: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[str]]:
    skipped_notes: list[str] = []
    try:
        tz_name = str(payload.get("tz", site.timezone_name))
        tz = pytz.timezone(tz_name)
    except Exception:
        tz_name = site.timezone_name
        tz = pytz.UTC

    try:
        time_datetimes = [t.astimezone(tz) for t in mdates.num2date(payload["times"])]
    except Exception:
        return [], ["Visibility samples are unavailable in the current plot state."]
    if not time_datetimes:
        return [], ["Visibility samples are unavailable in the current plot state."]

    observer = Observer(location=site.to_earthlocation(), timezone=tz_name)
    time_samples = Time(time_datetimes)
    moon_ra = np.array(payload.get("moon_ra", np.full(len(time_datetimes), np.nan)), dtype=float)
    moon_dec = np.array(payload.get("moon_dec", np.full(len(time_datetimes), np.nan)), dtype=float)
    if moon_ra.shape[0] != len(time_datetimes) or moon_dec.shape[0] != len(time_datetimes):
        return [], ["Moon position samples are unavailable in the current plot state."]
    moon_coords = SkyCoord(ra=moon_ra * u.deg, dec=moon_dec * u.deg)

    sun_alt_series = np.array(payload.get("sun_alt", np.full(len(time_datetimes), np.nan)), dtype=float)
    obs_sun_mask = np.isfinite(sun_alt_series) & (sun_alt_series <= float(sun_alt_limit))
    sample_hours = 24.0 / max(len(time_datetimes) - 1, 1)

    current_names = {_normalize_catalog_token(target.name) for target in targets}
    current_source_ids = {_normalize_catalog_token(target.source_object_id) for target in targets if target.source_object_id}
    current_coords = [target.skycoord for target in targets]

    ranked: list[dict[str, object]] = []

    for candidate_info in candidates:
        target = candidate_info.get("target")
        if not isinstance(target, Target):
            continue
        normalized_name = _normalize_catalog_token(target.name)
        normalized_source = _normalize_catalog_token(target.source_object_id)
        if normalized_name in current_names or (normalized_source and normalized_source in current_source_ids):
            continue
        if any(float(target.skycoord.separation(coord).deg) < 0.05 for coord in current_coords):
            continue

        try:
            fixed = FixedTarget(name=target.name, coord=target.skycoord)
            altaz = observer.altaz(time_samples, fixed)
            altitude = np.array(altaz.alt.deg, dtype=float)  # type: ignore[arg-type]
            moon_sep = np.array(target.skycoord.separation(moon_coords).deg, dtype=float)
            metrics = compute_target_metrics(
                altitude_deg=altitude,
                moon_sep_deg=moon_sep,
                limit_altitude=float(limit_altitude),
                sample_hours=sample_hours,
                priority=max(1, min(5, int(candidate_info.get("bhtom_priority", 3) or 3))),
                observed=False,
                valid_mask=obs_sun_mask,
            )
        except Exception:
            continue

        valid_mask = np.isfinite(altitude) & (altitude >= float(limit_altitude)) & obs_sun_mask
        if not valid_mask.any():
            continue

        valid_indices = np.where(valid_mask)[0]
        runs = np.split(valid_indices, np.where(np.diff(valid_indices) != 1)[0] + 1)
        best_run = max(runs, key=len)
        start_idx = int(best_run[0])
        end_idx = min(int(best_run[-1]) + 1, len(time_datetimes) - 1)
        window_start = time_datetimes[start_idx]
        window_end = time_datetimes[end_idx]
        night_airmass = _airmass_from_altitude_values(altitude[valid_mask])
        finite_night_airmass = night_airmass[np.isfinite(night_airmass)]
        best_airmass = float(np.min(finite_night_airmass)) if finite_night_airmass.size > 0 else None
        finite_window_moon_sep = moon_sep[best_run][np.isfinite(moon_sep[best_run])]
        min_window_moon_sep = (
            float(np.min(finite_window_moon_sep))
            if finite_window_moon_sep.size > 0
            else None
        )

        ranked.append(
            {
                "target": target,
                "metrics": metrics,
                "window_start": window_start,
                "window_end": window_end,
                "best_airmass": best_airmass,
                "min_window_moon_sep": min_window_moon_sep,
                "moon_sep_warning": (
                    float(min_moon_sep) > 0.0
                    and min_window_moon_sep is not None
                    and round(float(min_window_moon_sep), 1) < float(min_moon_sep)
                ),
                "added_to_plan": False,
                "importance": float(candidate_info.get("importance", 0.0) or 0.0),
                "bhtom_priority": int(candidate_info.get("bhtom_priority", 0) or 0),
                "sun_separation": candidate_info.get("sun_separation"),
            }
        )

    if not ranked:
        skipped_notes.append("No BHTOM targets matched the current filters and night window.")
        return [], skipped_notes

    ranked.sort(
        key=lambda item: (
            -float(item["importance"]),
            -int(item["bhtom_priority"]),
            -float(item["metrics"].score),  # type: ignore[index]
            -float(item["metrics"].hours_above_limit),  # type: ignore[index]
            item["window_start"],
            str(item["target"].name).lower(),  # type: ignore[index]
        )
    )
    return ranked, skipped_notes


def _format_duration_short(total_seconds: float) -> str:
    seconds = max(0, int(round(float(total_seconds))))
    if seconds % 3600 == 0:
        hours = max(1, seconds // 3600)
        return f"{hours}h"
    if seconds % 60 == 0:
        minutes = max(1, seconds // 60)
        return f"{minutes} min"
    return f"{seconds}s"


def bhtom_suggestion_source_message(source: str) -> str:
    ttl_txt = _format_duration_short(BHTOM_SUGGESTION_CACHE_TTL_S)
    normalized = str(source or "").strip().lower()
    if normalized == "cache":
        return (
            f"Source: cached BHTOM target list (TTL {ttl_txt}). "
            "Suggest Targets does not auto-refresh from the network while this cache is still valid."
        )
    if normalized == "network":
        return f"Source: fresh BHTOM target list fetched from the API. Results are cached for {ttl_txt}."
    return (
        f"Source: checking cache first (TTL {ttl_txt}). "
        "If no valid cache is found, AstroPlanner fetches fresh data from BHTOM."
    )


_bhtom_suggestion_source_message = bhtom_suggestion_source_message


class BhtomSuggestionWorker(QThread):
    """Background loader/ranker for BHTOM suggestions with incremental page updates."""

    pageReady = Signal(int, list, list, int, int)
    completed = Signal(int, list, list, list, str)

    def __init__(
        self,
        request_id: int,
        payload: dict[str, object],
        site: Site,
        targets: list[Target],
        limit_altitude: float,
        sun_alt_limit: float,
        min_moon_sep: float,
        bhtom_base_url: str,
        bhtom_token: str,
        cached_candidates: Optional[list[dict[str, object]]] = None,
        emit_partials: bool = True,
        parent=None,
    ):
        super().__init__(parent)
        self.setObjectName(self.__class__.__name__)
        self.request_id = int(request_id)
        self.payload = payload
        self.site = site
        self.targets = targets
        self.limit_altitude = float(limit_altitude)
        self.sun_alt_limit = float(sun_alt_limit)
        self.min_moon_sep = float(min_moon_sep)
        self.bhtom_base_url = bhtom_base_url
        self.bhtom_token = bhtom_token
        self.cached_candidates = list(cached_candidates) if cached_candidates else None
        self.emit_partials = bool(emit_partials)

    def run(self):
        try:
            candidates: list[dict[str, object]] = list(self.cached_candidates or [])
            final_notes: list[str] = []
            final_ranked: list[dict[str, object]] = []

            if candidates:
                final_ranked, final_notes = _rank_local_target_suggestions_from_candidates(
                    payload=self.payload,
                    site=self.site,
                    targets=self.targets,
                    limit_altitude=self.limit_altitude,
                    sun_alt_limit=self.sun_alt_limit,
                    min_moon_sep=self.min_moon_sep,
                    candidates=candidates,
                )
                if self.emit_partials:
                    self.pageReady.emit(
                        self.request_id,
                        final_ranked,
                        final_notes,
                        -1,
                        len(candidates),
                    )
            else:
                candidates = []
                for page in range(1, BHTOM_MAX_SUGGESTION_PAGES + 1):
                    if self.isInterruptionRequested():
                        self.completed.emit(self.request_id, [], [], [], "cancelled")
                        return

                    payload = _fetch_bhtom_target_page_payload(
                        endpoint_base_url=self.bhtom_base_url,
                        token=self.bhtom_token,
                        page=page,
                    )
                    items = _extract_bhtom_items(payload)
                    if not items:
                        if page == 1:
                            if isinstance(payload, dict):
                                keys = ", ".join(sorted(str(key) for key in payload.keys()))
                                raise RuntimeError(f"BHTOM returned an unexpected payload shape (keys: {keys or 'none'}).")
                            raise RuntimeError("BHTOM returned an unexpected payload shape.")
                        break

                    page_candidates: list[dict[str, object]] = []
                    for item in items:
                        if self.isInterruptionRequested():
                            self.completed.emit(self.request_id, [], [], [], "cancelled")
                            return
                        candidate = _build_bhtom_candidate_from_item(item)
                        if candidate is not None:
                            page_candidates.append(candidate)
                    candidates = dedupe_bhtom_candidates([*candidates, *page_candidates])

                    final_ranked, final_notes = _rank_local_target_suggestions_from_candidates(
                        payload=self.payload,
                        site=self.site,
                        targets=self.targets,
                        limit_altitude=self.limit_altitude,
                        sun_alt_limit=self.sun_alt_limit,
                        min_moon_sep=self.min_moon_sep,
                        candidates=candidates,
                    )

                    if self.emit_partials:
                        self.pageReady.emit(
                            self.request_id,
                            final_ranked,
                            final_notes,
                            page,
                            len(candidates),
                        )

                    if not _bhtom_payload_has_more(payload, page, len(items)):
                        break

            self.completed.emit(self.request_id, final_ranked, final_notes, candidates, "")
        except Exception as exc:  # noqa: BLE001
            self.completed.emit(self.request_id, [], [], [], str(exc))


class BhtomCandidatePrefetchWorker(QThread):
    """Background loader for BHTOM candidate cache refresh (no ranking)."""

    completed = Signal(int, list, str)

    def __init__(self, request_id: int, base_url: str, token: str, parent=None):
        super().__init__(parent)
        self.setObjectName(self.__class__.__name__)
        self.request_id = int(request_id)
        self.base_url = str(base_url)
        self.token = str(token)

    def run(self):
        try:
            candidates: list[dict[str, object]] = []
            for page in range(1, BHTOM_MAX_SUGGESTION_PAGES + 1):
                if self.isInterruptionRequested():
                    self.completed.emit(self.request_id, [], "cancelled")
                    return
                payload = _fetch_bhtom_target_page_payload(
                    endpoint_base_url=self.base_url,
                    token=self.token,
                    page=page,
                )
                items = _extract_bhtom_items(payload)
                if not items:
                    break
                page_candidates: list[dict[str, object]] = []
                for item in items:
                    if self.isInterruptionRequested():
                        self.completed.emit(self.request_id, [], "cancelled")
                        return
                    candidate = _build_bhtom_candidate_from_item(item)
                    if candidate is not None:
                        page_candidates.append(candidate)
                candidates = dedupe_bhtom_candidates([*candidates, *page_candidates])
                if not _bhtom_payload_has_more(payload, page, len(items)):
                    break
            if not candidates:
                raise RuntimeError("BHTOM returned no usable target candidates.")
            self.completed.emit(self.request_id, candidates, "")
        except Exception as exc:  # noqa: BLE001
            self.completed.emit(self.request_id, [], str(exc))


class BhtomObservatoryPresetWorker(QThread):
    """Background loader for BHTOM observatory/camera presets."""

    progress = Signal(int, int, str)
    completed = Signal(int, list, str)

    def __init__(self, request_id: int, base_url: str, token: str, parent=None):
        super().__init__(parent)
        self.setObjectName(self.__class__.__name__)
        self.request_id = int(request_id)
        self.base_url = str(base_url)
        self.token = str(token)

    def run(self):
        try:
            items: list[dict[str, Any]] = []
            for page in range(1, BHTOM_MAX_OBSERVATORY_PAGES + 1):
                if self.isInterruptionRequested():
                    self.completed.emit(self.request_id, [], "cancelled")
                    return
                payload = _fetch_bhtom_observatory_page_payload(
                    endpoint_base_url=self.base_url,
                    token=self.token,
                    page=page,
                )
                page_items = _extract_bhtom_observatory_items(payload)
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
                self.progress.emit(page, BHTOM_MAX_OBSERVATORY_PAGES, f"Loading BHTOM presets... page {page}")
                if not _bhtom_observatory_payload_has_more(payload, page, len(page_items)):
                    break
            presets = _build_bhtom_observatory_presets(items)
            if not presets:
                raise RuntimeError("BHTOM returned no usable observatory/camera presets.")
            self.completed.emit(self.request_id, presets, "")
        except Exception as exc:  # noqa: BLE001
            self.completed.emit(self.request_id, [], str(exc))


__all__ = [
    "BHTOM_API_BASE_URL",
    "BHTOM_MAX_OBSERVATORY_PAGES",
    "BHTOM_MAX_SUGGESTION_PAGES",
    "BHTOM_OBSERVATORY_CACHE_TTL_S",
    "BHTOM_OBSERVATORY_LIST_PATH",
    "BHTOM_PAGE_SIZE",
    "BHTOM_SUGGESTION_CACHE_TTL_S",
    "BHTOM_SUGGESTION_MIN_IMPORTANCE",
    "BHTOM_TARGET_LIST_PATH",
    "BhtomCandidatePrefetchWorker",
    "BhtomObservatoryPresetWorker",
    "BhtomSuggestionWorker",
    "_bhtom_observatory_payload_has_more",
    "_bhtom_payload_has_more",
    "_bhtom_suggestion_source_message",
    "_build_bhtom_candidate_from_item",
    "_build_bhtom_observatory_presets",
    "_extract_bhtom_items",
    "_extract_bhtom_observatory_items",
    "_fetch_bhtom_observatory_page_payload",
    "_fetch_bhtom_target_page_payload",
    "_pick_first_present",
    "_rank_local_target_suggestions_from_candidates",
    "bhtom_suggestion_source_message",
    "dedupe_bhtom_candidates",
]
