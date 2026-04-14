from __future__ import annotations

import copy
import hashlib
import html as html_module
import io
import json
import logging
import math
import re
import threading
from datetime import datetime, timedelta, timezone
from time import perf_counter
from typing import Callable, Optional
from urllib.parse import quote, urljoin
from urllib.request import Request, urlopen

import numpy as np
from PySide6.QtCore import QThread, Signal

from astroplanner.storage import AppStorage

try:
    from PIL import Image, ImageDraw
except Exception:  # pragma: no cover - optional runtime dependency
    Image = None  # type: ignore[assignment]
    ImageDraw = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


def _safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: object) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class WeatherLiveWorker(QThread):
    """Background weather loader with per-provider cache and incremental payload updates."""

    progress = Signal(str, int, int)
    partial = Signal(dict)
    completed = Signal(dict)

    _CACHE_LOCK = threading.Lock()
    _CACHE: dict[str, tuple[float, dict[str, object]]] = {}

    TTL_CONDITIONS_S = 120.0
    TTL_FORECAST_S = 600.0
    TTL_CLIMATOLOGY_S = 24.0 * 3600.0

    MONTH_NAMES = (
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    )

    def __init__(
        self,
        *,
        lat: float,
        lon: float,
        elev: float,
        custom_conditions_url: str = "",
        cloud_map_source: str = "earthenv",
        cloud_map_month: int = 1,
        force_refresh: bool = False,
        include_cloud_map: bool = True,
        include_satellite: bool = True,
        storage: AppStorage | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setObjectName(self.__class__.__name__)
        self.lat = float(lat)
        self.lon = float(lon)
        self.elev = float(elev)
        self.custom_conditions_url = str(custom_conditions_url or "").strip()
        self.cloud_map_source = str(cloud_map_source or "earthenv").strip().lower()
        self.cloud_map_month = max(1, min(12, int(cloud_map_month)))
        self.force_refresh = bool(force_refresh)
        self.include_cloud_map = bool(include_cloud_map)
        self.include_satellite = bool(include_satellite)
        self.storage = storage

    @staticmethod
    def _http_get_text(url: str, timeout_s: float = 20.0) -> str:
        req = Request(
            str(url),
            headers={
                "User-Agent": "Mozilla/5.0 (AstroPlanner Weather)",
                "Accept": "application/json,text/html,*/*",
            },
        )
        with urlopen(req, timeout=float(timeout_s)) as resp:
            return resp.read().decode("utf-8", errors="replace")

    @staticmethod
    def _http_get_json(url: str, timeout_s: float = 20.0) -> object:
        text = WeatherLiveWorker._http_get_text(url, timeout_s=timeout_s)
        return json.loads(text)

    @staticmethod
    def _http_post_json(url: str, payload: dict[str, object], timeout_s: float = 30.0) -> object:
        data = json.dumps(payload).encode("utf-8")
        req = Request(
            str(url),
            data=data,
            headers={
                "User-Agent": "Mozilla/5.0 (AstroPlanner Weather)",
                "Content-Type": "application/json",
                "Accept": "application/json,text/plain,*/*",
            },
        )
        with urlopen(req, timeout=float(timeout_s)) as resp:
            text = resp.read().decode("utf-8", errors="replace")
        return json.loads(text)

    @staticmethod
    def _http_get_bytes(url: str, timeout_s: float = 30.0) -> bytes:
        req = Request(
            str(url),
            headers={
                "User-Agent": "Mozilla/5.0 (AstroPlanner Weather)",
                "Accept": "image/*,*/*",
            },
        )
        with urlopen(req, timeout=float(timeout_s)) as resp:
            return bytes(resp.read())

    @staticmethod
    def _to_float_list(values: object) -> list[float]:
        out: list[float] = []
        if not isinstance(values, list):
            return out
        for value in values:
            fv = _safe_float(value)
            if fv is None or not math.isfinite(fv):
                continue
            out.append(float(fv))
        return out

    @staticmethod
    def _k_to_c(value_k: Optional[float]) -> Optional[float]:
        if value_k is None:
            return None
        return float(value_k) - 273.15

    @staticmethod
    def _avg(values: list[float]) -> Optional[float]:
        if not values:
            return None
        return float(sum(values) / len(values))

    @staticmethod
    def _first(values: list[float]) -> Optional[float]:
        return float(values[0]) if values else None

    @staticmethod
    def _extract_chart_series_by_name(payload: object) -> dict[str, list[float]]:
        if not isinstance(payload, dict):
            return {}
        series = payload.get("series")
        if not isinstance(series, list):
            return {}
        out: dict[str, list[float]] = {}
        for item in series:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip().lower()
            data = WeatherLiveWorker._to_float_list(item.get("data"))
            if name and data:
                out[name] = data
        return out

    def _cache_get(self, cache_key: str, ttl_s: float) -> Optional[dict[str, object]]:
        now = perf_counter()
        with WeatherLiveWorker._CACHE_LOCK:
            bucket = WeatherLiveWorker._CACHE.get(str(cache_key))
        if isinstance(bucket, tuple) and len(bucket) == 2:
            stamp, payload = bucket
            if now - float(stamp) <= float(ttl_s) and isinstance(payload, dict):
                return copy.deepcopy(payload)
        storage = getattr(self, "storage", None)
        if storage is None:
            return None
        cached = storage.cache.get_json("weather", str(cache_key))
        if isinstance(cached, dict):
            with WeatherLiveWorker._CACHE_LOCK:
                WeatherLiveWorker._CACHE[str(cache_key)] = (perf_counter(), copy.deepcopy(cached))
            return copy.deepcopy(cached)
        return None

    def _cache_set(self, cache_key: str, payload: dict[str, object], ttl_s: float | None = None) -> None:
        with WeatherLiveWorker._CACHE_LOCK:
            WeatherLiveWorker._CACHE[str(cache_key)] = (perf_counter(), copy.deepcopy(payload))
        storage = getattr(self, "storage", None)
        if storage is not None:
            try:
                storage.cache.set_json("weather", str(cache_key), payload, ttl_s=ttl_s)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to persist weather cache %s: %s", cache_key, exc)

    def _run_cached(
        self,
        *,
        cache_key: str,
        ttl_s: float,
        fetcher: Callable[[], dict[str, object]],
    ) -> tuple[dict[str, object], bool]:
        if not self.force_refresh:
            cached = self._cache_get(cache_key, ttl_s)
            if cached is not None:
                return cached, True
        payload = fetcher()
        self._cache_set(cache_key, payload, ttl_s=ttl_s)
        return copy.deepcopy(payload), False

    @staticmethod
    def _iso_to_timestamp(value: object, *, utc_offset_seconds: int = 0) -> Optional[int]:
        text = str(value or "").strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except Exception:
            return None
        if parsed.tzinfo is None:
            raw = int(parsed.replace(tzinfo=timezone.utc).timestamp())
            return int(raw - int(utc_offset_seconds))
        return int(parsed.timestamp())

    @staticmethod
    def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        r = 6371.0
        p1 = math.radians(float(lat1))
        p2 = math.radians(float(lat2))
        dp = p2 - p1
        dl = math.radians(float(lon2) - float(lon1))
        a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
        return 2.0 * r * math.asin(max(0.0, min(1.0, math.sqrt(a))))

    @staticmethod
    def _estimate_rh_from_temp_dew(temp_c: Optional[float], dew_c: Optional[float]) -> Optional[float]:
        if temp_c is None or dew_c is None:
            return None
        try:
            t = float(temp_c)
            d = float(dew_c)
            if not math.isfinite(t) or not math.isfinite(d):
                return None
            # Magnus approximation.
            gamma_t = (17.625 * t) / (243.04 + t)
            gamma_d = (17.625 * d) / (243.04 + d)
            rh = 100.0 * math.exp(gamma_d - gamma_t)
            return max(0.0, min(100.0, float(rh)))
        except Exception:
            return None

    @staticmethod
    def _metar_cover_to_cloud_pct(cover: object) -> Optional[float]:
        text = str(cover or "").strip().upper()
        if not text:
            return None
        mapping = {
            "CLR": 0.0,
            "SKC": 0.0,
            "NCD": 0.0,
            "FEW": 15.0,
            "SCT": 40.0,
            "BKN": 75.0,
            "OVC": 100.0,
            "VV": 100.0,
        }
        for key, value in mapping.items():
            if text.startswith(key):
                return float(value)
        return None

    @staticmethod
    def _earthenv_month_name(month_idx: int) -> str:
        idx = max(1, min(12, int(month_idx))) - 1
        return WeatherLiveWorker.MONTH_NAMES[idx]

    @staticmethod
    def _latlon_to_world_px(lat: float, lon: float, zoom: int) -> tuple[float, float]:
        z = max(0, int(zoom))
        n = float(2 ** z)
        lon_norm = ((float(lon) + 180.0) / 360.0) * n
        lat_clip = max(-85.05112878, min(85.05112878, float(lat)))
        lat_rad = math.radians(lat_clip)
        y_norm = (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) * 0.5 * n
        return lon_norm * 256.0, y_norm * 256.0

    @staticmethod
    def _earthenv_tile_url(mapid: str, token: str, z: int, x: int, y: int) -> str:
        mapid_txt = str(mapid or "").strip().strip("/")
        if token:
            return f"https://earthengine.googleapis.com/map/{mapid_txt}/{z}/{x}/{y}?token={quote(str(token))}"
        return f"https://earthengine.googleapis.com/v1alpha/{mapid_txt}/tiles/{z}/{x}/{y}"

    @staticmethod
    def _estimate_value_from_palette(
        rgb: tuple[int, int, int],
        palette: list[tuple[int, int, int]],
        vmin: float,
        vmax: float,
    ) -> Optional[float]:
        if len(palette) < 2:
            return None
        r, g, b = [float(c) for c in rgb]
        best_dist = float("inf")
        best_norm = 0.0
        n_segments = len(palette) - 1
        for idx in range(n_segments):
            p0 = np.array(palette[idx], dtype=float)
            p1 = np.array(palette[idx + 1], dtype=float)
            v = p1 - p0
            denom = float(np.dot(v, v))
            if denom <= 1e-9:
                t = 0.0
            else:
                t = float(np.dot(np.array([r, g, b]) - p0, v) / denom)
                t = max(0.0, min(1.0, t))
            p = p0 + t * v
            dist = float(np.sum((np.array([r, g, b]) - p) ** 2))
            if dist < best_dist:
                best_dist = dist
                best_norm = (float(idx) + float(t)) / float(n_segments)
        return float(vmin + best_norm * (vmax - vmin))

    def _sample_earthenv_point_precise(self, month_name: str) -> Optional[float]:
        """Try to sample EarthEnv cloud value from source data endpoint (not palette)."""
        sample_url = f"https://dev-dot-earthenv-dot-map-of-life.appspot.com/sample/cloud/{self.lon:.6f}/{self.lat:.6f}"
        payload = self._http_get_json(sample_url, timeout_s=20.0)
        if not isinstance(payload, dict):
            return None

        month_key = str(month_name or "").strip().lower()
        lookup_keys = {
            month_key,
            month_key[:3],
            f"{self.cloud_map_month:02d}",
            str(self.cloud_map_month),
        }
        lookup_keys = {key for key in lookup_keys if key}

        def _to_pct(value: object) -> Optional[float]:
            fv = _safe_float(value)
            if fv is None or not math.isfinite(fv):
                return None
            if fv > 100.0:
                fv = fv / 100.0
            return max(0.0, min(100.0, float(fv)))

        # Expected from app code: {layer_id: [{value: ...}]}
        for key, value in payload.items():
            key_txt = str(key).strip().lower()
            if not any(token in key_txt for token in lookup_keys):
                continue
            if isinstance(value, list) and value:
                first = value[0]
                if isinstance(first, dict):
                    for candidate_key in ("value", "val", "cloud", "cloud_pct", "mean"):
                        pct = _to_pct(first.get(candidate_key))
                        if pct is not None:
                            return pct
                pct = _to_pct(first)
                if pct is not None:
                    return pct
            if isinstance(value, dict):
                for candidate_key in ("value", "val", "cloud", "cloud_pct", "mean"):
                    pct = _to_pct(value.get(candidate_key))
                    if pct is not None:
                        return pct
            pct = _to_pct(value)
            if pct is not None:
                return pct

        # Fallback: if endpoint returns a single scalar-like payload.
        for candidate_key in ("value", "cloud_pct", "cloud", "mean"):
            if candidate_key in payload:
                pct = _to_pct(payload.get(candidate_key))
                if pct is not None:
                    return pct
        return None

    def _fetch_open_meteo(self) -> dict[str, object]:
        lat_txt = f"{self.lat:.5f}"
        lon_txt = f"{self.lon:.5f}"
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat_txt}&longitude={lon_txt}"
            "&hourly=temperature_2m,relative_humidity_2m,cloud_cover,surface_pressure,wind_speed_10m"
            "&current=temperature_2m,relative_humidity_2m,cloud_cover,surface_pressure,wind_speed_10m"
            "&forecast_days=3&timezone=auto"
        )
        payload = self._http_get_json(url, timeout_s=20.0)
        if not isinstance(payload, dict):
            raise RuntimeError("Open-Meteo returned an unexpected payload.")

        current = payload.get("current") if isinstance(payload.get("current"), dict) else {}
        hourly = payload.get("hourly") if isinstance(payload.get("hourly"), dict) else {}
        utc_offset = int(_safe_int(payload.get("utc_offset_seconds")) or 0)
        times_raw = hourly.get("time") if isinstance(hourly.get("time"), list) else []
        ts: list[int] = []
        for item in times_raw:
            t = self._iso_to_timestamp(item, utc_offset_seconds=utc_offset)
            if t is not None:
                ts.append(int(t))

        series = {
            "timestamps": ts[:96],
            "temp_c": self._to_float_list(hourly.get("temperature_2m"))[:96],
            "wind_ms": self._to_float_list(hourly.get("wind_speed_10m"))[:96],
            "cloud_pct": self._to_float_list(hourly.get("cloud_cover"))[:96],
            "rh_pct": self._to_float_list(hourly.get("relative_humidity_2m"))[:96],
            "pressure_hpa": self._to_float_list(hourly.get("surface_pressure"))[:96],
        }

        updated_ts = self._iso_to_timestamp(current.get("time"), utc_offset_seconds=utc_offset)
        updated_utc = (
            datetime.fromtimestamp(int(updated_ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            if updated_ts is not None
            else datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        )

        provider = {
            "label": "Open-Meteo",
            "source_label": "Open-Meteo",
            "temp_c": _safe_float(current.get("temperature_2m")),
            "wind_ms": _safe_float(current.get("wind_speed_10m")),
            "cloud_pct": _safe_float(current.get("cloud_cover")),
            "rh_pct": _safe_float(current.get("relative_humidity_2m")),
            "pressure_hpa": _safe_float(current.get("surface_pressure")),
            "status": "ok",
            "updated_utc": updated_utc,
            "note": "Public no-key API (forecast + near-real-time model output).",
            "categories": ["forecast", "conditions_model"],
        }
        return {"provider": provider, "series": series}

    def _fetch_meteo_icm(self) -> dict[str, object]:
        available = self._http_get_json("https://devmgramapi.meteo.pl/meteorograms/available", timeout_s=20.0)
        date_ts: Optional[int] = None
        if isinstance(available, dict):
            gfs = available.get("gfs")
            if isinstance(gfs, list) and gfs:
                date_ts = _safe_int(gfs[-1])
        if date_ts is None:
            raise RuntimeError("Meteo ICM: missing available run timestamp.")

        icm_payload = self._http_post_json(
            "https://devmgramapi.meteo.pl/meteorograms/gfs",
            {"date": int(date_ts), "point": {"lat": self.lat, "lon": self.lon}},
            timeout_s=30.0,
        )
        if not isinstance(icm_payload, dict):
            raise RuntimeError("Meteo ICM: unexpected payload.")
        icm_data = icm_payload.get("data")
        if not isinstance(icm_data, dict):
            raise RuntimeError("Meteo ICM: missing data block.")

        temp_block = icm_data.get("airtmp_point") if isinstance(icm_data.get("airtmp_point"), dict) else {}
        wind_block = (
            icm_data.get("wind10_sd_true_prev_point")
            if isinstance(icm_data.get("wind10_sd_true_prev_point"), dict)
            else {}
        )
        cloud_block = icm_data.get("cldtot_aver") if isinstance(icm_data.get("cldtot_aver"), dict) else {}
        rh_block = icm_data.get("realhum_aver") if isinstance(icm_data.get("realhum_aver"), dict) else {}

        temp_series = self._to_float_list(temp_block.get("data"))
        wind_series = self._to_float_list(wind_block.get("data"))
        cloud_raw = self._to_float_list(cloud_block.get("data"))
        rh_series = self._to_float_list(rh_block.get("data"))
        cloud_series: list[float] = []
        if cloud_raw:
            scale = 100.0 if max(cloud_raw) <= 1.5 else 1.0
            cloud_series = [max(0.0, min(100.0, v * scale)) for v in cloud_raw]

        first_ts = _safe_int(temp_block.get("first_timestamp"))
        interval_s = _safe_int(temp_block.get("interval"))
        ts: list[int] = []
        if first_ts is not None and interval_s is not None and interval_s > 0:
            ts = [int(first_ts + idx * interval_s) for idx in range(len(temp_series))]

        series = {
            "timestamps": ts[:96],
            "temp_c": temp_series[:96],
            "wind_ms": wind_series[:96],
            "cloud_pct": cloud_series[:96],
            "rh_pct": rh_series[:96],
            "pressure_hpa": [],
        }
        provider = {
            "label": "Meteo ICM",
            "source_label": "Meteo ICM",
            "temp_c": self._first(temp_series),
            "wind_ms": self._first(wind_series),
            "cloud_pct": self._first(cloud_series),
            "rh_pct": self._first(rh_series),
            "pressure_hpa": None,
            "status": "ok",
            "updated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "note": "GFS meteorogram point endpoint.",
            "categories": ["forecast"],
        }
        return {"provider": provider, "series": series}

    def _fetch_windy(self) -> dict[str, object]:
        lat_txt = f"{self.lat:.5f}"
        lon_txt = f"{self.lon:.5f}"
        windy_point_url = (
            f"https://node.windy.com/forecast/point/ecmwf/v2.9/{lat_txt}/{lon_txt}"
            "?includeNow=true&source=hp"
        )
        windy_point = self._http_get_json(windy_point_url, timeout_s=20.0)
        if not isinstance(windy_point, dict):
            raise RuntimeError("Windy: unexpected payload shape.")
        windy_data = windy_point.get("data")
        if not isinstance(windy_data, dict):
            raise RuntimeError("Windy: missing point data block.")

        temp_k = self._to_float_list(windy_data.get("temp"))
        temp_series = [float(v) for v in (self._k_to_c(t) for t in temp_k) if v is not None]
        wind_series = self._to_float_list(windy_data.get("wind"))
        rh_series = self._to_float_list(windy_data.get("rh"))
        ts_ms = self._to_float_list(windy_data.get("ts"))
        ts = [int(round(v / 1000.0)) for v in ts_ms]
        now = windy_point.get("now") if isinstance(windy_point.get("now"), dict) else {}

        provider = {
            "label": "Windy (ECMWF)",
            "source_label": "Windy",
            "temp_c": self._k_to_c(_safe_float(now.get("temp"))) or self._first(temp_series),
            "wind_ms": _safe_float(now.get("wind")) or self._first(wind_series),
            "cloud_pct": None,
            "rh_pct": self._first(rh_series),
            "pressure_hpa": None,
            "status": "ok",
            "updated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "note": "Live point forecast via Windy ECMWF endpoint.",
            "categories": ["forecast", "conditions_model"],
        }
        series = {
            "timestamps": ts[:96],
            "temp_c": temp_series[:96],
            "wind_ms": wind_series[:96],
            "cloud_pct": [],
            "rh_pct": rh_series[:96],
            "pressure_hpa": [],
        }
        return {"provider": provider, "series": series}

    def _fetch_meteoblue(self) -> dict[str, object]:
        lat_txt = f"{self.lat:.5f}"
        lon_txt = f"{self.lon:.5f}"
        mblue_point_url = (
            f"https://node.windy.com/forecast/point/mblue/v2.9/{lat_txt}/{lon_txt}"
            "?includeNow=true&source=hp"
        )
        mblue_point = self._http_get_json(mblue_point_url, timeout_s=20.0)
        if not isinstance(mblue_point, dict):
            raise RuntimeError("meteoblue: unexpected point payload.")
        mblue_data = mblue_point.get("data")
        if not isinstance(mblue_data, dict):
            raise RuntimeError("meteoblue: missing point data.")

        temp_k = self._to_float_list(mblue_data.get("temp"))
        temp_series = [float(v) for v in (self._k_to_c(t) for t in temp_k) if v is not None]
        wind_series = self._to_float_list(mblue_data.get("wind"))
        rh_series = self._to_float_list(mblue_data.get("rh"))
        ts_ms = self._to_float_list(mblue_data.get("ts"))
        ts = [int(round(v / 1000.0)) for v in ts_ms]

        met_url = f"https://node.windy.com/forecast/meteogram/mblue/v1.2/{lat_txt}/{lon_txt}?step=3"
        met_payload = self._http_get_json(met_url, timeout_s=20.0)
        met_data = met_payload.get("data") if isinstance(met_payload, dict) else None
        cloud_series: list[float] = []
        if isinstance(met_data, dict):
            keys = [key for key in met_data.keys() if str(key).startswith("cloud-")]
            layers: list[list[float]] = []
            for key in keys:
                values = self._to_float_list(met_data.get(key))
                if values:
                    layers.append(values)
            if layers:
                n = min(len(layer) for layer in layers)
                for idx in range(n):
                    sample = [layer[idx] for layer in layers if idx < len(layer)]
                    v = self._avg(sample)
                    cloud_series.append(0.0 if v is None else max(0.0, min(100.0, float(v))))

        now = mblue_point.get("now") if isinstance(mblue_point.get("now"), dict) else {}
        provider = {
            "label": "meteoblue",
            "source_label": "meteoblue",
            "temp_c": self._k_to_c(_safe_float(now.get("temp"))) or self._first(temp_series),
            "wind_ms": _safe_float(now.get("wind")) or self._first(wind_series),
            "cloud_pct": self._first(cloud_series),
            "rh_pct": self._first(rh_series),
            "pressure_hpa": None,
            "status": "ok",
            "updated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "note": "Live mblue model via Windy endpoints.",
            "categories": ["forecast", "conditions_model"],
        }
        series = {
            "timestamps": ts[:96],
            "temp_c": temp_series[:96],
            "wind_ms": wind_series[:96],
            "cloud_pct": cloud_series[:96],
            "rh_pct": rh_series[:96],
            "pressure_hpa": [],
        }
        return {"provider": provider, "series": series}

    def _fetch_metar(self) -> dict[str, object]:
        station: Optional[dict[str, object]] = None
        for delta in (0.7, 1.2, 2.5, 5.0, 8.0):
            bbox = (
                f"{self.lat - delta:.4f},{self.lon - delta:.4f},"
                f"{self.lat + delta:.4f},{self.lon + delta:.4f}"
            )
            url = f"https://aviationweather.gov/api/data/stationinfo?bbox={bbox}&format=json"
            response = self._http_get_json(url, timeout_s=20.0)
            if not isinstance(response, list) or not response:
                continue
            rows = [row for row in response if isinstance(row, dict)]
            if not rows:
                continue
            rows.sort(
                key=lambda row: self._haversine_km(
                    self.lat,
                    self.lon,
                    _safe_float(row.get("lat")) or 0.0,
                    _safe_float(row.get("lon")) or 0.0,
                )
            )
            station = rows[0]
            break
        if not isinstance(station, dict):
            raise RuntimeError("No nearby METAR station found.")

        station_id = str(station.get("icaoId") or station.get("id") or "").strip().upper()
        if not station_id:
            raise RuntimeError("METAR station id is missing.")
        metar_url = f"https://aviationweather.gov/api/data/metar?ids={quote(station_id)}&format=json&hours=12"
        metar_payload = self._http_get_json(metar_url, timeout_s=20.0)
        if not isinstance(metar_payload, list) or not metar_payload:
            raise RuntimeError(f"No METAR observations for {station_id}.")
        rows = [row for row in metar_payload if isinstance(row, dict)]
        if not rows:
            raise RuntimeError(f"METAR payload for {station_id} is empty.")

        latest = rows[0]
        temp = _safe_float(latest.get("temp"))
        dew = _safe_float(latest.get("dewp"))
        rh = self._estimate_rh_from_temp_dew(temp, dew)
        cloud = self._metar_cover_to_cloud_pct(latest.get("cover"))
        if cloud is None and isinstance(latest.get("clouds"), list):
            buckets = []
            for cloud_row in latest.get("clouds"):
                if not isinstance(cloud_row, dict):
                    continue
                cv = self._metar_cover_to_cloud_pct(cloud_row.get("cover"))
                if cv is not None:
                    buckets.append(float(cv))
            cloud = max(buckets) if buckets else None

        trend_rows = list(reversed(rows[:24]))
        ts: list[int] = []
        temp_series: list[float] = []
        wind_series: list[float] = []
        cloud_series: list[float] = []
        rh_series: list[float] = []
        pressure_series: list[float] = []
        for row in trend_rows:
            obs_ts = _safe_int(row.get("obsTime"))
            if obs_ts is None:
                continue
            ts.append(int(obs_ts))
            tv = _safe_float(row.get("temp"))
            wv = _safe_float(row.get("wspd"))
            pv = _safe_float(row.get("altim"))
            cv = self._metar_cover_to_cloud_pct(row.get("cover"))
            dv = _safe_float(row.get("dewp"))
            rv = self._estimate_rh_from_temp_dew(tv, dv)
            temp_series.append(float(tv) if tv is not None else float("nan"))
            wind_series.append(float(wv) if wv is not None else float("nan"))
            cloud_series.append(float(cv) if cv is not None else float("nan"))
            rh_series.append(float(rv) if rv is not None else float("nan"))
            pressure_series.append(float(pv) if pv is not None else float("nan"))

        updated_raw = latest.get("reportTime") or latest.get("receiptTime")
        updated_ts = self._iso_to_timestamp(updated_raw)
        updated_utc = (
            datetime.fromtimestamp(int(updated_ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            if updated_ts is not None
            else datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        )
        dist_km = self._haversine_km(
            self.lat,
            self.lon,
            _safe_float(station.get("lat")) or self.lat,
            _safe_float(station.get("lon")) or self.lon,
        )
        label = str(station.get("site") or station_id).strip()
        provider = {
            "label": f"METAR ({station_id})",
            "source_label": "METAR",
            "temp_c": temp,
            "wind_ms": _safe_float(latest.get("wspd")),
            "cloud_pct": cloud,
            "rh_pct": rh,
            "pressure_hpa": _safe_float(latest.get("altim")),
            "status": "ok",
            "updated_utc": updated_utc,
            "note": f"Nearest station: {label} ({dist_km:.1f} km).",
            "categories": ["conditions", "conditions_measured"],
        }
        series = {
            "timestamps": ts,
            "temp_c": temp_series,
            "wind_ms": wind_series,
            "cloud_pct": cloud_series,
            "rh_pct": rh_series,
            "pressure_hpa": pressure_series,
        }
        return {"provider": provider, "series": series}

    @staticmethod
    def _parse_custom_payload(payload: object, fallback_label: str) -> tuple[dict[str, object], dict[str, object]]:
        if isinstance(payload, list) and payload and isinstance(payload[0], dict):
            payload = payload[0]
        if not isinstance(payload, dict):
            raise RuntimeError("Custom URL must return a JSON object.")

        observations = payload.get("observations")
        if isinstance(observations, list) and any(isinstance(row, dict) for row in observations):
            return WeatherLiveWorker._parse_weathercom_pws_payload(payload, fallback_label)

        current = payload.get("current") if isinstance(payload.get("current"), dict) else payload
        if not isinstance(current, dict):
            raise RuntimeError("Custom JSON current block is invalid.")

        metric_values = {
            "temp_c": _safe_float(current.get("temp_c")),
            "wind_ms": _safe_float(current.get("wind_ms")),
            "cloud_pct": _safe_float(current.get("cloud_pct")),
            "rh_pct": _safe_float(current.get("rh_pct")),
            "pressure_hpa": _safe_float(current.get("pressure_hpa")),
        }
        if all(value is None for value in metric_values.values()):
            raise RuntimeError(
                "Custom JSON must include at least one of temp_c, wind_ms, cloud_pct, rh_pct or pressure_hpa "
                "in current/top-level object."
            )
        updated_utc = str(current.get("updated_utc") or payload.get("updated_utc") or "").strip()
        if not updated_utc:
            updated_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        label = str(current.get("source_label") or payload.get("source_label") or fallback_label).strip()
        missing_metrics = [key for key, value in metric_values.items() if value is None]
        status = "partial" if missing_metrics else "ok"
        note = "Custom station endpoint."
        if missing_metrics:
            note = f"Custom station endpoint. Missing: {', '.join(missing_metrics)}."
        provider = {
            "label": label or fallback_label,
            "source_label": label or fallback_label,
            "temp_c": metric_values["temp_c"],
            "wind_ms": metric_values["wind_ms"],
            "cloud_pct": metric_values["cloud_pct"],
            "rh_pct": metric_values["rh_pct"],
            "pressure_hpa": metric_values["pressure_hpa"],
            "status": status,
            "updated_utc": updated_utc,
            "note": note,
            "categories": ["conditions", "conditions_measured"],
        }

        series_payload = payload.get("series") if isinstance(payload.get("series"), dict) else {}
        ts = WeatherLiveWorker._to_float_list(series_payload.get("timestamps"))
        series = {
            "timestamps": [int(v) for v in ts],
            "temp_c": WeatherLiveWorker._to_float_list(series_payload.get("temp_c")),
            "wind_ms": WeatherLiveWorker._to_float_list(series_payload.get("wind_ms")),
            "cloud_pct": WeatherLiveWorker._to_float_list(series_payload.get("cloud_pct")),
            "rh_pct": WeatherLiveWorker._to_float_list(series_payload.get("rh_pct")),
            "pressure_hpa": WeatherLiveWorker._to_float_list(series_payload.get("pressure_hpa")),
        }
        return provider, series

    @staticmethod
    def _parse_weathercom_pws_payload(payload: dict[str, object], fallback_label: str) -> tuple[dict[str, object], dict[str, object]]:
        observations = payload.get("observations")
        if not isinstance(observations, list):
            raise RuntimeError("Weather.com PWS payload does not contain observations.")
        rows = [row for row in observations if isinstance(row, dict)]
        if not rows:
            raise RuntimeError("Weather.com PWS payload does not contain usable observations.")

        def _metric(row: dict[str, object], key: str) -> Optional[float]:
            metric_payload = row.get("metric")
            if not isinstance(metric_payload, dict):
                return None
            return _safe_float(metric_payload.get(key))

        def _pressure_hpa(row: dict[str, object]) -> Optional[float]:
            values = [value for value in (_metric(row, "pressureMax"), _metric(row, "pressureMin")) if value is not None]
            if not values:
                return None
            return float(sum(values) / len(values))

        def _best_metric(row: dict[str, object], keys: tuple[str, ...]) -> Optional[float]:
            for key in keys:
                value = _metric(row, key)
                if value is not None and math.isfinite(value):
                    return float(value)
            return None

        def _best_humidity(row: dict[str, object]) -> Optional[float]:
            for key in ("humidityAvg", "humidityHigh", "humidityLow"):
                value = _safe_float(row.get(key))
                if value is not None and math.isfinite(value):
                    return float(value)
            return None

        rows_sorted = sorted(
            rows,
            key=lambda r: _safe_float(r.get("epoch")) if _safe_float(r.get("epoch")) is not None else -1,
        )
        latest = rows_sorted[-1]
        station_id = str(latest.get("stationID") or "").strip()
        label = f"Weather.com PWS {station_id}" if station_id else fallback_label
        updated_utc = str(latest.get("obsTimeUtc") or "").strip()
        if not updated_utc:
            updated_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        temp_latest = _best_metric(latest, ("tempAvg", "tempHigh", "tempLow"))
        wind_latest = _best_metric(latest, ("windspeedAvg", "windspeedHigh", "windspeedLow"))
        rh_latest = _best_humidity(latest)
        pressure_latest = _pressure_hpa(latest)
        missing_metrics = [
            name
            for name, value in (
                ("temp_c", temp_latest),
                ("wind_ms", wind_latest),
                ("rh_pct", rh_latest),
                ("pressure_hpa", pressure_latest),
            )
            if value is None
        ]
        status = "partial" if missing_metrics else "ok"

        provider = {
            "label": label,
            "source_label": label,
            "temp_c": temp_latest,
            "wind_ms": wind_latest,
            "cloud_pct": None,
            "rh_pct": rh_latest,
            "pressure_hpa": pressure_latest,
            "status": status,
            "updated_utc": updated_utc,
            "note": "Weather.com PWS endpoint. cloud_pct is not provided by this feed.",
            "categories": ["conditions", "conditions_measured"],
        }

        timestamps: list[int] = []
        temp_series: list[float] = []
        wind_series: list[float] = []
        rh_series: list[float] = []
        pressure_series: list[float] = []
        for row in rows_sorted:
            epoch = _safe_float(row.get("epoch"))
            if epoch is not None and math.isfinite(epoch):
                timestamps.append(int(epoch))
            temp = _best_metric(row, ("tempAvg", "tempHigh", "tempLow"))
            if temp is not None and math.isfinite(temp):
                temp_series.append(float(temp))
            wind = _best_metric(row, ("windspeedAvg", "windspeedHigh", "windspeedLow"))
            if wind is not None and math.isfinite(wind):
                wind_series.append(float(wind))
            rh = _best_humidity(row)
            if rh is not None and math.isfinite(rh):
                rh_series.append(float(rh))
            pressure = _pressure_hpa(row)
            if pressure is not None and math.isfinite(pressure):
                pressure_series.append(float(pressure))

        series = {
            "timestamps": timestamps,
            "temp_c": temp_series,
            "wind_ms": wind_series,
            "cloud_pct": [],
            "rh_pct": rh_series,
            "pressure_hpa": pressure_series,
        }
        return provider, series

    def _fetch_custom_conditions(self) -> dict[str, object]:
        if not self.custom_conditions_url:
            return {
                "provider": {
                    "label": "Custom URL",
                    "source_label": "Custom URL",
                    "temp_c": None,
                    "wind_ms": None,
                    "cloud_pct": None,
                    "rh_pct": None,
                    "pressure_hpa": None,
                    "status": "disabled",
                    "updated_utc": "-",
                    "note": "Custom URL is not configured for the current observatory.",
                    "categories": ["conditions", "conditions_measured"],
                },
                "series": {
                    "timestamps": [],
                    "temp_c": [],
                    "wind_ms": [],
                    "cloud_pct": [],
                    "rh_pct": [],
                    "pressure_hpa": [],
                },
            }
        payload = self._http_get_json(self.custom_conditions_url, timeout_s=20.0)
        provider, series = self._parse_custom_payload(payload, "Custom URL")
        return {"provider": provider, "series": series}

    def _fetch_annual_cloud_climatology(self) -> dict[str, object]:
        lat_txt = f"{self.lat:.5f}"
        lon_txt = f"{self.lon:.5f}"
        elev_txt = f"{self.elev:.0f}"
        week_url = f"https://www.meteoblue.com/en/weather/week?lat={lat_txt}&lon={lon_txt}&asl={elev_txt}"
        week_html = self._http_get_text(week_url, timeout_s=20.0)
        match_climate = re.search(
            r'href="(/en/weather/historyclimate/climatemodelled/[^"]+)"',
            week_html,
            flags=re.IGNORECASE,
        )
        if not match_climate:
            raise RuntimeError("Unable to resolve meteoblue climate URL.")
        climate_url = urljoin("https://www.meteoblue.com", match_climate.group(1))
        climate_html = self._http_get_text(climate_url, timeout_s=20.0)
        match_cloud_chart = re.search(
            r'data-url="([^"]*climate_model/cloud_coverage[^"]+)"',
            climate_html,
            flags=re.IGNORECASE,
        )
        if not match_cloud_chart:
            raise RuntimeError("meteoblue cloud climatology endpoint not found.")
        cloud_chart_url = html_module.unescape(match_cloud_chart.group(1))
        if cloud_chart_url.startswith("//"):
            cloud_chart_url = f"https:{cloud_chart_url}"
        cloud_chart_payload = self._http_get_json(cloud_chart_url, timeout_s=25.0)
        cloud_series = self._extract_chart_series_by_name(cloud_chart_payload)
        sunny = cloud_series.get("sunny", [])
        partly = cloud_series.get("partly cloudy", [])
        overcast = cloud_series.get("overcast", [])
        n_months = min(len(sunny), len(partly), len(overcast))
        if n_months <= 0:
            raise RuntimeError("Cloud climatology series missing.")
        weighted_num = 0.0
        weighted_den = 0.0
        for idx in range(n_months):
            s = max(0.0, float(sunny[idx]))
            p = max(0.0, float(partly[idx]))
            o = max(0.0, float(overcast[idx]))
            days = s + p + o
            if days <= 0:
                continue
            cloud_pct = max(0.0, min(100.0, ((0.1 * s + 0.5 * p + 0.9 * o) / days) * 100.0))
            weighted_num += cloud_pct * days
            weighted_den += days
        annual_cloud_pct = weighted_num / weighted_den if weighted_den > 0 else None
        return {
            "annual_cloud_pct": annual_cloud_pct,
            "annual_cloud_note": "meteoblue climate (sunny/partly/overcast days weighted estimate).",
        }

    def _fetch_earthenv_cloud_map(self) -> dict[str, object]:
        if Image is None or ImageDraw is None:
            raise RuntimeError("Pillow is not available in the current environment.")
        month_name = self._earthenv_month_name(self.cloud_map_month)
        layer_url = f"https://dev-dot-earthenv-dot-map-of-life.appspot.com/map/cloud/{quote(month_name)}"
        layer_payload = self._http_get_json(layer_url, timeout_s=25.0)
        if not isinstance(layer_payload, dict):
            raise RuntimeError("EarthEnv map endpoint returned unexpected payload.")
        map_block = layer_payload.get("map") if isinstance(layer_payload.get("map"), dict) else {}
        mapid = str(map_block.get("mapid") or "").strip()
        token = str(map_block.get("token") or "").strip()
        if not mapid:
            raise RuntimeError("EarthEnv map id is missing.")

        layer_block = layer_payload.get("layer") if isinstance(layer_payload.get("layer"), dict) else {}
        viz = layer_block.get("viz_params") if isinstance(layer_block.get("viz_params"), dict) else {}
        vmin = _safe_float(viz.get("min"))
        vmax = _safe_float(viz.get("max"))
        palette_raw = str(viz.get("palette") or "").strip()
        palette: list[tuple[int, int, int]] = []
        if palette_raw:
            for item in palette_raw.split(","):
                txt = item.strip().lstrip("#")
                if len(txt) != 6:
                    continue
                try:
                    palette.append((int(txt[0:2], 16), int(txt[2:4], 16), int(txt[4:6], 16)))
                except Exception:
                    continue

        width = 860
        height = 380
        zoom = 4
        center_x, center_y = self._latlon_to_world_px(self.lat, self.lon, zoom)
        left = int(round(center_x - width / 2))
        top = int(round(center_y - height / 2))
        tile_from_x = math.floor(left / 256)
        tile_to_x = math.floor((left + width - 1) / 256)
        tile_from_y = math.floor(top / 256)
        tile_to_y = math.floor((top + height - 1) / 256)
        n_tiles = 2 ** zoom

        canvas = Image.new("RGB", (width, height), (12, 18, 26))
        draw = ImageDraw.Draw(canvas)

        for ty in range(tile_from_y, tile_to_y + 1):
            if ty < 0 or ty >= n_tiles:
                continue
            for tx in range(tile_from_x, tile_to_x + 1):
                tx_wrap = tx % n_tiles
                tile_url = self._earthenv_tile_url(mapid, token, zoom, tx_wrap, ty)
                try:
                    tile_bytes = self._http_get_bytes(tile_url, timeout_s=20.0)
                    tile_img = Image.open(io.BytesIO(tile_bytes)).convert("RGB")
                except Exception:
                    continue
                px = tx * 256 - left
                py = ty * 256 - top
                canvas.paste(tile_img, (int(px), int(py)))

        cx = int(round(width / 2.0))
        cy = int(round(height / 2.0))
        cross_color = (255, 72, 72)
        draw.line((cx - 12, cy, cx + 12, cy), fill=cross_color, width=2)
        draw.line((cx, cy - 12, cx, cy + 12), fill=cross_color, width=2)
        draw.ellipse((cx - 18, cy - 18, cx + 18, cy + 18), outline=(245, 245, 245), width=2)

        approx_pct: Optional[float] = None
        precise_caption = ""
        try:
            approx_pct = self._sample_earthenv_point_precise(month_name)
            if approx_pct is not None:
                precise_caption = " Sampled from EarthEnv source data."
        except Exception:
            approx_pct = None
        if approx_pct is None and palette and vmin is not None and vmax is not None:
            center_rgb = canvas.getpixel((cx, cy))
            approx_value = self._estimate_value_from_palette(center_rgb, palette, float(vmin), float(vmax))
            if approx_value is not None:
                approx_pct = float(approx_value) / 100.0 if float(vmax) >= 1000.0 else float(approx_value)
                approx_pct = max(0.0, min(100.0, approx_pct))
                precise_caption = " Estimated from rendered map palette."

        out = io.BytesIO()
        canvas.save(out, format="PNG")
        caption = f"EarthEnv monthly cloud map: {month_name}."
        if approx_pct is not None:
            caption += f" Approx cloud at marker: {approx_pct:.1f}%."
            caption += precise_caption
        caption += " Marker = observatory coordinates."
        return {
            "source": "earthenv",
            "month": int(self.cloud_map_month),
            "month_name": month_name,
            "image_bytes": out.getvalue(),
            "caption": caption,
            "approx_cloud_pct": approx_pct,
            "url": "https://www.earthenv.org/cloud",
        }

    @staticmethod
    def _satellite_area_for_coords(lat: float, lon: float) -> str:
        la = float(lat)
        lo = float(lon)
        if 20.0 <= la <= 75.0 and -25.0 <= lo <= 45.0:
            return "europe"
        if -40.0 <= la <= 45.0 and -30.0 <= lo <= 60.0:
            return "africa"
        if 25.0 <= la <= 50.0 and -45.0 <= lo <= 40.0:
            return "atlantic_ocean"
        if 25.0 <= la <= 50.0 and -10.0 <= lo <= 45.0:
            return "mediterranean"
        return "global"

    @staticmethod
    def _bounded_bbox(lat: float, lon: float, half_span_deg: float) -> tuple[float, float, float, float]:
        span = max(1.5, min(30.0, float(half_span_deg)))
        min_lat = max(-85.0, float(lat) - span)
        max_lat = min(85.0, float(lat) + span)
        min_lon = float(lon) - span
        max_lon = float(lon) + span
        if min_lon < -180.0:
            shift = -180.0 - min_lon
            min_lon += shift
            max_lon += shift
        if max_lon > 180.0:
            shift = max_lon - 180.0
            min_lon -= shift
            max_lon -= shift
        min_lon = max(-180.0, min_lon)
        max_lon = min(180.0, max_lon)
        return min_lon, min_lat, max_lon, max_lat

    @staticmethod
    def _draw_center_marker(image_bytes: bytes) -> bytes:
        if not image_bytes:
            return b""
        with Image.open(io.BytesIO(image_bytes)).convert("RGB") as img:
            w, h = img.size
            cx = int(round(w / 2.0))
            cy = int(round(h / 2.0))
            draw = ImageDraw.Draw(img)
            draw.line((cx - 16, cy, cx + 16, cy), fill=(255, 72, 72), width=3)
            draw.line((cx, cy - 16, cx, cy + 16), fill=(255, 72, 72), width=3)
            draw.ellipse((cx - 24, cy - 24, cx + 24, cy + 24), outline=(245, 245, 245), width=2)
            out = io.BytesIO()
            img.save(out, format="JPEG", quality=92)
            return out.getvalue()

    def _fetch_satellite(self) -> dict[str, object]:
        # Primary source: NASA GIBS WMS true-color layers, centered on observatory coordinates.
        # WMS 1.1.1 in EPSG:4326 expects bbox as minLon,minLat,maxLon,maxLat.
        min_lon, min_lat, max_lon, max_lat = self._bounded_bbox(self.lat, self.lon, half_span_deg=10.0)
        date_candidates = [datetime.now(timezone.utc).date() - timedelta(days=delta) for delta in (0, 1, 2)]
        layer_candidates = (
            "MODIS_Terra_CorrectedReflectance_TrueColor",
            "MODIS_Aqua_CorrectedReflectance_TrueColor",
            "VIIRS_SNPP_CorrectedReflectance_TrueColor",
        )
        last_error = "unknown"
        for layer in layer_candidates:
            for day in date_candidates:
                sat_url = (
                    "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
                    "?SERVICE=WMS&REQUEST=GetMap&VERSION=1.1.1"
                    f"&LAYERS={quote(layer)}&STYLES=&FORMAT=image/jpeg&TRANSPARENT=FALSE"
                    "&SRS=EPSG:4326&WIDTH=960&HEIGHT=540"
                    f"&BBOX={min_lon:.5f},{min_lat:.5f},{max_lon:.5f},{max_lat:.5f}"
                    f"&TIME={day.isoformat()}"
                )
                try:
                    raw = self._http_get_bytes(sat_url, timeout_s=30.0)
                    sat_bytes = self._draw_center_marker(raw)
                    if not sat_bytes:
                        raise RuntimeError("Empty satellite payload.")
                    return {
                        "url": sat_url,
                        "image_bytes": sat_bytes,
                        "caption": (
                            f"NASA GIBS {layer}, date {day.isoformat()} (UTC), centered on observatory."
                            f" BBOX {min_lon:.2f},{min_lat:.2f},{max_lon:.2f},{max_lat:.2f}."
                            " Marker = current observatory."
                        ),
                    }
                except Exception as exc:  # noqa: BLE001
                    last_error = str(exc)
                    continue
        raise RuntimeError(f"GIBS WMS request failed for all layers/dates: {last_error}")

    def _fetch_satellite_met_no(self) -> dict[str, object]:
        area = self._satellite_area_for_coords(self.lat, self.lon)
        sat_url = f"https://api.met.no/weatherapi/geosatellite/1.4/?area={quote(area)}&type=infrared"
        sat_bytes = self._http_get_bytes(sat_url, timeout_s=30.0)
        if not sat_bytes:
            raise RuntimeError("MET geosatellite endpoint returned empty payload.")
        return {
            "url": sat_url,
            "image_bytes": sat_bytes,
            "caption": (
                f"MET Norway geosatellite (infrared), area={area}, latest frame."
                " (Regional frame; not a centered site cutout.)"
            ),
        }

    def _fetch_satellite_wetterzentrale(self) -> dict[str, object]:
        sat_page_url = "https://www.wetterzentrale.de/en/reanalysis.php?map=1&model=sat&var=44"
        sat_page = self._http_get_text(sat_page_url, timeout_s=15.0)
        image_paths = re.findall(
            r"/maps/archive/\d{4}/sat/[A-Za-z0-9._-]+\.jpg",
            sat_page,
            flags=re.IGNORECASE,
        )
        if not image_paths:
            raise RuntimeError("No satellite image URL found on source page.")
        sat_url = urljoin("https://www.wetterzentrale.de", image_paths[-1])
        sat_bytes = self._http_get_bytes(sat_url, timeout_s=25.0)
        return {
            "url": sat_url,
            "image_bytes": sat_bytes,
            "caption": "wetterzentrale latest satellite frame.",
        }

    @staticmethod
    def _is_measured_conditions_provider(key: str, row: object) -> bool:
        if not isinstance(row, dict):
            return False
        key_txt = str(key or "").strip().lower()
        if key_txt in {"metar", "custom"}:
            return True
        categories = row.get("categories")
        return isinstance(categories, list) and "conditions_measured" in categories

    @classmethod
    def _conditions_source_keys(cls, provider_rows: dict[str, dict[str, object]]) -> list[str]:
        out: list[str] = []
        for key, row in provider_rows.items():
            if cls._is_measured_conditions_provider(str(key), row):
                out.append(str(key))
        return out

    def _emit_partial(self, payload: dict[str, object]) -> None:
        self.partial.emit(copy.deepcopy(payload))

    def run(self):
        payload: dict[str, object] = {
            "providers": {},
            "series": {},
            "averages": {},
            "annual_cloud_pct": None,
            "annual_cloud_note": "-",
            "cloud_map": (
                {
                    "source": self.cloud_map_source,
                    "month": int(self.cloud_map_month),
                    "month_name": self._earthenv_month_name(self.cloud_map_month),
                    "image_bytes": b"",
                    "caption": "Map is loading...",
                    "approx_cloud_pct": None,
                    "url": "https://www.earthenv.org/cloud",
                }
                if self.include_cloud_map
                else None
            ),
            "satellite": (
                {"url": "", "image_bytes": b"", "caption": "-"} if self.include_satellite else None
            ),
            "sections": {
                "forecast": {"status": "loading", "message": "Loading forecast providers..."},
                "conditions": {"status": "loading", "message": "Loading live conditions..."},
                "climatology": {
                    "status": "loading" if self.include_cloud_map else "idle",
                    "message": "Loading cloud climatology..." if self.include_cloud_map else "Cloud map not refreshed.",
                },
                "satellite": {
                    "status": "loading" if self.include_satellite else "idle",
                    "message": "Loading satellite preview..." if self.include_satellite else "Satellite not refreshed.",
                },
            },
            "errors": [],
        }

        providers: dict[str, dict[str, object]] = payload["providers"]  # type: ignore[assignment]
        series_map: dict[str, dict[str, object]] = payload["series"]  # type: ignore[assignment]
        errors: list[str] = payload["errors"]  # type: ignore[assignment]

        total_steps = 7 + (1 if self.include_cloud_map else 0) + (1 if self.include_satellite else 0)
        step = 0

        def advance(status: str) -> None:
            nonlocal step
            step += 1
            self.progress.emit(status, step, total_steps)

        lat_txt = f"{self.lat:.5f}"
        lon_txt = f"{self.lon:.5f}"
        custom_hash = hashlib.md5(self.custom_conditions_url.encode("utf-8")).hexdigest()[:10] if self.custom_conditions_url else "-"

        def update_provider(
            key: str,
            result: dict[str, object],
            *,
            cached: bool,
            on_error_label: str,
        ) -> None:
            provider = result.get("provider") if isinstance(result.get("provider"), dict) else None
            series = result.get("series") if isinstance(result.get("series"), dict) else None
            if not isinstance(provider, dict):
                raise RuntimeError(f"{on_error_label}: provider payload is invalid.")
            note = str(provider.get("note") or "").strip()
            if cached and note:
                provider["note"] = f"{note} (cached)"
            elif cached:
                provider["note"] = "cached"
            provider.setdefault("updated_utc", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
            providers[key] = provider
            if isinstance(series, dict):
                series_map[key] = series
            self._emit_partial(payload)

        try:
            advance("Loading Open-Meteo...")
            result, cached = self._run_cached(
                cache_key=f"weather:open_meteo:{lat_txt}:{lon_txt}",
                ttl_s=self.TTL_CONDITIONS_S,
                fetcher=self._fetch_open_meteo,
            )
            update_provider("open_meteo", result, cached=cached, on_error_label="Open-Meteo")
        except Exception as exc:  # noqa: BLE001
            providers["open_meteo"] = {
                "label": "Open-Meteo",
                "source_label": "Open-Meteo",
                "temp_c": None,
                "wind_ms": None,
                "cloud_pct": None,
                "rh_pct": None,
                "pressure_hpa": None,
                "status": "error",
                "updated_utc": "-",
                "note": str(exc),
                "categories": ["forecast", "conditions_model"],
            }
            errors.append(f"Open-Meteo: {exc}")
            self._emit_partial(payload)

        try:
            advance("Loading Meteo ICM...")
            result, cached = self._run_cached(
                cache_key=f"weather:meteo_icm:{lat_txt}:{lon_txt}",
                ttl_s=self.TTL_FORECAST_S,
                fetcher=self._fetch_meteo_icm,
            )
            update_provider("meteo_icm", result, cached=cached, on_error_label="Meteo ICM")
        except Exception as exc:  # noqa: BLE001
            providers["meteo_icm"] = {
                "label": "Meteo ICM",
                "source_label": "Meteo ICM",
                "temp_c": None,
                "wind_ms": None,
                "cloud_pct": None,
                "rh_pct": None,
                "pressure_hpa": None,
                "status": "error",
                "updated_utc": "-",
                "note": str(exc),
                "categories": ["forecast"],
            }
            errors.append(f"Meteo ICM: {exc}")
            self._emit_partial(payload)

        try:
            advance("Loading Windy...")
            result, cached = self._run_cached(
                cache_key=f"weather:windy:{lat_txt}:{lon_txt}",
                ttl_s=self.TTL_FORECAST_S,
                fetcher=self._fetch_windy,
            )
            update_provider("windy", result, cached=cached, on_error_label="Windy")
        except Exception as exc:  # noqa: BLE001
            providers["windy"] = {
                "label": "Windy",
                "source_label": "Windy",
                "temp_c": None,
                "wind_ms": None,
                "cloud_pct": None,
                "rh_pct": None,
                "pressure_hpa": None,
                "status": "error",
                "updated_utc": "-",
                "note": str(exc),
                "categories": ["forecast", "conditions_model"],
            }
            errors.append(f"Windy: {exc}")
            self._emit_partial(payload)

        try:
            advance("Loading meteoblue...")
            result, cached = self._run_cached(
                cache_key=f"weather:meteoblue:{lat_txt}:{lon_txt}",
                ttl_s=self.TTL_FORECAST_S,
                fetcher=self._fetch_meteoblue,
            )
            update_provider("meteoblue", result, cached=cached, on_error_label="meteoblue")
        except Exception as exc:  # noqa: BLE001
            providers["meteoblue"] = {
                "label": "meteoblue",
                "source_label": "meteoblue",
                "temp_c": None,
                "wind_ms": None,
                "cloud_pct": None,
                "rh_pct": None,
                "pressure_hpa": None,
                "status": "error",
                "updated_utc": "-",
                "note": str(exc),
                "categories": ["forecast", "conditions_model"],
            }
            errors.append(f"meteoblue: {exc}")
            self._emit_partial(payload)

        try:
            advance("Loading nearest METAR...")
            result, cached = self._run_cached(
                cache_key=f"weather:metar:{lat_txt}:{lon_txt}",
                ttl_s=self.TTL_CONDITIONS_S,
                fetcher=self._fetch_metar,
            )
            update_provider("metar", result, cached=cached, on_error_label="METAR")
        except Exception as exc:  # noqa: BLE001
            providers["metar"] = {
                "label": "METAR",
                "source_label": "METAR",
                "temp_c": None,
                "wind_ms": None,
                "cloud_pct": None,
                "rh_pct": None,
                "pressure_hpa": None,
                "status": "error",
                "updated_utc": "-",
                "note": str(exc),
                "categories": ["conditions", "conditions_measured"],
            }
            errors.append(f"METAR: {exc}")
            self._emit_partial(payload)

        try:
            advance("Loading custom conditions...")
            result, cached = self._run_cached(
                cache_key=f"weather:custom:{custom_hash}",
                ttl_s=self.TTL_CONDITIONS_S,
                fetcher=self._fetch_custom_conditions,
            )
            update_provider("custom", result, cached=cached, on_error_label="Custom URL")
        except Exception as exc:  # noqa: BLE001
            providers["custom"] = {
                "label": "Custom URL",
                "source_label": "Custom URL",
                "temp_c": None,
                "wind_ms": None,
                "cloud_pct": None,
                "rh_pct": None,
                "pressure_hpa": None,
                "status": "error",
                "updated_utc": "-",
                "note": str(exc),
                "categories": ["conditions", "conditions_measured"],
            }
            errors.append(f"Custom URL: {exc}")
            self._emit_partial(payload)

        try:
            advance("Loading cloud climatology...")
            cloud_payload, cached = self._run_cached(
                cache_key=f"weather:climatology:{lat_txt}:{lon_txt}:{self.elev:.0f}",
                ttl_s=self.TTL_CLIMATOLOGY_S,
                fetcher=self._fetch_annual_cloud_climatology,
            )
            payload["annual_cloud_pct"] = _safe_float(cloud_payload.get("annual_cloud_pct"))
            note = str(cloud_payload.get("annual_cloud_note") or "-").strip()
            payload["annual_cloud_note"] = f"{note} (cached)" if cached and note else note
            self._emit_partial(payload)
        except Exception as exc:  # noqa: BLE001
            payload["annual_cloud_pct"] = None
            payload["annual_cloud_note"] = f"Unavailable: {exc}"
            errors.append(f"Annual cloud climatology: {exc}")
            self._emit_partial(payload)

        if self.include_cloud_map:
            try:
                advance("Loading monthly cloud map...")
                if self.cloud_map_source == "earthenv":
                    map_payload, cached = self._run_cached(
                        cache_key=f"weather:earthenv:{self.cloud_map_month}:{lat_txt}:{lon_txt}",
                        ttl_s=self.TTL_CLIMATOLOGY_S,
                        fetcher=self._fetch_earthenv_cloud_map,
                    )
                    if cached:
                        caption = str(map_payload.get("caption") or "").strip()
                        if caption:
                            map_payload["caption"] = f"{caption} (cached)"
                    payload["cloud_map"] = map_payload
                else:
                    payload["cloud_map"] = {
                        "source": self.cloud_map_source,
                        "month": int(self.cloud_map_month),
                        "month_name": self._earthenv_month_name(self.cloud_map_month),
                        "image_bytes": b"",
                        "caption": "Cloud map source is not available in this build.",
                        "approx_cloud_pct": None,
                        "url": "https://www.earthenv.org/cloud",
                    }
                self._emit_partial(payload)
            except Exception as exc:  # noqa: BLE001
                payload["cloud_map"] = {
                    "source": "earthenv",
                    "month": int(self.cloud_map_month),
                    "month_name": self._earthenv_month_name(self.cloud_map_month),
                    "image_bytes": b"",
                    "caption": f"EarthEnv map unavailable: {exc}",
                    "approx_cloud_pct": None,
                    "url": "https://www.earthenv.org/cloud",
                }
                errors.append(f"Cloud map: {exc}")
                self._emit_partial(payload)

        if self.include_satellite:
            try:
                advance("Loading satellite preview...")
                sat_payload, cached = self._run_cached(
                    cache_key=f"weather:satellite:gibs:{lat_txt}:{lon_txt}",
                    ttl_s=self.TTL_CONDITIONS_S,
                    fetcher=self._fetch_satellite,
                )
                if cached:
                    caption = str(sat_payload.get("caption") or "").strip()
                    sat_payload["caption"] = f"{caption} (cached)" if caption else "cached"
                payload["satellite"] = sat_payload
                self._emit_partial(payload)
            except Exception as exc:  # noqa: BLE001
                try:
                    sat_payload, cached = self._run_cached(
                        cache_key=f"weather:satellite:met_no:{lat_txt}:{lon_txt}",
                        ttl_s=self.TTL_CONDITIONS_S,
                        fetcher=self._fetch_satellite_met_no,
                    )
                    if cached:
                        caption = str(sat_payload.get("caption") or "").strip()
                        sat_payload["caption"] = f"{caption} (cached)" if caption else "cached"
                    sat_payload["caption"] = (
                        f"{str(sat_payload.get('caption') or '').strip()} "
                        f"(fallback after primary satellite error: {exc})"
                    ).strip()
                    payload["satellite"] = sat_payload
                    self._emit_partial(payload)
                except Exception as exc_fallback:
                    try:
                        sat_payload, cached = self._run_cached(
                            cache_key="weather:satellite:wetterzentrale",
                            ttl_s=self.TTL_CONDITIONS_S,
                            fetcher=self._fetch_satellite_wetterzentrale,
                        )
                        if cached:
                            caption = str(sat_payload.get("caption") or "").strip()
                            sat_payload["caption"] = f"{caption} (cached)" if caption else "cached"
                        sat_payload["caption"] = (
                            f"{str(sat_payload.get('caption') or '').strip()} "
                            f"(fallback after errors: primary={exc}; met.no={exc_fallback})"
                        ).strip()
                        payload["satellite"] = sat_payload
                        self._emit_partial(payload)
                    except Exception as exc_last:
                        payload["satellite"] = {
                            "url": "",
                            "image_bytes": b"",
                            "caption": (
                                "Satellite preview unavailable: "
                                f"primary={exc}; met.no={exc_fallback}; wetterzentrale={exc_last}"
                            ),
                        }
                        errors.append(
                            "Satellite preview failed: "
                            f"primary={exc}; met.no={exc_fallback}; wetterzentrale={exc_last}"
                        )
                        self._emit_partial(payload)

        advance("Computing source aggregates...")
        condition_keys = self._conditions_source_keys(providers)
        avg_payload: dict[str, dict[str, object]] = {}
        for metric_key in ("temp_c", "wind_ms", "cloud_pct", "rh_pct", "pressure_hpa"):
            values: list[float] = []
            labels: list[str] = []
            for key in condition_keys:
                row = providers.get(key)
                if not isinstance(row, dict):
                    continue
                status = str(row.get("status") or "").strip().lower()
                if status not in {"ok", "partial"}:
                    continue
                value = _safe_float(row.get(metric_key))
                if value is None or not math.isfinite(value):
                    continue
                values.append(float(value))
                labels.append(str(row.get("label", key)))
            avg_payload[metric_key] = {
                "value": self._avg(values),
                "count": len(values),
                "sources": labels,
            }
        payload["averages"] = avg_payload

        forecast_ok = 0
        forecast_err = 0
        conditions_ok = 0
        conditions_err = 0
        for key, row in providers.items():
            if not isinstance(row, dict):
                continue
            categories = row.get("categories")
            if not isinstance(categories, list):
                continue
            status = str(row.get("status") or "").strip().lower()
            if "forecast" in categories:
                if status in {"ok", "partial"}:
                    forecast_ok += 1
                elif status == "error":
                    forecast_err += 1
            if self._is_measured_conditions_provider(str(key), row):
                if status in {"ok", "partial"}:
                    conditions_ok += 1
                elif status == "error":
                    conditions_err += 1

        payload["sections"] = {
            "forecast": {
                "status": "ok" if forecast_ok > 0 and forecast_err == 0 else ("partial" if forecast_ok > 0 else "error"),
                "message": f"Forecast providers ok: {forecast_ok}, errors: {forecast_err}.",
            },
            "conditions": {
                "status": "ok"
                if conditions_ok > 0 and conditions_err == 0
                else ("partial" if conditions_ok > 0 else "error"),
                "message": f"Measured conditions providers ok: {conditions_ok}, errors: {conditions_err}.",
            },
            "climatology": {
                "status": (
                    "ok" if _safe_float(payload.get("annual_cloud_pct")) is not None else "partial"
                )
                if self.include_cloud_map
                else "idle",
                "message": str(payload.get("annual_cloud_note") or "-")
                if self.include_cloud_map
                else "Cloud map not refreshed in this cycle.",
            },
            "satellite": {
                "status": (
                    "ok"
                    if isinstance(payload.get("satellite"), dict)
                    and payload.get("satellite", {}).get("image_bytes")
                    else "partial"
                )
                if self.include_satellite
                else "idle",
                "message": (
                    str(payload.get("satellite", {}).get("caption"))
                    if isinstance(payload.get("satellite"), dict)
                    else "-"
                )
                if self.include_satellite
                else "Satellite not refreshed in this cycle.",
            },
        }

        self._emit_partial(payload)
        self.completed.emit(payload)

__all__ = ["WeatherLiveWorker"]
