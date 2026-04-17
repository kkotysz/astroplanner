from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from PySide6.QtCore import Qt

from astroplanner.models import DEFAULT_LIMITING_MAGNITUDE, Site

if TYPE_CHECKING:
    from astro_planner import MainWindow


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
        if isinstance(value, str) and not value.strip():
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


class ObservatoryCoordinator:
    """Own observatory persistence, lookup, and combo refresh glue."""

    def __init__(self, planner: "MainWindow") -> None:
        self._planner = planner

    def observatories_config_path(self) -> Path:
        return Path(__file__).resolve().parents[1] / "config" / "default_observatories.json"

    def update_obs_combo_widths(self) -> None:
        planner = self._planner
        if not hasattr(planner, "obs_combo"):
            return
        names = list(planner.observatories.keys()) if hasattr(planner, "observatories") else []
        fm = planner.obs_combo.fontMetrics()
        longest_px = max((fm.horizontalAdvance(name) for name in names), default=220)
        combo_w = int(min(max(250, longest_px + 52), 460))
        popup_w = int(min(max(340, longest_px + 88), 700))
        planner.obs_combo.setMinimumWidth(combo_w)
        planner.obs_combo.setMaximumWidth(460)
        view = planner.obs_combo.view()
        if view is not None:
            view.setMinimumWidth(popup_w)
            view.setTextElideMode(Qt.ElideNone)

    def parse_custom_observatories_payload(self, payload: object) -> tuple[dict[str, Site], dict[str, str]]:
        items: list[dict[str, object]] = []
        if isinstance(payload, dict) and isinstance(payload.get("observatories"), list):
            payload = payload["observatories"]
        if isinstance(payload, list):
            items = [item for item in payload if isinstance(item, dict)]
        elif isinstance(payload, dict):
            for key, value in payload.items():
                if isinstance(value, dict):
                    entry = dict(value)
                    entry.setdefault("name", key)
                    items.append(entry)

        loaded: dict[str, Site] = {}
        preset_keys: dict[str, str] = {}
        for item in items:
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            try:
                lim_mag = _safe_float(item.get("limiting_magnitude", item.get("limitingMagnitude", DEFAULT_LIMITING_MAGNITUDE)))
                diameter_mm = _safe_float(
                    item.get("telescope_diameter_mm", item.get("telescopeDiameterMm", item.get("diameter_mm", 0.0)))
                )
                if diameter_mm is None:
                    diameter_m = _safe_float(item.get("telescope_diameter_m", item.get("telescopeDiameterM", 0.0)))
                    if diameter_m is not None:
                        diameter_mm = float(diameter_m) * 1000.0
                focal_mm = _safe_float(item.get("focal_length_mm", item.get("focalLengthMm", 0.0)))
                pixel_um = _safe_float(item.get("pixel_size_um", item.get("pixelSizeUm", item.get("pixel_um", 0.0))))
                detector_w = _safe_int(
                    item.get("detector_width_px", item.get("detectorWidthPx", item.get("detector_width", 0)))
                )
                detector_h = _safe_int(
                    item.get("detector_height_px", item.get("detectorHeightPx", item.get("detector_height", 0)))
                )
                loaded[name] = Site(
                    name=name,
                    latitude=float(item.get("latitude", 0.0)),
                    longitude=float(item.get("longitude", 0.0)),
                    elevation=float(item.get("elevation", 0.0)),
                    limiting_magnitude=float(lim_mag if lim_mag is not None else DEFAULT_LIMITING_MAGNITUDE),
                    telescope_diameter_mm=float(diameter_mm if diameter_mm is not None else 0.0),
                    focal_length_mm=float(focal_mm if focal_mm is not None else 0.0),
                    pixel_size_um=float(pixel_um if pixel_um is not None else 0.0),
                    detector_width_px=int(detector_w if detector_w is not None else 0),
                    detector_height_px=int(detector_h if detector_h is not None else 0),
                    custom_conditions_url=str(
                        item.get("custom_conditions_url", item.get("customConditionsUrl", item.get("weather_custom_conditions_url", "")))
                        or ""
                    ).strip(),
                )
                preset_key_raw = item.get("preset_key", item.get("presetKey", item.get("bhtom_preset_key", "custom")))
                preset_key = str(preset_key_raw or "custom").strip() or "custom"
                preset_keys[name] = preset_key
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping invalid saved observatory %r: %s", name, exc)
        return loaded, preset_keys

    def load_custom_observatories(self) -> tuple[dict[str, Site], dict[str, str]]:
        planner = self._planner
        storage = getattr(planner, "app_storage", None)
        if storage is not None:
            try:
                stored_items = storage.observatories.list_all()
                if stored_items:
                    return self.parse_custom_observatories_payload({"observatories": stored_items})
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to load observatories from storage: %s", exc)

        cfg_path = self.observatories_config_path()
        if cfg_path.exists():
            try:
                payload = json.loads(cfg_path.read_text(encoding="utf-8"))
                return self.parse_custom_observatories_payload(payload)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to parse %s: %s", cfg_path, exc)
        return {}, {}

    def save_custom_observatories(
        self,
        custom_sites: Optional[dict[str, Site]] = None,
        *,
        preset_keys: Optional[dict[str, str]] = None,
    ) -> None:
        planner = self._planner
        if custom_sites is None:
            source_sites = planner.observatories if hasattr(planner, "observatories") else {}
        else:
            source_sites = custom_sites
        if preset_keys is None:
            source_preset_keys = getattr(planner, "_observatory_preset_keys", {})
        else:
            source_preset_keys = preset_keys
        observatory_items: list[dict[str, object]] = []
        for name in sorted(source_sites.keys(), key=str.lower):
            site = source_sites[name]
            preset_key = str(source_preset_keys.get(name, "custom") or "custom")
            observatory_items.append(
                {
                    "name": name,
                    "latitude": float(site.latitude),
                    "longitude": float(site.longitude),
                    "elevation": float(site.elevation),
                    "limiting_magnitude": float(site.limiting_magnitude),
                    "telescope_diameter_mm": float(site.telescope_diameter_mm),
                    "telescope_diameter_m": float(site.telescope_diameter_mm) / 1000.0,
                    "focal_length_mm": float(site.focal_length_mm),
                    "pixel_size_um": float(site.pixel_size_um),
                    "detector_width_px": int(site.detector_width_px),
                    "detector_height_px": int(site.detector_height_px),
                    "custom_conditions_url": str(site.custom_conditions_url or "").strip(),
                    "preset_key": preset_key,
                }
            )
        storage = getattr(planner, "app_storage", None)
        if storage is not None:
            try:
                storage.observatories.replace_all(observatory_items)
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to save observatories to storage: %s", exc)
                return

    def refresh_observatory_combo(
        self,
        selected_name: Optional[str] = None,
        *,
        emit_selection_change: bool = False,
    ) -> None:
        planner = self._planner
        current = selected_name or planner.obs_combo.currentText()
        planner.obs_combo.blockSignals(True)
        planner.obs_combo.clear()
        planner.obs_combo.addItems(planner.observatories.keys())
        if current in planner.observatories:
            planner.obs_combo.setCurrentText(current)
        elif planner.obs_combo.count() > 0:
            planner.obs_combo.setCurrentIndex(0)
        planner.obs_combo.blockSignals(False)
        self.update_obs_combo_widths()
        if emit_selection_change:
            selected = planner.obs_combo.currentText().strip()
            if selected:
                planner._on_obs_change(selected)

    def lookup_observatory_coordinates(self, query: str) -> tuple[float, float, Optional[float], str]:
        q = query.strip()
        if not q:
            raise ValueError("Name cannot be empty.")

        params = urlencode({"format": "jsonv2", "limit": 1, "q": q})
        url = f"https://nominatim.openstreetmap.org/search?{params}"
        req = Request(
            url,
            headers={
                "User-Agent": "AstroPlanner/1.0 (desktop app)",
                "Accept": "application/json",
            },
        )
        try:
            with urlopen(req, timeout=10) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            raise RuntimeError(f"Location lookup failed: {exc}") from exc

        if not isinstance(payload, list) or not payload:
            raise ValueError("No location found for this name.")

        hit = payload[0]
        try:
            lat = float(hit.get("lat"))
            lon = float(hit.get("lon"))
        except Exception as exc:
            raise ValueError("Location found, but coordinates are invalid.") from exc
        display_name = str(hit.get("display_name", q))

        elevation: Optional[float] = None
        elev_params = urlencode({"latitude": f"{lat:.6f}", "longitude": f"{lon:.6f}"})
        elev_url = f"https://api.open-meteo.com/v1/elevation?{elev_params}"
        elev_req = Request(
            elev_url,
            headers={
                "User-Agent": "AstroPlanner/1.0 (desktop app)",
                "Accept": "application/json",
            },
        )
        try:
            with urlopen(elev_req, timeout=10) as resp:
                elev_payload = json.loads(resp.read().decode("utf-8"))
            elev_value = elev_payload.get("elevation")
            if isinstance(elev_value, list) and elev_value:
                elevation = float(elev_value[0])
            elif elev_value is not None:
                elevation = float(elev_value)
        except Exception:
            elevation = None

        return lat, lon, elevation, display_name


__all__ = ["ObservatoryCoordinator"]
