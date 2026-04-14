from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord
from PySide6.QtCore import QDate
from pydantic import BaseModel, ConfigDict, Field
from timezonefinder import TimezoneFinder


DEFAULT_LIMITING_MAGNITUDE = 19.0
_TZ_FINDER = TimezoneFinder()


def _safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class CalcRunStats:
    duration_s: float
    visible_targets: int
    total_targets: int


class Target(BaseModel):
    """A celestial target."""

    name: str = Field(..., description="Display name")
    ra: float = Field(..., description="Right Ascension in degrees")
    dec: float = Field(..., description="Declination in degrees")
    source_catalog: str = Field("", description="Resolver/backend used to create this target")
    source_object_id: str = Field("", description="Identifier returned by the source catalog")
    object_type: str = Field("", description="Target type label")
    magnitude: Optional[float] = Field(None, description="Apparent magnitude")
    size_arcmin: Optional[float] = Field(None, description="Apparent size in arcminutes")
    priority: int = Field(3, ge=1, le=5, description="User priority, 1 low to 5 high")
    observed: bool = Field(False, description="Whether target has already been observed")
    notes: str = Field("", description="Optional free-form note")
    plot_color: str = Field("", description="Optional custom plot color (hex)")

    @classmethod
    def from_skycoord(cls, name: str, coord: SkyCoord) -> "Target":  # noqa: D401
        return cls(name=name, ra=coord.ra.deg, dec=coord.dec.deg)  # type: ignore[arg-type]

    @property
    def skycoord(self) -> SkyCoord:  # noqa: D401
        return SkyCoord(ra=self.ra * u.deg, dec=self.dec * u.deg)


class Site(BaseModel):
    """Observation site."""

    name: str = "Custom"
    latitude: float = Field(..., description="Latitude in degrees")
    longitude: float = Field(..., description="Longitude in degrees")
    elevation: float = Field(0.0, description="Elevation in meters")
    limiting_magnitude: float = Field(
        DEFAULT_LIMITING_MAGNITUDE,
        description="Magnitude threshold used by observatory/telescope profile.",
    )
    telescope_diameter_mm: float = Field(0.0, description="Telescope aperture/diameter in mm.")
    focal_length_mm: float = Field(0.0, description="Telescope focal length in mm.")
    pixel_size_um: float = Field(0.0, description="Detector pixel size in micrometers.")
    detector_width_px: int = Field(0, description="Detector width in pixels.")
    detector_height_px: int = Field(0, description="Detector height in pixels.")
    custom_conditions_url: str = Field(
        "",
        description="Optional per-observatory JSON endpoint used by Weather -> Custom URL.",
    )

    def to_earthlocation(self) -> EarthLocation:  # noqa: D401
        return EarthLocation(lat=self.latitude * u.deg, lon=self.longitude * u.deg, height=self.elevation * u.m)

    @property
    def timezone_name(self) -> str:  # noqa: D401
        return _TZ_FINDER.timezone_at(lng=self.longitude, lat=self.latitude) or "UTC"

    @property
    def pixel_scale_arcsec_per_px(self) -> Optional[float]:  # noqa: D401
        focal = _safe_float(self.focal_length_mm)
        pixel = _safe_float(self.pixel_size_um)
        if focal is None or pixel is None or focal <= 0.0 or pixel <= 0.0:
            return None
        return float(206.265 * pixel / focal)

    @property
    def fov_arcmin(self) -> Optional[tuple[float, float]]:  # noqa: D401
        scale = self.pixel_scale_arcsec_per_px
        if scale is None:
            return None
        width = int(self.detector_width_px or 0)
        height = int(self.detector_height_px or 0)
        if width <= 0 or height <= 0:
            return None
        return (float(width) * scale / 60.0, float(height) * scale / 60.0)


class SessionSettings(BaseModel):
    """Per-night settings handed to the worker thread."""

    date: QDate
    site: Site
    limit_altitude: float = 35.0
    time_samples: int = 240

    model_config = ConfigDict(arbitrary_types_allowed=True)


def targets_match(left: Target, right: Target, max_sep_deg: float = 0.05) -> bool:
    left_source = str(left.source_object_id or "").strip().lower()
    right_source = str(right.source_object_id or "").strip().lower()
    if left_source and right_source and left_source == right_source:
        return True
    if str(left.name or "").strip().lower() == str(right.name or "").strip().lower():
        return True
    try:
        return float(left.skycoord.separation(right.skycoord).deg) < max_sep_deg
    except Exception:
        return False


_targets_match = targets_match

__all__ = [
    "CalcRunStats",
    "DEFAULT_LIMITING_MAGNITUDE",
    "SessionSettings",
    "Site",
    "Target",
    "_targets_match",
    "targets_match",
]
