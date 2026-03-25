from __future__ import annotations

from astropy import units as u
from astropy.coordinates import Angle, SkyCoord


def _is_sexagesimal(text: str) -> bool:
    t = text.strip().lower()
    return any(token in t for token in (":", "h", "m", "s", "d"))


def parse_ra_to_deg(text: str) -> float:
    """Parse right ascension in sexagesimal hours or decimal degrees."""
    t = text.strip()
    if _is_sexagesimal(t):
        return Angle(t, unit=u.hourangle).wrap_at(360 * u.deg).deg
    return float(t)


def parse_dec_to_deg(text: str) -> float:
    """Parse declination in sexagesimal or decimal degrees."""
    t = text.strip()
    if _is_sexagesimal(t):
        return Angle(t, unit=u.deg).deg
    return float(t)


def parse_ra_dec_query(query: str) -> tuple[float, float]:
    """Parse a free-form RA/Dec pair string into decimal degrees."""
    normalized = query.replace(",", " ").strip()
    if not normalized:
        raise ValueError("Empty coordinate string.")

    parts = normalized.split()
    if len(parts) == 2 and not any(_is_sexagesimal(part) for part in parts):
        return float(parts[0]), float(parts[1])

    for units in ((u.hourangle, u.deg), (u.deg, u.deg)):
        try:
            coord = SkyCoord(normalized, unit=units)
            return float(coord.ra.deg), float(coord.dec.deg)
        except Exception:
            continue

    if len(parts) < 2:
        raise ValueError("Unrecognised coordinates. Expected an RA/Dec pair.")

    if len(parts) == 2:
        return parse_ra_to_deg(parts[0]), parse_dec_to_deg(parts[1])

    mid = len(parts) // 2
    ra_str = " ".join(parts[:mid])
    dec_str = " ".join(parts[mid:])
    return parse_ra_to_deg(ra_str), parse_dec_to_deg(dec_str)
