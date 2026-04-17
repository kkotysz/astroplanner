from __future__ import annotations

import csv
import json
import logging
import math
import os
import warnings
from time import perf_counter
from typing import Any, Callable, Optional
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.simbad import Simbad
from PySide6.QtCore import QThread, Qt, Signal, Slot
from PySide6.QtWidgets import QApplication

from astroplanner.models import Target
from astroplanner.parsing import parse_dec_to_deg, parse_ra_dec_query, parse_ra_to_deg

try:
    from astroquery.exceptions import NoResultsWarning
except Exception:  # pragma: no cover - fallback only for older astroquery variants
    class NoResultsWarning(Warning):
        pass


logger = logging.getLogger(__name__)


TARGET_SEARCH_SOURCES: list[tuple[str, str]] = [
    ("simbad", "SIMBAD"),
    ("gaia_dr3", "Gaia DR3"),
    ("gaia_alerts", "Gaia Alerts"),
    ("tns", "TNS"),
    ("ned", "NED"),
    ("lsst", "LSST"),
]
TARGET_SOURCE_LABELS: dict[str, str] = {key: label for key, label in TARGET_SEARCH_SOURCES}
TARGET_SOURCE_LABELS["bhtom"] = "BHTOM"
TARGET_SOURCE_LABELS["coordinates"] = "Manual coordinates"
TARGET_SOURCE_LABELS["manual"] = "Manual target"

TNS_ENDPOINT_CHOICES: list[tuple[str, str]] = [
    ("production", "Production (www.wis-tns.org)"),
    ("sandbox", "Sandbox (sandbox.wis-tns.org)"),
]

TNS_ENDPOINT_BASE_URLS: dict[str, str] = {
    "production": "https://www.wis-tns.org/api",
    "sandbox": "https://sandbox.wis-tns.org/api",
}

SIMBAD_COMPACT_CACHE_TTL_S = 30 * 24 * 60 * 60
SIMBAD_COMPACT_NEGATIVE_CACHE_TTL_S = 6 * 60 * 60


def _decode_simbad_value(value: object) -> str:
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="ignore").strip()
    return str(value).strip()


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


def _target_source_label(source_key: object) -> str:
    key = _normalize_catalog_token(source_key)
    if not key:
        return "Saved target"
    return TARGET_SOURCE_LABELS.get(key, str(source_key).strip() or "Saved target")


def _target_magnitude_label(target: "Target") -> str:
    return "Last Mag" if _normalize_catalog_token(target.source_catalog) == "bhtom" else "Mag"


def _object_type_is_unknown(value: object) -> bool:
    token = _normalize_catalog_token(value)
    return token in {"", "-", "unknown", "unk", "n/a", "na", "none"}


def _normalize_catalog_display_name(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return " ".join(text.split())


def _simbad_column(result, *candidates: str) -> Optional[str]:
    if not hasattr(result, "colnames"):
        return None
    lookup = {name.lower(): name for name in result.colnames}
    for candidate in candidates:
        hit = lookup.get(candidate.lower())
        if hit:
            return hit
    return None


def _simbad_has_row(result, row_idx: int = 0) -> bool:
    if result is None:
        return False
    try:
        return len(result) > row_idx
    except Exception:
        return False


def _simbad_cell(result, column_name: str, row_idx: int = 0) -> Optional[object]:
    if not _simbad_has_row(result, row_idx):
        return None
    try:
        column = result[column_name]
    except Exception:
        return None
    try:
        if len(column) <= row_idx:
            return None
    except Exception:
        pass
    try:
        return column[row_idx]
    except Exception:
        return None


def _extract_simbad_metadata(result, row_idx: int = 0) -> tuple[Optional[float], str]:
    magnitude: Optional[float] = None
    object_type = ""
    if not _simbad_has_row(result, row_idx):
        return magnitude, object_type

    for candidates in (("V", "FLUX_V"), ("R", "FLUX_R"), ("B", "FLUX_B")):
        col = _simbad_column(result, *candidates)
        if col is None:
            continue
        raw = _simbad_cell(result, col, row_idx)
        if raw is None:
            continue
        if np.ma.is_masked(raw):
            continue
        try:
            magnitude = float(raw)
            break
        except (TypeError, ValueError):
            continue

    col = _simbad_column(result, "OTYPE")
    if col is not None:
        raw = _simbad_cell(result, col, row_idx)
        if raw is not None and not np.ma.is_masked(raw):
            object_type = _decode_simbad_value(raw)

    return magnitude, object_type


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


def _extract_simbad_photometry(result, row_idx: int = 0) -> dict[str, float]:
    photometry: dict[str, float] = {}
    if not _simbad_has_row(result, row_idx):
        return photometry

    for label, candidates in (
        ("B", ("B", "FLUX_B")),
        ("V", ("V", "FLUX_V")),
        ("R", ("R", "FLUX_R")),
    ):
        col = _simbad_column(result, *candidates)
        if col is None:
            continue
        raw = _simbad_cell(result, col, row_idx)
        if raw is None or np.ma.is_masked(raw):
            continue
        try:
            photometry[label] = float(raw)
        except (TypeError, ValueError):
            continue

    return photometry


def _extract_simbad_compact_measurements(result, row_idx: int = 0) -> dict[str, object]:
    details: dict[str, object] = {
        "photometry": _extract_simbad_photometry(result, row_idx=row_idx),
    }
    if not _simbad_has_row(result, row_idx):
        return details

    text_columns = {
        "sp_type": ("sp_type", "messpt.sptype"),
        "distance_unit": ("mesdistance.unit",),
    }
    float_columns = {
        "parallax_mas": ("plx_value", "mesplx.plx"),
        "parallax_err_mas": ("plx_err",),
        "distance_value": ("mesdistance.dist",),
        "distance_plus_err": ("mesdistance.plus_err",),
        "distance_minus_err": ("mesdistance.minus_err",),
        "teff_k": ("mesfe_h.teff",),
        "fe_h": ("mesfe_h.fe_h",),
        "size_major_arcmin": ("galdim_majaxis", "mesdiameter.diameter"),
        "size_minor_arcmin": ("galdim_minaxis",),
        "radial_velocity_kms": ("rvz_radvel", "mesvelocities.velvalue"),
        "radial_velocity_err_kms": ("rvz_err", "mesvelocities.meanerror"),
        "redshift": ("rvz_redshift",),
    }

    for key, candidates in text_columns.items():
        col = _simbad_column(result, *candidates)
        if col is None:
            continue
        raw = _simbad_cell(result, col, row_idx)
        if raw is None or np.ma.is_masked(raw):
            continue
        text = _decode_simbad_value(raw)
        if text:
            details[key] = text

    for key, candidates in float_columns.items():
        col = _simbad_column(result, *candidates)
        if col is None:
            continue
        value = _safe_float(_simbad_cell(result, col, row_idx))
        if value is not None and math.isfinite(value):
            details[key] = float(value)

    return details


def _extract_simbad_name(result, fallback: str, row_idx: int = 0) -> str:
    if not _simbad_has_row(result, row_idx):
        return fallback
    col = _simbad_column(result, "MAIN_ID", "main_id", "ID", "matched_id")
    if col is None:
        return fallback
    raw = _simbad_cell(result, col, row_idx)
    if raw is None or np.ma.is_masked(raw):
        return fallback
    value = _decode_simbad_value(raw)
    return value or fallback


def _simbad_row_coord(result, row_idx: int = 0) -> Optional[SkyCoord]:
    if not _simbad_has_row(result, row_idx):
        return None

    ra_deg_col = _simbad_column(result, "RA_d", "RA(deg)")
    dec_deg_col = _simbad_column(result, "DEC_d", "DEC(deg)")
    if ra_deg_col is not None and dec_deg_col is not None:
        ra_deg = _safe_float(_simbad_cell(result, ra_deg_col, row_idx))
        dec_deg = _safe_float(_simbad_cell(result, dec_deg_col, row_idx))
        if ra_deg is not None and dec_deg is not None:
            return SkyCoord(ra=float(ra_deg) * u.deg, dec=float(dec_deg) * u.deg, frame="icrs")

    ra_col = _simbad_column(result, "RA", "ra")
    dec_col = _simbad_column(result, "DEC", "dec")
    if ra_col is None or dec_col is None:
        return None
    ra_raw = _simbad_cell(result, ra_col, row_idx)
    dec_raw = _simbad_cell(result, dec_col, row_idx)
    if ra_raw is None or dec_raw is None or np.ma.is_masked(ra_raw) or np.ma.is_masked(dec_raw):
        return None
    ra_deg = _safe_float(ra_raw)
    dec_deg = _safe_float(dec_raw)
    if ra_deg is not None and dec_deg is not None:
        return SkyCoord(ra=float(ra_deg) * u.deg, dec=float(dec_deg) * u.deg, frame="icrs")
    ra_txt = _decode_simbad_value(ra_raw)
    dec_txt = _decode_simbad_value(dec_raw)
    if not ra_txt or not dec_txt:
        return None
    try:
        return SkyCoord(ra_txt, dec_txt, unit=(u.hourangle, u.deg), frame="icrs")
    except Exception:
        return None


def _simbad_best_row_index(result, reference_coord: Optional[SkyCoord] = None) -> int:
    if reference_coord is None or not _simbad_has_row(result):
        return 0
    try:
        total_rows = len(result)
    except Exception:
        return 0

    best_idx = 0
    best_sep = float("inf")
    for row_idx in range(total_rows):
        row_coord = _simbad_row_coord(result, row_idx=row_idx)
        if row_coord is None:
            continue
        try:
            sep_arcsec = float(row_coord.separation(reference_coord).arcsec)
        except Exception:
            continue
        if sep_arcsec < best_sep:
            best_sep = sep_arcsec
            best_idx = row_idx
    return best_idx


def _build_tns_marker(bot_id: int | str, bot_name: str) -> str:
    # Keep canonical format used in TNS FAQ examples.
    return f'tns_marker{{"tns_id": "{bot_id}", "type": "bot", "name": "{bot_name}"}}'


def _normalize_tns_endpoint_key(value: object) -> str:
    key = str(value or "").strip().lower()
    if key in TNS_ENDPOINT_BASE_URLS:
        return key
    if "sandbox" in key:
        return "sandbox"
    return "production"


def _tns_api_base_url(value: object) -> str:
    key = _normalize_tns_endpoint_key(value)
    return TNS_ENDPOINT_BASE_URLS[key]


class MetadataLookupWorker(QThread):
    """Background metadata fetch for SIMBAD magnitudes/types with cancellation."""

    completed = Signal(int, list)

    def __init__(self, request_id: int, names: list[str], parent=None):
        super().__init__(parent)
        self.setObjectName(self.__class__.__name__)
        self.request_id = request_id
        self.names = names
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        results: list[tuple[str, str, Optional[float], str]] = []
        if not self.names:
            self.completed.emit(self.request_id, results)
            return
        try:
            custom = Simbad()
            custom.add_votable_fields("V", "R", "B", "otype")
        except Exception:
            custom = None

        for name in self.names:
            if self._cancelled:
                break
            key = name.strip().lower()
            if not key:
                continue
            magnitude: Optional[float] = None
            object_type = ""
            main_id = key
            try:
                if custom is not None:
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=NoResultsWarning)
                            result = custom.query_object(name)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Metadata worker query failed for '%s': %s", name, exc)
                        result = None
                    if _simbad_has_row(result):
                        magnitude, object_type = _extract_simbad_metadata(result)
                        main_id = _extract_simbad_name(result, name).strip().lower() or key
            except Exception as exc:  # noqa: BLE001
                logger.warning("Metadata worker processing failed for '%s': %s", name, exc)
            results.append((key, main_id, magnitude, object_type))
        self.completed.emit(self.request_id, results)


class TargetResolver:
    """Resolve target names/coordinates and enrich target metadata."""

    def __init__(self, planner: object) -> None:
        object.__setattr__(self, "_planner", planner)

    def __getattr__(self, name: str):
        return getattr(self._planner, name)

    def __setattr__(self, name: str, value: object) -> None:
        if name == "_planner":
            object.__setattr__(self, name, value)
            return
        setattr(self._planner, name, value)

    def resolve(self, query: str, source: str = "simbad") -> Target:
        return self._resolve_target(query, source)

    def fetch_missing_magnitudes(self, targets: Optional[list[Target]] = None, emit_table: bool = True) -> int:
        return self._fetch_missing_magnitudes(targets, emit_table=emit_table)

    def fetch_missing_magnitudes_async(self) -> int:
        return self._fetch_missing_magnitudes_async()

    def _enrich_target_metadata_for_dialog(self, target: Target) -> None:
        self._fetch_missing_magnitudes([target], emit_table=False)

    def _cancel_metadata_lookup(self):
        worker = self._meta_worker
        if worker is None:
            return
        try:
            if worker.isRunning():
                worker.cancel()
                worker.quit()
                worker.wait()
        except Exception:  # noqa: BLE001
            pass
        self._meta_worker = None

    @Slot(int, list)
    def _on_metadata_lookup_completed(self, request_id: int, results: list):
        if request_id != self._meta_request_id:
            return
        updated_magnitude = 0
        updated_type = 0
        for key, main_id, magnitude, object_type in results:
            self._simbad_meta_cache[key] = (magnitude, object_type)
            if main_id:
                self._simbad_meta_cache[main_id] = (magnitude, object_type)

        for tgt in self.targets:
            cache_key = tgt.name.strip().lower()
            if not cache_key:
                continue
            cached = self._simbad_meta_cache.get(cache_key)
            if cached is None:
                continue
            magnitude, object_type = cached
            if tgt.magnitude is None and magnitude is not None:
                tgt.magnitude = float(magnitude)
                updated_magnitude += 1
            if not tgt.object_type and object_type:
                tgt.object_type = object_type
                updated_type += 1

        if updated_magnitude or updated_type:
            self._emit_table_data_changed()
            self._update_selected_details()
            logger.info(
                "Updated missing metadata from SIMBAD (magnitude=%d, object_type=%d)",
                updated_magnitude,
                updated_type,
            )
        self._meta_worker = None

    def _fetch_missing_magnitudes_async(self) -> int:
        pending_names: list[str] = []
        updated_magnitude = 0
        updated_type = 0
        for tgt in self.targets:
            if tgt.magnitude is not None and tgt.object_type:
                continue
            key = tgt.name.strip().lower()
            if not key:
                continue
            cached = self._simbad_meta_cache.get(key)
            if cached is not None:
                magnitude, object_type = cached
                if tgt.magnitude is None and magnitude is not None:
                    tgt.magnitude = float(magnitude)
                    updated_magnitude += 1
                if not tgt.object_type and object_type:
                    tgt.object_type = object_type
                    updated_type += 1
                continue
            pending_names.append(tgt.name)

        if updated_magnitude or updated_type:
            self._emit_table_data_changed()
            self._update_selected_details()

        names = sorted({name.strip() for name in pending_names if name.strip()})
        if not names:
            return updated_magnitude

        self._cancel_metadata_lookup()
        self._meta_request_id += 1
        worker = MetadataLookupWorker(self._meta_request_id, names, self._planner)
        worker.completed.connect(self._on_metadata_lookup_completed)
        worker.finished.connect(worker.deleteLater)
        self._meta_worker = worker
        worker.start()
        return updated_magnitude

    def _fetch_missing_magnitudes(self, targets: Optional[list[Target]] = None, emit_table: bool = True) -> int:
        pending = [t for t in (targets if targets is not None else self.targets) if t.magnitude is None or not t.object_type]
        if not pending:
            return 0

        updated_magnitude = 0
        updated_type = 0
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            for tgt in pending:
                key = tgt.name.strip().lower()
                if not key:
                    continue

                cached = self._simbad_meta_cache.get(key)
                if cached is None:
                    try:
                        custom = Simbad()
                        custom.add_votable_fields("V", "R", "B", "otype")
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=NoResultsWarning)
                            result = custom.query_object(tgt.name)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Magnitude lookup failed for '%s': %s", tgt.name, exc)
                        self._simbad_meta_cache[key] = (None, "")
                        continue

                    if not _simbad_has_row(result):
                        self._simbad_meta_cache[key] = (None, "")
                        continue

                    magnitude, object_type = _extract_simbad_metadata(result)
                    self._simbad_meta_cache[key] = (magnitude, object_type)
                    main_id = _extract_simbad_name(result, tgt.name).lower()
                    if main_id:
                        self._simbad_meta_cache[main_id] = (magnitude, object_type)
                    cached = (magnitude, object_type)

                magnitude, object_type = cached
                if tgt.magnitude is None and magnitude is not None:
                    tgt.magnitude = float(magnitude)
                    updated_magnitude += 1
                if not tgt.object_type and object_type:
                    tgt.object_type = object_type
                    updated_type += 1
        finally:
            QApplication.restoreOverrideCursor()

        if (updated_magnitude or updated_type) and emit_table:
            self._emit_table_data_changed()
            self._update_selected_details()
        if updated_magnitude or updated_type:
            logger.info(
                "Updated missing metadata from SIMBAD (magnitude=%d, object_type=%d)",
                updated_magnitude,
                updated_type,
            )
        return updated_magnitude

    def _resolve_target_from_coordinates(self, query: str) -> Optional[Target]:
        try:
            ra_deg, dec_deg = parse_ra_dec_query(query)
        except Exception:
            return None
        return Target(
            name=query,
            ra=ra_deg,
            dec=dec_deg,
            source_catalog="coordinates",
            source_object_id=query,
        )

    def _resolve_target_simbad(self, query: str) -> Target:
        try:
            custom = Simbad()
            custom.add_votable_fields("ra", "dec", "V", "R", "B", "otype")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=NoResultsWarning)
                result = custom.query_object(query)
            if _simbad_has_row(result):
                ra_col = _simbad_column(result, "RA", "ra")
                dec_col = _simbad_column(result, "DEC", "dec")
                if ra_col is None or dec_col is None:
                    raise KeyError("RA/DEC")
                ra_cell = _simbad_cell(result, ra_col, 0)
                dec_cell = _simbad_cell(result, dec_col, 0)
                if ra_cell is None or dec_cell is None:
                    raise KeyError("RA/DEC row")
                ra_raw = _decode_simbad_value(ra_cell)
                dec_raw = _decode_simbad_value(dec_cell)
                try:
                    ra_deg = float(ra_raw)
                    dec_deg = float(dec_raw)
                except ValueError:
                    ra_deg = parse_ra_to_deg(ra_raw)
                    dec_deg = parse_dec_to_deg(dec_raw)
                name_res = _extract_simbad_name(result, query)
                magnitude, object_type = _extract_simbad_metadata(result)
                self._simbad_meta_cache[query.strip().lower()] = (magnitude, object_type)
                self._simbad_meta_cache[name_res.strip().lower()] = (magnitude, object_type)
                return Target(
                    name=name_res,
                    ra=ra_deg,
                    dec=dec_deg,
                    source_catalog="simbad",
                    source_object_id=name_res or query,
                    magnitude=magnitude,
                    object_type=object_type,
                )
        except Exception as exc:
            logger.warning("Simbad resolver failed for '%s': %s", query, exc)

        try:
            coord = SkyCoord.from_name(query)
            target = Target.from_skycoord(query, coord)
            target.source_catalog = "simbad"
            target.source_object_id = query
            return target
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"No SIMBAD/Sesame match for '{query}'.") from exc

    def _resolve_target_gaia_dr3(self, query: str) -> Target:
        try:
            from astroquery.gaia import Gaia
        except Exception as exc:  # pragma: no cover - import guarded at runtime
            raise RuntimeError("Gaia DR3 resolver is unavailable (astroquery.gaia import failed).") from exc

        def _target_from_results(results, row_idx: int = 0) -> Optional[Target]:
            if not _simbad_has_row(results, row_idx):
                return None
            ra_col = _simbad_column(results, "ra")
            dec_col = _simbad_column(results, "dec")
            if ra_col is None or dec_col is None:
                return None
            ra_deg = _safe_float(_simbad_cell(results, ra_col, row_idx))
            dec_deg = _safe_float(_simbad_cell(results, dec_col, row_idx))
            if ra_deg is None or dec_deg is None:
                return None
            mag_col = _simbad_column(results, "phot_g_mean_mag")
            magnitude = _safe_float(_simbad_cell(results, mag_col, row_idx)) if mag_col else None
            src_col = _simbad_column(results, "source_id")
            source_id = _decode_simbad_value(_simbad_cell(results, src_col, row_idx)) if src_col else ""
            designation_col = _simbad_column(results, "designation")
            designation = _decode_simbad_value(_simbad_cell(results, designation_col, row_idx)) if designation_col else ""
            name = designation or (f"Gaia DR3 {source_id}".strip() if source_id else query)
            return Target(
                name=name,
                ra=float(ra_deg),
                dec=float(dec_deg),
                source_catalog="gaia_dr3",
                source_object_id=designation or source_id or name,
                magnitude=magnitude,
                object_type="Gaia DR3 source",
            )

        adql_queries: list[str] = []
        if query.isdigit():
            adql_queries.append(
                "SELECT TOP 1 source_id, designation, ra, dec, phot_g_mean_mag "
                "FROM gaiadr3.gaia_source "
                f"WHERE source_id = {int(query)}"
            )

        designation_candidates = [query]
        if not query.lower().startswith("gaia dr3"):
            designation_candidates.append(f"Gaia DR3 {query}")
        seen_designations: set[str] = set()
        for designation in designation_candidates:
            token = _normalize_catalog_token(designation)
            if token in seen_designations:
                continue
            seen_designations.add(token)
            safe_designation = designation.replace("'", "''")
            adql_queries.append(
                "SELECT TOP 1 source_id, designation, ra, dec, phot_g_mean_mag "
                "FROM gaiadr3.gaia_source "
                f"WHERE UPPER(designation) = UPPER('{safe_designation}')"
            )

        for adql in adql_queries:
            try:
                job = Gaia.launch_job(adql, dump_to_file=False)
                target = _target_from_results(job.get_results())
            except Exception as exc:
                logger.warning("Gaia DR3 query failed for '%s': %s", query, exc)
                continue
            if target is not None:
                return target

        # Try cross-identifiers from SIMBAD (e.g., "Gaia DR3 <source_id>").
        try:
            ids_result = Simbad.query_objectids(query)
            id_col = _simbad_column(ids_result, "ID", "id")
            if id_col is not None:
                for row_idx in range(len(ids_result)):
                    identifier_raw = _simbad_cell(ids_result, id_col, row_idx)
                    if identifier_raw is None or np.ma.is_masked(identifier_raw):
                        continue
                    identifier = _decode_simbad_value(identifier_raw)
                    if not identifier.lower().startswith("gaia dr3"):
                        continue
                    safe_designation = identifier.replace("'", "''")
                    adql = (
                        "SELECT TOP 1 source_id, designation, ra, dec, phot_g_mean_mag "
                        "FROM gaiadr3.gaia_source "
                        f"WHERE UPPER(designation) = UPPER('{safe_designation}')"
                    )
                    job = Gaia.launch_job(adql, dump_to_file=False)
                    target = _target_from_results(job.get_results())
                    if target is not None:
                        return target
        except Exception as exc:
            logger.warning("Gaia DR3 SIMBAD cross-id lookup failed for '%s': %s", query, exc)

        # Fallback: resolve by name, then cone search around that coordinate.
        try:
            coord = SkyCoord.from_name(query)
        except Exception as exc:
            raise ValueError(f"No Gaia DR3 match for '{query}'.") from exc

        try:
            job = Gaia.cone_search_async(coord, radius=120 * u.arcsec)
            results = job.get_results()
        except Exception as exc:
            raise RuntimeError(f"Gaia DR3 cone search failed for '{query}': {exc}") from exc

        if not _simbad_has_row(results):
            raise ValueError(f"No Gaia DR3 source found near '{query}'.")

        row_idx = 0
        ra_col = _simbad_column(results, "ra")
        dec_col = _simbad_column(results, "dec")
        if ra_col and dec_col and len(results) > 1:
            try:
                ra_vals = np.asarray(results[ra_col], dtype=float)
                dec_vals = np.asarray(results[dec_col], dtype=float)
                seps = SkyCoord(ra=ra_vals * u.deg, dec=dec_vals * u.deg).separation(coord)
                row_idx = int(np.nanargmin(seps.deg))
            except Exception:
                row_idx = 0

        target = _target_from_results(results, row_idx=row_idx)
        if target is None:
            raise ValueError(f"No Gaia DR3 source found near '{query}'.")
        return target

    def _load_gaia_alerts_cache(self, force_refresh: bool = False):
        ttl_seconds = 6 * 60 * 60
        age = perf_counter() - self._gaia_alerts_cache_loaded_at
        if self._gaia_alerts_cache and not force_refresh and age < ttl_seconds:
            return
        storage = getattr(self, "app_storage", None)
        if storage is not None and not force_refresh:
            try:
                cached_payload = storage.cache.get_json("gaia_alerts_catalog", "alerts.csv")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to read Gaia Alerts cache from storage: %s", exc)
            else:
                if isinstance(cached_payload, dict) and cached_payload:
                    self._gaia_alerts_cache = {
                        str(key): value
                        for key, value in cached_payload.items()
                        if isinstance(value, dict)
                    }
                    if self._gaia_alerts_cache:
                        self._gaia_alerts_cache_loaded_at = perf_counter()
                        return

        req = Request(
            "https://gsaweb.ast.cam.ac.uk/alerts/alerts.csv",
            headers={
                "User-Agent": "AstroPlanner/1.0 (desktop app)",
                "Accept": "text/csv,*/*;q=0.1",
            },
        )
        try:
            with urlopen(req, timeout=20) as resp:
                payload = resp.read().decode("utf-8", errors="replace")
        except Exception as exc:
            raise RuntimeError(f"Gaia Alerts download failed: {exc}") from exc

        parsed: dict[str, dict[str, str]] = {}
        for raw_row in csv.DictReader(payload.splitlines()):
            row: dict[str, str] = {}
            for key, value in raw_row.items():
                if key is None:
                    continue
                clean_key = str(key).strip().lstrip("#")
                row[clean_key] = str(value).strip() if value is not None else ""
            name = row.get("Name", "").strip()
            if not name:
                continue
            parsed[_normalize_catalog_token(name)] = row
            if name.lower().startswith("gaia"):
                alias = name[4:].strip()
                alias_key = _normalize_catalog_token(alias)
                if alias_key and alias_key not in parsed:
                    parsed[alias_key] = row

        if not parsed:
            raise RuntimeError("Gaia Alerts cache is empty after download.")
        self._gaia_alerts_cache = parsed
        self._gaia_alerts_cache_loaded_at = perf_counter()
        if storage is not None:
            try:
                storage.cache.set_json("gaia_alerts_catalog", "alerts.csv", parsed, ttl_s=ttl_seconds)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to persist Gaia Alerts cache: %s", exc)

    def _resolve_target_gaia_alerts(self, query: str) -> Target:
        self._load_gaia_alerts_cache()
        key = _normalize_catalog_token(query)
        row = self._gaia_alerts_cache.get(key)
        if row is None and not key.startswith("gaia"):
            row = self._gaia_alerts_cache.get(f"gaia{key}")
        if row is None:
            raise ValueError(f"Gaia Alerts object '{query}' was not found.")

        ra_deg = _safe_float(row.get("RaDeg"))
        dec_deg = _safe_float(row.get("DecDeg"))
        if ra_deg is None or dec_deg is None:
            raise ValueError(f"Gaia Alerts entry '{query}' has no valid coordinates.")

        magnitude = _safe_float(row.get("AlertMag"))
        object_type = row.get("Class", "").strip() or "Gaia Alert"
        name = row.get("Name", "").strip() or query
        return Target(
            name=name,
            ra=float(ra_deg),
            dec=float(dec_deg),
            source_catalog="gaia_alerts",
            source_object_id=name,
            magnitude=magnitude,
            object_type=object_type,
        )

    def _resolve_target_tns(self, query: str) -> Target:
        api_key = (os.getenv("TNS_API_KEY", "") or self.settings.value("general/tnsApiKey", "", type=str)).strip()
        if not api_key:
            raise RuntimeError("TNS requires API key. Set environment variable TNS_API_KEY.")
        endpoint_key = _normalize_tns_endpoint_key(self.settings.value("general/tnsEndpoint", "production", type=str))
        env_api_base = os.getenv("TNS_API_BASE_URL", "").strip()
        api_base = (env_api_base or _tns_api_base_url(endpoint_key)).rstrip("/")
        bot_id_raw = (os.getenv("TNS_BOT_ID", "") or self.settings.value("general/tnsBotId", "", type=str)).strip()
        bot_name = (os.getenv("TNS_BOT_NAME", "") or self.settings.value("general/tnsBotName", "", type=str)).strip()
        if not bot_id_raw or not bot_name:
            raise RuntimeError("TNS requires bot marker. Set TNS_BOT_ID and TNS_BOT_NAME.")
        try:
            bot_id = int(bot_id_raw)
        except ValueError as exc:
            raise RuntimeError("TNS_BOT_ID must be numeric.") from exc
        tns_marker = _build_tns_marker(bot_id, bot_name)

        def _tns_call(endpoint: str, req_payload: dict[str, object]) -> dict[str, Any]:
            body = urlencode(
                {
                    "api_key": api_key,
                    "data": json.dumps(req_payload),
                }
            ).encode("utf-8")
            req = Request(
                endpoint,
                data=body,
                headers={
                    "User-Agent": tns_marker,
                    "Accept": "application/json",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
            )
            try:
                with urlopen(req, timeout=20) as resp:
                    raw = resp.read().decode("utf-8")
            except HTTPError as exc:
                if exc.code in {401, 403}:
                    raise RuntimeError(
                        f"TNS unauthorized (401/403) on {endpoint_key}. Check API key, bot ID/name, bot activation and permissions."
                    ) from exc
                detail = ""
                try:
                    detail = exc.read().decode("utf-8", errors="ignore").strip()
                except Exception:
                    detail = ""
                if detail:
                    raise RuntimeError(f"TNS request failed ({exc.code}): {detail[:220]}") from exc
                raise RuntimeError(f"TNS request failed ({exc.code}).") from exc
            except Exception as exc:
                raise RuntimeError(f"TNS lookup failed: {exc}") from exc

            try:
                payload = json.loads(raw)
            except Exception as exc:
                raise RuntimeError("TNS returned non-JSON response.") from exc
            if not isinstance(payload, dict):
                raise RuntimeError("TNS returned invalid payload.")
            return payload

        def _extract_reply(payload: dict[str, Any]) -> Optional[dict[str, Any]]:
            data = payload.get("data")
            if isinstance(data, dict):
                reply = data.get("reply")
                if isinstance(reply, list) and reply and isinstance(reply[0], dict):
                    return reply[0]
                if isinstance(reply, dict) and reply:
                    return reply
                # Production TNS /get/object often returns the object directly in `data`.
                if data:
                    return data
            elif isinstance(data, list):
                # Some TNS responses wrap payload data in a list.
                if data and isinstance(data[0], dict) and "reply" in data[0]:
                    reply = data[0].get("reply")
                else:
                    reply = data
            else:
                reply = payload.get("reply")
            if isinstance(reply, list) and reply and isinstance(reply[0], dict):
                return reply[0]
            if isinstance(reply, dict) and reply:
                return reply
            return None

        def _extract_search_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
            data = payload.get("data")
            if isinstance(data, dict):
                reply = data.get("reply")
            elif isinstance(data, list):
                if data and isinstance(data[0], dict) and "reply" in data[0]:
                    reply = data[0].get("reply")
                else:
                    reply = data
            else:
                reply = payload.get("reply")

            if isinstance(reply, list):
                return [it for it in reply if isinstance(it, dict)]
            if isinstance(reply, dict):
                return [reply]
            return []

        raw_query = query.strip()
        if not raw_query:
            raise ValueError("TNS query cannot be empty.")

        candidates: list[str] = []

        def _add_candidate(name: str):
            item = name.strip()
            if item and item not in candidates:
                candidates.append(item)

        _add_candidate(raw_query)

        # Accept compact forms used on the website (e.g. "AT2025abcd", "SN2024xyz").
        low_raw = raw_query.lower()
        for prefix in ("at", "sn"):
            if low_raw.startswith(prefix):
                rest = raw_query[len(prefix):].lstrip()
                if rest:
                    _add_candidate(rest)
                    _add_candidate(f"{prefix.upper()} {rest}")
                    _add_candidate(f"{prefix.upper()}{rest}")
                break

        token = candidates[-1] if candidates else raw_query
        if token and token[0].isdigit():
            _add_candidate(f"AT {token}")
            _add_candidate(f"AT{token}")
            _add_candidate(f"SN {token}")
            _add_candidate(f"SN{token}")

        reply: Optional[dict[str, Any]] = None
        last_message = ""

        # 1) Direct object lookup attempts.
        for candidate in candidates:
            payload = _tns_call(
                f"{api_base}/get/object",
                {
                    "objname": candidate,
                    "objid": "",
                    "photometry": "0",
                    "spectra": "0",
                },
            )
            candidate_reply = _extract_reply(payload)
            if candidate_reply is not None:
                reply = candidate_reply
                break
            msg = str(payload.get("id_message", "")).strip()
            if msg:
                last_message = msg

        # 2) Search API fallback to discover canonical name/prefix.
        if reply is None:
            discovered_candidates: list[str] = []
            for candidate in candidates:
                payload = _tns_call(
                    f"{api_base}/get/search",
                    {
                        "objname": candidate,
                    },
                )
                items = _extract_search_items(payload)
                for item in items:
                    objname = str(item.get("objname", "") or "").strip()
                    prefix = str(item.get("prefix", "") or "").strip().upper()
                    full_name = str(item.get("name", "") or "").strip()
                    for name in (
                        full_name,
                        objname,
                        f"{prefix} {objname}" if prefix and objname else "",
                        f"{prefix}{objname}" if prefix and objname else "",
                    ):
                        n = name.strip()
                        if n and n not in discovered_candidates:
                            discovered_candidates.append(n)
                msg = str(payload.get("id_message", "")).strip()
                if msg:
                    last_message = msg

            for discovered in discovered_candidates:
                payload = _tns_call(
                    f"{api_base}/get/object",
                    {
                        "objname": discovered,
                        "objid": "",
                        "photometry": "0",
                        "spectra": "0",
                    },
                )
                candidate_reply = _extract_reply(payload)
                if candidate_reply is not None:
                    reply = candidate_reply
                    break
                msg = str(payload.get("id_message", "")).strip()
                if msg:
                    last_message = msg

        if reply is None:
            msg = last_message or "object not returned by API"
            raise ValueError(f"TNS did not return an object for '{query}' ({msg}).")

        lookup = {str(k).lower(): v for k, v in reply.items()}

        def _get_reply(*keys: str) -> object:
            for key in keys:
                if key.lower() in lookup:
                    return lookup[key.lower()]
            return None

        ra_deg = _safe_float(_get_reply("radeg", "ra_deg"))
        dec_deg = _safe_float(_get_reply("decdeg", "dec_deg"))
        if ra_deg is None:
            ra_raw = str(_get_reply("ra", "ra_hms", "ra_hms_str") or "").strip()
            if ra_raw:
                ra_deg = parse_ra_to_deg(ra_raw)
        if dec_deg is None:
            dec_raw = str(_get_reply("dec", "dec_dms", "dec_dms_str") or "").strip()
            if dec_raw:
                dec_deg = parse_dec_to_deg(dec_raw)
        if ra_deg is None or dec_deg is None:
            raise ValueError(f"TNS object '{query}' has no usable coordinates.")

        object_type_raw = _get_reply("object_type", "objtype", "type")
        object_type = ""
        if isinstance(object_type_raw, dict):
            object_type = str(
                object_type_raw.get("name")
                or object_type_raw.get("value")
                or ""
            ).strip()
        elif object_type_raw is not None:
            object_type = str(object_type_raw).strip()
        if not object_type:
            object_type = "TNS transient"

        magnitude = _safe_float(_get_reply("discoverymag", "discovery_mag", "max_mag"))
        name = str(_get_reply("objname", "name", "internal_name") or query).strip() or query
        prefix = str(_get_reply("name_prefix", "prefix") or "").strip().upper()
        if prefix and not name.lower().startswith(prefix.lower()):
            name = f"{prefix}{name}"
        return Target(
            name=name,
            ra=float(ra_deg),
            dec=float(dec_deg),
            source_catalog="tns",
            source_object_id=name,
            magnitude=magnitude,
            object_type=object_type,
        )

    def _resolve_target_ned(self, query: str) -> Target:
        try:
            from astroquery.ipac.ned import Ned
        except Exception as exc:  # pragma: no cover - import guarded at runtime
            raise RuntimeError("NED resolver is unavailable (astroquery.ipac.ned import failed).") from exc

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=NoResultsWarning)
                result = Ned.query_object(query)
        except Exception as exc:
            raise RuntimeError(f"NED lookup failed for '{query}': {exc}") from exc

        if not _simbad_has_row(result):
            raise ValueError(f"NED object '{query}' was not found.")

        ra_col = _simbad_column(result, "RA(deg)", "RA")
        dec_col = _simbad_column(result, "DEC(deg)", "Dec", "DEC")
        if ra_col is None or dec_col is None:
            raise ValueError(f"NED object '{query}' has no coordinate columns.")

        ra_deg = _safe_float(_simbad_cell(result, ra_col, 0))
        dec_deg = _safe_float(_simbad_cell(result, dec_col, 0))
        if ra_deg is None or dec_deg is None:
            raise ValueError(f"NED object '{query}' has invalid coordinate values.")

        name_col = _simbad_column(result, "Object Name", "Object_Name", "name")
        name_raw = _simbad_cell(result, name_col, 0) if name_col else None
        name = _decode_simbad_value(name_raw) if name_raw is not None else query

        type_col = _simbad_column(result, "Type", "Object Type", "objtype")
        type_raw = _simbad_cell(result, type_col, 0) if type_col else None
        object_type = _decode_simbad_value(type_raw) if type_raw is not None else ""
        if not object_type:
            object_type = "NED object"

        return Target(
            name=name or query,
            ra=float(ra_deg),
            dec=float(dec_deg),
            source_catalog="ned",
            source_object_id=(name or query),
            object_type=object_type,
        )

    def _resolve_target_lsst(self, query: str) -> Target:
        try:
            coord = SkyCoord.from_name(query)
        except Exception as exc:
            raise ValueError(
                f"LSST name lookup for '{query}' is unavailable. Use coordinates or another source."
            ) from exc
        target = Target.from_skycoord(query, coord)
        target.source_catalog = "lsst"
        target.source_object_id = query
        target.object_type = "LSST candidate"
        return target

    def _resolve_target(self, query: str, source: str = "simbad") -> Target:
        query = query.strip()
        if not query:
            raise ValueError("Query cannot be empty.")

        source_key = _normalize_catalog_token(source) or "simbad"
        source_labels = {key: label for key, label in TARGET_SEARCH_SOURCES}
        source_label = source_labels.get(source_key, source_key.upper())
        resolvers: dict[str, Callable[[str], Target]] = {
            "simbad": self._resolve_target_simbad,
            "gaia_dr3": self._resolve_target_gaia_dr3,
            "gaia_alerts": self._resolve_target_gaia_alerts,
            "tns": self._resolve_target_tns,
            "ned": self._resolve_target_ned,
            "lsst": self._resolve_target_lsst,
        }
        resolver = resolvers.get(source_key)
        if resolver is None:
            raise ValueError(f"Unsupported source '{source}'.")

        last_error: Optional[Exception] = None
        try:
            return resolver(query)
        except Exception as exc:
            last_error = exc
            logger.warning("%s resolver failed for '%s': %s", source_label, query, exc)

        fallback = self._resolve_target_from_coordinates(query)
        if fallback is not None:
            return fallback

        if last_error is None:
            raise ValueError(f"Unable to resolve '{query}' using {source_label}.")
        raise ValueError(f"Unable to resolve '{query}' using {source_label}: {last_error}") from last_error


__all__ = [
    "MetadataLookupWorker",
    "SIMBAD_COMPACT_CACHE_TTL_S",
    "SIMBAD_COMPACT_NEGATIVE_CACHE_TTL_S",
    "TARGET_SEARCH_SOURCES",
    "TARGET_SOURCE_LABELS",
    "TNS_ENDPOINT_BASE_URLS",
    "TNS_ENDPOINT_CHOICES",
    "TargetResolver",
    "_airmass_from_altitude_values",
    "_build_tns_marker",
    "_decode_simbad_value",
    "_extract_simbad_compact_measurements",
    "_extract_simbad_metadata",
    "_extract_simbad_name",
    "_extract_simbad_photometry",
    "_normalize_catalog_display_name",
    "_normalize_catalog_token",
    "_normalize_tns_endpoint_key",
    "_object_type_is_unknown",
    "_safe_float",
    "_safe_int",
    "_simbad_best_row_index",
    "_simbad_cell",
    "_simbad_column",
    "_simbad_has_row",
    "_simbad_row_coord",
    "_target_magnitude_label",
    "_target_source_label",
    "_tns_api_base_url",
]
