"""
Astronomical Observation Planner GUI
-----------------------------------
Python 3.12 | PySide6

This is a simple GUI application for planning astronomical observations. It allows you to:
- Load celestial targets from a JSON file.
- Select an observation site and date.
- Calculate visibility curves for each target.
- Plot the results using Matplotlib.
- Display current altitudes, azimuths, and separations from the Moon.

Dependencies
===========
- Python 3.12+
- PySide6
- Astroplan
- Astropy
- Matplotlib
- PyEphem
- Pydantic
- Astroquery
- TimezoneFinder
- NumPy

Run
===
```bash
python astro_planner.py
```

A tiny sample JSON file with targets is included below.  Save it as `example_targets.json` and load it.
```
[
    {"name": "M31", "ra": 10.684, "dec": 41.269},
    {"name": "Sirius", "ra": 6.752, "dec": -16.716},
    {"name": "Betelgeuse", "ra": 88.792939, "dec": 7.407064}
]
```
"""

# --- Imports --------------------------------------
from __future__ import annotations

# Standard library imports
import argparse
import base64
import copy
import csv
import hashlib
import html as html_module
import io
import json
import locale
import math
import os
import re
import sys
import threading
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urljoin, urlparse
from urllib.request import Request, urlopen
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, List, Optional
from time import perf_counter

import numpy as np
import pytz
import ephem
from astropy import units as u
from astropy.coordinates import SkyCoord, Angle
from astropy.time import Time
from astroplan import FixedTarget, Observer
from astroplan.observer import TargetAlwaysUpWarning

from astroquery.simbad import Simbad
try:
    from astroquery.exceptions import NoResultsWarning
except Exception:  # pragma: no cover - fallback only for older astroquery variants
    class NoResultsWarning(Warning):
        pass
from pydantic import ValidationError
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import font_manager as mpl_font_manager
from matplotlib import patheffects as mpl_patheffects
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
import warnings
import logging
import shiboken6 as shb   # PySide6 helper (isValid,objId)
from astroplanner.ai import (
    AIIntent,
    AI_CHAT_SPACING_CHOICES,
    AI_CHAT_TINT_CHOICES,
    AI_CHAT_WIDTH_CHOICES,
    ClassQuerySpec,
    CompareQuerySpec,
    KNOWLEDGE_DIR,
    KnowledgeNote,
    LLMConfig,
    LLMModelDiscoveryWorker,
    LLMWorker,
    ObjectQuerySpec,
    _format_knowledge_note_snippet,
    _knowledge_note_family,
    _load_knowledge_note,
    _looks_like_object_class_query,
    _looks_like_object_scoped_query,
    _looks_like_observing_guidance_query,
    _mentions_supernova_term,
    _normalize_knowledge_tag,
    _question_action_flags,
    _question_bhtom_type_markers,
    _requested_marker_family,
    _requested_marker_label,
    _requested_object_class_marker,
    _truncate_ai_memory_text,
    _type_label_class_family,
    _type_matches_requested_class,
)
from astroplanner.exporters import export_metrics_csv, export_observation_ics
from astroplanner.bhtom import (
    BHTOM_API_BASE_URL,
    BHTOM_MAX_OBSERVATORY_PAGES,
    BHTOM_MAX_SUGGESTION_PAGES,
    BHTOM_OBSERVATORY_CACHE_TTL_S,
    BHTOM_SUGGESTION_CACHE_TTL_S,
    BHTOM_SUGGESTION_MIN_IMPORTANCE,
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
from astroplanner.astronomy import AstronomyWorker, ClockWorker
from astroplanner.models import (
    CalcRunStats,
    DEFAULT_LIMITING_MAGNITUDE,
    SessionSettings,
    Site,
    Target,
    targets_match as _targets_match,
)
from astroplanner.parsing import parse_dec_to_deg, parse_ra_dec_query, parse_ra_to_deg
from astroplanner.scoring import TargetNightMetrics, compute_target_metrics
from astroplanner.seestar import (
    SEESTAR_ALP_LP_FILTER_AUTO,
    SEESTAR_ALP_LP_FILTER_OFF,
    SEESTAR_ALP_LP_FILTER_ON,
    SEESTAR_ALP_DEFAULT_BASE_URL,
    SEESTAR_ALP_DEFAULT_CLIENT_ID,
    SEESTAR_ALP_DEFAULT_DEVICE_NUM,
    SEESTAR_ALP_DEFAULT_GAIN,
    SEESTAR_ALP_DEFAULT_HONOR_QUEUE_TIMES,
    SEESTAR_ALP_DEFAULT_WAIT_UNTIL_LOCAL_TIME,
    SEESTAR_ALP_DEFAULT_STARTUP_SEQUENCE,
    SEESTAR_ALP_DEFAULT_STARTUP_POLAR_ALIGN,
    SEESTAR_ALP_DEFAULT_STARTUP_AUTO_FOCUS,
    SEESTAR_ALP_DEFAULT_STARTUP_DARK_FRAMES,
    SEESTAR_ALP_DEFAULT_CAPTURE_FLATS,
    SEESTAR_ALP_DEFAULT_FLATS_WAIT_S,
    SEESTAR_ALP_AF_MODE_OFF,
    SEESTAR_ALP_AF_MODE_PER_RUN,
    SEESTAR_ALP_AF_MODE_PER_TARGET,
    SEESTAR_ALP_DEFAULT_SCHEDULE_AUTOFOCUS_MODE,
    SEESTAR_ALP_DEFAULT_SCHEDULE_AUTOFOCUS,
    SEESTAR_ALP_DEFAULT_AUTOFOCUS_TRY_COUNT,
    SEESTAR_ALP_DEFAULT_DEW_HEATER_VALUE,
    SEESTAR_ALP_DEFAULT_PARK_AFTER_SESSION,
    SEESTAR_ALP_DEFAULT_SHUTDOWN_AFTER_SESSION,
    SEESTAR_ALP_DEFAULT_NUM_TRIES,
    SEESTAR_ALP_DEFAULT_PANEL_OVERLAP_PERCENT,
    SEESTAR_ALP_DEFAULT_RETRY_WAIT_S,
    SEESTAR_ALP_DEFAULT_STACK_EXPOSURE_MS,
    SEESTAR_ALP_DEFAULT_TIMEOUT_S,
    SEESTAR_ALP_DEFAULT_USE_AUTOFOCUS,
    SEESTAR_CAMPAIGN_PRESET_BL_LAC,
    SEESTAR_DEFAULT_BLOCK_MINUTES,
    SEESTAR_DEVICE_PROFILE_ALP,
    SEESTAR_DEVICE_PROFILE,
    SEESTAR_METHOD_ALP,
    SEESTAR_METHOD_GUIDED,
    SEESTAR_TEMPLATE_SCOPE_MULTI_TARGET,
    SEESTAR_TEMPLATE_SCOPE_SINGLE_TARGET,
    NightQueueSiteSnapshot,
    SeestarAlpAdapter,
    SeestarSessionTemplate,
    build_alp_web_ui_url,
    build_session_queue,
    default_science_checklist_items,
    normalize_seestar_alp_schedule_autofocus_mode,
    render_alp_backend_status_text,
    SeestarAlpClient,
    SeestarAlpConfig,
    SeestarGuidedAdapter,
)
from astroplanner.theme import (
    DEFAULT_DARK_MODE,
    DEFAULT_UI_THEME,
    THEME_CHOICES,
    COLORBLIND_LINE_COLORS,
    DEFAULT_LINE_COLORS,
    build_stylesheet,
    highlight_palette_for_theme,
    line_palette_for_theme,
    normalize_theme_key,
    resolve_theme_tokens,
)
from astroplanner.i18n import (
    SUPPORTED_UI_LANGUAGES,
    current_language,
    localize_widget_tree,
    resolve_language_choice,
    set_current_language,
    set_translated_text,
    set_translated_tooltip,
    translate_text,
)
from astroplanner.storage import AppStorage, SettingsAdapter
from astroplanner.targets_coordinator import TargetTableCoordinator
from astroplanner.ui.common import (
    SkeletonShimmerWidget,
    TargetTableView,
    _distribute_extra_table_width,
    _fit_dialog_to_screen,
)
from astroplanner.ui.add_target import AddTargetDialog, FinderChartWorker, MetadataLookupWorker
from astroplanner.ui.observatory import ObservatoryLookupWorker, ObservatoryManagerDialog
from astroplanner.ui.settings import GeneralSettingsDialog, TableSettingsDialog
from astroplanner.ui.seestar import SeestarSessionPlanDialog
from astroplanner.ui.suggestions import SuggestedTargetsDialog
from astroplanner.ui.targets import TargetTableModel
from astroplanner.weather import WeatherLiveWorker
from astroplanner.ui.weather import WeatherDialog

try:
    from PIL import Image, ImageDraw
except Exception:  # pragma: no cover - optional runtime dependency
    Image = None  # type: ignore[assignment]
    ImageDraw = None  # type: ignore[assignment]

try:
    import plotly
    from plotly.io import to_html as plotly_to_html
    from plotly.subplots import make_subplots as plotly_make_subplots
    import plotly.graph_objects as go

    _plotly_base_dir = Path(plotly.__file__).resolve().parent / "package_data"
    _PLOTLY_JS_BASE_DIR = str(_plotly_base_dir) if (_plotly_base_dir / "plotly.min.js").exists() else ""
    _HAS_PLOTLY = True
except Exception:  # pragma: no cover - optional runtime dependency
    plotly_to_html = None  # type: ignore[assignment]
    plotly_make_subplots = None  # type: ignore[assignment]
    go = None  # type: ignore[assignment]
    _PLOTLY_JS_BASE_DIR = ""
    _HAS_PLOTLY = False

# ── Logging configuration ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,                              # default level
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
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

CUTOUT_SURVEY_CHOICES: list[tuple[str, str, str]] = [
    ("dss2", "DSS2", "CDS/P/DSS2/color"),
    ("panstarrs", "PanSTARRS", "CDS/P/PanSTARRS/DR1/color-z-zg-g"),
    ("2mass", "2MASS", "CDS/P/2MASS/color"),
]
CUTOUT_VIEW_CHOICES: list[tuple[str, str]] = [
    ("aladin", "Aladin"),
    ("finderchart", "Finder chart"),
]
CUTOUT_DEFAULT_SURVEY_KEY = "dss2"
CUTOUT_DEFAULT_VIEW_KEY = "aladin"
CUTOUT_DEFAULT_FOV_ARCMIN = 15
CUTOUT_DEFAULT_SIZE_PX = 280
CUTOUT_MIN_FOV_ARCMIN = 5
CUTOUT_MAX_FOV_ARCMIN = 120
CUTOUT_MIN_SIZE_PX = 128
CUTOUT_MAX_SIZE_PX = 800
CUTOUT_CACHE_MAX = 24
# Fetch a slightly wider field for Aladin so zoom-out/pan has context around target.
CUTOUT_ALADIN_FETCH_MARGIN = 1.28
# Keep at least ~2 deg on the shorter axis for panning context without
# making telescope FOV overlays too small at max zoom.
CUTOUT_ALADIN_FETCH_MIN_ARCMIN = 120.0
# Prefer dynamic fetch size based on telescope FOV when available.
CUTOUT_ALADIN_FETCH_TELESCOPE_MARGIN = 5.0
CUTOUT_ALADIN_FETCH_TELESCOPE_MAX_ARCMIN = 480.0
# How much to widen fetched Aladin context when user zooms out past 1x.
CUTOUT_ALADIN_CONTEXT_STEP = 1.6
# Slight initial zoom so panning works immediately after load.
CUTOUT_ALADIN_INITIAL_PAN_ZOOM = 1.50
# Request higher native cutout resolution so zoom quality stays acceptable.
CUTOUT_ALADIN_FETCH_RES_MULT = 2.2
CUTOUT_ALADIN_FETCH_MIN_SHORT_PX = 900
CUTOUT_ALADIN_FETCH_MAX_EDGE_PX = 1440
FINDER_WORKER_TIMEOUT_MS = 10000
FINDER_RETRY_COOLDOWN_S = 45.0
SIMBAD_COMPACT_CACHE_TTL_S = 30 * 24 * 60 * 60
SIMBAD_COMPACT_NEGATIVE_CACHE_TTL_S = 6 * 60 * 60
QUICK_TARGETS_DEFAULT_COUNT = 10
QUICK_TARGETS_MIN_COUNT = 1
QUICK_TARGETS_MAX_COUNT = 50
VISIBILITY_AIRMASS_Y_MIN = 0.9
VISIBILITY_AIRMASS_Y_MAX = 2.1
VISIBILITY_AIRMASS_Y_TICKS = [0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.7, 1.9, 2.1]


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


def _normalize_cutout_survey_key(value: object) -> str:
    key = str(value or "").strip().lower()
    if key in {"decals", "cds/p/decals/dr5/color"}:
        # Legacy migration: DECaLS was removed from options.
        return "2mass"
    for survey_key, _, hips in CUTOUT_SURVEY_CHOICES:
        if key == survey_key or key == hips.lower():
            return survey_key
    return CUTOUT_DEFAULT_SURVEY_KEY


def _normalize_cutout_view_key(value: object) -> str:
    key = str(value or "").strip().lower()
    for view_key, _ in CUTOUT_VIEW_CHOICES:
        if key == view_key:
            return view_key
    return CUTOUT_DEFAULT_VIEW_KEY


def _cutout_survey_label(key: object) -> str:
    norm = _normalize_cutout_survey_key(key)
    for survey_key, label, _ in CUTOUT_SURVEY_CHOICES:
        if survey_key == norm:
            return label
    return "DSS2"


def _cutout_survey_hips(key: object) -> str:
    norm = _normalize_cutout_survey_key(key)
    for survey_key, _, hips in CUTOUT_SURVEY_CHOICES:
        if survey_key == norm:
            return hips
    return CUTOUT_SURVEY_CHOICES[0][2]


def _sanitize_cutout_fov_arcmin(value: object) -> int:
    try:
        ivalue = int(float(value))
    except Exception:
        ivalue = CUTOUT_DEFAULT_FOV_ARCMIN
    return max(CUTOUT_MIN_FOV_ARCMIN, min(CUTOUT_MAX_FOV_ARCMIN, ivalue))


def _sanitize_cutout_size_px(value: object) -> int:
    try:
        ivalue = int(float(value))
    except Exception:
        ivalue = CUTOUT_DEFAULT_SIZE_PX
    return max(CUTOUT_MIN_SIZE_PX, min(CUTOUT_MAX_SIZE_PX, ivalue))

# GUI imports (PySide6)
from PySide6.QtCore import (
    Property,
    QBuffer,
    QByteArray,
    QDate,
    QElapsedTimer,
    QEasingCurve,
    QEvent,
    QModelIndex,
    QObject,
    QPoint,
    QPropertyAnimation,
    QRectF,
    QSignalBlocker,
    Qt,
    QThread,
    Signal,
    Slot,
    QSize,
    QTimer,
    QCoreApplication,
    QItemSelectionModel,
    QStandardPaths,
    QUrl,
    QUrlQuery,
    QIODevice,
    QLocale,
)
from PySide6.QtGui import (
    QAction,
    QActionGroup,
    QBrush,
    QColor,
    QDoubleValidator,
    QFont,
    QFontDatabase,
    QFontMetrics,
    QDesktopServices,
    QIcon,
    QImage,
    QKeySequence,
    QLinearGradient,
    QPalette,
    QPainter,
    QPen,
    QPixmap,
    QShortcut,
    QTextDocument,
)
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkReply, QNetworkRequest
try:
    from PySide6.QtWebEngineWidgets import QWebEngineView

    _HAS_QTWEBENGINE = True
except Exception:  # pragma: no cover - optional runtime dependency
    QWebEngineView = None  # type: ignore[assignment]
    _HAS_QTWEBENGINE = False
from PySide6.QtWidgets import (
    QAbstractButton,
    QApplication,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDoubleSpinBox,
    QFrame,
    QFormLayout,
    QGridLayout,
    QHeaderView,
    QFileDialog,
    QGraphicsDropShadowEffect,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QInputDialog,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QSplitter,
    QStackedLayout,
    QSpinBox,
    QSlider,
    QStyle,
    QStyledItemDelegate,
    QSizePolicy,
    QTableView,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QToolButton,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
    QColorDialog,
    QDialog,
    QDialogButtonBox,
    QTextEdit,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt import NavigationToolbar2QT as NavigationToolbar
_HAS_MPL_CANVAS = True

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


def _configure_tab_widget(tab_widget: QTabWidget, *, document_mode: bool = True) -> QTabWidget:
    """Normalize tab widgets so native tab-bar chrome doesn't leak through custom themes."""
    tab_widget.setDocumentMode(document_mode)
    try:
        tab_widget.setUsesScrollButtons(True)
    except Exception:
        pass
    try:
        tab_widget.setElideMode(Qt.TextElideMode.ElideNone)
    except Exception:
        pass
    bar = tab_widget.tabBar()
    bar.setExpanding(False)
    bar.setElideMode(Qt.TextElideMode.ElideNone)
    if hasattr(bar, "setUsesScrollButtons"):
        try:
            bar.setUsesScrollButtons(True)
        except Exception:
            pass
    tab_widget.tabBar().setDrawBase(False)
    return tab_widget


_DISPLAY_FONT_PATH = Path(__file__).resolve().parent / "assets" / "fonts" / "Rajdhani-SemiBold.ttf"
_DISPLAY_FONT_LOADED = False
_DISPLAY_FONT_CSS_CACHE: Optional[str] = None
_DISPLAY_WEB_FONT_FAMILY = "Rajdhani Web"
_DISPLAY_FONT_QT_FAMILY: Optional[str] = None
_QT_FALLBACK_FONT_FAMILY: Optional[str] = None
_MPL_AVAILABLE_FONT_NAMES: Optional[set[str]] = None


def _preferred_display_font_family() -> str:
    """Return enforced display font family used across the app and plots."""
    return str(_DISPLAY_FONT_QT_FAMILY or "Rajdhani").strip() or "Rajdhani"


def _split_font_families(value: object) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    families: list[str] = []
    for chunk in text.split(","):
        name = str(chunk or "").strip().strip('"').strip("'").strip()
        if name:
            families.append(name)
    return families


def _platform_ui_font_candidates() -> list[str]:
    if sys.platform == "darwin":
        return [".AppleSystemUIFont", "SF Pro Text", "Avenir Next", "Helvetica Neue", "Arial"]
    if sys.platform.startswith("win"):
        return ["Segoe UI", "Arial"]
    return ["Noto Sans", "DejaVu Sans", "Liberation Sans", "Arial"]


def _resolved_platform_ui_font_family() -> str:
    global _QT_FALLBACK_FONT_FAMILY
    if _QT_FALLBACK_FONT_FAMILY:
        return _QT_FALLBACK_FONT_FAMILY
    for family in _platform_ui_font_candidates():
        if family:
            _QT_FALLBACK_FONT_FAMILY = family
            return family
    if _DISPLAY_FONT_QT_FAMILY:
        _QT_FALLBACK_FONT_FAMILY = _DISPLAY_FONT_QT_FAMILY
        return _QT_FALLBACK_FONT_FAMILY
    _QT_FALLBACK_FONT_FAMILY = "Arial"
    return _QT_FALLBACK_FONT_FAMILY


def _available_mpl_font_names() -> set[str]:
    global _MPL_AVAILABLE_FONT_NAMES
    if _MPL_AVAILABLE_FONT_NAMES is None:
        _MPL_AVAILABLE_FONT_NAMES = {str(font.name) for font in mpl_font_manager.fontManager.ttflist}
    return _MPL_AVAILABLE_FONT_NAMES


def _repolish_widget(widget: Optional[QWidget]) -> None:
    if widget is None:
        return
    style = widget.style()
    if style is None:
        widget.update()
        return
    style.unpolish(widget)
    style.polish(widget)
    widget.update()


def _set_dynamic_property(widget: Optional[QWidget], name: str, value: object) -> None:
    if widget is None:
        return
    widget.setProperty(name, value)


def _embedded_display_font_css() -> str:
    global _DISPLAY_FONT_CSS_CACHE
    if _DISPLAY_FONT_CSS_CACHE is not None:
        return _DISPLAY_FONT_CSS_CACHE
    if not _DISPLAY_FONT_PATH.exists():
        _DISPLAY_FONT_CSS_CACHE = ""
        return _DISPLAY_FONT_CSS_CACHE
    try:
        encoded = base64.b64encode(_DISPLAY_FONT_PATH.read_bytes()).decode("ascii")
    except Exception:
        _DISPLAY_FONT_CSS_CACHE = ""
        return _DISPLAY_FONT_CSS_CACHE
    _DISPLAY_FONT_CSS_CACHE = (
        "@font-face{"
        f"font-family:'{_DISPLAY_WEB_FONT_FAMILY}';"
        f"src:url(data:font/ttf;base64,{encoded}) format('truetype');"
        "font-style:normal;"
        "font-weight:600;"
        "font-display:swap;"
        "}"
    )
    return _DISPLAY_FONT_CSS_CACHE


def _plot_font_css_stack(theme_tokens: Optional[dict[str, str]] = None) -> str:
    # Enforce Rajdhani-first stack for all web/Plotly charts.
    fallback_families: list[str] = [_preferred_display_font_family(), "Rajdhani"]
    if theme_tokens:
        for body_font in _split_font_families(theme_tokens.get("font_family", "")):
            if body_font and body_font not in fallback_families:
                fallback_families.append(body_font)
    fallback_families.extend(_platform_ui_font_candidates())
    fallback_families.extend(["Arial", "Helvetica"])
    deduped: list[str] = []
    for family in fallback_families:
        normalized = str(family or "").strip()
        if normalized and normalized not in deduped and normalized != "Sans Serif":
            deduped.append(normalized)
    fallback = ", ".join(f'"{family}"' for family in deduped) or '"Arial"'
    if _DISPLAY_FONT_PATH.exists():
        return f'"{_DISPLAY_WEB_FONT_FAMILY}", {fallback}'
    return fallback


def _set_button_variant(button: Optional[QWidget], variant: str) -> None:
    _set_dynamic_property(button, "variant", str(variant))


def _set_label_tone(label: Optional[QWidget], tone: str) -> None:
    _set_dynamic_property(label, "tone", str(tone))


def _set_widget_invalid(widget: Optional[QWidget], invalid: bool) -> None:
    _set_dynamic_property(widget, "invalid", bool(invalid))


def _theme_tokens_from_widget(widget: object) -> dict[str, str]:
    current = widget
    for _ in range(6):
        tokens = getattr(current, "_theme_tokens", None)
        if isinstance(tokens, dict):
            return tokens
        parent = getattr(current, "parent", None)
        if not callable(parent):
            break
        current = parent()
        if current is None:
            break
    return {}


def _theme_color_from_widget(widget: object, key: str, fallback: str) -> str:
    tokens = _theme_tokens_from_widget(widget)
    value = tokens.get(key) if isinstance(tokens, dict) else None
    return str(value or fallback)


_CSS_RGBA_RE = re.compile(
    r"^rgba?\(\s*([0-9]{1,3})\s*,\s*([0-9]{1,3})\s*,\s*([0-9]{1,3})(?:\s*,\s*([0-9]*\.?[0-9]+))?\s*\)$",
    re.IGNORECASE,
)


def _parse_qcolor_token(value: object) -> QColor:
    if isinstance(value, QColor):
        color = QColor(value)
        return color if color.isValid() else QColor()

    text = str(value or "").strip()
    if not text:
        return QColor()

    color = QColor(text)
    if color.isValid():
        return color

    match = _CSS_RGBA_RE.match(text)
    if not match:
        return QColor()

    red, green, blue = (max(0, min(255, int(part))) for part in match.groups()[:3])
    alpha_token = match.group(4)
    if alpha_token is None or alpha_token == "":
        alpha = 255
    else:
        alpha_value = float(alpha_token)
        if alpha_value <= 1.0:
            alpha = int(round(max(0.0, min(1.0, alpha_value)) * 255.0))
        else:
            alpha = int(round(max(0.0, min(255.0, alpha_value))))
    return QColor(red, green, blue, alpha)


def _qcolor_from_token(value: object, fallback: object | None = None) -> QColor:
    color = _parse_qcolor_token(value)
    if color.isValid():
        return color
    if fallback is not None:
        fallback_color = _parse_qcolor_token(fallback)
        if fallback_color.isValid():
            return fallback_color
    return QColor()


def _theme_qcolor_from_widget(widget: object, key: str, fallback: str) -> QColor:
    color = _qcolor_from_token(_theme_color_from_widget(widget, key, fallback), fallback)
    if color.isValid():
        return color
    return _qcolor_from_token(fallback)


def _mix_qcolors_for_theme(first: QColor, second: QColor, first_ratio: float) -> QColor:
    ratio = max(0.0, min(1.0, float(first_ratio)))
    other_ratio = 1.0 - ratio
    return QColor(
        int(round(first.red() * ratio + second.red() * other_ratio)),
        int(round(first.green() * ratio + second.green() * other_ratio)),
        int(round(first.blue() * ratio + second.blue() * other_ratio)),
    )


def _qcolor_rgba_css_for_theme(color: QColor, alpha: float) -> str:
    use_color = QColor(color)
    if not use_color.isValid():
        use_color = QColor("#59f3ff")
    return f"rgba({use_color.red()}, {use_color.green()}, {use_color.blue()}, {max(0.0, min(1.0, float(alpha))):.3f})"


def _qcolor_rgba_mpl_for_theme(color: QColor, alpha: float) -> tuple[float, float, float, float]:
    use_color = QColor(color)
    if not use_color.isValid():
        use_color = QColor("#59f3ff")
    return (
        float(use_color.redF()),
        float(use_color.greenF()),
        float(use_color.blueF()),
        max(0.0, min(1.0, float(alpha))),
    )


def _softened_plot_grid_qcolor_from_tokens(theme_tokens: Optional[dict[str, str]]) -> QColor:
    tokens = theme_tokens if isinstance(theme_tokens, dict) else {}
    base = QColor(str(tokens.get("plot_grid", "#2f4666")))
    panel = QColor(str(tokens.get("plot_panel_bg", "#162334")))
    if not base.isValid():
        base = QColor("#2f4666")
    if not panel.isValid():
        panel = QColor("#162334")
    return _mix_qcolors_for_theme(base, panel, 0.34)


def _softened_plot_grid_css_from_tokens(theme_tokens: Optional[dict[str, str]], *, alpha: float = 1.0) -> str:
    return _qcolor_rgba_css_for_theme(_softened_plot_grid_qcolor_from_tokens(theme_tokens), alpha)


def _softened_plot_grid_rgba_from_tokens(theme_tokens: Optional[dict[str, str]], *, alpha: float = 1.0) -> tuple[float, float, float, float]:
    return _qcolor_rgba_mpl_for_theme(_softened_plot_grid_qcolor_from_tokens(theme_tokens), alpha)


def _composite_qcolor(foreground: QColor, background: QColor) -> QColor:
    fg = QColor(foreground)
    bg = QColor(background)
    if not fg.isValid():
        return bg if bg.isValid() else QColor()
    if not bg.isValid() or fg.alpha() >= 255:
        return fg
    alpha = fg.alphaF()
    out = QColor(
        round(fg.red() * alpha + bg.red() * (1.0 - alpha)),
        round(fg.green() * alpha + bg.green() * (1.0 - alpha)),
        round(fg.blue() * alpha + bg.blue() * (1.0 - alpha)),
    )
    return out


def _contrast_text_for_background(background: QColor, *, table_bg: Optional[QColor] = None) -> QColor:
    bg = QColor(background)
    if not bg.isValid():
        return QColor("#f6fbff")
    if table_bg is not None and table_bg.isValid() and bg.alpha() < 255:
        bg = _composite_qcolor(bg, table_bg)
    luminance = (
        0.2126 * float(bg.red()) +
        0.7152 * float(bg.green()) +
        0.0722 * float(bg.blue())
    ) / 255.0
    return QColor("#061118" if luminance > 0.58 else "#f6fbff")


def _icon_pen(color: QColor, width: float) -> QPen:
    pen = QPen(color)
    pen.setWidthF(width)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    return pen


def _button_icon_palette(widget: object) -> tuple[QColor, QColor, QColor]:
    variant = ""
    if hasattr(widget, "property"):
        try:
            variant = str(widget.property("variant") or "")
        except Exception:
            variant = ""
    if variant == "primary":
        fg = _theme_qcolor_from_widget(widget, "btn_text", "#f7fbff")
    elif variant == "ghost":
        fg = _theme_qcolor_from_widget(widget, "strip_label", "#a4b4c8")
    else:
        fg = _theme_qcolor_from_widget(widget, "section_title", "#eef4fc")
    accent = _theme_qcolor_from_widget(widget, "accent_primary", "#59f3ff")
    accent_secondary = _theme_qcolor_from_widget(widget, "accent_secondary", "#ff4da6")
    return fg, accent, accent_secondary


def _build_button_icon_pixmap(kind: str, fg: QColor, accent: QColor, accent_secondary: QColor, size: int = 18) -> QPixmap:
    px = QPixmap(size, size)
    px.fill(Qt.GlobalColor.transparent)
    painter = QPainter(px)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    stroke = max(1.5, float(size) * 0.11)
    painter.setPen(_icon_pen(fg, stroke))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    kind = str(kind or "").strip().lower()
    s = float(size)
    m = s * 0.2

    def line(x1: float, y1: float, x2: float, y2: float) -> None:
        painter.drawLine(int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))

    def ellipse(x: float, y: float, w: float, h: float) -> None:
        painter.drawEllipse(int(round(x)), int(round(y)), int(round(w)), int(round(h)))

    def arc(x: float, y: float, w: float, h: float, start_deg: float, span_deg: float) -> None:
        painter.drawArc(
            int(round(x)),
            int(round(y)),
            int(round(w)),
            int(round(h)),
            int(round(start_deg * 16)),
            int(round(span_deg * 16)),
        )

    def dot(x: float, y: float, r: float = 1.7) -> None:
        painter.save()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(accent_secondary)
        painter.drawEllipse(int(round(x - r)), int(round(y - r)), int(round(r * 2)), int(round(r * 2)))
        painter.restore()

    if kind in {"add", "new"}:
        line(s * 0.5, m, s * 0.5, s - m)
        line(m, s * 0.5, s - m, s * 0.5)
        dot(s * 0.77, s * 0.23)
    elif kind in {"quick", "flash"}:
        line(s * 0.56, m, s * 0.33, s * 0.48)
        line(s * 0.33, s * 0.48, s * 0.50, s * 0.48)
        line(s * 0.50, s * 0.48, s * 0.42, s - m)
        line(s * 0.42, s - m, s * 0.68, s * 0.54)
        line(s * 0.68, s * 0.54, s * 0.52, s * 0.54)
    elif kind in {"suggest", "spark", "ai"}:
        line(s * 0.5, m, s * 0.5, s - m)
        line(m, s * 0.5, s - m, s * 0.5)
        line(s * 0.26, s * 0.26, s * 0.74, s * 0.74)
        line(s * 0.74, s * 0.26, s * 0.26, s * 0.74)
        dot(s * 0.76, s * 0.24)
    elif kind in {"toggle", "observed", "ok", "apply"}:
        line(s * 0.23, s * 0.56, s * 0.42, s * 0.74)
        line(s * 0.42, s * 0.74, s * 0.78, s * 0.30)
        dot(s * 0.76, s * 0.24)
    elif kind in {"cancel", "clear"}:
        line(m, m, s - m, s - m)
        line(s - m, m, m, s - m)
    elif kind in {"remove", "delete"}:
        line(m, s * 0.5, s - m, s * 0.5)
    elif kind in {"edit", "pencil"}:
        line(s * 0.28, s * 0.72, s * 0.72, s * 0.28)
        line(s * 0.62, s * 0.24, s * 0.76, s * 0.38)
        line(s * 0.24, s * 0.76, s * 0.38, s * 0.62)
        dot(s * 0.78, s * 0.22)
    elif kind in {"load", "open"}:
        line(m, s * 0.72, s - m, s * 0.72)
        line(m, s * 0.72, m, s * 0.58)
        line(s - m, s * 0.72, s - m, s * 0.58)
        line(s * 0.5, m, s * 0.5, s * 0.58)
        line(s * 0.5, s * 0.58, s * 0.34, s * 0.42)
        line(s * 0.5, s * 0.58, s * 0.66, s * 0.42)
    elif kind in {"save"}:
        painter.drawRoundedRect(int(round(m)), int(round(m)), int(round(s - 2 * m)), int(round(s - 2 * m)), 3, 3)
        line(s * 0.32, m + 1, s * 0.32, s * 0.45)
        line(s * 0.32, s * 0.45, s * 0.68, s * 0.45)
        line(s * 0.28, s * 0.67, s * 0.72, s * 0.67)
    elif kind in {"weather", "cloud"}:
        ellipse(s * 0.22, s * 0.42, s * 0.28, s * 0.22)
        ellipse(s * 0.40, s * 0.32, s * 0.30, s * 0.26)
        ellipse(s * 0.56, s * 0.44, s * 0.22, s * 0.18)
        line(s * 0.24, s * 0.59, s * 0.72, s * 0.59)
        dot(s * 0.76, s * 0.28)
    elif kind in {"lookup", "resolve"}:
        ellipse(s * 0.22, s * 0.22, s * 0.36, s * 0.36)
        line(s * 0.52, s * 0.52, s * 0.74, s * 0.74)
        line(s * 0.40, s * 0.28, s * 0.40, s * 0.52)
        line(s * 0.28, s * 0.40, s * 0.52, s * 0.40)
    elif kind in {"send"}:
        line(m, s * 0.54, s - m, m)
        line(s - m, m, s * 0.62, s - m)
        line(s * 0.62, s - m, s * 0.50, s * 0.56)
        line(s * 0.50, s * 0.56, m, s * 0.54)
    elif kind in {"seestar", "target"}:
        ellipse(s * 0.22, s * 0.22, s * 0.56, s * 0.56)
        line(s * 0.5, m, s * 0.5, s * 0.34)
        line(s * 0.5, s * 0.66, s * 0.5, s - m)
        line(m, s * 0.5, s * 0.34, s * 0.5)
        line(s * 0.66, s * 0.5, s - m, s * 0.5)
        dot(s * 0.5, s * 0.5, 1.8)
    elif kind in {"refresh"}:
        painter.drawArc(int(round(m)), int(round(m)), int(round(s - 2 * m)), int(round(s - 2 * m)), int(25 * 16), int(280 * 16))
        line(s * 0.72, s * 0.18, s * 0.82, s * 0.18)
        line(s * 0.82, s * 0.18, s * 0.82, s * 0.30)
    elif kind in {"clock", "localtime"}:
        ellipse(s * 0.18, s * 0.18, s * 0.64, s * 0.64)
        line(s * 0.50, s * 0.50, s * 0.50, s * 0.30)
        line(s * 0.50, s * 0.50, s * 0.66, s * 0.58)
        dot(s * 0.74, s * 0.24)
    elif kind in {"utc", "globe"}:
        ellipse(s * 0.18, s * 0.18, s * 0.64, s * 0.64)
        arc(s * 0.18, s * 0.18, s * 0.64, s * 0.64, 0, 180)
        arc(s * 0.18, s * 0.18, s * 0.64, s * 0.64, 180, 180)
        line(s * 0.50, s * 0.20, s * 0.50, s * 0.80)
        arc(s * 0.30, s * 0.18, s * 0.40, s * 0.64, 90, 180)
        arc(s * 0.30, s * 0.18, s * 0.40, s * 0.64, 270, 180)
    elif kind in {"sidereal", "star"}:
        line(s * 0.50, m, s * 0.50, s - m)
        line(m, s * 0.50, s - m, s * 0.50)
        line(s * 0.26, s * 0.26, s * 0.74, s * 0.74)
        line(s * 0.74, s * 0.26, s * 0.26, s * 0.74)
        dot(s * 0.78, s * 0.24)
    elif kind in {"sun", "sun_alt"}:
        ellipse(s * 0.31, s * 0.31, s * 0.38, s * 0.38)
        line(s * 0.50, m, s * 0.50, s * 0.20)
        line(s * 0.50, s * 0.80, s * 0.50, s - m)
        line(m, s * 0.50, s * 0.20, s * 0.50)
        line(s * 0.80, s * 0.50, s - m, s * 0.50)
        line(s * 0.25, s * 0.25, s * 0.34, s * 0.34)
        line(s * 0.66, s * 0.66, s * 0.75, s * 0.75)
        line(s * 0.66, s * 0.34, s * 0.75, s * 0.25)
        line(s * 0.25, s * 0.75, s * 0.34, s * 0.66)
    elif kind in {"moon", "moon_alt"}:
        arc(s * 0.18, s * 0.16, s * 0.52, s * 0.68, 58, 244)
        arc(s * 0.34, s * 0.16, s * 0.36, s * 0.68, 110, 140)
        dot(s * 0.78, s * 0.28)
    elif kind in {"sunrise"}:
        line(m, s * 0.70, s - m, s * 0.70)
        arc(s * 0.28, s * 0.40, s * 0.44, s * 0.44, 0, 180)
        line(s * 0.50, s * 0.18, s * 0.50, s * 0.34)
        line(s * 0.34, s * 0.30, s * 0.40, s * 0.38)
        line(s * 0.66, s * 0.30, s * 0.60, s * 0.38)
    elif kind in {"sunset"}:
        line(m, s * 0.70, s - m, s * 0.70)
        arc(s * 0.28, s * 0.48, s * 0.44, s * 0.40, 180, 180)
        line(s * 0.50, s * 0.22, s * 0.50, s * 0.38)
        line(s * 0.50, s * 0.38, s * 0.42, s * 0.30)
        line(s * 0.50, s * 0.38, s * 0.58, s * 0.30)
    elif kind in {"moonphase", "phase"}:
        ellipse(s * 0.22, s * 0.18, s * 0.56, s * 0.56)
        arc(s * 0.34, s * 0.18, s * 0.34, s * 0.56, 90, 180)
        dot(s * 0.76, s * 0.26)
    elif kind in {"moonrise"}:
        arc(s * 0.18, s * 0.18, s * 0.52, s * 0.64, 58, 244)
        arc(s * 0.34, s * 0.18, s * 0.36, s * 0.64, 110, 140)
        line(s * 0.72, s * 0.74, s * 0.72, s * 0.36)
        line(s * 0.72, s * 0.36, s * 0.64, s * 0.44)
        line(s * 0.72, s * 0.36, s * 0.80, s * 0.44)
    elif kind in {"moonset"}:
        arc(s * 0.18, s * 0.18, s * 0.52, s * 0.64, 58, 244)
        arc(s * 0.34, s * 0.18, s * 0.36, s * 0.64, 110, 140)
        line(s * 0.72, s * 0.30, s * 0.72, s * 0.68)
        line(s * 0.72, s * 0.68, s * 0.64, s * 0.60)
        line(s * 0.72, s * 0.68, s * 0.80, s * 0.60)
    elif kind in {"window", "best_window"}:
        line(s * 0.24, s * 0.28, s * 0.24, s * 0.72)
        line(s * 0.76, s * 0.28, s * 0.76, s * 0.72)
        line(s * 0.24, s * 0.28, s * 0.76, s * 0.28)
        line(s * 0.24, s * 0.72, s * 0.76, s * 0.72)
        line(s * 0.50, s * 0.34, s * 0.50, s * 0.50)
        line(s * 0.50, s * 0.50, s * 0.62, s * 0.58)
    elif kind in {"notes", "note"}:
        painter.drawRoundedRect(int(round(s * 0.24)), int(round(s * 0.18)), int(round(s * 0.50)), int(round(s * 0.62)), 3, 3)
        line(s * 0.34, s * 0.36, s * 0.64, s * 0.36)
        line(s * 0.34, s * 0.50, s * 0.64, s * 0.50)
        line(s * 0.34, s * 0.64, s * 0.56, s * 0.64)
    elif kind in {"link", "open-link"}:
        line(s * 0.28, s * 0.72, s * 0.72, s * 0.28)
        line(s * 0.50, s * 0.28, s * 0.72, s * 0.28)
        line(s * 0.72, s * 0.28, s * 0.72, s * 0.50)
        line(s * 0.30, s * 0.34, s * 0.30, s * 0.70)
        line(s * 0.30, s * 0.70, s * 0.66, s * 0.70)
    elif kind in {"describe"}:
        painter.drawRoundedRect(int(round(m)), int(round(s * 0.24)), int(round(s - 2 * m)), int(round(s * 0.40)), 4, 4)
        line(s * 0.34, s * 0.66, s * 0.46, s * 0.56)
        line(s * 0.46, s * 0.56, s * 0.56, s * 0.66)
        dot(s * 0.76, s * 0.30)
    else:
        ellipse(m, m, s - 2 * m, s - 2 * m)
        dot(s * 0.72, s * 0.28)

    painter.end()
    return px


def _build_button_icon(button: object, kind: str, size: int = 18) -> QIcon:
    fg, accent, accent_secondary = _button_icon_palette(button)
    disabled = _theme_qcolor_from_widget(button, "state_disabled", "#7f8ca3")
    icon = QIcon()
    icon.addPixmap(_build_button_icon_pixmap(kind, fg, accent, accent_secondary, size), QIcon.Mode.Normal, QIcon.State.Off)
    icon.addPixmap(_build_button_icon_pixmap(kind, fg, accent_secondary, accent, size), QIcon.Mode.Active, QIcon.State.Off)
    icon.addPixmap(_build_button_icon_pixmap(kind, disabled, disabled, disabled, size), QIcon.Mode.Disabled, QIcon.State.Off)
    return icon


def _set_button_icon_kind(button: Optional[QWidget], kind: str, size: int = 16) -> None:
    if button is None or not hasattr(button, "setIcon"):
        return
    button.setProperty("icon_kind", str(kind))
    button.setProperty("icon_size", int(size))
    button.setIcon(_build_button_icon(button, kind, size))
    if hasattr(button, "setIconSize"):
        button.setIconSize(QSize(int(size), int(size)))


def _refresh_button_icons(root: Optional[QWidget]) -> None:
    if root is None:
        return
    widgets = [root, *root.findChildren(QWidget)]
    for widget in widgets:
        if isinstance(widget, QLabel):
            kind = widget.property("detail_icon_kind")
            if kind:
                size = int(widget.property("detail_icon_size") or 16)
                fg = _theme_qcolor_from_widget(widget, "section_title", "#eef4fc")
                accent = _theme_qcolor_from_widget(widget, "accent_primary", "#59f3ff")
                accent_secondary = _theme_qcolor_from_widget(widget, "accent_secondary", "#ff4da6")
                widget.setPixmap(_build_button_icon_pixmap(str(kind), fg, accent, accent_secondary, size))
                widget.setFixedSize(size, size)
                widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
                continue
        if not hasattr(widget, "property") or not hasattr(widget, "setIcon"):
            continue
        kind = widget.property("icon_kind")
        if not kind:
            continue
        size = int(widget.property("icon_size") or 16)
        widget.setIcon(_build_button_icon(widget, str(kind), size))
        if hasattr(widget, "setIconSize"):
            widget.setIconSize(QSize(size, size))
    _refresh_button_hover_glow(root)


_BUTTON_HOVER_GLOW_FILTER: Optional[QObject] = None


class _ButtonHoverGlowFilter(QObject):
    def eventFilter(self, watched, event):  # noqa: D401
        if not isinstance(watched, QWidget):
            return False
        event_type = event.type()
        if event_type in (QEvent.Type.Enter, QEvent.Type.HoverEnter):
            _apply_button_hover_glow(watched, True)
        elif event_type in (
            QEvent.Type.Leave,
            QEvent.Type.HoverLeave,
            QEvent.Type.Hide,
            QEvent.Type.EnabledChange,
            QEvent.Type.PaletteChange,
            QEvent.Type.StyleChange,
        ):
            _apply_button_hover_glow(watched, watched.isEnabled() and watched.underMouse())
        return False


def _apply_button_hover_glow(widget: QWidget, enabled: bool) -> None:
    if not isinstance(widget, QPushButton):
        return
    if bool(widget.property("weather_link")):
        enabled = False
    current = widget.graphicsEffect()
    if not enabled or not widget.isEnabled():
        if isinstance(current, QGraphicsDropShadowEffect):
            widget.setGraphicsEffect(None)
        return
    color = _theme_qcolor_from_widget(widget, "button_hover_glow", "#ff7ecf")
    aura = _theme_qcolor_from_widget(widget, "button_hover_aura", "#ff7ecf")
    effect: QGraphicsDropShadowEffect
    if isinstance(current, QGraphicsDropShadowEffect):
        effect = current
    else:
        effect = QGraphicsDropShadowEffect(widget)
        effect.setOffset(0.0, 0.0)
        widget.setGraphicsEffect(effect)
    glow_color = QColor(aura if aura.isValid() else color)
    if not glow_color.isValid():
        glow_color = QColor("#ff7ecf")
    variant = str(widget.property("variant") or "")
    blur_radius = 36.0 if variant in {"primary", "secondary"} else 32.0
    alpha_floor = 210 if variant in {"primary", "secondary"} else 188
    if glow_color.alpha() < alpha_floor:
        glow_color.setAlpha(alpha_floor)
    glow_color = QColor(
        max(glow_color.red(), color.red()),
        max(glow_color.green(), color.green()),
        max(glow_color.blue(), color.blue()),
        glow_color.alpha(),
    )
    effect.setBlurRadius(blur_radius)
    effect.setColor(glow_color)


def _refresh_button_hover_glow(root: Optional[QWidget]) -> None:
    global _BUTTON_HOVER_GLOW_FILTER
    if root is None:
        return
    if _BUTTON_HOVER_GLOW_FILTER is None:
        _BUTTON_HOVER_GLOW_FILTER = _ButtonHoverGlowFilter(QApplication.instance())
    widgets = [root, *root.findChildren(QWidget)]
    for widget in widgets:
        if not isinstance(widget, QPushButton):
            continue
        if not bool(widget.property("_hover_glow_installed")):
            widget.installEventFilter(_BUTTON_HOVER_GLOW_FILTER)
            widget.setProperty("_hover_glow_installed", True)
        _apply_button_hover_glow(widget, widget.isEnabled() and widget.underMouse())


def _set_detail_icon_kind(label: Optional[QLabel], kind: str, size: int = 16) -> None:
    if label is None:
        return
    label.setProperty("detail_icon_kind", str(kind))
    label.setProperty("detail_icon_size", int(size))
    fg = _theme_qcolor_from_widget(label, "section_title", "#eef4fc")
    accent = _theme_qcolor_from_widget(label, "accent_primary", "#59f3ff")
    accent_secondary = _theme_qcolor_from_widget(label, "accent_secondary", "#ff4da6")
    label.setPixmap(_build_button_icon_pixmap(kind, fg, accent, accent_secondary, size))
    label.setFixedSize(size, size)
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)


_LEGACY_TABLE_HIGHLIGHT_DEFAULTS = {
    "below": "#ffe0ea",
    "limit": "#fff2cf",
    "above": "#d9ffe9",
}
_LEGACY_TABLE_HIGHLIGHT_COLORBLIND = {
    "below": "#ffb18e",
    "limit": "#ffe680",
    "above": "#9bf985",
}


def _canonical_color_token(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    color = _qcolor_from_token(text)
    if color.isValid():
        return color.name(QColor.NameFormat.HexArgb).lower()
    return re.sub(r"\s+", "", text.lower())


def _resolve_table_highlight_color(settings: object, key: str, default_color: str) -> str:
    if settings is None or not hasattr(settings, "value"):
        return str(default_color)
    raw = str(settings.value(f"table/color/{key}", default_color) or "").strip()
    if not raw:
        return str(default_color)
    normalized = _canonical_color_token(raw)
    auto_colors = {
        _canonical_color_token(_LEGACY_TABLE_HIGHLIGHT_DEFAULTS.get(key, "")),
        _canonical_color_token(_LEGACY_TABLE_HIGHLIGHT_COLORBLIND.get(key, "")),
    }
    for theme_key, _label in THEME_CHOICES:
        for dark_enabled in (False, True):
            palette = highlight_palette_for_theme(theme_key, dark_enabled=dark_enabled, color_blind=False)
            auto_colors.update({
                _canonical_color_token(palette.below),
                _canonical_color_token(palette.limit),
                _canonical_color_token(palette.above),
            })
    colorblind_palette = highlight_palette_for_theme(DEFAULT_UI_THEME, dark_enabled=False, color_blind=True)
    auto_colors.update({
        _canonical_color_token(colorblind_palette.below),
        _canonical_color_token(colorblind_palette.limit),
        _canonical_color_token(colorblind_palette.above),
    })
    if normalized in auto_colors:
        return str(default_color)
    return raw


def _format_duration_short(total_seconds: float) -> str:
    seconds = max(0, int(round(float(total_seconds))))
    if seconds % 3600 == 0:
        hours = max(1, seconds // 3600)
        return f"{hours}h"
    if seconds % 60 == 0:
        minutes = max(1, seconds // 60)
        return f"{minutes} min"
    return f"{seconds}s"


def _bhtom_suggestion_source_message(source: str) -> str:
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


def _site_runtime_signature(site: Optional["Site"]) -> Optional[tuple[str, float, float, float, str]]:
    if not isinstance(site, Site):
        return None
    return (
        str(site.name or "").strip(),
        round(float(site.latitude), 6),
        round(float(site.longitude), 6),
        round(float(site.elevation), 2),
        str(site.timezone_name or "").strip(),
    )


def _same_runtime_site(first: Optional["Site"], second: Optional["Site"]) -> bool:
    first_sig = _site_runtime_signature(first)
    second_sig = _site_runtime_signature(second)
    return first_sig is not None and first_sig == second_sig


def _should_apply_observatory_change(
    current_name: str,
    current_site: Optional["Site"],
    next_name: str,
    next_site: Optional["Site"],
) -> bool:
    normalized_next = str(next_name or "").strip()
    if not normalized_next or not isinstance(next_site, Site):
        return False
    if str(current_name or "").strip() != normalized_next:
        return True
    return not _same_runtime_site(current_site, next_site)


def _style_dialog_button_box(
    button_box: Optional[QDialogButtonBox],
    *,
    ok: str = "secondary",
    cancel: str = "ghost",
    apply: str = "secondary",
    close: str = "secondary",
) -> None:
    if button_box is None:
        return
    mapping = {
        QDialogButtonBox.Ok: ok,
        QDialogButtonBox.Cancel: cancel,
        QDialogButtonBox.Apply: apply,
        QDialogButtonBox.Close: close,
    }
    icon_mapping = {
        QDialogButtonBox.Ok: "ok",
        QDialogButtonBox.Cancel: "cancel",
        QDialogButtonBox.Apply: "apply",
        QDialogButtonBox.Close: "cancel",
    }
    for role, variant in mapping.items():
        btn = button_box.button(role)
        if btn is not None:
            _set_button_variant(btn, variant)
            _set_button_icon_kind(btn, icon_mapping.get(role, "spark"), 14)


class NeonToggleSwitch(QAbstractButton):
    """Compact animated neon switch used for the Altitude/Airmass toggle."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("NeonToggleSwitch")
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self._offset = 0.0
        self._switch_anim = QPropertyAnimation(self, b"offset", self)
        self._switch_anim.setDuration(170)
        self._switch_anim.setEasingCurve(QEasingCurve.OutCubic)
        self.toggled.connect(self._animate_to_checked_state)
        self.setFixedSize(70, 34)

    def sizeHint(self) -> QSize:  # noqa: D401
        return QSize(70, 34)

    def minimumSizeHint(self) -> QSize:  # noqa: D401
        return self.sizeHint()

    def _get_offset(self) -> float:
        return float(self._offset)

    def _set_offset(self, value: float) -> None:
        self._offset = max(0.0, min(1.0, float(value)))
        self.update()

    offset = Property(float, _get_offset, _set_offset)

    def _animate_to_checked_state(self, checked: bool) -> None:
        try:
            self._switch_anim.stop()
        except Exception:
            pass
        self._switch_anim.setStartValue(self._offset)
        self._switch_anim.setEndValue(1.0 if checked else 0.0)
        self._switch_anim.start()

    def setChecked(self, checked: bool) -> None:  # type: ignore[override]
        changed = bool(checked) != self.isChecked()
        super().setChecked(bool(checked))
        if not changed:
            self._set_offset(1.0 if self.isChecked() else 0.0)

    def mouseReleaseEvent(self, event) -> None:  # noqa: D401
        if event.button() == Qt.LeftButton and self.rect().contains(event.position().toPoint()):
            self.toggle()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:  # noqa: D401
        if event.key() in (Qt.Key_Space, Qt.Key_Return, Qt.Key_Enter):
            self.toggle()
            event.accept()
            return
        super().keyPressEvent(event)

    def paintEvent(self, event) -> None:  # noqa: D401
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        track_rect = QRectF(1.0, 3.0, float(self.width() - 2), float(self.height() - 6))
        radius = track_rect.height() / 2.0
        panel_color = _theme_qcolor_from_widget(self, "plot_panel_bg", "#162334")
        primary_color = _theme_qcolor_from_widget(self, "accent_primary", "#18d8f2")
        secondary_color = _theme_qcolor_from_widget(self, "accent_secondary", "#ff4fd8")
        state_color = QColor(secondary_color if self.isChecked() else primary_color)
        border_color = _theme_qcolor_from_widget(self, "panel_border", "#20506a")
        knob_color = _theme_qcolor_from_widget(self, "btn_text", "#f6fbff")
        if self.isChecked():
            off_color = QColor(panel_color)
            mix_ratio = 0.26 + max(0.0, min(1.0, self._offset * 0.72))
        else:
            off_color = QColor(panel_color)
            mix_ratio = 0.74
        groove_mix = QColor(
            round(off_color.red() + (state_color.red() - off_color.red()) * mix_ratio),
            round(off_color.green() + (state_color.green() - off_color.green()) * mix_ratio),
            round(off_color.blue() + (state_color.blue() - off_color.blue()) * mix_ratio),
            round(off_color.alpha() + (state_color.alpha() - off_color.alpha()) * mix_ratio),
        )

        if self.isEnabled():
            glow_color = QColor(state_color)
            glow_color.setAlpha(int(76 + (72 * self._offset if self.isChecked() else 28)))
            painter.setPen(QPen(glow_color, 3.0))
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(track_rect.adjusted(0.5, 0.5, -0.5, -0.5), radius, radius)

        painter.setPen(QPen(border_color, 1.2))
        painter.setBrush(groove_mix)
        painter.drawRoundedRect(track_rect, radius, radius)

        highlight_rect = QRectF(track_rect.left() + 2.0, track_rect.top() + 2.0, track_rect.width() * 0.42, max(2.0, track_rect.height() * 0.28))
        highlight = QColor(state_color)
        highlight.setAlpha(46 if self.isEnabled() else 18)
        painter.setPen(Qt.NoPen)
        painter.setBrush(highlight)
        painter.drawRoundedRect(highlight_rect, highlight_rect.height() / 2.0, highlight_rect.height() / 2.0)

        scan_color = QColor(knob_color)
        scan_color.setAlpha(18 if self.isEnabled() else 8)
        painter.setPen(QPen(scan_color, 1.0))
        for idx in range(3):
            y = track_rect.top() + 5.0 + idx * 5.0
            painter.drawLine(track_rect.left() + 8.0, y, track_rect.right() - 8.0, y)

        handle_d = track_rect.height() - 4.0
        min_x = track_rect.left() + 2.0
        max_x = track_rect.right() - handle_d - 2.0
        handle_x = min_x + (max_x - min_x) * self._offset
        handle_rect = QRectF(handle_x, track_rect.top() + 2.0, handle_d, handle_d)

        handle_glow = QColor(state_color if self.isEnabled() else border_color)
        handle_glow.setAlpha(96 if self.isEnabled() else 28)
        painter.setPen(QPen(handle_glow, 2.0))
        painter.setBrush(knob_color)
        painter.drawEllipse(handle_rect)

        inner_dot = QColor(state_color if self.isChecked() else border_color)
        inner_dot.setAlpha(200 if self.isEnabled() else 90)
        painter.setPen(Qt.NoPen)
        painter.setBrush(inner_dot)
        painter.drawEllipse(handle_rect.center(), 2.1, 2.1)


class CoverImageLabel(QLabel):
    """Render pixmap preserving aspect ratio, centered in available space."""

    resized = Signal(int, int)
    zoomOutLimitReached = Signal()

    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self._source_pixmap = QPixmap()
        self._overlay_painter: Optional[Callable[..., None]] = None
        self._zoom_enabled = False
        self._zoom_factor = 1.0
        self._zoom_min = 1.0
        self._zoom_max = 18.0
        self._pan_offset = QPoint(0, 0)
        self._drag_active = False
        self._drag_start = QPoint(0, 0)
        self._drag_origin = QPoint(0, 0)

    def setPixmap(self, pixmap):  # type: ignore[override]
        if pixmap is None or pixmap.isNull():
            self._source_pixmap = QPixmap()
            self._drag_active = False
            super().setPixmap(QPixmap())
            self.reset_zoom()
            return
        self._source_pixmap = pixmap.copy()
        self._apply_cover_pixmap()

    def setText(self, text: str):  # type: ignore[override]
        self._source_pixmap = QPixmap()
        self._drag_active = False
        super().setText(text)
        self.reset_zoom()

    def resizeEvent(self, event):  # noqa: D401
        super().resizeEvent(event)
        if not self._source_pixmap.isNull():
            self._apply_cover_pixmap()
        self.resized.emit(self.width(), self.height())

    def wheelEvent(self, event):  # noqa: D401
        if not self._zoom_enabled or self._source_pixmap.isNull():
            super().wheelEvent(event)
            return
        delta = event.angleDelta().y()
        if delta == 0:
            return
        if delta < 0 and self._zoom_factor <= (self._zoom_min + 1e-3):
            self.zoomOutLimitReached.emit()
            event.accept()
            return
        step = float(delta) / 120.0
        factor = 1.17 ** step
        self.set_zoom(self._zoom_factor * factor)
        event.accept()

    def mousePressEvent(self, event):  # noqa: D401
        if (
            self._zoom_enabled
            and not self._source_pixmap.isNull()
            and event.button() == Qt.MouseButton.LeftButton
            and self._can_pan_current_view()
        ):
            self._drag_active = True
            self._drag_start = event.position().toPoint()
            self._drag_origin = QPoint(self._pan_offset)
            self._update_cursor()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):  # noqa: D401
        if self._drag_active and self._zoom_enabled:
            delta = event.position().toPoint() - self._drag_start
            self._pan_offset = self._drag_origin + delta
            self._apply_cover_pixmap()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):  # noqa: D401
        if self._drag_active and event.button() == Qt.MouseButton.LeftButton:
            self._drag_active = False
            self._update_cursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):  # noqa: D401
        if self._zoom_enabled and event.button() == Qt.MouseButton.LeftButton:
            self.reset_zoom()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def set_zoom_enabled(self, enabled: bool) -> None:
        self._zoom_enabled = bool(enabled)
        self._update_cursor()

    def set_overlay_painter(self, callback: Optional[Callable[..., None]]) -> None:
        self._overlay_painter = callback
        if not self._source_pixmap.isNull():
            self._apply_cover_pixmap()

    def set_zoom(self, factor: float) -> None:
        try:
            value = float(factor)
        except (TypeError, ValueError):
            value = 1.0
        value = max(self._zoom_min, min(self._zoom_max, value))
        if abs(value - self._zoom_factor) < 1e-4:
            return
        self._zoom_factor = value
        if abs(self._zoom_factor - 1.0) <= 1e-3:
            self._pan_offset = QPoint(0, 0)
        self._apply_cover_pixmap()
        self._update_cursor()

    def zoom_factor(self) -> float:
        return float(self._zoom_factor)

    def reset_zoom(self) -> None:
        self._zoom_factor = 1.0
        self._pan_offset = QPoint(0, 0)
        if not self._source_pixmap.isNull():
            self._apply_cover_pixmap()
        self._update_cursor()

    def _update_cursor(self) -> None:
        if not self._zoom_enabled or self._source_pixmap.isNull() or not self._can_pan_current_view():
            self.setCursor(Qt.CursorShape.ArrowCursor)
            return
        self.setCursor(Qt.CursorShape.ClosedHandCursor if self._drag_active else Qt.CursorShape.OpenHandCursor)

    def _can_pan_current_view(self) -> bool:
        if not self._zoom_enabled or self._source_pixmap.isNull():
            return False
        w = max(1, self.width())
        h = max(1, self.height())
        scale_target_w = max(1, int(round(w * self._zoom_factor)))
        scale_target_h = max(1, int(round(h * self._zoom_factor)))
        scaled = self._source_pixmap.scaled(
            scale_target_w,
            scale_target_h,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        return abs(int(scaled.width()) - int(w)) > 1 or abs(int(scaled.height()) - int(h)) > 1

    def _apply_cover_pixmap(self):
        if self._source_pixmap.isNull():
            super().setPixmap(QPixmap())
            self._update_cursor()
            return
        w = max(1, self.width())
        h = max(1, self.height())
        scale_target_w = max(1, int(round(w * self._zoom_factor)))
        scale_target_h = max(1, int(round(h * self._zoom_factor)))
        scaled = self._source_pixmap.scaled(
            scale_target_w,
            scale_target_h,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        max_dx = max(0, abs(scaled.width() - w) // 2)
        max_dy = max(0, abs(scaled.height() - h) // 2)
        self._pan_offset.setX(max(-max_dx, min(max_dx, self._pan_offset.x())))
        self._pan_offset.setY(max(-max_dy, min(max_dy, self._pan_offset.y())))
        canvas = QPixmap(w, h)
        canvas.fill(Qt.GlobalColor.transparent)
        painter = QPainter(canvas)
        x = ((w - scaled.width()) // 2) + self._pan_offset.x()
        y = ((h - scaled.height()) // 2) + self._pan_offset.y()
        painter.drawPixmap(x, y, scaled)
        if self._overlay_painter is not None:
            try:
                # Preferred signature:
                # overlay(painter, widget_w, widget_h, image_x, image_y, image_w, image_h)
                self._overlay_painter(painter, w, h, x, y, scaled.width(), scaled.height())
            except TypeError:
                try:
                    self._overlay_painter(painter, w, h)
                except Exception:
                    pass
            except Exception:
                pass
        painter.end()
        super().setPixmap(canvas)
        self._update_cursor()


def _normalized_css_color(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    color = QColor(text)
    if not color.isValid():
        return ""
    return color.name().lower()


def _swatch_text_color(value: object, *, dark: str = "#f6fbff", light: str = "#101720") -> str:
    color = _qcolor_from_token(value)
    if not color.isValid():
        return dark
    return dark if color.lightnessF() < 0.60 else light


_VALID_TABLE_COLOR_MODES = {"background", "text_glow"}
UI_FONT_SIZE_MIN = 9
UI_FONT_SIZE_MAX = 24


def _normalize_table_color_mode(value: object, *, default: str = "background") -> str:
    mode = str(value or "").strip().lower()
    if mode in _VALID_TABLE_COLOR_MODES:
        return mode
    return default


def _sanitize_ui_font_size(value: object, *, default: int = 11) -> int:
    try:
        size = int(value)
    except (TypeError, ValueError):
        size = int(default)
    return max(UI_FONT_SIZE_MIN, min(UI_FONT_SIZE_MAX, size))


class NoSelectBackgroundDelegate(QStyledItemDelegate):
    """Delegate that preserves model background for the target-name column when selected."""
    def paint(self, painter, option, index):
        # Disable the selected state to preserve background
        if option.state & QStyle.State_Selected:
            option.state &= ~QStyle.State_Selected
        super().paint(painter, option, index)


# --- Table Settings Dialog ---




# Number of time samples for visibility curve (lower = faster)
TIME_SAMPLES = 240 


# --------------------------------------------------
# --- Main Window ----------------------------------
# --------------------------------------------------
class MainWindow(QMainWindow):
    darkModeChanged = Signal(bool)
    bhtom_observatory_presets_changed = Signal(list, str)
    bhtom_observatory_presets_loading = Signal(bool, str)

    def eventFilter(self, watched, event):  # noqa: D401
        if hasattr(self, "ai_output") and watched is self.ai_output:
            if event.type() == QEvent.Type.Resize:
                new_width = self._ai_message_layout_width()
                last_width = int(getattr(self, "_ai_output_last_viewport_width", 0))
                if abs(new_width - last_width) >= 8:
                    self._ai_output_last_viewport_width = new_width
                    if getattr(self, "_ai_messages", None):
                        QTimer.singleShot(0, self._render_ai_messages)
        return super().eventFilter(watched, event)

    def _update_night_details_constraints(self):
        if not hasattr(self, "info_widget") or not hasattr(self, "info_card"):
            return
        self.info_widget.adjustSize()
        content_hint = self.info_widget.minimumSizeHint()
        content_h = max(self.info_widget.sizeHint().height(), content_hint.height())
        title_h = self.info_title_label.sizeHint().height() if hasattr(self, "info_title_label") else 20
        card_min_h = max(220, content_h + title_h + 30)
        card_min_w = max(330, int(getattr(self, "_night_details_fixed_min_width", 330)))
        self.info_card.setMinimumHeight(card_min_h)
        self.info_card.setMinimumWidth(card_min_w)
        if hasattr(self, "info_scroll"):
            self.info_scroll.setMinimumHeight(content_h + 6)
            self.info_scroll.setMinimumWidth(max(260, card_min_w - 18))
        if hasattr(self, "right_dashboard"):
            dashboard_base_min = max(460, int(getattr(self, "_right_dashboard_base_min_width", 460)))
            self.right_dashboard.setMinimumWidth(max(dashboard_base_min, card_min_w + 22))

    def _ensure_display_font_loaded(self) -> None:
        global _DISPLAY_FONT_LOADED, _DISPLAY_FONT_QT_FAMILY
        if _DISPLAY_FONT_LOADED:
            return
        if not _DISPLAY_FONT_PATH.exists():
            return
        try:
            mpl_font_manager.fontManager.addfont(str(_DISPLAY_FONT_PATH))
        except Exception:
            pass
        try:
            font_id = QFontDatabase.addApplicationFont(str(_DISPLAY_FONT_PATH))
        except Exception:
            font_id = -1
        if font_id >= 0:
            try:
                families = QFontDatabase.applicationFontFamilies(font_id)
            except Exception:
                families = []
            if families:
                _DISPLAY_FONT_QT_FAMILY = str(families[0] or "").strip() or None
                if _DISPLAY_FONT_QT_FAMILY:
                    if _MPL_AVAILABLE_FONT_NAMES is not None:
                        _MPL_AVAILABLE_FONT_NAMES.add(_DISPLAY_FONT_QT_FAMILY)
            _DISPLAY_FONT_LOADED = True

    def _pick_font_family(self, candidates: list[str]) -> str:
        self._ensure_display_font_loaded()
        if _DISPLAY_FONT_QT_FAMILY and _DISPLAY_FONT_QT_FAMILY in candidates:
            return _DISPLAY_FONT_QT_FAMILY
        for family in _platform_ui_font_candidates():
            if family in candidates:
                return family
        for family in candidates:
            if family and family != "Sans Serif":
                return family
        return _resolved_platform_ui_font_family()

    def _pick_matplotlib_font_family(self, candidates: list[str]) -> str:
        available = _available_mpl_font_names()
        for family in candidates:
            if family in available:
                return family
        return "DejaVu Sans"

    def _set_dark_mode_enabled(self, enabled: bool, persist: bool = True):
        enabled = bool(enabled)
        if enabled == getattr(self, "_dark_enabled", False):
            if persist:
                self.settings.setValue("general/darkMode", enabled)
            self._sync_dark_mode_menu_action()
            return
        self._dark_enabled = enabled
        if persist:
            self.settings.setValue("general/darkMode", self._dark_enabled)
        self._apply_styles()
        self._sync_dark_mode_menu_action()
        self.darkModeChanged.emit(self._dark_enabled)

    def _sync_dark_mode_menu_action(self) -> None:
        if not hasattr(self, "dark_act"):
            return
        blocker = QSignalBlocker(self.dark_act)
        self.dark_act.setChecked(bool(getattr(self, "_dark_enabled", False)))
        del blocker

    def _theme_color(self, key: str, fallback: str) -> str:
        tokens = getattr(self, "_theme_tokens", {})
        value = tokens.get(key) if isinstance(tokens, dict) else None
        return str(value or fallback)

    def _theme_qcolor(self, key: str, fallback: str) -> QColor:
        raw = self._theme_color(key, fallback)
        match = re.fullmatch(
            r"rgba?\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})(?:\s*,\s*([0-9.]+))?\s*\)",
            raw,
        )
        if match:
            red = max(0, min(255, int(match.group(1))))
            green = max(0, min(255, int(match.group(2))))
            blue = max(0, min(255, int(match.group(3))))
            alpha_group = match.group(4)
            alpha = 255
            if alpha_group is not None:
                try:
                    alpha_value = float(alpha_group)
                    alpha = int(round(255 * alpha_value)) if alpha_value <= 1.0 else int(round(alpha_value))
                except ValueError:
                    alpha = 255
                alpha = max(0, min(255, alpha))
            return QColor(red, green, blue, alpha)
        color = QColor(raw)
        if color.isValid():
            return color
        return QColor(fallback)

    def _accent_secondary_settings_key(self, theme_key: str | None = None) -> str:
        normalized = normalize_theme_key(theme_key or getattr(self, "_theme_name", DEFAULT_UI_THEME))
        return f"general/accentSecondaryColorByTheme/{normalized}"

    def _load_accent_secondary_override(self, theme_key: str | None = None) -> str:
        normalized = normalize_theme_key(theme_key or getattr(self, "_theme_name", DEFAULT_UI_THEME))
        per_theme_key = self._accent_secondary_settings_key(normalized)
        stored = _normalized_css_color(self.settings.value(per_theme_key, "", type=str))
        if stored:
            return stored
        return ""

    def _save_accent_secondary_override(self, theme_key: str, color: str) -> None:
        normalized = normalize_theme_key(theme_key)
        per_theme_key = self._accent_secondary_settings_key(normalized)
        normalized_color = _normalized_css_color(color)
        if normalized_color:
            self.settings.setValue(per_theme_key, normalized_color)
        else:
            self.settings.remove(per_theme_key)
        self.settings.remove("general/accentSecondaryColor")

    def _refresh_mpl_theme(self) -> None:
        plot_bg = self._theme_color("plot_bg", "#0f1825")
        plot_panel_bg = self._theme_color("plot_panel_bg", "#162334")
        plot_text = self._theme_color("plot_text", "#d7e4f0")
        plot_grid = self._theme_color("plot_grid", "#2f4666")
        plot_canvas = getattr(self, "plot_canvas", None)
        ax_alt = getattr(self, "ax_alt", None)
        if plot_canvas is not None and ax_alt is not None:
            plot_canvas.figure.patch.set_facecolor(plot_bg)
            ax_alt.set_facecolor(plot_panel_bg)
            ax_alt.tick_params(axis="x", colors=plot_text)
            ax_alt.tick_params(axis="y", colors=plot_text)
            for spine in ax_alt.spines.values():
                spine.set_color(plot_grid)
            ax_alt.xaxis.label.set_color(plot_text)
            ax_alt.yaxis.label.set_color(plot_text)
            ax_alt.title.set_color(plot_text)
        if hasattr(self, "polar_canvas") and hasattr(self, "polar_ax"):
            self.polar_canvas.figure.patch.set_facecolor(plot_bg)
            self.polar_ax.set_facecolor(plot_panel_bg)
            self.polar_ax.tick_params(axis="x", colors=plot_text, pad=0)
            self.polar_ax.tick_params(axis="y", colors=plot_text, pad=1)
            self.polar_ax.grid(True, color=plot_grid, alpha=0.36, linestyle="--", linewidth=0.7)
            for spine in self.polar_ax.spines.values():
                spine.set_color(plot_grid)
            if hasattr(self, "polar_scatter"):
                self.polar_scatter.set_color(self._theme_color("polar_target", "#59f3ff"))
            if hasattr(self, "selected_scatter"):
                self.selected_scatter.set_color(self._theme_color("polar_selected", "#ff4fd8"))
            if hasattr(self, "sun_marker"):
                self.sun_marker.set_color(self._theme_color("polar_sun", "#ffb224"))
            if hasattr(self, "moon_marker"):
                self.moon_marker.set_color(self._theme_color("polar_moon", "#dbe7ff"))
            if hasattr(self, "radar_sweep_line") and self.radar_sweep_line is not None:
                self.radar_sweep_line.set_color(self._qcolor_rgba_mpl(self._theme_qcolor("accent_secondary", "#ff4fd8"), 0.0))
            if hasattr(self, "radar_sweep_glow_line") and self.radar_sweep_glow_line is not None:
                self.radar_sweep_glow_line.set_color(self._qcolor_rgba_mpl(self._theme_qcolor("accent_secondary_soft", "#d38cff"), 0.48))
            if hasattr(self, "radar_sweep_core") and self.radar_sweep_core is not None:
                self.radar_sweep_core.set_visible(False)
            if hasattr(self, "radar_sweep_mesh") and self.radar_sweep_mesh is not None:
                try:
                    self.radar_sweep_mesh.set_cmap(self._build_radar_sweep_cmap())
                except Exception:
                    pass
            if hasattr(self, "_radar_echo_artists") or hasattr(self, "radar_sweep_mesh") or (hasattr(self, "radar_echo_scatter") and self.radar_echo_scatter is not None):
                self._refresh_radar_sweep_artists(redraw=False)
        if plot_canvas is not None:
            plot_canvas.draw_idle()
        if hasattr(self, "polar_canvas"):
            self.polar_canvas.draw_idle()

    def _apply_styles(self):
        """Apply a custom stylesheet, fonts, and default icon sizes."""
        self._ensure_display_font_loaded()
        display_family = str(getattr(self, "_display_font_family", "") or _preferred_display_font_family()).strip()
        self._display_font_family = display_family
        body_family = display_family
        font_size = _sanitize_ui_font_size(getattr(self, "_ui_font_size", 11))
        app_font = QFont(body_family)
        app_font.setPointSize(font_size)
        QApplication.setFont(app_font)
        plt.rcParams["font.family"] = [
            _preferred_display_font_family(),
            "Rajdhani",
            "DejaVu Sans",
            "Arial",
            "Helvetica",
        ]
        self._theme_name = normalize_theme_key(getattr(self, "_theme_name", DEFAULT_UI_THEME))
        theme_overrides: dict[str, str] = {}
        if getattr(self, "_accent_secondary_override", ""):
            theme_overrides["btn2"] = str(self._accent_secondary_override)
        self._theme_tokens = resolve_theme_tokens(
            self._theme_name,
            dark_enabled=getattr(self, "_dark_enabled", False),
            ui_font_size=font_size,
            font_family=body_family,
            display_font_family=display_family,
            overrides=theme_overrides or None,
        )
        if hasattr(self, "_visibility_web_html_cache"):
            self._visibility_web_html_cache.clear()
        stylesheet = build_stylesheet(
            self._theme_name,
            dark_enabled=getattr(self, "_dark_enabled", False),
            ui_font_size=font_size,
            font_family=body_family,
            display_font_family=display_family,
            overrides=theme_overrides or None,
        )
        self.setStyleSheet(stylesheet)
        ai_window = getattr(self, "ai_window", None)
        if isinstance(ai_window, QDialog):
            ai_window.setStyleSheet(stylesheet)
        weather_window = getattr(self, "weather_window", None)
        if isinstance(weather_window, QDialog):
            weather_window.setStyleSheet(stylesheet)
        if getattr(self, "visibility_web", None) is not None:
            try:
                self.visibility_web.setStyleSheet(f"background:{self._theme_color('plot_bg', '#0f192b')};")
                self.visibility_web.page().setBackgroundColor(self._theme_qcolor("plot_bg", "#0f192b"))
            except Exception:
                pass
        if getattr(self, "visibility_plot_host", None) is not None:
            try:
                self.visibility_plot_host.setStyleSheet(f"background:{self._theme_color('plot_bg', '#0f192b')};")
            except Exception:
                pass
        if getattr(self, "visibility_loading_widget", None) is not None:
            try:
                self.visibility_loading_widget.setStyleSheet(f"background:{self._theme_color('plot_bg', '#0f192b')};")
            except Exception:
                pass
        bootstrap_placeholder = getattr(self, "_visibility_bootstrap_placeholder", None)
        if isinstance(bootstrap_placeholder, QWidget):
            try:
                bootstrap_placeholder.setStyleSheet(f"background:{self._theme_color('plot_bg', '#0f192b')};")
            except Exception:
                pass
        for placeholder in getattr(self, "_cutout_image_placeholders", {}).values():
            if isinstance(placeholder, QWidget):
                try:
                    placeholder.setStyleSheet(f"background:{self._theme_color('plot_bg', '#0f192b')};")
                except Exception:
                    pass
        for stack in getattr(self, "_cutout_image_stacks", {}).values():
            host = stack.parentWidget() if isinstance(stack, QStackedLayout) else None
            if isinstance(host, QWidget):
                try:
                    host.setStyleSheet(f"background:{self._theme_color('plot_bg', '#0f192b')};")
                except Exception:
                    pass
        self._refresh_target_color_map()
        self._refresh_mpl_theme()
        self._refresh_date_nav_icons()
        self._refresh_plot_mode_switch()
        self._update_plot_mode_label_metrics()
        if isinstance(getattr(self, "last_payload", None), dict):
            self._render_visibility_web_plot(self.last_payload)
        if hasattr(self, "table_model") and hasattr(self, "settings"):
            palette = highlight_palette_for_theme(
                getattr(self, "_theme_name", DEFAULT_UI_THEME),
                dark_enabled=bool(getattr(self, "_dark_enabled", False)),
                color_blind=bool(getattr(self, "color_blind_mode", False)),
            )
            default_colors = {"below": palette.below, "limit": palette.limit, "above": palette.above}
            self.table_model.highlight_colors = {
                key: _qcolor_from_token(_resolve_table_highlight_color(self.settings, key, default_colors[key]), default_colors[key])
                for key in default_colors
            }
            if hasattr(self, "_emit_table_data_changed"):
                self._emit_table_data_changed()
        if isinstance(weather_window, WeatherDialog):
            self._refresh_weather_window_context()
        _refresh_button_icons(self)
        if isinstance(ai_window, QDialog):
            _refresh_button_icons(ai_window)
        if isinstance(weather_window, QDialog):
            _refresh_button_icons(weather_window)
        self._update_night_details_constraints()
        # Apply icon size globally
        self.setIconSize(QSize(20, 20))

    def _build_date_nav_icon_pixmap(self, kind: str, stroke: QColor, size: int = 20) -> QPixmap:
        px = QPixmap(size, size)
        px.fill(Qt.GlobalColor.transparent)
        painter = QPainter(px)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        pen = QPen(stroke)
        pen.setWidthF(max(1.5, size * 0.11))
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        if kind == "prev":
            painter.drawLine(int(size * 0.60), int(size * 0.24), int(size * 0.37), int(size * 0.50))
            painter.drawLine(int(size * 0.60), int(size * 0.76), int(size * 0.37), int(size * 0.50))
        elif kind == "next":
            painter.drawLine(int(size * 0.40), int(size * 0.24), int(size * 0.63), int(size * 0.50))
            painter.drawLine(int(size * 0.40), int(size * 0.76), int(size * 0.63), int(size * 0.50))
        else:
            m = size * 0.22
            arc = int(size - 2 * m)
            painter.drawArc(int(m), int(m), arc, arc, int(35 * 16), int(290 * 16))
            tip_x = int(size * 0.77)
            tip_y = int(size * 0.30)
            painter.drawLine(tip_x, tip_y, int(size * 0.64), int(size * 0.30))
            painter.drawLine(tip_x, tip_y, int(size * 0.77), int(size * 0.43))
        painter.end()
        return px

    def _build_date_nav_icon(self, kind: str) -> QIcon:
        base = self.palette().color(QPalette.ColorRole.ButtonText)
        if not base.isValid():
            base = self._theme_qcolor("section_title", "#d8e6f5")
        normal = QColor(base)
        normal.setAlpha(225)
        active = QColor(base)
        active.setAlpha(255)
        disabled = QColor(base)
        disabled.setAlpha(95)
        icon = QIcon()
        icon.addPixmap(self._build_date_nav_icon_pixmap(kind, normal), QIcon.Mode.Normal, QIcon.State.Off)
        icon.addPixmap(self._build_date_nav_icon_pixmap(kind, active), QIcon.Mode.Active, QIcon.State.Off)
        icon.addPixmap(self._build_date_nav_icon_pixmap(kind, disabled), QIcon.Mode.Disabled, QIcon.State.Off)
        return icon

    def _refresh_date_nav_icons(self):
        for attr, kind in (
            ("prev_day_btn", "prev"),
            ("today_btn", "today"),
            ("next_day_btn", "next"),
        ):
            btn = getattr(self, attr, None)
            if isinstance(btn, QToolButton):
                btn.setIcon(self._build_date_nav_icon(kind))

    def _build_web_loading_placeholder(self, title: str, message: str) -> QWidget:
        widget = QWidget(self)
        widget.setAttribute(Qt.WA_StyledBackground, True)
        widget.setStyleSheet(f"background:{self._theme_color('plot_bg', '#121b29')};")
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)
        layout.addStretch(1)
        card = QFrame(widget)
        card.setObjectName("VisibilityLoadingCard")
        card.setMinimumWidth(420)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(30, 26, 30, 26)
        card_layout.setSpacing(14)
        title_label = QLabel(title, card)
        title_label.setObjectName("SectionTitle")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setMinimumWidth(360)
        skeleton = SkeletonShimmerWidget("plot", card)
        skeleton.setMinimumHeight(260)
        hint_label = QLabel(message, card)
        hint_label.setObjectName("SectionHint")
        hint_label.setAlignment(Qt.AlignCenter)
        hint_label.setWordWrap(True)
        hint_label.setMinimumWidth(380)
        hint_label.setMaximumWidth(520)
        hint_label.setMinimumHeight(54)
        hint_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.MinimumExpanding)
        card_layout.addWidget(title_label, 0, Qt.AlignHCenter)
        card_layout.addWidget(skeleton, 0)
        card_layout.addWidget(hint_label, 0, Qt.AlignHCenter)
        layout.addWidget(card, 0, Qt.AlignCenter)
        layout.addStretch(1)
        widget._loading_hint_label = hint_label  # type: ignore[attr-defined]
        widget._loading_skeleton = skeleton  # type: ignore[attr-defined]
        return widget

    def _build_cutout_loading_placeholder(self, title: str, message: str) -> QWidget:
        widget = QWidget(self)
        widget.setAttribute(Qt.WA_StyledBackground, True)
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        card = QFrame(widget)
        card.setObjectName("CutoutImage")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(14, 14, 14, 14)
        card_layout.setSpacing(10)
        title_label = QLabel(title, card)
        title_label.setObjectName("SectionTitle")
        title_label.setAlignment(Qt.AlignCenter)
        skeleton = SkeletonShimmerWidget("image", card)
        skeleton.setMinimumHeight(220)
        hint_label = QLabel(message, card)
        hint_label.setObjectName("SectionHint")
        hint_label.setWordWrap(True)
        hint_label.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(title_label, 0, Qt.AlignHCenter)
        card_layout.addWidget(skeleton, 1)
        card_layout.addWidget(hint_label, 0, Qt.AlignHCenter)
        layout.addWidget(card, 1)
        widget._loading_hint_label = hint_label  # type: ignore[attr-defined]
        return widget

    def _ensure_visibility_plot_widgets(self) -> None:
        if bool(getattr(self, "_visibility_plot_widgets_ready", False)):
            return

        if self.plot_canvas is None or self.ax_alt is None:
            self.plot_canvas = FigureCanvas(Figure(figsize=(6, 4), tight_layout=True))
            self.plot_canvas.setContentsMargins(0, 0, 0, 0)
            self.plot_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.ax_alt = self.plot_canvas.figure.subplots()
            self.ax_alt.margins(x=0, y=0)

        if self._use_visibility_web and QWebEngineView is not None:
            self.visibility_web = QWebEngineView(self)
            self.visibility_web.setMinimumHeight(320)
            self.visibility_web.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.visibility_web.setStyleSheet(f"background:{self._theme_color('plot_bg', '#121b29')};")
            self.visibility_web.setToolTip("Interactive visibility chart. Use Plotly controls in the top-right to zoom, pan and reset.")
            self.visibility_loading_widget = self._build_web_loading_placeholder(
                "Visibility Plot",
                "Computing night tracks…",
            )
            self.visibility_plot_stack = QStackedLayout()
            self.visibility_plot_stack.setContentsMargins(0, 0, 0, 0)
            self.visibility_plot_host = QWidget(self)
            self.visibility_plot_host.setAttribute(Qt.WA_StyledBackground, True)
            self.visibility_plot_host.setStyleSheet(f"background:{self._theme_color('plot_bg', '#121b29')};")
            self.visibility_plot_host.setLayout(self.visibility_plot_stack)
            self.visibility_plot_stack.addWidget(self.visibility_loading_widget)
            self.visibility_plot_stack.addWidget(self.visibility_web)
            self.visibility_plot_stack.setCurrentWidget(self.visibility_loading_widget)
            self.visibility_web.loadFinished.connect(self._on_visibility_web_load_finished)
            plot_content_widget = self.visibility_plot_host
        else:
            self.plot_toolbar = NavigationToolbar(self.plot_canvas, self)
            self.plot_toolbar.setIconSize(QSize(16, 16))
            self.plot_toolbar.layout().setContentsMargins(0, 0, 0, 0)
            self.plot_toolbar.layout().setSpacing(0)
            self.plot_toolbar.addSeparator()
            self.plot_toolbar.addWidget(self.plot_mode_widget)
            self.plot_toolbar.setMaximumHeight(self.plot_toolbar.sizeHint().height())
            plot_content_widget = self.plot_canvas

        placeholder = getattr(self, "_visibility_bootstrap_placeholder", None)
        plot_layout = getattr(self, "plot_card_layout", None)
        if isinstance(plot_layout, QVBoxLayout):
            if isinstance(placeholder, QWidget):
                plot_layout.removeWidget(placeholder)
                placeholder.hide()
            if self.plot_toolbar is not None and plot_layout.indexOf(self.plot_toolbar) < 0:
                plot_layout.insertWidget(0, self.plot_toolbar)
            if isinstance(plot_content_widget, QWidget) and plot_layout.indexOf(plot_content_widget) < 0:
                plot_layout.addWidget(plot_content_widget, 1)

        self._visibility_plot_widgets_ready = True
        self._refresh_mpl_theme()

    def _ensure_visibility_plot_placeholder_message(self, message: str) -> None:
        placeholder = getattr(self, "_visibility_bootstrap_placeholder", None)
        hint_label = getattr(placeholder, "_loading_hint_label", None)
        if isinstance(hint_label, QLabel):
            hint_label.setText(message)

    def _create_cutout_image_stack(
        self,
        parent: QWidget,
        *,
        kind: str,
        title: str,
    ) -> tuple[QWidget, CoverImageLabel]:
        host = QWidget(parent)
        host.setAttribute(Qt.WA_StyledBackground, True)
        host.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        stack = QStackedLayout(host)
        stack.setContentsMargins(0, 0, 0, 0)
        placeholder = self._build_cutout_loading_placeholder(title, f"Loading {title.lower()}…")
        image_label = CoverImageLabel("Select a target", host)
        image_label.setObjectName("CutoutImage")
        image_label.setProperty("cutout_image", True)
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setWordWrap(True)
        image_label.setScaledContents(False)
        image_label.setMinimumSize(1, 1)
        image_label.setMaximumSize(16777215, 16777215)
        image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        stack.addWidget(placeholder)
        stack.addWidget(image_label)
        stack.setCurrentWidget(image_label)
        self._cutout_image_stacks[kind] = stack
        self._cutout_image_placeholders[kind] = placeholder
        self._cutout_image_labels[kind] = image_label
        return host, image_label

    def _set_cutout_image_loading(self, kind: str, message: str, *, visible: bool = True) -> None:
        stack = self._cutout_image_stacks.get(kind)
        placeholder = self._cutout_image_placeholders.get(kind)
        image_label = self._cutout_image_labels.get(kind)
        if stack is None or placeholder is None or image_label is None:
            return
        hint_label = getattr(placeholder, "_loading_hint_label", None)
        if isinstance(hint_label, QLabel):
            hint_label.setText(message)
        if visible:
            stack.setCurrentWidget(placeholder)
        else:
            stack.setCurrentWidget(image_label)

    def _set_visibility_loading_state(self, message: str, *, visible: bool = True) -> None:
        stack = getattr(self, "visibility_plot_stack", None)
        placeholder = getattr(self, "visibility_loading_widget", None)
        web_view = getattr(self, "visibility_web", None)
        if stack is None or placeholder is None or web_view is None:
            return
        hint_label = getattr(placeholder, "_loading_hint_label", None)
        if isinstance(hint_label, QLabel):
            hint_label.setText(message)
        if visible:
            stack.setCurrentWidget(placeholder)
        else:
            stack.setCurrentWidget(web_view)

    @Slot(bool)
    def _on_visibility_web_load_finished(self, ok: bool) -> None:
        if ok:
            self._visibility_web_has_content = True
            self._set_visibility_loading_state("", visible=False)
            self._apply_visibility_web_selection_style()
            return
        self._visibility_web_has_content = False
        self._set_visibility_loading_state("Unable to render interactive chart.", visible=True)

    def _line_palette(self) -> list[str]:
        palette = line_palette_for_theme(
            getattr(self, "_theme_name", DEFAULT_UI_THEME),
            dark_enabled=getattr(self, "_dark_enabled", False),
            color_blind=bool(getattr(self, "color_blind_mode", False)),
        )
        return palette or (COLORBLIND_LINE_COLORS if self.color_blind_mode else DEFAULT_LINE_COLORS)

    @staticmethod
    def _target_color_key(target: Target) -> str:
        source_key = _normalize_catalog_token(getattr(target, "source_object_id", ""))
        if source_key:
            return f"id:{source_key}"
        name_key = _normalize_catalog_token(getattr(target, "name", ""))
        if name_key:
            return f"name:{name_key}"
        ra = _safe_float(getattr(target, "ra", None))
        dec = _safe_float(getattr(target, "dec", None))
        if ra is not None and dec is not None and math.isfinite(ra) and math.isfinite(dec):
            return f"coord:{ra:.6f},{dec:.6f}"
        return f"obj:{id(target)}"

    def _ensure_auto_target_color_palette(self, palette: Optional[list[str]] = None) -> list[str]:
        use_palette = palette if palette is not None else self._line_palette()
        signature = tuple(str(color) for color in use_palette)
        prev_signature = tuple(getattr(self, "_auto_target_color_palette_signature", ()))
        if signature != prev_signature:
            self._auto_target_color_palette_signature = signature
            self._auto_target_color_map = {}
        return use_palette

    def _target_plot_color_css(
        self,
        target: Target,
        index: int,
        palette: Optional[list[str]] = None,
    ) -> str:
        custom_css = _normalized_css_color(target.plot_color)
        if custom_css:
            return custom_css
        use_palette = self._ensure_auto_target_color_palette(palette)
        if not use_palette:
            return self._theme_color("accent_primary", "#4da3ff")
        key = self._target_color_key(target)
        auto_map = getattr(self, "_auto_target_color_map", {})
        cached = _normalized_css_color(auto_map.get(key, ""))
        if cached:
            return cached
        color_css = str(use_palette[len(auto_map) % len(use_palette)])
        normalized = _normalized_css_color(color_css) or color_css
        auto_map[key] = str(normalized)
        self._auto_target_color_map = auto_map
        return str(normalized)

    def _airmass_from_altitude(self, altitude_deg: object) -> np.ndarray:
        altitude = np.asarray(altitude_deg, dtype=float)
        airmass = np.full_like(altitude, np.nan, dtype=float)
        valid = np.isfinite(altitude) & (altitude > 0.0)
        if np.any(valid):
            alt_valid = altitude[valid]
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                denom = np.sin(np.radians(alt_valid)) + 0.50572 * np.power(alt_valid + 6.07995, -1.6364)
                airmass[valid] = 1.0 / denom
        return airmass

    def _plot_y_values(self, altitude_deg: object) -> np.ndarray:
        altitude = np.asarray(altitude_deg, dtype=float)
        if not self._plot_airmass:
            return altitude
        return self._airmass_from_altitude(altitude)

    def _plot_limit_value(self) -> float:
        limit_alt = float(self.limit_spin.value())
        if not self._plot_airmass:
            return limit_alt
        limit_airmass = self._airmass_from_altitude(np.array([limit_alt], dtype=float))
        value = _safe_float(limit_airmass[0])
        return value if value is not None else 1.0

    def _visibility_time_window(
        self,
        data: dict,
        tz,
    ) -> tuple[datetime, datetime, dict[str, datetime]]:
        event_map: dict[str, datetime] = {}
        for key in (
            "sunset",
            "dusk_civ",
            "dusk_naut",
            "dusk",
            "dawn",
            "dawn_naut",
            "dawn_civ",
            "sunrise",
            "moonrise",
            "moonset",
        ):
            try:
                event_map[key] = mdates.num2date(data[key]).astimezone(tz)
            except Exception:
                continue

        sunset_dt = event_map.get("sunset")
        sunrise_dt = event_map.get("sunrise")
        if isinstance(sunset_dt, datetime) and isinstance(sunrise_dt, datetime):
            start_dt = sunset_dt - timedelta(hours=1)
            end_dt = sunrise_dt + timedelta(hours=1)
            if end_dt > start_dt:
                return start_dt, end_dt, event_map

        obs_date = self.date_edit.date()
        start_noon_naive = datetime(obs_date.year(), obs_date.month(), obs_date.day(), 12, 0)
        next_date = obs_date.addDays(1)
        end_noon_naive = datetime(next_date.year(), next_date.month(), next_date.day(), 12, 0)
        try:
            start_dt = tz.localize(start_noon_naive)
            end_dt = tz.localize(end_noon_naive)
        except Exception:
            center_dt = mdates.num2date(data["midnight"]).astimezone(tz)
            start_dt = center_dt - timedelta(hours=12)
            end_dt = center_dt + timedelta(hours=12)
        return start_dt, end_dt, event_map

    def _visibility_grid_color(self, *, alpha: float = 1.0) -> str:
        base = self._theme_qcolor("plot_grid", "#2f4666")
        panel = self._theme_qcolor("plot_panel_bg", "#162334")
        softened = self._mix_qcolors(base, panel, 0.34)
        return self._qcolor_rgba_css(softened, alpha)

    def _visibility_grid_rgba(self, *, alpha: float = 1.0) -> tuple[float, float, float, float]:
        base = self._theme_qcolor("plot_grid", "#2f4666")
        panel = self._theme_qcolor("plot_panel_bg", "#162334")
        softened = self._mix_qcolors(base, panel, 0.34)
        return self._qcolor_rgba_mpl(softened, alpha)

    def _visibility_context_key_from_parts(
        self,
        *,
        site_name: str,
        latitude: float,
        longitude: float,
        elevation: float,
        obs_date,
        time_samples: int,
        limit_altitude: float,
    ) -> str:
        return (
            f"{site_name}|{latitude:.6f}|{longitude:.6f}|{elevation:.1f}|"
            f"{obs_date.toString('yyyy-MM-dd')}|{int(time_samples)}|{float(limit_altitude):.1f}"
        )

    def _current_visibility_context_key(self) -> str:
        try:
            return self._visibility_context_key_from_parts(
                site_name=str(self.obs_combo.currentText()),
                latitude=self._read_site_float(self.lat_edit),
                longitude=self._read_site_float(self.lon_edit),
                elevation=self._read_site_float(self.elev_edit),
                obs_date=self.date_edit.date(),
                time_samples=self.settings.value("general/timeSamples", 240, type=int),
                limit_altitude=float(self.limit_spin.value()),
            )
        except Exception:
            return ""

    def _show_visibility_matplotlib_placeholder(self, title: str, message: str) -> None:
        if getattr(self, "_use_visibility_web", False):
            return
        if getattr(self, "ax_alt", None) is None or getattr(self, "plot_canvas", None) is None:
            return
        plot_bg = self._theme_color("plot_bg", "#0f1825")
        plot_panel_bg = self._theme_color("plot_panel_bg", "#162334")
        plot_text = self._theme_color("plot_text", "#d7e4f0")
        self.ax_alt.clear()
        self.plot_canvas.figure.patch.set_facecolor(plot_bg)
        self.ax_alt.set_facecolor(plot_panel_bg)
        self.ax_alt.set_xticks([])
        self.ax_alt.set_yticks([])
        for spine in self.ax_alt.spines.values():
            spine.set_visible(False)
        self.ax_alt.text(
            0.5,
            0.56,
            title,
            transform=self.ax_alt.transAxes,
            ha="center",
            va="center",
            color=plot_text,
            fontsize=15,
            fontweight="bold",
        )
        self.ax_alt.text(
            0.5,
            0.46,
            message,
            transform=self.ax_alt.transAxes,
            ha="center",
            va="center",
            color=self._theme_color("section_hint", plot_text),
            fontsize=10,
            wrap=True,
        )
        self.plot_canvas.draw_idle()

    def _begin_visibility_refresh(self, message: str) -> None:
        self._visibility_web_has_content = False
        if not bool(getattr(self, "_visibility_plot_widgets_ready", False)):
            self._ensure_visibility_plot_placeholder_message(message)
        if getattr(self, "visibility_web", None) is not None:
            self.visibility_web.setEnabled(False)
            self._set_visibility_loading_state(message, visible=True)
        self._show_visibility_matplotlib_placeholder("Visibility Plot", message)

    def _configure_main_plot_y_axis(self) -> None:
        if not self._plot_airmass:
            self.ax_alt.set_ylabel("Altitude (°)")
            self.ax_alt.set_ylim(0, 90)
            self.ax_alt.set_yticks([0, 15, 30, 45, 60, 75, 90])
            return

        self.ax_alt.set_ylabel("Airmass")
        self.ax_alt.set_ylim(VISIBILITY_AIRMASS_Y_MAX, VISIBILITY_AIRMASS_Y_MIN)
        self.ax_alt.set_yticks(VISIBILITY_AIRMASS_Y_TICKS)
        self.ax_alt.set_yticklabels([f"{tick:.1f}" for tick in VISIBILITY_AIRMASS_Y_TICKS])

    def _update_plot_mode_label_metrics(self) -> None:
        if not hasattr(self, "plot_mode_alt_label") or not hasattr(self, "plot_mode_airmass_label"):
            return
        for label, text, align in (
            (self.plot_mode_alt_label, "Altitude", Qt.AlignRight | Qt.AlignVCenter),
            (self.plot_mode_airmass_label, "Airmass", Qt.AlignLeft | Qt.AlignVCenter),
        ):
            metrics_font = QFont(label.font())
            metrics_font.setWeight(QFont.Weight.Bold)
            width = QFontMetrics(metrics_font).horizontalAdvance(text) + 28
            label.setMinimumWidth(max(width, 102))
            label.setAlignment(align)
        if hasattr(self, "plot_mode_widget"):
            switch_width = 70
            if hasattr(self, "airmass_toggle_btn") and self.airmass_toggle_btn is not None:
                switch_width = max(switch_width, self.airmass_toggle_btn.minimumSizeHint().width())
            self.plot_mode_widget.setMinimumWidth(
                self.plot_mode_alt_label.minimumWidth() + self.plot_mode_airmass_label.minimumWidth() + switch_width + 52
            )
            self.plot_mode_widget.setMinimumHeight(44)

    def _animate_plot_mode_switch(self) -> None:
        labels = (
            (getattr(self, "plot_mode_alt_label", None), not self._plot_airmass),
            (getattr(self, "plot_mode_airmass_label", None), self._plot_airmass),
        )
        for anim in self._plot_mode_animations:
            try:
                anim.stop()
            except Exception:
                pass
        self._plot_mode_animations.clear()
        for label, active in labels:
            effect = getattr(label, "_opacity_effect", None)
            if label is None or effect is None:
                continue
            target = 1.0 if active else 0.68
            anim = QPropertyAnimation(effect, b"opacity", self)
            anim.setDuration(180)
            anim.setStartValue(effect.opacity())
            anim.setEndValue(target)
            anim.setEasingCurve(QEasingCurve.OutCubic)
            anim.finished.connect(lambda a=anim: self._plot_mode_animations.remove(a) if a in self._plot_mode_animations else None)
            self._plot_mode_animations.append(anim)
            anim.start()

    def _refresh_plot_mode_switch(self) -> None:
        if not hasattr(self, "plot_mode_alt_label") or not hasattr(self, "plot_mode_airmass_label"):
            return
        if hasattr(self, "airmass_toggle_btn") and self.airmass_toggle_btn is not None:
            blocker = QSignalBlocker(self.airmass_toggle_btn)
            self.airmass_toggle_btn.setChecked(bool(self._plot_airmass))
            del blocker
        for label, active in (
            (self.plot_mode_alt_label, not self._plot_airmass),
            (self.plot_mode_airmass_label, self._plot_airmass),
        ):
            font = QFont(label.font())
            font.setWeight(QFont.Weight.Bold if active else QFont.Weight.Medium)
            label.setFont(font)
            _set_label_tone(label, "info" if active else "muted")
            effect = getattr(label, "_opacity_effect", None)
            if effect is not None:
                effect.setOpacity(1.0 if active else 0.68)
        self._update_plot_mode_label_metrics()

    @staticmethod
    def _polar_rgba_array(color: QColor, alphas: np.ndarray) -> np.ndarray:
        if not color.isValid():
            color = QColor("#59f3ff")
        red = color.redF()
        green = color.greenF()
        blue = color.blueF()
        rows = []
        for alpha in alphas:
            rows.append((red, green, blue, float(max(0.0, min(1.0, alpha)))))
        return np.array(rows, dtype=float) if rows else np.empty((0, 4), dtype=float)

    def _ensure_radar_echo_artists(self, count: int) -> None:
        artists = list(getattr(self, "_radar_echo_artists", []))
        while len(artists) < count:
            artist, = self.polar_ax.plot(
                [],
                [],
                linestyle="",
                marker="o",
                markersize=1.5,
                markeredgewidth=1.35,
                alpha=0.0,
                zorder=4,
            )
            artists.append(artist)
        while len(artists) > count:
            artist = artists.pop()
            try:
                artist.remove()
            except Exception:
                pass
        self._radar_echo_artists = artists

    def _build_radar_sweep_cmap(self) -> LinearSegmentedColormap:
        base = self._theme_qcolor("accent_secondary_soft", "#d38cff")
        r = float(base.redF())
        g = float(base.greenF())
        b = float(base.blueF())
        return LinearSegmentedColormap.from_list(
            "astroplanner_radar_sweep",
            [
                (0.0, (r, g, b, 0.0)),
                (0.28, (r, g, b, 0.04)),
                (0.68, (r, g, b, 0.20)),
                (1.0, (r, g, b, 0.54)),
            ],
            N=512,
        )

    @staticmethod
    def _radar_sector_vertices(theta_start: float, theta_end: float, outer_radius: float = 90.0, samples: int = 12) -> np.ndarray:
        arc_thetas = np.linspace(theta_start, theta_end, max(4, int(samples)))
        vertices: list[tuple[float, float]] = [(theta_start, 0.0)]
        vertices.extend((float(theta), float(outer_radius)) for theta in arc_thetas)
        vertices.append((theta_end, 0.0))
        vertices.append((theta_start, 0.0))
        return np.array(vertices, dtype=float)

    def _refresh_radar_sweep_artists(self, *, redraw: bool = True, delta_s: Optional[float] = None) -> None:
        if (
            not hasattr(self, "radar_sweep_line")
            or not hasattr(self, "polar_ax")
        ):
            return
        enabled = bool(getattr(self, "_radar_sweep_enabled", False))
        if not enabled:
            self.radar_sweep_line.set_data([], [])
            self.radar_sweep_glow_line.set_data([], [])
            if hasattr(self, "radar_sweep_core"):
                self.radar_sweep_core.set_offsets(np.empty((0, 2)))
            if hasattr(self, "radar_sweep_mesh") and self.radar_sweep_mesh is not None:
                self.radar_sweep_mesh.set_array(np.zeros_like(self._radar_sweep_mesh_values).ravel())
                self.radar_sweep_mesh.set_visible(False)
            self._ensure_radar_echo_artists(0)
            self._radar_echo_strengths = np.zeros(0, dtype=float)
            self.polar_scatter.set_alpha(0.52)
            if hasattr(self, "selected_scatter"):
                self.selected_scatter.set_alpha(1.0)
            if redraw and hasattr(self, "polar_canvas"):
                self.polar_canvas.draw_idle()
            return

        theta = float(getattr(self, "_radar_sweep_angle", 0.0)) % (2.0 * math.pi)
        self.radar_sweep_line.set_data([theta, theta], [0.0, 90.0])
        self.radar_sweep_glow_line.set_data([theta, theta], [0.0, 90.0])
        if hasattr(self, "radar_sweep_core"):
            self.radar_sweep_core.set_offsets(np.empty((0, 2)))
        self.polar_scatter.set_alpha(0.0)
        if hasattr(self, "selected_scatter"):
            self.selected_scatter.set_alpha(1.0)
        if hasattr(self, "radar_sweep_mesh") and self.radar_sweep_mesh is not None:
            centers = getattr(self, "_radar_sweep_theta_centers", np.empty(0, dtype=float))
            if isinstance(centers, np.ndarray) and centers.size:
                trail_extent = math.pi / 2.05
                deltas = (theta - centers) % (2.0 * math.pi)
                strengths_1d = np.zeros_like(centers)
                mask = deltas <= trail_extent
                if np.any(mask):
                    normalized = 1.0 - (deltas[mask] / trail_extent)
                    strengths_1d[mask] = np.power(normalized, 1.28)
                radial_rows = len(self._radar_sweep_radius_edges) - 1
                mesh_values = np.repeat(strengths_1d[np.newaxis, :], radial_rows, axis=0)
                self._radar_sweep_mesh_values = mesh_values
                self.radar_sweep_mesh.set_array(mesh_values.ravel())
                self.radar_sweep_mesh.set_visible(bool(np.any(mesh_values > 0.0005)))
        coords = getattr(self, "_radar_target_coords", np.empty((0, 2)))
        strengths = getattr(self, "_radar_echo_strengths", np.zeros(0, dtype=float))
        if not isinstance(strengths, np.ndarray) or strengths.shape[0] != (coords.shape[0] if isinstance(coords, np.ndarray) else 0):
            strengths = np.zeros(coords.shape[0] if isinstance(coords, np.ndarray) else 0, dtype=float)
        if delta_s is not None and strengths.size:
            speed_multiplier = max(0.4, min(2.6, float(getattr(self, "_radar_sweep_speed", 140)) / 100.0))
            revolution_s = 1.0 / speed_multiplier
            decay_tau = max(0.06, revolution_s / 6.0)
            strengths *= math.exp(-max(0.0, float(delta_s)) / decay_tau)
        if isinstance(coords, np.ndarray) and coords.size:
            deltas = np.abs(np.angle(np.exp(1j * (coords[:, 0] - theta))))
            sweep_width = np.deg2rad(8.5)
            mask = deltas <= sweep_width
            if np.any(mask):
                strengths[mask] = np.maximum(strengths[mask], 1.0)
            visible = strengths > 0.0002
            if np.any(visible):
                offsets = coords[visible]
                vis_strength = strengths[visible]
                main_sizes = 1.8 + (np.power(vis_strength, 0.78) * 3.6)
                main_alphas = np.clip(np.power(vis_strength, 1.55) * 0.52, 0.0, 0.52)
                echo_color = self._theme_qcolor("polar_target", "#59f3ff")
                self._ensure_radar_echo_artists(len(offsets))
                for idx, ((theta_value, radius_value), size_value, alpha_value) in enumerate(
                    zip(offsets, main_sizes, main_alphas)
                ):
                    artist = self._radar_echo_artists[idx]
                    artist.set_data([float(theta_value)], [float(radius_value)])
                    artist.set_markersize(float(size_value))
                    artist.set_markeredgewidth(1.05 + (float(alpha_value) * 1.35))
                    artist.set_color(self._qcolor_rgba_mpl(echo_color, float(alpha_value)))
                    artist.set_alpha(float(alpha_value))
                for artist in self._radar_echo_artists[len(offsets):]:
                    artist.set_data([], [])
                    artist.set_alpha(0.0)
            else:
                for artist in getattr(self, "_radar_echo_artists", []):
                    artist.set_data([], [])
                    artist.set_alpha(0.0)
        else:
            self._ensure_radar_echo_artists(0)
        self._radar_echo_strengths = strengths
        if redraw and hasattr(self, "polar_canvas"):
            self.polar_canvas.draw_idle()

    def _update_radar_sweep_state(self) -> None:
        enabled = bool(getattr(self, "_radar_sweep_enabled", False))
        timer = getattr(self, "_radar_sweep_timer", None)
        if timer is None:
            return
        if enabled:
            if hasattr(self, "_radar_sweep_clock"):
                try:
                    self._radar_sweep_clock.start()
                except Exception:
                    pass
            if not timer.isActive():
                timer.start()
        else:
            timer.stop()
            self._radar_sweep_clock.invalidate()
        self._refresh_radar_sweep_artists(redraw=True)

    @Slot()
    def _advance_radar_sweep(self) -> None:
        elapsed_ms = 16.0
        if hasattr(self, "_radar_sweep_clock"):
            if not self._radar_sweep_clock.isValid():
                self._radar_sweep_clock.start()
            elapsed_ms = float(self._radar_sweep_clock.restart())
        delta_s = max(1.0 / 240.0, min(0.05, elapsed_ms / 1000.0))
        speed_multiplier = max(0.4, min(2.6, float(getattr(self, "_radar_sweep_speed", 140)) / 100.0))
        degrees_per_second = 360.0 * speed_multiplier
        self._radar_sweep_angle = (
            float(getattr(self, "_radar_sweep_angle", 0.0)) + math.radians(degrees_per_second * delta_s)
        ) % (2.0 * math.pi)
        self._refresh_radar_sweep_artists(redraw=True, delta_s=delta_s)

    def _reset_plot_navigation_home(self) -> None:
        toolbar = getattr(self, "plot_toolbar", None)
        if toolbar is None:
            return
        try:
            toolbar.update()
            toolbar.push_current()
        except Exception:
            pass

    def _selected_target_names(self) -> list[str]:
        if not hasattr(self, "table_view"):
            return []
        selection_model = self.table_view.selectionModel()
        if selection_model is None:
            return []
        names: list[str] = []
        for index in selection_model.selectedRows():
            row = index.row()
            if 0 <= row < len(self.targets):
                names.append(self.targets[row].name)
        return names

    def _schedule_selected_cutout_update(self, target: Optional[Target]) -> None:
        self._pending_selected_cutout_target = target
        if getattr(self, "_defer_startup_preview_updates", False):
            return
        if hasattr(self, "_selected_cutout_update_timer"):
            self._selected_cutout_update_timer.start()
            return
        self._flush_selected_cutout_update()

    @Slot()
    def _flush_selected_cutout_update(self) -> None:
        if getattr(self, "_defer_startup_preview_updates", False):
            return
        target = self._pending_selected_cutout_target
        self._pending_selected_cutout_target = None
        self._update_cutout_preview_for_target(target)

    def _refresh_visibility_matplotlib_mode_only(self, data: Optional[dict] = None) -> None:
        payload = data if isinstance(data, dict) else getattr(self, "last_payload", None)
        if not isinstance(payload, dict):
            return
        self._ensure_visibility_plot_widgets()
        if getattr(self, "ax_alt", None) is None or getattr(self, "plot_canvas", None) is None:
            return
        times = payload.get("times")
        if not isinstance(times, np.ndarray) and not isinstance(times, list):
            return
        sample_count = len(times)
        limit = float(self.limit_spin.value())
        sun_alt_series = np.array(payload.get("sun_alt", np.full(sample_count, np.nan)), dtype=float)
        sun_alt_limit = self._sun_alt_limit()
        obs_sun_mask = np.isfinite(sun_alt_series) & (sun_alt_series <= sun_alt_limit)

        base_lines: dict[str, Any] = {}
        high_lines: dict[str, Any] = {}
        for name, line, is_over in list(getattr(self, "vis_lines", [])):
            if is_over:
                high_lines[name] = line
            else:
                base_lines[name] = line

        for tgt in self.targets:
            row = payload.get(tgt.name)
            if not isinstance(row, dict):
                continue
            alt = np.array(row.get("altitude", np.full(sample_count, np.nan)), dtype=float)
            if alt.shape[0] != sample_count:
                continue
            base_line = base_lines.get(tgt.name)
            if base_line is not None:
                alt_vis = np.array(alt, copy=True)
                alt_vis[~(np.isfinite(alt) & (alt > 0.0))] = np.nan
                base_line.set_ydata(self._plot_y_values(alt_vis))
            high_line = high_lines.get(tgt.name)
            if high_line is not None:
                alt_high = np.array(alt, copy=True)
                alt_high[~(np.isfinite(alt) & (alt >= limit) & obs_sun_mask)] = np.nan
                high_line.set_ydata(self._plot_y_values(alt_high))

        if getattr(self, "sun_line", None) is not None and "sun_alt" in payload:
            self.sun_line.set_ydata(self._plot_y_values(payload["sun_alt"]))
        if getattr(self, "moon_line", None) is not None and "moon_alt" in payload:
            self.moon_line.set_ydata(self._plot_y_values(payload["moon_alt"]))
        if getattr(self, "limit_line", None) is not None:
            limit_value = self._plot_limit_value()
            self.limit_line.set_ydata([limit_value, limit_value])
            self.limit_line.set_label("Limit Airmass" if self._plot_airmass else "Limit Altitude")

        self._configure_main_plot_y_axis()
        plot_text = self._theme_color("plot_text", "#d7e4f0")
        self.ax_alt.xaxis.label.set_color(plot_text)
        self.ax_alt.yaxis.label.set_color(plot_text)
        self.ax_alt.tick_params(axis="x", colors=plot_text)
        self.ax_alt.tick_params(axis="y", colors=plot_text)
        self.plot_canvas.draw_idle()

    def _visibility_web_render_signature(
        self,
        data: dict,
        *,
        now_override: Optional[datetime] = None,
    ) -> str:
        current_now = now_override if isinstance(now_override, datetime) else data.get("now_local")
        now_key = (
            current_now.strftime("%Y-%m-%d %H:%M")
            if isinstance(current_now, datetime)
            else ""
        )
        target_signature = [
            (
                str(tgt.name),
                _normalized_css_color(getattr(tgt, "plot_color", "")),
                bool(self.table_model.row_enabled[idx]) if idx < len(self.table_model.row_enabled) else True,
            )
            for idx, tgt in enumerate(self.targets)
        ]
        signature_payload = {
            "context": str(data.get("site_key", "") or self._current_visibility_context_key()),
            "mode": "airmass" if self._plot_airmass else "altitude",
            "theme": str(getattr(self, "_theme_name", DEFAULT_UI_THEME)),
            "dark": bool(getattr(self, "_dark_enabled", False)),
            "date": self.date_edit.date().toString("yyyy-MM-dd") if hasattr(self, "date_edit") else "",
            "sun": bool(self.sun_check.isChecked()) if hasattr(self, "sun_check") else True,
            "moon": bool(self.moon_check.isChecked()) if hasattr(self, "moon_check") else True,
            "targets": target_signature,
            "now": now_key,
        }
        serialized = json.dumps(signature_payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(serialized.encode("utf-8")).hexdigest()

    def _store_visibility_web_html_cache(self, cache_key: str, html: str) -> None:
        cache = getattr(self, "_visibility_web_html_cache", {})
        cache[str(cache_key)] = str(html)
        while len(cache) > 8:
            cache.pop(next(iter(cache)))
        self._visibility_web_html_cache = cache

    def _schedule_visibility_plot_refresh(self, *, delay_ms: int = 0) -> None:
        if not isinstance(getattr(self, "last_payload", None), dict):
            return
        if getattr(self, "_use_visibility_web", False):
            self._ensure_visibility_plot_widgets()
        self._visibility_plot_refresh_timer.start(max(0, int(delay_ms)))

    @Slot()
    def _flush_visibility_plot_refresh(self) -> None:
        if isinstance(getattr(self, "last_payload", None), dict):
            self._render_visibility_web_plot(self.last_payload)

    def _apply_visibility_web_selection_style(self, selected_names: Optional[set[str]] = None) -> None:
        if not bool(getattr(self, "_use_visibility_web", False)):
            return
        if not bool(getattr(self, "_visibility_web_has_content", False)):
            return
        web_view = getattr(self, "visibility_web", None)
        if web_view is None:
            return
        page = web_view.page() if hasattr(web_view, "page") else None
        if page is None:
            return
        if selected_names is None:
            selected_names = set(self._selected_target_names())
        selected_list = sorted(str(name) for name in selected_names if str(name).strip())
        selected_json = json.dumps(selected_list, ensure_ascii=False)
        script = (
            "(function(){"
            f"const selected = new Set({selected_json});"
            "const gd=document.querySelector('.plotly-graph-div');"
            "if(!gd||!window.Plotly||!Array.isArray(gd.data)){return 0;}"
            "const indices=[];const widths=[];const opacities=[];"
            "for(let i=0;i<gd.data.length;i++){"
            "const tr=gd.data[i]||{};const meta=tr.meta;"
            "if(!meta||typeof meta!=='object'||meta.kind!=='target'){continue;}"
            "const name=String(meta.target||'');if(!name){continue;}"
            "const seg=String(meta.segment||'');"
            "const isSelected=selected.has(name);"
            "if(seg==='base'){indices.push(i);widths.push(1.4);opacities.push(isSelected?0.40:0.28);}"
            "else if(seg==='high'){indices.push(i);widths.push(isSelected?2.8:1.9);opacities.push(isSelected?1.00:0.92);}"
            "}"
            "if(!indices.length){return 0;}"
            "Plotly.restyle(gd, {'line.width': widths, 'opacity': opacities}, indices);"
            "return indices.length;"
            "})();"
        )
        try:
            page.runJavaScript(script)
        except Exception:
            return

    def _apply_visibility_line_style(self, line: object, *, is_over: bool, is_selected: bool) -> None:
        if line is None:
            return
        try:
            line.set_solid_capstyle("round")
            line.set_solid_joinstyle("round")
            line.set_linewidth(2.3 if (is_over and is_selected) else 1.4)
            line.set_alpha(1.0 if (is_over and is_selected) else (0.7 if is_over else 0.3))
            if is_over and is_selected:
                line_color = QColor(str(getattr(line, "get_color", lambda: "")()))
                if not line_color.isValid():
                    line_color = self._theme_qcolor("accent_secondary", "#ff4fd8")
                line.set_path_effects(
                    [
                        mpl_patheffects.Stroke(
                            linewidth=16.0,
                            foreground=self._qcolor_rgba_mpl(line_color, 0.012),
                        ),
                        mpl_patheffects.Stroke(
                            linewidth=13.7,
                            foreground=self._qcolor_rgba_mpl(line_color, 0.020),
                        ),
                        mpl_patheffects.Stroke(
                            linewidth=11.5,
                            foreground=self._qcolor_rgba_mpl(line_color, 0.032),
                        ),
                        mpl_patheffects.Stroke(
                            linewidth=9.2,
                            foreground=self._qcolor_rgba_mpl(line_color, 0.048),
                        ),
                        mpl_patheffects.Stroke(
                            linewidth=7.2,
                            foreground=self._qcolor_rgba_mpl(line_color, 0.070),
                        ),
                        mpl_patheffects.Stroke(
                            linewidth=5.6,
                            foreground=self._qcolor_rgba_mpl(line_color, 0.102),
                        ),
                        mpl_patheffects.Stroke(
                            linewidth=4.3,
                            foreground=self._qcolor_rgba_mpl(line_color, 0.142),
                        ),
                        mpl_patheffects.Normal(),
                    ]
                )
            else:
                line.set_path_effects([])
        except Exception:
            return

    def _build_visibility_plotly_html(
        self,
        data: dict,
        *,
        now_override: Optional[datetime] = None,
    ) -> Optional[str]:
        if not (_HAS_PLOTLY and plotly_to_html is not None and go is not None):
            return None
        try:
            tz = pytz.timezone(str(data.get("tz", "UTC") or "UTC"))
        except Exception:
            tz = pytz.UTC
        try:
            times = [t.astimezone(tz) for t in mdates.num2date(data["times"])]
        except Exception:
            return None
        if len(times) < 2:
            return None

        use_tokens = getattr(
            self,
            "_theme_tokens",
            resolve_theme_tokens(
                getattr(self, "_theme_name", DEFAULT_UI_THEME),
                dark_enabled=bool(getattr(self, "_dark_enabled", False)),
            ),
        )
        dark = bool(getattr(self, "_dark_enabled", False))
        plot_bg = str(use_tokens.get("plot_bg", "#121b29"))
        plot_panel_bg = str(use_tokens.get("plot_panel_bg", plot_bg))
        plot_text = str(use_tokens.get("plot_text", "#d7e4f0" if dark else "#253347"))
        plot_grid = str(use_tokens.get("plot_grid", "#2f4666" if dark else "#d6deea"))
        plot_guide = str(use_tokens.get("plot_guide", plot_grid))
        plot_limit = str(use_tokens.get("plot_limit", "#ff5d8f"))
        plot_now = str(use_tokens.get("plot_now", "#ff3df0"))
        plot_sun = str(use_tokens.get("plot_sun", "#ffb224"))
        plot_moon = str(use_tokens.get("plot_moon", "#d7e2ff"))
        civil_col = str(use_tokens.get("plot_twilight_civil", "#ffd166"))
        naut_col = str(use_tokens.get("plot_twilight_naut", "#5ab6ff"))
        astro_col = str(use_tokens.get("plot_twilight_astro", "#8094c8"))
        plot_font = _plot_font_css_stack(use_tokens)

        limit = float(self.limit_spin.value())
        sun_alt_limit = self._sun_alt_limit()
        sun_alt_series = np.array(data.get("sun_alt", np.full(len(times), np.nan)), dtype=float)
        obs_sun_mask = np.isfinite(sun_alt_series) & (sun_alt_series <= sun_alt_limit)
        line_palette = self._line_palette()
        target_colors = [
            self._target_plot_color_css(tgt, idx, line_palette)
            for idx, tgt in enumerate(self.targets)
        ]
        row_enabled = list(getattr(self.table_model, "row_enabled", []))
        if len(row_enabled) != len(self.targets):
            row_enabled = [True] * len(self.targets)
        start_dt, end_dt, event_map = self._visibility_time_window(data, tz)
        grid_css = self._visibility_grid_color(alpha=0.42)
        guide_css = self._visibility_grid_color(alpha=0.24)

        fig = go.Figure()
        y_title = "Airmass" if self._plot_airmass else "Altitude (°)"
        y_precision = ".2f" if self._plot_airmass else ".1f"

        def _visible_series(values: np.ndarray, mask: np.ndarray) -> list[float]:
            series = np.array(values, copy=True, dtype=float)
            series[~mask] = np.nan
            return [float(v) if np.isfinite(v) else np.nan for v in series]

        for idx, tgt in enumerate(self.targets):
            if idx >= len(row_enabled) or not row_enabled[idx]:
                continue
            row = data.get(tgt.name)
            if not isinstance(row, dict):
                continue
            alt = np.array(row.get("altitude", np.full(len(times), np.nan)), dtype=float)
            if alt.shape[0] != len(times):
                continue
            color = target_colors[idx] if idx < len(target_colors) else self._target_plot_color_css(tgt, idx, line_palette)
            vis_mask = np.isfinite(alt) & (alt > 0.0)
            if not vis_mask.any():
                continue
            base_y = _visible_series(self._plot_y_values(alt), vis_mask)
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=base_y,
                    mode="lines",
                    name=tgt.name,
                    showlegend=False,
                    hoverinfo="skip",
                    line={"color": color, "width": 1.4, "dash": "dash"},
                    opacity=0.28,
                    meta={"kind": "target", "target": tgt.name, "segment": "base"},
                )
            )

            high_mask = np.isfinite(alt) & (alt >= limit) & obs_sun_mask
            if high_mask.any():
                high_y = _visible_series(self._plot_y_values(alt), high_mask)
                fig.add_trace(
                    go.Scatter(
                        x=times,
                        y=high_y,
                        mode="lines",
                        name=tgt.name,
                        showlegend=False,
                        line={"color": color, "width": 1.9},
                        opacity=0.92,
                        hovertemplate=f"{tgt.name}<br>%{{x|%H:%M}}<br>{y_title}: %{{y:{y_precision}}}<extra></extra>",
                        meta={"kind": "target", "target": tgt.name, "segment": "high"},
                    )
                )

        for start_key, end_key, color in (
            ("sunset", "dusk_civ", civil_col),
            ("dusk_civ", "dusk_naut", naut_col),
            ("dusk_naut", "dusk", astro_col),
            ("dawn", "dawn_naut", astro_col),
            ("dawn_naut", "dawn_civ", naut_col),
            ("dawn_civ", "sunrise", civil_col),
        ):
            start_evt = event_map.get(start_key)
            end_evt = event_map.get(end_key)
            if start_evt is None or end_evt is None or start_evt >= end_evt:
                continue
            x0 = max(start_evt, start_dt)
            x1 = min(end_evt, end_dt)
            if x0 >= x1:
                continue
            fig.add_vrect(x0=x0, x1=x1, fillcolor=color, opacity=0.22, line_width=0, layer="below")

        for _key, evt in event_map.items():
            if start_dt <= evt <= end_dt:
                fig.add_vline(x=evt, line={"color": guide_css, "width": 1, "dash": "dash"}, opacity=1.0)

        limit_value = self._plot_limit_value()
        fig.add_hline(y=limit_value, line={"color": plot_limit, "width": 1.0}, opacity=0.55)

        if self.sun_check.isChecked() and "sun_alt" in data:
            sun_y = [float(v) if np.isfinite(v) else np.nan for v in self._plot_y_values(data["sun_alt"])]
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=sun_y,
                    mode="lines",
                    name="Sun",
                    showlegend=False,
                    line={"color": plot_sun, "width": 1.2},
                    opacity=0.82,
                    hovertemplate=f"Sun<br>%{{x|%H:%M}}<br>{y_title}: %{{y:{y_precision}}}<extra></extra>",
                )
            )
        if self.moon_check.isChecked() and "moon_alt" in data:
            moon_y = [float(v) if np.isfinite(v) else np.nan for v in self._plot_y_values(data["moon_alt"])]
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=moon_y,
                    mode="lines",
                    name="Moon",
                    showlegend=False,
                    line={"color": plot_moon, "width": 1.2},
                    opacity=0.84,
                    hovertemplate=f"Moon<br>%{{x|%H:%M}}<br>{y_title}: %{{y:{y_precision}}}<extra></extra>",
                )
            )

        now_dt = now_override
        if not isinstance(now_dt, datetime):
            payload_now = data.get("now_local")
            now_dt = payload_now if isinstance(payload_now, datetime) else datetime.now(tz)
        if now_dt.tzinfo is None:
            now_dt = tz.localize(now_dt)
        else:
            now_dt = now_dt.astimezone(tz)
        fig.add_vline(x=now_dt, line={"color": plot_now, "width": 1.4, "dash": "dot"})

        date_label = self.date_edit.date().toString("yyyy-MM-dd") if hasattr(self, "date_edit") else now_dt.strftime("%Y-%m-%d")
        fig.update_layout(
            template="plotly_dark" if dark else "plotly_white",
            title={"text": f"Date: {date_label}", "x": 0.5, "xanchor": "center"},
            margin={"l": 56, "r": 20, "t": 48, "b": 52},
            hovermode="x unified",
            paper_bgcolor=plot_bg,
            plot_bgcolor=plot_panel_bg,
            font={"color": plot_text, "family": plot_font},
            showlegend=False,
            dragmode="pan",
        )
        fig.update_xaxes(
            range=[start_dt, end_dt],
            showgrid=True,
            gridcolor=grid_css,
            zerolinecolor=grid_css,
            color=plot_text,
            tickformat="%H:%M",
            title_text="Time (local)",
        )
        if not self._plot_airmass:
            fig.update_yaxes(
                title_text="Altitude (°)",
                range=[0, 90],
                tickvals=[0, 15, 30, 45, 60, 75, 90],
                showgrid=True,
                gridcolor=grid_css,
                zerolinecolor=grid_css,
                color=plot_text,
            )
        else:
            fig.update_yaxes(
                title_text="Airmass",
                range=[VISIBILITY_AIRMASS_Y_MAX, VISIBILITY_AIRMASS_Y_MIN],
                tickvals=VISIBILITY_AIRMASS_Y_TICKS,
                ticktext=[f"{tick:.1f}" for tick in VISIBILITY_AIRMASS_Y_TICKS],
                showgrid=True,
                gridcolor=grid_css,
                zerolinecolor=grid_css,
                color=plot_text,
            )

        html_fragment = plotly_to_html(
            fig,
            include_plotlyjs=False,
            full_html=False,
            config={
                "displaylogo": False,
                "responsive": True,
                "scrollZoom": True,
                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            },
        )
        html_fragment = html_fragment.replace("<div>", "<div id='plot-host' style='width:100%;height:100%;min-height:100%;'>", 1)
        graph_id_match = re.search(r'<div id="([^"]+)" class="plotly-graph-div"', html_fragment)
        graph_id = graph_id_match.group(1) if graph_id_match else ""
        resize_script = ""
        if graph_id:
            resize_script = (
                "<script>"
                "(function(){"
                f"const gd=document.getElementById('{graph_id}');"
                "if(!gd||!window.Plotly){return;}"
                "const resize=()=>{"
                "gd.style.width='100%';"
                "gd.style.height='100%';"
                "if(window.Plotly&&Plotly.Plots){Plotly.Plots.resize(gd);}"
                "};"
                "window.addEventListener('resize', resize);"
                "if(window.ResizeObserver){new ResizeObserver(resize).observe(document.body);}"
                "setTimeout(resize,0);"
                "setTimeout(resize,120);"
                "})();"
                "</script>"
            )
        font_face_css = _embedded_display_font_css()
        html_head = (
            "<!doctype html><html><head><meta charset='utf-8'>"
            "<style>"
            f"{font_face_css}"
            f"html,body,#plot-host{{margin:0;width:100%;height:100%;overflow:hidden;background:{plot_bg};}}"
            f"html,body,#plot-host,.plotly-graph-div{{font-family:{plot_font};}}"
            ".plotly-graph-div{width:100%!important;height:100%!important;}"
            "</style>"
        )
        if _PLOTLY_JS_BASE_DIR:
            return (
                f"{html_head}<script src='plotly.min.js'></script></head><body>"
                f"{html_fragment}{resize_script}</body></html>"
            )
        return (
            f"{html_head}<script src='https://cdn.plot.ly/plotly-3.4.0.min.js'></script></head><body>"
            f"{html_fragment}{resize_script}</body></html>"
        )

    def _render_visibility_web_plot(
        self,
        data: Optional[dict] = None,
        *,
        now_override: Optional[datetime] = None,
    ) -> None:
        payload = data if isinstance(data, dict) else getattr(self, "last_payload", None)
        if not isinstance(payload, dict):
            return
        if getattr(self, "_use_visibility_web", False):
            self._ensure_visibility_plot_widgets()
        web_view = getattr(self, "visibility_web", None)
        if web_view is None:
            return
        cache_key = self._visibility_web_render_signature(payload, now_override=now_override)
        html = self._visibility_web_html_cache.get(cache_key)
        if not html:
            html = self._build_visibility_plotly_html(payload, now_override=now_override)
            if html:
                self._store_visibility_web_html_cache(cache_key, html)
        if not html:
            self._visibility_web_has_content = False
            self._set_visibility_loading_state("Unable to render interactive chart.", visible=True)
            return
        if not self._visibility_web_has_content:
            self._set_visibility_loading_state("Rendering interactive chart…", visible=True)
        if _PLOTLY_JS_BASE_DIR:
            web_view.setHtml(html, QUrl.fromLocalFile(str(_PLOTLY_JS_BASE_DIR) + "/"))
            return
        web_view.setHtml(html)

    def _refresh_target_color_map(self, palette: Optional[list[str]] = None):
        use_palette = palette if palette is not None else self._line_palette()
        self.table_model.color_map.clear()
        for idx, tgt in enumerate(self.targets):
            self.table_model.color_map[tgt.name] = QColor(
                self._target_plot_color_css(tgt, idx, use_palette)
            )

    def _cutout_render_dimensions_px(self, label: Optional[QLabel] = None) -> tuple[int, int]:
        base = _sanitize_cutout_size_px(getattr(self, "_cutout_size_px", CUTOUT_DEFAULT_SIZE_PX))
        probe = label if label is not None else getattr(self, "aladin_image_label", None)
        w = int(getattr(probe, "width", lambda: 0)()) if probe is not None else 0
        h = int(getattr(probe, "height", lambda: 0)()) if probe is not None else 0
        if (w < 32 or h < 32) and hasattr(self, "cutout_tabs"):
            tw = int(self.cutout_tabs.width())
            th = int(self.cutout_tabs.height() - self.cutout_tabs.tabBar().height())
            if tw > 0 and th > 0:
                w = max(w, tw)
                h = max(h, th)
        if w < 32 or h < 32:
            return base, base

        ratio = max(0.4, min(2.5, float(w) / float(h)))
        if ratio >= 1.0:
            out_h = base
            out_w = int(round(base * ratio))
        else:
            out_w = base
            out_h = int(round(base / ratio))

        max_side = 1400
        largest = max(out_w, out_h)
        if largest > max_side:
            scale = max_side / float(largest)
            out_w = int(round(out_w * scale))
            out_h = int(round(out_h * scale))
        step = 8
        out_w = max(128, int(round(out_w / step) * step))
        out_h = max(128, int(round(out_h / step) * step))
        return out_w, out_h

    def _cutout_fov_axes_arcmin(self, width_px: int, height_px: int) -> tuple[float, float]:
        base = max(1.0, float(self._cutout_fov_arcmin))
        if width_px <= 0 or height_px <= 0:
            return base, base
        ratio = float(width_px) / float(height_px)
        if ratio >= 1.0:
            fov_y = base
            fov_x = base * ratio
        else:
            fov_x = base
            fov_y = base / ratio
        return fov_x, fov_y

    def _aladin_fetch_fov_axes_arcmin(self, width_px: int, height_px: int) -> tuple[float, float]:
        fov_x, fov_y = self._cutout_fov_axes_arcmin(width_px, height_px)
        margin = max(1.0, float(CUTOUT_ALADIN_FETCH_MARGIN))
        fetch_x = float(fov_x) * margin
        fetch_y = float(fov_y) * margin
        min_short_axis = self._aladin_fetch_min_short_axis_arcmin()
        if width_px > 0 and height_px > 0:
            ratio = max(0.25, min(4.0, float(width_px) / float(height_px)))
            if ratio >= 1.0:
                # Height is the shorter axis in landscape.
                fetch_y = max(fetch_y, min_short_axis)
                fetch_x = max(fetch_x, fetch_y * ratio)
            else:
                # Width is the shorter axis in portrait.
                fetch_x = max(fetch_x, min_short_axis)
                fetch_y = max(fetch_y, fetch_x / ratio)
        else:
            fetch_x = max(fetch_x, min_short_axis)
            fetch_y = max(fetch_y, min_short_axis)
        context_factor = max(1.0, float(getattr(self, "_aladin_context_factor", 1.0)))
        if context_factor > 1.0:
            fetch_x *= context_factor
            fetch_y *= context_factor
        return fetch_x, fetch_y

    def _aladin_fetch_min_short_axis_arcmin(self) -> float:
        fallback = max(1.0, float(CUTOUT_ALADIN_FETCH_MIN_ARCMIN))
        tel_fov = self._site_telescope_fov_arcmin()
        if tel_fov is None:
            return fallback
        tel_short = min(float(tel_fov[0]), float(tel_fov[1]))
        if not math.isfinite(tel_short) or tel_short <= 0.0:
            return fallback
        dynamic = tel_short * max(1.0, float(CUTOUT_ALADIN_FETCH_TELESCOPE_MARGIN))
        dynamic = max(float(CUTOUT_MIN_FOV_ARCMIN), dynamic)
        dynamic = min(float(CUTOUT_ALADIN_FETCH_TELESCOPE_MAX_ARCMIN), dynamic)
        return dynamic

    def _aladin_fetch_dimensions_px(self, width_px: int, height_px: int) -> tuple[int, int]:
        w = max(1, int(width_px))
        h = max(1, int(height_px))
        short_axis = max(1, min(w, h))
        min_short_px = max(256, int(CUTOUT_ALADIN_FETCH_MIN_SHORT_PX))
        scale = max(
            1.0,
            float(CUTOUT_ALADIN_FETCH_RES_MULT),
            float(min_short_px) / float(short_axis),
        )
        out_w = max(64, int(round(float(w) * scale)))
        out_h = max(64, int(round(float(h) * scale)))
        max_edge = max(min_short_px, int(CUTOUT_ALADIN_FETCH_MAX_EDGE_PX))
        largest = max(out_w, out_h)
        if largest > max_edge:
            down = float(max_edge) / float(largest)
            out_w = max(64, int(round(float(out_w) * down)))
            out_h = max(64, int(round(float(out_h) * down)))
        step = 8
        out_w = max(64, int(round(out_w / step) * step))
        out_h = max(64, int(round(out_h / step) * step))
        return out_w, out_h

    def _site_telescope_fov_arcmin(self, site: Optional[Site] = None) -> Optional[tuple[float, float]]:
        site_obj = site
        if site_obj is None:
            if hasattr(self, "obs_combo") and hasattr(self, "observatories"):
                site_obj = self.observatories.get(self.obs_combo.currentText())
            if site_obj is None and hasattr(self, "table_model"):
                site_obj = self.table_model.site
        if site_obj is None:
            return None
        fov = site_obj.fov_arcmin
        if fov is None:
            return None
        fov_x = _safe_float(fov[0])
        fov_y = _safe_float(fov[1])
        if fov_x is None or fov_y is None:
            return None
        if not math.isfinite(fov_x) or not math.isfinite(fov_y) or fov_x <= 0.0 or fov_y <= 0.0:
            return None
        return float(fov_x), float(fov_y)

    def _telescope_overlay_signature(self, site: Optional[Site] = None) -> str:
        fov = self._site_telescope_fov_arcmin(site)
        if fov is None:
            return "none"
        return f"{fov[0]:.3f}x{fov[1]:.3f}"

    def _fit_cutout_base_fov_to_telescope(self, tel_fov_x: float, tel_fov_y: float, width_px: int, height_px: int) -> int:
        ratio = 1.0
        if width_px > 0 and height_px > 0:
            ratio = max(0.25, min(4.0, float(width_px) / float(height_px)))
        if ratio >= 1.0:
            required_base = max(float(tel_fov_y), float(tel_fov_x) / ratio)
        else:
            required_base = max(float(tel_fov_x), float(tel_fov_y) * ratio)
        return _sanitize_cutout_fov_arcmin(int(round(required_base)))

    def _sync_cutout_fov_to_site(self, site: Optional[Site] = None, persist: bool = False) -> bool:
        tel_fov = self._site_telescope_fov_arcmin(site)
        if tel_fov is None:
            return False
        render_w, render_h = self._cutout_render_dimensions_px(getattr(self, "aladin_image_label", None))
        fitted = self._fit_cutout_base_fov_to_telescope(tel_fov[0], tel_fov[1], render_w, render_h)
        if fitted == int(self._cutout_fov_arcmin):
            return False
        self._cutout_fov_arcmin = int(fitted)
        if persist:
            self.settings.setValue("general/cutoutFovArcmin", int(self._cutout_fov_arcmin))
        return True

    def _telescope_overlay_rect(
        self,
        width_px: int,
        height_px: int,
        margin_px: int = 0,
        fov_axes: Optional[tuple[float, float]] = None,
    ) -> Optional[tuple[int, int, int, int]]:
        tel_fov = self._site_telescope_fov_arcmin()
        if tel_fov is None:
            return None
        if fov_axes is None:
            cutout_fov_x, cutout_fov_y = self._cutout_fov_axes_arcmin(width_px, height_px)
        else:
            cutout_fov_x, cutout_fov_y = float(fov_axes[0]), float(fov_axes[1])
        if cutout_fov_x <= 0.0 or cutout_fov_y <= 0.0:
            return None
        rel_w = max(0.0, min(1.0, float(tel_fov[0]) / float(cutout_fov_x)))
        rel_h = max(0.0, min(1.0, float(tel_fov[1]) / float(cutout_fov_y)))
        if rel_w <= 0.0 or rel_h <= 0.0:
            return None
        safe_margin = max(0, int(margin_px))
        avail_w = max(8, int(width_px) - 2 * safe_margin)
        avail_h = max(8, int(height_px) - 2 * safe_margin)
        rect_w = max(8, int(round(float(avail_w) * rel_w)))
        rect_h = max(8, int(round(float(avail_h) * rel_h)))
        rect_w = min(rect_w, avail_w)
        rect_h = min(rect_h, avail_h)
        x0 = safe_margin + max(0, (avail_w - rect_w) // 2)
        y0 = safe_margin + max(0, (avail_h - rect_h) // 2)
        return x0, y0, rect_w, rect_h

    def _paint_telescope_fov_overlay(
        self,
        painter: QPainter,
        w: int,
        h: int,
        fill: bool = True,
        color: Optional[QColor] = None,
        offset_x: int = 0,
        offset_y: int = 0,
        fov_axes: Optional[tuple[float, float]] = None,
        min_margin_px: int = 4,
    ) -> None:
        pen_width = max(1, int(min(w, h) * 0.005))
        margin = max(int(min_margin_px), int(math.ceil(pen_width * 0.8)))
        overlay = self._telescope_overlay_rect(w, h, margin_px=margin, fov_axes=fov_axes)
        if overlay is None:
            return
        x0, y0, rw, rh = overlay
        overlay_color = color or self._theme_qcolor("overlay_fov", "#59f3ff")
        overlay_color.setAlpha(180 if color is None else overlay_color.alpha())
        pen = QPen(overlay_color)
        pen.setWidth(pen_width)
        pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        if fill:
            fill_color = QColor(overlay_color)
            fill_color.setAlpha(24)
            painter.setBrush(fill_color)
        else:
            painter.setBrush(Qt.BrushStyle.NoBrush)
        # Qt drawRect includes the right/bottom border pixels; shrink by 1 px
        # so dashed edges are not clipped when overlay touches panel bounds.
        draw_w = max(2, int(rw) - 1)
        draw_h = max(2, int(rh) - 1)
        painter.drawRect(int(x0 + offset_x), int(y0 + offset_y), draw_w, draw_h)

    def _cutout_fov_text(
        self,
        width_px: int,
        height_px: int,
        fetch_margin: bool = False,
        fov_axes: Optional[tuple[float, float]] = None,
    ) -> str:
        if fov_axes is not None:
            fov_x, fov_y = float(fov_axes[0]), float(fov_axes[1])
        elif fetch_margin:
            fov_x, fov_y = self._aladin_fetch_fov_axes_arcmin(width_px, height_px)
        else:
            fov_x, fov_y = self._cutout_fov_axes_arcmin(width_px, height_px)
        return f"{fov_x:.1f}x{fov_y:.1f} arcmin"

    def _cutout_key_for_target(self, target: Target, width_px: int, height_px: int) -> str:
        fetch_w, fetch_h = self._aladin_fetch_dimensions_px(width_px, height_px)
        return (
            f"{self._cutout_survey_key}:{self._cutout_fov_arcmin}:{width_px}x{height_px}:"
            f"fetch{fetch_w}x{fetch_h}:"
            f"ctx{float(getattr(self, '_aladin_context_factor', 1.0)):.2f}:"
            f"fetchm{CUTOUT_ALADIN_FETCH_MARGIN:.2f}:minfov{CUTOUT_ALADIN_FETCH_MIN_ARCMIN:.1f}:"
            f"telfetchm{CUTOUT_ALADIN_FETCH_TELESCOPE_MARGIN:.2f}:"
            f"telfetchmax{CUTOUT_ALADIN_FETCH_TELESCOPE_MAX_ARCMIN:.1f}:"
            f"res{CUTOUT_ALADIN_FETCH_RES_MULT:.2f}:minpx{CUTOUT_ALADIN_FETCH_MIN_SHORT_PX}:maxpx{CUTOUT_ALADIN_FETCH_MAX_EDGE_PX}:"
            f"{self._telescope_overlay_signature()}:"
            f"{target.ra:.6f},{target.dec:.6f}"
        )

    def _aladin_visible_image_rect(
        self,
        widget_w: int,
        widget_h: int,
        image_x: int,
        image_y: int,
        image_w: int,
        image_h: int,
    ) -> tuple[int, int, int, int]:
        if widget_w <= 0 or widget_h <= 0:
            return 0, 0, 1, 1
        if image_w <= 0 or image_h <= 0:
            return 0, 0, int(widget_w), int(widget_h)
        left = max(0, int(image_x))
        top = max(0, int(image_y))
        right = min(int(widget_w), int(image_x + image_w))
        bottom = min(int(widget_h), int(image_y + image_h))
        if right <= left or bottom <= top:
            return 0, 0, int(widget_w), int(widget_h)
        return left, top, int(right - left), int(bottom - top)

    def _aladin_visible_fov_axes_arcmin(
        self,
        widget_w: int,
        widget_h: int,
        image_x: int,
        image_y: int,
        image_w: int,
        image_h: int,
    ) -> tuple[float, float]:
        base_x, base_y = self._aladin_fetch_fov_axes_arcmin(image_w, image_h)
        _ = (image_x, image_y)
        # FOV should react to zoom level; panning must not change FOV value.
        frac_x = max(0.02, min(1.0, float(max(1, widget_w)) / float(max(1, image_w))))
        frac_y = max(0.02, min(1.0, float(max(1, widget_h)) / float(max(1, image_h))))
        return float(base_x) * frac_x, float(base_y) * frac_y

    def _cutout_resize_signature_for_target(self, target: Optional[Target]) -> Optional[tuple]:
        if target is None:
            return None
        aw, ah = self._cutout_render_dimensions_px(getattr(self, "aladin_image_label", None))
        fw, fh = self._cutout_render_dimensions_px(getattr(self, "finder_image_label", None))
        return (
            target.name.strip().lower(),
            self._cutout_view_key,
            self._cutout_survey_key,
            int(self._cutout_fov_arcmin),
            round(float(getattr(self, "_aladin_context_factor", 1.0)), 3),
            self._telescope_overlay_signature(),
            int(aw),
            int(ah),
            int(fw),
            int(fh),
        )

    @Slot(int, int)
    def _schedule_cutout_resize_refresh(self, *_):
        if getattr(self, "_shutting_down", False):
            return
        timer = getattr(self, "_cutout_resize_timer", None)
        if timer is not None:
            timer.start()

    @Slot()
    def _on_cutout_resize_timeout(self):
        if getattr(self, "_shutting_down", False):
            return
        target = self._selected_target_or_none()
        if target is None:
            return
        fov_changed = self._sync_cutout_fov_to_site()
        sig = self._cutout_resize_signature_for_target(target)
        if sig is None:
            return
        if (not fov_changed) and sig == self._cutout_last_resize_signature:
            return
        self._cutout_last_resize_signature = sig
        self._update_cutout_preview_for_target(target)

    def _set_cutout_placeholder(self, text: str):
        if not hasattr(self, "aladin_image_label"):
            return
        for kind, label in (
            ("aladin", self.aladin_image_label),
            ("finder", getattr(self, "finder_image_label", None)),
        ):
            if label is None:
                continue
            label.setPixmap(QPixmap())
            label.setText(text)
            self._set_cutout_image_loading(kind, text, visible=False)

    def _paint_aladin_static_overlay(
        self,
        painter: QPainter,
        w: int,
        h: int,
        image_x: Optional[int] = None,
        image_y: Optional[int] = None,
        image_w: Optional[int] = None,
        image_h: Optional[int] = None,
    ) -> None:
        painter.setRenderHint(QPainter.Antialiasing, True)
        iw = int(image_w if image_w is not None else w)
        ih = int(image_h if image_h is not None else h)
        ix = int(image_x if image_x is not None else 0)
        iy = int(image_y if image_y is not None else 0)
        _ = (ix, iy)

        # Keep crosshair + telescope FOV fixed in viewport center.
        cx = w // 2
        cy = h // 2
        radius = max(10, int(min(w, h) * 0.14))

        # Crosshair and center ring for quick visual centering.
        crosshair = self._theme_qcolor("overlay_crosshair", "#ff5d8f")
        crosshair.setAlpha(225)
        pen_cross = QPen(crosshair)
        pen_cross.setWidth(max(1, int(min(w, h) * 0.01)))
        painter.setPen(pen_cross)
        span = max(14, int(min(w, h) * 0.18))
        painter.drawLine(cx - span, cy, cx + span, cy)
        painter.drawLine(cx, cy - span, cx, cy + span)
        ring_color = self._theme_qcolor("overlay_strip_text", "#eef4fc")
        ring_color.setAlpha(210)
        pen_ring = QPen(ring_color)
        pen_ring.setWidth(max(1, int(min(w, h) * 0.008)))
        painter.setPen(pen_ring)
        painter.drawEllipse(cx - radius, cy - radius, radius * 2, radius * 2)
        overlay_fov_axes = self._aladin_visible_fov_axes_arcmin(
            w,
            h,
            ix,
            iy,
            iw,
            ih,
        )
        self._paint_telescope_fov_overlay(
            painter,
            w,
            h,
            fov_axes=overlay_fov_axes,
        )

        # Keep survey/FOV visible directly on Aladin image.
        strip_h = max(16, int(h * 0.10))
        strip_y = h - strip_h
        strip_bg = self._theme_qcolor("overlay_strip_bg", "#08111d")
        strip_bg.setAlpha(165)
        painter.fillRect(0, strip_y, w, strip_h, strip_bg)
        strip_text = self._theme_qcolor("overlay_strip_text", "#eef4fc")
        strip_text.setAlpha(240)
        painter.setPen(strip_text)
        meta_txt = f"{_cutout_survey_label(self._cutout_survey_key)} | {self._cutout_fov_text(w, h, fov_axes=overlay_fov_axes)}"
        painter.drawText(8, strip_y, max(8, w - 16), strip_h, Qt.AlignVCenter | Qt.AlignLeft, meta_txt)

    def _build_aladin_overlay_pixmap(self, source: QPixmap) -> QPixmap:
        if source.isNull():
            return source
        view = source.copy()
        painter = QPainter(view)
        self._paint_aladin_static_overlay(painter, view.width(), view.height())
        painter.end()
        return view

    def _build_finder_overlay_pixmap(self, source: QPixmap) -> QPixmap:
        if source.isNull():
            return source
        view = source.copy()
        painter = QPainter(view)
        painter.setRenderHint(QPainter.Antialiasing, True)
        w = view.width()
        h = view.height()
        cx = w // 2
        cy = h // 2

        # Centered finder reticle (manual, to avoid astroplan offset on non-square frames).
        crosshair = self._theme_qcolor("overlay_crosshair", "#ff5d8f")
        crosshair.setAlpha(225)
        pen_cross = QPen(crosshair)
        pen_cross.setWidth(max(1, int(min(w, h) * 0.010)))
        painter.setPen(pen_cross)
        span = max(16, int(min(w, h) * 0.10))
        gap = max(6, int(span * 0.34))
        painter.drawLine(cx - span, cy, cx - gap, cy)
        painter.drawLine(cx + gap, cy, cx + span, cy)
        painter.drawLine(cx, cy - span, cx, cy - gap)
        painter.drawLine(cx, cy + gap, cx, cy + span)
        self._paint_telescope_fov_overlay(
            painter,
            w,
            h,
            fill=False,
            color=self._theme_qcolor("overlay_crosshair", "#ff5d8f"),
            min_margin_px=8,
        )

        strip_h = max(16, int(h * 0.10))
        strip_bg = self._theme_qcolor("overlay_strip_bg", "#08111d")
        strip_bg.setAlpha(165)
        painter.fillRect(0, h - strip_h, w, strip_h, strip_bg)
        strip_text = self._theme_qcolor("overlay_strip_text", "#eef4fc")
        strip_text.setAlpha(240)
        painter.setPen(strip_text)
        painter.drawText(
            8,
            h - strip_h,
            w - 16,
            strip_h,
            Qt.AlignVCenter | Qt.AlignLeft,
            f"FOV: {self._cutout_fov_text(w, h)}",
        )
        painter.end()
        return view

    def _pixmap_to_png_bytes(self, pixmap: QPixmap) -> bytes:
        if pixmap.isNull():
            return b""
        payload = QByteArray()
        buffer = QBuffer(payload)
        if not buffer.open(QIODevice.WriteOnly):
            return b""
        try:
            ok = pixmap.save(buffer, "PNG")
        finally:
            buffer.close()
        return bytes(payload) if ok else b""

    def _load_pixmap_from_storage_cache(self, namespace: str, key: str) -> Optional[QPixmap]:
        storage = getattr(self, "app_storage", None)
        if storage is None or not key:
            return None
        try:
            payload = storage.cache.get_json(namespace, key)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to read %s image cache '%s': %s", namespace, key, exc)
            return None
        if not isinstance(payload, dict):
            return None
        image_bytes = payload.get("image_bytes")
        if not isinstance(image_bytes, (bytes, bytearray)):
            return None
        pixmap = QPixmap()
        if not pixmap.loadFromData(bytes(image_bytes), "PNG") or pixmap.isNull():
            return None
        return pixmap

    def _persist_pixmap_to_storage_cache(self, namespace: str, key: str, pixmap: QPixmap) -> None:
        storage = getattr(self, "app_storage", None)
        if storage is None or not key or pixmap.isNull():
            return
        payload = self._pixmap_to_png_bytes(pixmap)
        if not payload:
            return
        try:
            storage.cache.set_json(namespace, key, {"image_bytes": payload}, ttl_s=14 * 24 * 60 * 60)
            storage.cache.prune_namespace(namespace, max_entries=160)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to persist %s image cache '%s': %s", namespace, key, exc)

    def _cache_cutout_pixmap(self, key: str, pixmap: QPixmap, *, persist: bool = True):
        self._cutout_cache[key] = pixmap
        if key in self._cutout_cache_order:
            self._cutout_cache_order.remove(key)
        self._cutout_cache_order.append(key)
        targets_n = len(getattr(self, "targets", []))
        cache_limit = max(CUTOUT_CACHE_MAX, min(160, int(targets_n) + 24))
        while len(self._cutout_cache_order) > cache_limit:
            stale = self._cutout_cache_order.pop(0)
            self._cutout_cache.pop(stale, None)
        if persist:
            self._persist_pixmap_to_storage_cache("cutout_preview", key, pixmap)

    def _cache_finder_pixmap(self, key: str, pixmap: QPixmap, *, persist: bool = True):
        self._finder_cache[key] = pixmap
        if key in self._finder_cache_order:
            self._finder_cache_order.remove(key)
        self._finder_cache_order.append(key)
        targets_n = len(getattr(self, "targets", []))
        cache_limit = max(CUTOUT_CACHE_MAX, min(160, int(targets_n) + 24))
        while len(self._finder_cache_order) > cache_limit:
            stale = self._finder_cache_order.pop(0)
            self._finder_cache.pop(stale, None)
        if persist:
            self._persist_pixmap_to_storage_cache("finder_preview", key, pixmap)

    def _find_cutout_cache_variant(self, target: Target) -> Optional[tuple[str, QPixmap]]:
        coord_suffix = f"{target.ra:.6f},{target.dec:.6f}"
        prefix = f"{self._cutout_survey_key}:{self._cutout_fov_arcmin}:"
        ctx_token = f"ctx{float(getattr(self, '_aladin_context_factor', 1.0)):.2f}:"
        overlay_token = f"{self._telescope_overlay_signature()}:"
        for key in reversed(self._cutout_cache_order):
            if not key.endswith(coord_suffix):
                continue
            if not key.startswith(prefix):
                continue
            if ctx_token not in key:
                continue
            if overlay_token not in key:
                continue
            pix = self._cutout_cache.get(key)
            if pix is None or pix.isNull():
                continue
            return key, pix
        return None

    def _find_finder_cache_variant(self, target: Target) -> Optional[tuple[str, QPixmap]]:
        coord_suffix = f"{target.ra:.6f},{target.dec:.6f}"
        prefix = f"{self._cutout_survey_key}:{self._cutout_fov_arcmin}:"
        overlay_token = f"{self._telescope_overlay_signature()}:"
        # Finder chart is independent from Aladin context zoom factor,
        # so allow reusing cached variants across ctx changes.
        for key in reversed(self._finder_cache_order):
            if not key.endswith(coord_suffix):
                continue
            if not key.startswith(prefix):
                continue
            if overlay_token not in key:
                continue
            pix = self._finder_cache.get(key)
            if pix is None or pix.isNull():
                continue
            return key, pix
        return None

    def _show_finder_aladin_fallback(self, key: str, text_if_missing: str) -> bool:
        if not hasattr(self, "finder_image_label"):
            return False
        # Keep Finder tab dedicated to finder output (no Aladin substitution).
        fallback = self._finder_cache.get(key)
        if fallback is not None and not fallback.isNull():
            self.finder_image_label.setText("")
            self.finder_image_label.setPixmap(fallback)
            self._set_cutout_image_loading("finder", "", visible=False)
            self._finder_displayed_key = key
            return True
        self.finder_image_label.setPixmap(QPixmap())
        self.finder_image_label.setText(text_if_missing)
        self._set_cutout_image_loading("finder", text_if_missing, visible=False)
        self._finder_displayed_key = ""
        return False

    def _ensure_aladin_pan_ready(self) -> None:
        if not hasattr(self, "aladin_image_label"):
            return
        label = self.aladin_image_label
        try:
            zoom_now = float(label.zoom_factor())
        except Exception:
            zoom_now = 1.0
        if abs(zoom_now - 1.0) > 1e-3:
            return
        label.set_zoom(float(CUTOUT_ALADIN_INITIAL_PAN_ZOOM))

    def _set_finder_status(self, text: str, busy: bool = False):
        if not hasattr(self, "status_finder_label") or not hasattr(self, "status_finder_progress"):
            return
        self.status_finder_label.setText(text)
        if busy:
            self.status_finder_progress.setRange(0, 0)
            self.status_finder_progress.show()
            return
        self.status_finder_progress.hide()
        self.status_finder_progress.setRange(0, 1)
        self.status_finder_progress.setValue(0)

    def _set_aladin_status(self, text: str, busy: bool = False):
        if not hasattr(self, "status_aladin_label") or not hasattr(self, "status_aladin_progress"):
            return
        self.status_aladin_label.setText(text)
        if busy:
            self.status_aladin_progress.setRange(0, 0)
            self.status_aladin_progress.show()
            return
        self.status_aladin_progress.hide()
        self.status_aladin_progress.setRange(0, 1)
        self.status_aladin_progress.setValue(0)

    def _finder_prefetch_done_status(self) -> None:
        total = max(int(getattr(self, "_finder_prefetch_total", 0)), int(getattr(self, "_finder_prefetch_completed", 0)))
        cached = max(0, int(getattr(self, "_finder_prefetch_cached", 0)))
        self._set_finder_status(
            f"Finder: prefetch done ({int(self._finder_prefetch_completed)}/{total}, {cached} cached)",
            busy=False,
        )

    def _aladin_prefetch_done_status(self) -> None:
        total = max(int(getattr(self, "_cutout_prefetch_total", 0)), int(getattr(self, "_cutout_prefetch_completed", 0)))
        cached = max(0, int(getattr(self, "_cutout_prefetch_cached", 0)))
        self._set_aladin_status(
            f"Aladin: prefetch done ({int(self._cutout_prefetch_completed)}/{total}, {cached} cached)",
            busy=False,
        )

    def _set_bhtom_status(self, text: str, busy: bool = False):
        if not hasattr(self, "status_bhtom_label") or not hasattr(self, "status_bhtom_progress"):
            return
        self.status_bhtom_label.setText(text)
        if busy:
            self.status_bhtom_progress.setRange(0, 0)
            self.status_bhtom_progress.show()
            return
        self.status_bhtom_progress.hide()
        self.status_bhtom_progress.setRange(0, 1)
        self.status_bhtom_progress.setValue(0)

    def _stop_finder_workers(self, aggressive: bool = False):
        workers = list(getattr(self, "_finder_workers", []))
        alive: list[FinderChartWorker] = []
        for worker in workers:
            try:
                if not shb.isValid(worker):
                    continue
            except Exception:
                continue

            running = False
            try:
                running = worker.isRunning()
            except Exception:
                running = False

            if running:
                try:
                    worker.requestInterruption()
                    worker.quit()
                except Exception:
                    pass
                if aggressive:
                    stopped = False
                    try:
                        stopped = worker.wait(5500)
                    except Exception:
                        stopped = False

            still_running = False
            try:
                still_running = shb.isValid(worker) and worker.isRunning()
            except Exception:
                still_running = False
            if still_running:
                alive.append(worker)
        self._finder_workers = alive
        if self._finder_worker is not None and self._finder_worker not in alive:
            self._finder_worker = None

    def _cancel_finder_chart_worker(self):
        had_pending = bool(self._finder_pending_key)
        if hasattr(self, "_finder_timeout_timer"):
            self._finder_timeout_timer.stop()
        self._stop_finder_workers(aggressive=False)
        self._finder_pending_key = ""
        self._finder_pending_name = ""
        self._finder_pending_background = False
        self._finder_prefetch_queue.clear()
        self._finder_prefetch_enqueued_keys.clear()
        self._finder_prefetch_total = 0
        self._finder_prefetch_completed = 0
        self._finder_prefetch_cached = 0
        self._finder_prefetch_active = False
        self._finder_request_id += 1
        self._set_finder_status("Finder: cancelled" if had_pending else "Finder: idle", busy=False)

    @Slot(int, str, bytes, str)
    def _on_finder_chart_completed(self, request_id: int, key: str, payload: bytes, err: str):
        if hasattr(self, "_finder_timeout_timer"):
            self._finder_timeout_timer.stop()
        if request_id != self._finder_request_id:
            return
        if key != self._finder_pending_key:
            return
        pending_name = str(getattr(self, "_finder_pending_name", "") or "")
        self._finder_pending_key = ""
        self._finder_pending_name = ""
        was_background = bool(getattr(self, "_finder_pending_background", False))
        self._finder_pending_background = False

        if err or not payload:
            status_text = "Finder: unavailable"
            if err and err.lower() == "cancelled":
                status_text = "Finder: cancelled"
            if not was_background:
                self._set_finder_status(status_text, busy=False)
            else:
                self._finder_prefetch_completed += 1
            if err and err.lower() != "cancelled":
                logger.warning("Finder chart generation failed for key '%s': %s", key, err)
                self._finder_retry_after[key] = perf_counter() + FINDER_RETRY_COOLDOWN_S
            if not was_background:
                self._show_finder_aladin_fallback(key, "Finder chart unavailable")
            self._drain_finder_prefetch_queue()
            if was_background and not self._finder_pending_key and not self._finder_prefetch_queue:
                self._finder_prefetch_done_status()
                self._finder_prefetch_active = False
            return

        image = QImage.fromData(payload, "PNG")
        if image.isNull():
            if not was_background:
                self._set_finder_status("Finder: decode failed", busy=False)
            else:
                self._finder_prefetch_completed += 1
            self._finder_retry_after[key] = perf_counter() + FINDER_RETRY_COOLDOWN_S
            if not was_background:
                self._show_finder_aladin_fallback(key, "Finder chart decode failed")
            self._drain_finder_prefetch_queue()
            if was_background and not self._finder_pending_key and not self._finder_prefetch_queue:
                self._finder_prefetch_done_status()
                self._finder_prefetch_active = False
            return
        pix = QPixmap.fromImage(image)
        if pix.isNull():
            if not was_background:
                self._set_finder_status("Finder: decode failed", busy=False)
            else:
                self._finder_prefetch_completed += 1
            self._finder_retry_after[key] = perf_counter() + FINDER_RETRY_COOLDOWN_S
            if not was_background:
                self._show_finder_aladin_fallback(key, "Finder chart decode failed")
            self._drain_finder_prefetch_queue()
            if was_background and not self._finder_pending_key and not self._finder_prefetch_queue:
                self._finder_prefetch_done_status()
                self._finder_prefetch_active = False
            return
        pix_with_overlay = self._build_finder_overlay_pixmap(pix)
        self._cache_finder_pixmap(key, pix_with_overlay)
        self._finder_retry_after.pop(key, None)
        if not was_background:
            name_hint = pending_name.strip()
            if name_hint:
                self._set_finder_status(f"Finder: ready ({name_hint})", busy=False)
            else:
                self._set_finder_status("Finder: ready", busy=False)
        if (not was_background) and hasattr(self, "finder_image_label"):
            self.finder_image_label.setText("")
            self.finder_image_label.setPixmap(pix_with_overlay)
            self._set_cutout_image_loading("finder", "", visible=False)
            self._finder_displayed_key = key
        if was_background:
            self._finder_prefetch_completed += 1
        self._drain_finder_prefetch_queue()
        if was_background and not self._finder_pending_key and not self._finder_prefetch_queue:
            self._finder_prefetch_done_status()
            self._finder_prefetch_active = False

    def _on_finder_chart_worker_finished(self, worker: FinderChartWorker):
        workers = getattr(self, "_finder_workers", None)
        if isinstance(workers, list) and worker in workers:
            workers.remove(worker)
        if self._finder_worker is worker:
            self._finder_worker = None
        if self._finder_pending_key:
            return
        if self._finder_prefetch_queue:
            self._drain_finder_prefetch_queue()
            return
        if self._finder_prefetch_active and self._finder_prefetch_completed >= self._finder_prefetch_total:
            self._finder_prefetch_done_status()
            self._finder_prefetch_active = False

    @Slot()
    def _on_finder_chart_timeout(self):
        key = self._finder_pending_key
        if not key:
            return
        pending_name = str(getattr(self, "_finder_pending_name", "") or "")
        was_background = bool(getattr(self, "_finder_pending_background", False))
        self._finder_pending_key = ""
        self._finder_pending_name = ""
        self._finder_pending_background = False
        self._finder_request_id += 1
        self._finder_retry_after[key] = perf_counter() + FINDER_RETRY_COOLDOWN_S
        self._stop_finder_workers(aggressive=True)
        if not was_background:
            if pending_name.strip():
                self._set_finder_status(f"Finder: timeout ({pending_name})", busy=False)
            else:
                self._set_finder_status("Finder: timeout", busy=False)
            self._show_finder_aladin_fallback(key, "Finder chart timeout")
        else:
            self._finder_prefetch_completed += 1
        logger.warning("Finder chart timed out for key '%s'", key)
        if not self._finder_workers:
            self._drain_finder_prefetch_queue()
        if was_background and not self._finder_pending_key and not self._finder_prefetch_queue and not self._finder_workers:
            self._finder_prefetch_done_status()
            self._finder_prefetch_active = False

    def _update_finder_chart_for_target(
        self,
        target: Target,
        key: str,
        *,
        background: bool = False,
        cache_only: bool = False,
    ):
        if not hasattr(self, "finder_image_label"):
            return
        if (
            not background
            and key == getattr(self, "_finder_displayed_key", "")
            and not self._finder_pending_key
        ):
            return
        cached = self._finder_cache.get(key)
        if cached is not None and not cached.isNull():
            if (not background) and hasattr(self, "_finder_timeout_timer"):
                self._finder_timeout_timer.stop()
            if not background:
                self.finder_image_label.setText("")
                self.finder_image_label.setPixmap(cached)
                self._set_cutout_image_loading("finder", "", visible=False)
                self._set_finder_status(f"Finder: cached ({target.name})", busy=False)
                self._finder_displayed_key = key
            return
        persisted = self._load_pixmap_from_storage_cache("finder_preview", key)
        if persisted is not None and not persisted.isNull():
            self._cache_finder_pixmap(key, persisted, persist=False)
            if (not background) and hasattr(self, "_finder_timeout_timer"):
                self._finder_timeout_timer.stop()
            if not background:
                self.finder_image_label.setText("")
                self.finder_image_label.setPixmap(persisted)
                self._set_cutout_image_loading("finder", "", visible=False)
                self._set_finder_status(f"Finder: cached ({target.name})", busy=False)
                self._finder_displayed_key = key
            return
        cached_variant = self._find_finder_cache_variant(target)
        if cached_variant is not None:
            _variant_key, variant_pix = cached_variant
            self._cache_finder_pixmap(key, variant_pix)
            if (not background) and hasattr(self, "_finder_timeout_timer"):
                self._finder_timeout_timer.stop()
            if not background:
                self.finder_image_label.setText("")
                self.finder_image_label.setPixmap(variant_pix)
                self._set_cutout_image_loading("finder", "", visible=False)
                self._set_finder_status(f"Finder: cached ({target.name})", busy=False)
                self._finder_displayed_key = key
            return
        if cache_only:
            if not background:
                self._set_finder_status(f"Finder: cache miss ({target.name})", busy=False)
            return
        if self._finder_pending_key == key:
            if not background:
                pending_name = str(getattr(self, "_finder_pending_name", "") or "").strip()
                if pending_name:
                    self._set_finder_status(f"Finder: loading {pending_name}...", busy=True)
                    self._set_cutout_image_loading("finder", f"Loading finder chart for {pending_name}…", visible=True)
                else:
                    self._set_finder_status("Finder: loading...", busy=True)
                    self._set_cutout_image_loading("finder", "Loading finder chart…", visible=True)
            return
        if background and self._finder_pending_key:
            self._enqueue_finder_prefetch(target, key)
            return

        retry_after = float(self._finder_retry_after.get(key, 0.0))
        now = perf_counter()
        if retry_after > now:
            secs = max(1, int(round(retry_after - now)))
            if not background:
                self._set_finder_status(f"Finder: retry in {secs}s", busy=False)
                self._show_finder_aladin_fallback(key, f"Finder chart unavailable ({secs}s)")
            return

        if not background:
            self.finder_image_label.setPixmap(QPixmap())
            self.finder_image_label.setText("Loading finder chart…")
            self._set_cutout_image_loading("finder", f"Loading finder chart for {target.name}…", visible=True)

        self._finder_request_id += 1
        req_id = self._finder_request_id
        self._finder_pending_key = key
        self._finder_pending_name = target.name
        self._finder_pending_background = background
        render_w, render_h = self._cutout_render_dimensions_px(getattr(self, "finder_image_label", None))
        old_worker = self._finder_worker
        if (not background) and old_worker is not None:
            try:
                if old_worker.isRunning():
                    old_worker.requestInterruption()
            except Exception:
                pass
        worker = FinderChartWorker(
            request_id=req_id,
            key=key,
            name=target.name,
            ra_deg=target.ra,
            dec_deg=target.dec,
            survey_key=self._cutout_survey_key,
            fov_arcmin=self._cutout_fov_arcmin,
            width_px=render_w,
            height_px=render_h,
            parent=self,
        )
        worker.completed.connect(self._on_finder_chart_completed)
        worker.finished.connect(lambda w=worker: self._on_finder_chart_worker_finished(w))
        worker.finished.connect(worker.deleteLater)
        self._finder_workers.append(worker)
        self._finder_worker = worker
        if not background:
            self._set_finder_status(f"Finder: loading {target.name}...", busy=True)
        else:
            total = max(int(getattr(self, "_finder_prefetch_total", 0)), 1)
            done = int(getattr(self, "_finder_prefetch_completed", 0))
            cached = int(getattr(self, "_finder_prefetch_cached", 0))
            current_idx = min(total, max(1, done + 1))
            queued_left = len(getattr(self, "_finder_prefetch_queue", []))
            self._set_finder_status(
                f"Finder: prefetch {current_idx}/{total} ({cached} cached) {target.name} (queue {queued_left})",
                busy=True,
            )
        worker.start()
        if hasattr(self, "_finder_timeout_timer"):
            self._finder_timeout_timer.start()

    def _enqueue_finder_prefetch(self, target: Optional[Target], key: str) -> str:
        if target is None or not key:
            return "skipped"
        cached = self._finder_cache.get(key)
        if cached is not None and not cached.isNull():
            return "cached"
        cached_variant = self._find_finder_cache_variant(target)
        if cached_variant is not None:
            _variant_key, variant_pix = cached_variant
            # Alias current exact key to cached variant to prevent duplicate fetches.
            self._cache_finder_pixmap(key, variant_pix)
            return "cached"
        if key == self._finder_pending_key:
            return "skipped"
        if key in self._finder_prefetch_enqueued_keys:
            return "skipped"
        retry_after = float(self._finder_retry_after.get(key, 0.0))
        if retry_after > perf_counter():
            return "skipped"
        self._finder_prefetch_queue.append((target, key))
        self._finder_prefetch_enqueued_keys.add(key)
        return "queued"

    def _drain_finder_prefetch_queue(self) -> None:
        if self._finder_pending_key:
            return
        while self._finder_prefetch_queue and not self._finder_pending_key:
            target, key = self._finder_prefetch_queue.pop(0)
            self._finder_prefetch_enqueued_keys.discard(key)
            if target not in self.targets:
                continue
            self._update_finder_chart_for_target(target, key, background=True)

    def _prefetch_finder_charts_for_all_targets(self, prioritize: Optional[Target] = None) -> None:
        if not self.targets:
            return
        # Keep one stable batch at a time to avoid progress inflation.
        if self._finder_prefetch_active and (self._finder_pending_key or self._finder_prefetch_queue):
            return
        if self._finder_pending_key or self._finder_prefetch_queue:
            return
        candidates: list[Target] = []
        if prioritize is not None:
            candidates.append(prioritize)
        for candidate in self.targets:
            if prioritize is not None and candidate is prioritize:
                continue
            candidates.append(candidate)
        if not candidates:
            return

        self._finder_prefetch_total = len(candidates)
        self._finder_prefetch_completed = 0
        self._finder_prefetch_cached = 0
        self._finder_prefetch_active = True

        render_w, render_h = self._cutout_render_dimensions_px(getattr(self, "finder_image_label", None))
        for candidate in candidates:
            key = self._cutout_key_for_target(candidate, render_w, render_h)
            enqueue_state = self._enqueue_finder_prefetch(candidate, key)
            if enqueue_state == "queued":
                pass
            else:
                self._finder_prefetch_completed += 1
                if enqueue_state == "cached":
                    self._finder_prefetch_cached += 1

        if self._finder_prefetch_completed >= self._finder_prefetch_total:
            self._finder_prefetch_done_status()
            self._finder_prefetch_active = False
            return
        self._drain_finder_prefetch_queue()

    def _start_cutout_request(
        self,
        target: Target,
        key: str,
        *,
        render_w: int,
        render_h: int,
        fetch_w: int,
        fetch_h: int,
        background: bool = False,
    ) -> None:
        self._cutout_request_id += 1
        self._cutout_pending_key = key
        self._cutout_pending_name = target.name
        self._cutout_pending_background = background
        if not background:
            self.aladin_image_label.setPixmap(QPixmap())
            self.aladin_image_label.setText("Loading…")
            self._set_cutout_image_loading("aladin", f"Loading Aladin preview for {target.name}…", visible=True)
            self._set_aladin_status(f"Aladin: loading {target.name}...", busy=True)
        else:
            total = max(int(getattr(self, "_cutout_prefetch_total", 0)), 1)
            done = int(getattr(self, "_cutout_prefetch_completed", 0))
            cached = int(getattr(self, "_cutout_prefetch_cached", 0))
            current_idx = min(total, max(1, done + 1))
            queued_left = len(getattr(self, "_cutout_prefetch_queue", []))
            self._set_aladin_status(
                f"Aladin: prefetch {current_idx}/{total} ({cached} cached) {target.name} (queue {queued_left})",
                busy=True,
            )

        query = QUrlQuery()
        query.addQueryItem("hips", _cutout_survey_hips(self._cutout_survey_key))
        query.addQueryItem("ra", f"{target.ra:.8f}")
        query.addQueryItem("dec", f"{target.dec:.8f}")
        fov_x_arcmin, _ = self._aladin_fetch_fov_axes_arcmin(fetch_w, fetch_h)
        query.addQueryItem("fov", f"{(fov_x_arcmin / 60.0):.6f}")
        query.addQueryItem("width", str(fetch_w))
        query.addQueryItem("height", str(fetch_h))
        query.addQueryItem("projection", "TAN")
        query.addQueryItem("coordsys", "icrs")
        query.addQueryItem("format", "png")

        url = QUrl("https://alasky.cds.unistra.fr/hips-image-services/hips2fits")
        url.setQuery(query)
        req = QNetworkRequest(url)
        req.setAttribute(QNetworkRequest.Http2AllowedAttribute, False)
        if hasattr(req, "setTransferTimeout"):
            req.setTransferTimeout(12000)
        req.setRawHeader(b"User-Agent", b"AstroPlanner/1.0 (cutout)")
        req.setRawHeader(b"Accept", b"image/png,image/*;q=0.8,*/*;q=0.2")
        reply = self._cutout_manager.get(req)
        reply.setProperty("cutout_request_id", self._cutout_request_id)
        reply.setProperty("cutout_key", key)
        reply.setProperty("cutout_name", target.name)
        reply.setProperty("cutout_background", 1 if background else 0)
        self._cutout_reply = reply

    def _enqueue_cutout_prefetch(
        self,
        target: Optional[Target],
        key: str,
        *,
        render_w: int,
        render_h: int,
        fetch_w: int,
        fetch_h: int,
    ) -> str:
        if target is None or not key:
            return "skipped"
        cached = self._cutout_cache.get(key)
        if cached is not None and not cached.isNull():
            return "cached"
        cached_variant = self._find_cutout_cache_variant(target)
        if cached_variant is not None:
            _variant_key, variant_pix = cached_variant
            # Alias current exact key to cached variant to prevent duplicate fetches.
            self._cache_cutout_pixmap(key, variant_pix)
            return "cached"
        if key == self._cutout_pending_key:
            return "skipped"
        if key in self._cutout_prefetch_enqueued_keys:
            return "skipped"
        self._cutout_prefetch_queue.append((target, key, int(render_w), int(render_h), int(fetch_w), int(fetch_h)))
        self._cutout_prefetch_enqueued_keys.add(key)
        return "queued"

    def _drain_cutout_prefetch_queue(self) -> None:
        if self._cutout_pending_key:
            return
        while self._cutout_prefetch_queue and not self._cutout_pending_key:
            target, key, render_w, render_h, fetch_w, fetch_h = self._cutout_prefetch_queue.pop(0)
            self._cutout_prefetch_enqueued_keys.discard(key)
            if target not in self.targets:
                if self._cutout_prefetch_active:
                    self._cutout_prefetch_completed += 1
                continue
            self._start_cutout_request(
                target,
                key,
                render_w=render_w,
                render_h=render_h,
                fetch_w=fetch_w,
                fetch_h=fetch_h,
                background=True,
            )

    def _prefetch_cutouts_for_all_targets(self, prioritize: Optional[Target] = None) -> None:
        if not self.targets:
            return
        # Keep one stable batch at a time to avoid progress inflation.
        if self._cutout_prefetch_active and (self._cutout_pending_key or self._cutout_prefetch_queue):
            return
        if self._cutout_pending_key or self._cutout_prefetch_queue:
            return
        candidates: list[Target] = []
        if prioritize is not None:
            candidates.append(prioritize)
        for candidate in self.targets:
            if prioritize is not None and candidate is prioritize:
                continue
            candidates.append(candidate)
        if not candidates:
            return

        self._cutout_prefetch_total = len(candidates)
        self._cutout_prefetch_completed = 0
        self._cutout_prefetch_cached = 0
        self._cutout_prefetch_active = True

        render_w, render_h = self._cutout_render_dimensions_px(getattr(self, "aladin_image_label", None))
        fetch_w, fetch_h = self._aladin_fetch_dimensions_px(render_w, render_h)
        for candidate in candidates:
            key = self._cutout_key_for_target(candidate, render_w, render_h)
            enqueue_state = self._enqueue_cutout_prefetch(
                candidate,
                key,
                render_w=render_w,
                render_h=render_h,
                fetch_w=fetch_w,
                fetch_h=fetch_h,
            )
            if enqueue_state == "queued":
                pass
            else:
                self._cutout_prefetch_completed += 1
                if enqueue_state == "cached":
                    self._cutout_prefetch_cached += 1

        if self._cutout_prefetch_completed >= self._cutout_prefetch_total:
            self._aladin_prefetch_done_status()
            self._cutout_prefetch_active = False
            return
        self._drain_cutout_prefetch_queue()

    def _update_cutout_preview_for_target(self, target: Optional[Target], *, cache_only: bool = False):
        if not hasattr(self, "cutout_image_label"):
            return
        use_cache_only = bool(cache_only)

        if target is None:
            if self._cutout_reply is not None and not self._cutout_reply.isFinished():
                self._cutout_reply.abort()
            self._cutout_reply = None
            self._cutout_pending_key = ""
            self._cutout_pending_name = ""
            self._cutout_pending_background = False
            self._cutout_displayed_key = ""
            self._finder_displayed_key = ""
            self._set_cutout_placeholder("Select a target")
            self._set_aladin_status("Aladin: idle", busy=False)
            self._set_finder_status("Finder: idle", busy=False)
            return

        self._sync_cutout_fov_to_site()
        render_w, render_h = self._cutout_render_dimensions_px(getattr(self, "aladin_image_label", None))
        fetch_w, fetch_h = self._aladin_fetch_dimensions_px(render_w, render_h)
        key = self._cutout_key_for_target(target, render_w, render_h)
        self._cutout_last_resize_signature = self._cutout_resize_signature_for_target(target)
        show_finder = getattr(self, "_cutout_view_key", "aladin") == "finderchart"
        finder_already_displayed = (not show_finder) or (
            key == getattr(self, "_finder_displayed_key", "")
            and not self._finder_pending_key
        )
        if (
            key == getattr(self, "_cutout_displayed_key", "")
            and not self._cutout_pending_key
            and finder_already_displayed
        ):
            return
        if show_finder:
            self._update_finder_chart_for_target(target, key, cache_only=use_cache_only)

        if key in self._cutout_cache:
            aladin_pix = self._cutout_cache[key]
            self.aladin_image_label.setText("")
            self.aladin_image_label.setPixmap(aladin_pix)
            self._set_cutout_image_loading("aladin", "", visible=False)
            self._ensure_aladin_pan_ready()
            self._set_aladin_status(f"Aladin: cached ({target.name})", busy=False)
            self._prefetch_cutouts_for_all_targets(prioritize=target)
            if show_finder:
                self._update_finder_chart_for_target(target, key, cache_only=use_cache_only)
            else:
                self._prefetch_finder_charts_for_all_targets(prioritize=target)
            self._cutout_displayed_key = key
            return

        persisted_cutout = self._load_pixmap_from_storage_cache("cutout_preview", key)
        if persisted_cutout is not None and not persisted_cutout.isNull():
            self._cache_cutout_pixmap(key, persisted_cutout, persist=False)
            self.aladin_image_label.setText("")
            self.aladin_image_label.setPixmap(persisted_cutout)
            self._set_cutout_image_loading("aladin", "", visible=False)
            self._ensure_aladin_pan_ready()
            self._set_aladin_status(f"Aladin: cached ({target.name})", busy=False)
            self._prefetch_cutouts_for_all_targets(prioritize=target)
            if show_finder:
                self._update_finder_chart_for_target(target, key, cache_only=use_cache_only)
            else:
                self._prefetch_finder_charts_for_all_targets(prioritize=target)
            self._cutout_displayed_key = key
            return

        cached_variant = self._find_cutout_cache_variant(target)
        if cached_variant is not None:
            cached_key, cached_pix = cached_variant
            self.aladin_image_label.setText("")
            self.aladin_image_label.setPixmap(cached_pix)
            self._set_cutout_image_loading("aladin", "", visible=False)
            self._ensure_aladin_pan_ready()
            self._set_aladin_status(f"Aladin: cached ({target.name})", busy=False)
            self._prefetch_cutouts_for_all_targets(prioritize=target)
            if show_finder:
                self._update_finder_chart_for_target(target, key, cache_only=use_cache_only)
            else:
                self._prefetch_finder_charts_for_all_targets(prioritize=target)
            self._cutout_displayed_key = cached_key
            return
        if use_cache_only:
            self._set_aladin_status(f"Aladin: cache miss ({target.name})", busy=False)
            if show_finder:
                self._update_finder_chart_for_target(target, key, cache_only=True)
            if not getattr(self, "_cutout_displayed_key", ""):
                self._set_cutout_placeholder("Preview not cached yet")
            return

        if key == self._cutout_pending_key and self._cutout_reply is not None and not self._cutout_reply.isFinished():
            pending_name = str(getattr(self, "_cutout_pending_name", "") or "").strip()
            if pending_name:
                self._set_aladin_status(f"Aladin: loading {pending_name}...", busy=True)
                self._set_cutout_image_loading("aladin", f"Loading Aladin preview for {pending_name}…", visible=True)
            else:
                self._set_aladin_status("Aladin: loading...", busy=True)
                self._set_cutout_image_loading("aladin", "Loading Aladin preview…", visible=True)
            return

        if self._cutout_reply is not None and not self._cutout_reply.isFinished():
            if self._cutout_pending_background and self._cutout_prefetch_active:
                self._cutout_prefetch_completed += 1
            self._cutout_reply.abort()
        self._cutout_reply = None
        self._cutout_pending_key = ""
        self._cutout_pending_name = ""
        self._cutout_pending_background = False

        self._start_cutout_request(
            target,
            key,
            render_w=render_w,
            render_h=render_h,
            fetch_w=fetch_w,
            fetch_h=fetch_h,
            background=False,
        )

    @Slot(QNetworkReply)
    def _on_cutout_reply(self, reply: QNetworkReply):
        req_id = int(reply.property("cutout_request_id") or 0)
        key = str(reply.property("cutout_key") or "")
        target_name = str(reply.property("cutout_name") or "").strip()
        is_background = bool(int(reply.property("cutout_background") or 0))

        # Ignore stale or unrelated replies.
        if req_id != self._cutout_request_id or key != self._cutout_pending_key:
            reply.deleteLater()
            return

        self._cutout_reply = None
        self._cutout_pending_key = ""
        self._cutout_pending_name = ""
        self._cutout_pending_background = False

        if reply.error() != QNetworkReply.NoError:
            err = reply.error()
            if err != QNetworkReply.OperationCanceledError:
                logger.warning("Cutout fetch failed for '%s': %s", target_name or key, reply.errorString())
                if not is_background:
                    self._set_cutout_placeholder("Preview unavailable")
                    label = target_name or "target"
                    self._set_aladin_status(f"Aladin: unavailable ({label})", busy=False)
            if is_background:
                self._cutout_prefetch_completed += 1
            reply.deleteLater()
            self._drain_cutout_prefetch_queue()
            if is_background and not self._cutout_pending_key and not self._cutout_prefetch_queue:
                self._aladin_prefetch_done_status()
                self._cutout_prefetch_active = False
            return

        payload = bytes(reply.readAll())
        reply.deleteLater()
        pixmap = QPixmap()
        if not payload or not pixmap.loadFromData(payload):
            if not is_background:
                self._set_cutout_placeholder("Preview decode failed")
                label = target_name or "target"
                self._set_aladin_status(f"Aladin: decode failed ({label})", busy=False)
            else:
                self._cutout_prefetch_completed += 1
                self._drain_cutout_prefetch_queue()
                if not self._cutout_pending_key and not self._cutout_prefetch_queue:
                    self._aladin_prefetch_done_status()
                    self._cutout_prefetch_active = False
            return

        self._cache_cutout_pixmap(key, pixmap)
        target_lc = target_name.strip().lower()
        target = next((t for t in self.targets if t.name.strip().lower() == target_lc), None)
        if not is_background:
            self.aladin_image_label.setText("")
            self.aladin_image_label.setPixmap(pixmap)
            self._set_cutout_image_loading("aladin", "", visible=False)
            self._ensure_aladin_pan_ready()
            label = target_name or "target"
            self._set_aladin_status(f"Aladin: ready ({label})", busy=False)
            self._cutout_displayed_key = key
            if target is not None and hasattr(self, "finder_image_label"):
                should_render_finder = getattr(self, "_cutout_view_key", "aladin") == "finderchart"
                if should_render_finder:
                    self._update_finder_chart_for_target(target, key)
                else:
                    self._prefetch_finder_charts_for_all_targets(prioritize=target)
            self._prefetch_cutouts_for_all_targets(prioritize=target)
        else:
            self._cutout_prefetch_completed += 1
        self._drain_cutout_prefetch_queue()
        if is_background and not self._cutout_pending_key and not self._cutout_prefetch_queue:
            self._aladin_prefetch_done_status()
            self._cutout_prefetch_active = False

    def _clear_cutout_cache(self):
        if self._cutout_reply is not None and not self._cutout_reply.isFinished():
            self._cutout_reply.abort()
        self._cancel_finder_chart_worker()
        self._cutout_reply = None
        self._cutout_prefetch_queue.clear()
        self._cutout_prefetch_enqueued_keys.clear()
        self._cutout_prefetch_total = 0
        self._cutout_prefetch_completed = 0
        self._cutout_prefetch_cached = 0
        self._cutout_prefetch_active = False
        self._cutout_cache.clear()
        self._cutout_cache_order.clear()
        self._finder_cache.clear()
        self._finder_cache_order.clear()
        self._finder_retry_after.clear()
        self._cutout_last_resize_signature = None
        self._cutout_displayed_key = ""
        self._finder_displayed_key = ""
        self._cutout_pending_key = ""
        self._cutout_pending_name = ""
        self._cutout_pending_background = False
        self._set_aladin_status("Aladin: idle", busy=False)

    def _selected_target_or_none(self) -> Optional[Target]:
        rows = self._selected_rows() if hasattr(self, "table_view") else []
        if rows and 0 <= rows[0] < len(self.targets):
            return self.targets[rows[0]]
        return None

    @Slot(int)
    def _on_cutout_tab_changed(self, index: int):
        self._cutout_view_key = "finderchart" if index == 1 else "aladin"
        self.settings.setValue("general/cutoutView", self._cutout_view_key)
        if self._cutout_view_key != "finderchart":
            self._cancel_finder_chart_worker()
        self._update_cutout_preview_for_target(self._selected_target_or_none())

    @Slot()
    def _aladin_zoom_in(self):
        if not hasattr(self, "aladin_image_label"):
            return
        self.aladin_image_label.set_zoom(self.aladin_image_label.zoom_factor() * 1.15)

    def _aladin_expand_context(self) -> None:
        context_now = max(1.0, float(getattr(self, "_aladin_context_factor", 1.0)))
        step = max(1.05, float(CUTOUT_ALADIN_CONTEXT_STEP))
        context_next = min(8.0, context_now * step)
        if context_next <= context_now + 1e-4:
            return
        self._aladin_context_factor = context_next
        self._update_cutout_preview_for_target(self._selected_target_or_none())

    @Slot()
    def _aladin_zoom_out(self):
        if not hasattr(self, "aladin_image_label"):
            return
        label = self.aladin_image_label
        current = float(label.zoom_factor())
        if current > 1.02:
            label.set_zoom(current / 1.15)
            return
        # Already at base scale: widen fetched Aladin context.
        self._aladin_expand_context()

    @Slot()
    def _aladin_zoom_reset(self):
        if not hasattr(self, "aladin_image_label"):
            return
        # Reset only viewport state (zoom/pan) in current image.
        # Do not change context factor and do not refetch cutout.
        self.aladin_image_label.reset_zoom()
        self.aladin_image_label.set_zoom(float(CUTOUT_ALADIN_INITIAL_PAN_ZOOM))

    @Slot(str)
    def _on_obs_change(self, name: str, *, defer_replot: bool = False):
        logger.info("Observatory switched to %s", name)
        """Populate site fields when an observatory is selected."""
        site = self.observatories.get(name)
        if site is None:
            return
        if hasattr(self, "settings"):
            self.settings.setValue("general/defaultSite", name)
        self._prime_selected_observatory(name)
        # Update the table model and replot with debounce
        self._clear_table_dynamic_cache()
        self.target_metrics.clear()
        self.target_windows.clear()
        self.last_payload = None
        # Reset color mapping until new calculation assigns fresh colors
        self.table_model.color_map.clear()
        self.table_model.layoutChanged.emit()
        self._validate_site_inputs()
        if not defer_replot:
            self._begin_visibility_refresh(f"Updating view for {name}…")
            self._replot_timer.start()
            if hasattr(self, "aladin_image_label"):
                self._set_cutout_image_loading("aladin", f"Loading Aladin preview for {name}…", visible=True)
            if hasattr(self, "finder_image_label"):
                self._set_cutout_image_loading("finder", f"Loading finder chart for {name}…", visible=True)
            weather_window = getattr(self, "weather_window", None)
            if isinstance(weather_window, WeatherDialog):
                weather_window._set_weather_plot_loading("conditions", "Updating conditions…", visible=True)
                weather_window._set_weather_plot_loading("meteogram", "Updating meteogram…", visible=True)
                weather_window._set_weather_image_loading("cloud_map", "Updating cloud climatology…", visible=True)
                weather_window._set_weather_image_loading("satellite", "Updating satellite preview…", visible=True)
        self._obs_change_finalize_pending = True
        self._schedule_plan_autosave()
        QTimer.singleShot(0, lambda n=str(name): self._finalize_observatory_change(n))

    def _prime_selected_observatory(self, name: str) -> bool:
        site = self.observatories.get(name)
        if site is None:
            return False
        self.lat_edit.setText(f"{site.latitude}")
        self.lon_edit.setText(f"{site.longitude}")
        self.elev_edit.setText(f"{site.elevation}")
        self.table_model.site = site
        self._validate_site_inputs()
        return True

    def _finalize_observatory_change(self, name: str) -> None:
        if not hasattr(self, "obs_combo") or self.obs_combo.currentText() != name:
            return
        site = self.observatories.get(name)
        if site is None:
            return
        self._obs_change_finalize_pending = False
        # Start the lightweight realtime worker first so the UI can repaint
        # immediately; heavier panel rebuilds are deferred one more tick.
        self._start_clock_worker()
        QTimer.singleShot(0, lambda n=str(name): self._refresh_observatory_dependent_views(n))

    def _refresh_observatory_dependent_views(self, name: str) -> None:
        if not hasattr(self, "obs_combo") or self.obs_combo.currentText() != name:
            return
        site = self.observatories.get(name)
        if site is None:
            return
        self._sync_cutout_fov_to_site(site)
        if not getattr(self, "_defer_startup_preview_updates", False):
            self._update_cutout_preview_for_target(self._selected_target_or_none())
        self._refresh_weather_window_context()

    def _current_limiting_magnitude(self) -> float:
        site = None
        if hasattr(self, "obs_combo") and hasattr(self, "observatories"):
            site = self.observatories.get(self.obs_combo.currentText())
        if site is None and hasattr(self, "table_model"):
            site = self.table_model.site
        if site is not None:
            value = _safe_float(getattr(site, "limiting_magnitude", None))
            if value is not None and math.isfinite(value):
                return float(value)
        return DEFAULT_LIMITING_MAGNITUDE

    def _quick_targets_config(self) -> dict[str, object]:
        count = self.settings.value("general/quickTargetsCount", QUICK_TARGETS_DEFAULT_COUNT, type=int)
        min_importance = self.settings.value(
            "general/quickTargetsMinImportance",
            BHTOM_SUGGESTION_MIN_IMPORTANCE,
            type=float,
        )
        try:
            count = int(count)
        except Exception:
            count = QUICK_TARGETS_DEFAULT_COUNT
        count = max(QUICK_TARGETS_MIN_COUNT, min(QUICK_TARGETS_MAX_COUNT, count))
        try:
            min_importance = float(min_importance)
        except Exception:
            min_importance = float(BHTOM_SUGGESTION_MIN_IMPORTANCE)
        if not math.isfinite(min_importance) or min_importance < 0.0:
            min_importance = float(BHTOM_SUGGESTION_MIN_IMPORTANCE)
        return {
            "count": count,
            "min_importance": min_importance,
            "use_score_filter": self.settings.value("general/quickTargetsUseScoreFilter", True, type=bool),
            "use_moon_filter": self.settings.value("general/quickTargetsUseMoonFilter", True, type=bool),
            "use_limiting_mag": self.settings.value("general/quickTargetsUseLimitingMag", True, type=bool),
        }

    def _update_quick_targets_button_tooltip(self) -> None:
        if not hasattr(self, "quick_targets_btn"):
            return
        cfg = self._quick_targets_config()
        count = int(cfg["count"])
        use_score = bool(cfg["use_score_filter"])
        use_moon = bool(cfg["use_moon_filter"])
        use_lim_mag = bool(cfg["use_limiting_mag"])
        parts: list[str] = []
        if use_score:
            parts.append("Score")
        if use_moon:
            parts.append("Moon Sep")
        if use_lim_mag:
            parts.append("Limiting mag")
        filters_txt = ", ".join(parts) if parts else "no extra filters"
        self.quick_targets_btn.setToolTip(
            f"Add top {count} suggested targets sorted by score ({filters_txt})."
        )

    def _seestar_config(self) -> dict[str, object]:
        method = str(self.settings.value("general/seestarMethod", SEESTAR_METHOD_GUIDED, type=str) or SEESTAR_METHOD_GUIDED).strip().lower()
        if method not in {SEESTAR_METHOD_GUIDED, SEESTAR_METHOD_ALP}:
            method = SEESTAR_METHOD_GUIDED
        alp_base_url = (
            str(self.settings.value("general/seestarAlpBaseUrl", SEESTAR_ALP_DEFAULT_BASE_URL, type=str) or SEESTAR_ALP_DEFAULT_BASE_URL)
            .strip()
            .rstrip("/")
        ) or SEESTAR_ALP_DEFAULT_BASE_URL
        alp_device_num = self.settings.value("general/seestarAlpDeviceNum", SEESTAR_ALP_DEFAULT_DEVICE_NUM, type=int)
        alp_client_id = self.settings.value("general/seestarAlpClientId", SEESTAR_ALP_DEFAULT_CLIENT_ID, type=int)
        alp_timeout_s = self.settings.value("general/seestarAlpTimeoutSec", SEESTAR_ALP_DEFAULT_TIMEOUT_S, type=float)
        alp_gain = self.settings.value("general/seestarAlpGain", SEESTAR_ALP_DEFAULT_GAIN, type=int)
        alp_panel_overlap = self.settings.value(
            "general/seestarAlpPanelOverlapPercent",
            SEESTAR_ALP_DEFAULT_PANEL_OVERLAP_PERCENT,
            type=int,
        )
        alp_use_autofocus = self.settings.value(
            "general/seestarAlpUseAutofocus",
            SEESTAR_ALP_DEFAULT_USE_AUTOFOCUS,
            type=bool,
        )
        alp_num_tries = self.settings.value("general/seestarAlpNumTries", SEESTAR_ALP_DEFAULT_NUM_TRIES, type=int)
        alp_retry_wait_s = self.settings.value(
            "general/seestarAlpRetryWaitSec",
            SEESTAR_ALP_DEFAULT_RETRY_WAIT_S,
            type=int,
        )
        alp_stack_exposure_ms = self.settings.value(
            "general/seestarAlpStackExposureMs",
            SEESTAR_ALP_DEFAULT_STACK_EXPOSURE_MS,
            type=int,
        )
        alp_lp_filter_mode = self.settings.value(
            "general/seestarAlpLpFilterMode",
            SEESTAR_ALP_LP_FILTER_AUTO,
            type=str,
        )
        alp_honor_queue_times = self.settings.value(
            "general/seestarAlpHonorQueueTimes",
            SEESTAR_ALP_DEFAULT_HONOR_QUEUE_TIMES,
            type=bool,
        )
        alp_wait_until_local_time = self.settings.value(
            "general/seestarAlpWaitUntilLocalTime",
            SEESTAR_ALP_DEFAULT_WAIT_UNTIL_LOCAL_TIME,
            type=str,
        )
        alp_startup_enabled = self.settings.value(
            "general/seestarAlpStartupEnabled",
            SEESTAR_ALP_DEFAULT_STARTUP_SEQUENCE,
            type=bool,
        )
        alp_startup_polar_align = self.settings.value(
            "general/seestarAlpStartupPolarAlign",
            SEESTAR_ALP_DEFAULT_STARTUP_POLAR_ALIGN,
            type=bool,
        )
        alp_startup_auto_focus = self.settings.value(
            "general/seestarAlpStartupAutoFocus",
            SEESTAR_ALP_DEFAULT_STARTUP_AUTO_FOCUS,
            type=bool,
        )
        alp_startup_dark_frames = self.settings.value(
            "general/seestarAlpStartupDarkFrames",
            SEESTAR_ALP_DEFAULT_STARTUP_DARK_FRAMES,
            type=bool,
        )
        alp_capture_flats = self.settings.value(
            "general/seestarAlpCaptureFlatsBeforeSession",
            SEESTAR_ALP_DEFAULT_CAPTURE_FLATS,
            type=bool,
        )
        alp_flats_wait_s = self.settings.value(
            "general/seestarAlpFlatsWaitSec",
            SEESTAR_ALP_DEFAULT_FLATS_WAIT_S,
            type=int,
        )
        alp_schedule_autofocus_legacy = self.settings.value(
            "general/seestarAlpScheduleAutofocusBeforeEachTarget",
            SEESTAR_ALP_DEFAULT_SCHEDULE_AUTOFOCUS,
            type=bool,
        )
        alp_schedule_autofocus_mode = normalize_seestar_alp_schedule_autofocus_mode(
            self.settings.value(
                "general/seestarAlpScheduleAutofocusMode",
                SEESTAR_ALP_DEFAULT_SCHEDULE_AUTOFOCUS_MODE,
                type=str,
            ),
            legacy_enabled=bool(alp_schedule_autofocus_legacy),
        )
        alp_schedule_autofocus_try_count = self.settings.value(
            "general/seestarAlpScheduleAutofocusTryCount",
            SEESTAR_ALP_DEFAULT_AUTOFOCUS_TRY_COUNT,
            type=int,
        )
        alp_dew_heater_value = self.settings.value(
            "general/seestarAlpDewHeaterValue",
            SEESTAR_ALP_DEFAULT_DEW_HEATER_VALUE,
            type=int,
        )
        alp_park_after_session = self.settings.value(
            "general/seestarAlpParkAfterSession",
            SEESTAR_ALP_DEFAULT_PARK_AFTER_SESSION,
            type=bool,
        )
        alp_shutdown_after_session = self.settings.value(
            "general/seestarAlpShutdownAfterSession",
            SEESTAR_ALP_DEFAULT_SHUTDOWN_AFTER_SESSION,
            type=bool,
        )
        session_template_key = self.settings.value(
            "general/seestarSessionTemplateKey",
            "",
            type=str,
        )
        session_repeat_count = self.settings.value(
            "general/seestarSessionRepeatCount",
            1,
            type=int,
        )
        session_minutes_per_run = self.settings.value(
            "general/seestarSessionMinutesPerRun",
            SEESTAR_DEFAULT_BLOCK_MINUTES,
            type=int,
        )
        session_gap_seconds = self.settings.value(
            "general/seestarSessionGapSeconds",
            0,
            type=int,
        )
        session_require_checklist = self.settings.value(
            "general/seestarSessionRequireChecklist",
            False,
            type=bool,
        )
        session_checklist_text = self.settings.value(
            "general/seestarSessionChecklistText",
            "",
            type=str,
        )
        session_notes = self.settings.value("general/seestarSessionTemplateNotes", "", type=str)
        try:
            alp_device_num = int(alp_device_num)
        except Exception:
            alp_device_num = SEESTAR_ALP_DEFAULT_DEVICE_NUM
        try:
            alp_client_id = int(alp_client_id)
        except Exception:
            alp_client_id = SEESTAR_ALP_DEFAULT_CLIENT_ID
        try:
            alp_timeout_s = float(alp_timeout_s)
        except Exception:
            alp_timeout_s = SEESTAR_ALP_DEFAULT_TIMEOUT_S
        try:
            alp_gain = int(alp_gain)
        except Exception:
            alp_gain = SEESTAR_ALP_DEFAULT_GAIN
        try:
            alp_panel_overlap = int(alp_panel_overlap)
        except Exception:
            alp_panel_overlap = SEESTAR_ALP_DEFAULT_PANEL_OVERLAP_PERCENT
        try:
            alp_num_tries = int(alp_num_tries)
        except Exception:
            alp_num_tries = SEESTAR_ALP_DEFAULT_NUM_TRIES
        try:
            alp_retry_wait_s = int(alp_retry_wait_s)
        except Exception:
            alp_retry_wait_s = SEESTAR_ALP_DEFAULT_RETRY_WAIT_S
        try:
            alp_stack_exposure_ms = int(alp_stack_exposure_ms)
        except Exception:
            alp_stack_exposure_ms = SEESTAR_ALP_DEFAULT_STACK_EXPOSURE_MS
        try:
            alp_flats_wait_s = int(alp_flats_wait_s)
        except Exception:
            alp_flats_wait_s = SEESTAR_ALP_DEFAULT_FLATS_WAIT_S
        try:
            alp_schedule_autofocus_try_count = int(alp_schedule_autofocus_try_count)
        except Exception:
            alp_schedule_autofocus_try_count = SEESTAR_ALP_DEFAULT_AUTOFOCUS_TRY_COUNT
        try:
            alp_dew_heater_value = int(alp_dew_heater_value)
        except Exception:
            alp_dew_heater_value = SEESTAR_ALP_DEFAULT_DEW_HEATER_VALUE
        try:
            session_repeat_count = int(session_repeat_count)
        except Exception:
            session_repeat_count = 1
        try:
            session_minutes_per_run = int(session_minutes_per_run)
        except Exception:
            session_minutes_per_run = SEESTAR_DEFAULT_BLOCK_MINUTES
        try:
            session_gap_seconds = int(session_gap_seconds)
        except Exception:
            session_gap_seconds = 0
        return {
            "method": method,
            "alp": SeestarAlpConfig(
                base_url=alp_base_url,
                device_num=max(0, alp_device_num),
                client_id=max(1, alp_client_id),
                timeout_s=max(1.0, alp_timeout_s),
                gain=max(0, alp_gain),
                panel_overlap_percent=max(0, alp_panel_overlap),
                use_autofocus=bool(alp_use_autofocus),
                num_tries=max(1, alp_num_tries),
                retry_wait_s=max(0, alp_retry_wait_s),
                target_integration_override_min=0,
                stack_exposure_ms=max(0, alp_stack_exposure_ms),
                lp_filter_mode=str(alp_lp_filter_mode or SEESTAR_ALP_LP_FILTER_AUTO).strip().lower(),
                honor_queue_times=bool(alp_honor_queue_times),
                wait_until_local_time=str(alp_wait_until_local_time or "").strip(),
                startup_enabled=bool(alp_startup_enabled),
                startup_polar_align=bool(alp_startup_polar_align),
                startup_auto_focus=bool(alp_startup_auto_focus),
                startup_dark_frames=bool(alp_startup_dark_frames),
                capture_flats_before_session=bool(alp_capture_flats),
                flats_wait_s=max(0, alp_flats_wait_s),
                schedule_autofocus_mode=alp_schedule_autofocus_mode,
                schedule_autofocus_before_each_target=(alp_schedule_autofocus_mode == SEESTAR_ALP_AF_MODE_PER_RUN),
                schedule_autofocus_try_count=max(1, alp_schedule_autofocus_try_count),
                dew_heater_value=alp_dew_heater_value if alp_dew_heater_value >= 0 else SEESTAR_ALP_DEFAULT_DEW_HEATER_VALUE,
                park_after_session=bool(alp_park_after_session),
                shutdown_after_session=bool(alp_shutdown_after_session),
            ),
            "template": SeestarSessionTemplate(
                key=str(session_template_key or "").strip(),
                name="",
                scope=SEESTAR_TEMPLATE_SCOPE_MULTI_TARGET,
                repeat_count=max(1, session_repeat_count),
                minutes_per_run=max(1, session_minutes_per_run),
                gap_seconds=max(0, session_gap_seconds),
                require_science_checklist=bool(session_require_checklist),
                science_checklist_items=[
                    line.strip()
                    for line in str(session_checklist_text or "").splitlines()
                    if line.strip()
                ],
                template_notes=str(session_notes or "").strip(),
                lp_filter_mode=str(alp_lp_filter_mode or SEESTAR_ALP_LP_FILTER_AUTO).strip().lower(),
                gain=max(0, alp_gain),
                panel_overlap_percent=max(0, alp_panel_overlap),
                use_autofocus=bool(alp_use_autofocus),
                num_tries=max(1, alp_num_tries),
                retry_wait_s=max(0, alp_retry_wait_s),
                target_integration_override_min=0,
                stack_exposure_ms=max(0, alp_stack_exposure_ms),
                honor_queue_times=bool(alp_honor_queue_times),
                wait_until_local_time=str(alp_wait_until_local_time or "").strip(),
                startup_enabled=bool(alp_startup_enabled),
                startup_polar_align=bool(alp_startup_polar_align),
                startup_auto_focus=bool(alp_startup_auto_focus),
                startup_dark_frames=bool(alp_startup_dark_frames),
                capture_flats_before_session=bool(alp_capture_flats),
                flats_wait_s=max(0, alp_flats_wait_s),
                schedule_autofocus_mode=alp_schedule_autofocus_mode,
                schedule_autofocus_before_each_target=(alp_schedule_autofocus_mode == SEESTAR_ALP_AF_MODE_PER_RUN),
                schedule_autofocus_try_count=max(1, alp_schedule_autofocus_try_count),
                dew_heater_value=alp_dew_heater_value if alp_dew_heater_value >= 0 else SEESTAR_ALP_DEFAULT_DEW_HEATER_VALUE,
                park_after_session=bool(alp_park_after_session),
                shutdown_after_session=bool(alp_shutdown_after_session),
            ),
        }

    def _update_seestar_button_tooltip(self) -> None:
        if not hasattr(self, "seestar_session_btn"):
            return
        cfg = self._seestar_config()
        backend_label = "ALP service" if str(cfg["method"]) == SEESTAR_METHOD_ALP else "guided handoff"
        template = cfg["template"]
        tooltip = (
            "Build a Seestar S50 session from the current Targets table order. "
            f"Backend: {backend_label}, defaults: {int(template.repeat_count)}x {int(template.minutes_per_run)} min, "
            f"gap {int(template.gap_seconds)} s."
        )
        self.seestar_session_btn.setToolTip(tooltip)
        if hasattr(self, "seestar_session_act"):
            self.seestar_session_act.setToolTip(tooltip)
            self.seestar_session_act.setStatusTip(tooltip)

    def _current_site_snapshot(self) -> NightQueueSiteSnapshot:
        selected_name = self.obs_combo.currentText() if hasattr(self, "obs_combo") else ""
        preset_site = self.observatories.get(selected_name) if hasattr(self, "observatories") else None
        tz_name = "UTC"
        if isinstance(getattr(self, "last_payload", None), dict):
            tz_name = str(self.last_payload.get("tz", "UTC"))
        elif preset_site is not None:
            tz_name = preset_site.timezone_name
        return NightQueueSiteSnapshot(
            name=selected_name or getattr(preset_site, "name", "") or "Current site",
            latitude=self._read_site_float(self.lat_edit),
            longitude=self._read_site_float(self.lon_edit),
            elevation=self._read_site_float(self.elev_edit),
            timezone=tz_name,
            limiting_magnitude=self._current_limiting_magnitude(),
            telescope_diameter_mm=float(getattr(preset_site, "telescope_diameter_mm", 0.0) or 0.0),
            focal_length_mm=float(getattr(preset_site, "focal_length_mm", 0.0) or 0.0),
            pixel_size_um=float(getattr(preset_site, "pixel_size_um", 0.0) or 0.0),
            detector_width_px=int(getattr(preset_site, "detector_width_px", 0) or 0),
            detector_height_px=int(getattr(preset_site, "detector_height_px", 0) or 0),
        )

    def _seestar_night_bounds(self) -> Optional[tuple[datetime, datetime, str]]:
        payload = self.last_payload if isinstance(getattr(self, "last_payload", None), dict) else None
        if payload is None or "times" not in payload:
            return None
        tz_name = str(payload.get("tz", "UTC"))
        try:
            tz = pytz.timezone(tz_name)
        except Exception:
            tz = pytz.UTC
            tz_name = "UTC"
        try:
            times = [t.astimezone(tz) for t in mdates.num2date(payload["times"])]
        except Exception:
            return None
        if not times:
            return None
        sun_alt_series = np.array(payload.get("sun_alt", np.full(len(times), np.nan)), dtype=float)
        if sun_alt_series.shape[0] != len(times):
            return None
        sun_mask = np.isfinite(sun_alt_series) & (sun_alt_series <= self._sun_alt_limit())
        if not sun_mask.any():
            return None
        indices = np.where(sun_mask)[0]
        start_idx = int(indices[0])
        end_idx = min(int(indices[-1]) + 1, len(times) - 1)
        start_dt = times[start_idx]
        end_dt = times[end_idx]
        if end_dt <= start_dt:
            return None
        return start_dt, end_dt, tz_name

    @Slot()
    def _open_seestar_session(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(self, "Seestar Session", "Wait for the current visibility calculation to finish.")
            return
        night_bounds = self._seestar_night_bounds()
        if night_bounds is None:
            QMessageBox.information(
                self,
                "Seestar Session",
                "Run a visibility calculation first so the Seestar queue uses the current night bounds.",
            )
            return
        session_dialog = SeestarSessionPlanDialog(self)
        if session_dialog.exec() != int(QDialog.Accepted):
            return
        self._update_seestar_button_tooltip()
        night_start, night_end, tz_name = night_bounds
        method = str(session_dialog.method() or SEESTAR_METHOD_GUIDED).strip().lower()
        session_template = session_dialog.session_template()
        session_items = [item for item in session_dialog.session_items() if item.enabled]
        alp_config = session_dialog.alp_config()
        if not session_items:
            QMessageBox.information(self, "Seestar Session", "Enable at least one target in the session table.")
            return
        device_profile = SEESTAR_DEVICE_PROFILE_ALP if method == SEESTAR_METHOD_ALP else SEESTAR_DEVICE_PROFILE
        site_snapshot = self._current_site_snapshot()
        if (
            str(session_template.scope or SEESTAR_TEMPLATE_SCOPE_MULTI_TARGET) == SEESTAR_TEMPLATE_SCOPE_SINGLE_TARGET
            and len(session_items) != 1
        ):
            QMessageBox.information(
                self,
                "Seestar Session",
                "The selected template is single-target. Enable exactly one target in the session table.",
            )
            return
        if session_template.require_science_checklist and not session_template.science_checklist_items:
            session_template = session_template.model_copy(update={"science_checklist_items": default_science_checklist_items()})
        queue = build_session_queue(
            session_items,
            session_template=session_template,
            site_snapshot=site_snapshot,
            night_start_local=night_start,
            night_end_local=night_end,
            timezone=tz_name,
            device_profile=device_profile,
            start_cursor_local=datetime.now(night_start.tzinfo) if night_start.tzinfo is not None else datetime.now(),
        )
        if not queue.blocks:
            QMessageBox.information(self, "Seestar Session", "No session blocks were generated from the current Targets table.")
            return
        adapter = (
            SeestarAlpAdapter(alp_config)
            if method == SEESTAR_METHOD_ALP
            else SeestarGuidedAdapter()
        )
        bundle = adapter.build_handoff_bundle(queue)
        adapter.open_handoff_dialog(bundle, parent=self)

    def _observatories_config_path(self) -> Path:
        return Path(__file__).resolve().parent / "config" / "default_observatories.json"

    def _update_obs_combo_widths(self):
        if not hasattr(self, "obs_combo"):
            return
        names = list(self.observatories.keys()) if hasattr(self, "observatories") else []
        fm = self.obs_combo.fontMetrics()
        longest_px = max((fm.horizontalAdvance(name) for name in names), default=220)
        combo_w = int(min(max(250, longest_px + 52), 460))
        popup_w = int(min(max(340, longest_px + 88), 700))
        self.obs_combo.setMinimumWidth(combo_w)
        self.obs_combo.setMaximumWidth(460)
        view = self.obs_combo.view()
        if view is not None:
            view.setMinimumWidth(popup_w)
            view.setTextElideMode(Qt.ElideNone)

    def _parse_custom_observatories_payload(self, payload: object) -> tuple[dict[str, Site], dict[str, str]]:
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
            except Exception as exc:
                logger.warning("Skipping invalid saved observatory %r: %s", name, exc)
        return loaded, preset_keys

    def _load_custom_observatories(self) -> tuple[dict[str, Site], dict[str, str]]:
        storage = getattr(self, "app_storage", None)
        if storage is not None:
            try:
                stored_items = storage.observatories.list_all()
                if stored_items:
                    return self._parse_custom_observatories_payload({"observatories": stored_items})
            except Exception as exc:
                logger.warning("Failed to load observatories from storage: %s", exc)

        cfg_path = self._observatories_config_path()
        if cfg_path.exists():
            try:
                payload = json.loads(cfg_path.read_text(encoding="utf-8"))
                return self._parse_custom_observatories_payload(payload)
            except Exception as exc:
                logger.warning("Failed to parse %s: %s", cfg_path, exc)
        return {}, {}

    def _save_custom_observatories(
        self,
        custom_sites: Optional[dict[str, Site]] = None,
        *,
        preset_keys: Optional[dict[str, str]] = None,
    ):
        if custom_sites is None:
            source_sites = self.observatories if hasattr(self, "observatories") else {}
        else:
            source_sites = custom_sites
        if preset_keys is None:
            source_preset_keys = getattr(self, "_observatory_preset_keys", {})
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
        storage = getattr(self, "app_storage", None)
        if storage is not None:
            try:
                storage.observatories.replace_all(observatory_items)
                return
            except Exception as exc:
                logger.warning("Failed to save observatories to storage: %s", exc)
                return

    def _refresh_observatory_combo(
        self,
        selected_name: Optional[str] = None,
        *,
        emit_selection_change: bool = False,
    ):
        current = selected_name or self.obs_combo.currentText()
        self.obs_combo.blockSignals(True)
        self.obs_combo.clear()
        self.obs_combo.addItems(self.observatories.keys())
        if current in self.observatories:
            self.obs_combo.setCurrentText(current)
        elif self.obs_combo.count() > 0:
            self.obs_combo.setCurrentIndex(0)
        self.obs_combo.blockSignals(False)
        self._update_obs_combo_widths()
        if emit_selection_change:
            selected = self.obs_combo.currentText().strip()
            if selected:
                self._on_obs_change(selected)

    def _lookup_observatory_coordinates(self, query: str) -> tuple[float, float, Optional[float], str]:
        q = query.strip()
        if not q:
            raise ValueError("Name cannot be empty.")

        # 1) Geocode place name to lat/lon (OpenStreetMap Nominatim)
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

        # 2) Optional elevation fetch (Open-Meteo)
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
            # Elevation is optional for this feature.
            elevation = None

        return lat, lon, elevation, display_name

    @Slot()
    def _open_observatory_manager(self):
        current_name = self.obs_combo.currentText() if hasattr(self, "obs_combo") else ""
        dlg = ObservatoryManagerDialog(
            self.observatories,
            self,
            preset_keys=getattr(self, "_observatory_preset_keys", {}),
            selected_name=current_name,
        )
        if dlg.exec() != QDialog.Accepted:
            return
        updated = dlg.observatories()
        updated_preset_keys = dlg.preset_keys()
        if not updated:
            QMessageBox.warning(self, "No Observatories", "At least one observatory must remain configured.")
            return
        previous = self.obs_combo.currentText() if hasattr(self, "obs_combo") else ""
        self.observatories = updated
        self._observatory_preset_keys = {
            name: str(updated_preset_keys.get(name, "custom") or "custom")
            for name in self.observatories
        }
        self._save_custom_observatories(self.observatories, preset_keys=self._observatory_preset_keys)
        selected = previous if previous in self.observatories else next(iter(self.observatories.keys()), "")
        self._refresh_observatory_combo(selected_name=selected)
        if selected:
            self.settings.setValue("general/defaultSite", selected)
            self._on_obs_change(selected)

    @Slot()
    def _open_add_observatory_dialog(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Add Observatory")
        dlg.setObjectName("AddObservatoryDialog")
        layout = QFormLayout(dlg)

        name_edit = QLineEdit(dlg)
        name_edit.setPlaceholderText("Name")
        lookup_btn = QPushButton("Lookup", dlg)
        lookup_btn.setToolTip("Find coordinates from place name")
        _set_button_variant(lookup_btn, "secondary")
        _set_button_icon_kind(lookup_btn, "lookup")
        name_row = QWidget(dlg)
        name_row_l = QHBoxLayout(name_row)
        name_row_l.setContentsMargins(0, 0, 0, 0)
        name_row_l.setSpacing(6)
        name_row_l.addWidget(name_edit, 1)
        name_row_l.addWidget(lookup_btn)
        lat_edit = QLineEdit(dlg)
        lat_edit.setPlaceholderText("Latitude")
        lat_edit.setValidator(QDoubleValidator(-90.0, 90.0, 6, dlg))
        lon_edit = QLineEdit(dlg)
        lon_edit.setPlaceholderText("Longitude")
        lon_edit.setValidator(QDoubleValidator(-180.0, 180.0, 6, dlg))
        elev_edit = QLineEdit(dlg)
        elev_edit.setPlaceholderText("Elevation (m)")
        elev_edit.setValidator(QDoubleValidator(-1000.0, 20000.0, 2, dlg))
        limiting_mag_spin = QDoubleSpinBox(dlg)
        limiting_mag_spin.setRange(-5.0, 30.0)
        limiting_mag_spin.setDecimals(1)
        limiting_mag_spin.setSingleStep(0.1)
        limiting_mag_spin.setValue(self._current_limiting_magnitude())
        limiting_mag_spin.setToolTip("Telescope limiting magnitude used as red-warning threshold in suggested targets.")
        custom_conditions_url_edit = QLineEdit(dlg)
        custom_conditions_url_edit.setPlaceholderText("https://example.com/station.json")
        lookup_info = QLabel("", dlg)
        lookup_info.setWordWrap(True)
        _set_label_tone(lookup_info, "info")

        layout.addRow("Name:", name_row)
        layout.addRow("Lat:", lat_edit)
        layout.addRow("Lon:", lon_edit)
        layout.addRow("Elev:", elev_edit)
        layout.addRow("Lim. Mag:", limiting_mag_spin)
        layout.addRow("Custom conditions URL:", custom_conditions_url_edit)
        layout.addRow("", lookup_info)

        def _lookup_coords():
            query = name_edit.text().strip()
            if not query:
                QMessageBox.warning(dlg, "Missing Name", "Enter observatory name/place first.")
                return
            lookup_btn.setEnabled(False)
            old_txt = lookup_btn.text()
            lookup_btn.setText("...")
            try:
                lat, lon, elev, display = self._lookup_observatory_coordinates(query)
            except Exception as exc:
                QMessageBox.warning(dlg, "Lookup Failed", str(exc))
                return
            finally:
                lookup_btn.setText(old_txt)
                lookup_btn.setEnabled(True)

            lat_edit.setText(f"{lat:.6f}")
            lon_edit.setText(f"{lon:.6f}")
            if elev is not None:
                elev_edit.setText(f"{elev:.1f}")
            elif not elev_edit.text().strip():
                elev_edit.setText("0")
            lookup_info.setText(display)

        lookup_btn.clicked.connect(_lookup_coords)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dlg)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        _style_dialog_button_box(buttons)
        layout.addWidget(buttons)
        _fit_dialog_to_screen(
            dlg,
            preferred_width=720,
            preferred_height=520,
            min_width=520,
            min_height=420,
        )

        while dlg.exec() == QDialog.Accepted:
            name = name_edit.text().strip()
            if not name:
                QMessageBox.warning(self, "Invalid Observatory", "Name cannot be empty.")
                continue
            if name in self.observatories:
                QMessageBox.warning(self, "Invalid Observatory", f"Observatory '{name}' already exists.")
                continue
            try:
                template_site = self.observatories.get(self.obs_combo.currentText())
                site = Site(
                    name=name,
                    latitude=float(lat_edit.text()),
                    longitude=float(lon_edit.text()),
                    elevation=float(elev_edit.text()),
                    limiting_magnitude=float(limiting_mag_spin.value()),
                    custom_conditions_url=custom_conditions_url_edit.text().strip(),
                    telescope_diameter_mm=float(getattr(template_site, "telescope_diameter_mm", 0.0) or 0.0),
                    focal_length_mm=float(getattr(template_site, "focal_length_mm", 0.0) or 0.0),
                    pixel_size_um=float(getattr(template_site, "pixel_size_um", 0.0) or 0.0),
                    detector_width_px=int(getattr(template_site, "detector_width_px", 0) or 0),
                    detector_height_px=int(getattr(template_site, "detector_height_px", 0) or 0),
                )
            except Exception:
                QMessageBox.warning(
                    self,
                    "Invalid Coordinates",
                    "Please enter valid latitude, longitude, elevation and limiting magnitude.",
                )
                continue

            self.observatories[name] = site
            self._observatory_preset_keys[name] = "custom"
            self._save_custom_observatories()
            self._refresh_observatory_combo(selected_name=name)
            self.settings.setValue("general/defaultSite", name)
            self._on_obs_change(name)
            return
    def __init__(self):
        super().__init__()
        self.setObjectName(self.__class__.__name__)
        self._ensure_display_font_loaded()
        fallback_family = _preferred_display_font_family()
        if fallback_family not in {"", "Sans Serif"}:
            QFont.insertSubstitution("Sans Serif", fallback_family)
            QFont.insertSubstitution("SF Pro Text", fallback_family)
            QFont.insertSubstitution("SF Pro Display", fallback_family)
            bootstrap_font = QFont(fallback_family)
            bootstrap_font.setPointSize(11)
            QApplication.setFont(bootstrap_font)
        # Initialize clock_worker early so callbacks can safely reference it
        self.clock_worker = None
        # Keep references to every ClockWorker we create so they can all be stopped
        self._clock_workers: list[ClockWorker] = []
        self.last_payload = None
        # Prevent new workers once we start shutting down
        self._shutting_down = False
        self.setWindowTitle("Astronomical Observation Planner")
        self.resize(1360, 820)
        self.setMinimumSize(1180, 720)

        # Persistent user settings
        self.settings = _create_app_settings()
        self.app_storage = getattr(self.settings, "storage", None)
        try:
            self._system_locale_name = str(QLocale.system().name() or "")
        except Exception:
            self._system_locale_name = str(locale.getdefaultlocale()[0] or "")
        self._ui_language_choice = str(self.settings.value("general/uiLanguage", "system", type=str) or "system")
        self._ui_language = set_current_language(self._ui_language_choice, system_locale=self._system_locale_name)
        self._workspace_plan_id = ""
        self._active_plan_id = ""
        self._active_plan_kind = "workspace"
        self._active_plan_name = "Workspace"
        self._plan_autosave_timer = QTimer(self)
        self._plan_autosave_timer.setSingleShot(True)
        self._plan_autosave_timer.timeout.connect(self._flush_plan_autosave)
        self._suspend_plan_autosave = False
        self._defer_startup_preview_updates = False
        self._startup_restore_pending = True
        self._obs_change_finalize_pending = False
        logger.info("Using settings file: %s", self.settings.fileName())
        self._migrate_table_settings_schema()
        self._dark_enabled = self.settings.value("general/darkMode", DEFAULT_DARK_MODE, type=bool)
        self._theme_name = normalize_theme_key(self.settings.value("general/uiTheme", DEFAULT_UI_THEME, type=str))
        self._ui_font_size = _sanitize_ui_font_size(self.settings.value("general/uiFontSize", 11, type=int))
        self._display_font_family = _preferred_display_font_family()
        startup_font = QFont(self._display_font_family)
        startup_font.setPointSize(self._ui_font_size)
        QApplication.setFont(startup_font)

        # ── Polar-path visibility flags (persisted) ───────────────────────────
        self.show_sun_path  = self.settings.value("general/showSunPath",  True, type=bool)
        self.show_moon_path = self.settings.value("general/showMoonPath", True, type=bool)
        self.show_obj_path  = self.settings.value("general/showObjPath",  True, type=bool)
        
        # state holders
        self.targets: List[Target] = []
        self._auto_target_color_map: dict[str, str] = {}
        self._auto_target_color_palette_signature: tuple[str, ...] = tuple()
        self.worker: Optional[AstronomyWorker] = None  # keep reference!
        self.target_metrics: dict[str, TargetNightMetrics] = {}
        self.target_windows: dict[str, tuple[datetime, datetime]] = {}
        self._simbad_meta_cache: dict[str, tuple[Optional[float], str]] = {}
        self._simbad_compact_cache: dict[str, dict[str, object]] = {}
        self._knowledge_notes_cache: Optional[list[KnowledgeNote]] = None
        self._gaia_alerts_cache: dict[str, dict[str, str]] = {}
        self._gaia_alerts_cache_loaded_at = 0.0
        self._meta_worker: Optional[MetadataLookupWorker] = None
        self._meta_request_id = 0
        self._calc_started_at = 0.0
        self._last_calc_stats = CalcRunStats(duration_s=0.0, visible_targets=0, total_targets=0)
        self._queued_plan_run = False
        self._pending_visibility_context_key = ""
        self._clock_polar_tick = 0
        self.color_blind_mode = self.settings.value("general/colorBlindMode", False, type=bool)
        self._accent_secondary_override = self._load_accent_secondary_override(self._theme_name)
        self._radar_sweep_enabled = self.settings.value("general/radarSweepAnimation", False, type=bool)
        self._radar_sweep_speed = max(40, min(260, self.settings.value("general/radarSweepSpeed", 140, type=int)))
        self._radar_sweep_angle = 0.0
        self._plot_airmass = self.settings.value("general/plotAirmass", False, type=bool)
        self._plot_mode_animations: list[QPropertyAnimation] = []
        self._cutout_view_key = _normalize_cutout_view_key(
            self.settings.value("general/cutoutView", CUTOUT_DEFAULT_VIEW_KEY, type=str)
        )
        self._cutout_survey_key = _normalize_cutout_survey_key(
            self.settings.value("general/cutoutSurvey", CUTOUT_DEFAULT_SURVEY_KEY, type=str)
        )
        self._cutout_fov_arcmin = _sanitize_cutout_fov_arcmin(
            self.settings.value("general/cutoutFovArcmin", CUTOUT_DEFAULT_FOV_ARCMIN, type=int)
        )
        self._cutout_size_px = _sanitize_cutout_size_px(
            self.settings.value("general/cutoutSizePx", CUTOUT_DEFAULT_SIZE_PX, type=int)
        )
        self._aladin_context_factor = 1.0
        self.llm_config = LLMConfig(
            url=self.settings.value("llm/serverUrl", LLMConfig.DEFAULT_URL, type=str),
            model=self.settings.value("llm/model", LLMConfig.DEFAULT_MODEL, type=str),
            timeout_s=self.settings.value("llm/timeoutSec", LLMConfig.DEFAULT_TIMEOUT_S, type=int),
            max_tokens=self.settings.value("llm/maxTokens", LLMConfig.DEFAULT_MAX_TOKENS, type=int),
            enable_thinking=self.settings.value("llm/enableThinking", LLMConfig.DEFAULT_ENABLE_THINKING, type=bool),
            enable_chat_memory=self.settings.value(
                "llm/enableChatMemory",
                LLMConfig.DEFAULT_ENABLE_CHAT_MEMORY,
                type=bool,
            ),
        )
        self._llm_chat_font_size_pt = max(
            9,
            int(self.settings.value("llm/chatFontSizePt", LLMConfig.DEFAULT_CHAT_FONT_PT, type=int)),
        )
        self._ai_chat_spacing = str(
            self.settings.value("llm/chatSpacing", LLMConfig.DEFAULT_CHAT_SPACING, type=str)
            or LLMConfig.DEFAULT_CHAT_SPACING
        ).strip().lower()
        self._ai_chat_tint_strength = str(
            self.settings.value("llm/chatTintStrength", LLMConfig.DEFAULT_CHAT_TINT_STRENGTH, type=str)
            or LLMConfig.DEFAULT_CHAT_TINT_STRENGTH
        ).strip().lower()
        self._ai_chat_width = str(
            self.settings.value("llm/chatMessageWidth", LLMConfig.DEFAULT_CHAT_WIDTH, type=str)
            or LLMConfig.DEFAULT_CHAT_WIDTH
        ).strip().lower()
        self._ai_status_error_clear_s = max(
            0,
            int(self.settings.value("llm/statusErrorClearSec", LLMConfig.DEFAULT_STATUS_ERROR_CLEAR_S, type=int)),
        )
        self._llm_worker: Optional[LLMWorker] = None
        self._llm_active_tag = ""
        self._llm_active_requested_class = ""
        self._llm_active_primary_target: Optional[Target] = None
        self._llm_warmup_silent = False
        self._llm_startup_warmup_attempted = False
        self._ai_last_knowledge_titles: list[str] = []
        self._llm_last_warmup_at = 0.0
        self._llm_last_warmup_key: tuple[str, str] = ("", "")
        self._ai_runtime_status = ""
        self._ai_runtime_status_tone = "info"
        self._ai_messages: list[dict[str, Any]] = []
        self._ai_message_widget_refs: list[dict[str, Any]] = []
        self._ai_stream_message_index: Optional[int] = None
        self._ai_scroll_bottom_pending = False
        self._ai_stream_render_timer = QTimer(self)
        self._ai_stream_render_timer.setSingleShot(True)
        self._ai_stream_render_timer.timeout.connect(self._flush_ai_stream_render)
        self._ai_status_clear_timer = QTimer(self)
        self._ai_status_clear_timer.setSingleShot(True)
        self._ai_status_clear_timer.timeout.connect(self._clear_ai_runtime_status)
        self._cutout_manager = QNetworkAccessManager(self)
        self._cutout_manager.finished.connect(self._on_cutout_reply)
        self._cutout_request_id = 0
        self._cutout_pending_key = ""
        self._cutout_pending_name = ""
        self._cutout_pending_background = False
        self._cutout_displayed_key = ""
        self._cutout_reply: Optional[QNetworkReply] = None
        self._cutout_cache: dict[str, QPixmap] = {}
        self._cutout_cache_order: list[str] = []
        self._cutout_prefetch_queue: list[tuple[Target, str, int, int, int, int]] = []
        self._cutout_prefetch_enqueued_keys: set[str] = set()
        self._cutout_prefetch_total = 0
        self._cutout_prefetch_completed = 0
        self._cutout_prefetch_cached = 0
        self._cutout_prefetch_active = False
        self._finder_cache: dict[str, QPixmap] = {}
        self._finder_cache_order: list[str] = []
        self._bhtom_candidate_cache_key: Optional[tuple[str, str]] = None
        self._bhtom_candidate_cache: Optional[list[dict[str, object]]] = None
        self._bhtom_candidate_cache_loaded_at = 0.0
        self._bhtom_last_network_fetch_key: Optional[tuple[str, str]] = None
        self._bhtom_ranked_suggestions_cache: list[dict[str, object]] = []
        self._bhtom_candidate_prefetch_worker: Optional[BhtomCandidatePrefetchWorker] = None
        self._bhtom_candidate_prefetch_request_id = 0
        self._bhtom_observatory_cache_key: Optional[tuple[str, str]] = None
        self._bhtom_observatory_cache: Optional[list[dict[str, object]]] = None
        self._bhtom_observatory_cache_loaded_at = 0.0
        self._bhtom_observatory_worker: Optional[BhtomObservatoryPresetWorker] = None
        self._bhtom_observatory_worker_request_id = 0
        self._bhtom_observatory_loading_message = ""
        self._observatory_preset_keys: dict[str, str] = {}
        self._bhtom_worker: Optional[BhtomSuggestionWorker] = None
        self._bhtom_worker_request_id = 0
        self._bhtom_worker_mode = ""
        self._bhtom_worker_cache_key: Optional[tuple[str, str]] = None
        self._bhtom_worker_source = ""
        self._bhtom_dialog: Optional[SuggestedTargetsDialog] = None
        self._finder_workers: list[FinderChartWorker] = []
        self._finder_worker: Optional[FinderChartWorker] = None
        self._finder_request_id = 0
        self._finder_pending_key = ""
        self._finder_pending_name = ""
        self._finder_pending_background = False
        self._finder_displayed_key = ""
        self._finder_prefetch_queue: list[tuple[Target, str]] = []
        self._finder_prefetch_enqueued_keys: set[str] = set()
        self._finder_prefetch_total = 0
        self._finder_prefetch_completed = 0
        self._finder_prefetch_cached = 0
        self._finder_prefetch_active = False
        self._finder_retry_after: dict[str, float] = {}
        self._cutout_image_stacks: dict[str, QStackedLayout] = {}
        self._cutout_image_placeholders: dict[str, QWidget] = {}
        self._cutout_image_labels: dict[str, CoverImageLabel] = {}
        self._finder_timeout_timer = QTimer(self)
        self._finder_timeout_timer.setSingleShot(True)
        self._finder_timeout_timer.setInterval(FINDER_WORKER_TIMEOUT_MS)
        self._finder_timeout_timer.timeout.connect(self._on_finder_chart_timeout)
        self._startup_weather_worker: Optional[WeatherLiveWorker] = None
        self._cutout_resize_timer = QTimer(self)
        self._cutout_resize_timer.setSingleShot(True)
        self._cutout_resize_timer.setInterval(260)
        self._cutout_resize_timer.timeout.connect(self._on_cutout_resize_timeout)
        self._cutout_last_resize_signature: Optional[tuple] = None
        self._visibility_plot_refresh_timer = QTimer(self)
        self._visibility_plot_refresh_timer.setSingleShot(True)
        self._visibility_plot_refresh_timer.setInterval(0)
        self._visibility_plot_refresh_timer.timeout.connect(self._flush_visibility_plot_refresh)
        self._selected_cutout_update_timer = QTimer(self)
        self._selected_cutout_update_timer.setSingleShot(True)
        self._selected_cutout_update_timer.setInterval(90)
        self._selected_cutout_update_timer.timeout.connect(self._flush_selected_cutout_update)
        self._pending_selected_cutout_target: Optional[Target] = None
        self._last_vis_selected_names: set[str] = set()
        self._last_polar_selection_signature: tuple[str, ...] = ()
        # Keep web visibility selection highlight enabled via lightweight Plotly.restyle.
        self._visibility_web_sync_selection = True
        self._visibility_web_html_cache: dict[str, str] = {}

        # UI ------------------------------------------------
        self.table_model = TargetTableModel(self.targets, site=None, parent=self)
        self.table_view = TargetTableView()
        self._table_coordinator = TargetTableCoordinator(self)
        self._table_coordinator.bind()

        # Polar plot for alt-az projection
        self.polar_canvas = FigureCanvas(Figure(figsize=(4, 4)))
        self.polar_ax = self.polar_canvas.figure.add_subplot(projection='polar')
        self.polar_ax.set_theta_zero_location('N')
        self.polar_ax.set_theta_direction(-1)
        # Plot placeholders: targets, selected target, sun, moon
        self.polar_scatter = self.polar_ax.scatter([], [], c='blue', marker='o', s=10, label='Targets', alpha=0.5, picker=True)
        self.selected_scatter = self.polar_ax.scatter([], [], c='red', marker='x', s=40, alpha=1, label='Selected')
        self.radar_echo_glow_scatter = self.polar_ax.scatter([], [], c='cyan', marker='o', s=10, alpha=1.0, linewidths=3.2, zorder=3)
        self.radar_echo_scatter = self.polar_ax.scatter([], [], c='cyan', marker='o', s=10, alpha=1.0, linewidths=1.35, zorder=4)
        self.radar_echo_glow_scatter.set_visible(False)
        self.radar_echo_scatter.set_visible(False)
        self._radar_echo_artists: list[Any] = []
        self.radar_sweep_glow_line, = self.polar_ax.plot(
            [],
            [],
            color='cyan',
            linewidth=3.2,
            alpha=0.42,
            zorder=2,
            solid_capstyle='round',
            solid_joinstyle='round',
            antialiased=True,
        )
        self.radar_sweep_line, = self.polar_ax.plot(
            [],
            [],
            color='magenta',
            linewidth=0.9,
            alpha=0.0,
            zorder=3,
            solid_capstyle='round',
            solid_joinstyle='round',
            antialiased=True,
        )
        self.radar_sweep_core = self.polar_ax.scatter([], [], c='magenta', marker='o', s=58, alpha=0.0, linewidths=0.0, zorder=3)
        self.radar_sweep_core.set_visible(False)
        self._radar_sweep_theta_edges = np.linspace(0.0, 2.0 * math.pi, 721)
        self._radar_sweep_theta_centers = (self._radar_sweep_theta_edges[:-1] + self._radar_sweep_theta_edges[1:]) * 0.5
        self._radar_sweep_radius_edges = np.linspace(0.0, 90.0, 9)
        self._radar_sweep_mesh_values = np.zeros(
            (len(self._radar_sweep_radius_edges) - 1, len(self._radar_sweep_theta_edges) - 1),
            dtype=float,
        )
        self.radar_sweep_mesh = self.polar_ax.pcolormesh(
            self._radar_sweep_theta_edges,
            self._radar_sweep_radius_edges,
            self._radar_sweep_mesh_values,
            shading='flat',
            cmap=self._build_radar_sweep_cmap(),
            vmin=0.0,
            vmax=1.0,
            antialiased=False,
            zorder=1,
        )
        self.radar_sweep_mesh.set_visible(False)
        # Placeholder for selected-object path trace
        self.selected_trace_line = None
        # Placeholder for altitude limit circle
        self.limit_circle = None
        # Placeholder for celestial pole marker
        self.pole_marker = None
        self.sun_marker = self.polar_ax.scatter([], [], c='orange', marker='o', s=100, label='Sun')
        self.moon_marker = self.polar_ax.scatter([], [], c='silver', marker='o', s=100, label='Moon')
        # Place-holders for path lines
        self.sun_path_line  = None
        self.moon_path_line = None
        self.polar_ax.set_rlim(0, 90)
        self.polar_ax.set_rlabel_position(135)
        # Will map scatter points to target indices for picking
        self.polar_indices: list[int] = []
        # Label cardinal directions on the polar plot
        self.polar_ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
        self.polar_ax.set_xticklabels(['N', 'E', 'S', 'W'])
        # Draw cardinal labels in white for visibility on dark background
        self.polar_ax.tick_params(axis='x', colors='white', pad=0)
        # Radial ticks for altitudes 20°, 40°, 60°, and 80° (r = 90 - altitude), labels hidden
        alt_ticks = [20, 40, 60, 80]
        r_ticks = [90 - a for a in alt_ticks]
        self.polar_ax.set_yticks(r_ticks)
        # Hide radial tick labels for a cleaner look
        self.polar_ax.set_yticklabels([])
        self.polar_ax.tick_params(axis='y', pad=1)
        # Keep the polar axes background white
        self.polar_ax.set_facecolor('white')
        self.polar_canvas.figure.patch.set_alpha(0)
        # Make the canvas widget itself transparent
        self.polar_canvas.setAttribute(Qt.WA_TranslucentBackground)
        self.polar_canvas.setStyleSheet("background: transparent;")
        # Update polar when table selection changes
        self.table_view.selectionModel().selectionChanged.connect(self._update_polar_selection)
        # Also highlight visibility curves on selection
        self.table_view.selectionModel().selectionChanged.connect(self._update_vis_selection)
        self.table_view.selectionModel().selectionChanged.connect(self._update_selected_details)
        # Holder for visibility line artists per target, with over-limit flag
        self.vis_lines: list[tuple[str, Any, bool]] = []
        self.limit_line = None
        # Connect pick event for polar scatter
        self.polar_canvas.mpl_connect('pick_event', self._on_polar_pick)
        self.polar_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.polar_canvas.setMinimumWidth(250)
        self.polar_canvas.setMinimumHeight(220)
        self.polar_canvas.setMaximumWidth(16777215)
        # Keep enough edge room so cardinal labels are not clipped, but let the
        # polar plot fill more of the available canvas area on large screens.
        self.polar_canvas.figure.subplots_adjust(left=0.03, right=0.97, bottom=0.06, top=0.97)
        self._radar_target_coords = np.empty((0, 2), dtype=float)
        self._radar_echo_strengths = np.zeros(0, dtype=float)
        self._radar_sweep_timer = QTimer(self)
        self._radar_sweep_timer.setInterval(16)
        try:
            self._radar_sweep_timer.setTimerType(Qt.PreciseTimer)
        except Exception:
            pass
        self._radar_sweep_timer.timeout.connect(self._advance_radar_sweep)
        self._radar_sweep_clock = QElapsedTimer()

        # Debounce frequent requests for plotting
        self._replot_timer = QTimer(self)
        self._replot_timer.setSingleShot(True)
        self._replot_timer.setInterval(300)  # ms
        self._replot_timer.timeout.connect(self._run_plan)

        self.date_edit = QDateEdit()
        # Initialize to current observing night
        self._change_to_today()
        self.date_edit.setCalendarPopup(True)
        self.date_edit.setMinimumWidth(130)
        self.date_edit.setMaximumWidth(130)

        # Site coordinate inputs and limit-altitude
        self.lat_edit = QLineEdit("-24.59", self)
        self.lat_edit.setMinimumWidth(92)
        self.lat_edit.setMaximumWidth(95)
        self.lat_edit.setValidator(QDoubleValidator(-90.0, 90.0, 6, self))
        self.lon_edit = QLineEdit("-70.19", self)
        self.lon_edit.setMinimumWidth(92)
        self.lon_edit.setMaximumWidth(95)
        self.lon_edit.setValidator(QDoubleValidator(-180.0, 180.0, 6, self))
        self.elev_edit = QLineEdit("2800", self)
        self.elev_edit.setMinimumWidth(70)
        self.elev_edit.setMaximumWidth(70)
        self.elev_edit.setValidator(QDoubleValidator(-1000.0, 20000.0, 2, self))
        self.limit_spin = QSpinBox(minimum=-90, maximum=90, value=35)
        self.limit_spin.setMinimumWidth(74)
        self.limit_spin.setMaximumWidth(74)
        init_sun_alt_limit = self.settings.value("general/sunAltLimit", -10, type=int)
        self.sun_alt_limit_spin = QSpinBox(minimum=-90, maximum=90, value=init_sun_alt_limit)
        self.sun_alt_limit_spin.setMinimumWidth(84)
        self.sun_alt_limit_spin.setMaximumWidth(90)
        self.sun_alt_limit_spin.setToolTip("Sun altitude threshold for start/end of observing window (deg)")

        # Observatory selection
        self._builtin_observatories = {
            "OCM": Site(
                name="OCM",
                latitude=-24.59,
                longitude=-70.19,
                elevation=2800,
                limiting_magnitude=DEFAULT_LIMITING_MAGNITUDE,
            ),
            "Białków": Site(
                name="Białków",
                latitude=51.474248,
                longitude=16.657821,
                elevation=128,
                limiting_magnitude=DEFAULT_LIMITING_MAGNITUDE,
            ),
            "Roque de los Muchachos": Site(
                name="Roque de los Muchachos",
                latitude=28.4522,
                longitude=-17.5330,
                elevation=2426,
                limiting_magnitude=DEFAULT_LIMITING_MAGNITUDE,
            ),
        }
        loaded_observatories, loaded_preset_keys = self._load_custom_observatories()
        if loaded_observatories:
            # SQLite is the source of truth for user observatories.
            self.observatories = dict(loaded_observatories)
            self._observatory_preset_keys = {
                name: str(loaded_preset_keys.get(name, "custom") or "custom")
                for name in self.observatories
            }
        else:
            self.observatories = dict(self._builtin_observatories)
            self._observatory_preset_keys = {name: "custom" for name in self.observatories}
            self._save_custom_observatories(self.observatories, preset_keys=self._observatory_preset_keys)
        self.obs_combo = QComboBox()
        self.obs_combo.addItems(self.observatories.keys())
        init_site = self.settings.value("general/defaultSite", "OCM", type=str)
        if init_site not in self.observatories:
            init_site = "OCM" if "OCM" in self.observatories else (next(iter(self.observatories.keys()), ""))
        self.obs_combo.blockSignals(True)
        self.obs_combo.setCurrentText(init_site)
        self.obs_combo.blockSignals(False)
        self.obs_combo.currentTextChanged.connect(self._on_obs_change)
        # Prime the initial site fields without kicking background work before
        # workspace restore decides what the real startup context should be.
        if init_site:
            self._prime_selected_observatory(init_site)
        self.obs_combo.setMinimumContentsLength(26)
        self.obs_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.obs_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._update_obs_combo_widths()
        self.add_obs_btn = QToolButton()
        self.add_obs_btn.setText("+")
        self.add_obs_btn.setToolTip("Open observatory manager")
        self.add_obs_btn.setFixedSize(24, 24)
        self.add_obs_btn.clicked.connect(self._open_observatory_manager)

        # Debounced connections for date and limit spin
        self.date_edit.dateChanged.connect(lambda _: self._replot_timer.start())
        self.date_edit.dateChanged.connect(lambda _: self._refresh_weather_window_context())
        self.date_edit.dateChanged.connect(lambda *_: self._schedule_plan_autosave())
        self.limit_spin.valueChanged.connect(self._update_limit)
        self.sun_alt_limit_spin.valueChanged.connect(self._on_sun_alt_limit_changed)
        # Re-plot when latitude, longitude, or elevation is changed
        self.lat_edit.editingFinished.connect(self._on_site_inputs_changed)
        self.lon_edit.editingFinished.connect(self._on_site_inputs_changed)
        self.elev_edit.editingFinished.connect(self._on_site_inputs_changed)

        self._use_visibility_web = bool(_HAS_QTWEBENGINE and _HAS_PLOTLY and QWebEngineView is not None)
        self._visibility_web_minute_key = ""
        self._visibility_web_has_content = False
        self.plot_canvas = None
        self.ax_alt = None
        self.plot_toolbar = None
        self.visibility_web = None
        self.visibility_loading_widget = None
        self.visibility_plot_stack = None
        self.visibility_plot_host = None
        self._visibility_plot_widgets_ready = False
        self._visibility_bootstrap_placeholder = self._build_web_loading_placeholder(
            "Visibility Plot",
            "Preparing chart…",
        )
        self._visibility_bootstrap_placeholder.setMinimumHeight(320)

        self.plot_mode_widget = QWidget(self)
        self.plot_mode_widget.setObjectName("PlotModeWidget")
        self.plot_mode_widget.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        plot_mode_layout = QHBoxLayout(self.plot_mode_widget)
        plot_mode_layout.setContentsMargins(12, 2, 12, 2)
        plot_mode_layout.setSpacing(12)
        self.plot_mode_alt_label = QLabel("Altitude", self.plot_mode_widget)
        self.plot_mode_airmass_label = QLabel("Airmass", self.plot_mode_widget)
        for label in (self.plot_mode_alt_label, self.plot_mode_airmass_label):
            effect = QGraphicsOpacityEffect(label)
            effect.setOpacity(1.0)
            label.setGraphicsEffect(effect)
            label._opacity_effect = effect  # type: ignore[attr-defined]
        self.airmass_toggle_btn = NeonToggleSwitch(self.plot_mode_widget)
        self.airmass_toggle_btn.setObjectName("PlotModeSwitch")
        self.airmass_toggle_btn.setToolTip("Switch the main plot Y-axis between altitude and airmass")
        self.airmass_toggle_btn.setChecked(bool(self._plot_airmass))
        self.airmass_toggle_btn.toggled.connect(self._on_plot_mode_switch_changed)
        plot_mode_layout.addWidget(self.plot_mode_alt_label)
        plot_mode_layout.addWidget(self.airmass_toggle_btn)
        plot_mode_layout.addWidget(self.plot_mode_airmass_label)
        self._refresh_plot_mode_switch()
        self._update_plot_mode_label_metrics()
        if self._use_visibility_web:
            self.plot_controls_bar = QWidget(self)
            self.plot_controls_bar.setObjectName("PlotControlsBar")
            plot_controls_layout = QHBoxLayout(self.plot_controls_bar)
            plot_controls_layout.setContentsMargins(6, 2, 6, 2)
            plot_controls_layout.setSpacing(8)
            self.plot_hint_label = QLabel(
                "Interactive visibility chart. Use Plotly controls in the top-right for zoom and pan.",
                self.plot_controls_bar,
            )
            self.plot_hint_label.setObjectName("SectionHint")
            plot_controls_layout.addWidget(self.plot_hint_label, 1)
            plot_controls_layout.addWidget(self.plot_mode_widget, 0, Qt.AlignRight)
        else:
            self.plot_controls_bar = None
            self.plot_hint_label = None

        # top controls
        # Date selector with previous/next day buttons
        self.prev_day_btn = QToolButton()
        self.prev_day_btn.setObjectName("DateNavButton")
        self.prev_day_btn.setToolTip("Previous day")
        self.prev_day_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.prev_day_btn.setIconSize(QSize(17, 17))
        self.prev_day_btn.setFixedSize(30, 30)
        self.prev_day_btn.clicked.connect(lambda: self._change_date(-1))

        self.today_btn = QToolButton()
        self.today_btn.setObjectName("DateNavButton")
        self.today_btn.setToolTip("Today")
        self.today_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.today_btn.setIconSize(QSize(17, 17))
        self.today_btn.setFixedSize(30, 30)
        self.today_btn.clicked.connect(lambda: self._change_to_today())

        self.next_day_btn = QToolButton()
        self.next_day_btn.setObjectName("DateNavButton")
        self.next_day_btn.setToolTip("Next day")
        self.next_day_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.next_day_btn.setIconSize(QSize(17, 17))
        self.next_day_btn.setFixedSize(30, 30)
        self.next_day_btn.clicked.connect(lambda: self._change_date(1))
        self._refresh_date_nav_icons()

        # --- Compact date chooser ---
        date_widget = QWidget()
        date_layout = QHBoxLayout()
        date_layout.setContentsMargins(0, 0, 0, 0)
        date_layout.setSpacing(4)
        date_layout.addWidget(self.prev_day_btn)
        date_layout.addWidget(self.today_btn)
        date_layout.addWidget(self.date_edit)
        date_layout.addWidget(self.next_day_btn)
        date_widget.setLayout(date_layout)

        self.coord_error_label = QLabel("")
        self.coord_error_label.setWordWrap(True)
        _set_label_tone(self.coord_error_label, "error")

        session_layout = QVBoxLayout()
        session_layout.setContentsMargins(2, 2, 2, 2)
        session_layout.setSpacing(3)
        session_row_top = QHBoxLayout()
        session_row_top.setSpacing(5)
        session_row_top.addWidget(QLabel("Obs"))
        session_row_top.addWidget(self.obs_combo, 2)
        session_row_top.addWidget(self.add_obs_btn)
        session_row_top.addWidget(QLabel("Date"))
        session_row_top.addWidget(date_widget, 1)
        session_row_top.addWidget(QLabel("Alt"))
        session_row_top.addWidget(self.limit_spin)
        session_row_top.addWidget(QLabel("Sun ≤"))
        session_row_top.addWidget(self.sun_alt_limit_spin)
        session_row_top.addStretch(1)
        session_layout.addLayout(session_row_top)

        # Coordinates are managed by observatory selection/add dialog.
        # Keep editors hidden as internal data holders for calculations.
        self.lat_edit.setVisible(False)
        self.lon_edit.setVisible(False)
        self.elev_edit.setVisible(False)

        self.coord_error_label.setVisible(False)

        self.min_moon_sep_spin = QSpinBox(minimum=0, maximum=180, value=self.settings.value("general/minMoonSep", 0, type=int))
        self.min_moon_sep_spin.setMinimumWidth(84)
        self.min_moon_sep_spin.setMaximumWidth(92)

        self.min_score_spin = QSpinBox(minimum=0, maximum=100, value=self.settings.value("general/minScore", 0, type=int))
        self.min_score_spin.setMinimumWidth(84)
        self.min_score_spin.setMaximumWidth(92)

        self.hide_observed_chk = QCheckBox("Hide observed")
        self.hide_observed_chk.setChecked(self.settings.value("general/hideObserved", False, type=bool))
        self.min_moon_sep_spin.valueChanged.connect(self._on_filter_change)
        self.min_score_spin.valueChanged.connect(self._on_filter_change)
        self.hide_observed_chk.stateChanged.connect(lambda _: self._on_filter_change())

        # Sun/Moon visibility toggles
        self.sun_check = QCheckBox("Show Sun")
        self.sun_check.setChecked(False)
        self.sun_check.stateChanged.connect(self._toggle_visibility)

        self.moon_check = QCheckBox("Show Moon")
        self.moon_check.setChecked(True)
        self.moon_check.stateChanged.connect(self._toggle_visibility)

        for chk, min_w in (
            (self.hide_observed_chk, 132),
            (self.sun_check, 102),
            (self.moon_check, 112),
        ):
            chk.setMinimumWidth(min_w)
            chk.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)

        filters_layout = QVBoxLayout()
        filters_layout.setContentsMargins(2, 2, 2, 2)
        filters_layout.setSpacing(3)

        filters_row_top = QHBoxLayout()
        filters_row_top.setSpacing(6)
        filters_row_top.addWidget(QLabel("Moon ≥"))
        filters_row_top.addWidget(self.min_moon_sep_spin)
        filters_row_top.addSpacing(6)
        filters_row_top.addWidget(QLabel("Score ≥"))
        filters_row_top.addWidget(self.min_score_spin)
        filters_row_top.addStretch(1)
        filters_layout.addLayout(filters_row_top)

        filters_row_bottom = QHBoxLayout()
        filters_row_bottom.setSpacing(8)
        filters_row_bottom.addWidget(self.hide_observed_chk)
        filters_row_bottom.addWidget(self.sun_check)
        filters_row_bottom.addWidget(self.moon_check)
        filters_row_bottom.addStretch(1)
        filters_layout.addLayout(filters_row_bottom)

        add_btn = QPushButton("Add")
        add_btn.setMinimumHeight(24)
        add_btn.clicked.connect(self._add_target_dialog)
        add_btn.setToolTip("Add a new target to the plan")
        _set_button_variant(add_btn, "primary")
        _set_button_icon_kind(add_btn, "add")

        toggle_obs_btn = QPushButton("Observed")
        toggle_obs_btn.setMinimumHeight(24)
        toggle_obs_btn.clicked.connect(self._toggle_observed_selected)
        toggle_obs_btn.setToolTip("Toggle observed state for the selected target")
        _set_button_variant(toggle_obs_btn, "neutral")
        _set_button_icon_kind(toggle_obs_btn, "toggle")
        self.edit_object_btn = QPushButton("Edit")
        self.edit_object_btn.setMinimumHeight(24)
        self.edit_object_btn.clicked.connect(self._edit_object_for_selected)
        self.edit_object_btn.setEnabled(False)
        self.edit_object_btn.setToolTip("Edit the selected target")
        _set_button_variant(self.edit_object_btn, "neutral")
        _set_button_icon_kind(self.edit_object_btn, "edit")

        self.open_saved_plan_btn = QPushButton("Open Plan")
        self.open_saved_plan_btn.setMinimumHeight(24)
        self.open_saved_plan_btn.clicked.connect(self._open_saved_plan)
        self.open_saved_plan_btn.setToolTip("Open a saved plan from local storage")
        _set_button_variant(self.open_saved_plan_btn, "secondary")
        _set_button_icon_kind(self.open_saved_plan_btn, "load")
        self.save_plan_as_btn = QPushButton("Save Plan")
        self.save_plan_as_btn.setMinimumHeight(24)
        self.save_plan_as_btn.clicked.connect(self._save_plan_as)
        self.save_plan_as_btn.setToolTip("Save the current plan under a chosen name")
        _set_button_variant(self.save_plan_as_btn, "secondary")
        _set_button_icon_kind(self.save_plan_as_btn, "save")
        suggest_targets_btn = QPushButton("Suggest")
        suggest_targets_btn.setMinimumHeight(24)
        suggest_targets_btn.clicked.connect(self._ai_suggest_targets)
        suggest_targets_btn.setToolTip(
            "Check cached BHTOM targets first (TTL 1h). "
            "A fresh fetch starts only when the cache is missing or expired."
        )
        _set_button_variant(suggest_targets_btn, "primary")
        _set_button_icon_kind(suggest_targets_btn, "suggest")
        self.quick_targets_btn = QPushButton("Quick")
        self.quick_targets_btn.setMinimumHeight(24)
        self.quick_targets_btn.clicked.connect(self._quick_add_suggested_targets)
        self._update_quick_targets_button_tooltip()
        _set_button_variant(self.quick_targets_btn, "primary")
        _set_button_icon_kind(self.quick_targets_btn, "quick")
        self.seestar_session_btn = QPushButton("Seestar")
        self.seestar_session_btn.setMinimumHeight(24)
        self.seestar_session_btn.clicked.connect(self._open_seestar_session)
        self.seestar_session_btn.setToolTip("Open Seestar session settings")
        _set_button_variant(self.seestar_session_btn, "neutral")
        _set_button_icon_kind(self.seestar_session_btn, "seestar")
        self.weather_btn = QPushButton("Weather")
        self.weather_btn.setMinimumHeight(24)
        self.weather_btn.setToolTip("Open weather workspace (forecast/conditions/cloud/satellite tabs)")
        self.weather_btn.clicked.connect(self._open_weather_window)
        _set_button_variant(self.weather_btn, "secondary")
        _set_button_icon_kind(self.weather_btn, "weather")
        self.ai_toggle_btn = QPushButton("AI")
        self.ai_toggle_btn.setMinimumHeight(24)
        self.ai_toggle_btn.setCheckable(True)
        self.ai_toggle_btn.setChecked(False)
        self.ai_toggle_btn.setToolTip("Show or hide the AI assistant window")
        self.ai_toggle_btn.toggled.connect(self._toggle_ai_panel)
        _set_button_variant(self.ai_toggle_btn, "secondary")
        _set_button_icon_kind(self.ai_toggle_btn, "ai")

        session_strip = QWidget()
        session_strip.setObjectName("SessionStrip")
        session_strip.setLayout(session_layout)
        filters_strip = QWidget()
        filters_strip.setObjectName("FiltersStrip")
        filters_strip.setLayout(filters_layout)

        top_controls = QFrame()
        top_controls.setObjectName("TopControlsBar")
        top_controls_l = QHBoxLayout()
        top_controls_l.setContentsMargins(6, 4, 6, 4)
        top_controls_l.setSpacing(8)
        top_controls_l.addWidget(session_strip, 3)
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setFrameShadow(QFrame.Sunken)
        top_controls_l.addWidget(sep)
        top_controls_l.addWidget(filters_strip, 2)
        top_controls.setLayout(top_controls_l)
        top_controls.setMinimumHeight(70)
        top_controls.setMaximumHeight(140)
        _set_dynamic_property(top_controls, "accented", True)

        actions_bar = QFrame()
        actions_bar.setObjectName("ActionsBar")
        actions_l = QHBoxLayout()
        actions_l.setContentsMargins(10, 4, 10, 4)
        actions_l.setSpacing(8)
        actions_l.addWidget(add_btn)
        actions_l.addWidget(suggest_targets_btn)
        actions_l.addWidget(self.quick_targets_btn)
        actions_l.addWidget(self.seestar_session_btn)
        actions_l.addWidget(self.weather_btn)
        actions_l.addWidget(toggle_obs_btn)
        actions_l.addWidget(self.edit_object_btn)
        actions_l.addWidget(self.open_saved_plan_btn)
        actions_l.addWidget(self.save_plan_as_btn)
        actions_l.addWidget(self.ai_toggle_btn)
        actions_l.addStretch(1)
        actions_bar.setLayout(actions_l)
        actions_bar.setMinimumHeight(40)
        actions_bar.setMaximumHeight(50)
        self._update_seestar_button_tooltip()
        _set_dynamic_property(actions_bar, "accented", True)

        plot_card = QFrame()
        plot_card.setObjectName("PlotCard")
        plot_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        _set_dynamic_property(plot_card, "accented", True)
        plot_l = QVBoxLayout()
        plot_l.setContentsMargins(2, 2, 2, 2)
        plot_l.setSpacing(2)
        if self._use_visibility_web and self.plot_controls_bar is not None:
            if self.plot_controls_bar is not None:
                plot_l.addWidget(self.plot_controls_bar)
        plot_l.addWidget(self._visibility_bootstrap_placeholder, 1)
        plot_card.setLayout(plot_l)
        self.plot_card = plot_card
        self.plot_card_layout = plot_l

        self.table_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.table_view.setAlternatingRowColors(True)

        # Info panel for sun/moon events
        self.info_widget = QWidget()
        info_layout = QHBoxLayout()
        info_layout.setContentsMargins(0, 0, 0, 0)
        info_layout.setSpacing(10)

        left_info_widget = QWidget()
        left_info_form = QFormLayout()
        left_info_form.setContentsMargins(0, 0, 0, 0)
        left_info_form.setHorizontalSpacing(6)
        left_info_form.setVerticalSpacing(2)
        left_info_form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        left_info_widget.setLayout(left_info_form)

        info_sep = QFrame()
        info_sep.setFrameShape(QFrame.VLine)
        info_sep.setFrameShadow(QFrame.Sunken)

        right_info_widget = QWidget()
        right_info_form = QFormLayout()
        right_info_form.setContentsMargins(0, 0, 0, 0)
        right_info_form.setHorizontalSpacing(6)
        right_info_form.setVerticalSpacing(2)
        right_info_form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        right_info_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        right_info_widget.setLayout(right_info_form)

        info_layout.addWidget(left_info_widget, 1)
        info_layout.addWidget(info_sep)
        info_layout.addWidget(right_info_widget, 1)
        self.info_widget.setLayout(info_layout)

        self.sunrise_label = QLabel("-")
        self.sunset_label = QLabel("-")
        self.moonrise_label = QLabel("-")
        self.moonset_label = QLabel("-")
        self.moonphase_bar = QProgressBar(self)
        self.moonphase_bar.setRange(0, 100)
        self.moonphase_bar.setValue(0)
        self.moonphase_bar.setFormat("%p%%")
        self.moonphase_bar.setTextVisible(True)
        self.moonphase_bar.setMinimumWidth(130)
        self.moonphase_bar.setMaximumWidth(190)
        self.moonphase_bar.setMinimumHeight(20)
        # Current time labels
        self.localtime_label = QLabel("-")
        self.utctime_label = QLabel("-")
        # Fonts for info panel: labels bold, values italic
        label_font = QFont(self.font())
        label_font.setPointSize(10)
        label_font.setWeight(QFont.Weight.DemiBold)
        value_font = QFont(self.font())
        value_font.setPointSize(10)
        self.info_label_font = label_font
        self.info_value_font = value_font
        value_line_h = max(16, QFontMetrics(self.info_value_font).lineSpacing())
        self.sun_alt_label = QLabel("-")
        self.moon_alt_label = QLabel("-")
        self.sidereal_label = QLabel("-")

        # Selected target details
        self.sel_name_label = QLabel("-")
        self.sel_type_label = QLabel("-")
        self.sel_score_label = QLabel("-")
        self.sel_window_label = QLabel("-")
        self.sel_notes_label = QLabel("-")
        for dynamic_value_label in (self.sel_name_label, self.sel_type_label, self.sel_notes_label):
            dynamic_value_label.setWordWrap(True)
            dynamic_value_label.setMinimumWidth(0)
            dynamic_value_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
            dynamic_value_label.setMinimumHeight(value_line_h + 2)
        # Reserve two text lines for target/type so short values don't collapse the row
        # and medium-length values don't cause visible "jumping" between selections.
        reserved_two_lines_h = value_line_h * 2 + 4
        self.sel_name_label.setMinimumHeight(reserved_two_lines_h)
        self.sel_type_label.setMinimumHeight(reserved_two_lines_h)

        details_row = QWidget()
        details_layout = QVBoxLayout()
        details_layout.setContentsMargins(0, 0, 0, 0)
        details_layout.setSpacing(2)
        details_layout.addWidget(self.sel_notes_label)
        details_row.setLayout(details_layout)

        def _add_info_row(form: QFormLayout, row_idx: int, icon_kind: str, title: str, value_widget: QWidget):
            title_widget = QWidget()
            title_layout = QHBoxLayout(title_widget)
            title_layout.setContentsMargins(0, 0, 0, 0)
            title_layout.setSpacing(7)
            icon_label = QLabel(title_widget)
            _set_detail_icon_kind(icon_label, icon_kind, 16)
            title_label = QLabel(title, title_widget)
            title_label.setFont(self.info_label_font)
            title_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            title_layout.addWidget(icon_label, 0, Qt.AlignVCenter)
            title_layout.addWidget(title_label, 0, Qt.AlignVCenter)
            title_layout.addStretch(1)
            if isinstance(value_widget, QLabel):
                value_widget.setFont(self.info_value_font)
                value_widget.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            form.insertRow(row_idx, title_widget, value_widget)
            return title_label

        _add_info_row(left_info_form, 0, "clock", "Local time:", self.localtime_label)
        _add_info_row(left_info_form, 1, "utc", "UTC time:", self.utctime_label)
        _add_info_row(left_info_form, 2, "sidereal", "Sidereal time:", self.sidereal_label)
        _add_info_row(left_info_form, 3, "sunset", "Sunset:", self.sunset_label)
        _add_info_row(left_info_form, 4, "sunrise", "Sunrise:", self.sunrise_label)
        _add_info_row(left_info_form, 5, "moonphase", "Moon phase:", self.moonphase_bar)
        _add_info_row(left_info_form, 6, "moonrise", "Moonrise:", self.moonrise_label)
        _add_info_row(left_info_form, 7, "moonset", "Moonset:", self.moonset_label)

        _add_info_row(right_info_form, 0, "sun", "Sun altitude:", self.sun_alt_label)
        _add_info_row(right_info_form, 1, "moon", "Moon altitude:", self.moon_alt_label)
        _add_info_row(right_info_form, 2, "target", "Selected target:", self.sel_name_label)
        self.sel_type_title_label = _add_info_row(right_info_form, 3, "describe", "Type / Last Mag:", self.sel_type_label)
        _add_info_row(right_info_form, 4, "sidereal", "Score:", self.sel_score_label)
        _add_info_row(right_info_form, 5, "window", "Best window:", self.sel_window_label)
        _add_info_row(right_info_form, 6, "notes", "Notes:", details_row)
        self.sel_name_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.sel_type_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.sel_notes_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        self.info_widget.setContentsMargins(0, 0, 0, 0)

        table_card = QFrame()
        table_card.setObjectName("TableCard")
        table_card.setMinimumHeight(180)
        _set_dynamic_property(table_card, "accented", True)
        table_l = QVBoxLayout()
        table_l.setContentsMargins(2, 2, 2, 2)
        table_l.setSpacing(2)
        table_title_row = QWidget()
        table_title_row_l = QHBoxLayout(table_title_row)
        table_title_row_l.setContentsMargins(0, 0, 0, 0)
        table_title_row_l.setSpacing(8)
        table_title = QLabel("Targets")
        table_title.setObjectName("SectionTitle")
        table_hint = QLabel("Left-click: select row | Double left-click (Name): edit object | Right-click: more options")
        table_hint.setObjectName("SectionHint")
        table_title_row_l.addWidget(table_title)
        table_title_row_l.addWidget(table_hint)
        table_title_row_l.addStretch(1)
        table_l.addWidget(table_title_row)
        table_l.addWidget(self.table_view)
        table_card.setLayout(table_l)

        polar_card = QFrame()
        polar_card.setObjectName("PolarCard")
        _set_dynamic_property(polar_card, "accented", True)
        polar_l = QVBoxLayout()
        polar_l.setContentsMargins(2, 2, 2, 2)
        polar_l.setSpacing(1)
        polar_title = QLabel("Sky View")
        polar_title.setObjectName("SectionTitle")
        polar_l.addWidget(polar_title)

        cutout_frame = QFrame()
        cutout_frame.setObjectName("CutoutFrame")
        _set_dynamic_property(cutout_frame, "accented", True)
        cutout_frame.setMinimumWidth(220)
        cutout_frame.setMaximumWidth(560)
        cutout_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        cutout_l = QVBoxLayout()
        cutout_l.setContentsMargins(0, 0, 0, 0)
        cutout_l.setSpacing(0)
        self.cutout_tabs = _configure_tab_widget(QTabWidget(self))
        self.cutout_tabs.setTabPosition(QTabWidget.North)
        self.cutout_tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        aladin_host, self.aladin_image_label = self._create_cutout_image_stack(
            self.cutout_tabs,
            kind="aladin",
            title="Aladin",
        )
        self.aladin_image_label.set_zoom_enabled(True)
        self.aladin_image_label.set_overlay_painter(self._paint_aladin_static_overlay)
        self.aladin_image_label.setToolTip(
            "Mouse wheel: zoom (wheel down at 1x loads wider field) | Drag: pan | Double left-click: reset zoom"
        )
        self.aladin_image_label.zoomOutLimitReached.connect(self._aladin_expand_context)
        self.aladin_image_label.resized.connect(self._schedule_cutout_resize_refresh)
        # Backward-compatible alias used by existing cutout helpers.
        self.cutout_image_label = self.aladin_image_label

        finder_host, self.finder_image_label = self._create_cutout_image_stack(
            self.cutout_tabs,
            kind="finder",
            title="Finder Chart",
        )
        self.finder_image_label.resized.connect(self._schedule_cutout_resize_refresh)

        aladin_tab = QWidget(self.cutout_tabs)
        aladin_tab.setProperty("cutout_page", True)
        aladin_tab.setAttribute(Qt.WA_StyledBackground, True)
        aladin_tab_l = QHBoxLayout(aladin_tab)
        aladin_tab_l.setContentsMargins(0, 0, 0, 0)
        aladin_tab_l.setSpacing(2)
        aladin_tab_l.addWidget(aladin_host, 1)
        aladin_zoom_col = QWidget(aladin_tab)
        aladin_zoom_col.setProperty("cutout_tool_col", True)
        aladin_zoom_col.setAttribute(Qt.WA_StyledBackground, True)
        aladin_zoom_col_l = QVBoxLayout(aladin_zoom_col)
        aladin_zoom_col_l.setContentsMargins(2, 2, 2, 2)
        aladin_zoom_col_l.setSpacing(4)
        zoom_btn_size = 30
        aladin_zoom_in_btn = QToolButton(aladin_zoom_col)
        aladin_zoom_in_btn.setText("+")
        aladin_zoom_in_btn.setToolTip("Zoom in")
        aladin_zoom_in_btn.setFixedSize(zoom_btn_size, zoom_btn_size)
        aladin_zoom_in_btn.clicked.connect(self._aladin_zoom_in)
        aladin_zoom_out_btn = QToolButton(aladin_zoom_col)
        aladin_zoom_out_btn.setText("−")
        aladin_zoom_out_btn.setToolTip("Zoom out")
        aladin_zoom_out_btn.setFixedSize(zoom_btn_size, zoom_btn_size)
        aladin_zoom_out_btn.clicked.connect(self._aladin_zoom_out)
        aladin_zoom_reset_btn = QToolButton(aladin_zoom_col)
        aladin_zoom_reset_btn.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
        aladin_zoom_reset_btn.setIconSize(QSize(14, 14))
        aladin_zoom_reset_btn.setToolTip("Reset zoom to 1:1")
        aladin_zoom_reset_btn.setFixedSize(zoom_btn_size, zoom_btn_size)
        aladin_zoom_reset_btn.clicked.connect(self._aladin_zoom_reset)
        aladin_zoom_col_l.addWidget(aladin_zoom_in_btn)
        aladin_zoom_col_l.addWidget(aladin_zoom_out_btn)
        aladin_zoom_col_l.addWidget(aladin_zoom_reset_btn)
        aladin_zoom_col_l.addStretch(1)
        aladin_tab_l.addWidget(aladin_zoom_col, 0, Qt.AlignTop)
        finder_tab = QWidget(self.cutout_tabs)
        finder_tab.setProperty("cutout_page", True)
        finder_tab.setAttribute(Qt.WA_StyledBackground, True)
        finder_tab_l = QVBoxLayout(finder_tab)
        finder_tab_l.setContentsMargins(0, 0, 0, 0)
        finder_tab_l.addWidget(finder_host, 1)
        self.cutout_tabs.addTab(aladin_tab, "Aladin")
        self.cutout_tabs.addTab(finder_tab, "Finder chart")
        self.cutout_tabs.currentChanged.connect(self._on_cutout_tab_changed)
        self.cutout_tabs.setCurrentIndex(1 if self._cutout_view_key == "finderchart" else 0)
        cutout_l.addWidget(self.cutout_tabs, 1)
        cutout_frame.setLayout(cutout_l)

        sky_row = QHBoxLayout()
        sky_row.setContentsMargins(0, 0, 0, 0)
        sky_row.setSpacing(2)
        sky_row.addWidget(self.polar_canvas, 2)
        sky_row.addWidget(cutout_frame, 3)
        polar_l.addLayout(sky_row, stretch=1)
        polar_card.setLayout(polar_l)

        info_card = QFrame()
        info_card.setObjectName("InfoCard")
        _set_dynamic_property(info_card, "accented", True)
        info_l = QVBoxLayout()
        info_l.setContentsMargins(2, 2, 2, 2)
        info_l.setSpacing(2)
        info_title = QLabel("Night Details")
        info_title.setObjectName("SectionTitle")
        self.info_title_label = info_title
        info_l.addWidget(info_title)
        info_scroll = QScrollArea()
        info_scroll.setWidgetResizable(True)
        info_scroll.setFrameShape(QFrame.NoFrame)
        info_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        info_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        info_scroll.setWidget(self.info_widget)
        self.info_scroll = info_scroll
        info_l.addWidget(info_scroll, stretch=1)
        info_card.setLayout(info_l)
        self.info_card = info_card
        info_card.setMinimumHeight(220)
        self._night_details_fixed_min_width = max(330, self.info_widget.minimumSizeHint().width() + 26)

        right_dashboard = QSplitter(Qt.Vertical)
        right_dashboard.addWidget(polar_card)
        right_dashboard.addWidget(info_card)
        right_dashboard.setHandleWidth(1)
        right_dashboard.setStretchFactor(0, 3)
        right_dashboard.setStretchFactor(1, 2)
        right_dashboard.setSizes([390, 300])
        right_dashboard.setMinimumWidth(460)
        self._right_dashboard_base_min_width = 460
        right_dashboard.setMaximumWidth(16777215)
        right_dashboard.setChildrenCollapsible(False)
        self.right_dashboard = right_dashboard
        self._update_night_details_constraints()

        top_splitter = QSplitter(Qt.Horizontal)
        top_splitter.addWidget(plot_card)
        top_splitter.addWidget(right_dashboard)
        top_splitter.setHandleWidth(1)
        top_splitter.setStretchFactor(0, 8)
        top_splitter.setStretchFactor(1, 1)
        top_splitter.setSizes([1000, 660])
        top_splitter.setChildrenCollapsible(False)

        main_vertical = QSplitter(Qt.Vertical)
        main_vertical.addWidget(top_splitter)
        main_vertical.addWidget(table_card)
        main_vertical.setHandleWidth(1)
        main_vertical.setStretchFactor(0, 4)
        main_vertical.setStretchFactor(1, 3)
        main_vertical.setSizes([500, 300])
        main_vertical.setChildrenCollapsible(False)

        main_area = QWidget()
        main_l = QVBoxLayout()
        main_l.setContentsMargins(0, 0, 0, 0)
        main_l.setSpacing(0)
        main_l.addWidget(main_vertical)
        main_area.setLayout(main_l)
        self.ai_window: Optional[QDialog] = None
        self.weather_window: Optional[WeatherDialog] = None

        container = QWidget()
        container.setObjectName("RootContainer")
        container_l = QVBoxLayout()
        container_l.setContentsMargins(0, 0, 0, 0)
        container_l.setSpacing(2)
        container_l.addWidget(top_controls)
        container_l.addWidget(main_area, 1)
        container_l.addWidget(actions_bar)
        container.setLayout(container_l)
        self.setCentralWidget(container)
        required_main_width = max(top_controls.minimumSizeHint().width(), actions_bar.minimumSizeHint().width()) + 24
        self.setMinimumWidth(max(self.minimumWidth(), required_main_width))
        # Apply persisted settings once all widgets exist
        self._load_settings()

        # Apply custom GUI styles
        self._apply_styles()
        # Build the menu bar and shortcuts
        self._build_actions()
        self.status_filters_label = QLabel("Filters: 0")
        self.status_bhtom_label = QLabel("BHTOM: idle")
        self.status_bhtom_progress = QProgressBar(self)
        self.status_bhtom_progress.setMinimumWidth(96)
        self.status_bhtom_progress.setMaximumWidth(140)
        self.status_bhtom_progress.setTextVisible(False)
        self.status_bhtom_progress.setRange(0, 0)
        self.status_bhtom_progress.hide()
        self.status_aladin_label = QLabel("Aladin: idle")
        self.status_aladin_progress = QProgressBar(self)
        self.status_aladin_progress.setMinimumWidth(96)
        self.status_aladin_progress.setMaximumWidth(140)
        self.status_aladin_progress.setTextVisible(False)
        self.status_aladin_progress.setRange(0, 0)
        self.status_aladin_progress.hide()
        self.status_finder_label = QLabel("Finder: idle")
        self.status_finder_progress = QProgressBar(self)
        self.status_finder_progress.setMinimumWidth(96)
        self.status_finder_progress.setMaximumWidth(140)
        self.status_finder_progress.setTextVisible(False)
        self.status_finder_progress.setRange(0, 0)
        self.status_finder_progress.hide()
        self.status_calc_label = QLabel("Last calc: -")
        self.statusBar().addPermanentWidget(self.status_filters_label)
        self.statusBar().addPermanentWidget(self.status_bhtom_label)
        self.statusBar().addPermanentWidget(self.status_bhtom_progress)
        self.statusBar().addPermanentWidget(self.status_aladin_label)
        self.statusBar().addPermanentWidget(self.status_aladin_progress)
        self.statusBar().addPermanentWidget(self.status_finder_label)
        self.statusBar().addPermanentWidget(self.status_finder_progress)
        self.statusBar().addPermanentWidget(self.status_calc_label)
        # Ensure threads are cleaned up on application exit
        app = QApplication.instance()
        if app:
            app.aboutToQuit.connect(self._cleanup_threads)

        # Start real‑time clock updates for time labels (but avoid double‑launch)
        self.clock_worker = getattr(self, "clock_worker", None)
        if (
            self.table_model.site
            and self.clock_worker is None
            and not getattr(self, "_obs_change_finalize_pending", False)
            and not getattr(self, "_startup_restore_pending", False)
        ):
            self._start_clock_worker()
        self._clock_timer = QTimer(self)
        self._clock_timer.timeout.connect(self._update_clock)
        self._clock_timer.start(1000)
        self._update_status_bar()
        self._update_selected_details()
        self._apply_localization()
        QTimer.singleShot(0, self._run_startup_sequence)

    def _apply_localization(self) -> None:
        lang = resolve_language_choice(
            getattr(self, "_ui_language", current_language()),
            system_locale=getattr(self, "_system_locale_name", ""),
        )
        set_current_language(lang, system_locale=getattr(self, "_system_locale_name", ""))
        self._ui_language = lang
        localize_widget_tree(self, lang)
        if hasattr(self, "table_model"):
            try:
                self.table_model.headerDataChanged.emit(Qt.Horizontal, 0, self.table_model.columnCount() - 1)
            except Exception:
                pass
        for dialog in (
            getattr(self, "ai_window", None),
            getattr(self, "weather_window", None),
            getattr(self, "_bhtom_dialog", None),
        ):
            if not isinstance(dialog, QDialog):
                continue
            localize_widget_tree(dialog, lang)
            model = getattr(dialog, "table_model", None)
            if model is not None and hasattr(model, "columnCount") and hasattr(model, "headerDataChanged"):
                try:
                    model.headerDataChanged.emit(Qt.Horizontal, 0, model.columnCount() - 1)
                except Exception:
                    pass

    def _start_clock_worker(self):
        # Don’t create new workers while exiting
        if getattr(self, "_shutting_down", False):
            return
        # Determine site, fallback to default if none
        site = self.table_model.site or Site(name="Default", latitude=0.0, longitude=0.0, elevation=0.0)
        existing_worker = self.clock_worker
        if (
            existing_worker is not None
            and existing_worker.isRunning()
            and _same_runtime_site(getattr(existing_worker, "site", None), site)
        ):
            logger.debug("Reusing existing ClockWorker for site %s", site.name)
            return
        # Stop any existing clock worker
        if existing_worker:
            self.clock_worker = None
            try:
                existing_worker.request_stop()
            except Exception:
                try:
                    existing_worker.stop()
                except Exception:
                    pass
        logger.info("Launching new ClockWorker for site %s", site.name)
        # Create and track new worker instance
        worker = ClockWorker(site, self.targets, self)
        self.clock_worker = worker
        self._clock_workers.append(worker)
        # Connect updates only from this specific worker to avoid stale threads
        worker.updated.connect(lambda data, w=worker: self._handle_clock_update(data) if self.clock_worker is w else None)
        # Start the new worker
        worker.start()
        # Ensure finished threads clean themselves up
        worker.finished.connect(lambda w=worker: self._clock_workers.remove(w) if w in self._clock_workers else None)
        worker.finished.connect(worker.deleteLater)

    # --------------------------------------------------
    # Qt close handler – ensure threads are stopped
    # --------------------------------------------------
    def closeEvent(self, event):
        """Stop background threads before the window closes."""
        try:
            if hasattr(self, "_plan_autosave_timer") and self._plan_autosave_timer.isActive():
                self._plan_autosave_timer.stop()
            self._flush_plan_autosave()
        except Exception:
            pass
        try:
            self._cleanup_threads()
        except Exception as exc:
            logger.exception("Error during thread cleanup: %s", exc)
        try:
            self.settings.sync()
        except Exception:
            pass
        super().closeEvent(event)

    # --------------------------------------------------
    # Qt close handler – ensure threads are stopped
    # --------------------------------------------------

    # .....................................................
    # menu & shortcuts
    # .....................................................
    def _build_actions(self) -> None:  # noqa: D401
        """Create shortcuts and populate the menu bar."""
        # ----- Actions -----
        self.load_act = QAction("Load targets…", self, shortcut=QKeySequence("Ctrl+L"))
        self.load_act.triggered.connect(self._load_targets)

        self.open_saved_plan_act = QAction("Open saved plan…", self, shortcut=QKeySequence("Ctrl+Shift+L"))
        self.open_saved_plan_act.triggered.connect(self._open_saved_plan)
        self.import_plan_json_act = QAction("Import plan JSON…", self)
        self.import_plan_json_act.triggered.connect(self._import_plan_json)
        self.save_plan_as_act = QAction("Save plan as…", self, shortcut=QKeySequence("Ctrl+Shift+S"))
        self.save_plan_as_act.triggered.connect(self._save_plan_as)

        self.add_act = QAction("Add target…", self, shortcut=QKeySequence("Ctrl+N"))
        self.add_act.triggered.connect(self._add_target_dialog)

        self.toggle_obs_act = QAction("Toggle observed", self, shortcut=QKeySequence("Ctrl+Shift+O"))
        self.toggle_obs_act.triggered.connect(self._toggle_observed_selected)

        self.exp_act = QAction("Export plan…", self, shortcut=QKeySequence("Ctrl+E"))
        self.exp_act.triggered.connect(self._export_plan)
        self.seestar_session_act = QAction("Seestar Session…", self)
        self.seestar_session_act.triggered.connect(self._open_seestar_session)

        self.dark_act = QAction("Toggle Dark/Light Mode", self, shortcut=QKeySequence("Ctrl+D"))
        self.dark_act.setCheckable(True)
        self.dark_act.setChecked(bool(getattr(self, "_dark_enabled", False)))
        self.dark_act.setShortcutContext(Qt.ApplicationShortcut)
        self.dark_act.toggled.connect(lambda checked: self._set_dark_mode_enabled(bool(checked), persist=True))
        self.quit_act = QAction("Quit", self, shortcut=QKeySequence.Quit)
        self.quit_act.setMenuRole(QAction.QuitRole)
        self.quit_act.setShortcutContext(Qt.ApplicationShortcut)
        self.quit_act.triggered.connect(self.close)
        self.ai_describe_act = QAction("Describe selected object", self, shortcut=QKeySequence("Ctrl+I"))
        self.ai_describe_act.triggered.connect(self._ai_describe_target)
        self.ai_suggest_act = QAction("Suggest targets for tonight", self, shortcut=QKeySequence("Ctrl+Shift+I"))
        self.ai_suggest_act.triggered.connect(self._ai_suggest_targets)
        self.ai_suggest_act.setToolTip(
            "Check cached BHTOM targets first (TTL 1h). "
            "A fresh fetch starts only when the cache is missing or expired."
        )
        self.ai_suggest_act.setStatusTip(self.ai_suggest_act.toolTip())
        self.weather_act = QAction("Weather Window…", self)
        self.weather_act.triggered.connect(self._open_weather_window)
        self.ai_toggle_panel_act = QAction("Toggle AI window", self)
        self.ai_toggle_panel_act.triggered.connect(
            lambda: self.ai_toggle_btn.setChecked(not self.ai_toggle_btn.isChecked())
        )
        self.ai_warmup_act = QAction("Warm up LLM", self)
        self.ai_warmup_act.triggered.connect(self._warmup_llm_manual)
        self.ai_clear_chat_act = QAction("Clear AI chat", self)
        self.ai_clear_chat_act.triggered.connect(self._clear_ai_messages)
        self.obs_history_act = QAction("Observation History…", self)
        self.obs_history_act.triggered.connect(self._show_observation_history)
        self.ai_focus_input_act = QAction("Focus AI input", self)
        self.ai_focus_input_act.triggered.connect(self._focus_ai_input)
        self.ai_settings_act = QAction("AI Settings…", self)
        self.ai_settings_act.triggered.connect(self._open_ai_settings)

        self.view_obs_preset_act = QAction("Observation Columns", self, checkable=True)
        self.view_obs_preset_act.triggered.connect(lambda: self._apply_column_preset("observation"))
        self.view_full_preset_act = QAction("Full Columns", self, checkable=True)
        self.view_full_preset_act.triggered.connect(lambda: self._apply_column_preset("full"))
        preset_group = QActionGroup(self)
        preset_group.setExclusive(True)
        preset_group.addAction(self.view_obs_preset_act)
        preset_group.addAction(self.view_full_preset_act)
        current_preset = self.settings.value("table/viewPreset", "full", type=str)
        if current_preset == "full":
            self.view_full_preset_act.setChecked(True)
        else:
            self.view_obs_preset_act.setChecked(True)

        # Make shortcuts work even if the action isn't in a visible menu
        for act in (
            self.load_act,
            self.open_saved_plan_act,
            self.import_plan_json_act,
            self.save_plan_as_act,
            self.add_act,
            self.toggle_obs_act,
            self.exp_act,
            self.seestar_session_act,
            self.dark_act,
            self.quit_act,
            self.ai_describe_act,
            self.ai_suggest_act,
            self.ai_warmup_act,
            self.ai_clear_chat_act,
            self.ai_focus_input_act,
            self.ai_settings_act,
            self.weather_act,
            self.obs_history_act,
        ):
            self.addAction(act)

        # ----- Menu bar -----
        menubar = self.menuBar()
        if sys.platform == "darwin":
            menubar.setNativeMenuBar(True)
        menubar.setVisible(True)
        file_menu = menubar.addMenu("&File")
        file_menu.addAction(self.load_act)
        file_menu.addAction(self.open_saved_plan_act)
        file_menu.addAction(self.import_plan_json_act)
        file_menu.addAction(self.save_plan_as_act)
        file_menu.addAction(self.exp_act)
        file_menu.addAction(self.seestar_session_act)
        file_menu.addSeparator()
        file_menu.addAction(self.dark_act)
        file_menu.addSeparator()
        file_menu.addAction(self.quit_act)

        target_menu = menubar.addMenu("&Targets")
        target_menu.addAction(self.add_act)
        target_menu.addAction(self.ai_suggest_act)
        target_menu.addAction(self.toggle_obs_act)

        view_menu = menubar.addMenu("&View")
        view_menu.addAction(self.view_obs_preset_act)
        view_menu.addAction(self.view_full_preset_act)
        view_menu.addSeparator()
        view_menu.addAction(self.weather_act)
        view_menu.addAction(self.obs_history_act)

        ai_menu = menubar.addMenu("&AI")
        ai_menu.addAction(self.ai_describe_act)
        ai_menu.addSeparator()
        ai_menu.addAction(self.ai_focus_input_act)
        ai_menu.addAction(self.ai_toggle_panel_act)
        ai_menu.addAction(self.ai_warmup_act)
        ai_menu.addAction(self.ai_clear_chat_act)
        ai_menu.addSeparator()
        ai_menu.addAction(self.ai_settings_act)

        # Settings menu
        settings_menu = menubar.addMenu("&Settings")
        self.gen_settings_act = QAction("General Settings…", self)
        self.gen_settings_act.setMenuRole(QAction.NoRole)
        self.gen_settings_act.setShortcuts([QKeySequence("Ctrl+,"), QKeySequence("Ctrl+;")])
        self.gen_settings_act.setShortcutContext(Qt.ApplicationShortcut)
        self.gen_settings_act.triggered.connect(self.open_general_settings)
        settings_menu.addAction(self.gen_settings_act)
        self.tbl_settings_act = QAction("Table Settings…", self)
        self.tbl_settings_act.setMenuRole(QAction.NoRole)
        self.tbl_settings_act.triggered.connect(self.open_table_settings)
        settings_menu.addAction(self.tbl_settings_act)
        self.obs_manager_act = QAction("Observatory Manager…", self)
        self.obs_manager_act.setMenuRole(QAction.NoRole)
        self.obs_manager_act.triggered.connect(self._open_observatory_manager)
        settings_menu.addAction(self.obs_manager_act)

        self.addAction(self.gen_settings_act)
        self.addAction(self.tbl_settings_act)
        self.addAction(self.obs_manager_act)

    def _load_settings(self):
        """Load saved settings and apply both Table and General."""
        self._apply_table_settings()
        # Apply default sort column
        val = self.settings.value("table/defaultSortColumn", TargetTableModel.COL_SCORE)
        try:
            default_sort = int(val)
        except (TypeError, ValueError):
            default_sort = TargetTableModel.COL_SCORE
        # Validate index
        ncols = self.table_model.columnCount()
        if 0 <= default_sort < ncols:
            self.table_view.sortByColumn(default_sort, Qt.AscendingOrder)
        self._apply_general_settings()

    def _migrate_table_settings_schema(self) -> None:
        version = self.settings.value("table/columnSchemaVersion", 0, type=int)
        if version < 2:
            old_column_count = 12
            for idx in range(old_column_count - 1, -1, -1):
                old_key = f"table/col{idx}"
                if not self.settings.contains(old_key):
                    continue
                self.settings.setValue(f"table/col{idx + 1}", self.settings.value(old_key, type=bool))
            self.settings.setValue("table/col0", True)

            if self.settings.contains("table/defaultSortColumn"):
                try:
                    old_sort = int(self.settings.value("table/defaultSortColumn", TargetTableModel.COL_SCORE))
                except (TypeError, ValueError):
                    old_sort = TargetTableModel.COL_SCORE
                self.settings.setValue(
                    "table/defaultSortColumn",
                    min(old_sort + 1, TargetTableModel.COL_ACTIONS),
                )
            version = 2

        if version < 3:
            # Insert the new Last Mag/Mag column at index 10.
            old_column_count = 13
            inserted_col = TargetTableModel.COL_MAG
            for idx in range(old_column_count - 1, inserted_col - 1, -1):
                old_key = f"table/col{idx}"
                if not self.settings.contains(old_key):
                    continue
                self.settings.setValue(f"table/col{idx + 1}", self.settings.value(old_key, type=bool))
            self.settings.setValue(f"table/col{inserted_col}", True)

            if self.settings.contains("table/defaultSortColumn"):
                try:
                    old_sort = int(self.settings.value("table/defaultSortColumn", TargetTableModel.COL_SCORE))
                except (TypeError, ValueError):
                    old_sort = TargetTableModel.COL_SCORE
                if old_sort >= inserted_col:
                    old_sort += 1
                self.settings.setValue(
                    "table/defaultSortColumn",
                    min(old_sort, TargetTableModel.COL_ACTIONS),
                )
            version = 3

        if version < 4:
            color_mode = _normalize_table_color_mode(
                self.settings.value("table/colorMode", "", type=str),
                default="",
            )
            if color_mode not in _VALID_TABLE_COLOR_MODES:
                self.settings.setValue("table/colorMode", "background")
            version = 4

        if version < 5:
            color_mode = _normalize_table_color_mode(
                self.settings.value("table/colorMode", "", type=str),
                default="background",
            )
            if color_mode not in _VALID_TABLE_COLOR_MODES:
                self.settings.setValue("table/colorMode", "background")
            version = 5

        self.settings.setValue("table/columnSchemaVersion", version)

    def _recompute_recommended_order_cache(self) -> None:
        coordinator = getattr(self, "_table_coordinator", None)
        if coordinator is not None:
            coordinator.recompute_recommended_order_cache()

    def _apply_table_settings(self):
        coordinator = getattr(self, "_table_coordinator", None)
        if coordinator is not None:
            coordinator.apply_table_settings()

    def _table_matches_observation_preset(self) -> bool:
        coordinator = getattr(self, "_table_coordinator", None)
        if coordinator is None:
            return False
        return coordinator.table_matches_observation_preset()

    def _apply_column_preset(self, preset: str, save: bool = True):
        coordinator = getattr(self, "_table_coordinator", None)
        if coordinator is not None:
            coordinator.apply_column_preset(preset, save=save)

    def _schedule_table_column_width_refresh(self, reset_widths: bool = False) -> None:
        coordinator = getattr(self, "_table_coordinator", None)
        if coordinator is not None:
            coordinator.schedule_table_column_width_refresh(reset_widths=reset_widths)

    def _refresh_table_column_widths(self) -> None:
        coordinator = getattr(self, "_table_coordinator", None)
        if coordinator is not None:
            coordinator.refresh_table_column_widths()

    def open_table_settings(self):
        """Open the Table Settings dialog."""
        dlg = TableSettingsDialog(self)
        dlg.exec()

    def _apply_general_settings(self):
        """Apply default site."""
        s = self.settings
        prev_show_sun_path = bool(getattr(self, "show_sun_path", True))
        prev_show_moon_path = bool(getattr(self, "show_moon_path", True))
        prev_show_obj_path = bool(getattr(self, "show_obj_path", True))
        prev_dark_enabled = bool(getattr(self, "_dark_enabled", DEFAULT_DARK_MODE))
        prev_theme_name = str(getattr(self, "_theme_name", DEFAULT_UI_THEME))
        prev_accent_secondary = str(getattr(self, "_accent_secondary_override", "") or "")
        prev_ui_font_size = int(getattr(self, "_ui_font_size", 11))
        prev_color_blind_mode = bool(getattr(self, "color_blind_mode", False))
        prev_radar_sweep_enabled = bool(getattr(self, "_radar_sweep_enabled", False))
        prev_radar_sweep_speed = int(getattr(self, "_radar_sweep_speed", 140))
        prev_ui_language_choice = str(getattr(self, "_ui_language_choice", "system") or "system")
        prev_ui_language = str(getattr(self, "_ui_language", current_language()) or current_language())
        self.show_sun_path  = self.settings.value("general/showSunPath",  True, type=bool)
        self.show_moon_path = self.settings.value("general/showMoonPath", True, type=bool)
        self.show_obj_path  = self.settings.value("general/showObjPath",  True, type=bool)
        self._dark_enabled = self.settings.value("general/darkMode", DEFAULT_DARK_MODE, type=bool)
        self._theme_name = normalize_theme_key(self.settings.value("general/uiTheme", DEFAULT_UI_THEME, type=str))
        self._accent_secondary_override = self._load_accent_secondary_override(self._theme_name)
        self._ui_font_size = _sanitize_ui_font_size(self.settings.value("general/uiFontSize", 11, type=int))
        self._ui_language_choice = str(self.settings.value("general/uiLanguage", "system", type=str) or "system")
        self._ui_language = resolve_language_choice(
            self._ui_language_choice,
            system_locale=getattr(self, "_system_locale_name", ""),
        )
        set_current_language(self._ui_language, system_locale=getattr(self, "_system_locale_name", ""))
        self.color_blind_mode = self.settings.value("general/colorBlindMode", False, type=bool)
        self._radar_sweep_enabled = self.settings.value("general/radarSweepAnimation", False, type=bool)
        self._radar_sweep_speed = max(40, min(260, self.settings.value("general/radarSweepSpeed", 140, type=int)))
        new_cutout_view_key = _normalize_cutout_view_key(
            self.settings.value("general/cutoutView", CUTOUT_DEFAULT_VIEW_KEY, type=str)
        )
        new_cutout_survey_key = _normalize_cutout_survey_key(
            self.settings.value("general/cutoutSurvey", CUTOUT_DEFAULT_SURVEY_KEY, type=str)
        )
        new_cutout_fov_arcmin = _sanitize_cutout_fov_arcmin(
            self.settings.value("general/cutoutFovArcmin", CUTOUT_DEFAULT_FOV_ARCMIN, type=int)
        )
        new_cutout_size_px = _sanitize_cutout_size_px(
            self.settings.value("general/cutoutSizePx", CUTOUT_DEFAULT_SIZE_PX, type=int)
        )
        view_changed = new_cutout_view_key != self._cutout_view_key
        cutout_changed = (
            new_cutout_survey_key != self._cutout_survey_key
            or new_cutout_fov_arcmin != self._cutout_fov_arcmin
            or new_cutout_size_px != self._cutout_size_px
        )
        self._cutout_survey_key = new_cutout_survey_key
        self._cutout_fov_arcmin = new_cutout_fov_arcmin
        self._cutout_size_px = new_cutout_size_px
        self._cutout_view_key = new_cutout_view_key
        self.llm_config.url = (
            self.settings.value("llm/serverUrl", LLMConfig.DEFAULT_URL, type=str)
            or LLMConfig.DEFAULT_URL
        ).strip().rstrip("/") or LLMConfig.DEFAULT_URL
        self.llm_config.model = (
            self.settings.value("llm/model", LLMConfig.DEFAULT_MODEL, type=str)
            or LLMConfig.DEFAULT_MODEL
        ).strip() or LLMConfig.DEFAULT_MODEL
        self.llm_config.timeout_s = max(
            15,
            int(self.settings.value("llm/timeoutSec", LLMConfig.DEFAULT_TIMEOUT_S, type=int)),
        )
        self.llm_config.max_tokens = max(
            32,
            int(self.settings.value("llm/maxTokens", LLMConfig.DEFAULT_MAX_TOKENS, type=int)),
        )
        self.llm_config.enable_thinking = bool(
            self.settings.value("llm/enableThinking", LLMConfig.DEFAULT_ENABLE_THINKING, type=bool)
        )
        self.llm_config.enable_chat_memory = bool(
            self.settings.value("llm/enableChatMemory", LLMConfig.DEFAULT_ENABLE_CHAT_MEMORY, type=bool)
        )
        self._llm_chat_font_size_pt = max(
            9,
            int(self.settings.value("llm/chatFontSizePt", LLMConfig.DEFAULT_CHAT_FONT_PT, type=int)),
        )
        self._ai_chat_spacing = str(
            self.settings.value("llm/chatSpacing", LLMConfig.DEFAULT_CHAT_SPACING, type=str)
            or LLMConfig.DEFAULT_CHAT_SPACING
        ).strip().lower()
        self._ai_chat_tint_strength = str(
            self.settings.value("llm/chatTintStrength", LLMConfig.DEFAULT_CHAT_TINT_STRENGTH, type=str)
            or LLMConfig.DEFAULT_CHAT_TINT_STRENGTH
        ).strip().lower()
        self._ai_chat_width = str(
            self.settings.value("llm/chatMessageWidth", LLMConfig.DEFAULT_CHAT_WIDTH, type=str)
            or LLMConfig.DEFAULT_CHAT_WIDTH
        ).strip().lower()
        self._ai_status_error_clear_s = max(
            0,
            int(self.settings.value("llm/statusErrorClearSec", LLMConfig.DEFAULT_STATUS_ERROR_CLEAR_S, type=int)),
        )
        style_changed = any(
            (
                prev_dark_enabled != self._dark_enabled,
                prev_theme_name != self._theme_name,
                prev_accent_secondary != self._accent_secondary_override,
                prev_ui_font_size != self._ui_font_size,
                prev_color_blind_mode != self.color_blind_mode,
            )
        )
        polar_visuals_changed = any(
            (
                prev_show_sun_path != self.show_sun_path,
                prev_show_moon_path != self.show_moon_path,
                prev_show_obj_path != self.show_obj_path,
                prev_radar_sweep_enabled != self._radar_sweep_enabled,
                prev_radar_sweep_speed != self._radar_sweep_speed,
            )
        )
        language_changed = any(
            (
                prev_ui_language_choice != self._ui_language_choice,
                prev_ui_language != self._ui_language,
            )
        )
        self._refresh_ai_warm_indicator()
        self._refresh_ai_context_hint()
        self._apply_ai_chat_font()
        self._render_ai_messages()

        if hasattr(self, "min_moon_sep_spin"):
            self.min_moon_sep_spin.blockSignals(True)
            self.min_moon_sep_spin.setValue(s.value("general/minMoonSep", 0, type=int))
            self.min_moon_sep_spin.blockSignals(False)
        if hasattr(self, "min_score_spin"):
            self.min_score_spin.blockSignals(True)
            self.min_score_spin.setValue(s.value("general/minScore", 0, type=int))
            self.min_score_spin.blockSignals(False)
        if hasattr(self, "hide_observed_chk"):
            self.hide_observed_chk.blockSignals(True)
            self.hide_observed_chk.setChecked(s.value("general/hideObserved", False, type=bool))
            self.hide_observed_chk.blockSignals(False)
        if hasattr(self, "sun_alt_limit_spin"):
            self.sun_alt_limit_spin.blockSignals(True)
            self.sun_alt_limit_spin.setValue(s.value("general/sunAltLimit", -10, type=int))
            self.sun_alt_limit_spin.blockSignals(False)
        if hasattr(self, "cutout_tabs"):
            tab_idx = 1 if self._cutout_view_key == "finderchart" else 0
            self.cutout_tabs.blockSignals(True)
            self.cutout_tabs.setCurrentIndex(tab_idx)
            self.cutout_tabs.blockSignals(False)
        if cutout_changed and hasattr(self, "cutout_image_label"):
            self._clear_cutout_cache()
        if (cutout_changed or view_changed) and hasattr(self, "cutout_image_label"):
            self._update_cutout_preview_for_target(self._selected_target_or_none())
        self._update_quick_targets_button_tooltip()
        self._update_seestar_button_tooltip()

        # If a payload is already cached, refresh the polar plot right away
        if getattr(self, "last_payload", None) and polar_visuals_changed:
            self._update_polar_positions(self.last_payload)
        ds = s.value("general/defaultSite", type=str)
        if ds in self.observatories and self.obs_combo.currentText() != ds:
            self.obs_combo.setCurrentText(ds)
        if style_changed:
            self._apply_table_settings()
            self._apply_styles()
        if language_changed:
            self._apply_localization()
        self._update_radar_sweep_state()
        self._update_status_bar()
        self._refresh_weather_window_context()

    def open_general_settings(self, initial_tab: Optional[str] = None):
        dlg = GeneralSettingsDialog(self, initial_tab=initial_tab)
        dlg.exec()

    @Slot()
    def _open_ai_settings(self) -> None:
        self.open_general_settings(initial_tab="AI")

    def _refresh_ai_context_hint(self) -> None:
        memory_enabled = bool(getattr(self.llm_config, "enable_chat_memory", False))
        if memory_enabled:
            hint_text = (
                "Short chat memory enabled.\n"
                "Last 1-2 turns may be reused."
            )
            hint_tooltip = (
                "Examples: 'Którą z tych SN najlepiej dziś obserwować?' or "
                "'How should I observe 8C0716_714 tonight?'\n\n"
                "Short chat memory can help follow-up questions, but requests may be slower."
            )
            input_tooltip = (
                "The AI chat is saved per active plan/workspace. "
                "Prompts may reuse the last 1-2 user/LLM turns for follow-up questions. "
                "Use complete questions when the context could be ambiguous."
            )
        else:
            hint_text = (
                "No chat memory.\n"
                "Ask complete questions with the object or class name."
            )
            hint_tooltip = (
                "Examples: 'Którą SN z BHTOM najlepiej dziś obserwować?' or "
                "'How should I observe 8C0716_714 tonight?'"
            )
            input_tooltip = (
                "The AI chat is saved per active plan/workspace. "
                "Prompts do not reuse previous turns, so use complete questions with the object or class name, "
                "e.g. 'Która SN z BHTOM jest dziś najlepsza?'"
            )

        if hasattr(self, "ai_context_hint"):
            set_translated_text(self.ai_context_hint, hint_text, current_language())
            set_translated_tooltip(self.ai_context_hint, hint_tooltip, current_language())
        if hasattr(self, "ai_input"):
            set_translated_tooltip(self.ai_input, input_tooltip, current_language())

    def _refresh_ai_knowledge_hint(self, titles: Optional[list[str]] = None) -> None:
        if titles is not None:
            self._ai_last_knowledge_titles = [str(title).strip() for title in titles if str(title).strip()]
        current_titles = list(getattr(self, "_ai_last_knowledge_titles", []) or [])
        if current_titles:
            if len(current_titles) > 3:
                visible = ", ".join(current_titles[:3]) + f" (+{len(current_titles) - 3} more)"
            else:
                visible = ", ".join(current_titles)
            text = f"Using local knowledge: {visible}"
            tooltip = "Local knowledge notes used for the most recent AI prompt:\n- " + "\n- ".join(current_titles)
        else:
            text = "Using local knowledge: none"
            tooltip = "No local knowledge notes were added to the most recent AI prompt."
        if hasattr(self, "ai_knowledge_hint"):
            set_translated_text(self.ai_knowledge_hint, text, current_language())
            set_translated_tooltip(self.ai_knowledge_hint, tooltip, current_language())

    @Slot()
    def _focus_ai_input(self) -> None:
        ai_window = self._ensure_ai_window()
        if hasattr(self, "ai_toggle_btn") and not self.ai_toggle_btn.isChecked():
            self.ai_toggle_btn.setChecked(True)
        if isinstance(ai_window, QDialog):
            ai_window.show()
            ai_window.raise_()
            ai_window.activateWindow()
        if hasattr(self, "ai_input"):
            self.ai_input.setFocus(Qt.OtherFocusReason)
            self.ai_input.selectAll()

    def _sun_alt_limit(self) -> float:
        if not hasattr(self, "sun_alt_limit_spin"):
            return -10.0
        return float(self.sun_alt_limit_spin.value())

    @Slot(int)
    def _on_sun_alt_limit_changed(self, value: int):
        self.settings.setValue("general/sunAltLimit", int(value))
        self._update_status_bar()
        self._refresh_weather_window_context()
        self._schedule_plan_autosave()
        if hasattr(self, "progress"):
            self._replot_timer.start()

    @Slot()
    def _on_filter_change(self, *_):
        self.settings.setValue("general/minMoonSep", self.min_moon_sep_spin.value())
        self.settings.setValue("general/minScore", self.min_score_spin.value())
        self.settings.setValue("general/hideObserved", self.hide_observed_chk.isChecked())
        self.table_model.row_enabled = self._recompute_row_enabled_from_current()
        self._apply_table_row_visibility()
        self._emit_table_data_changed()
        self._update_status_bar()
        self._schedule_plan_autosave()
        if hasattr(self, "progress"):
            self._replot_timer.start()

    def _passes_active_filters(self, target: Target, score: float, moon_sep_now: float) -> bool:
        min_moon_sep = float(self.min_moon_sep_spin.value()) if hasattr(self, "min_moon_sep_spin") else 0.0
        min_score = float(self.min_score_spin.value()) if hasattr(self, "min_score_spin") else 0.0
        hide_observed = self.hide_observed_chk.isChecked() if hasattr(self, "hide_observed_chk") else False
        if hide_observed and target.observed:
            return False
        if score < min_score:
            return False
        if not np.isfinite(moon_sep_now):
            return min_moon_sep <= 0.0
        # Keep filter semantics aligned with table display precision (0.1 deg).
        moon_sep_cmp = round(float(moon_sep_now), 1)
        if moon_sep_cmp < min_moon_sep:
            return False
        return True

    def _recompute_row_enabled_from_current(self) -> list[bool]:
        row_enabled: list[bool] = []
        for idx, tgt in enumerate(self.targets):
            score = float(self.table_model.scores[idx]) if idx < len(self.table_model.scores) else 0.0
            moon_sep_now = (
                float(self.table_model.current_seps[idx])
                if idx < len(self.table_model.current_seps)
                else float("nan")
            )
            row_enabled.append(self._passes_active_filters(tgt, score, moon_sep_now))
        return row_enabled

    def _selected_rows(self) -> list[int]:
        sel_model = self.table_view.selectionModel()
        if sel_model is None:
            return []
        return sorted({idx.row() for idx in sel_model.selectedRows()})

    def _schedule_primary_target_selection(self) -> None:
        coordinator = getattr(self, "_table_coordinator", None)
        if coordinator is not None:
            coordinator.schedule_primary_target_selection()

    @Slot()
    def _ensure_primary_target_selected(self) -> None:
        if not hasattr(self, "table_view") or not hasattr(self, "table_model"):
            return
        rows = self.table_model.rowCount()
        if rows <= 0:
            self._update_selected_details()
            return
        current_rows = [row for row in self._selected_rows() if 0 <= row < rows and not self.table_view.isRowHidden(row)]
        if current_rows:
            return
        target_row = next((row for row in range(rows) if not self.table_view.isRowHidden(row)), None)
        if target_row is None:
            self._update_selected_details()
            return
        sel_model = self.table_view.selectionModel()
        if sel_model is None:
            return
        idx = self.table_model.index(target_row, TargetTableModel.COL_NAME)
        if not idx.isValid():
            return
        sel_model.select(idx, QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows)
        self.table_view.setCurrentIndex(idx)
        self.table_view.scrollTo(idx, QTableView.PositionAtCenter)

    def _update_status_bar(self):
        if not hasattr(self, "status_filters_label") or not hasattr(self, "status_calc_label"):
            return
        parts: list[str] = []
        if hasattr(self, "min_score_spin") and self.min_score_spin.value() > 0:
            parts.append(f"score≥{self.min_score_spin.value()}")
        if hasattr(self, "min_moon_sep_spin") and self.min_moon_sep_spin.value() > 0:
            parts.append(f"moon≥{self.min_moon_sep_spin.value()}°")
        if hasattr(self, "hide_observed_chk") and self.hide_observed_chk.isChecked():
            parts.append(translate_text("hide observed", current_language()))
        if hasattr(self, "sun_alt_limit_spin"):
            parts.append(f"sun≤{self._sun_alt_limit():.0f}°")

        total = len(self.targets)
        visible = sum(1 for flag in self.table_model.row_enabled if flag) if self.table_model.row_enabled else total
        active_filters = ", ".join(parts) if parts else translate_text("none", current_language())
        set_translated_text(
            self.status_filters_label,
            f"Filters: {active_filters} | Visible: {visible}/{total}",
            current_language(),
        )

        if self.worker is not None and self.worker.isRunning():
            set_translated_text(self.status_calc_label, "Last calc: running...", current_language())
            return
        stats = self._last_calc_stats
        if stats.total_targets > 0:
            set_translated_text(
                self.status_calc_label,
                f"Last calc: {stats.duration_s:.2f}s | {stats.visible_targets}/{stats.total_targets}",
                current_language(),
            )
        else:
            set_translated_text(self.status_calc_label, "Last calc: -", current_language())

    @Slot(object, object)
    def _update_selected_details(self, *_):
        rows = self._selected_rows()
        if not rows:
            self.sel_name_label.setText("-")
            set_translated_text(self.sel_type_title_label, "Type / Mag:", current_language())
            self.sel_type_label.setText("-")
            self.sel_score_label.setText("-")
            self.sel_window_label.setText("-")
            self.sel_notes_label.setText("-")
            self.edit_object_btn.setEnabled(False)
            if hasattr(self, "ai_describe_act"):
                self.ai_describe_act.setEnabled(False)
            self._schedule_selected_cutout_update(None)
            self._update_status_bar()
            return

        row = rows[0]
        if not (0 <= row < len(self.targets)):
            self._schedule_selected_cutout_update(None)
            return
        tgt = self.targets[row]
        self._ensure_known_target_type(tgt)
        self.sel_name_label.setText(tgt.name)
        set_translated_text(
            self.sel_type_title_label,
            "Type / Last Mag:" if _target_magnitude_label(tgt) == "Last Mag" else "Type / Mag:",
            current_language(),
        )
        mag_txt = f"{tgt.magnitude:.2f}" if tgt.magnitude is not None else "-"
        type_txt = tgt.object_type or "-"
        self.sel_type_label.setText(f"{type_txt} / {_target_magnitude_label(tgt)} {mag_txt}")

        metrics = self.target_metrics.get(tgt.name)
        if metrics:
            self.sel_score_label.setText(f"{metrics.score:.1f} | {metrics.hours_above_limit:.1f}h")
        else:
            self.sel_score_label.setText("-")

        window = self.target_windows.get(tgt.name)
        if window:
            tz_name = "UTC"
            if isinstance(getattr(self, "last_payload", None), dict):
                tz_name = str(self.last_payload.get("tz", "UTC"))
            try:
                tz = pytz.timezone(tz_name)
            except Exception:
                tz = pytz.UTC
            start = window[0].astimezone(tz).strftime("%H:%M")
            end = window[1].astimezone(tz).strftime("%H:%M")
            self.sel_window_label.setText(f"{start} - {end}")
        else:
            self.sel_window_label.setText("-")
        self.sel_notes_label.setText(tgt.notes or "-")
        self.edit_object_btn.setEnabled(True)
        if hasattr(self, "ai_describe_act"):
            self.ai_describe_act.setEnabled(True)
        self._schedule_selected_cutout_update(tgt)
        self._update_status_bar()

    def _edit_notes_for_row(self, row: int):
        if not (0 <= row < len(self.targets)):
            return
        tgt = self.targets[row]
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Notes: {tgt.name}")
        layout = QVBoxLayout(dlg)
        notes_edit = QTextEdit(dlg)
        notes_edit.setPlainText(tgt.notes or "")
        notes_edit.setMinimumHeight(120)
        layout.addWidget(notes_edit)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dlg)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        _style_dialog_button_box(buttons)
        layout.addWidget(buttons)
        _fit_dialog_to_screen(
            dlg,
            preferred_width=760,
            preferred_height=520,
            min_width=560,
            min_height=340,
        )
        if dlg.exec() != QDialog.Accepted:
            return
        tgt.notes = notes_edit.toPlainText().strip()
        idx = self.table_model.index(row, TargetTableModel.COL_NAME)
        self.table_model.dataChanged.emit(idx, idx, [Qt.ToolTipRole, Qt.DisplayRole])
        self._update_selected_details()
        self._schedule_plan_autosave()

    def _edit_object_for_row(self, row: int):
        if not (0 <= row < len(self.targets)):
            return
        tgt = self.targets[row]
        old_name = tgt.name

        dlg = QDialog(self)
        dlg.setWindowTitle(f"Edit Object: {tgt.name}")
        dlg.setObjectName("EditObjectDialog")
        form = QFormLayout(dlg)

        name_edit = QLineEdit(dlg)
        name_edit.setText(tgt.name)
        ra_edit = QLineEdit(dlg)
        ra_edit.setText(Angle(tgt.ra, u.deg).to_string(unit=u.hourangle, sep=":", pad=True, precision=2))
        ra_edit.setPlaceholderText("HH:MM:SS or degrees")
        dec_edit = QLineEdit(dlg)
        dec_edit.setText(Angle(tgt.dec, u.deg).to_string(unit=u.deg, sep=":", alwayssign=True, pad=True, precision=2))
        dec_edit.setPlaceholderText("±DD:MM:SS or degrees")
        type_edit = QLineEdit(dlg)
        type_edit.setText(tgt.object_type or "")
        mag_edit = QLineEdit(dlg)
        mag_edit.setPlaceholderText("Optional")
        mag_edit.setValidator(QDoubleValidator(-40.0, 40.0, 4, dlg))
        if tgt.magnitude is not None:
            mag_edit.setText(f"{tgt.magnitude:.2f}")
        line_palette = self._line_palette()
        color_edit = QLineEdit(dlg)
        color_edit.setPlaceholderText("Auto (palette) or #RRGGBB")
        color_edit.setText(_normalized_css_color(tgt.plot_color))
        color_preview = QLabel("Auto", dlg)
        color_preview.setAlignment(Qt.AlignCenter)
        color_preview.setMinimumWidth(112)
        color_preview.setFrameShape(QFrame.StyledPanel)
        pick_color_btn = QPushButton("Pick...", dlg)
        pick_color_btn.setMinimumWidth(78)
        auto_color_btn = QPushButton("Auto", dlg)
        auto_color_btn.setMinimumWidth(70)

        def _default_color_for_row() -> str:
            mapped = self.table_model.color_map.get(tgt.name)
            if isinstance(mapped, QColor) and mapped.isValid():
                return mapped.name()
            return self._target_plot_color_css(tgt, row, line_palette)

        def _refresh_color_preview():
            color_css = _normalized_css_color(color_edit.text())
            solid_border = self._theme_color("panel_border", "#50627a")
            dashed_border = self._theme_color("section_hint", solid_border)
            if color_css:
                sample = QColor(color_css)
                fg = "#111111" if sample.lightness() > 145 else "#f5f7fb"
                color_preview.setText(color_css.upper())
                color_preview.setStyleSheet(
                    f"background:{color_css}; color:{fg}; border:1px solid {solid_border}; border-radius:4px; padding:2px 6px;"
                )
                return
            fallback = _default_color_for_row()
            sample = QColor(fallback)
            fg = "#111111" if sample.lightness() > 145 else "#f5f7fb"
            color_preview.setText("Auto")
            color_preview.setStyleSheet(
                f"background:{fallback}; color:{fg}; border:1px dashed {dashed_border}; border-radius:4px; padding:2px 6px;"
            )

        def _pick_plot_color():
            seed = _normalized_css_color(color_edit.text()) or _default_color_for_row()
            chosen = QColorDialog.getColor(QColor(seed), dlg, "Pick object color")
            if chosen.isValid():
                color_edit.setText(chosen.name())

        color_edit.textChanged.connect(lambda _text: _refresh_color_preview())
        pick_color_btn.clicked.connect(_pick_plot_color)
        auto_color_btn.clicked.connect(color_edit.clear)
        color_row = QHBoxLayout()
        color_row.setContentsMargins(0, 0, 0, 0)
        color_row.setSpacing(8)
        color_row.addWidget(color_edit, 1)
        color_row.addWidget(pick_color_btn)
        color_row.addWidget(auto_color_btn)
        color_row.addWidget(color_preview)
        color_widget = QWidget(dlg)
        color_widget.setLayout(color_row)
        _refresh_color_preview()
        priority_spin = QSpinBox(dlg)
        priority_spin.setRange(1, 5)
        priority_spin.setValue(tgt.priority)
        observed_chk = QCheckBox("Observed", dlg)
        observed_chk.setChecked(tgt.observed)
        notes_edit = QTextEdit(dlg)
        notes_edit.setPlainText(tgt.notes or "")
        notes_edit.setMinimumHeight(120)

        form.addRow("Name:", name_edit)
        form.addRow("RA:", ra_edit)
        form.addRow("Dec:", dec_edit)
        form.addRow("Type:", type_edit)
        form.addRow("Magnitude:", mag_edit)
        form.addRow("Color:", color_widget)
        form.addRow("Priority:", priority_spin)
        form.addRow("", observed_chk)
        form.addRow("Notes:", notes_edit)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dlg)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        _style_dialog_button_box(buttons)
        form.addRow(buttons)
        _fit_dialog_to_screen(
            dlg,
            preferred_width=920,
            preferred_height=760,
            min_width=620,
            min_height=480,
        )

        while dlg.exec() == QDialog.Accepted:
            name = name_edit.text().strip()
            if not name:
                QMessageBox.warning(dlg, "Invalid Name", "Name cannot be empty.")
                continue
            try:
                ra_deg = parse_ra_to_deg(ra_edit.text().strip())
                dec_deg = parse_dec_to_deg(dec_edit.text().strip())
            except Exception as exc:
                QMessageBox.warning(dlg, "Invalid Coordinates", f"Cannot parse RA/Dec: {exc}")
                continue
            if not -90.0 <= dec_deg <= 90.0:
                QMessageBox.warning(dlg, "Invalid Coordinates", "Declination must be between -90 and +90 degrees.")
                continue

            mag_txt = mag_edit.text().strip()
            magnitude: Optional[float] = None
            if mag_txt:
                try:
                    magnitude = float(mag_txt)
                except ValueError:
                    QMessageBox.warning(dlg, "Invalid Magnitude", "Magnitude must be a numeric value.")
                    continue
            color_txt = color_edit.text().strip()
            plot_color = _normalized_css_color(color_txt)
            if color_txt and not plot_color:
                QMessageBox.warning(
                    dlg,
                    "Invalid Color",
                    "Color must be a valid CSS color (for example #4a86e8 or steelblue).",
                )
                continue

            tgt.name = name
            tgt.ra = float(ra_deg) % 360.0
            tgt.dec = float(dec_deg)
            tgt.object_type = type_edit.text().strip()
            tgt.magnitude = magnitude
            tgt.plot_color = plot_color
            tgt.priority = priority_spin.value()
            was_observed = bool(tgt.observed)
            tgt.observed = observed_chk.isChecked()
            tgt.notes = notes_edit.toPlainText().strip()

            # Metrics/windows are coordinate and condition dependent; refresh on next run.
            self.target_metrics.pop(old_name, None)
            self.target_metrics.pop(tgt.name, None)
            self.target_windows.pop(old_name, None)
            self.target_windows.pop(tgt.name, None)
            if old_name != tgt.name:
                self.table_model.color_map.pop(old_name, None)
            self._refresh_target_color_map()
            self._recompute_recommended_order_cache()

            self._emit_table_data_changed()
            self._update_selected_details()
            self._update_status_bar()
            self._record_observation_if_needed(tgt, was_observed=was_observed, source="edit_object")
            self._schedule_plan_autosave()
            self._replot_timer.start()
            return

    @Slot()
    def _edit_object_for_selected(self):
        rows = self._selected_rows()
        if not rows:
            QMessageBox.information(self, "No target selected", "Select a target row first.")
            return
        self._edit_object_for_row(rows[0])

    @Slot(QModelIndex)
    def _on_table_double_click(self, index: QModelIndex):
        if not index.isValid():
            return
        if index.column() == TargetTableModel.COL_OBSERVED:
            self._toggle_observed_row(index.row())
            return
        if index.column() == TargetTableModel.COL_NAME:
            self._edit_object_for_row(index.row())

    @Slot(object)
    def _open_table_context_menu(self, pos):
        index = self.table_view.indexAt(pos)
        if not index.isValid():
            return
        sel_model = self.table_view.selectionModel()
        if sel_model is None:
            return
        if not sel_model.isRowSelected(index.row(), QModelIndex()):
            sel_model.select(index, QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows)
        rows = self._selected_rows()
        if not rows:
            return
        target = self.targets[rows[0]]

        menu = QMenu(self.table_view)
        menu.setAttribute(Qt.WA_TranslucentBackground, False)
        menu.setWindowOpacity(1.0)
        menu.setMinimumWidth(240)
        menu_font = QFont(menu.font())
        menu_font.setPointSize(max(menu_font.pointSize(), 11))
        menu.setFont(menu_font)
        edit_obj_act = menu.addAction("Edit object...")
        open_bhtom_act = None
        if _normalize_catalog_token(target.source_catalog) == "bhtom":
            open_bhtom_act = menu.addAction("Open in BHTOM")
        toggle_act = menu.addAction("Toggle observed")
        menu.addSeparator()
        delete_act = menu.addAction("Remove selected")
        chosen = menu.exec(self.table_view.viewport().mapToGlobal(pos))
        if chosen == edit_obj_act:
            self._edit_object_for_row(rows[0])
        elif open_bhtom_act is not None and chosen == open_bhtom_act:
            self._open_target_in_bhtom(target)
        elif chosen == toggle_act:
            self._toggle_observed_selected()
        elif chosen == delete_act:
            self._delete_selected_targets()

    def _open_target_in_bhtom(self, target: Target) -> None:
        if _normalize_catalog_token(target.source_catalog) != "bhtom":
            return
        slug = quote(target.name.strip(), safe="")
        if not slug:
            return
        QDesktopServices.openUrl(QUrl(f"{self._bhtom_api_base_url()}/targets/{slug}/"))

    def _read_site_float(self, edit: QLineEdit) -> float:
        return float(edit.text().strip().replace(",", "."))

    def _validate_site_inputs(self) -> bool:
        invalid: list[QLineEdit] = []
        errors: list[str] = []
        try:
            lat = self._read_site_float(self.lat_edit)
            if not -90.0 <= lat <= 90.0:
                raise ValueError("Latitude must be within [-90, 90].")
        except Exception:
            invalid.append(self.lat_edit)
            errors.append("Invalid latitude")
        try:
            lon = self._read_site_float(self.lon_edit)
            if not -180.0 <= lon <= 180.0:
                raise ValueError("Longitude must be within [-180, 180].")
        except Exception:
            invalid.append(self.lon_edit)
            errors.append("Invalid longitude")
        try:
            self._read_site_float(self.elev_edit)
        except Exception:
            invalid.append(self.elev_edit)
            errors.append("Invalid elevation")

        if hasattr(self, "coord_error_label"):
            self.coord_error_label.setText("; ".join(errors))
            self.coord_error_label.setVisible(bool(errors))
        for edit in (self.lat_edit, self.lon_edit, self.elev_edit):
            _set_widget_invalid(edit, edit in invalid)
        return not invalid

    @Slot()
    def _on_site_inputs_changed(self):
        if not self._validate_site_inputs():
            return
        try:
            site = self._build_runtime_site_from_inputs()
        except (ValidationError, ValueError):
            return
        self.table_model.site = site
        self._start_clock_worker()
        self._schedule_plan_autosave()
        self._replot_timer.start()

    def _build_runtime_site_from_inputs(self) -> Site:
        selected_name = self.obs_combo.currentText() if hasattr(self, "obs_combo") else ""
        template: Optional[Site] = None
        if selected_name and hasattr(self, "observatories"):
            maybe = self.observatories.get(selected_name)
            if isinstance(maybe, Site):
                template = maybe
        if template is None:
            current_site = getattr(self.table_model, "site", None) if hasattr(self, "table_model") else None
            if isinstance(current_site, Site):
                template = current_site
        return Site(
            name=selected_name or (template.name if isinstance(template, Site) else "Custom"),
            latitude=self._read_site_float(self.lat_edit),
            longitude=self._read_site_float(self.lon_edit),
            elevation=self._read_site_float(self.elev_edit),
            limiting_magnitude=self._current_limiting_magnitude(),
            telescope_diameter_mm=float(getattr(template, "telescope_diameter_mm", 0.0) or 0.0),
            focal_length_mm=float(getattr(template, "focal_length_mm", 0.0) or 0.0),
            pixel_size_um=float(getattr(template, "pixel_size_um", 0.0) or 0.0),
            detector_width_px=int(getattr(template, "detector_width_px", 0) or 0),
            detector_height_px=int(getattr(template, "detector_height_px", 0) or 0),
            custom_conditions_url=str(getattr(template, "custom_conditions_url", "") or "").strip(),
        )

    def _cleanup_threads(self):
        """Stop timer and threads cleanly on exit (order matters)."""
        logger.info("Begin cleanup_threads")
        # Signal that shutdown has begun
        self._shutting_down = True
        cutout_resize_timer = getattr(self, "_cutout_resize_timer", None)
        if cutout_resize_timer is not None:
            try:
                cutout_resize_timer.stop()
            except Exception:
                pass
        # 1) Kill the periodic timer *first* so it can’t spawn new workers
        clock_timer = getattr(self, "_clock_timer", None)
        if clock_timer is not None:
            try:
                # Disconnect to avoid late-queued timeouts spawning new workers
                if hasattr(clock_timer, "timeout"):
                    clock_timer.timeout.disconnect(self._update_clock)
                clock_timer.stop()
            except Exception:
                pass
            logger.debug("Clock timer stopped")
            self._clock_timer = None

        # Stop pending cutout network request.
        cutout_reply = getattr(self, "_cutout_reply", None)
        if cutout_reply is not None and not cutout_reply.isFinished():
            try:
                cutout_reply.abort()
            except Exception:
                pass
        self._cutout_reply = None
        self._cutout_pending_key = ""
        self._cutout_pending_name = ""
        self._cutout_pending_background = False
        self._cutout_prefetch_queue.clear()
        self._cutout_prefetch_enqueued_keys.clear()
        self._cutout_prefetch_total = 0
        self._cutout_prefetch_completed = 0
        self._cutout_prefetch_cached = 0
        self._cutout_prefetch_active = False

        # 2) Stop *all* ClockWorkers created during this session
        for w in list(getattr(self, "_clock_workers", [])):
            try:
                # Skip if C++ object is already destroyed
                if shb.isValid(w) and w.isRunning():
                    # Ask it to stop only if still alive
                    w.running = False
                    if hasattr(w, "_wait_cond"):
                        w._wait_cond.wakeAll()
                    w.quit()
                    w.wait()
            except RuntimeError:
                # Already deleted; nothing to do
                pass
            except Exception as exc:
                logger.exception("Failed to stop ClockWorker: %s", exc)
        logger.debug("All ClockWorkers stopped (%d total)", len(getattr(self, "_clock_workers", [])))
        # Clear list
        self._clock_workers = []
        self.clock_worker = None

        # 3) Stop the AstronomyWorker if it’s still running
        worker = getattr(self, "worker", None)
        if worker is not None and worker.isRunning():
            try:
                worker.quit()
                worker.wait()
                logger.debug("AstronomyWorker stopped")
            except Exception as exc:
                logger.exception("Failed to stop AstronomyWorker: %s", exc)
        llm_worker = getattr(self, "_llm_worker", None)
        if llm_worker is not None:
            try:
                if llm_worker.isRunning():
                    llm_worker.requestInterruption()
                    llm_worker.quit()
                    if not llm_worker.wait(1200):
                        llm_worker.terminate()
                        llm_worker.wait(400)
            except Exception as exc:
                logger.exception("Failed to stop LLMWorker: %s", exc)
            self._llm_worker = None
        meta_worker = getattr(self, "_meta_worker", None)
        if meta_worker is not None:
            try:
                if meta_worker.isRunning():
                    meta_worker.cancel()
                    meta_worker.quit()
                    meta_worker.wait()
            except Exception as exc:  # noqa: BLE001
                logger.exception("Failed to stop MetadataLookupWorker: %s", exc)
            self._meta_worker = None
        startup_weather_worker = getattr(self, "_startup_weather_worker", None)
        if startup_weather_worker is not None:
            try:
                if startup_weather_worker.isRunning():
                    startup_weather_worker.requestInterruption()
                    startup_weather_worker.quit()
                    if not startup_weather_worker.wait(1500):
                        startup_weather_worker.terminate()
                        startup_weather_worker.wait(400)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Failed to stop startup WeatherLiveWorker: %s", exc)
            self._startup_weather_worker = None
        bhtom_worker = getattr(self, "_bhtom_worker", None)
        if bhtom_worker is not None:
            try:
                if bhtom_worker.isRunning():
                    bhtom_worker.requestInterruption()
                    bhtom_worker.quit()
                    if not bhtom_worker.wait(1500):
                        bhtom_worker.terminate()
                        bhtom_worker.wait(400)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Failed to stop BhtomSuggestionWorker: %s", exc)
            self._bhtom_worker = None
            self._bhtom_worker_mode = ""
            self._bhtom_worker_cache_key = None
            self._bhtom_worker_source = ""
            self._bhtom_dialog = None
        cand_worker = getattr(self, "_bhtom_candidate_prefetch_worker", None)
        if cand_worker is not None:
            try:
                if cand_worker.isRunning():
                    cand_worker.requestInterruption()
                    cand_worker.quit()
                    if not cand_worker.wait(1500):
                        cand_worker.terminate()
                        cand_worker.wait(400)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Failed to stop BhtomCandidatePrefetchWorker: %s", exc)
            self._bhtom_candidate_prefetch_worker = None
        obs_worker = getattr(self, "_bhtom_observatory_worker", None)
        if obs_worker is not None:
            try:
                if obs_worker.isRunning():
                    obs_worker.requestInterruption()
                    obs_worker.quit()
                    if not obs_worker.wait(1500):
                        obs_worker.terminate()
                        obs_worker.wait(400)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Failed to stop BhtomObservatoryPresetWorker: %s", exc)
            self._bhtom_observatory_worker = None
            self._bhtom_observatory_loading_message = ""
        self._stop_finder_workers(aggressive=True)
        self._finder_worker = None
        self._finder_pending_key = ""
        self._finder_pending_name = ""
        self._finder_pending_background = False
        self._finder_prefetch_queue.clear()
        self._finder_prefetch_enqueued_keys.clear()
        self._finder_prefetch_total = 0
        self._finder_prefetch_completed = 0
        self._finder_prefetch_cached = 0
        self._finder_prefetch_active = False
        self._finder_retry_after.clear()

    def _apply_default_sort(self):
        """Apply default sort column from settings."""
        default_sort = self.settings.value("table/defaultSortColumn", TargetTableModel.COL_SCORE, type=int)
        cols = self.table_model.columnCount()
        if 0 <= default_sort < cols:
            self.table_model.sort(default_sort, Qt.AscendingOrder)
            self.table_view.horizontalHeader().setSortIndicator(default_sort, Qt.AscendingOrder)

    def _reapply_current_table_sort(self) -> None:
        if not hasattr(self, "table_view") or not hasattr(self, "table_model"):
            return
        header = self.table_view.horizontalHeader()
        section = header.sortIndicatorSection()
        order = header.sortIndicatorOrder()
        cols = self.table_model.columnCount()
        if 0 <= section < cols:
            self.table_model.sort(section, order)
            header.setSortIndicator(section, order)
            return
        self._apply_default_sort()

    def _plan_target_key(self, target: Target) -> str:
        source_catalog = str(target.source_catalog or "").strip().lower()
        source_object_id = str(target.source_object_id or "").strip().lower()
        if source_catalog and source_object_id:
            return f"{source_catalog}:{source_object_id}"
        return _normalize_catalog_token(target.name) or f"{target.ra:.6f}:{target.dec:.6f}"

    def _active_plan_storage_id(self) -> str:
        return str(self._active_plan_id or self._workspace_plan_id or "")

    def _serialize_current_targets(self) -> list[dict[str, object]]:
        return [target.model_dump(mode="json") for target in self.targets]

    def _current_site_snapshot(self) -> dict[str, object]:
        site = getattr(self.table_model, "site", None)
        if isinstance(site, Site):
            return site.model_dump(mode="json")
        if hasattr(self, "obs_combo"):
            name = self.obs_combo.currentText().strip()
            obs = self.observatories.get(name) if hasattr(self, "observatories") else None
            if isinstance(obs, Site):
                return obs.model_dump(mode="json")
            try:
                fallback = Site(
                    name=name or "Custom",
                    latitude=self._read_site_float(self.lat_edit),
                    longitude=self._read_site_float(self.lon_edit),
                    elevation=self._read_site_float(self.elev_edit),
                    limiting_magnitude=self._current_limiting_magnitude(),
                )
            except Exception:
                return {}
            return fallback.model_dump(mode="json")
        return {}

    def _current_plan_snapshot(self) -> dict[str, object]:
        header = self.table_view.horizontalHeader() if hasattr(self, "table_view") else None
        selected = self._selected_target_or_none()
        default_sort_column = TargetTableModel.COL_SCORE
        if header is not None:
            default_sort_column = int(header.sortIndicatorSection())
        return {
            "date": self.date_edit.date().toString(Qt.ISODate) if hasattr(self, "date_edit") else "",
            "site_name": self.obs_combo.currentText().strip() if hasattr(self, "obs_combo") else "",
            "site_snapshot": self._current_site_snapshot(),
            "limit_altitude": float(self.limit_spin.value()) if hasattr(self, "limit_spin") else 30.0,
            "sun_alt_limit": float(self._sun_alt_limit()),
            "min_moon_sep": float(self.min_moon_sep_spin.value()) if hasattr(self, "min_moon_sep_spin") else 0.0,
            "min_score": float(self.min_score_spin.value()) if hasattr(self, "min_score_spin") else 0.0,
            "hide_observed": bool(self.hide_observed_chk.isChecked()) if hasattr(self, "hide_observed_chk") else False,
            "selected_target_name": str(selected.name if isinstance(selected, Target) else ""),
            "view_preset": "observation" if self._table_matches_observation_preset() else "full",
            "default_sort_column": int(default_sort_column),
        }

    def _set_plan_context(self, *, plan_id: str, plan_kind: str, plan_name: str) -> None:
        self._active_plan_id = str(plan_id or "")
        self._active_plan_kind = str(plan_kind or "workspace")
        self._active_plan_name = str(plan_name or "Workspace")
        if self._active_plan_kind == "workspace":
            self._workspace_plan_id = self._active_plan_id
        self._load_ai_messages_for_active_plan()

    def _serialize_ai_messages_for_storage(self) -> list[dict[str, object]]:
        serialized: list[dict[str, object]] = []
        for idx, message in enumerate(self._ai_messages):
            payload: dict[str, object] = {
                "kind": str(message.get("kind", "info") or "info"),
                "text": str(message.get("text", "") or ""),
                "created_at": float(message.get("created_at", datetime.now(timezone.utc).timestamp()) or datetime.now(timezone.utc).timestamp()),
                "sort_order": idx,
            }
            action_targets = message.get("action_targets")
            if isinstance(action_targets, list):
                payload["action_targets"] = [
                    target.model_dump(mode="json")
                    for target in action_targets
                    if isinstance(target, Target)
                ]
            requested_class = str(message.get("requested_class", "") or "").strip()
            if requested_class:
                payload["requested_class"] = requested_class
            primary_target = message.get("primary_target")
            if isinstance(primary_target, Target):
                payload["primary_target"] = primary_target.model_dump(mode="json")
            suggested_targets = message.get("suggested_targets")
            if isinstance(suggested_targets, list):
                payload["suggested_targets"] = [
                    target.model_dump(mode="json")
                    for target in suggested_targets
                    if isinstance(target, Target)
                ]
            serialized.append(payload)
        return serialized

    def _deserialize_ai_messages_from_storage(self, rows: list[dict[str, object]]) -> list[dict[str, Any]]:
        restored: list[dict[str, Any]] = []
        for row in rows:
            message: dict[str, Any] = {
                "kind": str(row.get("kind", "info") or "info"),
                "text": str(row.get("text", "") or ""),
                "created_at": float(row.get("created_at", 0.0) or 0.0),
            }
            action_targets_payload = row.get("action_targets")
            if isinstance(action_targets_payload, list):
                action_targets: list[Target] = []
                for target_payload in action_targets_payload:
                    if not isinstance(target_payload, dict):
                        continue
                    try:
                        action_targets.append(Target(**target_payload))
                    except Exception:
                        continue
                if action_targets:
                    message["action_targets"] = action_targets
            requested_class = str(row.get("requested_class", "") or "").strip()
            if requested_class:
                message["requested_class"] = requested_class
            primary_target_payload = row.get("primary_target")
            if isinstance(primary_target_payload, dict):
                try:
                    message["primary_target"] = Target(**primary_target_payload)
                except Exception:
                    pass
            suggested_targets_payload = row.get("suggested_targets")
            if isinstance(suggested_targets_payload, list):
                suggested_targets: list[Target] = []
                for target_payload in suggested_targets_payload:
                    if not isinstance(target_payload, dict):
                        continue
                    try:
                        suggested_targets.append(Target(**target_payload))
                    except Exception:
                        continue
                if suggested_targets:
                    message["suggested_targets"] = suggested_targets
            restored.append(message)
        return restored

    def _load_ai_messages_for_active_plan(self) -> None:
        storage = getattr(self, "app_storage", None)
        plan_id = self._active_plan_storage_id()
        if storage is None or not plan_id:
            self._ai_messages = []
            self._ai_message_widget_refs = []
            if hasattr(self, "ai_output_layout"):
                self._render_ai_messages()
            self._refresh_ai_panel_action_buttons()
            return
        try:
            rows = storage.chat_history.list_messages(plan_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load AI history for %s: %s", plan_id, exc)
            rows = []
        self._ai_messages = self._deserialize_ai_messages_from_storage(rows)
        self._ai_message_widget_refs = []
        if hasattr(self, "ai_output_layout"):
            self._render_ai_messages()
        self._refresh_ai_panel_action_buttons()

    def _preload_cached_runtime_state_on_startup(self) -> None:
        self._prefetch_bhtom_observatory_presets_on_startup()
        self._prefetch_bhtom_candidates_on_startup()
        self._preload_cached_preview_images_on_startup()
        selected_target = self._selected_target_or_none()
        if selected_target is not None:
            self._update_cutout_preview_for_target(selected_target)
            self._prefetch_finder_charts_for_all_targets(prioritize=selected_target)
        else:
            self._prefetch_cutouts_for_all_targets()
            self._prefetch_finder_charts_for_all_targets()
        self._prefetch_weather_on_startup()

    def _preload_cached_preview_images_on_startup(self) -> None:
        selected_target = self._selected_target_or_none()
        if selected_target is None or not hasattr(self, "aladin_image_label"):
            return
        render_w, render_h = self._cutout_render_dimensions_px(getattr(self, "aladin_image_label", None))
        loaded_cutouts = 0
        loaded_finders = 0
        key = self._cutout_key_for_target(selected_target, render_w, render_h)
        cutout = self._load_pixmap_from_storage_cache("cutout_preview", key)
        if cutout is not None and not cutout.isNull():
            self._cache_cutout_pixmap(key, cutout, persist=False)
            loaded_cutouts += 1
        finder = self._load_pixmap_from_storage_cache("finder_preview", key)
        if finder is not None and not finder.isNull():
            self._cache_finder_pixmap(key, finder, persist=False)
            loaded_finders += 1
        if loaded_cutouts:
            logger.info("Preloaded %d cached Aladin preview into RAM.", loaded_cutouts)
        if loaded_finders:
            logger.info("Preloaded %d cached finder preview into RAM.", loaded_finders)

    def _run_startup_sequence(self) -> None:
        if getattr(self, "_shutting_down", False):
            return
        if hasattr(self, "_replot_timer") and self._replot_timer.isActive():
            self._replot_timer.stop()
        self._defer_startup_preview_updates = True
        restored_workspace = False
        try:
            restored_workspace = self._restore_workspace_on_startup(defer_visual_refresh=True)
        finally:
            self._defer_startup_preview_updates = False
            self._startup_restore_pending = False
        self._validate_site_inputs()
        current_site_name = self.obs_combo.currentText().strip() if hasattr(self, "obs_combo") else ""
        if current_site_name and not getattr(self, "_obs_change_finalize_pending", False):
            self._finalize_observatory_change(current_site_name)
        if restored_workspace:
            QTimer.singleShot(0, self._update_selected_details)
        if not restored_workspace:
            self._replot_timer.start()
        QTimer.singleShot(120, self._preload_cached_runtime_state_on_startup)
        QTimer.singleShot(900, self._warmup_llm_on_startup)

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
            parent=self,
        )
        prefetch_worker.completed.connect(self._on_bhtom_candidate_prefetch_completed)
        prefetch_worker.finished.connect(lambda w=prefetch_worker: self._on_bhtom_candidate_prefetch_finished(w))
        prefetch_worker.finished.connect(prefetch_worker.deleteLater)
        self._bhtom_candidate_prefetch_worker = prefetch_worker
        self._set_bhtom_status("BHTOM: refreshing cache...", busy=True)
        prefetch_worker.start()
        return True

    @Slot(int, list, str)
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

    def _prefetch_weather_on_startup(self) -> None:
        if getattr(self, "_shutting_down", False):
            return
        site = self._current_site_for_weather()
        if not isinstance(site, Site):
            return
        existing = getattr(self, "_startup_weather_worker", None)
        if isinstance(existing, WeatherLiveWorker) and existing.isRunning():
            return

        cloud_month_mode = str(
            self.settings.value("weather/cloudMapMonthMode", "session_month", type=str) or "session_month"
        ).strip().lower()
        cloud_month = int(QDate.currentDate().month()) if cloud_month_mode == "current_month" else int(self.date_edit.date().month())
        worker = WeatherLiveWorker(
            lat=float(site.latitude),
            lon=float(site.longitude),
            elev=float(site.elevation),
            custom_conditions_url=str(getattr(site, "custom_conditions_url", "") or "").strip(),
            cloud_map_source=str(self.settings.value("weather/cloudMapSource", "earthenv", type=str) or "earthenv").strip(),
            cloud_map_month=cloud_month,
            force_refresh=False,
            include_cloud_map=True,
            include_satellite=True,
            storage=getattr(self, "app_storage", None),
            parent=self,
        )
        worker.progress.connect(self._on_startup_weather_progress)
        worker.completed.connect(self._on_startup_weather_completed)
        worker.finished.connect(lambda w=worker: self._on_startup_weather_finished(w))
        worker.finished.connect(worker.deleteLater)
        self._startup_weather_worker = worker
        worker.start()

    @Slot(str, int, int)
    def _on_startup_weather_progress(self, status: str, _step: int, _total: int) -> None:
        logger.info("Startup weather refresh: %s", str(status or "").strip() or "running")

    @Slot(dict)
    def _on_startup_weather_completed(self, payload: dict) -> None:
        if not isinstance(payload, dict):
            return
        sections = payload.get("sections")
        if isinstance(sections, dict):
            forecast = sections.get("forecast") if isinstance(sections.get("forecast"), dict) else {}
            conditions = sections.get("conditions") if isinstance(sections.get("conditions"), dict) else {}
            logger.info(
                "Startup weather refresh ready (forecast=%s, conditions=%s).",
                str(forecast.get("status", "unknown") or "unknown"),
                str(conditions.get("status", "unknown") or "unknown"),
            )

    def _on_startup_weather_finished(self, worker: WeatherLiveWorker) -> None:
        if getattr(self, "_startup_weather_worker", None) is worker:
            self._startup_weather_worker = None

    def _persist_ai_messages_to_storage(self, *, allow_empty_clear: bool = False) -> None:
        storage = getattr(self, "app_storage", None)
        plan_id = self._active_plan_storage_id()
        if storage is None or not plan_id:
            return
        try:
            serialized = self._serialize_ai_messages_for_storage()
            if serialized:
                storage.chat_history.replace_messages(plan_id, serialized)
            elif allow_empty_clear:
                storage.chat_history.clear(plan_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to persist AI history for %s: %s", plan_id, exc)

    def _persist_workspace_now(self) -> None:
        storage = getattr(self, "app_storage", None)
        if storage is None or self._suspend_plan_autosave:
            return
        try:
            record = storage.plans.save_workspace(self._current_plan_snapshot(), self._serialize_current_targets())
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to persist workspace: %s", exc)
            return
        workspace_id = str(record.get("id", "") or "")
        if workspace_id:
            self._workspace_plan_id = workspace_id
            if self._active_plan_kind == "workspace" or not self._active_plan_id:
                self._active_plan_id = workspace_id
                self._active_plan_kind = "workspace"
                self._active_plan_name = str(record.get("name", "Workspace") or "Workspace")

    def _persist_active_plan_now(self) -> None:
        if self._suspend_plan_autosave:
            return
        self._persist_workspace_now()
        storage = getattr(self, "app_storage", None)
        if storage is None or self._active_plan_kind != "saved" or not self._active_plan_id:
            self._persist_ai_messages_to_storage()
            return
        try:
            record = storage.plans.save_named(
                self._active_plan_name,
                self._current_plan_snapshot(),
                self._serialize_current_targets(),
                plan_id=self._active_plan_id,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to autosave active plan '%s': %s", self._active_plan_name, exc)
        else:
            self._active_plan_id = str(record.get("id", self._active_plan_id) or self._active_plan_id)
            self._active_plan_name = str(record.get("name", self._active_plan_name) or self._active_plan_name)
        self._persist_ai_messages_to_storage()

    def _schedule_plan_autosave(self) -> None:
        if self._suspend_plan_autosave or getattr(self, "app_storage", None) is None:
            return
        self._plan_autosave_timer.start(500)

    def _flush_plan_autosave(self) -> None:
        self._persist_active_plan_now()

    def _ensure_named_site_available(self, site_snapshot: dict[str, object], *, preferred_name: str) -> None:
        if not site_snapshot:
            return
        site_name = str(preferred_name or site_snapshot.get("name") or "Custom").strip() or "Custom"
        site_snapshot = dict(site_snapshot)
        site_snapshot["name"] = site_name
        try:
            site = Site(**site_snapshot)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping invalid stored site snapshot for '%s': %s", site_name, exc)
            return
        if site_name not in self.observatories:
            self.observatories[site_name] = site
            self._observatory_preset_keys[site_name] = "custom"
            self._save_custom_observatories()
            self._refresh_observatory_combo(selected_name=site_name)

    def _apply_plan_payload(
        self,
        snapshot: dict[str, object],
        target_payloads: list[dict[str, object]],
        *,
        plan_id: str,
        plan_kind: str,
        plan_name: str,
        defer_visual_refresh: bool = False,
        apply_snapshot_date: bool = True,
    ) -> None:
        loaded_targets: list[Target] = []
        for payload in target_payloads:
            try:
                loaded_targets.append(Target(**payload))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping invalid stored target payload: %s", exc)
        site_snapshot = snapshot.get("site_snapshot")
        site_name = str(snapshot.get("site_name", "") or "")
        if isinstance(site_snapshot, dict):
            self._ensure_named_site_available(site_snapshot, preferred_name=site_name)

        selected_target_name = str(snapshot.get("selected_target_name", "") or "")
        current_site_name = self.obs_combo.currentText().strip() if hasattr(self, "obs_combo") else ""
        current_site = self.table_model.site if isinstance(getattr(self.table_model, "site", None), Site) else None
        target_site = self.observatories.get(site_name) if site_name else None
        self.obs_combo.blockSignals(True)
        try:
            if site_name and site_name in self.observatories:
                self.obs_combo.setCurrentText(site_name)
        finally:
            self.obs_combo.blockSignals(False)
        if _should_apply_observatory_change(current_site_name, current_site, site_name, target_site):
            self._on_obs_change(site_name, defer_replot=defer_visual_refresh)
        elif isinstance(site_snapshot, dict):
            try:
                restored_site = Site(**site_snapshot)
            except Exception:
                restored_site = None
            if isinstance(restored_site, Site):
                self.lat_edit.setText(f"{restored_site.latitude}")
                self.lon_edit.setText(f"{restored_site.longitude}")
                self.elev_edit.setText(f"{restored_site.elevation}")
                self.table_model.site = restored_site

        if apply_snapshot_date:
            iso_date = str(snapshot.get("date", "") or "").strip()
            if iso_date:
                qdate = QDate.fromString(iso_date, Qt.ISODate)
                if qdate.isValid():
                    self.date_edit.setDate(qdate)
        else:
            self.date_edit.setDate(QDate.currentDate())
        if "limit_altitude" in snapshot:
            self.limit_spin.setValue(int(round(float(snapshot.get("limit_altitude", self.limit_spin.value()) or self.limit_spin.value()))))
        if "sun_alt_limit" in snapshot:
            self.sun_alt_limit_spin.setValue(int(round(float(snapshot.get("sun_alt_limit", self.sun_alt_limit_spin.value()) or self.sun_alt_limit_spin.value()))))
        if "min_moon_sep" in snapshot:
            self.min_moon_sep_spin.setValue(int(round(float(snapshot.get("min_moon_sep", self.min_moon_sep_spin.value()) or self.min_moon_sep_spin.value()))))
        if "min_score" in snapshot:
            self.min_score_spin.setValue(int(round(float(snapshot.get("min_score", self.min_score_spin.value()) or self.min_score_spin.value()))))
        if "hide_observed" in snapshot:
            self.hide_observed_chk.setChecked(bool(snapshot.get("hide_observed")))

        self.target_metrics.clear()
        self.target_windows.clear()
        self.table_model.reset_targets(loaded_targets)
        self._recompute_recommended_order_cache()
        self._apply_table_settings()
        default_preset = str(self.settings.value("table/viewPreset", "full", type=str) or "full")
        preset = str(snapshot.get("view_preset", default_preset) or default_preset)
        self._apply_column_preset(preset, save=False)
        sort_col_raw = snapshot.get("default_sort_column", TargetTableModel.COL_SCORE)
        try:
            sort_col = int(sort_col_raw)
        except Exception:
            sort_col = TargetTableModel.COL_SCORE
        if 0 <= sort_col < self.table_model.columnCount():
            self.table_view.sortByColumn(sort_col, Qt.AscendingOrder)
        else:
            self._apply_default_sort()
        self._fetch_missing_magnitudes_async()
        self._set_plan_context(plan_id=plan_id, plan_kind=plan_kind, plan_name=plan_name)
        if selected_target_name:
            for row_idx, target in enumerate(self.targets):
                if _normalize_catalog_token(target.name) == _normalize_catalog_token(selected_target_name):
                    self.table_view.selectRow(row_idx)
                    break
        if not defer_visual_refresh:
            self._update_selected_details()
        self._refresh_target_color_map()
        self._emit_table_data_changed()
        self._run_plan()

    def _restore_plan_record(
        self,
        record: dict[str, object],
        *,
        defer_visual_refresh: bool = False,
        apply_snapshot_date: bool = False,
    ) -> None:
        snapshot = dict(record.get("snapshot") or {})
        target_payloads = [
            dict(payload)
            for payload in record.get("targets", [])
            if isinstance(payload, dict)
        ]
        self._suspend_plan_autosave = True
        try:
            self._apply_plan_payload(
                snapshot,
                target_payloads,
                plan_id=str(record.get("id", "") or ""),
                plan_kind=str(record.get("plan_kind", "workspace") or "workspace"),
                plan_name=str(record.get("name", "Workspace") or "Workspace"),
                defer_visual_refresh=defer_visual_refresh,
                apply_snapshot_date=apply_snapshot_date,
            )
        finally:
            self._suspend_plan_autosave = False
        self._persist_workspace_now()

    def _restore_workspace_on_startup(self, *, defer_visual_refresh: bool = False) -> bool:
        storage = getattr(self, "app_storage", None)
        if storage is None:
            return False
        try:
            workspace = storage.plans.load_workspace()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to restore workspace from storage: %s", exc)
            workspace = None
        if workspace:
            self._restore_plan_record(
                workspace,
                defer_visual_refresh=defer_visual_refresh,
                apply_snapshot_date=False,
            )
            if str(workspace.get("id", "")).strip():
                self._workspace_plan_id = str(workspace.get("id"))
            return True
        self._persist_workspace_now()
        self._load_ai_messages_for_active_plan()
        return False

    def _load_plan_from_json_path(self, file_path: str, *, persist_workspace: bool = True) -> None:
        try:
            with open(file_path, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
            target_payloads = [dict(entry) for entry in data if isinstance(entry, dict)]
        except Exception as e:
            QMessageBox.critical(self, "Load plan error", str(e))
            return
        self._suspend_plan_autosave = True
        try:
            self._apply_plan_payload(
                {},
                target_payloads,
                plan_id=str(self._workspace_plan_id or ""),
                plan_kind="workspace",
                plan_name="Workspace",
            )
        finally:
            self._suspend_plan_autosave = False
        if persist_workspace:
            self._persist_workspace_now()
            self._set_plan_context(
                plan_id=str(self._workspace_plan_id or self._active_plan_id),
                plan_kind="workspace",
                plan_name="Workspace",
            )
            self._persist_ai_messages_to_storage()

    @Slot()
    def _import_plan_json(self):
        fn, _ = QFileDialog.getOpenFileName(
            self, "Import plan JSON", str(Path.cwd()), "JSON files (*.json)"
        )
        if not fn:
            return
        self._load_plan_from_json_path(fn, persist_workspace=True)

    @Slot()
    def _load_plan(self):
        self._open_saved_plan()

    @Slot()
    def _save_plan_as(self) -> None:
        storage = getattr(self, "app_storage", None)
        if storage is None:
            QMessageBox.information(self, "Save Plan", "SQLite storage is not available.")
            return
        default_name = self._active_plan_name if self._active_plan_kind == "saved" else ""
        name, ok = QInputDialog.getText(self, "Save Plan As", "Plan name:", text=default_name)
        if not ok:
            return
        normalized_name = str(name or "").strip()
        if not normalized_name:
            QMessageBox.information(self, "Save Plan", "Plan name cannot be empty.")
            return
        try:
            record = storage.plans.save_named(
                normalized_name,
                self._current_plan_snapshot(),
                self._serialize_current_targets(),
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Save Plan", f"Could not save the plan:\n{exc}")
            return
        self._persist_workspace_now()
        self._set_plan_context(
            plan_id=str(record.get("id", "") or ""),
            plan_kind="saved",
            plan_name=str(record.get("name", normalized_name) or normalized_name),
        )
        self._persist_ai_messages_to_storage()
        self.statusBar().showMessage(f"Saved plan '{self._active_plan_name}'.", 4000)

    @Slot()
    def _open_saved_plan(self) -> None:
        storage = getattr(self, "app_storage", None)
        if storage is None:
            QMessageBox.information(self, "Open Saved Plan", "SQLite storage is not available.")
            return
        try:
            plans = storage.plans.list_saved()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Open Saved Plan", f"Could not read saved plans:\n{exc}")
            return
        if not plans:
            QMessageBox.information(self, "Open Saved Plan", "No saved plans were found.")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Open Saved Plan")
        dlg.resize(520, 420)
        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        search_edit = QLineEdit(dlg)
        search_edit.setPlaceholderText("Filter saved plans...")
        plan_list = QListWidget(dlg)
        info_label = QLabel(dlg)
        info_label.setObjectName("SectionHint")
        delete_btn = QPushButton("Delete", dlg)
        _set_button_variant(delete_btn, "neutral")
        buttons = QDialogButtonBox(QDialogButtonBox.Open | QDialogButtonBox.Cancel, parent=dlg)
        open_btn = buttons.button(QDialogButtonBox.Open)

        current_plan_map: dict[str, dict[str, object]] = {}

        def refresh_items() -> None:
            current_plan_map.clear()
            plan_list.clear()
            query = str(search_edit.text() or "").strip().lower()
            for item in plans:
                plan_name = str(item.get("name", "") or "")
                if query and query not in plan_name.lower():
                    continue
                label = f"{plan_name} ({int(item.get('target_count', 0) or 0)} targets)"
                widget_item = QListWidgetItem(label, plan_list)
                plan_id = str(item.get("id", "") or "")
                widget_item.setData(Qt.UserRole, plan_id)
                current_plan_map[plan_id] = item
            has_selection = plan_list.count() > 0
            if has_selection:
                plan_list.setCurrentRow(0)
            open_btn.setEnabled(has_selection)
            delete_btn.setEnabled(has_selection)

        def update_info() -> None:
            current_item = plan_list.currentItem()
            if current_item is None:
                info_label.setText("No saved plan selected.")
                return
            plan_id = str(current_item.data(Qt.UserRole) or "")
            plan = current_plan_map.get(plan_id, {})
            updated_at = float(plan.get("updated_at", 0.0) or 0.0)
            if updated_at > 0:
                stamp = datetime.fromtimestamp(updated_at, tz=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M")
            else:
                stamp = "unknown"
            info_label.setText(
                f"Targets: {int(plan.get('target_count', 0) or 0)} | Last update: {stamp}"
            )

        def delete_selected() -> None:
            current_item = plan_list.currentItem()
            if current_item is None:
                return
            plan_id = str(current_item.data(Qt.UserRole) or "")
            plan = current_plan_map.get(plan_id, {})
            plan_name = str(plan.get("name", "this plan") or "this plan")
            answer = QMessageBox.question(
                dlg,
                "Delete Saved Plan",
                f"Delete '{plan_name}' from SQLite storage?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
            try:
                deleted = storage.plans.delete_plan(plan_id)
            except Exception as exc:  # noqa: BLE001
                QMessageBox.warning(dlg, "Delete Saved Plan", f"Could not delete the plan:\n{exc}")
                return
            if not deleted:
                QMessageBox.information(dlg, "Delete Saved Plan", "The selected plan no longer exists.")
                return
            nonlocal plans
            plans = [item for item in plans if str(item.get("id", "")) != plan_id]
            refresh_items()
            update_info()

        search_edit.textChanged.connect(lambda _text: (refresh_items(), update_info()))
        plan_list.currentItemChanged.connect(lambda *_args: update_info())
        plan_list.itemDoubleClicked.connect(lambda *_args: dlg.accept())
        delete_btn.clicked.connect(delete_selected)

        layout.addWidget(search_edit)
        layout.addWidget(plan_list, 1)
        layout.addWidget(info_label)
        controls_row = QHBoxLayout()
        controls_row.addWidget(delete_btn)
        controls_row.addStretch(1)
        layout.addLayout(controls_row)
        layout.addWidget(buttons)

        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        refresh_items()
        update_info()
        if dlg.exec() != QDialog.Accepted:
            return
        current_item = plan_list.currentItem()
        if current_item is None:
            return
        plan_id = str(current_item.data(Qt.UserRole) or "")
        try:
            record = storage.plans.load_plan(plan_id)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Open Saved Plan", f"Could not load the plan:\n{exc}")
            return
        if not record:
            QMessageBox.information(self, "Open Saved Plan", "The selected plan is no longer available.")
            return
        self._restore_plan_record(record)
        self.statusBar().showMessage(f"Opened '{record.get('name', 'saved plan')}'.", 4000)

    @Slot()
    def _show_observation_history(self) -> None:
        storage = getattr(self, "app_storage", None)
        if storage is None:
            QMessageBox.information(self, "Observation History", "SQLite storage is not available.")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Observation History")
        dlg.resize(880, 460)
        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        search_edit = QLineEdit(dlg)
        search_edit.setPlaceholderText("Search by target or site...")
        table = QTableWidget(dlg)
        table.setColumnCount(5)
        table.setHorizontalHeaderLabels(["Observed At", "Target", "Site", "Source", "Notes"])
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectRows)
        table.setSelectionMode(QTableWidget.SingleSelection)
        table.verticalHeader().setVisible(False)
        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.Stretch)
        export_btn = QPushButton("Export CSV", dlg)
        buttons = QDialogButtonBox(QDialogButtonBox.Close, parent=dlg)
        current_rows: list[dict[str, object]] = []

        def load_rows() -> list[dict[str, object]]:
            try:
                return storage.observation_log.list_entries(search=search_edit.text())
            except Exception as exc:  # noqa: BLE001
                QMessageBox.warning(dlg, "Observation History", f"Could not load history:\n{exc}")
                return []

        def refresh_table() -> None:
            rows = load_rows()
            current_rows[:] = rows
            table.setRowCount(len(rows))
            for row_idx, row in enumerate(rows):
                observed_at = float(row.get("observed_at", 0.0) or 0.0)
                if observed_at > 0:
                    stamp = datetime.fromtimestamp(observed_at, tz=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M")
                else:
                    stamp = "-"
                values = [
                    stamp,
                    str(row.get("target_name", "") or ""),
                    str(row.get("site_name", "") or ""),
                    str(row.get("source", "") or ""),
                    str(row.get("notes", "") or ""),
                ]
                for col_idx, value in enumerate(values):
                    table.setItem(row_idx, col_idx, QTableWidgetItem(value))
            table.resizeRowsToContents()

        def export_csv() -> None:
            if not current_rows:
                QMessageBox.information(dlg, "Observation History", "No observation entries to export.")
                return
            default_name = f"astroplanner-observation-history-{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
            file_path, _ = QFileDialog.getSaveFileName(
                dlg,
                "Export Observation History",
                str(Path.home() / default_name),
                "CSV files (*.csv)",
            )
            if not file_path:
                return
            try:
                with open(file_path, "w", encoding="utf-8", newline="") as fh:
                    writer = csv.writer(fh)
                    writer.writerow(["observed_at", "target_name", "target_key", "site_name", "source", "notes"])
                    for row in current_rows:
                        writer.writerow(
                            [
                                float(row.get("observed_at", 0.0) or 0.0),
                                str(row.get("target_name", "") or ""),
                                str(row.get("target_key", "") or ""),
                                str(row.get("site_name", "") or ""),
                                str(row.get("source", "") or ""),
                                str(row.get("notes", "") or ""),
                            ]
                        )
            except Exception as exc:  # noqa: BLE001
                QMessageBox.warning(dlg, "Observation History", f"Could not export CSV:\n{exc}")
                return
            self.statusBar().showMessage(f"Exported observation history to {Path(file_path).name}.", 4000)

        search_edit.textChanged.connect(lambda _text: refresh_table())
        export_btn.clicked.connect(export_csv)
        buttons.rejected.connect(dlg.reject)

        layout.addWidget(search_edit)
        layout.addWidget(table, 1)
        controls = QHBoxLayout()
        controls.addWidget(export_btn)
        controls.addStretch(1)
        layout.addLayout(controls)
        layout.addWidget(buttons)
        refresh_table()
        dlg.exec()

    def _record_observation_if_needed(self, target: Target, *, was_observed: bool, source: str) -> None:
        if was_observed or not target.observed:
            return
        storage = getattr(self, "app_storage", None)
        if storage is None:
            return
        site_payload = self._current_site_snapshot()
        site_name = str(site_payload.get("name", self.obs_combo.currentText() if hasattr(self, "obs_combo") else "") or "")
        try:
            storage.observation_log.append(
                target_name=target.name,
                target_key=self._plan_target_key(target),
                target_payload=target.model_dump(mode="json"),
                site_name=site_name,
                site_payload=site_payload,
                notes=str(target.notes or ""),
                source=source,
                plan_id=self._active_plan_storage_id() or None,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to persist observation log entry for %s: %s", target.name, exc)

    # .....................................................
    # file / target helpers
    # .....................................................
    @Slot()
    def _load_targets(self):  # noqa: D401
        """Load a CSV/TSV/MPC target list file."""
        fn, _ = QFileDialog.getOpenFileName(
            self,
            "Open target list",
            str(Path.cwd()),
            "Text files (*.csv *.tsv *.txt)",
        )
        if not fn:
            return

        try:
            loaded_targets: list[Target] = []
            self.target_metrics.clear()
            self.target_windows.clear()
            with open(fn, newline="", encoding="utf-8") as fh:
                dialect = csv.Sniffer().sniff(fh.read(2048))
                fh.seek(0)
                rdr = csv.DictReader(fh, dialect=dialect)
                for row in rdr:
                    priority = int(row.get("priority", 3) or 3)
                    priority = min(5, max(1, priority))
                    magnitude = row.get("magnitude") or None
                    size_arcmin = row.get("size_arcmin") or None
                    observed_raw = str(row.get("observed", "")).strip().lower()
                    loaded_targets.append(
                        Target(
                            name=row["name"],
                            ra=parse_ra_to_deg(row["ra"]),
                            dec=parse_dec_to_deg(row["dec"]),
                            object_type=row.get("object_type", row.get("type", "")) or "",
                            magnitude=float(magnitude) if magnitude is not None else None,
                            size_arcmin=float(size_arcmin) if size_arcmin is not None else None,
                            priority=priority,
                            observed=observed_raw in {"1", "true", "yes", "y", "tak"},
                            notes=row.get("notes", "") or "",
                        )
                    )
            self.table_model.reset_targets(loaded_targets)
        except (FileNotFoundError, KeyError, ValueError, ValidationError, csv.Error) as exc:
            QMessageBox.critical(self, "Load error", str(exc))
            return

        # Refresh table view after successful load
        # Reapply settings & default sort after layout change
        self._apply_table_settings()
        self._apply_default_sort()
        self._fetch_missing_magnitudes_async()

    @Slot()
    def _add_target_dialog(self):
        """Prompt user for a target name or coordinate string and append it."""
        default_source = _normalize_catalog_token(
            self.settings.value("general/targetSearchSource", "simbad", type=str)
        )
        dlg = AddTargetDialog(
            self._resolve_target,
            self._enrich_target_metadata_for_dialog,
            self,
            source_options=TARGET_SEARCH_SOURCES,
            default_source=default_source,
        )
        if dlg.exec() != QDialog.Accepted:
            return
        self.settings.setValue("general/targetSearchSource", dlg.selected_source())
        try:
            target = dlg.build_target()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Resolve error", str(exc))
            return
        self._append_target_to_plan(target)

    def _plan_contains_target(self, target: Target) -> bool:
        return any(_targets_match(existing, target) for existing in self.targets)

    def _try_fast_append_visibility_update(self, target: Target) -> bool:
        if getattr(self, "worker", None) is not None and self.worker.isRunning():
            return False
        payload = getattr(self, "last_payload", None)
        if not isinstance(payload, dict):
            return False
        current_key = self._current_visibility_context_key()
        payload_key = str(payload.get("site_key", "") or "")
        if not current_key or not payload_key or payload_key != current_key:
            return False
        if target.name in payload:
            return True
        if "times" not in payload or "moon_ra" not in payload or "moon_dec" not in payload:
            return False
        if not self._validate_site_inputs():
            return False
        try:
            site = self._build_runtime_site_from_inputs()
            tz_name = str(payload.get("tz", site.timezone_name) or site.timezone_name)
            observer = Observer(location=site.to_earthlocation(), timezone=tz_name)
            times_num = np.array(payload["times"], dtype=float)
            times = Time(mdates.num2date(times_num))
            fixed = FixedTarget(name=target.name, coord=target.skycoord)
            altaz = observer.altaz(times, fixed)
            moon_ra = np.array(payload.get("moon_ra", []), dtype=float)
            moon_dec = np.array(payload.get("moon_dec", []), dtype=float)
            if moon_ra.size != times_num.size or moon_dec.size != times_num.size:
                return False
            moon_coords = SkyCoord(ra=moon_ra * u.deg, dec=moon_dec * u.deg)
            moon_sep = target.skycoord.separation(moon_coords).deg
            merged = dict(payload)
            merged[target.name] = {
                "altitude": np.array(altaz.alt.deg, dtype=float),  # type: ignore[arg-type]
                "azimuth": np.array(altaz.az.deg, dtype=float),  # type: ignore[arg-type]
                "moon_sep": np.array(moon_sep, dtype=float),
            }
            self._update_plot(merged)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.info("Fast append visibility update failed for %s: %s", target.name, exc)
            return False

    def _append_target_to_plan(
        self,
        target: Target,
        refresh: bool = True,
        notify_duplicate: bool = True,
    ) -> bool:
        if self._plan_contains_target(target):
            if notify_duplicate:
                QMessageBox.information(self, "Already in plan", f"{target.name} is already present in the current plan.")
            return False

        target_copy = Target(**target.model_dump())
        self._ensure_known_target_type(target_copy)
        self.table_model.append_target(target_copy)
        if refresh:
            self._recompute_recommended_order_cache()
            self._apply_table_settings()
            self._apply_default_sort()
            self._refresh_target_color_map()
            self._emit_table_data_changed()
            self._fetch_missing_magnitudes_async()
            if not self._try_fast_append_visibility_update(target_copy):
                self._replot_timer.start()

            for row_idx, existing in enumerate(self.targets):
                if existing is target_copy or _targets_match(existing, target_copy):
                    self.table_view.selectRow(row_idx)
                    self.table_view.scrollTo(self.table_model.index(row_idx, TargetTableModel.COL_NAME))
                    break
            self._update_selected_details()
            self._prefetch_cutouts_for_all_targets(prioritize=self._selected_target_or_none())
            self._prefetch_finder_charts_for_all_targets(prioritize=self._selected_target_or_none())
        self._schedule_plan_autosave()
        return True

    @Slot()
    def _run_plan(self):
        """Kick off the worker thread unless one is already running."""
        logger.info("Starting new visibility calculation …")
        self._ensure_visibility_plot_widgets()
        if self.worker and self.worker.isRunning():
            self._queued_plan_run = True
            try:
                self.worker.requestInterruption()
            except Exception:
                pass
            self._begin_visibility_refresh("Updating visibility for the new context…")
            return
        if not self._validate_site_inputs():
            return
        try:
            site = self._build_runtime_site_from_inputs()
            settings = SessionSettings(
                date=self.date_edit.date(),
                site=site,
                limit_altitude=float(self.limit_spin.value()),
                time_samples=self.settings.value("general/timeSamples", 240, type=int),
            )
        except (ValidationError, ValueError) as exc:
            QMessageBox.critical(self, "Invalid input", str(exc))
            return

        self.table_model.site = site
        self._queued_plan_run = False
        self._pending_visibility_context_key = self._visibility_context_key_from_parts(
            site_name=site.name,
            latitude=float(site.latitude),
            longitude=float(site.longitude),
            elevation=float(site.elevation),
            obs_date=settings.date,
            time_samples=int(settings.time_samples),
            limit_altitude=float(settings.limit_altitude),
        )
        # Update the observation limit for table coloring
        self.table_model.limit = settings.limit_altitude
        self._emit_table_data_changed()
        self._calc_started_at = perf_counter()
        self._update_status_bar()

        self.worker = AstronomyWorker(self.targets, settings, parent=self)
        self.worker.aborted.connect(self._on_astronomy_worker_aborted)
        self.worker.finished.connect(self._update_plot)
        self.worker.finished.connect(lambda _: self.plot_canvas.setEnabled(True))
        if getattr(self, "visibility_web", None) is not None:
            self.worker.finished.connect(lambda _: self.visibility_web.setEnabled(True))
        # disable canvas during computation
        self.plot_canvas.setEnabled(False)
        if getattr(self, "visibility_web", None) is not None:
            self.visibility_web.setEnabled(False)
        self._begin_visibility_refresh("Computing night tracks…")
        self.worker.start()

    def _remove_target(self, row: int):
        if not (0 <= row < len(self.targets)):
            return
        removed_name = self.targets[row].name
        self.target_metrics.pop(removed_name, None)
        self.target_windows.pop(removed_name, None)
        self.table_model.remove_rows([row])
        self._recompute_recommended_order_cache()
        # Reapply settings & default sort after layout change
        self._apply_table_settings()
        self._apply_default_sort()
        self._update_selected_details()
        # Debounce and update plot after removing a target
        self._schedule_plan_autosave()
        self._replot_timer.start()

    @Slot()
    def _delete_selected_targets(self):
        """Delete all currently selected targets via Ctrl/Cmd+Delete."""
        # Collect selected rows, sort descending to avoid reindexing issues
        rows = self._selected_rows()
        if not rows:
            return
        removed = self.table_model.remove_rows(rows)
        for tgt in removed:
            self.target_metrics.pop(tgt.name, None)
            self.target_windows.pop(tgt.name, None)
        self._recompute_recommended_order_cache()
        self._apply_table_settings()
        self._apply_default_sort()
        self._update_selected_details()
        self._schedule_plan_autosave()
        self._replot_timer.start()

    def _toggle_observed_row(self, row: int):
        if 0 <= row < len(self.targets):
            was_observed = bool(self.targets[row].observed)
            self.targets[row].observed = not self.targets[row].observed
            self._emit_table_data_changed()
            self._update_selected_details()
            self._record_observation_if_needed(self.targets[row], was_observed=was_observed, source="toggle_row")
            self._schedule_plan_autosave()
            if self.last_payload is not None:
                self._update_plot(self.last_payload)
            else:
                self._replot_timer.start()

    @Slot()
    def _toggle_observed_selected(self):
        rows = self._selected_rows()
        if not rows:
            return
        for row in rows:
            if 0 <= row < len(self.targets):
                target = self.targets[row]
                was_observed = bool(target.observed)
                target.observed = not target.observed
                self._record_observation_if_needed(target, was_observed=was_observed, source="toggle_selected")
        self._emit_table_data_changed()
        self._update_selected_details()
        self._schedule_plan_autosave()
        if self.last_payload is not None:
            self._update_plot(self.last_payload)
        else:
            self._replot_timer.start()

    def _emit_table_data_changed(self):
        coordinator = getattr(self, "_table_coordinator", None)
        if coordinator is not None:
            coordinator.emit_table_data_changed()

    def _apply_table_row_visibility(self):
        coordinator = getattr(self, "_table_coordinator", None)
        if coordinator is not None:
            coordinator.apply_table_row_visibility()

    def _clear_table_dynamic_cache(self):
        coordinator = getattr(self, "_table_coordinator", None)
        if coordinator is not None:
            coordinator.clear_table_dynamic_cache()

    def _refresh_table_buttons(self):
        # Row widgets were replaced by a context menu to keep the table fast.
        return

    @Slot()
    def _on_astronomy_worker_aborted(self) -> None:
        worker = getattr(self, "worker", None)
        if worker is not None and worker.isRunning():
            QTimer.singleShot(40, self._on_astronomy_worker_aborted)
            return
        self.worker = None
        if self._queued_plan_run:
            self._queued_plan_run = False
            QTimer.singleShot(0, self._run_plan)

    @Slot(dict)
    def _update_plot(self, data: dict):
        """Redraw the altitude plot with new data from the worker."""
        logger.info("Altitude plot refresh (%d targets)", len(self.targets))
        self._ensure_visibility_plot_widgets()
        if self.plot_canvas is None or self.ax_alt is None:
            return
        sender = self.sender()
        if sender is getattr(self, "worker", None):
            self.worker = None
        payload_key = str(data.get("site_key", "") or "")
        current_key = self._current_visibility_context_key()
        if payload_key and current_key and payload_key != current_key:
            if self._queued_plan_run:
                self._queued_plan_run = False
                QTimer.singleShot(0, self._run_plan)
            return
        self.last_payload = data
        # Keep full visibility data around for polar path plotting
        self.full_payload = data
        # Reset stored visibility lines for this redraw
        self.vis_lines.clear()
        self.ax_alt.clear()
        plot_bg = self._theme_color("plot_bg", "#0f1825")
        plot_panel_bg = self._theme_color("plot_panel_bg", "#162334")
        plot_text = self._theme_color("plot_text", "#d7e4f0")
        plot_grid = self._theme_color("plot_grid", "#2f4666")
        plot_guide = self._theme_color("plot_guide", "#62748a")
        soft_grid_color = self._mix_qcolors(self._theme_qcolor("plot_grid", plot_grid), self._theme_qcolor("plot_panel_bg", plot_panel_bg), 0.34)
        self.plot_canvas.figure.patch.set_facecolor(plot_bg)
        self.ax_alt.set_facecolor(plot_panel_bg)

        # Localise the timezone
        tz = pytz.timezone(data.get("tz", "UTC"))

        # Convert times array to that timezone
        times = [t.astimezone(tz) for t in mdates.num2date(data["times"])]
        times_nums = mdates.date2num(times)
        # Generate line colors from palette, overridden by optional per-target color.
        line_palette = self._line_palette()
        target_colors = [
            self._target_plot_color_css(tgt, idx, line_palette)
            for idx, tgt in enumerate(self.targets)
        ]
        limit = float(self.limit_spin.value())
        sample_hours = 24.0 / max(len(times) - 1, 1)

        self.target_metrics.clear()
        self.target_windows.clear()
        score_vals: list[float] = []
        hour_vals: list[float] = []
        row_enabled: list[bool] = []
        sun_alt_series = np.array(data.get("sun_alt", np.full(len(times), np.nan)), dtype=float)
        sun_alt_limit = self._sun_alt_limit()
        obs_sun_mask = np.isfinite(sun_alt_series) & (sun_alt_series <= sun_alt_limit)
        tz_name = data.get("tz", "UTC")
        site = Site(
            name="",
            latitude=self._read_site_float(self.lat_edit),
            longitude=self._read_site_float(self.lon_edit),
            elevation=self._read_site_float(self.elev_edit),
            limiting_magnitude=self._current_limiting_magnitude(),
        )
        observer_now = Observer(location=site.to_earthlocation(), timezone=tz_name)
        now_dt = datetime.now(pytz.timezone(tz_name))
        # Prepare ephem observer once and use it for both filtering and labels.
        eph_obs = ephem.Observer()
        eph_obs.lat = str(site.latitude)
        eph_obs.lon = str(site.longitude)
        eph_obs.elevation = site.elevation
        eph_obs.date = now_dt
        moon = ephem.Moon(eph_obs)
        moon_coord = SkyCoord(ra=Angle(moon.ra, u.rad), dec=Angle(moon.dec, u.rad))

        # Keep table Name cell colors aligned with plot line colors.
        self._refresh_target_color_map(line_palette)

        for idx, tgt in enumerate(self.targets):
            alt = np.array(data[tgt.name]["altitude"], dtype=float)
            moon_sep_series = np.array(data[tgt.name].get("moon_sep", np.full_like(alt, np.nan)), dtype=float)
            color = target_colors[idx] if idx < len(target_colors) else self._target_plot_color_css(tgt, idx, line_palette)
            metrics = compute_target_metrics(
                altitude_deg=alt,
                moon_sep_deg=moon_sep_series,
                limit_altitude=limit,
                sample_hours=sample_hours,
                priority=tgt.priority,
                observed=tgt.observed,
                valid_mask=obs_sun_mask,
            )
            self.target_metrics[tgt.name] = metrics
            score_vals.append(metrics.score)
            hour_vals.append(metrics.hours_above_limit)

            limit_mask = (alt >= limit) & obs_sun_mask
            if limit_mask.any():
                vis_idx = np.where(limit_mask)[0]
                runs = np.split(vis_idx, np.where(np.diff(vis_idx) != 1)[0] + 1)
                best_run = max(runs, key=len)
                start_idx = int(best_run[0])
                end_idx = min(int(best_run[-1]) + 1, len(times) - 1)
                self.target_windows[tgt.name] = (times[start_idx], times[end_idx])

            moon_sep_now = float(tgt.skycoord.separation(moon_coord).deg)
            passes_filters = self._passes_active_filters(tgt, metrics.score, moon_sep_now)
            row_enabled.append(passes_filters)
            if not passes_filters:
                continue

            # Points above horizon
            vis_mask = np.isfinite(alt) & (alt > 0)
            if not vis_mask.any():
                continue
            # Dashed base path for full visible range.
            # Use NaN outside mask so matplotlib breaks segments instead of drawing
            # straight connector lines through non-visible intervals.
            alt_vis = np.array(alt, copy=True)
            alt_vis[~vis_mask] = np.nan
            plot_alt_vis = self._plot_y_values(alt_vis)
            base_line, = self.ax_alt.plot(
                times_nums, plot_alt_vis,
                color=color, linewidth=1.4,
                linestyle="--", alpha=0.3, zorder=1
            )
            self.vis_lines.append((tgt.name, base_line, False))
            # Solid overlay for portions above limit
            high_mask = np.isfinite(alt) & (alt >= limit) & obs_sun_mask
            if high_mask.any():
                alt_high = np.array(alt, copy=True)
                alt_high[~high_mask] = np.nan
                plot_alt_high = self._plot_y_values(alt_high)
                solid_line, = self.ax_alt.plot(
                    times_nums, plot_alt_high,
                    color=color, linewidth=1.4,
                    linestyle="-", alpha=1.0, zorder=2
                )
                self.vis_lines.append((tgt.name, solid_line, True))

        # ------------------------------------------------------------------
        # Compute and cache current alt, az, sep for each target for the table

        current_alts: list[float] = []
        current_azs: list[float] = []
        current_seps: list[float] = []
        if self.targets:
            try:
                coords_now = SkyCoord(
                    ra=np.array([float(t.ra) for t in self.targets], dtype=float) * u.deg,
                    dec=np.array([float(t.dec) for t in self.targets], dtype=float) * u.deg,
                )
                altaz_now_all = observer_now.altaz(Time(now_dt), coords_now)
                alt_vals = np.array(altaz_now_all.alt.deg, dtype=float)  # type: ignore[arg-type]
                az_vals = np.array(altaz_now_all.az.deg, dtype=float)  # type: ignore[arg-type]
                sep_vals = np.array(coords_now.separation(moon_coord).deg, dtype=float)
                current_alts = [float(value) for value in np.ravel(alt_vals)]
                current_azs = [float(value) for value in np.ravel(az_vals)]
                current_seps = [float(value) for value in np.ravel(sep_vals)]
                if len(current_alts) != len(self.targets):
                    raise ValueError("Unexpected alt/az vector size.")
            except Exception:
                current_alts = []
                current_azs = []
                current_seps = []
                for tgt in self.targets:
                    fixed = FixedTarget(name=tgt.name, coord=tgt.skycoord)
                    altaz_now = observer_now.altaz(Time(now_dt), fixed)
                    current_alts.append(float(altaz_now.alt.deg))  # type: ignore[arg-type]
                    current_azs.append(float(altaz_now.az.deg))  # type: ignore[arg-type]
                    current_seps.append(float(tgt.skycoord.separation(moon_coord).deg))  # type: ignore[arg-type]

        # Assign to model and refresh table
        self.table_model.current_alts = current_alts
        self.table_model.current_azs = current_azs
        self.table_model.current_seps = current_seps
        self.table_model.scores = score_vals
        self.table_model.hours_above_limit = hour_vals
        self.table_model.row_enabled = row_enabled
        self._recompute_recommended_order_cache()
        self._reapply_current_table_sort()
        self._apply_table_row_visibility()
        self._emit_table_data_changed()

        # Compute and display current sun and moon altitudes
        sun_obs = ephem.Sun(eph_obs)
        moon_obs = ephem.Moon(eph_obs)
        sun_alt_curr = sun_obs.alt * 180.0 / math.pi
        moon_alt_curr = moon_obs.alt * 180.0 / math.pi
        self.sun_alt_label.setText(f"{sun_alt_curr:.1f}°")
        self.moon_alt_label.setText(f"{moon_alt_curr:.1f}°")

        # Compute and display sidereal time
        sidereal = Time(now_dt).sidereal_time('apparent', site.to_earthlocation().lon)
        self.sidereal_label.setText(sidereal.to_string(unit=u.hour, sep=":", pad=True, precision=0))

        # ------------------------------------------------------------------
        # Twilight shading (civil, nautical, astronomical), only when valid
        # ------------------------------------------------------------------
        civil_col = self._theme_color("plot_twilight_civil", "#FFF2CC")
        naut_col = self._theme_color("plot_twilight_naut", "#CCE5FF")
        astro_col = self._theme_color("plot_twilight_astro", "#D9D9D9")
        start_dt, end_dt, ev = self._visibility_time_window(data, tz)
        self.ax_alt.set_xlim(start_dt, end_dt)
        xmin, xmax = self.ax_alt.get_xlim()

        # Segments to shade, only if both endpoints exist and start < end, and within window
        segments = [
            ("sunset", "dusk_civ", civil_col),
            ("dusk_civ", "dusk_naut", naut_col),
            ("dusk_naut", "dusk", astro_col),
            ("dawn", "dawn_naut", astro_col),
            ("dawn_naut", "dawn_civ", naut_col),
            ("dawn_civ", "sunrise", civil_col),
        ]
        for start_key, end_key, col in segments:
            if start_key in ev and end_key in ev:
                s_num = mdates.date2num(ev[start_key])
                e_num = mdates.date2num(ev[end_key])
                # Only shade if the segment is within the visible window
                if s_num < e_num and s_num < xmax and e_num > xmin:
                    # Clip to window
                    s_dt = max(float(s_num), float(xmin))
                    e_dt = min(float(e_num), float(xmax))
                    self.ax_alt.axvspan(mdates.num2date(s_dt), mdates.num2date(e_dt),
                                        color=col, alpha=0.4, zorder=0)

        # Guide lines at each valid boundary
        for key, dt in ev.items():
            num = mdates.date2num(dt)
            if xmin <= num <= xmax:
                self.ax_alt.axvline(dt, color=self._qcolor_rgba_mpl(soft_grid_color, 0.24), linestyle="--", alpha=1.0, linewidth=0.9)

        # ------------------------------------------------------------------
        # Red limiting‑altitude line
        # ------------------------------------------------------------------
        limit_line_value = self._plot_limit_value()
        limit_line_label = "Limit Airmass" if self._plot_airmass else "Limit Altitude"
        self.limit_line = self.ax_alt.axhline(
            limit_line_value,
            color=self._theme_color("plot_limit", "#ff5d8f"),
            linestyle="-",
            linewidth=0.5,
            alpha=0.4,
            label=limit_line_label,
        )

        # Reset line references
        self.sun_line = None
        self.moon_line = None

        # Sun altitude curve (always plot, visibility controlled)
        if "sun_alt" in data:
            sun_plot_values = self._plot_y_values(data["sun_alt"])
            self.sun_line, = self.ax_alt.plot(
                times, sun_plot_values,
                color=self._theme_color("plot_sun", "orange"), linewidth=1.2, linestyle='-',
                alpha=0.8, label="Sun"
            )
            self.sun_line.set_visible(self.sun_check.isChecked())

        # Moon altitude curve (always plot, visibility controlled)
        if "moon_alt" in data:
            moon_plot_values = self._plot_y_values(data["moon_alt"])
            self.moon_line, = self.ax_alt.plot(
                times, moon_plot_values,
                color=self._theme_color("plot_moon", "silver"), linewidth=1.2, linestyle='-',
                alpha=0.8, label="Moon"
            )
            self.moon_line.set_visible(self.moon_check.isChecked())

        # Update info panel labels in local time
        fmt = "%Y-%m-%d %H:%M"
        if "sunrise" in ev:
            self.sunrise_label.setText(ev["sunrise"].strftime(fmt))
        else:
            self.sunrise_label.setText("-")
        if "sunset" in ev:
            self.sunset_label.setText(ev["sunset"].strftime(fmt))
        else:
            self.sunset_label.setText("-")
        if "moonrise" in ev:
            self.moonrise_label.setText(ev["moonrise"].strftime(fmt))
        else:
            self.moonrise_label.setText("-")
        if "moonset" in ev:
            self.moonset_label.setText(ev["moonset"].strftime(fmt))
        else:
            self.moonset_label.setText("-")
        # Use cached moon_phase percent
        phase_pct = float(data.get("moon_phase", 0.0))
        phase_value = int(max(0, min(100, round(phase_pct))))
        self.moonphase_bar.setValue(phase_value)
        self.moonphase_bar.setFormat(f"{phase_value}%")
        self._configure_main_plot_y_axis()
        self.ax_alt.set_xlabel("Time (local)")
        self.ax_alt.xaxis.label.set_color(plot_text)
        self.ax_alt.yaxis.label.set_color(plot_text)
        self.ax_alt.tick_params(axis="x", colors=plot_text)
        self.ax_alt.tick_params(axis="y", colors=plot_text)
        for spine in self.ax_alt.spines.values():
            spine.set_color(self._qcolor_css(soft_grid_color))
        self.ax_alt.grid(True, color=self._qcolor_rgba_mpl(soft_grid_color, 0.16), alpha=1.0, linestyle="--", linewidth=0.6)
        # self.ax_alt.legend(loc="upper right")
        # Hour labels in the observer's local timezone
        self.ax_alt.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=tz))
        # Display selected observation date
        date_str = self.date_edit.date().toString("yyyy-MM-dd")
        self.ax_alt.set_title(f"Date: {date_str}", color=plot_text)
        # Current time indicator
        now = datetime.now(tz)
        data["now_local"] = now
        self.now_line = self.ax_alt.axvline(
            float(mdates.date2num(now)),
            color=self._theme_color("plot_now", "magenta"),
            linestyle=":",
            linewidth=1.2,
            label="Now",
        )

        # Update time labels
        # Local time
        self.localtime_label.setText(now.strftime("%Y-%m-%d %H:%M:%S"))
        # UTC time (with globe icon)
        now_utc = datetime.now(timezone.utc)
        self.utctime_label.setText(f"{now_utc.strftime('%Y-%m-%d %H:%M:%S')}")
        self._refresh_weather_window_context()

        # Apply default alpha and width based on altitude limit
        for name, line, is_over in self.vis_lines:
            self._apply_visibility_line_style(line, is_over=is_over, is_selected=False)
        # Highlight selected targets over limit
        sel_rows = [i.row() for i in self.table_view.selectionModel().selectedRows()]
        sel_names = [self.targets[i].name for i in sel_rows]
        for name, line, is_over in self.vis_lines:
            self._apply_visibility_line_style(line, is_over=is_over, is_selected=(name in sel_names))
        visible_targets = sum(1 for flag in row_enabled if flag)
        if self._calc_started_at > 0:
            self._last_calc_stats = CalcRunStats(
                duration_s=max(0.0, perf_counter() - self._calc_started_at),
                visible_targets=visible_targets,
                total_targets=len(self.targets),
            )
            self._calc_started_at = 0.0
        self._refresh_cached_bhtom_suggestions()
        self._update_selected_details()
        self._update_status_bar()
        self._reset_plot_navigation_home()
        self.plot_canvas.draw_idle()
        self._render_visibility_web_plot(data)
        self._update_polar_positions(data)
        # Force one selected-path refresh after a full recompute.
        # Selection signatures are debounced in live selection updates.
        self._last_polar_selection_signature = ()
        self._update_polar_selection(None, None)
        if self._queued_plan_run:
            self._queued_plan_run = False
            QTimer.singleShot(0, self._run_plan)

    @Slot()
    def _toggle_visibility(self):
        """Show or hide sun and moon lines without recalculation."""
        if hasattr(self, 'sun_line') and self.sun_line:
            self.sun_line.set_visible(self.sun_check.isChecked())
        if hasattr(self, 'moon_line') and self.moon_line:
            self.moon_line.set_visible(self.moon_check.isChecked())
        plot_canvas = getattr(self, "plot_canvas", None)
        if plot_canvas is not None:
            plot_canvas.draw_idle()
        if isinstance(getattr(self, "last_payload", None), dict):
            self._render_visibility_web_plot(self.last_payload)

    @Slot(bool)
    def _on_plot_mode_switch_changed(self, checked: bool):
        if self._plot_airmass == checked:
            self._refresh_plot_mode_switch()
            return
        self._plot_airmass = checked
        self._refresh_plot_mode_switch()
        self._animate_plot_mode_switch()
        self.settings.setValue("general/plotAirmass", self._plot_airmass)
        if isinstance(self.last_payload, dict):
            if getattr(self, "_use_visibility_web", False):
                self._schedule_visibility_plot_refresh()
            else:
                self._refresh_visibility_matplotlib_mode_only(self.last_payload)


    @Slot()
    def _update_clock(self):
        # Skip if we’re in the middle of shutdown
        if self._shutting_down:
            return
        if self.clock_worker is None and self.table_model.site:
            self._start_clock_worker()


    @Slot(dict)
    def _handle_clock_update(self, data):
        self.localtime_label.setText(data["now_local"].strftime("%Y-%m-%d %H:%M:%S"))
        self.utctime_label.setText(data["now_utc"].strftime("%Y-%m-%d %H:%M:%S"))
        self._refresh_weather_window_context(rebuild=False)
        self.sun_alt_label.setText(f"{data['sun_alt']:.1f}°")
        self.moon_alt_label.setText(f"{data['moon_alt']:.1f}°")
        self.table_model.current_alts = data["alts"]
        self.table_model.current_azs = data["azs"]
        self.table_model.current_seps = data["seps"]
        self.table_model.row_enabled = self._recompute_row_enabled_from_current()
        self._reapply_current_table_sort()
        self._apply_table_row_visibility()
        self._emit_table_data_changed()
        self._update_status_bar()

        # Update sidereal time based on local time and site longitude
        if hasattr(self, 'last_payload') and self.last_payload:
            now = data["now_local"]
            # sidereal time calculation
            from astropy.time import Time
            if self.table_model.site is not None:
                sidereal = Time(data["now_local"]).sidereal_time('apparent', self.table_model.site.to_earthlocation().lon)
                # Format as HH:MM:SS
                self.sidereal_label.setText(sidereal.to_string(unit=u.hour, sep=":", pad=True, precision=0))
            else:
                self.sidereal_label.setText("-")
            if hasattr(self, 'now_line') and self.now_line:
                self.now_line.set_xdata([float(mdates.date2num(now)), float(mdates.date2num(now))])
            else:
                self.now_line = self.ax_alt.axvline(
                    float(mdates.date2num(now)), color="magenta", linestyle=":", linewidth=1.2, label="Now"
                )
            self.plot_canvas.draw_idle()
            current_minute_key = now.strftime("%Y-%m-%d %H:%M")
            if getattr(self, "_visibility_web_minute_key", "") != current_minute_key:
                self._visibility_web_minute_key = current_minute_key
                self.last_payload["now_local"] = now
                self._render_visibility_web_plot(self.last_payload, now_override=now)
            self._update_polar_positions(data, dynamic_only=True)


    @Slot()
    def _update_polar_selection(self, selected, deselected):
        """Update highlight for selected targets on polar plot."""
        # Gather selected rows
        sel_rows = sorted(idx.row() for idx in self.table_view.selectionModel().selectedRows())
        selection_signature = tuple(
            self.targets[row].name
            for row in sel_rows
            if 0 <= row < len(self.targets)
        )
        if selection_signature == getattr(self, "_last_polar_selection_signature", ()):
            return
        self._last_polar_selection_signature = selection_signature
        # Prepare coordinates for selected targets
        sel_coords = []
        for row in sel_rows:
            if row < len(self.table_model.row_enabled) and not self.table_model.row_enabled[row]:
                continue
            alt = self.table_model.current_alts[row] if row < len(self.table_model.current_alts) else None
            az = self.table_model.current_azs[row] if row < len(self.table_model.current_azs) else None
            if alt is None or az is None or alt <= 0:
                continue
            theta = np.deg2rad(az)
            r = 90 - alt
            sel_coords.append((theta, r))
        if sel_coords:
            arr = np.array(sel_coords)
            self.selected_scatter.set_offsets(arr)
        else:
            self.selected_scatter.set_offsets(np.empty((0, 2)))
        # Skip drawing trace if user turned object paths off
        if not self.show_obj_path:
            if self.selected_trace_line:
                try:
                    self.selected_trace_line.remove()
                except Exception:
                    pass
                self.selected_trace_line = None
            self.polar_canvas.draw_idle()
            return
        # Plot the full sky path of the selected target from rise to set
        if self.selected_trace_line:
            try:
                self.selected_trace_line.remove()
            except Exception:
                pass
        self.selected_trace_line = None

        if sel_rows:
            idx0 = sel_rows[0]
            name = self.targets[idx0].name
            if not hasattr(self, "full_payload") or not isinstance(self.full_payload, dict) or name not in self.full_payload:
                self.selected_trace_line = None
                self.polar_canvas.draw_idle()
                return
            alt_arr = np.array(self.full_payload[name]["altitude"])
            az_arr  = np.array(self.full_payload[name]["azimuth"])
            # Only points above horizon
            mask = alt_arr > 0
            vis_idx = np.where(mask)[0]
            if vis_idx.size == 0:
                self.selected_trace_line = None
                self.polar_canvas.draw_idle()
                return
            # Build full theta/r arrays by handling each visible segment separately
            theta_full = np.array([], dtype=float)
            r_full = np.array([], dtype=float)
            # Split into contiguous runs of indices (rise/set segmentation)
            runs = np.split(vis_idx, np.where(np.diff(vis_idx) != 1)[0] + 1)
            for run in runs:
                theta_seg = np.deg2rad(az_arr[run])
                r_seg = 90 - alt_arr[run]
                # Break at azimuth wrap discontinuities
                dtheta = np.abs(np.diff(theta_seg))
                wrap_pts = np.where(dtheta > np.pi)[0] + 1
                for wp in reversed(wrap_pts):
                    theta_seg = np.insert(theta_seg, wp, np.nan)
                    r_seg = np.insert(r_seg, wp, np.nan)
                # Append segment, then a NaN to separate from next
                theta_full = np.concatenate([theta_full, theta_seg, [np.nan]])
                r_full = np.concatenate([r_full, r_seg, [np.nan]])
            trace, = self.polar_ax.plot(
                theta_full, r_full,
                color=self._theme_color("polar_selected_path", "#8cff84"),
                linewidth=0.8,
                linestyle=':',
                alpha=0.7,
                zorder=1,
            )
            self.selected_trace_line = trace
        # Redraw the polar canvas to reflect changes
        self.polar_canvas.draw_idle()

    @Slot(object)
    def _on_polar_pick(self, event):
        """Select table row when a polar scatter point is clicked."""
        if event.artist is not self.polar_scatter:
            return
        inds = event.ind
        if not len(inds):
            return
        ptr = inds[0]
        # Map to the actual target index
        i = self.polar_indices[ptr]
        # Clear previous selection and select the clicked row
        sel_model = self.table_view.selectionModel()
        sel_model.clearSelection()
        idx = self.table_model.index(i, 0)
        sel_model.select(idx, QItemSelectionModel.Select | QItemSelectionModel.Rows)
        # Update selected scatter marker
        alt = self.table_model.current_alts[i]
        az = self.table_model.current_azs[i]
        theta = np.deg2rad(az)
        r = 90 - alt
        self.selected_scatter.set_offsets(np.array([[theta, r]]))
        self.polar_canvas.draw_idle()

    @Slot(object, object)
    def _update_vis_selection(self, selected, deselected):
        """Adjust visibility plot alpha and width based on table selection."""
        sel_rows = [idx.row() for idx in self.table_view.selectionModel().selectedRows()]
        sel_names = {
            self.targets[i].name
            for i in sel_rows
            if 0 <= i < len(self.targets)
        }
        previous_sel_names = set(getattr(self, "_last_vis_selected_names", set()))
        if sel_names == previous_sel_names:
            return
        changed_names = sel_names.symmetric_difference(previous_sel_names)
        self._last_vis_selected_names = set(sel_names)

        for name, line, is_over in self.vis_lines:
            if name not in changed_names:
                continue
            self._apply_visibility_line_style(line, is_over=is_over, is_selected=(name in sel_names))
        plot_canvas = getattr(self, "plot_canvas", None)
        if plot_canvas is not None:
            plot_canvas.draw_idle()
        if (
            bool(getattr(self, "_visibility_web_sync_selection", False))
            and isinstance(getattr(self, "last_payload", None), dict)
        ):
            self._apply_visibility_web_selection_style(sel_names)

    @staticmethod
    def _build_polar_visible_path(
        alt_series: object,
        az_series: object,
    ) -> tuple[np.ndarray, np.ndarray]:
        alt_arr = np.array(alt_series, dtype=float).ravel()
        az_arr = np.array(az_series, dtype=float).ravel()
        n = min(alt_arr.size, az_arr.size)
        if n <= 0:
            return np.array([], dtype=float), np.array([], dtype=float)
        alt_arr = alt_arr[:n]
        az_arr = np.mod(az_arr[:n], 360.0)
        mask = np.isfinite(alt_arr) & np.isfinite(az_arr) & (alt_arr > 0.0)
        vis_idx = np.where(mask)[0]
        if vis_idx.size == 0:
            return np.array([], dtype=float), np.array([], dtype=float)

        theta_segments: list[np.ndarray] = []
        r_segments: list[np.ndarray] = []
        runs = np.split(vis_idx, np.where(np.diff(vis_idx) != 1)[0] + 1)
        for run in runs:
            if run.size == 0:
                continue
            theta_seg = np.deg2rad(az_arr[run])
            r_seg = 90.0 - alt_arr[run]
            wrap_pts = np.where(np.abs(np.diff(theta_seg)) > np.pi)[0] + 1
            for wp in reversed(wrap_pts):
                theta_seg = np.insert(theta_seg, int(wp), np.nan)
                r_seg = np.insert(r_seg, int(wp), np.nan)
            theta_segments.append(theta_seg)
            r_segments.append(r_seg)
            theta_segments.append(np.array([np.nan], dtype=float))
            r_segments.append(np.array([np.nan], dtype=float))

        if not theta_segments:
            return np.array([], dtype=float), np.array([], dtype=float)
        return np.concatenate(theta_segments), np.concatenate(r_segments)

    @Slot(dict)
    def _update_polar_positions(self, data: dict, dynamic_only: bool = False):
        """Update all markers on the polar plot based on latest alt-az data."""
        has_sun_path = "sun_alt" in data and "sun_az" in data
        has_moon_path = "moon_alt" in data and "moon_az" in data
        full_refresh = not dynamic_only and (has_sun_path or has_moon_path)

        if "alts" in data and "azs" in data:
            alt_list = data["alts"]
            az_list = data["azs"]
        else:
            alt_list = self.table_model.current_alts
            az_list = self.table_model.current_azs

        tgt_coords = []
        self.polar_indices = []
        for i in range(len(self.targets)):
            if i >= len(alt_list) or i >= len(az_list):
                continue
            if i < len(self.table_model.row_enabled) and not self.table_model.row_enabled[i]:
                continue
            alt = alt_list[i]
            az = az_list[i]
            if alt is None or alt <= 0:
                continue
            tgt_coords.append((np.deg2rad(az), 90 - alt))
            self.polar_indices.append(i)
        self._radar_target_coords = np.array(tgt_coords, dtype=float) if tgt_coords else np.empty((0, 2), dtype=float)
        self.polar_scatter.set_offsets(self._radar_target_coords if tgt_coords else np.empty((0, 2)))

        sel_coords = []
        for row in self._selected_rows():
            if row >= len(alt_list) or row >= len(az_list):
                continue
            alt = alt_list[row]
            az = az_list[row]
            if alt is None or alt <= 0:
                continue
            sel_coords.append((np.deg2rad(az), 90 - alt))
        self.selected_scatter.set_offsets(np.array(sel_coords) if sel_coords else np.empty((0, 2)))

        site = self.table_model.site
        if site is None:
            self.sun_marker.set_offsets(np.empty((0, 2)))
            self.moon_marker.set_offsets(np.empty((0, 2)))
            self._refresh_radar_sweep_artists(redraw=False)
            self.polar_canvas.draw_idle()
            return

        now_local = data.get("now_local")
        if now_local is not None:
            eph_obs = ephem.Observer()
            eph_obs.lat = str(site.latitude)
            eph_obs.lon = str(site.longitude)
            eph_obs.elevation = site.elevation
            eph_obs.date = now_local

            sun = ephem.Sun(eph_obs)
            sun_alt = sun.alt * 180.0 / math.pi  # type: ignore[arg-type]
            sun_az = sun.az * 180.0 / math.pi  # type: ignore[arg-type]
            if sun_alt > 0:
                self.sun_marker.set_offsets(np.array([[np.deg2rad(sun_az), 90 - sun_alt]]))
            else:
                self.sun_marker.set_offsets(np.empty((0, 2)))

            moon = ephem.Moon(eph_obs)
            moon_alt = moon.alt * 180.0 / math.pi  # type: ignore[arg-type]
            moon_az = moon.az * 180.0 / math.pi  # type: ignore[arg-type]
            if moon_alt > 0:
                self.moon_marker.set_offsets(np.array([[np.deg2rad(moon_az), 90 - moon_alt]]))
            else:
                self.moon_marker.set_offsets(np.empty((0, 2)))

        if full_refresh:
            for line_attr in ("sun_path_line", "moon_path_line"):
                line = getattr(self, line_attr, None)
                if line is None:
                    continue
                try:
                    line.remove()
                except Exception:
                    pass
                setattr(self, line_attr, None)

            if self.show_sun_path and has_sun_path:
                theta, r = self._build_polar_visible_path(data["sun_alt"], data["sun_az"])
                if theta.size > 0 and r.size > 0:
                    self.sun_path_line, = self.polar_ax.plot(
                        theta,
                        r,
                        color=self._theme_color("polar_sun", "gold"),
                        linewidth=0.9,
                        linestyle="--",
                        alpha=0.7,
                        zorder=1,
                    )

            if self.show_moon_path and has_moon_path:
                theta, r = self._build_polar_visible_path(data["moon_alt"], data["moon_az"])
                if theta.size > 0 and r.size > 0:
                    self.moon_path_line, = self.polar_ax.plot(
                        theta,
                        r,
                        color=self._theme_color("polar_moon", "silver"),
                        linewidth=0.9,
                        linestyle="--",
                        alpha=0.7,
                        zorder=1,
                    )

            signature = (round(site.latitude, 6), int(self.limit_spin.value()))
            if getattr(self, "_polar_static_signature", None) != signature:
                if self.pole_marker:
                    try:
                        if isinstance(self.pole_marker, (list, tuple)):
                            for art in self.pole_marker:
                                art.remove()
                        else:
                            self.pole_marker.remove()
                    except Exception:
                        pass
                pole_alt = site.latitude if site.latitude >= 0 else -site.latitude
                pole_az = 0.0 if site.latitude >= 0 else 180.0
                r_pol = 90 - pole_alt
                theta_pol = np.deg2rad(pole_az)
                circle = self.polar_ax.scatter(
                    [theta_pol], [r_pol],
                    facecolors='none',
                    edgecolors=self._theme_color("polar_pole", "purple"),
                    marker='o',
                    s=80,
                    linewidths=1.5,
                    zorder=3,
                    alpha=0.3,
                )
                dot = self.polar_ax.scatter(
                    [theta_pol],
                    [r_pol],
                    c=self._theme_color("polar_pole", "purple"),
                    marker='.',
                    s=30,
                    zorder=4,
                    alpha=0.3,
                )
                self.pole_marker = (circle, dot)

                if self.limit_circle:
                    try:
                        self.limit_circle.remove()
                    except Exception:
                        pass
                lim = self.limit_spin.value()
                theta_full = np.linspace(0, 2 * math.pi, 200)
                r_full = np.full_like(theta_full, 90 - lim)
                self.limit_circle, = self.polar_ax.plot(
                    theta_full,
                    r_full,
                    color=self._theme_color("polar_limit", "#ff5d8f"),
                    linestyle='-',
                    linewidth=0.5,
                    alpha=0.4,
                )
                self._polar_static_signature = signature

        self._clock_polar_tick += 1
        self._refresh_radar_sweep_artists(redraw=False)
        self.polar_canvas.draw_idle()

    @Slot()
    def _toggle_dark(self):
        """Toggle dark mode while preserving the base stylesheet."""
        self._set_dark_mode_enabled(not self._dark_enabled, persist=True)

    @Slot()
    def _export_plan(self):
        """Write targets JSON, plot PNG, session CSV, and calendar ICS."""
        out_dir = QFileDialog.getExistingDirectory(
            self, "Select export directory", str(Path.cwd())
        )
        if not out_dir:
            return
        out_path = Path(out_dir)
        # JSON
        with open(out_path / "plan_targets.json", "w", encoding="utf-8") as fh:
            json.dump([t.model_dump() for t in self.targets], fh, indent=2)
        # PNG
        self._ensure_visibility_plot_widgets()
        if self.plot_canvas is not None:
            self.plot_canvas.figure.savefig(out_path / "plan_plot.png", dpi=150)
        # CSV summary and ICS window schedule
        tz_name = "UTC"
        if isinstance(getattr(self, "last_payload", None), dict):
            tz_name = str(self.last_payload.get("tz", "UTC"))
        elif self.table_model.site:
            tz_name = self.table_model.site.timezone_name

        rows: list[dict[str, object]] = []
        ics_events: list[dict[str, object]] = []
        for tgt in self.targets:
            metrics = self.target_metrics.get(tgt.name)
            window = self.target_windows.get(tgt.name)
            start_txt = window[0].strftime("%Y-%m-%d %H:%M") if window else ""
            end_txt = window[1].strftime("%Y-%m-%d %H:%M") if window else ""
            rows.append({
                "name": tgt.name,
                "ra_deg": round(tgt.ra, 6),
                "dec_deg": round(tgt.dec, 6),
                "priority": tgt.priority,
                "observed": tgt.observed,
                "score": metrics.score if metrics else 0.0,
                "hours_above_limit": metrics.hours_above_limit if metrics else 0.0,
                "max_altitude_deg": metrics.max_altitude_deg if metrics else 0.0,
                "peak_moon_sep_deg": metrics.peak_moon_sep_deg if metrics else 0.0,
                "window_start_local": start_txt,
                "window_end_local": end_txt,
                "notes": tgt.notes,
            })
            if window:
                score_txt = f"{metrics.score:.1f}" if metrics else "0.0"
                ics_events.append({
                    "title": f"Observe {tgt.name}",
                    "start": window[0],
                    "end": window[1],
                    "description": f"Target score {score_txt}, priority {tgt.priority}.",
                })

        export_metrics_csv(out_path / "plan_summary.csv", rows)
        export_observation_ics(out_path / "plan_schedule.ics", tz_name=tz_name, events=ics_events)
        QMessageBox.information(self, "Export complete", f"Wrote files to {out_path}")

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
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
        worker = MetadataLookupWorker(self._meta_request_id, names, self)
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

    def _build_ai_panel(self, parent: Optional[QWidget] = None) -> QWidget:
        host = parent if parent is not None else self
        panel = QFrame(host)
        panel.setObjectName("AIAssistantPanel")
        _set_dynamic_property(panel, "accented", True)

        layout = QHBoxLayout(panel)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        btn_col = QVBoxLayout()
        btn_col.setSpacing(6)

        describe_btn = QPushButton("Describe Object")
        describe_btn.clicked.connect(self._ai_describe_target)
        btn_col.addWidget(describe_btn)
        _set_button_variant(describe_btn, "secondary")
        _set_button_icon_kind(describe_btn, "describe")
        describe_btn.ensurePolished()

        describe_hint = QLabel("Uses local metadata only.\nNo LLM request.", panel)
        describe_hint.setObjectName("SectionHint")
        describe_hint.setWordWrap(True)
        btn_col.addWidget(describe_hint)

        self.ai_copy_last_btn = QPushButton("Copy Last Reply")
        self.ai_copy_last_btn.clicked.connect(self._copy_last_ai_response)
        _set_button_variant(self.ai_copy_last_btn, "ghost")
        _set_button_icon_kind(self.ai_copy_last_btn, "describe")
        btn_col.addWidget(self.ai_copy_last_btn)

        self.ai_export_chat_btn = QPushButton("Export Chat")
        self.ai_export_chat_btn.clicked.connect(self._export_ai_chat)
        _set_button_variant(self.ai_export_chat_btn, "ghost")
        _set_button_icon_kind(self.ai_export_chat_btn, "save")
        btn_col.addWidget(self.ai_export_chat_btn)

        self.ai_reset_appearance_btn = QPushButton("Reset Appearance")
        self.ai_reset_appearance_btn.clicked.connect(self._reset_ai_chat_appearance)
        _set_button_variant(self.ai_reset_appearance_btn, "ghost")
        _set_button_icon_kind(self.ai_reset_appearance_btn, "refresh")
        btn_col.addWidget(self.ai_reset_appearance_btn)

        self.ai_settings_btn = QPushButton("AI Settings")
        self.ai_settings_btn.clicked.connect(self._open_ai_settings)
        _set_button_variant(self.ai_settings_btn, "ghost")
        _set_button_icon_kind(self.ai_settings_btn, "edit")
        btn_col.addWidget(self.ai_settings_btn)

        self.ai_context_hint = QLabel("", panel)
        self.ai_context_hint.setObjectName("SectionHint")
        self.ai_context_hint.setWordWrap(True)
        btn_col.addWidget(self.ai_context_hint)

        self.ai_knowledge_hint = QLabel("Using local knowledge: none", panel)
        self.ai_knowledge_hint.setObjectName("SectionHint")
        self.ai_knowledge_hint.setWordWrap(True)
        self.ai_knowledge_hint.setToolTip(
            "Shows which local knowledge notes were added to the most recent AI prompt."
        )
        btn_col.addWidget(self.ai_knowledge_hint)

        warmup_btn = QPushButton("Warm Up LLM", panel)
        warmup_btn.clicked.connect(self._warmup_llm_manual)
        _set_button_variant(warmup_btn, "ghost")
        _set_button_icon_kind(warmup_btn, "refresh")
        warmup_btn.setToolTip("Send a lightweight request to warm the current LLM model.")
        btn_col.addWidget(warmup_btn)

        clear_btn = QPushButton("Clear", panel)
        clear_btn.clicked.connect(self._clear_ai_messages)
        _set_button_variant(clear_btn, "ghost")
        _set_button_icon_kind(clear_btn, "clear")
        btn_col.addWidget(clear_btn)

        status_row = QWidget(panel)
        status_row_l = QHBoxLayout(status_row)
        status_row_l.setContentsMargins(0, 0, 0, 0)
        status_row_l.setSpacing(0)

        self.ai_warm_badge_label = QLabel(f"● cold {self.llm_config.model}", status_row)
        self.ai_warm_badge_label.setObjectName("SectionHint")
        self.ai_warm_badge_label.setToolTip(self.llm_config.model)
        status_row_l.addWidget(self.ai_warm_badge_label, 1)
        _set_label_tone(self.ai_warm_badge_label, "muted")
        btn_col.addWidget(status_row)

        btn_col.addStretch(1)
        btn_widget = QWidget(panel)
        btn_widget.setLayout(btn_col)
        describe_width = max(
            210,
            describe_btn.sizeHint().width() + 24,
            self.ai_copy_last_btn.sizeHint().width() + 24,
            self.ai_export_chat_btn.sizeHint().width() + 24,
            self.ai_reset_appearance_btn.sizeHint().width() + 24,
            self.ai_settings_btn.sizeHint().width() + 24,
            warmup_btn.sizeHint().width() + 24,
            clear_btn.sizeHint().width() + 24,
            244,
        )
        describe_btn.setMinimumWidth(describe_width - 12)
        describe_hint.setFixedWidth(describe_width - 12)
        self.ai_copy_last_btn.setMinimumWidth(describe_width - 12)
        self.ai_export_chat_btn.setMinimumWidth(describe_width - 12)
        self.ai_reset_appearance_btn.setMinimumWidth(describe_width - 12)
        self.ai_settings_btn.setMinimumWidth(describe_width - 12)
        self.ai_context_hint.setFixedWidth(describe_width - 12)
        self.ai_knowledge_hint.setFixedWidth(describe_width - 12)
        warmup_btn.setMinimumWidth(describe_width - 12)
        clear_btn.setMinimumWidth(describe_width - 12)
        btn_widget.setFixedWidth(describe_width)
        layout.addWidget(btn_widget)

        self._ai_output_placeholder_text = (
            "AI responses will appear here.\n"
            "Configure Ollama or another local LLM in Settings -> General Settings."
        )
        self.ai_output = QScrollArea(panel)
        self.ai_output.setWidgetResizable(True)
        self.ai_output.setFrameShape(QFrame.NoFrame)
        self.ai_output.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.ai_output.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.ai_output.setObjectName("AIChatScroll")
        self.ai_output.installEventFilter(self)
        self._ai_output_last_viewport_width = 0
        self.ai_output_content = QWidget(self.ai_output)
        self.ai_output_content.setObjectName("AIChatContent")
        self.ai_output_layout = QVBoxLayout(self.ai_output_content)
        self.ai_output_layout.setContentsMargins(12, 12, 12, 12)
        self.ai_output_layout.setSpacing(0)
        self.ai_output.setWidget(self.ai_output_content)
        self._apply_ai_chat_font()
        center_widget = QWidget(panel)
        center_col = QVBoxLayout(center_widget)
        center_col.setContentsMargins(0, 0, 0, 0)
        center_col.setSpacing(8)
        center_col.addWidget(self.ai_output, 1)

        composer_row = QHBoxLayout()
        composer_row.setContentsMargins(0, 0, 0, 0)
        composer_row.setSpacing(8)

        self.ai_input = QLineEdit(center_widget)
        self.ai_input.setPlaceholderText("Ask about tonight or the selected object...")
        self.ai_input.returnPressed.connect(self._send_ai_query)
        composer_row.addWidget(self.ai_input, 1)

        send_btn = QPushButton("Send", center_widget)
        send_btn.clicked.connect(self._send_ai_query)
        _set_button_variant(send_btn, "primary")
        _set_button_icon_kind(send_btn, "send")
        send_btn.setToolTip(
            "Send the current question to the local LLM. Object-specific questions are auto-routed to a lighter selected-target prompt."
        )
        send_btn.setMinimumWidth(max(132, send_btn.sizeHint().width() + 16))
        composer_row.addWidget(send_btn)
        center_col.addLayout(composer_row)
        layout.addWidget(center_widget, 1)
        self._refresh_ai_context_hint()
        self._refresh_ai_knowledge_hint()
        self._refresh_ai_panel_action_buttons()
        self._refresh_ai_warm_indicator()

        return panel

    def _build_ai_window(self) -> QDialog:
        window = QDialog(self)
        window.setObjectName("AIAssistantWindow")
        window.setWindowTitle("AI Assistant")
        window.setModal(False)
        _fit_dialog_to_screen(
            window,
            preferred_width=1180,
            preferred_height=620,
            min_width=860,
            min_height=360,
        )

        window_l = QVBoxLayout(window)
        window_l.setContentsMargins(8, 8, 8, 8)
        window_l.setSpacing(6)
        panel = self._build_ai_panel(window)
        window_l.addWidget(panel, 1)
        window.finished.connect(self._on_ai_window_finished)
        localize_widget_tree(window, current_language())
        return window

    def _ensure_ai_window(self) -> QDialog:
        ai_window = getattr(self, "ai_window", None)
        if isinstance(ai_window, QDialog):
            return ai_window
        ai_window = self._build_ai_window()
        ai_window.setStyleSheet(self.styleSheet())
        self.ai_window = ai_window
        if getattr(self, "_ai_messages", None):
            self._render_ai_messages()
        self._refresh_ai_panel_action_buttons()
        self._refresh_ai_context_hint()
        self._refresh_ai_knowledge_hint()
        self._refresh_ai_warm_indicator()
        return ai_window

    @Slot(bool)
    def _toggle_ai_panel(self, checked: bool) -> None:
        ai_window = self._ensure_ai_window() if checked else getattr(self, "ai_window", None)
        if isinstance(ai_window, QDialog):
            if checked:
                ai_window.show()
                ai_window.raise_()
                ai_window.activateWindow()
                self._warmup_llm_if_needed()
            else:
                ai_window.hide()
        if hasattr(self, "ai_toggle_btn"):
            set_translated_text(self.ai_toggle_btn, "Hide AI" if checked else "AI", current_language())

    @Slot(int)
    def _on_ai_window_finished(self, _result: int) -> None:
        if not hasattr(self, "ai_toggle_btn"):
            return
        if not self.ai_toggle_btn.isChecked():
            return
        blocker = QSignalBlocker(self.ai_toggle_btn)
        self.ai_toggle_btn.setChecked(False)
        del blocker
        set_translated_text(self.ai_toggle_btn, "AI", current_language())

    def _llm_warmup_cache_key(self) -> tuple[str, str]:
        return (
            str(getattr(self.llm_config, "url", "") or "").strip().rstrip("/"),
            str(getattr(self.llm_config, "model", "") or "").strip(),
        )

    def _llm_is_warm(self) -> bool:
        cache_key = self._llm_warmup_cache_key()
        if self._llm_last_warmup_key != cache_key:
            return False
        return (perf_counter() - float(self._llm_last_warmup_at)) < 600.0

    def _start_llm_warmup(
        self,
        *,
        force: bool = False,
        user_initiated: bool = False,
        silent: bool = False,
    ) -> bool:
        worker = self._llm_worker
        if worker is not None and worker.isRunning():
            if user_initiated:
                self._set_ai_status("AI assistant is busy.", tone="warning")
            return False
        if not force and self._llm_is_warm():
            if not silent:
                self._set_ai_status("Ready", tone="info")
            return False
        self._llm_warmup_silent = bool(silent)
        self._llm_active_requested_class = ""
        self._llm_active_primary_target = None
        if not silent:
            self._set_ai_status("Warming up LLM...", tone="info")
        worker = LLMWorker(
            config=self.llm_config,
            prompt="Reply with OK.",
            system_prompt="Reply with exactly OK.",
            tag="warmup",
            parent=self,
        )
        worker.responseChunk.connect(self._on_ai_chunk)
        worker.responseReady.connect(self._on_ai_response)
        worker.errorOccurred.connect(self._on_ai_error)
        worker.finished.connect(self._on_ai_worker_finished)
        self._llm_worker = worker
        self._llm_active_tag = "warmup"
        self._refresh_ai_warm_indicator()
        worker.start()
        return True

    @Slot()
    def _warmup_llm_if_needed(self) -> None:
        self._start_llm_warmup(force=False, user_initiated=False, silent=False)

    @Slot()
    def _warmup_llm_manual(self) -> None:
        self._start_llm_warmup(force=True, user_initiated=True, silent=False)

    @Slot()
    def _warmup_llm_on_startup(self) -> None:
        if self._llm_startup_warmup_attempted or getattr(self, "_shutting_down", False):
            return
        self._llm_startup_warmup_attempted = True
        self._start_llm_warmup(force=False, user_initiated=False, silent=True)

    def _refresh_ai_warm_indicator(self) -> None:
        configured_model = str(getattr(self.llm_config, "model", "") or "").strip() or LLMConfig.DEFAULT_MODEL
        is_warm = self._llm_is_warm()
        active_tag = str(getattr(self, "_llm_active_tag", "") or "").strip().lower()
        runtime_status = str(getattr(self, "_ai_runtime_status", "") or "").strip().lower()
        model_text = configured_model
        if is_warm and self._llm_last_warmup_key[1]:
            model_text = self._llm_last_warmup_key[1]

        badge_text = f"● cold {model_text}"
        badge_tone = "muted"
        badge_glow_color = self._theme_qcolor("section_hint", "#8fa3b8")

        if runtime_status == "warm-up failed":
            badge_text = f"! warm-up failed {model_text}"
            badge_tone = "warning"
            badge_glow_color = self._theme_qcolor("state_warning", "#ffcc66")
        elif active_tag == "warmup":
            badge_text = f"◌ warming {model_text}"
            badge_tone = "info"
            badge_glow_color = self._theme_qcolor("state_info", "#59f3ff")
        elif active_tag and runtime_status.startswith("stream"):
            badge_text = f"✦ streaming {model_text}"
            badge_tone = "info"
            badge_glow_color = self._theme_qcolor("state_info", "#59f3ff")
        elif active_tag:
            badge_text = f"✦ thinking {model_text}"
            badge_tone = "info"
            badge_glow_color = self._theme_qcolor("state_info", "#59f3ff")
        elif is_warm:
            badge_text = f"● warm {model_text}"
            badge_tone = "success"
            badge_glow_color = self._theme_qcolor("state_success", "#67ff9a")

        if hasattr(self, "ai_warm_badge_label"):
            set_translated_text(self.ai_warm_badge_label, badge_text, current_language())
            set_translated_tooltip(self.ai_warm_badge_label, badge_text, current_language())
            _set_label_tone(self.ai_warm_badge_label, badge_tone)
            self._apply_label_glow_effect(self.ai_warm_badge_label, badge_glow_color, blur_radius=22.0)

    def _set_ai_status(self, text: str, *, tone: str = "info") -> None:
        self._ai_runtime_status = str(text)
        self._ai_runtime_status_tone = str(tone)
        if tone in {"warning", "error"} and self._ai_status_error_clear_s > 0:
            self._ai_status_clear_timer.start(int(self._ai_status_error_clear_s * 1000))
        else:
            self._ai_status_clear_timer.stop()
        self._refresh_ai_warm_indicator()

    @Slot()
    def _clear_ai_runtime_status(self) -> None:
        if str(getattr(self, "_llm_active_tag", "") or "").strip():
            return
        self._ai_runtime_status = ""
        self._ai_runtime_status_tone = "info"
        self._refresh_ai_warm_indicator()

    @staticmethod
    def _apply_label_glow_effect(label: Optional[QLabel], color: QColor, *, blur_radius: float) -> None:
        if label is None:
            return
        current = label.graphicsEffect()
        if not isinstance(current, QGraphicsDropShadowEffect):
            effect = QGraphicsDropShadowEffect(label)
            effect.setOffset(0.0, 0.0)
            label.setGraphicsEffect(effect)
        else:
            effect = current
        glow_color = QColor(color)
        if not glow_color.isValid():
            glow_color = QColor("#59f3ff")
        if glow_color.alpha() < 210:
            glow_color.setAlpha(210)
        effect.setBlurRadius(float(blur_radius))
        effect.setColor(glow_color)

    def _current_site_for_weather(self) -> Optional[Site]:
        site = getattr(self.table_model, "site", None) if hasattr(self, "table_model") else None
        if isinstance(site, Site):
            return site
        if hasattr(self, "obs_combo") and hasattr(self, "observatories"):
            selected = self.obs_combo.currentText()
            maybe = self.observatories.get(selected)
            if isinstance(maybe, Site):
                return maybe
        return None

    def _refresh_weather_window_context(self, *, rebuild: bool = True) -> None:
        window = getattr(self, "weather_window", None)
        if not isinstance(window, WeatherDialog):
            return
        site = self._current_site_for_weather()
        obs_name = site.name if isinstance(site, Site) else (
            self.obs_combo.currentText() if hasattr(self, "obs_combo") else "-"
        )
        date = self.date_edit.date() if hasattr(self, "date_edit") else QDate.currentDate()
        local_time_text = self.localtime_label.text() if hasattr(self, "localtime_label") else "-"
        utc_time_text = self.utctime_label.text() if hasattr(self, "utctime_label") else "-"
        sunrise_text = self.sunrise_label.text() if hasattr(self, "sunrise_label") else "-"
        sunset_text = self.sunset_label.text() if hasattr(self, "sunset_label") else "-"
        moonrise_text = self.moonrise_label.text() if hasattr(self, "moonrise_label") else "-"
        moonset_text = self.moonset_label.text() if hasattr(self, "moonset_label") else "-"
        moon_phase_pct = int(self.moonphase_bar.value()) if hasattr(self, "moonphase_bar") else 0
        window.set_context(
            site=site,
            obs_name=obs_name,
            date=date,
            sun_alt_limit=self._sun_alt_limit(),
            local_time_text=local_time_text,
            utc_time_text=utc_time_text,
            sunrise_text=sunrise_text,
            sunset_text=sunset_text,
            moonrise_text=moonrise_text,
            moonset_text=moonset_text,
            moon_phase_pct=moon_phase_pct,
            rebuild=rebuild,
        )

    @Slot()
    def _open_weather_window(self) -> None:
        window = getattr(self, "weather_window", None)
        if not isinstance(window, WeatherDialog):
            window = WeatherDialog(self)
            window.setStyleSheet(self.styleSheet())
            self.weather_window = window
        self._refresh_weather_window_context()
        window.show()
        window.raise_()
        window.activateWindow()

    def _selected_target_row_index(self) -> Optional[int]:
        rows = self._selected_rows() if hasattr(self, "table_view") else []
        if rows and 0 <= rows[0] < len(self.targets):
            return int(rows[0])
        return None

    def _build_llm_target_summary_line(
        self,
        row_index: int,
        target: Target,
        *,
        include_current_snapshot: bool = True,
    ) -> str:
        self._ensure_known_target_type(target)
        details: list[str] = []
        order_values = getattr(self.table_model, "order_values", [])
        if row_index < len(order_values):
            order_value = _safe_int(order_values[row_index])
            if isinstance(order_value, int) and order_value > 0:
                details.append(f"order {order_value}")
        details.append(f"pri {target.priority}")
        if target.observed:
            details.append("observed")
        if target.object_type and not _object_type_is_unknown(target.object_type):
            details.append(f"type {target.object_type}")
        class_family = self._target_class_family(target)
        if class_family:
            details.append(f"family {class_family}")

        metrics = self.target_metrics.get(target.name)
        if metrics is not None:
            details.extend(
                [
                    f"score {metrics.score:.1f}",
                    f"best {self._format_target_best_window_compact(target) or 'none'}",
                    f"over {metrics.hours_above_limit:.1f} h",
                    f"max alt {metrics.max_altitude_deg:.0f} deg",
                ]
            )
        else:
            details.append("visibility not calculated")

        if include_current_snapshot:
            if row_index < len(self.table_model.current_alts):
                alt_now = self.table_model.current_alts[row_index]
                if math.isfinite(alt_now):
                    details.append(f"now alt {alt_now:.1f} deg")
            if row_index < len(self.table_model.current_seps):
                moon_sep_now = self.table_model.current_seps[row_index]
                if math.isfinite(moon_sep_now):
                    details.append(f"now moon sep {moon_sep_now:.1f} deg")

        return f"  - {target.name}: " + ", ".join(details)

    def _session_context_target_indices(self, *, max_items: int = 8) -> tuple[list[int], int]:
        if not self.targets:
            return [], 0

        row_enabled = list(getattr(self.table_model, "row_enabled", []))
        visible_indices = [idx for idx, enabled in enumerate(row_enabled) if enabled] if row_enabled else []
        candidate_indices = visible_indices or list(range(len(self.targets)))
        selected_row = self._selected_target_row_index()

        def _sort_key(idx: int) -> tuple[object, ...]:
            order_values = getattr(self.table_model, "order_values", [])
            raw_order = order_values[idx] if idx < len(order_values) else 0
            order_value = _safe_int(raw_order) or 0
            metrics = self.target_metrics.get(self.targets[idx].name)
            score = float(metrics.score) if metrics is not None else -1.0
            hours_above = float(metrics.hours_above_limit) if metrics is not None else -1.0
            return (
                0 if order_value > 0 else 1,
                order_value if order_value > 0 else 10**9,
                -score,
                -hours_above,
                _normalize_catalog_display_name(self.targets[idx].name).lower(),
            )

        ranked = sorted(candidate_indices, key=_sort_key)
        summary_indices: list[int] = []
        for idx in ranked:
            if idx == selected_row:
                continue
            summary_indices.append(idx)
            if len(summary_indices) >= max_items:
                break
        omitted_count = max(0, len(candidate_indices) - len(summary_indices))
        return summary_indices, omitted_count

    def _build_session_context(
        self,
        *,
        include_current_snapshot: bool = True,
        user_question: str = "",
    ) -> str:
        parts: list[str] = []
        date_str = self.date_edit.date().toString("yyyy-MM-dd")
        parts.append(f"Observation date: {date_str}")
        site = self.table_model.site
        if site:
            parts.append(
                f"Site: {site.name} (lat {site.latitude:.3f}, lon {site.longitude:.3f}, "
                f"elev {site.elevation:.0f} m, timezone {site.timezone_name})"
            )
        parts.append(f"Altitude limit: {self.limit_spin.value()} deg")
        parts.append(f"Sun altitude threshold: {self._sun_alt_limit():.0f} deg")
        if hasattr(self, "min_moon_sep_spin"):
            parts.append(f"Moon separation threshold: {float(self.min_moon_sep_spin.value()):.0f} deg")
        if hasattr(self, "min_score_spin"):
            parts.append(f"Score threshold: {float(self.min_score_spin.value()):.1f}")

        payload = self.last_payload if isinstance(self.last_payload, dict) else None
        tz_name = site.timezone_name if site else "UTC"
        try:
            tz = pytz.timezone(tz_name)
        except Exception:
            tz = pytz.UTC
        now_local = None
        if payload:
            tz_name = str(payload.get("tz", tz_name or "UTC"))
            try:
                tz = pytz.timezone(tz_name)
            except Exception:
                tz = pytz.UTC
            now_local = payload.get("now_local")

            if isinstance(now_local, datetime):
                try:
                    now_local = now_local.astimezone(tz)
                except Exception:
                    pass

            def _fmt_event(key: str) -> str:
                raw = payload.get(key)
                if raw is None:
                    return "N/A"
                try:
                    dt = mdates.num2date(float(raw)).astimezone(tz)
                    return dt.strftime("%Y-%m-%d %H:%M")
                except Exception:
                    return "N/A"

            if isinstance(now_local, datetime):
                parts.append(f"Current local time: {now_local.strftime('%Y-%m-%d %H:%M:%S')}")
            parts.append(
                "Night events: "
                f"sunset {_fmt_event('sunset')}, sunrise {_fmt_event('sunrise')}, "
                f"astronomical night starts {_fmt_event('dusk')}, "
                f"astronomical night ends {_fmt_event('dawn')}"
            )
            moon_phase = payload.get("moon_phase")
            if moon_phase is not None:
                try:
                    parts.append(f"Moon phase: {float(moon_phase):.1f}%")
                except Exception:
                    pass

        if not self.targets:
            parts.append("Targets in plan: none")
            suggestion_context = self._build_bhtom_suggestion_shortlist_context(
                user_question=user_question,
                max_items=5,
            )
            if suggestion_context:
                parts.append(suggestion_context)
            return "\n".join(parts)

        visible_count = sum(bool(enabled) for enabled in getattr(self.table_model, "row_enabled", []))
        if visible_count > 0:
            parts.append(f"Targets in plan: {len(self.targets)} total, {visible_count} visible under current filters")
        else:
            parts.append(f"Targets in plan: {len(self.targets)}")

        class_query = self._parse_class_query_spec(user_question)
        requested_marker = class_query.requested_class if class_query is not None else ""
        selected_row = self._selected_target_row_index()
        if selected_row is not None and 0 <= selected_row < len(self.targets):
            selected_target = self.targets[selected_row]
            self._ensure_known_target_type(selected_target)
            if not requested_marker or _type_matches_requested_class(selected_target.object_type, requested_marker):
                parts.append(
                    "Selected target:\n"
                    + self._build_llm_target_summary_line(
                        selected_row,
                        selected_target,
                        include_current_snapshot=include_current_snapshot,
                    )
                )

        summary_indices, omitted_count = self._session_context_target_indices(
            max_items=8,
        )
        if requested_marker:
            filtered_indices: list[int] = []
            for idx in summary_indices:
                self._ensure_known_target_type(self.targets[idx])
                if _type_matches_requested_class(self.targets[idx].object_type, requested_marker):
                    filtered_indices.append(idx)
            omitted_count += max(0, len(summary_indices) - len(filtered_indices))
            summary_indices = filtered_indices
        if summary_indices:
            rows = [
                self._build_llm_target_summary_line(
                    idx,
                    self.targets[idx],
                    include_current_snapshot=include_current_snapshot,
                )
                for idx in summary_indices
            ]
            parts.append("Target shortlist:\n" + "\n".join(rows))
            if omitted_count > 0:
                parts.append(f"Additional visible targets omitted from prompt: {omitted_count}")

        suggestion_context = self._build_bhtom_suggestion_shortlist_context(
            user_question=user_question,
            max_items=5,
        )
        if suggestion_context:
            parts.append(suggestion_context)
        return "\n".join(parts)

    def _build_bhtom_suggestion_shortlist_context(
        self,
        *,
        user_question: str = "",
        max_items: int = 5,
    ) -> str:
        suggestions = list(getattr(self, "_bhtom_ranked_suggestions_cache", []) or [])
        if not suggestions:
            return ""

        type_markers = list(_question_bhtom_type_markers(user_question))
        class_query = self._parse_class_query_spec(user_question)
        if class_query is not None:
            marker = _normalize_knowledge_tag(class_query.requested_class)
            if marker and marker not in type_markers:
                type_markers.append(marker)
        shortlist = suggestions
        shortlist_label = "Cached BHTOM suggestion shortlist (not yet in plan)"
        max_rows = max_items

        if type_markers:
            filtered: list[dict[str, object]] = []
            for item in suggestions:
                target = item.get("target")
                if not isinstance(target, Target):
                    continue
                haystack = " ".join(
                    [
                        _normalize_catalog_display_name(target.name).lower(),
                        _normalize_catalog_display_name(target.source_object_id).lower(),
                        _normalize_catalog_display_name(target.object_type).lower(),
                    ]
                )
                if any(marker in haystack for marker in type_markers):
                    filtered.append(item)
            if filtered:
                shortlist = filtered
                max_rows = max(max_items, 10)
                shortlist_label = (
                    "Cached BHTOM suggestion shortlist matching this question "
                    "(not yet in plan)"
                )

        rows: list[str] = []
        for item in shortlist:
            target = item.get("target")
            metrics = item.get("metrics")
            window_start = item.get("window_start")
            window_end = item.get("window_end")
            if not isinstance(target, Target) or not isinstance(metrics, TargetNightMetrics):
                continue
            if not isinstance(window_start, datetime) or not isinstance(window_end, datetime):
                continue

            details: list[str] = []
            if target.object_type and not _object_type_is_unknown(target.object_type):
                details.append(f"type {target.object_type}")
            importance = _safe_float(item.get("importance"))
            if importance is not None and math.isfinite(importance):
                details.append(f"importance {importance:.1f}")
            if target.magnitude is not None and math.isfinite(float(target.magnitude)):
                details.append(f"{_target_magnitude_label(target).lower()} {float(target.magnitude):.2f}")
            details.append(f"score {metrics.score:.1f}")
            details.append(f"best {window_start.strftime('%H:%M')}-{window_end.strftime('%H:%M')}")
            best_airmass = _safe_float(item.get("best_airmass"))
            if best_airmass is not None and math.isfinite(best_airmass):
                details.append(f"min airmass {best_airmass:.2f}")
            rows.append(f"  - {target.name}: " + ", ".join(details))
            if len(rows) >= max_rows:
                break

        if not rows:
            return ""
        omitted = max(0, len(shortlist) - len(rows))
        summary = shortlist_label
        if omitted > 0:
            summary = f"{summary} (+{omitted} more)"
        return summary + ":\n" + "\n".join(rows)

    def _target_class_family(self, target: Target) -> str:
        self._ensure_known_target_type(target)
        return _type_label_class_family(target.object_type)

    def _find_referenced_target_in_question(self, text: str) -> Optional[Target]:
        raw_text = str(text or "").strip()
        if not raw_text:
            return None

        candidates: list[tuple[int, Target]] = [(0, target) for target in self.targets]
        suggestions = list(getattr(self, "_bhtom_ranked_suggestions_cache", []) or [])
        for item in suggestions:
            target = item.get("target")
            if isinstance(target, Target):
                candidates.append((1, target))

        matched: list[tuple[int, int, int, Target]] = []
        seen: set[str] = set()
        for source_rank, target in candidates:
            dedupe_key = _normalize_catalog_token(target.source_object_id or target.name)
            if not dedupe_key or dedupe_key in seen:
                continue
            positions: list[tuple[int, int]] = []
            for candidate_name in {target.name.strip(), str(target.source_object_id or "").strip()}:
                if not candidate_name:
                    continue
                pos = self._find_normalized_text_position(raw_text, candidate_name)
                if pos is None:
                    continue
                match_len = len(re.sub(r"[^a-z0-9]+", "", _normalize_catalog_display_name(candidate_name).lower()))
                positions.append((pos, match_len))
            if not positions:
                continue
            seen.add(dedupe_key)
            best_pos, best_len = min(positions, key=lambda item: (item[0], -item[1]))
            matched.append((best_pos, -best_len, source_rank, target))

        if not matched:
            return None
        matched.sort(key=lambda item: (item[0], item[1], item[2], item[3].name.lower()))
        return matched[0][3]

    def _find_referenced_targets_in_question(self, text: str, *, max_targets: int = 6) -> list[Target]:
        raw_text = str(text or "").strip()
        if not raw_text:
            return []

        candidates: list[tuple[int, Target]] = [(0, target) for target in self.targets]
        suggestions = list(getattr(self, "_bhtom_ranked_suggestions_cache", []) or [])
        for item in suggestions:
            target = item.get("target")
            if isinstance(target, Target):
                candidates.append((1, target))

        matched: list[tuple[int, int, int, str, Target]] = []
        seen: set[str] = set()
        for source_rank, target in candidates:
            dedupe_key = _normalize_catalog_token(target.source_object_id or target.name)
            if not dedupe_key or dedupe_key in seen:
                continue
            positions: list[tuple[int, int]] = []
            for candidate_name in {target.name.strip(), str(target.source_object_id or "").strip()}:
                if not candidate_name:
                    continue
                pos = self._find_normalized_text_position(raw_text, candidate_name)
                if pos is None:
                    continue
                match_len = len(re.sub(r"[^a-z0-9]+", "", _normalize_catalog_display_name(candidate_name).lower()))
                positions.append((pos, match_len))
            if not positions:
                continue
            seen.add(dedupe_key)
            best_pos, best_len = min(positions, key=lambda item: (item[0], -item[1]))
            matched.append((best_pos, -best_len, source_rank, target.name.lower(), target))

        matched.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
        return [item[4] for item in matched[: max(1, int(max_targets))]]

    def _plan_row_index_for_target(self, target: Target) -> Optional[int]:
        for idx, existing in enumerate(self.targets):
            if _targets_match(existing, target):
                return idx
        return None

    def _lookup_target_observing_candidate(self, target: Target) -> Optional[dict[str, object]]:
        row_index = self._plan_row_index_for_target(target)
        if row_index is not None and 0 <= row_index < len(self.targets):
            plan_target = self.targets[row_index]
            self._ensure_known_target_type(plan_target)
            metrics = self.target_metrics.get(plan_target.name)
            if metrics is not None:
                current_alt = None
                if row_index < len(self.table_model.current_alts):
                    alt_now = self.table_model.current_alts[row_index]
                    if math.isfinite(alt_now):
                        current_alt = float(alt_now)
                moon_sep = None
                if row_index < len(self.table_model.current_seps):
                    sep_now = self.table_model.current_seps[row_index]
                    if math.isfinite(sep_now):
                        moon_sep = float(sep_now)
                return {
                    "target": plan_target,
                    "metrics": metrics,
                    "best_window": self._format_target_best_window_compact(plan_target),
                    "current_alt": current_alt,
                    "moon_sep": moon_sep,
                    "source": "plan",
                }

        suggestions = list(getattr(self, "_bhtom_ranked_suggestions_cache", []) or [])
        for item in suggestions:
            suggestion_target = item.get("target")
            metrics = item.get("metrics")
            if not isinstance(suggestion_target, Target) or not isinstance(metrics, TargetNightMetrics):
                continue
            if not _targets_match(suggestion_target, target):
                continue
            self._ensure_known_target_type(suggestion_target)
            best_window = ""
            window_start = item.get("window_start")
            window_end = item.get("window_end")
            if isinstance(window_start, datetime) and isinstance(window_end, datetime):
                best_window = f"{window_start.strftime('%H:%M')}-{window_end.strftime('%H:%M')}"
            return {
                "target": suggestion_target,
                "metrics": metrics,
                "best_window": best_window,
                "current_alt": None,
                "moon_sep": _safe_float(item.get("min_window_moon_sep")),
                "source": "bhtom",
                "best_airmass": _safe_float(item.get("best_airmass")),
                "importance": _safe_float(item.get("importance")),
            }

        self._ensure_known_target_type(target)
        metrics = self.target_metrics.get(target.name)
        if metrics is None:
            return None
        return {
            "target": target,
            "metrics": metrics,
            "best_window": self._format_target_best_window_compact(target),
            "current_alt": None,
            "moon_sep": None,
            "source": _normalize_catalog_token(target.source_catalog) or "target",
        }

    def _load_knowledge_notes(self) -> list[KnowledgeNote]:
        cached = getattr(self, "_knowledge_notes_cache", None)
        if cached is not None:
            return cached

        notes: list[KnowledgeNote] = []
        try:
            paths = sorted(
                path
                for path in KNOWLEDGE_DIR.rglob("*.md")
                if path.is_file()
                and "_templates" not in path.parts
                and path.name != "_index.md"
            )
        except Exception:
            paths = []

        for path in paths:
            note = _load_knowledge_note(path)
            if note is not None:
                notes.append(note)
        self._knowledge_notes_cache = notes
        return notes

    def _select_knowledge_notes(
        self,
        *,
        question: str,
        target: Optional[Target] = None,
        max_notes: int = 3,
        max_chars: int = 1600,
    ) -> list[KnowledgeNote]:
        notes = self._load_knowledge_notes()
        if not notes:
            return []

        request_tags = self._knowledge_request_tags(question, target=target)
        if not request_tags:
            return []

        requested_family = _requested_marker_family(_requested_object_class_marker(question))
        if not requested_family and target is not None:
            requested_family = self._target_class_family(target)

        if target is None and not requested_family:
            global_note_tags = {
                "bhtom",
                "last-mag-vs-mag",
                "simbad",
                "tns",
                "gaia-alerts",
                "best-window",
                "moonlight",
                "choosing-between-similar-targets",
                "small-scope-practicality",
            }
            if not (request_tags & global_note_tags):
                return []

        ranked: list[tuple[int, KnowledgeNote]] = []
        for note in notes:
            note_family = _knowledge_note_family(note)
            if requested_family and "object-classes" in note.path.parts:
                if note_family != requested_family:
                    continue
            score = self._knowledge_note_score(note, request_tags=request_tags, question=question, target=target)
            if score <= 0:
                continue
            ranked.append((score, note))
        if not ranked:
            return []

        ranked.sort(key=lambda item: (-item[0], item[1].path.as_posix()))
        selected: list[KnowledgeNote] = []
        total_chars = 0
        for _, note in ranked[:max_notes]:
            snippet = _format_knowledge_note_snippet(note)
            next_total = total_chars + len(snippet) + (2 if selected else 0)
            if selected and next_total > max_chars:
                break
            selected.append(note)
            total_chars = next_total
        return selected

    def _knowledge_request_tags(self, question: str, target: Optional[Target] = None) -> set[str]:
        tags: set[str] = set()
        for marker in _question_bhtom_type_markers(question):
            normalized = _normalize_knowledge_tag(marker)
            if normalized:
                tags.add(normalized)

        requested_marker = _requested_object_class_marker(question)
        if requested_marker:
            tags.add(_normalize_knowledge_tag(requested_marker))

        normalized_question = _normalize_catalog_display_name(question).lower()
        keyword_map = {
            "moonlight": (
                "moon", "moonlight", "moon sep", "moon separation", "ksiezyc", "księżyc",
                "lun", "bright moon",
            ),
            "bhtom": ("bhtom", "importance", "last mag", "last magnitude"),
            "last-mag-vs-mag": ("last mag", "last magnitude", "catalog mag", "mag vs last mag"),
            "simbad": ("simbad",),
            "tns": ("tns", "transient name server"),
            "gaia-alerts": ("gaia alert", "gaia alerts", "gaia-alerts"),
            "best-window": (
                "best window", "window", "over limit", "score", "order", "kolejn", "airmass",
                "altitude", "wysok", "horizon", "horyzont", "when observe", "kiedy obserw",
            ),
            "practical-observing": ("observe", "obserw", "how should i", "jak powinienem"),
            "choosing-between-similar-targets": (
                "which is best", "which one is best", "ktory najlepiej", "która najlepiej",
                "porown", "porówn", "compare", "comparison", "vs", "better target",
            ),
            "small-scope-practicality": (
                "small scope", "small telescope", "small setup", "seestar",
                "easy target", "practical", "realistically", "ma sens",
            ),
        }
        for tag, markers in keyword_map.items():
            if any(marker in normalized_question for marker in markers):
                tags.add(tag)

        if target is not None:
            self._ensure_known_target_type(target)
            family = self._target_class_family(target)
            if family == "AGN":
                tags.add("agn")
                type_norm = _normalize_catalog_display_name(target.object_type).lower()
                if "qso" in type_norm or "quasar" in type_norm:
                    tags.add("qso")
            elif family == "Supernova":
                tags.update({"supernova", "sn"})
            elif family == "Nova":
                tags.add("nova")
            elif family == "Cataclysmic variable":
                tags.add("cv")
            elif family == "Galaxy":
                tags.add("galaxy")
            elif family == "Star":
                tags.add("star")
                type_norm = _normalize_catalog_display_name(target.object_type).lower()
                if "variable" in type_norm:
                    tags.add("variable-star")
            if "cluster" in _normalize_catalog_display_name(target.object_type).lower():
                tags.add("open-cluster")
            source_tag = _normalize_knowledge_tag(target.source_catalog)
            if source_tag:
                tags.add(source_tag)
            if _normalize_catalog_token(target.source_catalog) == "bhtom":
                tags.add("bhtom")

        return {tag for tag in tags if tag}

    def _knowledge_note_score(
        self,
        note: KnowledgeNote,
        *,
        request_tags: set[str],
        question: str,
        target: Optional[Target],
    ) -> int:
        score = 0
        note_tags = set(note.tags)
        overlap = request_tags & note_tags
        score += len(overlap) * 4

        path_tokens = {
            _normalize_knowledge_tag(note.path.stem),
            _normalize_knowledge_tag(note.path.parent.name),
        }
        score += len(request_tags & path_tokens) * 3

        normalized_question = _normalize_catalog_display_name(question).lower()
        if note.summary and any(tag.replace("-", " ") in normalized_question for tag in note_tags):
            score += 2

        if target is not None:
            if "bhtom" in note_tags and _normalize_catalog_token(target.source_catalog) == "bhtom":
                score += 2
            if "agn" in note_tags and self._target_class_family(target) == "AGN":
                score += 2
            if {"supernova", "sn"} & note_tags and self._target_class_family(target) == "Supernova":
                score += 2

        return score

    def _build_knowledge_context(
        self,
        *,
        question: str,
        target: Optional[Target] = None,
        max_notes: int = 3,
        max_chars: int = 1600,
    ) -> str:
        notes = self._select_knowledge_notes(
            question=question,
            target=target,
            max_notes=max_notes,
            max_chars=max_chars,
        )
        if not notes:
            return ""
        snippets = [_format_knowledge_note_snippet(note) for note in notes]
        return "Local knowledge notes:\n" + "\n".join(snippets)

    def _build_local_object_fact_answer(self, question: str, *, target: Optional[Target] = None) -> Optional[str]:
        target = target or self._resolve_object_query_target(question, selected_target=self._selected_target_or_none())
        if target is None:
            return None

        type_label = self._ensure_known_target_type(target).strip()
        family = self._target_class_family(target)
        if not type_label and not family:
            return None

        normalized = _normalize_catalog_display_name(question).lower()
        requested_marker = _requested_object_class_marker(question)
        class_label_map = {
            "agn": "AGN",
            "qso": "QSO",
            "seyfert": "Seyfert",
            "blazar": "blazar",
            "supernova": "supernova",
            "nova": "nova",
            "xrb": "X-ray binary",
            "cv": "cataclysmic variable",
            "galaxy": "galaxy",
            "star": "star",
        }
        if requested_marker and _looks_like_object_class_query(normalized):
            matches = _type_matches_requested_class(type_label, requested_marker)
            class_label = class_label_map.get(requested_marker, requested_marker)
            if matches:
                detail = type_label or family or class_label
                answer = f"Yes. {target.name} is classified as {detail}."
                if family and family != detail:
                    answer += f" Broad class: {family}."
                return answer
            detail = type_label or family or "unknown"
            answer = f"No. Current metadata classifies {target.name} as {detail}."
            if family and family != detail:
                answer += f" Broad class: {family}."
            return answer

        type_markers = (
            "what type",
            "type of object",
            "what is",
            "what kind",
            "jaki typ",
            "jaki to typ",
            "jaki to obiekt",
            "co to za obiekt",
            "czym jest",
            "co to jest",
        )
        if any(marker in normalized for marker in type_markers):
            detail = type_label or family
            if not detail:
                return None
            answer = f"{target.name} is classified as {detail}."
            if family and family != detail:
                answer += f" Broad class: {family}."
            return answer

        return None

    def _parse_class_query_spec(self, question: str) -> Optional[ClassQuerySpec]:
        normalized = _normalize_catalog_display_name(question).lower()
        if not normalized:
            return None

        explicit_requested_class = _requested_object_class_marker(normalized)
        filter_flags = _question_action_flags(
            normalized,
            {
                "bhtom_only": (
                    "bhtom only",
                    "only bhtom",
                    "only from bhtom",
                    "from bhtom only",
                    "tylko z bhtom",
                    "tylko bhtom",
                ),
                "exclude_observed": (
                    "exclude observed",
                    "without observed",
                    "hide observed",
                    "not observed",
                    "unobserved",
                    "nieobserw",
                    "nie obserw",
                    "bez obserw",
                ),
                "prefer_brighter": (
                    "brighter",
                    "brightest",
                    "jaśniejs",
                    "jasniejs",
                    "jaśniejsze",
                    "jasniejsze",
                    "najjaś",
                    "najas",
                ),
            },
        )
        flags = _question_action_flags(
            normalized,
            {
                "observe": (
                    "which",
                    "best",
                    "recommend",
                    "observe",
                    "obserw",
                    "moge",
                    "mogę",
                    "today",
                    "tonight",
                    "dzis",
                    "dziś",
                    "któr",
                    "jaki",
                ),
                "list": (
                    "other",
                    "others",
                    "more",
                    "list",
                    "show",
                    "give me",
                    "top ",
                    "jakie",
                    "jakie sa",
                    "jakie są",
                    "inne",
                    "wymien",
                    "wymień",
                    "pokaz",
                    "pokaż",
                    "lista",
                    "podaj",
                ),
                "more": (
                    "other",
                    "others",
                    "more",
                    "another",
                    "inne",
                    "kolejne",
                    "więcej",
                    "wiecej",
                    "next",
                ),
                "choice": (
                    "which one",
                    "which is best",
                    "which is better",
                    "ktory",
                    "która",
                    "ktora",
                    "który",
                ),
            },
        )
        has_action_marker = bool(flags["observe"])
        wants_list = bool(flags["list"]) or bool(re.search(r"\b([2-9]|10)\b", normalized))
        wants_more = bool(flags["more"])
        choice_followup = bool(flags["choice"])
        prefer_bhtom_only = bool(filter_flags["bhtom_only"])
        exclude_observed = bool(filter_flags["exclude_observed"])
        prefer_brighter = bool(filter_flags["prefer_brighter"])
        semantic_followup = prefer_bhtom_only or exclude_observed or prefer_brighter
        requested_class = explicit_requested_class
        if not requested_class and (wants_list or wants_more or choice_followup or semantic_followup):
            requested_class = str(self._recent_ai_conversation_state().get("requested_class", "") or "").strip()

        if not requested_class:
            return None
        if not explicit_requested_class and not (wants_list or wants_more or choice_followup or semantic_followup):
            return None
        if explicit_requested_class and not (
            has_action_marker or wants_list or wants_more or choice_followup or semantic_followup
        ):
            return None

        if semantic_followup and not wants_list and not choice_followup:
            wants_list = True
        count = self._class_query_requested_count(question, default_count=5 if wants_list else 3)
        return ClassQuerySpec(
            requested_class=requested_class,
            count=count,
            wants_list=wants_list,
            wants_more=wants_more,
            choice_followup=choice_followup,
            prefer_bhtom_only=prefer_bhtom_only,
            exclude_observed=exclude_observed,
            exclude_previous_results=wants_more,
            prefer_brighter=prefer_brighter,
        )

    def _looks_like_class_observing_query(self, text: str) -> bool:
        return self._parse_class_query_spec(text) is not None

    def _collect_class_observing_candidates(self, class_query: ClassQuerySpec) -> list[dict[str, object]]:
        marker = _normalize_knowledge_tag(class_query.requested_class)
        if not marker:
            return []

        candidates: list[dict[str, object]] = []
        seen: set[str] = set()
        for row_index, target in enumerate(self.targets):
            self._ensure_known_target_type(target)
            if not _type_matches_requested_class(target.object_type, marker):
                continue
            if class_query.prefer_bhtom_only and _normalize_catalog_token(target.source_catalog) != "bhtom":
                continue
            if class_query.exclude_observed and bool(target.observed):
                continue
            metrics = self.target_metrics.get(target.name)
            if metrics is None:
                continue
            current_alt = None
            if row_index < len(self.table_model.current_alts):
                alt_now = self.table_model.current_alts[row_index]
                if math.isfinite(alt_now):
                    current_alt = float(alt_now)
            moon_sep = None
            if row_index < len(self.table_model.current_seps):
                sep_now = self.table_model.current_seps[row_index]
                if math.isfinite(sep_now):
                    moon_sep = float(sep_now)
            dedupe_key = _normalize_catalog_token(target.source_object_id or target.name)
            if not dedupe_key or dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            candidates.append(
                {
                    "target": target,
                    "metrics": metrics,
                    "best_window": self._format_target_best_window_compact(target),
                    "current_alt": current_alt,
                    "moon_sep": moon_sep,
                    "source": "plan",
                }
            )

        suggestions = list(getattr(self, "_bhtom_ranked_suggestions_cache", []) or [])
        for item in suggestions:
            target = item.get("target")
            metrics = item.get("metrics")
            if not isinstance(target, Target) or not isinstance(metrics, TargetNightMetrics):
                continue
            self._ensure_known_target_type(target)
            if not _type_matches_requested_class(target.object_type, marker):
                continue
            if class_query.exclude_observed and bool(target.observed):
                continue
            dedupe_key = _normalize_catalog_token(target.source_object_id or target.name)
            if not dedupe_key or dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            best_window = ""
            window_start = item.get("window_start")
            window_end = item.get("window_end")
            if isinstance(window_start, datetime) and isinstance(window_end, datetime):
                best_window = f"{window_start.strftime('%H:%M')}-{window_end.strftime('%H:%M')}"
            candidates.append(
                {
                    "target": target,
                    "metrics": metrics,
                    "best_window": best_window,
                    "current_alt": None,
                    "moon_sep": _safe_float(item.get("min_window_moon_sep")),
                    "source": "bhtom",
                }
            )
        return candidates

    def _recent_ai_conversation_state(self, *, max_messages: int = 8) -> dict[str, Any]:
        requested_class = ""
        primary_target: Optional[Target] = None
        suggested_targets: list[Target] = []

        considered = 0
        for message in reversed(self._ai_messages):
            kind = str(message.get("kind", "") or "").strip().lower()
            if kind not in {"user", "ai"}:
                continue
            considered += 1
            if not requested_class:
                requested_class = str(message.get("requested_class", "") or "").strip()
            if primary_target is None:
                candidate_target = message.get("primary_target")
                if isinstance(candidate_target, Target):
                    primary_target = candidate_target
            if not suggested_targets:
                candidate_targets = message.get("suggested_targets")
                if isinstance(candidate_targets, list):
                    suggested_targets = [target for target in candidate_targets if isinstance(target, Target)]
            if requested_class and primary_target is not None and suggested_targets:
                break
            if considered >= max_messages:
                break

        if primary_target is None and suggested_targets:
            primary_target = suggested_targets[0]

        return {
            "requested_class": requested_class,
            "primary_target": primary_target,
            "suggested_targets": suggested_targets,
        }

    def _resolve_recent_class_marker(self, *, max_messages: int = 6) -> str:
        recent_state = self._recent_ai_conversation_state(max_messages=max_messages)
        requested_class = str(recent_state.get("requested_class", "") or "").strip()
        if requested_class:
            return requested_class

        recent_user_texts: list[str] = []
        for message in reversed(self._ai_messages):
            kind = str(message.get("kind", "") or "").strip().lower()
            if kind != "user":
                continue
            text = str(message.get("text", "") or "").strip()
            if not text:
                continue
            recent_user_texts.append(text)
            if len(recent_user_texts) >= max_messages:
                break

        for text in recent_user_texts:
            marker = _requested_object_class_marker(text)
            if marker:
                return marker
        return ""

    @staticmethod
    def _class_query_requested_count(question: str, *, default_count: int = 3, max_count: int = 10) -> int:
        normalized = _normalize_catalog_display_name(question).lower()
        if not normalized:
            return default_count

        digit_match = re.search(r"\b([1-9]|10)\b", normalized)
        if digit_match:
            return max(1, min(max_count, int(digit_match.group(1))))

        word_map = {
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10,
            "jeden": 1,
            "dwa": 2,
            "trzy": 3,
            "cztery": 4,
            "piec": 5,
            "pięć": 5,
            "szesc": 6,
            "sześć": 6,
            "siedem": 7,
            "osiem": 8,
            "dziewiec": 9,
            "dziewięć": 9,
            "dziesiec": 10,
            "dziesięć": 10,
        }
        for token, value in word_map.items():
            if re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", normalized):
                return max(1, min(max_count, value))
        return default_count

    def _build_local_class_observing_response(self, question: str) -> Optional[dict[str, Any]]:
        class_query = self._parse_class_query_spec(question)
        if class_query is None:
            return None

        recent_state = self._recent_ai_conversation_state()
        requested_marker = class_query.requested_class
        if not requested_marker:
            return None

        candidates = self._collect_class_observing_candidates(class_query)
        label = _requested_marker_label(requested_marker)
        is_polish = any(
            token in _normalize_catalog_display_name(question).lower()
            for token in ("jak ", "obserw", "dzis", "dziś", "któr", "jaki", "mogę", "moge")
        )
        if not candidates:
            if is_polish:
                return {
                    "text": f"Brak pasujących celów klasy {label} w planie ani w cache BHTOM.",
                    "requested_class": requested_marker,
                    "primary_target": None,
                    "suggested_targets": [],
                }
            return {
                "text": f"No matching {label} targets are available in the plan or cached BHTOM suggestions.",
                "requested_class": requested_marker,
                "primary_target": None,
                "suggested_targets": [],
            }

        wants_list = bool(class_query.wants_list)
        wants_more = bool(class_query.wants_more)
        choice_followup = bool(class_query.choice_followup)
        requested_count = int(class_query.count)
        excluded_targets: list[Target] = []
        if class_query.exclude_previous_results:
            recent_suggested = list(recent_state.get("suggested_targets", []) or [])
            for recent_target in recent_suggested:
                if not isinstance(recent_target, Target):
                    continue
                self._ensure_known_target_type(recent_target)
                if _type_matches_requested_class(recent_target.object_type, requested_marker):
                    excluded_targets.append(recent_target)
            if not excluded_targets:
                recent_target = recent_state.get("primary_target")
                if isinstance(recent_target, Target):
                    self._ensure_known_target_type(recent_target)
                    if _type_matches_requested_class(recent_target.object_type, requested_marker):
                        excluded_targets.append(recent_target)

        def _sort_key(item: dict[str, object]) -> tuple[object, ...]:
            metrics = item.get("metrics")
            score = float(metrics.score) if isinstance(metrics, TargetNightMetrics) else -1.0
            hours_above = float(metrics.hours_above_limit) if isinstance(metrics, TargetNightMetrics) else -1.0
            current_alt = _safe_float(item.get("current_alt"))
            current_alt_sort = float(current_alt) if current_alt is not None else -1.0
            target = item.get("target")
            magnitude = float(target.magnitude) if isinstance(target, Target) and target.magnitude is not None and math.isfinite(float(target.magnitude)) else float("inf")
            source_rank = 0 if str(item.get("source") or "") == "plan" else 1
            name = target.name if isinstance(target, Target) else ""
            if class_query.prefer_brighter:
                return (magnitude, -score, -hours_above, -current_alt_sort, source_rank, name.lower())
            return (-score, -hours_above, -current_alt_sort, magnitude, source_rank, name.lower())

        candidates.sort(key=_sort_key)
        if choice_followup:
            recent_keys = {
                _normalize_catalog_token(target.source_object_id or target.name)
                for target in list(recent_state.get("suggested_targets", []) or [])
                if isinstance(target, Target)
            }
            if recent_keys:
                filtered_candidates = []
                for item in candidates:
                    item_target = item.get("target")
                    if not isinstance(item_target, Target):
                        continue
                    item_key = _normalize_catalog_token(item_target.source_object_id or item_target.name)
                    if item_key and item_key in recent_keys:
                        filtered_candidates.append(item)
                if filtered_candidates:
                    candidates = filtered_candidates
        if excluded_targets:
            excluded_keys = {
                _normalize_catalog_token(target.source_object_id or target.name)
                for target in excluded_targets
                if isinstance(target, Target)
            }
            filtered_candidates = []
            for item in candidates:
                item_target = item.get("target")
                if not isinstance(item_target, Target):
                    continue
                item_key = _normalize_catalog_token(item_target.source_object_id or item_target.name)
                if item_key and item_key in excluded_keys:
                    continue
                filtered_candidates.append(item)
            if filtered_candidates:
                candidates = filtered_candidates
            elif wants_more:
                if is_polish:
                    return {
                        "text": f"Nie mam już więcej celów klasy {label} poza tymi, które już pokazałem.",
                        "requested_class": requested_marker,
                        "primary_target": excluded_targets[0] if excluded_targets else None,
                        "suggested_targets": list(excluded_targets),
                    }
                return {
                    "text": f"No more {label} targets are available beyond the ones already shown.",
                    "requested_class": requested_marker,
                    "primary_target": excluded_targets[0] if excluded_targets else None,
                    "suggested_targets": list(excluded_targets),
                }

        top = candidates[0]
        target = top.get("target")
        metrics = top.get("metrics")
        if not isinstance(target, Target) or not isinstance(metrics, TargetNightMetrics):
            return None

        displayed_targets: list[Target] = []
        if wants_list:
            rows: list[str] = []
            for item in candidates[: max(1, requested_count)]:
                item_target = item.get("target")
                item_metrics = item.get("metrics")
                if not isinstance(item_target, Target) or not isinstance(item_metrics, TargetNightMetrics):
                    continue
                displayed_targets.append(item_target)
                item_window = str(item.get("best_window") or "").strip()
                item_alt = _safe_float(item.get("current_alt"))
                item_source = str(item.get("source") or "plan")
                parts = [f"score {item_metrics.score:.1f}"]
                if item_window:
                    parts.append(f"best {item_window}")
                if item_alt is not None and math.isfinite(item_alt):
                    parts.append(f"now alt {item_alt:.1f}°")
                if item_source == "bhtom":
                    parts.append("BHTOM")
                rows.append(f"- {item_target.name}: " + ", ".join(parts))
            if is_polish:
                return {
                    "text": f"{label} dostępne teraz:\n" + "\n".join(rows),
                    "requested_class": requested_marker,
                    "primary_target": displayed_targets[0] if displayed_targets else target,
                    "suggested_targets": displayed_targets,
                }
            return {
                "text": f"{label} options now:\n" + "\n".join(rows),
                "requested_class": requested_marker,
                "primary_target": displayed_targets[0] if displayed_targets else target,
                "suggested_targets": displayed_targets,
            }

        best_window = str(top.get("best_window") or "").strip()
        current_alt = _safe_float(top.get("current_alt"))
        source = str(top.get("source") or "plan")
        details = [f"score {metrics.score:.1f}"]
        if best_window:
            details.append(f"best {best_window}")
        details.append(f"max alt {metrics.max_altitude_deg:.0f}°")
        if current_alt is not None and math.isfinite(current_alt):
            details.append(f"now alt {current_alt:.1f}°")
        if source == "bhtom":
            details.append("from BHTOM cache")

        backups: list[str] = []
        displayed_targets.append(target)
        for item in candidates[1:3]:
            backup_target = item.get("target")
            backup_metrics = item.get("metrics")
            if not isinstance(backup_target, Target) or not isinstance(backup_metrics, TargetNightMetrics):
                continue
            backups.append(f"{backup_target.name} ({backup_metrics.score:.1f})")
            displayed_targets.append(backup_target)

        if is_polish:
            answer = f"Najlepszy {label} teraz: {target.name} — {', '.join(details)}."
            if backups:
                answer += " Rezerwa: " + ", ".join(backups) + "."
            return {
                "text": answer,
                "requested_class": requested_marker,
                "primary_target": target,
                "suggested_targets": displayed_targets,
            }

        answer = f"Best {label} now: {target.name} — {', '.join(details)}."
        if backups:
            answer += " Backups: " + ", ".join(backups) + "."
        return {
            "text": answer,
            "requested_class": requested_marker,
            "primary_target": target,
            "suggested_targets": displayed_targets,
        }

    def _build_local_class_observing_answer(self, question: str) -> Optional[str]:
        response = self._build_local_class_observing_response(question)
        if not response:
            return None
        text = str(response.get("text", "") or "").strip()
        return text or None

    def _build_local_object_observing_answer(self, question: str, *, target: Optional[Target] = None) -> Optional[str]:
        if not _looks_like_observing_guidance_query(question):
            return None

        target = target or self._resolve_object_query_target(question, selected_target=self._selected_target_or_none())
        if target is None:
            return None

        self._ensure_known_target_type(target)
        family = self._target_class_family(target)
        metrics = self.target_metrics.get(target.name)
        best_window = self._format_target_best_window_compact(target)

        row_index: Optional[int] = None
        for idx, existing in enumerate(self.targets):
            if _targets_match(existing, target):
                row_index = idx
                break

        moon_sep_now: Optional[float] = None
        if row_index is not None and row_index < len(self.table_model.current_seps):
            candidate = self.table_model.current_seps[row_index]
            if math.isfinite(candidate):
                moon_sep_now = float(candidate)

        moon_sep_threshold = float(self.min_moon_sep_spin.value()) if hasattr(self, "min_moon_sep_spin") else 0.0
        magnitude_label = _target_magnitude_label(target)
        magnitude_text = (
            f"{magnitude_label} {float(target.magnitude):.2f}"
            if target.magnitude is not None and math.isfinite(float(target.magnitude))
            else ""
        )

        is_polish = any(
            token in _normalize_catalog_display_name(question).lower()
            for token in ("jak ", "obserw", "dzis", "dziś", "księ", "ksie", "powinienem")
        )

        if is_polish:
            lines: list[str] = []
            lead = f"{target.name}: "
            lead_parts: list[str] = []
            if best_window:
                lead_parts.append(f"najlepsze okno {best_window}")
            if metrics is not None:
                lead_parts.append(f"max alt {metrics.max_altitude_deg:.0f}°")
                lead_parts.append(f"ponad limit {metrics.hours_above_limit:.1f} h")
            if magnitude_text:
                lead_parts.append(magnitude_text)
            if lead_parts:
                lines.append(lead + ", ".join(lead_parts) + ".")
            else:
                lines.append(f"{target.name}: obserwuj go w najwyższym segmencie dostępnego okna.")

            if family in {"Supernova", "Nova", "Cataclysmic variable", "AGN"}:
                lines.append("To kompaktowy cel punktowy: priorytet to niski airmass, stabilne prowadzenie i dłuższa ciągła integracja.")
            elif family == "Galaxy":
                lines.append("To cel rozmyty: ważniejsze są ciemniejsze niebo i najwyższy fragment okna niż sam score.")
            else:
                lines.append("Priorytetem jest najwyższy fragment okna i możliwie niski airmass, a nie krótkie przeskoki między celami.")

            if moon_sep_now is not None and moon_sep_threshold > 0:
                if moon_sep_now >= moon_sep_threshold:
                    lines.append(f"Separacja od Księżyca {moon_sep_now:.1f}° jest bezpieczna względem progu {moon_sep_threshold:.0f}°.")
                else:
                    lines.append(f"Separacja od Księżyca {moon_sep_now:.1f}° jest poniżej progu {moon_sep_threshold:.0f}°; kontrast może być gorszy.")

            return "\n".join(f"- {line}" for line in lines[:3])

        lines_en: list[str] = []
        lead_parts_en: list[str] = []
        if best_window:
            lead_parts_en.append(f"best window {best_window}")
        if metrics is not None:
            lead_parts_en.append(f"max alt {metrics.max_altitude_deg:.0f}°")
            lead_parts_en.append(f"over limit {metrics.hours_above_limit:.1f} h")
        if magnitude_text:
            lead_parts_en.append(magnitude_text)
        if lead_parts_en:
            lines_en.append(f"{target.name}: " + ", ".join(lead_parts_en) + ".")
        else:
            lines_en.append(f"{target.name}: observe it during the highest-altitude segment of the available window.")

        if family in {"Supernova", "Nova", "Cataclysmic variable", "AGN"}:
            lines_en.append("Treat it as a compact point source: prioritize low airmass, stable tracking, and longer continuous integration.")
        elif family == "Galaxy":
            lines_en.append("Treat it as a diffuse target: dark sky and the highest clean segment matter more than score alone.")
        else:
            lines_en.append("Prioritize the highest segment of the window and lower airmass rather than hopping between short looks.")

        if moon_sep_now is not None and moon_sep_threshold > 0:
            if moon_sep_now >= moon_sep_threshold:
                lines_en.append(f"Moon separation {moon_sep_now:.1f}° is safely above the {moon_sep_threshold:.0f}° threshold.")
            else:
                lines_en.append(f"Moon separation {moon_sep_now:.1f}° is below the {moon_sep_threshold:.0f}° threshold, so contrast may suffer.")

        return "\n".join(f"- {line}" for line in lines_en[:3])

    @staticmethod
    def _find_normalized_text_position(text: str, candidate: str) -> Optional[int]:
        normalized_text = _normalize_catalog_display_name(text).lower()
        normalized_candidate = _normalize_catalog_display_name(candidate).lower()
        if not normalized_text or not normalized_candidate:
            return None
        direct_pos = normalized_text.find(normalized_candidate)
        if direct_pos >= 0:
            return direct_pos
        compact_text = re.sub(r"[^a-z0-9]+", "", normalized_text)
        compact_candidate = re.sub(r"[^a-z0-9]+", "", normalized_candidate)
        if len(compact_candidate) < 5:
            return None
        compact_pos = compact_text.find(compact_candidate)
        if compact_pos >= 0:
            return compact_pos
        return None

    def _extract_addable_bhtom_targets_from_ai_text(self, text: str, *, max_items: int = 4) -> list[Target]:
        raw_text = str(text or "").strip()
        if not raw_text:
            return []
        suggestions = list(getattr(self, "_bhtom_ranked_suggestions_cache", []) or [])
        if not suggestions:
            return []

        matched: list[tuple[int, int, Target]] = []
        seen: set[str] = set()
        for rank, item in enumerate(suggestions):
            target = item.get("target")
            if not isinstance(target, Target):
                continue
            if self._plan_contains_target(target):
                continue
            positions: list[int] = []
            for candidate_name in {target.name.strip(), str(target.source_object_id or "").strip()}:
                if not candidate_name:
                    continue
                pos = self._find_normalized_text_position(raw_text, candidate_name)
                if pos is not None:
                    positions.append(pos)
            if not positions:
                continue
            dedupe_key = _normalize_catalog_token(target.source_object_id or target.name)
            if not dedupe_key or dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            matched.append((min(positions), rank, target))

        matched.sort(key=lambda item: (item[0], item[1], item[2].name.lower()))
        return [target for _, _, target in matched[: max(1, int(max_items))]]

    def _resolve_recent_chat_target_reference(self, *, max_messages: int = 6) -> Optional[Target]:
        recent_state = self._recent_ai_conversation_state(max_messages=max_messages)
        primary_target = recent_state.get("primary_target")
        if isinstance(primary_target, Target):
            return primary_target
        suggested_targets = recent_state.get("suggested_targets")
        if isinstance(suggested_targets, list):
            for target in suggested_targets:
                if isinstance(target, Target):
                    return target
        if not bool(getattr(self.llm_config, "enable_chat_memory", False)):
            return None

        recent_user_texts: list[str] = []
        recent_ai_texts: list[str] = []
        for message in reversed(self._ai_messages):
            kind = str(message.get("kind", "") or "").strip().lower()
            if kind not in {"user", "ai"}:
                continue
            text = str(message.get("text", "") or "").strip()
            if not text:
                continue
            if kind == "user":
                recent_user_texts.append(text)
            else:
                recent_ai_texts.append(text)
            if len(recent_user_texts) + len(recent_ai_texts) >= max_messages:
                break

        for text in recent_user_texts:
            target = self._find_referenced_target_in_question(text)
            if target is not None:
                return target
        for text in recent_ai_texts:
            target = self._find_referenced_target_in_question(text)
            if target is not None:
                return target
        return None

    def _parse_object_query_spec(
        self,
        question: str,
        *,
        selected_target: Optional[Target] = None,
    ) -> Optional[ObjectQuerySpec]:
        text = str(question or "").strip()
        if not text:
            return None
        resolved_target = self._resolve_object_query_target(question, selected_target=selected_target)
        object_scoped = _looks_like_object_scoped_query(question)
        wants_guidance = _looks_like_observing_guidance_query(question)
        wants_fact = _looks_like_object_class_query(question) or any(
            marker in _normalize_catalog_display_name(question).lower()
            for marker in (
                "what type",
                "type of object",
                "what is",
                "what kind",
                "jaki typ",
                "jaki to typ",
                "jaki to obiekt",
                "co to za obiekt",
                "czym jest",
                "co to jest",
            )
        )
        wants_selected_llm = self._should_auto_route_selected_target_query(question, resolved_target)
        blocked_no_selection = bool(object_scoped and resolved_target is None)
        if not (object_scoped or wants_guidance or wants_fact or wants_selected_llm or blocked_no_selection):
            return None
        return ObjectQuerySpec(
            target=resolved_target,
            object_scoped=object_scoped,
            wants_guidance=wants_guidance,
            wants_fact=wants_fact,
            wants_selected_llm=wants_selected_llm,
            blocked_no_selection=blocked_no_selection,
        )

    def _parse_compare_query_spec(
        self,
        question: str,
        *,
        selected_target: Optional[Target] = None,
    ) -> Optional[CompareQuerySpec]:
        normalized = _normalize_catalog_display_name(question).lower()
        if not normalized:
            return None

        flags = _question_action_flags(
            normalized,
            {
                "compare": (
                    "compare",
                    "comparison",
                    "vs",
                    "versus",
                    "between",
                    "porown",
                    "porówn",
                    "better",
                    "lepszy",
                    "lepsza",
                ),
                "choose": (
                    "which one",
                    "which is better",
                    "which is best",
                    "which target",
                    "which object",
                    "ktory",
                    "który",
                    "ktora",
                    "która",
                    "best of",
                    "best between",
                ),
                "brightness": (
                    "brighter",
                    "brightest",
                    "jaśniejs",
                    "jasniejs",
                    "jaśniejsze",
                    "jasniejsze",
                    "najjaś",
                    "najas",
                ),
                "reason": (
                    "why",
                    "reason",
                    "justify",
                    "uzasad",
                    "dlaczego",
                    "czemu",
                ),
            },
        )

        explicit_targets = self._find_referenced_targets_in_question(question, max_targets=6)
        if len(explicit_targets) == 1 and isinstance(selected_target, Target) and not _targets_match(explicit_targets[0], selected_target):
            explicit_targets.append(selected_target)

        if len(explicit_targets) < 2 and (flags["compare"] or flags["choose"]):
            recent_targets = [
                target
                for target in list(self._recent_ai_conversation_state().get("suggested_targets", []) or [])
                if isinstance(target, Target)
            ]
            if len(recent_targets) >= 2:
                explicit_targets = recent_targets[: min(5, len(recent_targets))]

        if len(explicit_targets) < 2:
            return None
        if not (flags["compare"] or flags["choose"]):
            return None

        deduped: list[Target] = []
        seen: set[str] = set()
        for target in explicit_targets:
            dedupe_key = _normalize_catalog_token(target.source_object_id or target.name)
            if not dedupe_key or dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            deduped.append(target)

        if len(deduped) < 2:
            return None

        return CompareQuerySpec(
            targets=tuple(deduped),
            criterion="brightness" if flags["brightness"] else "overall",
            return_best_only=bool(flags["choose"]),
            include_reason=bool(flags["reason"] or flags["choose"]),
        )

    def _build_local_compare_response(
        self,
        question: str,
        *,
        compare_query: CompareQuerySpec,
    ) -> Optional[dict[str, Any]]:
        items: list[dict[str, object]] = []
        seen: set[str] = set()
        for target in compare_query.targets:
            entry = self._lookup_target_observing_candidate(target)
            if not isinstance(entry, dict):
                continue
            candidate_target = entry.get("target")
            metrics = entry.get("metrics")
            if not isinstance(candidate_target, Target) or not isinstance(metrics, TargetNightMetrics):
                continue
            dedupe_key = _normalize_catalog_token(candidate_target.source_object_id or candidate_target.name)
            if not dedupe_key or dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            items.append(entry)

        if len(items) < 2:
            return None

        is_polish = any(
            token in _normalize_catalog_display_name(question).lower()
            for token in ("porown", "porówn", "któr", "ktora", "która", "lepszy", "lepsza")
        )

        def _sort_key(item: dict[str, object]) -> tuple[object, ...]:
            target = item["target"]
            metrics = item["metrics"]
            assert isinstance(target, Target)
            assert isinstance(metrics, TargetNightMetrics)
            magnitude = (
                float(target.magnitude)
                if target.magnitude is not None and math.isfinite(float(target.magnitude))
                else float("inf")
            )
            current_alt = _safe_float(item.get("current_alt"))
            current_alt_sort = float(current_alt) if current_alt is not None else -1.0
            if compare_query.criterion == "brightness":
                return (magnitude, -float(metrics.score), -current_alt_sort, target.name.lower())
            return (-float(metrics.score), -float(metrics.hours_above_limit), -current_alt_sort, magnitude, target.name.lower())

        items.sort(key=_sort_key)
        best_item = items[0]
        best_target = best_item["target"]
        best_metrics = best_item["metrics"]
        assert isinstance(best_target, Target)
        assert isinstance(best_metrics, TargetNightMetrics)

        def _format_item_summary(item: dict[str, object]) -> str:
            target = item["target"]
            metrics = item["metrics"]
            assert isinstance(target, Target)
            assert isinstance(metrics, TargetNightMetrics)
            bits = [f"score {metrics.score:.1f}"]
            magnitude_label = _target_magnitude_label(target)
            if target.magnitude is not None and math.isfinite(float(target.magnitude)):
                bits.append(f"{magnitude_label} {float(target.magnitude):.2f}")
            best_window = str(item.get("best_window") or "").strip()
            if best_window:
                bits.append(f"best {best_window}")
            current_alt = _safe_float(item.get("current_alt"))
            if current_alt is not None and math.isfinite(current_alt):
                bits.append(f"now alt {current_alt:.1f}°")
            best_airmass = _safe_float(item.get("best_airmass"))
            if best_airmass is not None and math.isfinite(best_airmass):
                bits.append(f"min airmass {best_airmass:.2f}")
            return f"{target.name} ({', '.join(bits)})"

        compared_targets = [item["target"] for item in items if isinstance(item.get("target"), Target)]
        compared_names = ", ".join(target.name for target in compared_targets[:5])
        action_targets = [target for target in compared_targets if not self._plan_contains_target(target)]

        if compare_query.return_best_only:
            reasons: list[str] = []
            if compare_query.criterion == "brightness":
                if best_target.magnitude is not None and math.isfinite(float(best_target.magnitude)):
                    reasons.append(f"brightest with {_target_magnitude_label(best_target).lower()} {float(best_target.magnitude):.2f}")
            else:
                reasons.append(f"highest score {best_metrics.score:.1f}")
                if best_metrics.hours_above_limit > 0:
                    reasons.append(f"over limit {best_metrics.hours_above_limit:.1f} h")
            best_window = str(best_item.get("best_window") or "").strip()
            if best_window:
                reasons.append(f"best {best_window}")
            if is_polish:
                reason_text = ", ".join(reasons[:3]) if reasons else "najlepszy łączny profil obserwacyjny"
                text = f"Najlepszy wybór między {compared_names}: {best_target.name} — {reason_text}."
                return {
                    "text": text,
                    "primary_target": best_target,
                    "suggested_targets": compared_targets,
                    "action_targets": action_targets,
                }
            reason_text = ", ".join(reasons[:3]) if reasons else "best combined observing profile"
            text = f"Best choice between {compared_names}: {best_target.name} — {reason_text}."
            return {
                "text": text,
                "primary_target": best_target,
                "suggested_targets": compared_targets,
                "action_targets": action_targets,
            }

        rows = [_format_item_summary(item) for item in items]
        if is_polish:
            text = "Porównanie:\n" + "\n".join(f"- {row}" for row in rows)
        else:
            text = "Comparison:\n" + "\n".join(f"- {row}" for row in rows)
        return {
            "text": text,
            "primary_target": best_target,
            "suggested_targets": compared_targets,
            "action_targets": action_targets,
        }

    def _resolve_object_query_target(self, question: str, *, selected_target: Optional[Target]) -> Optional[Target]:
        explicit_target = self._find_referenced_target_in_question(question)
        if explicit_target is not None:
            return explicit_target

        if _looks_like_object_scoped_query(question):
            recent_target = self._resolve_recent_chat_target_reference()
            if recent_target is not None:
                return recent_target
            return selected_target

        if selected_target is not None and self._should_auto_route_selected_target_query(question, selected_target):
            return selected_target
        return None

    def _should_auto_route_selected_target_query(self, text: str, target: Optional[Target]) -> bool:
        if target is None:
            return False
        normalized = _normalize_catalog_display_name(text).lower()
        if not normalized:
            return False
        if _looks_like_object_scoped_query(normalized):
            return True

        selected_tokens = [
            _normalize_catalog_display_name(target.name).lower(),
            _normalize_catalog_display_name(target.source_object_id).lower(),
        ]
        mentions_selected_target = any(token and token in normalized for token in selected_tokens)
        if not mentions_selected_target:
            return False

        object_markers = (
            "describe",
            "details",
            "detail",
            "summary",
            "summarize",
            "tell me about",
            "what is",
            "what are",
            "info",
            "information",
            "object",
            "target",
            "obiekt",
            "opisz",
            "szczegoly",
            "szczegóły",
            "informacje",
        )
        session_markers = (
            "tonight",
            "plan",
            "schedule",
            "compare",
            "comparison",
            "order",
            "rank",
            "ranking",
            "which target",
            "which object",
            "best target",
            "other targets",
            "lista",
            "porown",
            "porówn",
            "kolejn",
            "harmonogram",
        )
        return any(marker in normalized for marker in object_markers) and not any(
            marker in normalized for marker in session_markers
        )

    def _dispatch_selected_target_llm_question(self, target: Target, question: str, *, label: str) -> None:
        prompt = self._build_selected_target_llm_prompt(target, question)
        self._dispatch_llm(
            prompt,
            tag="chat_selected",
            label=label,
            primary_target=target,
        )

    def _build_deterministic_observation_order(self) -> tuple[list[dict[str, object]], list[str]]:
        payload = self.full_payload if isinstance(getattr(self, "full_payload", None), dict) else None
        if not payload:
            return [], ["Run a visibility calculation first so night windows are available."]

        try:
            tz = pytz.timezone(str(payload.get("tz", "UTC")))
        except Exception:
            tz = pytz.UTC

        try:
            times = [t.astimezone(tz) for t in mdates.num2date(payload["times"])]
        except Exception:
            return [], ["Visibility samples are unavailable in the current plot state."]

        if not times:
            return [], ["Visibility samples are unavailable in the current plot state."]

        limit = float(self.limit_spin.value())
        sun_alt_limit = self._sun_alt_limit()
        sun_alt_series = np.array(payload.get("sun_alt", np.full(len(times), np.nan)), dtype=float)
        obs_sun_mask = np.isfinite(sun_alt_series) & (sun_alt_series <= sun_alt_limit)

        considered_rows = set(range(len(self.targets)))
        if self.table_model.row_enabled:
            visible_rows = {idx for idx, enabled in enumerate(self.table_model.row_enabled) if enabled}
            if visible_rows:
                considered_rows = visible_rows

        valid_items: list[dict[str, object]] = []
        invalid_notes: list[str] = []

        for idx, target in enumerate(self.targets):
            if idx not in considered_rows:
                continue

            target_payload = payload.get(target.name)
            if not isinstance(target_payload, dict):
                invalid_notes.append(f"{target.name}: missing altitude series in current plot data.")
                continue

            alt = np.array(target_payload.get("altitude", []), dtype=float)
            if alt.shape[0] != len(times):
                invalid_notes.append(f"{target.name}: incomplete altitude series in current plot data.")
                continue

            valid_mask = np.isfinite(alt) & (alt >= limit) & obs_sun_mask
            metrics = self.target_metrics.get(target.name)
            if not valid_mask.any():
                invalid_notes.append(f"{target.name}: no valid observing window under current constraints.")
                continue

            valid_indices = np.where(valid_mask)[0]
            runs = np.split(valid_indices, np.where(np.diff(valid_indices) != 1)[0] + 1)
            run_candidates: list[dict[str, object]] = []
            for run in runs:
                if len(run) == 0:
                    continue
                start_idx = int(run[0])
                end_idx = min(int(run[-1]) + 1, len(times) - 1)
                peak_idx = int(run[np.argmax(alt[run])])
                start_dt = times[start_idx]
                end_dt = times[end_idx]
                duration_h = max(0.0, (end_dt - start_dt).total_seconds() / 3600.0)
                still_rising = peak_idx >= int(run[-1])
                run_candidates.append(
                    {
                        "window_start": start_dt,
                        "window_end": end_dt,
                        "peak_time": times[peak_idx],
                        "window_hours": duration_h,
                        "still_rising": still_rising,
                    }
                )

            if not run_candidates:
                invalid_notes.append(f"{target.name}: no valid observing window under current constraints.")
                continue

            selected_run = min(
                run_candidates,
                key=lambda item: (
                    int(bool(item["still_rising"])),
                    item["window_end"] if bool(item["still_rising"]) else item["window_start"],
                    item["window_start"] if bool(item["still_rising"]) else item["peak_time"],
                    item["window_hours"],
                    item["peak_time"],
                ),
            )

            valid_items.append(
                {
                    "row_index": idx,
                    "name": target.name,
                    "priority": int(target.priority),
                    "score": float(metrics.score) if metrics else 0.0,
                    "hours_above_limit": float(metrics.hours_above_limit) if metrics else 0.0,
                    "max_altitude_deg": float(metrics.max_altitude_deg) if metrics else 0.0,
                    "peak_moon_sep_deg": float(metrics.peak_moon_sep_deg) if metrics else 0.0,
                    "window_start": selected_run["window_start"],
                    "window_end": selected_run["window_end"],
                    "peak_time": selected_run["peak_time"],
                    "window_hours": selected_run["window_hours"],
                    "still_rising": selected_run["still_rising"],
                }
            )

        valid_items.sort(
            key=lambda item: (
                int(bool(item["still_rising"])),
                item["window_end"] if bool(item["still_rising"]) else item["window_start"],
                item["window_start"] if bool(item["still_rising"]) else item["peak_time"],
                item["window_hours"],
                item["peak_time"],
                item["window_end"],
                item["window_start"],
                -int(item["priority"]),
                -float(item["score"]),
                str(item["name"]).lower(),
            )
        )
        return valid_items, invalid_notes

    _SYSTEM_PROMPT = (
        "You are an expert astronomy observation assistant. "
        "Provide concise, practical, and accurate guidance for planning observations. "
        "Keep answers compact by default and avoid unnecessary background. "
        "Do not repeat the same metric, sentence, or recommendation. "
        "If the user asks about a specific class such as QSO, AGN, supernova, nova, CV, or galaxy, restrict the answer to that class only. "
        "Use only the provided session context when answering questions about tonight's plan. "
        "If local knowledge notes are included, treat them as grounded heuristics and caveats for practical advice. "
        "Do not introduce targets that are not listed in the provided session context unless the user explicitly asks for off-plan suggestions. "
        "If a cached BHTOM suggestion shortlist is included in the session context, you may use only those listed BHTOM suggestions as off-plan candidates. "
        "Use target names exactly as they appear in the provided context; do not swap in unrelated Messier, NGC, or other catalog examples. "
        "Interpret dusk as the start of the observing night window and dawn as the end of the observing night window. "
        "If the data is missing or ambiguous, say so instead of guessing."
    )

    def _format_target_coords_compact(self, target: Target) -> str:
        ra_txt = Angle(target.ra, u.deg).to_string(unit=u.hourangle, sep=":", pad=True, precision=0)
        dec_txt = Angle(target.dec, u.deg).to_string(unit=u.deg, sep=":", alwayssign=True, pad=True, precision=0)
        return f"{ra_txt} {dec_txt}"

    def _format_target_best_window_compact(self, target: Target) -> str:
        window = self.target_windows.get(target.name)
        if window is None:
            return ""
        tz_name = "UTC"
        if isinstance(getattr(self, "last_payload", None), dict):
            tz_name = str(self.last_payload.get("tz", "UTC"))
        try:
            tz = pytz.timezone(tz_name)
        except Exception:
            tz = pytz.UTC
        try:
            start = window[0].astimezone(tz).strftime("%H:%M")
            end = window[1].astimezone(tz).strftime("%H:%M")
        except Exception:
            start = window[0].strftime("%H:%M")
            end = window[1].strftime("%H:%M")
        return f"{start}-{end}"

    def _build_fast_target_llm_context(self, target: Target) -> str:
        self._ensure_known_target_type(target)
        lines: list[str] = [f"Name: {target.name}", f"Source: {_target_source_label(target.source_catalog)}"]

        source_id = target.source_object_id.strip()
        if source_id and _normalize_catalog_token(source_id) != _normalize_catalog_token(target.name):
            lines.append(f"Catalog ID: {source_id}")
        if target.object_type and not _object_type_is_unknown(target.object_type):
            lines.append(f"Type: {target.object_type}")
        class_family = self._target_class_family(target)
        if class_family:
            lines.append(f"Class family: {class_family}")

        bhtom_importance = self._bhtom_importance_for_target(target)
        if bhtom_importance is not None:
            lines.append(f"BHTOM importance: {bhtom_importance:.1f}")

        if target.magnitude is not None:
            lines.append(f"{_target_magnitude_label(target)}: {target.magnitude:.2f}")
        lines.append(f"Coords: {self._format_target_coords_compact(target)}")

        row_index: Optional[int] = None
        for idx, existing in enumerate(self.targets):
            if _targets_match(existing, target):
                row_index = idx
                break

        metrics = self.target_metrics.get(target.name)
        tonight_parts: list[str] = []
        best_window = self._format_target_best_window_compact(target)
        if best_window:
            tonight_parts.append(f"best {best_window}")
        if metrics is not None:
            tonight_parts.append(f"max alt {metrics.max_altitude_deg:.0f} deg")
            tonight_parts.append(f"over limit {metrics.hours_above_limit:.1f} h")
            tonight_parts.append(f"score {metrics.score:.1f}")
        if row_index is not None:
            if row_index < len(self.table_model.current_alts):
                alt_now = self.table_model.current_alts[row_index]
                if math.isfinite(alt_now):
                    tonight_parts.append(f"now alt {alt_now:.1f} deg")
            if row_index < len(self.table_model.current_seps):
                moon_sep_now = self.table_model.current_seps[row_index]
                if math.isfinite(moon_sep_now):
                    tonight_parts.append(f"now moon sep {moon_sep_now:.1f} deg")
        if tonight_parts:
            lines.append("Tonight: " + ", ".join(tonight_parts))

        return "\n".join(lines)

    @staticmethod
    def _format_compact_number(value: float, *, decimals_small: int = 3) -> str:
        abs_value = abs(float(value))
        if abs_value >= 100:
            return f"{value:.0f}"
        if abs_value >= 10:
            return f"{value:.1f}"
        if abs_value >= 1:
            return f"{value:.2f}"
        return f"{value:.{decimals_small}f}"

    @staticmethod
    def _format_signed_value(value: float, decimals: int = 2) -> str:
        return f"{float(value):+.{decimals}f}"

    def _is_star_like_target(self, target: Target, details: dict[str, object]) -> bool:
        object_type = _normalize_catalog_token(target.object_type)
        if "*" in object_type:
            return True
        sp_type = str(details.get("sp_type", "")).strip()
        if sp_type:
            return True
        return any(marker in object_type for marker in ("star", "nova", "binary", "cv"))

    def _format_distance_text(self, details: dict[str, object]) -> str:
        distance_value = details.get("distance_value")
        unit = str(details.get("distance_unit", "")).strip()
        if not isinstance(distance_value, (int, float)) or not unit:
            return ""
        value = float(distance_value)
        plus_err = details.get("distance_plus_err")
        minus_err = details.get("distance_minus_err")
        value_txt = self._format_compact_number(value, decimals_small=3)
        if isinstance(plus_err, (int, float)) and isinstance(minus_err, (int, float)):
            plus_abs = abs(float(plus_err))
            minus_abs = abs(float(minus_err))
            if math.isfinite(plus_abs) and math.isfinite(minus_abs):
                if abs(plus_abs - minus_abs) <= max(0.1, 0.05 * max(plus_abs, minus_abs, 1.0)):
                    err_txt = self._format_compact_number(max(plus_abs, minus_abs), decimals_small=3)
                    return f"{value_txt} +/- {err_txt} {unit}"
                plus_txt = self._format_compact_number(plus_abs, decimals_small=3)
                minus_txt = self._format_compact_number(minus_abs, decimals_small=3)
                return f"{value_txt} +{plus_txt}/-{minus_txt} {unit}"
        return f"{value_txt} {unit}"

    def _format_size_text(self, details: dict[str, object]) -> str:
        major = details.get("size_major_arcmin")
        minor = details.get("size_minor_arcmin")
        if not isinstance(major, (int, float)):
            return ""
        major_txt = self._format_compact_number(float(major), decimals_small=2)
        if isinstance(minor, (int, float)) and math.isfinite(float(minor)):
            minor_val = float(minor)
            if abs(float(major) - minor_val) > 0.05:
                minor_txt = self._format_compact_number(minor_val, decimals_small=2)
                return f"{major_txt} x {minor_txt} arcmin"
        return f"{major_txt} arcmin"

    def _format_kinematics_text(self, details: dict[str, object]) -> str:
        parts: list[str] = []
        radial_velocity = details.get("radial_velocity_kms")
        if isinstance(radial_velocity, (int, float)):
            rv_txt = self._format_compact_number(float(radial_velocity), decimals_small=2)
            parts.append(f"rv {rv_txt} km/s")
        redshift = details.get("redshift")
        if isinstance(redshift, (int, float)) and math.isfinite(float(redshift)):
            parts.append(f"z {float(redshift):+.5f}")
        return ", ".join(parts)

    def _get_simbad_compact_data(self, target: Target) -> dict[str, object]:
        cache_keys = [
            target.name.strip().lower(),
            target.source_object_id.strip().lower(),
        ]
        storage = getattr(self, "app_storage", None)
        primary_key = cache_keys[0]
        if primary_key:
            primary_cached = self._simbad_compact_cache.get(primary_key)
            if primary_cached is not None:
                return dict(primary_cached)
            if storage is not None:
                try:
                    persisted = storage.cache.get_json("simbad_compact", primary_key)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to read SIMBAD compact cache for '%s': %s", target.name, exc)
                else:
                    if isinstance(persisted, dict):
                        self._simbad_compact_cache[primary_key] = dict(persisted)
                        return dict(persisted)
        secondary_key = cache_keys[1]
        if secondary_key and secondary_key != primary_key:
            secondary_cached = self._simbad_compact_cache.get(secondary_key)
            if secondary_cached is not None:
                secondary_status = str(secondary_cached.get("_simbad_status", "")).strip().lower()
                if secondary_status == "matched":
                    return dict(secondary_cached)
            if storage is not None:
                try:
                    persisted = storage.cache.get_json("simbad_compact", secondary_key)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to read SIMBAD compact cache for '%s': %s", target.name, exc)
                else:
                    if isinstance(persisted, dict):
                        self._simbad_compact_cache[secondary_key] = dict(persisted)
                        if str(persisted.get("_simbad_status", "")).strip().lower() == "matched":
                            return dict(persisted)

        query_candidates: list[str] = []
        for candidate in (target.name, target.source_object_id):
            query = candidate.strip()
            if query and query.lower() not in {item.lower() for item in query_candidates}:
                query_candidates.append(query)

        try:
            custom = Simbad()
            custom.add_votable_fields(
                "V",
                "R",
                "B",
                "sp",
                "parallax",
                "mesdistance",
                "mesfe_h",
                "velocity",
                "galdim_majaxis",
                "galdim_minaxis",
            )
            result = None
            row_idx = 0
            match_mode = ""
            for query in query_candidates:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=NoResultsWarning)
                    result = custom.query_object(query)
                if _simbad_has_row(result):
                    match_mode = "name"
                    break
            if not _simbad_has_row(result):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=NoResultsWarning)
                    result = custom.query_region(target.skycoord, radius=120 * u.arcsec)
                row_idx = _simbad_best_row_index(result, reference_coord=target.skycoord)
                if _simbad_has_row(result):
                    match_mode = "coordinates"
        except Exception as exc:  # noqa: BLE001
            logger.warning("SIMBAD compact lookup failed for '%s': %s", target.name, exc)
            payload = {"_simbad_status": "lookup_failed"}
            if storage is not None:
                for key in cache_keys:
                    if not key:
                        continue
                    try:
                        storage.cache.set_json("simbad_compact", key, payload, ttl_s=SIMBAD_COMPACT_NEGATIVE_CACHE_TTL_S)
                    except Exception:
                        pass
            return payload

        if not _simbad_has_row(result):
            compact_data = {"_simbad_status": "not_found"}
            for key in cache_keys:
                if key:
                    self._simbad_compact_cache[key] = dict(compact_data)
                    if storage is not None:
                        try:
                            storage.cache.set_json("simbad_compact", key, compact_data, ttl_s=SIMBAD_COMPACT_NEGATIVE_CACHE_TTL_S)
                        except Exception:
                            pass
            return compact_data

        compact_data = _extract_simbad_compact_measurements(result, row_idx=row_idx)
        compact_data["_simbad_status"] = "matched"
        if match_mode:
            compact_data["_simbad_match_mode"] = match_mode
        if match_mode == "coordinates":
            row_coord = _simbad_row_coord(result, row_idx=row_idx)
            if row_coord is not None:
                try:
                    compact_data["_simbad_sep_arcsec"] = float(row_coord.separation(target.skycoord).arcsec)
                except Exception:
                    pass
        main_id = _extract_simbad_name(result, target.name, row_idx=row_idx).strip().lower()
        for key in (*cache_keys, main_id):
            if key:
                self._simbad_compact_cache[key] = dict(compact_data)
                if storage is not None:
                    try:
                        storage.cache.set_json("simbad_compact", key, compact_data, ttl_s=SIMBAD_COMPACT_CACHE_TTL_S)
                    except Exception:
                        pass
        return compact_data

    def _build_compact_target_description(self, target: Target) -> str:
        self._ensure_known_target_type(target)
        lines = [f"Source: {_target_source_label(target.source_catalog)}"]
        bhtom_importance = self._bhtom_importance_for_target(target)

        source_id = target.source_object_id.strip()
        if source_id and _normalize_catalog_token(source_id) != _normalize_catalog_token(target.name):
            lines.append(f"Catalog ID: {source_id}")
        if target.object_type:
            lines.append(f"Type: {target.object_type}")
        class_family = self._target_class_family(target)
        if class_family:
            lines.append(f"Class family: {class_family}")
        if bhtom_importance is not None:
            lines.append(f"BHTOM importance: {bhtom_importance:.1f}")
        details = self._get_simbad_compact_data(target)
        simbad_status = str(details.get("_simbad_status", "")).strip().lower()
        simbad_match_mode = str(details.get("_simbad_match_mode", "")).strip().lower()
        if simbad_status == "matched":
            if simbad_match_mode == "coordinates":
                simbad_sep_arcsec = _safe_float(details.get("_simbad_sep_arcsec"))
                if simbad_sep_arcsec is not None and math.isfinite(simbad_sep_arcsec):
                    lines.append(f"SIMBAD: coordinate match ({simbad_sep_arcsec:.1f}\")")
                else:
                    lines.append("SIMBAD: coordinate match")
            else:
                lines.append("SIMBAD: name match")
        elif simbad_status == "not_found":
            lines.append("SIMBAD: not found")
        elif simbad_status == "lookup_failed":
            lines.append("SIMBAD: unavailable")
        photometry = details.get("photometry", {})
        if photometry:
            phot_txt = ", ".join(
                f"{band} {float(photometry[band]):.2f}" for band in ("B", "V", "R")
                if isinstance(photometry, dict) and band in photometry
            )
            if phot_txt:
                lines.append(f"Photometry: {phot_txt}")
        elif target.magnitude is not None:
            lines.append(f"{_target_magnitude_label(target)}: {target.magnitude:.2f}")
        lines.append(f"Coords: {self._format_target_coords_compact(target)}")

        is_star_like = self._is_star_like_target(target, details)
        if is_star_like:
            stellar_parts: list[str] = []
            sp_type = str(details.get("sp_type", "")).strip()
            if sp_type:
                stellar_parts.append(f"SpT {sp_type}")
            parallax = details.get("parallax_mas")
            if isinstance(parallax, (int, float)):
                parallax_txt = f"{float(parallax):.3f} mas"
                parallax_err = details.get("parallax_err_mas")
                if isinstance(parallax_err, (int, float)) and math.isfinite(float(parallax_err)):
                    parallax_txt = f"{float(parallax):.3f} +/- {float(parallax_err):.3f} mas"
                stellar_parts.append(f"parallax {parallax_txt}")
            if stellar_parts:
                lines.append("Stellar: " + ", ".join(stellar_parts))

            distance_txt = self._format_distance_text(details)
            if distance_txt:
                lines.append(f"Distance: {distance_txt}")
        else:
            size_txt = self._format_size_text(details)
            if size_txt:
                lines.append(f"Angular size: {size_txt}")
            kinematics_txt = self._format_kinematics_text(details)
            if kinematics_txt:
                lines.append(f"Kinematics: {kinematics_txt}")

        physical_parts: list[str] = []
        teff_k = details.get("teff_k")
        if is_star_like and isinstance(teff_k, (int, float)):
            physical_parts.append(f"Teff {float(teff_k):.0f} K")
        fe_h = details.get("fe_h")
        if isinstance(fe_h, (int, float)):
            physical_parts.append(f"[Fe/H] {self._format_signed_value(float(fe_h), decimals=2)}")
        if physical_parts:
            lines.append(("Atmosphere: " if is_star_like else "Physical: ") + ", ".join(physical_parts))

        metrics = self.target_metrics.get(target.name)
        best_window = self._format_target_best_window_compact(target)
        tonight_parts: list[str] = []
        if best_window:
            tonight_parts.append(f"best {best_window}")
        if metrics is not None:
            tonight_parts.append(f"max alt {metrics.max_altitude_deg:.0f} deg")
            tonight_parts.append(f"over limit {metrics.hours_above_limit:.1f} h")
            tonight_parts.append(f"score {metrics.score:.1f}")
        elif self.targets:
            tonight_parts.append("run visibility calculation for tonight's window")
        if tonight_parts:
            lines.append("Tonight: " + ", ".join(tonight_parts))
        return "\n".join(lines)

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
        prefetch_worker = BhtomObservatoryPresetWorker(req_id, base_url=base_url, token=token, parent=self)
        prefetch_worker.progress.connect(self._on_bhtom_observatory_prefetch_progress)
        prefetch_worker.completed.connect(self._on_bhtom_observatory_prefetch_completed)
        prefetch_worker.finished.connect(lambda w=prefetch_worker: self._on_bhtom_observatory_prefetch_finished(w))
        prefetch_worker.finished.connect(prefetch_worker.deleteLater)
        self._bhtom_observatory_worker = prefetch_worker
        prefetch_worker.start()
        return True

    @Slot(int, int, str)
    def _on_bhtom_observatory_prefetch_progress(self, page: int, _total_pages: int, message: str) -> None:
        if self._bhtom_observatory_worker is None:
            return
        txt = str(message or "").strip() or f"Loading BHTOM presets... page {int(page)}"
        self._bhtom_observatory_loading_message = txt
        self.bhtom_observatory_presets_loading.emit(True, txt)

    @Slot(int, list, str)
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
            QMessageBox.information(self, title, "A BHTOM request is already in progress.")
            return False

        context, error = self._build_bhtom_suggestion_context()
        if context is None:
            title = "Quick Targets" if mode == "quick" else "Suggest Targets"
            QMessageBox.warning(self, title, error or "Unable to prepare BHTOM request.")
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
            parent=self,
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

    @Slot(int, list, list, int, int)
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

    @Slot(int, list, list, list, str)
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
                        QMessageBox.warning(self, "Suggest Targets", error)
                elif mode == "quick":
                    QMessageBox.warning(self, "Quick Targets", error)
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

    @Slot(int)
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

    @Slot()
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

    def _resolve_ai_intent(self, question: str) -> AIIntent:
        text = str(question or "").strip()
        if not text:
            return AIIntent(kind="empty", question="", label="")

        selected_target = self._selected_target_or_none()
        class_query = self._parse_class_query_spec(text)
        object_query = self._parse_object_query_spec(text, selected_target=selected_target)
        compare_query = self._parse_compare_query_spec(text, selected_target=selected_target)
        if compare_query is not None:
            local_compare_response = self._build_local_compare_response(text, compare_query=compare_query)
            if local_compare_response is not None:
                primary_target = local_compare_response.get("primary_target")
                suggested_targets = tuple(
                    target
                    for target in (local_compare_response.get("suggested_targets") or [])
                    if isinstance(target, Target)
                )
                action_targets = tuple(
                    target
                    for target in (local_compare_response.get("action_targets") or [])
                    if isinstance(target, Target)
                )
                return AIIntent(
                    kind="local_compare",
                    question=text,
                    label=text,
                    target=primary_target if isinstance(primary_target, Target) else None,
                    requested_class=class_query.requested_class if class_query is not None else "",
                    class_query=class_query,
                    object_query=object_query,
                    compare_query=compare_query,
                    local_text=str(local_compare_response.get("text", "") or "").strip(),
                    suggested_targets=suggested_targets,
                    action_targets=action_targets,
                )

        local_class_response = self._build_local_class_observing_response(text)
        if local_class_response is not None:
            requested_class = str(local_class_response.get("requested_class", "") or "").strip()
            primary_target = local_class_response.get("primary_target")
            suggested_targets = tuple(
                target
                for target in (local_class_response.get("suggested_targets") or [])
                if isinstance(target, Target)
            )
            action_targets = tuple(
                target for target in suggested_targets if not self._plan_contains_target(target)
            )
            return AIIntent(
                kind="local_class",
                question=text,
                label=text,
                target=primary_target if isinstance(primary_target, Target) else None,
                requested_class=requested_class,
                class_query=class_query,
                object_query=object_query,
                compare_query=compare_query,
                local_text=str(local_class_response.get("text", "") or "").strip(),
                suggested_targets=suggested_targets,
                action_targets=action_targets,
            )

        resolved_object_target = object_query.target if object_query is not None else None
        local_observing_answer = self._build_local_object_observing_answer(text, target=resolved_object_target)
        if local_observing_answer is not None:
            return AIIntent(
                kind="local_object_guidance",
                question=text,
                label=text,
                target=resolved_object_target if isinstance(resolved_object_target, Target) else None,
                object_query=object_query,
                compare_query=compare_query,
                local_text=local_observing_answer,
            )

        local_fact_answer = self._build_local_object_fact_answer(text, target=resolved_object_target)
        if local_fact_answer is not None:
            return AIIntent(
                kind="local_object_fact",
                question=text,
                label=text,
                target=resolved_object_target if isinstance(resolved_object_target, Target) else None,
                object_query=object_query,
                compare_query=compare_query,
                local_text=local_fact_answer,
            )

        if object_query is not None and object_query.blocked_no_selection:
            return AIIntent(kind="blocked_no_selection", question=text, label=text)

        if object_query is not None and object_query.wants_selected_llm and isinstance(resolved_object_target, Target):
            return AIIntent(
                kind="llm_selected",
                question=text,
                label=text,
                target=resolved_object_target if isinstance(resolved_object_target, Target) else None,
                object_query=object_query,
                compare_query=compare_query,
            )

        return AIIntent(
            kind="llm_session",
            question=text,
            label=text,
            target=resolved_object_target if isinstance(resolved_object_target, Target) else None,
            requested_class=class_query.requested_class if class_query is not None else "",
            class_query=class_query,
            object_query=object_query,
            compare_query=compare_query,
        )

    def _execute_ai_intent(self, intent: AIIntent) -> None:
        if intent.kind == "empty":
            return
        if intent.kind == "blocked_no_selection":
            QMessageBox.information(
                self,
                "No selection",
                "Select one target first, or use a session-wide question.",
            )
            return

        worker = self._llm_worker
        if worker is not None and worker.isRunning():
            self._append_ai_message(
                "The AI assistant is still processing the previous request.",
                is_error=True,
            )
            return

        if hasattr(self, "ai_input"):
            self.ai_input.clear()
        if hasattr(self, "ai_toggle_btn") and not self.ai_toggle_btn.isChecked():
            self.ai_toggle_btn.setChecked(True)

        if intent.kind == "local_class":
            self._refresh_ai_knowledge_hint([])
            self._append_ai_message(
                intent.label,
                is_user=True,
                requested_class=intent.requested_class,
                primary_target=intent.target,
            )
            self._append_ai_message(
                intent.local_text,
                is_ai=True,
                requested_class=intent.requested_class,
                primary_target=intent.target,
                suggested_targets=list(intent.suggested_targets),
                action_targets=list(intent.action_targets),
            )
            self._set_ai_status("Ready", tone="info")
            return

        if intent.kind == "local_compare":
            self._refresh_ai_knowledge_hint([])
            self._append_ai_message(
                intent.label,
                is_user=True,
                requested_class=intent.requested_class,
                primary_target=intent.target,
            )
            self._append_ai_message(
                intent.local_text,
                is_ai=True,
                requested_class=intent.requested_class,
                primary_target=intent.target,
                suggested_targets=list(intent.suggested_targets),
                action_targets=list(intent.action_targets),
            )
            self._set_ai_status("Ready", tone="info")
            return

        if intent.kind == "local_object_guidance":
            self._refresh_ai_knowledge_hint([])
            self._append_ai_message(intent.label, is_user=True, primary_target=intent.target)
            self._append_ai_message(intent.local_text, is_ai=True, primary_target=intent.target)
            self._set_ai_status("Ready", tone="info")
            return

        if intent.kind == "local_object_fact":
            self._refresh_ai_knowledge_hint([])
            self._append_ai_message(intent.label, is_user=True, primary_target=intent.target)
            self._append_ai_message(intent.local_text, is_ai=True, primary_target=intent.target)
            self._set_ai_status("Ready", tone="info")
            return

        if intent.kind == "llm_selected" and isinstance(intent.target, Target):
            prompt = self._build_selected_target_llm_prompt(intent.target, intent.question)
            self._dispatch_llm(
                prompt,
                tag="chat_selected",
                label=intent.label,
                requested_class=intent.requested_class,
                primary_target=intent.target,
            )
            return

        if intent.kind != "llm_session":
            return

        knowledge_target = (
            intent.target
            if isinstance(intent.target, Target) and _looks_like_object_scoped_query(intent.question)
            else intent.target or self._find_referenced_target_in_question(intent.question)
        )
        context = self._build_session_context(user_question=intent.question)
        recent_memory = self._build_recent_chat_memory_block()
        knowledge_notes = self._select_knowledge_notes(question=intent.question, target=knowledge_target)
        knowledge_context = (
            "Local knowledge notes:\n"
            + "\n".join(_format_knowledge_note_snippet(note) for note in knowledge_notes)
            if knowledge_notes
            else ""
        )
        self._refresh_ai_knowledge_hint([note.title for note in knowledge_notes])
        prompt_sections = []
        if recent_memory:
            prompt_sections.append(recent_memory)
        if knowledge_context:
            prompt_sections.append(knowledge_context)
        prompt_sections.append(
            f"Current session context:\n{context}\n\nUser question: {intent.question}"
        )
        prompt = "\n\n".join(prompt_sections)
        self._dispatch_llm(
            prompt,
            tag="chat",
            label=intent.label,
            requested_class=intent.requested_class,
            primary_target=intent.target,
        )

    @Slot()
    def _send_ai_query(self) -> None:
        text = self.ai_input.text().strip() if hasattr(self, "ai_input") else ""
        if not text:
            return
        intent = self._resolve_ai_intent(text)
        self._execute_ai_intent(intent)

    def _build_selected_target_llm_prompt(self, target: Target, question: str) -> str:
        compact_description = self._build_fast_target_llm_context(target)
        recent_memory = self._build_recent_chat_memory_block()
        knowledge_notes = self._select_knowledge_notes(question=question, target=target)
        knowledge_context = (
            "Local knowledge notes:\n"
            + "\n".join(_format_knowledge_note_snippet(note) for note in knowledge_notes)
            if knowledge_notes
            else ""
        )
        self._refresh_ai_knowledge_hint([note.title for note in knowledge_notes])
        prompt_sections: list[str] = []
        if recent_memory:
            prompt_sections.append(recent_memory)
        if knowledge_context:
            prompt_sections.append(knowledge_context)
        prompt_sections.append(
            f"Selected object context:\n"
            f"{compact_description}\n\n"
            f"User question about the selected object: {question}\n\n"
            "Answer in at most 3 short sentences or 3 short bullets and stay grounded in the selected object context. "
            "Treat the provided Type and Class family fields as authoritative for classification questions. "
            "Do not switch to a different object. Do not recommend other targets unless the user explicitly asks. "
            "Do not repeat the same metric, sentence, or phrase."
        )
        return "\n\n".join(prompt_sections)

    def _build_fast_general_llm_prompt(self, question: str) -> str:
        recent_memory = self._build_recent_chat_memory_block()
        resolved_object_target = self._resolve_object_query_target(
            question,
            selected_target=self._selected_target_or_none(),
        )
        knowledge_target = (
            resolved_object_target
            if _looks_like_object_scoped_query(question)
            else self._find_referenced_target_in_question(question)
        )
        knowledge_notes = self._select_knowledge_notes(question=question, target=knowledge_target)
        knowledge_context = (
            "Local knowledge notes:\n"
            + "\n".join(_format_knowledge_note_snippet(note) for note in knowledge_notes)
            if knowledge_notes
            else ""
        )
        self._refresh_ai_knowledge_hint([note.title for note in knowledge_notes])
        prompt_sections: list[str] = []
        if recent_memory:
            prompt_sections.append(recent_memory)
        if knowledge_context:
            prompt_sections.append(knowledge_context)
        prompt_sections.append(
            f"User question: {question}\n\n"
            "Answer concisely in no more than 4 short sentences. "
            "Do not assume details about any selected object unless the question explicitly asks about it."
        )
        return "\n\n".join(prompt_sections)

    def _build_recent_chat_memory_block(self, *, max_messages: int = 4) -> str:
        if not bool(getattr(self.llm_config, "enable_chat_memory", False)):
            return ""

        recent_sections: list[str] = []
        for message in reversed(self._ai_messages):
            kind = str(message.get("kind", "") or "").strip().lower()
            if kind not in {"user", "ai"}:
                continue
            text = _truncate_ai_memory_text(str(message.get("text", "") or "").strip())
            if not text:
                continue
            role = "User" if kind == "user" else "LLM"
            recent_sections.append(f"{role}: {text}")
            if len(recent_sections) >= max_messages:
                break

        if not recent_sections:
            return ""
        recent_sections.reverse()
        return "Recent chat turns:\n" + "\n".join(recent_sections)

    @Slot()
    def _send_ai_selected_target_query(self) -> None:
        target = self._selected_target_or_none()

        typed_text = self.ai_input.text().strip() if hasattr(self, "ai_input") else ""
        question = typed_text or "Give a concise summary of this selected object for tonight's observing."
        if not typed_text and target is None:
            QMessageBox.information(self, "No selection", "Select one target first.")
            return
        if typed_text and hasattr(self, "ai_input"):
            self.ai_input.clear()

        if target is not None and (not typed_text or self._should_auto_route_selected_target_query(question, target)):
            label = typed_text or f"Summarize {target.name}"
            self._dispatch_selected_target_llm_question(target, question, label=label)
            return

        prompt = self._build_fast_general_llm_prompt(question)
        self._dispatch_llm(prompt, tag="chat_fast", label=question)

    @Slot()
    def _ai_describe_target(self) -> None:
        target = self._selected_target_or_none()
        if target is None:
            QMessageBox.information(self, "No selection", "Select one target first.")
            return
        if hasattr(self, "ai_toggle_btn") and not self.ai_toggle_btn.isChecked():
            self.ai_toggle_btn.setChecked(True)
        self._refresh_ai_knowledge_hint([])
        self._append_ai_message(f"Describe {target.name}", is_user=True)
        self._append_ai_message(self._build_compact_target_description(target), is_ai=True)
        worker = self._llm_worker
        if worker is None or not worker.isRunning():
            self._set_ai_status("Ready", tone="info")

    @Slot()
    def _ai_suggest_targets(self) -> None:
        self._set_ai_status("Loading suggestions...", tone="info")
        if self._bhtom_worker is not None and self._bhtom_worker.isRunning():
            QMessageBox.information(self, "Suggest Targets", "A BHTOM request is already in progress.")
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
            parent=self,
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

    def _dispatch_llm(
        self,
        prompt: str,
        tag: str,
        label: str,
        *,
        requested_class: str = "",
        primary_target: Optional[Target] = None,
    ) -> None:
        worker = self._llm_worker
        if worker is not None and worker.isRunning():
            self._append_ai_message(
                "The AI assistant is still processing the previous request.",
                is_error=True,
            )
            return
        if hasattr(self, "ai_toggle_btn") and not self.ai_toggle_btn.isChecked():
            self.ai_toggle_btn.setChecked(True)
        self._append_ai_message(
            label,
            is_user=True,
            requested_class=requested_class,
            primary_target=primary_target,
        )
        self._start_ai_response_message()
        self._set_ai_status("Thinking...", tone="info")
        self._llm_active_requested_class = str(requested_class or "").strip()
        self._llm_active_primary_target = primary_target if isinstance(primary_target, Target) else None
        worker = LLMWorker(
            config=self.llm_config,
            prompt=prompt,
            system_prompt=self._SYSTEM_PROMPT,
            tag=tag,
            parent=self,
        )
        worker.responseChunk.connect(self._on_ai_chunk)
        worker.responseReady.connect(self._on_ai_response)
        worker.errorOccurred.connect(self._on_ai_error)
        worker.finished.connect(self._on_ai_worker_finished)
        self._llm_worker = worker
        self._llm_active_tag = tag
        self._refresh_ai_warm_indicator()
        worker.start()

    @Slot(str, str)
    def _on_ai_chunk(self, tag: str, text: str) -> None:
        if not text:
            return
        if tag == "warmup":
            return
        self._set_ai_status("Streaming...", tone="info")
        self._append_ai_stream_chunk(text)

    @Slot(str, str)
    def _on_ai_response(self, tag: str, text: str) -> None:
        if tag == "warmup":
            self._llm_last_warmup_key = self._llm_warmup_cache_key()
            self._llm_last_warmup_at = perf_counter()
            self._set_ai_status("Ready", tone="info")
            return
        logger.info("LLM response received (tag=%s, length=%d)", tag, len(text))
        self._finalize_ai_response(text)

    @Slot(str)
    def _on_ai_error(self, message: str) -> None:
        if self._llm_active_tag == "warmup":
            logger.warning("LLM warm-up error: %s", message)
            if self._llm_warmup_silent:
                self._ai_runtime_status = ""
                self._refresh_ai_warm_indicator()
            else:
                self._set_ai_status("Warm-up failed", tone="warning")
            return
        logger.warning("LLM error: %s", message)
        self._fail_ai_response(message)

    @Slot()
    def _on_ai_worker_finished(self) -> None:
        active_tag = self._llm_active_tag
        if active_tag != "warmup":
            self._set_ai_status("Ready", tone="info")
        idx = self._ai_stream_message_index
        if idx is not None and 0 <= idx < len(self._ai_messages):
            if not self._ai_messages[idx].get("text", "").strip():
                self._ai_messages.pop(idx)
                self._render_ai_messages()
                self._persist_ai_messages_to_storage()
        self._ai_stream_message_index = None
        sender = self.sender()
        if sender is self._llm_worker:
            self._llm_worker = None
        if active_tag == "warmup":
            self._llm_warmup_silent = False
        self._llm_active_requested_class = ""
        self._llm_active_primary_target = None
        self._llm_active_tag = ""
        self._refresh_ai_warm_indicator()

    def _clear_ai_messages(self) -> None:
        self._ai_stream_render_timer.stop()
        self._ai_messages.clear()
        self._ai_message_widget_refs.clear()
        self._ai_stream_message_index = None
        if hasattr(self, "ai_output_layout"):
            self._render_ai_messages()
        self._refresh_ai_panel_action_buttons()
        self._persist_ai_messages_to_storage(allow_empty_clear=True)

    def _apply_ai_chat_font(self) -> None:
        if not hasattr(self, "ai_output"):
            return
        font = self.ai_output.font()
        font.setPointSize(int(getattr(self, "_llm_chat_font_size_pt", LLMConfig.DEFAULT_CHAT_FONT_PT)))
        self.ai_output.setFont(font)
        if hasattr(self, "ai_output_content"):
            self.ai_output_content.setFont(font)
        if getattr(self, "_ai_messages", None):
            self._render_ai_messages()

    def _last_ai_response_text(self) -> str:
        for message in reversed(self._ai_messages):
            if str(message.get("kind", "")) != "ai":
                continue
            text = str(message.get("text", "")).strip()
            if text:
                return text
        return ""

    def _build_ai_chat_transcript(self) -> str:
        role_map = {
            "user": "You",
            "ai": "LLM",
            "error": "Error",
            "info": "Info",
        }
        sections: list[str] = []
        for message in self._ai_messages:
            text = str(message.get("text", "")).strip()
            if not text:
                continue
            kind = str(message.get("kind", "info"))
            role = role_map.get(kind, kind.title() or "Info")
            sections.append(f"{role}\n{text}")
        return "\n\n".join(sections).strip()

    def _refresh_ai_panel_action_buttons(self) -> None:
        has_messages = any(str(message.get("text", "")).strip() for message in self._ai_messages)
        has_last_ai = bool(self._last_ai_response_text())
        if hasattr(self, "ai_copy_last_btn"):
            self.ai_copy_last_btn.setEnabled(has_last_ai)
        if hasattr(self, "ai_export_chat_btn"):
            self.ai_export_chat_btn.setEnabled(has_messages)

    def _visible_action_targets_for_message(self, message: dict[str, Any]) -> list[Target]:
        targets = message.get("action_targets")
        if not isinstance(targets, list):
            return []
        visible: list[Target] = []
        for target in targets:
            if not isinstance(target, Target):
                continue
            if self._plan_contains_target(target):
                continue
            visible.append(target)
        return visible

    def _add_ai_suggested_target(self, target: Target) -> None:
        added = self._append_target_to_plan(target, refresh=True, notify_duplicate=False)
        if added:
            self._set_ai_status(f"Added {target.name} to the plan.", tone="info")
        else:
            self._set_ai_status(f"{target.name} is already in the plan.", tone="warning")
        self._render_ai_messages()

    @Slot()
    def _copy_last_ai_response(self) -> None:
        text = self._last_ai_response_text()
        if not text:
            self._set_ai_status("No AI response to copy.", tone="warning")
            return
        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(text)
        self._set_ai_status("Copied last AI response.", tone="info")

    @Slot()
    def _export_ai_chat(self) -> None:
        transcript = self._build_ai_chat_transcript()
        if not transcript:
            self._set_ai_status("No AI chat to export.", tone="warning")
            return
        default_name = f"astroplanner-ai-chat-{datetime.now().strftime('%Y%m%d-%H%M%S')}.txt"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export AI Chat",
            str(Path.home() / default_name),
            "Text files (*.txt);;Markdown files (*.md);;All files (*)",
        )
        if not file_path:
            return
        try:
            Path(file_path).write_text(transcript + "\n", encoding="utf-8")
        except Exception as exc:
            QMessageBox.warning(self, "Export Failed", f"Could not export AI chat:\n{exc}")
            self._set_ai_status("AI chat export failed.", tone="warning")
            return
        self._set_ai_status(f"Exported AI chat to {Path(file_path).name}.", tone="info")

    @Slot()
    def _reset_ai_chat_appearance(self) -> None:
        self.settings.setValue("llm/chatFontSizePt", LLMConfig.DEFAULT_CHAT_FONT_PT)
        self.settings.setValue("llm/chatSpacing", LLMConfig.DEFAULT_CHAT_SPACING)
        self.settings.setValue("llm/chatTintStrength", LLMConfig.DEFAULT_CHAT_TINT_STRENGTH)
        self.settings.setValue("llm/chatMessageWidth", LLMConfig.DEFAULT_CHAT_WIDTH)
        self.settings.setValue("llm/statusErrorClearSec", LLMConfig.DEFAULT_STATUS_ERROR_CLEAR_S)
        self._llm_chat_font_size_pt = LLMConfig.DEFAULT_CHAT_FONT_PT
        self._ai_chat_spacing = LLMConfig.DEFAULT_CHAT_SPACING
        self._ai_chat_tint_strength = LLMConfig.DEFAULT_CHAT_TINT_STRENGTH
        self._ai_chat_width = LLMConfig.DEFAULT_CHAT_WIDTH
        self._ai_status_error_clear_s = LLMConfig.DEFAULT_STATUS_ERROR_CLEAR_S
        self._apply_ai_chat_font()
        self._render_ai_messages()
        self._set_ai_status("AI appearance reset to defaults.", tone="info")

    @staticmethod
    def _mix_qcolors(first: QColor, second: QColor, first_ratio: float) -> QColor:
        ratio = max(0.0, min(1.0, float(first_ratio)))
        other_ratio = 1.0 - ratio
        return QColor(
            int(round(first.red() * ratio + second.red() * other_ratio)),
            int(round(first.green() * ratio + second.green() * other_ratio)),
            int(round(first.blue() * ratio + second.blue() * other_ratio)),
        )

    @staticmethod
    def _qcolor_css(color: QColor) -> str:
        return color.name(QColor.HexRgb)

    @staticmethod
    def _qcolor_rgba_css(color: QColor, alpha: float) -> str:
        use_color = QColor(color)
        if not use_color.isValid():
            use_color = QColor("#59f3ff")
        return f"rgba({use_color.red()}, {use_color.green()}, {use_color.blue()}, {max(0.0, min(1.0, float(alpha))):.3f})"

    @staticmethod
    def _qcolor_rgba_mpl(color: QColor, alpha: float) -> tuple[float, float, float, float]:
        use_color = QColor(color)
        if not use_color.isValid():
            use_color = QColor("#59f3ff")
        return (
            float(use_color.redF()),
            float(use_color.greenF()),
            float(use_color.blueF()),
            max(0.0, min(1.0, float(alpha))),
        )

    @staticmethod
    def _format_ai_message_inline_markdown(text: str) -> str:
        rendered = str(text)
        rendered = re.sub(r"`([^`\n]+)`", r"<code>\1</code>", rendered)
        rendered = re.sub(r"\*\*([^*\n][\s\S]*?)\*\*", r"<strong>\1</strong>", rendered)
        rendered = re.sub(r"(?<!\*)\*([^*\n][\s\S]*?)\*(?!\*)", r"<em>\1</em>", rendered)
        return rendered

    def _render_ai_message_body(self, raw_text: str, *, muted_text: QColor) -> str:
        text = str(raw_text or "")
        if not text.strip():
            return f'<span style="color:{self._qcolor_css(muted_text)}"><i>...</i></span>'

        blocks: list[str] = []
        paragraph_lines: list[str] = []
        list_kind: Optional[str] = None
        list_items: list[str] = []

        def flush_paragraph() -> None:
            nonlocal paragraph_lines
            if not paragraph_lines:
                return
            joined = "<br>".join(
                self._format_ai_message_inline_markdown(html_module.escape(line))
                for line in paragraph_lines
            )
            blocks.append(f"<div>{joined}</div>")
            paragraph_lines = []

        def flush_list() -> None:
            nonlocal list_kind, list_items
            if not list_kind or not list_items:
                list_kind = None
                list_items = []
                return
            if list_kind == "ol":
                rows_html = "".join(
                    (
                        '<tr>'
                        f'<td valign="top" style="padding:0 8px 4px 0;white-space:nowrap;font-weight:600;">{idx}.</td>'
                        f'<td valign="top" style="padding:0 0 4px 0;">'
                        f'{self._format_ai_message_inline_markdown(html_module.escape(item))}'
                        "</td>"
                        "</tr>"
                    )
                    for idx, item in enumerate(list_items, start=1)
                )
                blocks.append(
                    '<table style="margin:4px 0 4px 0;border:none;border-collapse:collapse;">'
                    f"{rows_html}"
                    "</table>"
                )
            else:
                items_html = "".join(
                    f"<li>{self._format_ai_message_inline_markdown(html_module.escape(item))}</li>"
                    for item in list_items
                )
                blocks.append(
                    f'<ul style="margin:4px 0 4px 18px;padding:0;">{items_html}</ul>'
                )
            list_kind = None
            list_items = []

        for raw_line in text.splitlines():
            line = raw_line.rstrip()
            stripped = line.strip()
            if not stripped:
                flush_paragraph()
                flush_list()
                continue

            bullet_match = re.match(r"^\s*[-*]\s+(.+)$", line)
            ordered_match = re.match(r"^\s*\d+[.)]\s+(.+)$", line)
            if bullet_match:
                flush_paragraph()
                if list_kind not in (None, "ul"):
                    flush_list()
                list_kind = "ul"
                list_items.append(bullet_match.group(1).strip())
                continue
            if ordered_match:
                flush_paragraph()
                if list_kind not in (None, "ol"):
                    flush_list()
                list_kind = "ol"
                list_items.append(ordered_match.group(1).strip())
                continue

            flush_list()
            paragraph_lines.append(stripped)

        flush_paragraph()
        flush_list()
        return "".join(blocks)

    def _perform_ai_scroll_to_bottom(self) -> None:
        if not hasattr(self, "ai_output"):
            return
        if hasattr(self, "ai_output_content"):
            self.ai_output_content.adjustSize()
            layout = self.ai_output_content.layout()
            if layout is not None:
                layout.activate()
        scroll = self.ai_output.verticalScrollBar()
        if scroll is not None:
            scroll.setValue(scroll.maximum())

    def _scroll_ai_output_to_bottom(self, *, deferred: bool = True) -> None:
        if not hasattr(self, "ai_output"):
            return
        self._perform_ai_scroll_to_bottom()
        if not deferred:
            return
        if self._ai_scroll_bottom_pending:
            return
        self._ai_scroll_bottom_pending = True

        def _deferred_scroll() -> None:
            self._ai_scroll_bottom_pending = False
            self._perform_ai_scroll_to_bottom()
            QTimer.singleShot(0, self._perform_ai_scroll_to_bottom)

        QTimer.singleShot(0, _deferred_scroll)

    def _apply_ai_message_body_widget(
        self,
        body: QLabel,
        bubble: QFrame,
        *,
        body_html: str,
        font_pt: int,
        line_height: float,
        text_color: QColor,
        bubble_pad_h: int,
        bubble_max_width: int,
        stream_fixed_width: bool,
    ) -> None:
        body_rich_text = (
            f'<div style="color:{self._qcolor_css(text_color)};'
            f'font-size:{font_pt}pt;line-height:{line_height};">'
            f"{body_html}</div>"
        )
        body.setText(body_rich_text)
        max_body_width = max(150, bubble_max_width - (2 * bubble_pad_h))
        if stream_fixed_width:
            body_width = max_body_width
        else:
            doc = QTextDocument()
            doc.setDefaultFont(body.font())
            doc.setDocumentMargin(0)
            doc.setHtml(body_rich_text)
            body_width = max(150, min(max_body_width, int(math.ceil(doc.idealWidth())) + 2))
        body.setFixedWidth(body_width)
        bubble.setFixedWidth(body_width + (2 * bubble_pad_h))

    def _build_ai_message_widget(
        self,
        *,
        kind: str,
        body_html: str,
        show_headers: bool,
        font_pt: int,
        header_pt: int,
        outer_margin: int,
        header_margin: int,
        bubble_pad_h: int,
        bubble_pad_v: int,
        line_height: float,
        message_max_width_ratio: float,
        text_color: QColor,
        muted_text: QColor,
        label_color: QColor,
        border_color: QColor,
        background_color: QColor,
        header_icon_html: str,
        header_text: str,
        align_right: bool,
        streaming: bool = False,
        action_targets: Optional[list[Target]] = None,
    ) -> tuple[QWidget, dict[str, Any]]:
        viewport_width = self._ai_message_layout_width()
        bubble_max_width = max(240, int(viewport_width * message_max_width_ratio))

        row = QWidget(self.ai_output_content)
        row_l = QHBoxLayout(row)
        row_l.setContentsMargins(0, outer_margin, 0, outer_margin)
        row_l.setSpacing(0)

        stack = QWidget(row)
        stack_l = QVBoxLayout(stack)
        stack_l.setContentsMargins(0, 0, 0, 0)
        stack_l.setSpacing(header_margin)

        if show_headers:
            header = QLabel(stack)
            header.setTextFormat(Qt.RichText)
            header.setText(
                f'<span style="color:{self._qcolor_css(label_color)};'
                f'font-size:{header_pt}pt;font-weight:700;letter-spacing:0.8px;text-transform:uppercase;">'
                f'{header_icon_html}{html_module.escape(header_text)}</span>'
            )
            header.setTextInteractionFlags(Qt.NoTextInteraction)
            header.setStyleSheet("background:transparent;border:none;padding:0;margin:0;")
            stack_l.addWidget(header, 0, Qt.AlignRight if align_right else Qt.AlignLeft)

        bubble = QFrame(stack)
        bubble.setObjectName("AIChatBubble")
        bubble.setMaximumWidth(bubble_max_width)
        bubble.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        top_left = 14 if not align_right else 14
        top_right = 14 if align_right else 14
        bottom_left = 5 if not align_right else 14
        bottom_right = 5 if align_right else 14
        bubble.setStyleSheet(
            "QFrame#AIChatBubble {"
            f"background:{self._qcolor_rgba_css(background_color, background_color.alphaF())};"
            f"border:1px solid {self._qcolor_css(border_color)};"
            f"border-top-left-radius:{top_left}px;"
            f"border-top-right-radius:{top_right}px;"
            f"border-bottom-left-radius:{bottom_left}px;"
            f"border-bottom-right-radius:{bottom_right}px;"
            "}"
        )
        shadow = QGraphicsDropShadowEffect(bubble)
        shadow.setBlurRadius(18)
        shadow.setOffset(0, 6)
        shadow.setColor(QColor(border_color.red(), border_color.green(), border_color.blue(), 42))
        bubble.setGraphicsEffect(shadow)

        bubble_l = QVBoxLayout(bubble)
        bubble_l.setContentsMargins(bubble_pad_h, bubble_pad_v, bubble_pad_h, bubble_pad_v)
        bubble_l.setSpacing(0)

        body = QLabel(bubble)
        body.setWordWrap(True)
        body.setTextFormat(Qt.RichText)
        body.setTextInteractionFlags(Qt.TextSelectableByMouse)
        body.setStyleSheet(
            "background:transparent;border:none;padding:0;margin:0;"
            f"color:{self._qcolor_css(text_color)};"
        )
        self._apply_ai_message_body_widget(
            body,
            bubble,
            body_html=body_html,
            font_pt=font_pt,
            line_height=line_height,
            text_color=text_color,
            bubble_pad_h=bubble_pad_h,
            bubble_max_width=bubble_max_width,
            stream_fixed_width=streaming,
        )
        bubble_l.addWidget(body)

        if action_targets:
            action_box = QWidget(bubble)
            action_box_l = QVBoxLayout(action_box)
            action_box_l.setContentsMargins(0, 8, 0, 0)
            action_box_l.setSpacing(6)
            for target in action_targets:
                add_btn = QPushButton(f"Add {target.name}", action_box)
                _set_button_variant(add_btn, "ghost")
                _set_button_icon_kind(add_btn, "add")
                add_btn.clicked.connect(lambda _checked=False, t=target: self._add_ai_suggested_target(t))
                action_box_l.addWidget(add_btn, 0, Qt.AlignLeft)
            bubble_l.addWidget(action_box, 0, Qt.AlignLeft)

        stack_l.addWidget(bubble, 0, Qt.AlignRight if align_right else Qt.AlignLeft)

        if align_right:
            row_l.addStretch(1)
            row_l.addWidget(stack, 0, Qt.AlignRight)
        else:
            row_l.addWidget(stack, 0, Qt.AlignLeft)
            row_l.addStretch(1)
        return row, {
            "row": row,
            "bubble": bubble,
            "body": body,
            "font_pt": font_pt,
            "line_height": line_height,
            "text_color": QColor(text_color),
            "muted_text": QColor(muted_text),
            "bubble_pad_h": bubble_pad_h,
            "bubble_max_width": bubble_max_width,
            "streaming": streaming,
        }

    def _ai_message_layout_width(self) -> int:
        if not hasattr(self, "ai_output"):
            return 320
        viewport_width = 0
        try:
            viewport_width = int(self.ai_output.viewport().width())
        except Exception:
            viewport_width = int(self.ai_output.width())
        reserve = 0
        scroll = self.ai_output.verticalScrollBar() if hasattr(self.ai_output, "verticalScrollBar") else None
        if scroll is not None and not scroll.isVisible():
            try:
                reserve = max(0, int(scroll.sizeHint().width()))
            except Exception:
                reserve = 14
        return max(320, viewport_width - reserve)

    def _update_ai_message_widget(self, idx: int, *, streaming: bool) -> bool:
        if not (0 <= idx < len(self._ai_messages)) or not (0 <= idx < len(self._ai_message_widget_refs)):
            return False
        refs = self._ai_message_widget_refs[idx]
        body = refs.get("body")
        bubble = refs.get("bubble")
        if not isinstance(body, QLabel) or not isinstance(bubble, QFrame):
            return False
        message = self._ai_messages[idx]
        body_html = self._render_ai_message_body(str(message.get("text", "")), muted_text=QColor(refs["muted_text"]))
        self._apply_ai_message_body_widget(
            body,
            bubble,
            body_html=body_html,
            font_pt=int(refs["font_pt"]),
            line_height=float(refs["line_height"]),
            text_color=QColor(refs["text_color"]),
            bubble_pad_h=int(refs["bubble_pad_h"]),
            bubble_max_width=int(refs["bubble_max_width"]),
            stream_fixed_width=streaming,
        )
        refs["streaming"] = streaming
        body.updateGeometry()
        bubble.updateGeometry()
        row = refs.get("row")
        if isinstance(row, QWidget):
            row.updateGeometry()
        self.ai_output_content.updateGeometry()
        return True

    def _render_ai_messages(self) -> None:
        if not hasattr(self, "ai_output"):
            return
        font_pt = max(9, int(getattr(self, "_llm_chat_font_size_pt", LLMConfig.DEFAULT_CHAT_FONT_PT)))
        header_pt = max(8, font_pt - 2)
        spacing_mode = str(getattr(self, "_ai_chat_spacing", LLMConfig.DEFAULT_CHAT_SPACING) or LLMConfig.DEFAULT_CHAT_SPACING).strip().lower()
        tint_mode = str(getattr(self, "_ai_chat_tint_strength", LLMConfig.DEFAULT_CHAT_TINT_STRENGTH) or LLMConfig.DEFAULT_CHAT_TINT_STRENGTH).strip().lower()
        width_mode = str(getattr(self, "_ai_chat_width", LLMConfig.DEFAULT_CHAT_WIDTH) or LLMConfig.DEFAULT_CHAT_WIDTH).strip().lower()
        show_headers = True
        if spacing_mode == "compact":
            outer_margin = 4
            header_margin = 2
            bubble_pad_v = 4
            bubble_pad_h = 7
            line_height = 1.28
        else:
            outer_margin = 6
            header_margin = 3
            bubble_pad_v = 6
            bubble_pad_h = 9
            line_height = 1.35
        tint_scale_map = {
            "low": 0.75,
            "medium": 1.10,
            "high": 1.45,
        }
        tint_scale = float(tint_scale_map.get(tint_mode, 1.00))
        max_width_map = {
            "narrow": 0.34,
            "normal": 0.48,
            "wide": 0.62,
        }
        message_max_width_ratio = float(max_width_map.get(width_mode, 0.76))
        palette = self.ai_output.palette()
        base = palette.color(QPalette.Base)
        text_color = palette.color(QPalette.Text)
        highlight = palette.color(QPalette.Highlight)
        accent_red = self._theme_qcolor("state_error", "#dc2626")
        accent_primary = self._theme_qcolor("accent_primary", "#59f3ff")
        accent_secondary = self._theme_qcolor("accent_secondary", "#ff4fd8")
        accent_secondary_soft = self._theme_qcolor("accent_secondary_soft", "#d38cff")

        user_border = self._mix_qcolors(accent_primary, highlight, 0.60)
        ai_border = self._mix_qcolors(accent_secondary, accent_secondary_soft, 0.58)
        error_border = self._mix_qcolors(accent_red, base, 0.38)
        muted_text = self._mix_qcolors(text_color, base, 0.42)
        user_bg = self._mix_qcolors(accent_primary, base, 0.14)
        ai_bg = self._mix_qcolors(accent_secondary_soft, base, 0.10)
        error_bg = self._mix_qcolors(accent_red, base, 0.08)
        user_label = self._mix_qcolors(accent_primary, text_color, 0.82)
        ai_label = self._mix_qcolors(accent_secondary, text_color, 0.76)
        error_label = self._mix_qcolors(accent_red, text_color, 0.80)
        self.ai_output.setUpdatesEnabled(False)
        try:
            while self.ai_output_layout.count():
                item = self.ai_output_layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
            self._ai_message_widget_refs = []

            if not self._ai_messages:
                placeholder = QLabel(self._ai_output_placeholder_text, self.ai_output_content)
                placeholder.setObjectName("SectionHint")
                placeholder.setWordWrap(True)
                placeholder.setTextInteractionFlags(Qt.NoTextInteraction)
                self.ai_output_layout.addWidget(placeholder, 0, Qt.AlignTop)
                self.ai_output_layout.addStretch(1)
                self._refresh_ai_panel_action_buttons()
                return

            for idx, message in enumerate(self._ai_messages):
                kind = message.get("kind", "ai")
                raw_text = str(message.get("text", ""))
                body_html = self._render_ai_message_body(raw_text, muted_text=muted_text)
                is_streaming_message = bool(
                    kind == "ai"
                    and self._ai_stream_message_index is not None
                    and idx == self._ai_stream_message_index
                    and self._llm_active_tag not in {"", "warmup"}
                )
                if kind == "user":
                    widget, refs = self._build_ai_message_widget(
                        kind="user",
                        body_html=body_html,
                        show_headers=show_headers,
                        font_pt=font_pt,
                        header_pt=header_pt,
                        outer_margin=outer_margin,
                        header_margin=header_margin,
                        bubble_pad_h=bubble_pad_h,
                        bubble_pad_v=bubble_pad_v,
                        line_height=line_height,
                        message_max_width_ratio=message_max_width_ratio,
                        text_color=text_color,
                        muted_text=muted_text,
                        label_color=user_label,
                        border_color=user_border,
                        background_color=QColor.fromRgbF(user_bg.redF(), user_bg.greenF(), user_bg.blueF(), min(0.98, 0.96 * tint_scale)),
                        header_icon_html=(
                            f'<span style="display:inline-block;min-width:16px;height:16px;line-height:16px;'
                            f'padding:0 5px;margin:0 5px 0 0;text-align:center;border-radius:999px;'
                            f'background:{self._qcolor_rgba_css(user_label, 0.16)};'
                            f'border:1px solid {self._qcolor_css(user_border)};'
                            f'color:{self._qcolor_css(user_label)}">&#8250;</span>'
                        ),
                        header_text="You",
                        align_right=True,
                        streaming=False,
                    )
                elif kind == "error":
                    widget, refs = self._build_ai_message_widget(
                        kind="error",
                        body_html=body_html,
                        show_headers=show_headers,
                        font_pt=font_pt,
                        header_pt=header_pt,
                        outer_margin=outer_margin,
                        header_margin=header_margin,
                        bubble_pad_h=bubble_pad_h,
                        bubble_pad_v=bubble_pad_v,
                        line_height=line_height,
                        message_max_width_ratio=message_max_width_ratio,
                        text_color=text_color,
                        muted_text=muted_text,
                        label_color=error_label,
                        border_color=error_border,
                        background_color=QColor.fromRgbF(error_bg.redF(), error_bg.greenF(), error_bg.blueF(), min(0.98, 1.00 * tint_scale)),
                        header_icon_html=(
                            f'<span style="display:inline-block;min-width:16px;height:16px;line-height:16px;'
                            f'padding:0 5px;margin:0 5px 0 0;text-align:center;border-radius:999px;'
                            f'background:{self._qcolor_rgba_css(error_label, 0.16)};'
                            f'border:1px solid {self._qcolor_css(error_border)};'
                            f'color:{self._qcolor_css(error_label)}">!</span>'
                        ),
                        header_text="Error",
                        align_right=False,
                        streaming=False,
                    )
                else:
                    action_targets = [] if is_streaming_message else self._visible_action_targets_for_message(message)
                    widget, refs = self._build_ai_message_widget(
                        kind="ai",
                        body_html=body_html,
                        show_headers=show_headers,
                        font_pt=font_pt,
                        header_pt=header_pt,
                        outer_margin=outer_margin,
                        header_margin=header_margin,
                        bubble_pad_h=bubble_pad_h,
                        bubble_pad_v=bubble_pad_v,
                        line_height=line_height,
                        message_max_width_ratio=message_max_width_ratio,
                        text_color=text_color,
                        muted_text=muted_text,
                        label_color=ai_label,
                        border_color=ai_border,
                        background_color=QColor.fromRgbF(ai_bg.redF(), ai_bg.greenF(), ai_bg.blueF(), min(0.98, 0.98 * tint_scale)),
                        header_icon_html=(
                            f'<span style="display:inline-block;min-width:16px;height:16px;line-height:16px;'
                            f'padding:0 5px;margin:0 5px 0 0;text-align:center;border-radius:999px;'
                            f'background:{self._qcolor_rgba_css(ai_label, 0.16)};'
                            f'border:1px solid {self._qcolor_css(ai_border)};'
                            f'color:{self._qcolor_css(ai_label)}">&#10022;</span>'
                        ),
                        header_text="LLM",
                        align_right=False,
                        streaming=is_streaming_message,
                        action_targets=action_targets,
                    )
                self.ai_output_layout.addWidget(widget, 0, Qt.AlignTop)
                self._ai_message_widget_refs.append(refs)
            self.ai_output_layout.addStretch(1)
        finally:
            self.ai_output.setUpdatesEnabled(True)

        self._scroll_ai_output_to_bottom(deferred=True)
        self._refresh_ai_panel_action_buttons()

    def _flush_ai_stream_render(self) -> None:
        idx = self._ai_stream_message_index
        if idx is None:
            return
        if not self._update_ai_message_widget(idx, streaming=True):
            self._render_ai_messages()
            return
        self._scroll_ai_output_to_bottom(deferred=False)
        self._refresh_ai_panel_action_buttons()

    def _start_ai_response_message(self) -> None:
        self._ai_stream_render_timer.stop()
        self._ai_stream_message_index = self._append_ai_message("", is_ai=True)

    def _append_ai_stream_chunk(self, text: str) -> None:
        if not text:
            return
        idx = self._ai_stream_message_index
        if idx is None or not (0 <= idx < len(self._ai_messages)):
            idx = self._append_ai_message("", is_ai=True)
            self._ai_stream_message_index = idx
        self._ai_messages[idx]["text"] = self._ai_messages[idx].get("text", "") + text
        if not self._ai_stream_render_timer.isActive():
            self._ai_stream_render_timer.start(45)

    def _finalize_ai_response(self, text: str) -> None:
        self._ai_stream_render_timer.stop()
        requested_class = str(getattr(self, "_llm_active_requested_class", "") or "").strip()
        primary_target = getattr(self, "_llm_active_primary_target", None)
        idx = self._ai_stream_message_index
        if idx is not None and 0 <= idx < len(self._ai_messages):
            streamed_text = str(self._ai_messages[idx].get("text", "") or "")
            final_text = str(text or "").strip() or streamed_text
            action_targets = self._extract_addable_bhtom_targets_from_ai_text(final_text)
            if not action_targets and streamed_text and streamed_text != final_text:
                action_targets = self._extract_addable_bhtom_targets_from_ai_text(streamed_text)
            self._ai_messages[idx]["kind"] = "ai"
            self._ai_messages[idx]["text"] = final_text
            self._ai_messages[idx]["action_targets"] = action_targets
            if requested_class:
                self._ai_messages[idx]["requested_class"] = requested_class
            if isinstance(primary_target, Target):
                self._ai_messages[idx]["primary_target"] = primary_target
            if action_targets:
                self._ai_messages[idx]["suggested_targets"] = list(action_targets)
            self._ai_stream_message_index = None
            self._render_ai_messages()
        else:
            final_text = str(text or "").strip()
            self._append_ai_message(
                final_text,
                is_ai=True,
                action_targets=self._extract_addable_bhtom_targets_from_ai_text(final_text),
                requested_class=requested_class,
                primary_target=primary_target if isinstance(primary_target, Target) else None,
            )
        self._ai_stream_message_index = None
        self._persist_ai_messages_to_storage()

    def _fail_ai_response(self, message: str) -> None:
        self._ai_stream_render_timer.stop()
        idx = self._ai_stream_message_index
        if idx is not None and 0 <= idx < len(self._ai_messages):
            current_text = self._ai_messages[idx].get("text", "")
            if current_text.strip():
                self._append_ai_message(message, is_error=True)
            else:
                self._ai_messages[idx]["kind"] = "error"
                self._ai_messages[idx]["text"] = message
                self._render_ai_messages()
        else:
            self._append_ai_message(message, is_error=True)
        self._ai_stream_message_index = None
        self._persist_ai_messages_to_storage()

    def _append_ai_message(
        self,
        text: str,
        *,
        is_user: bool = False,
        is_ai: bool = False,
        is_error: bool = False,
        action_targets: Optional[list[Target]] = None,
        requested_class: str = "",
        primary_target: Optional[Target] = None,
        suggested_targets: Optional[list[Target]] = None,
    ) -> int:
        kind = "user" if is_user else "error" if is_error else "ai" if is_ai else "info"
        payload: dict[str, Any] = {
            "kind": kind,
            "text": str(text),
            "created_at": datetime.now(timezone.utc).timestamp(),
        }
        if action_targets:
            payload["action_targets"] = list(action_targets)
        requested_class_value = str(requested_class or "").strip()
        if requested_class_value:
            payload["requested_class"] = requested_class_value
        if isinstance(primary_target, Target):
            payload["primary_target"] = primary_target
        if suggested_targets:
            payload["suggested_targets"] = [target for target in suggested_targets if isinstance(target, Target)]
        self._ai_messages.append(payload)
        self._render_ai_messages()
        if not is_ai or self._llm_active_tag in {"", "warmup"}:
            self._persist_ai_messages_to_storage()
        return len(self._ai_messages) - 1

    @Slot(int)
    def _change_date(self, offset_days: int):
        """Shift the selected date by the given number of days and re-plot."""
        new_date = self.date_edit.date().addDays(offset_days)
        self.date_edit.setDate(new_date)
        self._schedule_plan_autosave()
        self._replot_timer.start()

    @Slot()
    def _change_to_today(self):
        """
        Reset date picker to the current observing night: use previous calendar date before local noon.
        This ensures pressing 'Today' after midnight and before noon still shows the previous night's date.
        """
        # Determine local timezone (site or system)
        if self.table_model.site:
            tz = pytz.timezone(self.table_model.site.timezone_name)
        else:
            tz = datetime.now().astimezone().tzinfo
        now_local = datetime.now(tz)
        # If before noon local time, use yesterday; otherwise use today
        local_qdate = QDate(now_local.year, now_local.month, now_local.day)
        if now_local.hour < 12:
            new_date = local_qdate.addDays(-1)
        else:
            new_date = local_qdate
        self.date_edit.setDate(new_date)
        self._schedule_plan_autosave()
        self._replot_timer.start()

    @Slot()
    def _update_limit(self):
        """Update limit altitude, refresh table warnings, and replot."""
        # Update table model limit so coloring reflects the new threshold
        new_limit = float(self.limit_spin.value())
        self.table_model.limit = new_limit
        self._emit_table_data_changed()
        self._schedule_plan_autosave()
        # Replot visibility with updated limit
        if self.last_payload is not None:
            self._update_plot(self.last_payload)
        else:
            self._replot_timer.start()

# --------------------------------------------------
# --- Main entry -----------------------------------
# --------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Astronomical Observation Planner")
    parser.add_argument('--plan', '-p', help='Path to JSON plan file to load and plot on startup')
    args = parser.parse_args()

    app = QApplication(sys.argv)
    win = MainWindow()

    # If a plan file is specified, import it into the workspace and immediately plot.
    if args.plan:
        try:
            win._load_plan_from_json_path(args.plan, persist_workspace=True)
        except Exception as e:
            QMessageBox.critical(None, "Startup Load Error", f"Failed to load plan '{args.plan}': {e}")

    win.show()
    sys.exit(app.exec())
