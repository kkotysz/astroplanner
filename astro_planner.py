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
import csv
import html as html_module
import io
import json
import math
import os
import sys
import threading
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, List, Optional
from time import perf_counter

import numpy as np
import pytz
import ephem
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord, Angle
from astropy.time import Time
from astroplan import FixedTarget, Observer
from astroplan.plots import plot_finder_image
from astroplan.observer import TargetAlwaysUpWarning

from astroquery.simbad import Simbad
try:
    from astroquery.exceptions import NoResultsWarning
except Exception:  # pragma: no cover - fallback only for older astroquery variants
    class NoResultsWarning(Warning):
        pass
from pydantic import BaseModel, Field, ValidationError, ConfigDict
from timezonefinder import TimezoneFinder
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import font_manager as mpl_font_manager
from matplotlib.figure import Figure
import warnings
import logging
import shiboken6 as shb   # PySide6 helper (isValid,objId)
from astroplanner.exporters import export_metrics_csv, export_observation_ics
from astroplanner.parsing import parse_dec_to_deg, parse_ra_dec_query, parse_ra_to_deg
from astroplanner.scoring import TargetNightMetrics, compute_target_metrics
from astroplanner.theme import (
    DEFAULT_UI_THEME,
    THEME_CHOICES,
    COLORBLIND_HIGHLIGHT,
    COLORBLIND_LINE_COLORS,
    DEFAULT_HIGHLIGHT,
    DEFAULT_LINE_COLORS,
    build_stylesheet,
    normalize_theme_key,
)

# ── Logging configuration ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,                              # default level
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
_TZ_FINDER = TimezoneFinder()
_FINDER_PATCH_LOCK = threading.Lock()

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
FINDER_HTTP_TIMEOUT_S = 5.0
FINDER_WORKER_TIMEOUT_MS = 10000
FINDER_RETRY_COOLDOWN_S = 45.0
BHTOM_API_BASE_URL = "https://bh-tom2.astrouw.edu.pl"
BHTOM_TARGET_LIST_PATH = "/targets/getTargetList/"
BHTOM_MAX_SUGGESTION_PAGES = 5
BHTOM_PAGE_SIZE = 200
BHTOM_SUGGESTION_MIN_IMPORTANCE = 2.0
BHTOM_SUGGESTION_CACHE_TTL_S = 60 * 60


@dataclass(frozen=True)
class CalcRunStats:
    duration_s: float
    visible_targets: int
    total_targets: int


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


def _finder_survey_candidates(key: object) -> list[str]:
    norm = _normalize_cutout_survey_key(key)
    mapping: dict[str, list[str]] = {
        "dss2": ["DSS2 Red", "DSS", "DSS1 Red", "DSS2 IR", "2MASS-K"],
        "panstarrs": ["PanSTARRS g", "DSS2 Red", "DSS", "2MASS-K"],
        "2mass": ["2MASS-K", "DSS2 Red", "DSS"],
    }
    return mapping.get(norm, ["DSS"])


def _plot_finder_image_compat(
    target: FixedTarget,
    survey: str,
    fov_radius,
    ax,
    width_px: int,
    height_px: int,
) -> None:
    from astroquery.skyview import SkyView

    kwargs = {
        "survey": survey,
        "fov_radius": fov_radius,
        "ax": ax,
        "grid": False,
        "reticle": False,
        "style_kwargs": {"cmap": "Greys", "origin": "lower"},
    }
    with _FINDER_PATCH_LOCK:
        original_request = SkyView._request
        original_url = str(getattr(SkyView, "URL", "") or "")

        if original_url.startswith("http://"):
            # Some networks reset plain HTTP to SkyView; HTTPS is much more stable.
            SkyView.URL = "https://" + original_url[len("http://"):]

        def _request_with_timeout(method, url, **inner_kwargs):
            if inner_kwargs.get("timeout") in (None, 0):
                inner_kwargs["timeout"] = FINDER_HTTP_TIMEOUT_S
            return original_request(method, url, **inner_kwargs)

        SkyView._request = _request_with_timeout
        original = SkyView.get_images

        def _compat_get_images(*args, **inner_kwargs):
            inner_kwargs.pop("grid", None)
            if "show_progress" not in inner_kwargs:
                inner_kwargs["show_progress"] = False
            radius = inner_kwargs.pop("radius", None)
            if radius is not None:
                ratio = max(0.4, min(2.5, float(width_px) / max(1.0, float(height_px))))
                diameter = 2.0 * radius
                if ratio >= 1.0:
                    inner_kwargs.setdefault("height", diameter)
                    inner_kwargs.setdefault("width", diameter * ratio)
                else:
                    inner_kwargs.setdefault("width", diameter)
                    inner_kwargs.setdefault("height", diameter / ratio)
            if inner_kwargs.get("pixels") in (None, 0, ""):
                inner_kwargs["pixels"] = f"{max(64, int(width_px))},{max(64, int(height_px))}"
            return original(*args, **inner_kwargs)

        SkyView.get_images = _compat_get_images
        try:
            try:
                plot_finder_image(target, **kwargs)
                return
            except IndexError as exc:
                # astroplan blindly indexes SkyView.get_images()[0][0].
                # SkyView occasionally returns an empty list for a field/survey pair.
                raise LookupError(f"SkyView returned no image for survey '{survey}'") from exc
            except TypeError as exc:
                # astroplan<->astroquery API mismatch: newer astroquery dropped "grid"
                if "grid" not in str(exc).lower():
                    raise
                try:
                    plot_finder_image(target, **kwargs)
                    return
                except IndexError as inner_exc:
                    raise LookupError(f"SkyView returned no image for survey '{survey}'") from inner_exc
        finally:
            SkyView.get_images = original
            SkyView._request = original_request
            if original_url:
                SkyView.URL = original_url


def _render_finder_chart_png_bytes(
    name: str,
    ra_deg: float,
    dec_deg: float,
    survey_key: str,
    fov_arcmin: int,
    width_px: int,
    height_px: int,
) -> bytes:
    width = max(168, min(int(width_px), 1400))
    height = max(168, min(int(height_px), 1400))
    dpi = 100.0
    inches_w = max(1.6, float(width) / dpi)
    inches_h = max(1.6, float(height) / dpi)
    fixed = FixedTarget(
        name=name.strip() or "Target",
        coord=SkyCoord(ra=float(ra_deg) * u.deg, dec=float(dec_deg) * u.deg),
    )
    fov_radius = max(1.0, float(fov_arcmin) / 2.0) * u.arcmin
    failures: list[str] = []

    for survey in _finder_survey_candidates(survey_key):
        fig = Figure(figsize=(inches_w, inches_h), dpi=dpi)
        ax = fig.add_subplot(111)
        try:
            _plot_finder_image_compat(fixed, survey, fov_radius, ax, width, height)
            # Fill the whole image area (no matplotlib frame margins),
            # so finder chart occupies the same visual size as Aladin cutout.
            fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
            ax.set_position([0.0, 0.0, 1.0, 1.0])
            ax.set_title("")
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            canvas = FigureCanvasAgg(fig)
            buf = io.BytesIO()
            canvas.print_png(buf)
            payload = buf.getvalue()
            if payload:
                return payload
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{survey}: {exc}")
        finally:
            try:
                plt.close(fig)
            except Exception:
                pass

    if failures:
        raise RuntimeError(" / ".join(failures))
    raise RuntimeError("Finder chart unavailable")


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
    QAbstractTableModel,
    QDate,
    QEvent,
    QModelIndex,
    QSignalBlocker,
    Qt,
    QThread,
    QMutex, QWaitCondition,
    Signal,
    Slot,
    QSize,
    QTimer,
    QCoreApplication,
    QItemSelectionModel,
    QSettings,
    QUrl,
    QUrlQuery,
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
    QPalette,
    QPainter,
    QPen,
    QPixmap,
    QShortcut,
)
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkReply, QNetworkRequest
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDoubleSpinBox,
    QFrame,
    QFormLayout,
    QHeaderView,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QProgressDialog,
    QScrollArea,
    QSplitter,
    QSpinBox,
    QSlider,
    QStyle,
    QSizePolicy,
    QTableView,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QColorDialog,
    QDialog,
    QDialogButtonBox,
    QTextEdit,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_agg import FigureCanvasAgg

# --- Custom delegate to preserve model background for column 0 even when selected ---
from PySide6.QtWidgets import QStyledItemDelegate, QStyle

# --- Custom QTableView to manage delete shortcut (macOS only) ----------
class TargetTableView(QTableView):
    """
    Emit `deleteRequested` when ⌘+Backspace is pressed, and suppress
    Ctrl+Backspace so that only the macOS-native shortcut deletes rows.
    """
    deleteRequested = Signal()

    def keyPressEvent(self, event):
        """
        Only a *pure* ⌘‑Backspace / ⌘‑Delete on macOS (or Ctrl‑Backspace / Ctrl‑Delete
        on other platforms) triggers row deletion.  All other Backspace/Delete
        combinations are swallowed so Qt’s default handler never sees them.
        """
        key = event.key()
        # We care only about the two physical keys that map to “erase” on macOS
        if key in (Qt.Key_Backspace, Qt.Key_Delete):
            mods = event.modifiers()
            # On macOS the Command (⌘) key is reported as Qt.ControlModifier,
            # while the physical Control key is Qt.MetaModifier.  Everywhere
            # else Ctrl is Qt.ControlModifier.
            desired_mod = Qt.ControlModifier if sys.platform == "darwin" else Qt.ControlModifier
            if mods == desired_mod:
                self.deleteRequested.emit()
            # Swallow the event in every case so nothing else processes it
            return
        # All other keys → default processing
        super().keyPressEvent(event)


class CoverImageLabel(QLabel):
    """Render pixmap preserving aspect ratio, centered in available space."""

    resized = Signal(int, int)

    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self._source_pixmap = QPixmap()

    def setPixmap(self, pixmap):  # type: ignore[override]
        if pixmap is None or pixmap.isNull():
            self._source_pixmap = QPixmap()
            super().setPixmap(QPixmap())
            return
        self._source_pixmap = pixmap.copy()
        self._apply_cover_pixmap()

    def setText(self, text: str):  # type: ignore[override]
        self._source_pixmap = QPixmap()
        super().setText(text)

    def resizeEvent(self, event):  # noqa: D401
        super().resizeEvent(event)
        if not self._source_pixmap.isNull():
            self._apply_cover_pixmap()
        self.resized.emit(self.width(), self.height())

    def _apply_cover_pixmap(self):
        if self._source_pixmap.isNull():
            super().setPixmap(QPixmap())
            return
        w = max(1, self.width())
        h = max(1, self.height())
        scaled = self._source_pixmap.scaled(
            w,
            h,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        canvas = QPixmap(w, h)
        canvas.fill(Qt.GlobalColor.transparent)
        painter = QPainter(canvas)
        x = (w - scaled.width()) // 2
        y = (h - scaled.height()) // 2
        painter.drawPixmap(x, y, scaled)
        painter.end()
        super().setPixmap(canvas)


def _normalized_css_color(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    color = QColor(text)
    if not color.isValid():
        return ""
    return color.name().lower()


class NoSelectBackgroundDelegate(QStyledItemDelegate):
    """Delegate that preserves model background for the target-name column when selected."""
    def paint(self, painter, option, index):
        # Disable the selected state to preserve background
        if option.state & QStyle.State_Selected:
            option.state &= ~QStyle.State_Selected
        super().paint(painter, option, index)

# --- Table Settings Dialog ---
class TableSettingsDialog(QDialog):
    """Dialog to configure table parameters."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName(self.__class__.__name__)
        self.setWindowTitle("Table Settings")
        layout = QFormLayout(self)
        # Row height
        self.row_height_spin = QSpinBox(self)
        self.row_height_spin.setRange(10, 100)
        init_h = parent.settings.value("table/rowHeight", 24, type=int) if parent and hasattr(parent, "settings") else 24
        self.row_height_spin.setValue(init_h)
        layout.addRow("Row height:", self.row_height_spin)
        # Name-column width
        self.first_col_width_spin = QSpinBox(self)
        self.first_col_width_spin.setRange(50, 500)
        init_w = parent.settings.value("table/firstColumnWidth", 100, type=int) if parent and hasattr(parent, "settings") else 100
        self.first_col_width_spin.setValue(init_w)
        layout.addRow("Name column width:", self.first_col_width_spin)

        # Font size
        self.font_spin = QSpinBox(self)
        self.font_spin.setRange(8, 16)
        init_fs = parent.settings.value("table/fontSize", 11, type=int) if parent and hasattr(parent, "settings") else 11
        self.font_spin.setValue(init_fs)
        layout.addRow("Font size:", self.font_spin)

        # Column visibility
        self.col_checks = {}
        if parent is not None and hasattr(parent, "table_model"):
            for idx, lbl in enumerate(parent.table_model.headers[:-1]):
                chk = QCheckBox(lbl, self)
                val = parent.settings.value(f"table/col{idx}", True, type=bool) if parent and hasattr(parent, "settings") else True
                chk.setChecked(val)
                layout.addRow(f"Show {lbl}:", chk)
                self.col_checks[idx] = chk

        # Default sort column
        self.sort_combo = QComboBox(self)
        # Populate with column headers
        headers = parent.table_model.headers if parent and hasattr(parent, "table_model") else []
        self.sort_combo.addItems(headers)
        init_sort = (
            parent.settings.value("table/defaultSortColumn", TargetTableModel.COL_SCORE, type=int)
            if parent and hasattr(parent, "settings")
            else TargetTableModel.COL_SCORE
        )
        # Clamp to valid range
        if 0 <= init_sort < len(headers):
            self.sort_combo.setCurrentIndex(init_sort)
        layout.addRow("Default sort column:", self.sort_combo)

        # Highlight colors
        default_colors = {"below":"#ff8080","limit":"#ffff80","above":"#b3ffb3"}
        self.selected_colors: dict[str, str] = {}
        def _pick_color(key, btn):
            col = QColorDialog.getColor(
                QColor(self.selected_colors.get(key, default_colors[key])),
                self,
                f"Pick {key} color",
            )
            if col.isValid():
                btn.setStyleSheet(f"background:{col.name()}")
                self.selected_colors[key] = col.name()
        for key in ("below","limit","above"):
            btn = QPushButton(f"{key.capitalize()} highlight", self)
            init = parent.settings.value(f"table/color/{key}", default_colors[key]) if parent and hasattr(parent, "settings") else default_colors[key]
            self.selected_colors[key] = str(init)
            btn.setStyleSheet(f"background:{init}")
            btn.clicked.connect(lambda _,k=key,b=btn: _pick_color(k,b))
            layout.addRow(f"{key.capitalize()} color:", btn)
            setattr(self, f"{key}_btn", btn)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.Apply, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        apply_btn = buttons.button(QDialogButtonBox.Apply)
        if apply_btn is not None:
            apply_btn.clicked.connect(self._apply_changes)
        layout.addWidget(buttons)
        self.setMinimumWidth(max(420, self.sizeHint().width()))

    def _apply_changes(self):
        s = self.parent().settings
        s.setValue("table/rowHeight", self.row_height_spin.value())
        s.setValue("table/firstColumnWidth", self.first_col_width_spin.value())
        s.setValue("table/fontSize", self.font_spin.value())
        for idx, chk in self.col_checks.items():
            s.setValue(f"table/col{idx}", chk.isChecked())
        for key in ("below","limit","above"):
            s.setValue(f"table/color/{key}", self.selected_colors.get(key))
        # Save default sort column
        s.setValue("table/defaultSortColumn", self.sort_combo.currentIndex())
        self.parent()._apply_table_settings()

    def accept(self):
        self._apply_changes()
        super().accept()

# --- General Settings Dialog ---
class GeneralSettingsDialog(QDialog):
    """Configure default site, date, samples & clock refresh."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName(self.__class__.__name__)
        self.setWindowTitle("General Settings")
        self._original_dark_mode = (
            parent.settings.value("general/darkMode", False, type=bool)
            if parent and hasattr(parent, "settings")
            else False
        )
        root_layout = QVBoxLayout(self)
        tabs = QTabWidget(self)
        tabs.setDocumentMode(True)
        root_layout.addWidget(tabs)

        def _make_tab(title: str) -> QFormLayout:
            page = QWidget(self)
            form = QFormLayout(page)
            form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
            form.setContentsMargins(12, 12, 12, 12)
            form.setSpacing(10)
            tabs.addTab(page, title)
            return form

        general_layout = _make_tab("General")
        cutout_layout = _make_tab("Cutout")
        ai_layout = _make_tab("AI")
        integrations_layout = _make_tab("Integrations")
        plot_layout = _make_tab("Plot")

        # Default Observatory
        self.site_combo = QComboBox(self)
        self.site_combo.addItems(parent.observatories.keys()) # type: ignore[attr-defined]
        init_site = parent.settings.value("general/defaultSite", parent.obs_combo.currentText(), type=str) if parent and hasattr(parent, "settings") else "Custom"
        self.site_combo.setCurrentText(init_site)
        general_layout.addRow("Default Observatory:", self.site_combo)

        # Visibility samples
        self.ts_spin = QSpinBox(self)
        self.ts_spin.setRange(50, 1000)
        init_ts = parent.settings.value("general/timeSamples", 240, type=int) if parent and hasattr(parent, "settings") else 240
        self.ts_spin.setValue(init_ts)
        general_layout.addRow("Visibility samples:", self.ts_spin)

        # Global UI font size
        self.ui_font_spin = QSpinBox(self)
        self.ui_font_spin.setRange(9, 16)
        self.ui_font_spin.setSuffix(" pt")
        init_ui_font = parent.settings.value("general/uiFontSize", 11, type=int) if parent and hasattr(parent, "settings") else 11
        self.ui_font_spin.setValue(max(9, min(16, int(init_ui_font))))
        general_layout.addRow("UI font size:", self.ui_font_spin)

        # UI theme
        self.theme_combo = QComboBox(self)
        for key, label in THEME_CHOICES:
            self.theme_combo.addItem(label, key)
        init_theme = (
            normalize_theme_key(parent.settings.value("general/uiTheme", DEFAULT_UI_THEME, type=str))
            if parent and hasattr(parent, "settings")
            else DEFAULT_UI_THEME
        )
        theme_idx = self.theme_combo.findData(init_theme)
        if theme_idx >= 0:
            self.theme_combo.setCurrentIndex(theme_idx)
        general_layout.addRow("Color theme:", self.theme_combo)

        self.dark_mode_chk = QCheckBox("Enable dark mode", self)
        self.dark_mode_chk.setChecked(
            parent.settings.value("general/darkMode", False, type=bool)
            if parent and hasattr(parent, "settings")
            else False
        )
        self.dark_mode_chk.toggled.connect(self._preview_dark_mode)
        if parent and hasattr(parent, "darkModeChanged"):
            parent.darkModeChanged.connect(self._sync_dark_mode_checkbox)
        general_layout.addRow(self.dark_mode_chk)

        # Cutout defaults
        self.cutout_view_combo = QComboBox(self)
        for key, label in CUTOUT_VIEW_CHOICES:
            self.cutout_view_combo.addItem(label, key)
        init_cutout_view = (
            _normalize_cutout_view_key(parent.settings.value("general/cutoutView", CUTOUT_DEFAULT_VIEW_KEY, type=str))
            if parent and hasattr(parent, "settings")
            else CUTOUT_DEFAULT_VIEW_KEY
        )
        cutout_view_idx = self.cutout_view_combo.findData(init_cutout_view)
        if cutout_view_idx >= 0:
            self.cutout_view_combo.setCurrentIndex(cutout_view_idx)

        self.cutout_survey_combo = QComboBox(self)
        for key, label, _ in CUTOUT_SURVEY_CHOICES:
            self.cutout_survey_combo.addItem(label, key)
        init_cutout_survey = (
            _normalize_cutout_survey_key(parent.settings.value("general/cutoutSurvey", CUTOUT_DEFAULT_SURVEY_KEY, type=str))
            if parent and hasattr(parent, "settings")
            else CUTOUT_DEFAULT_SURVEY_KEY
        )
        cutout_survey_idx = self.cutout_survey_combo.findData(init_cutout_survey)
        if cutout_survey_idx >= 0:
            self.cutout_survey_combo.setCurrentIndex(cutout_survey_idx)

        self.cutout_fov_spin = QSpinBox(self)
        self.cutout_fov_spin.setRange(CUTOUT_MIN_FOV_ARCMIN, CUTOUT_MAX_FOV_ARCMIN)
        self.cutout_fov_spin.setSuffix(" arcmin")
        init_cutout_fov = (
            _sanitize_cutout_fov_arcmin(parent.settings.value("general/cutoutFovArcmin", CUTOUT_DEFAULT_FOV_ARCMIN, type=int))
            if parent and hasattr(parent, "settings")
            else CUTOUT_DEFAULT_FOV_ARCMIN
        )
        self.cutout_fov_spin.setValue(init_cutout_fov)

        self.cutout_size_spin = QSpinBox(self)
        self.cutout_size_spin.setRange(CUTOUT_MIN_SIZE_PX, CUTOUT_MAX_SIZE_PX)
        self.cutout_size_spin.setSingleStep(16)
        self.cutout_size_spin.setSuffix(" px")
        init_cutout_size = (
            _sanitize_cutout_size_px(parent.settings.value("general/cutoutSizePx", CUTOUT_DEFAULT_SIZE_PX, type=int))
            if parent and hasattr(parent, "settings")
            else CUTOUT_DEFAULT_SIZE_PX
        )
        self.cutout_size_spin.setValue(init_cutout_size)

        cutout_hdr = QLabel("Cutout defaults")
        cutout_hdr.setObjectName("SectionHint")
        cutout_layout.addRow(cutout_hdr)
        cutout_layout.addRow("Cutout view:", self.cutout_view_combo)
        cutout_layout.addRow("Cutout survey:", self.cutout_survey_combo)
        cutout_layout.addRow("Cutout FOV:", self.cutout_fov_spin)
        cutout_layout.addRow("Cutout size:", self.cutout_size_spin)

        # Local LLM defaults
        self.llm_url_edit = QLineEdit(self)
        self.llm_url_edit.setPlaceholderText(LLMConfig.DEFAULT_URL)
        self.llm_url_edit.setText(
            parent.settings.value("llm/serverUrl", LLMConfig.DEFAULT_URL, type=str)
            if parent and hasattr(parent, "settings")
            else LLMConfig.DEFAULT_URL
        )
        self.llm_model_edit = QLineEdit(self)
        self.llm_model_edit.setPlaceholderText(LLMConfig.DEFAULT_MODEL)
        self.llm_model_edit.setText(
            parent.settings.value("llm/model", LLMConfig.DEFAULT_MODEL, type=str)
            if parent and hasattr(parent, "settings")
            else LLMConfig.DEFAULT_MODEL
        )

        llm_hdr = QLabel("Local AI assistant (optional)")
        llm_hdr.setObjectName("SectionHint")
        ai_layout.addRow(llm_hdr)
        ai_layout.addRow("LLM server URL:", self.llm_url_edit)
        ai_layout.addRow("LLM model:", self.llm_model_edit)

        # BHTOM credentials (optional, used for Suggest Targets)
        self.bhtom_api_edit = QLineEdit(self)
        self.bhtom_api_edit.setEchoMode(QLineEdit.Password)
        self.bhtom_api_edit.setPlaceholderText("API token for BHTOM target suggestions")
        self.bhtom_api_edit.setText(
            parent.settings.value("general/bhtomApiToken", "", type=str)
            if parent and hasattr(parent, "settings")
            else ""
        )

        bhtom_hdr = QLabel("BHTOM suggestions (optional)")
        bhtom_hdr.setObjectName("SectionHint")
        integrations_layout.addRow(bhtom_hdr)
        integrations_layout.addRow("BHTOM API token:", self.bhtom_api_edit)

        # TNS credentials (optional, used for TNS resolver)
        self.tns_api_edit = QLineEdit(self)
        self.tns_api_edit.setEchoMode(QLineEdit.Password)
        self.tns_api_edit.setPlaceholderText("API key from TNS bot")
        self.tns_api_edit.setText(parent.settings.value("general/tnsApiKey", "", type=str) if parent and hasattr(parent, "settings") else "")
        self.tns_bot_id_edit = QLineEdit(self)
        self.tns_bot_id_edit.setPlaceholderText("Numeric bot ID")
        self.tns_bot_id_edit.setText(parent.settings.value("general/tnsBotId", "", type=str) if parent and hasattr(parent, "settings") else "")
        self.tns_bot_name_edit = QLineEdit(self)
        self.tns_bot_name_edit.setPlaceholderText("Exact bot name")
        self.tns_bot_name_edit.setText(parent.settings.value("general/tnsBotName", "", type=str) if parent and hasattr(parent, "settings") else "")
        self.tns_endpoint_combo = QComboBox(self)
        for key, label in TNS_ENDPOINT_CHOICES:
            self.tns_endpoint_combo.addItem(label, key)
        init_tns_endpoint = (
            _normalize_tns_endpoint_key(parent.settings.value("general/tnsEndpoint", "production", type=str))
            if parent and hasattr(parent, "settings")
            else "production"
        )
        endpoint_idx = self.tns_endpoint_combo.findData(init_tns_endpoint)
        if endpoint_idx >= 0:
            self.tns_endpoint_combo.setCurrentIndex(endpoint_idx)
        self.tns_test_btn = QPushButton("Test TNS credentials", self)
        self.tns_test_btn.clicked.connect(self._test_tns_credentials)

        tns_hdr = QLabel("TNS credentials (optional)")
        tns_hdr.setObjectName("SectionHint")
        integrations_layout.addRow(tns_hdr)
        integrations_layout.addRow("TNS API key:", self.tns_api_edit)
        integrations_layout.addRow("TNS Bot ID:", self.tns_bot_id_edit)
        integrations_layout.addRow("TNS Bot name:", self.tns_bot_name_edit)
        integrations_layout.addRow("TNS endpoint:", self.tns_endpoint_combo)
        integrations_layout.addRow("", self.tns_test_btn)

        # Polar-plot path options
        self.sun_path_chk  = QCheckBox("Plot Sun path",  self)
        self.moon_path_chk = QCheckBox("Plot Moon path", self)
        self.obj_path_chk  = QCheckBox("Plot object paths", self)
        self.color_blind_chk = QCheckBox("Color-blind friendly palette", self)

        self.sun_path_chk.setChecked(parent.settings.value("general/showSunPath",  True, type=bool)) if parent and hasattr(parent, "settings") else True
        self.moon_path_chk.setChecked(parent.settings.value("general/showMoonPath", True, type=bool)) if parent and hasattr(parent, "settings") else True
        self.obj_path_chk.setChecked(parent.settings.value("general/showObjPath",  True, type=bool)) if parent and hasattr(parent, "settings") else True
        self.color_blind_chk.setChecked(parent.settings.value("general/colorBlindMode", False, type=bool)) if parent and hasattr(parent, "settings") else False

        plot_layout.addRow(self.sun_path_chk)
        plot_layout.addRow(self.moon_path_chk)
        plot_layout.addRow(self.obj_path_chk)
        plot_layout.addRow(self.color_blind_chk)

        # OK / Cancel
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.Apply, self)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        apply_btn = btns.button(QDialogButtonBox.Apply)
        if apply_btn is not None:
            apply_btn.clicked.connect(self._apply_changes)
        root_layout.addWidget(btns)
        self.setMinimumSize(560, 460)
        self.resize(640, 520)

    @Slot(bool)
    def _preview_dark_mode(self, checked: bool):
        parent = self.parent()
        if parent and hasattr(parent, "_set_dark_mode_enabled"):
            parent._set_dark_mode_enabled(checked, persist=False)

    @Slot(bool)
    def _sync_dark_mode_checkbox(self, enabled: bool):
        blocker = QSignalBlocker(self.dark_mode_chk)
        self.dark_mode_chk.setChecked(enabled)
        del blocker

    def _apply_changes(self):
        s = self.parent().settings
        rerun_plan = any(
            (
                s.value("general/defaultSite", self.site_combo.currentText(), type=str) != self.site_combo.currentText(),
                s.value("general/timeSamples", self.ts_spin.value(), type=int) != self.ts_spin.value(),
                s.value("general/showSunPath", self.sun_path_chk.isChecked(), type=bool) != self.sun_path_chk.isChecked(),
                s.value("general/showMoonPath", self.moon_path_chk.isChecked(), type=bool) != self.moon_path_chk.isChecked(),
                s.value("general/showObjPath", self.obj_path_chk.isChecked(), type=bool) != self.obj_path_chk.isChecked(),
                s.value("general/colorBlindMode", self.color_blind_chk.isChecked(), type=bool) != self.color_blind_chk.isChecked(),
            )
        )
        s.setValue("general/defaultSite", self.site_combo.currentText())
        s.setValue("general/timeSamples", self.ts_spin.value())
        s.setValue("general/uiFontSize", self.ui_font_spin.value())
        s.setValue("general/uiTheme", self.theme_combo.currentData())
        s.setValue("general/darkMode", self.dark_mode_chk.isChecked())
        s.setValue("general/showSunPath",  self.sun_path_chk.isChecked())
        s.setValue("general/showMoonPath", self.moon_path_chk.isChecked())
        s.setValue("general/showObjPath",  self.obj_path_chk.isChecked())
        s.setValue("general/colorBlindMode", self.color_blind_chk.isChecked())
        s.setValue("general/cutoutView", _normalize_cutout_view_key(self.cutout_view_combo.currentData()))
        s.setValue("general/cutoutSurvey", _normalize_cutout_survey_key(self.cutout_survey_combo.currentData()))
        s.setValue("general/cutoutFovArcmin", _sanitize_cutout_fov_arcmin(self.cutout_fov_spin.value()))
        s.setValue("general/cutoutSizePx", _sanitize_cutout_size_px(self.cutout_size_spin.value()))
        llm_url = self.llm_url_edit.text().strip() or LLMConfig.DEFAULT_URL
        llm_model = self.llm_model_edit.text().strip() or LLMConfig.DEFAULT_MODEL
        s.setValue("llm/serverUrl", llm_url)
        s.setValue("llm/model", llm_model)
        s.setValue("general/bhtomApiToken", self.bhtom_api_edit.text().strip())
        s.setValue("general/tnsApiKey", self.tns_api_edit.text().strip())
        s.setValue("general/tnsBotId", self.tns_bot_id_edit.text().strip())
        s.setValue("general/tnsBotName", self.tns_bot_name_edit.text().strip())
        s.setValue("general/tnsEndpoint", _normalize_tns_endpoint_key(self.tns_endpoint_combo.currentData()))
        self.parent()._apply_general_settings()
        if rerun_plan:
            self.parent()._run_plan()
        self._original_dark_mode = self.dark_mode_chk.isChecked()

    def accept(self):
        self._apply_changes()
        super().accept()

    def reject(self):
        parent = self.parent()
        if parent and hasattr(parent, "_set_dark_mode_enabled"):
            parent._set_dark_mode_enabled(self._original_dark_mode, persist=True)
        super().reject()

    @Slot()
    def _test_tns_credentials(self):
        api_key = self.tns_api_edit.text().strip()
        bot_id_raw = self.tns_bot_id_edit.text().strip()
        bot_name = self.tns_bot_name_edit.text().strip()
        if not api_key or not bot_id_raw or not bot_name:
            QMessageBox.warning(
                self,
                "Missing TNS Fields",
                "Fill TNS API key, Bot ID and Bot name before testing.",
            )
            return
        try:
            bot_id = int(bot_id_raw)
        except ValueError:
            QMessageBox.warning(self, "Invalid Bot ID", "TNS Bot ID must be numeric.")
            return

        marker = _build_tns_marker(bot_id, bot_name)
        endpoint_key = _normalize_tns_endpoint_key(self.tns_endpoint_combo.currentData())
        api_base = _tns_api_base_url(endpoint_key).rstrip("/")
        body = urlencode(
            {
                "api_key": api_key,
                "data": json.dumps({"objname": "2023ixf"}),
            }
        ).encode("utf-8")
        req = Request(
            f"{api_base}/get/search",
            data=body,
            headers={
                "User-Agent": marker,
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            try:
                with urlopen(req, timeout=20) as resp:
                    raw = resp.read().decode("utf-8", errors="replace")
            except HTTPError as exc:
                if exc.code in {401, 403}:
                    QMessageBox.critical(
                        self,
                        "TNS Unauthorized",
                        f"Unauthorized (401/403) on {endpoint_key}. Check API key, bot ID/name and bot permissions.",
                    )
                    return
                detail = ""
                try:
                    detail = exc.read().decode("utf-8", errors="ignore").strip()
                except Exception:
                    detail = ""
                msg = f"TNS request failed ({exc.code})."
                if detail:
                    msg = f"{msg}\n\n{detail[:280]}"
                QMessageBox.critical(self, "TNS Error", msg)
                return
            except Exception as exc:
                QMessageBox.critical(self, "TNS Error", f"Request failed: {exc}")
                return

            try:
                payload = json.loads(raw)
            except Exception:
                QMessageBox.warning(
                    self,
                    "TNS Response",
                    f"Received response from {endpoint_key}, but it was not valid JSON. Credentials may still be valid.",
                )
                return

            id_code = payload.get("id_code") if isinstance(payload, dict) else None
            id_msg = str(payload.get("id_message", "")).strip() if isinstance(payload, dict) else ""
            if id_code in {401, 403} or "unauthorized" in id_msg.lower():
                QMessageBox.critical(
                    self,
                    "TNS Unauthorized",
                    f"TNS rejected credentials on {endpoint_key}: {id_msg or id_code}",
                )
                return

            QMessageBox.information(
                self,
                "TNS OK",
                f"Credentials look valid on {endpoint_key}.\n{id_msg or 'Authentication passed.'}",
            )
        finally:
            QApplication.restoreOverrideCursor()


class AddTargetDialog(QDialog):
    """Two-step target add dialog with lazy-expanded metadata details."""

    def __init__(
        self,
        resolver,
        metadata_fetcher: Optional[Callable[[Target], None]] = None,
        parent=None,
        source_options: Optional[list[tuple[str, str]]] = None,
        default_source: str = "simbad",
    ):
        super().__init__(parent)
        self.setObjectName(self.__class__.__name__)
        self.setWindowTitle("Add Target")
        self._resolver = resolver
        self._metadata_fetcher = metadata_fetcher
        self._source_options = source_options or list(TARGET_SEARCH_SOURCES)
        self._resolved_target: Optional[Target] = None
        self._resolved_query = ""
        self._resolved_source = ""

        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.source_combo = QComboBox(self)
        for key, label in self._source_options:
            self.source_combo.addItem(label, key)
        source_idx = self.source_combo.findData(_normalize_catalog_token(default_source))
        if source_idx >= 0:
            self.source_combo.setCurrentIndex(source_idx)
        form.addRow("Source:", self.source_combo)
        self.query_edit = QLineEdit(self)
        self.query_edit.setPlaceholderText("Name or RA Dec")
        form.addRow("Query:", self.query_edit)
        layout.addLayout(form)

        top_row = QHBoxLayout()
        self.resolve_btn = QPushButton("Resolve", self)
        self.resolve_btn.clicked.connect(self._resolve_target)
        top_row.addWidget(self.resolve_btn)
        top_row.addStretch(1)
        layout.addLayout(top_row)

        self.details_widget = QWidget(self)
        details_form = QFormLayout(self.details_widget)
        self.name_edit = QLineEdit(self)
        self.name_edit.setPlaceholderText("Resolved name")
        self.name_edit.setEnabled(False)
        self.ra_label = QLabel("-")
        self.dec_label = QLabel("-")
        self.mag_label = QLabel("-")
        self.type_edit = QLineEdit(self)
        self.type_edit.setPlaceholderText("Object type")
        self.type_edit.setEnabled(False)
        self.priority_spin = QSpinBox(self)
        self.priority_spin.setRange(1, 5)
        self.priority_spin.setValue(3)
        self.notes_edit = QTextEdit(self)
        self.notes_edit.setPlaceholderText("Optional notes for this target...")
        self.notes_edit.setMinimumHeight(70)
        self.notes_edit.setMaximumHeight(90)

        details_form.addRow("Name:", self.name_edit)
        details_form.addRow("RA:", self.ra_label)
        details_form.addRow("Dec:", self.dec_label)
        details_form.addRow("Magnitude:", self.mag_label)
        details_form.addRow("Type:", self.type_edit)
        details_form.addRow("Priority:", self.priority_spin)
        details_form.addRow("Notes:", self.notes_edit)
        self.details_widget.setVisible(False)
        layout.addWidget(self.details_widget)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        ok_btn = self.button_box.button(QDialogButtonBox.Ok)
        if ok_btn:
            ok_btn.setEnabled(False)
        layout.addWidget(self.button_box)
        self.setMinimumWidth(max(500, self.sizeHint().width()))

        self.query_edit.returnPressed.connect(self._resolve_target)
        self.query_edit.textChanged.connect(self._on_query_changed)
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)

    def _current_source(self) -> str:
        return _normalize_catalog_token(self.source_combo.currentData())

    def _invalidate_resolution(self):
        self._resolved_target = None
        ok_btn = self.button_box.button(QDialogButtonBox.Ok)
        if ok_btn:
            ok_btn.setEnabled(False)

    def _on_query_changed(self, text: str):
        if text.strip() != self._resolved_query:
            self._invalidate_resolution()

    @Slot(int)
    def _on_source_changed(self, _index: int):
        if self._current_source() != self._resolved_source:
            self._invalidate_resolution()

    @Slot()
    def _resolve_target(self):
        query = self.query_edit.text().strip()
        source = self._current_source()
        if not query:
            QMessageBox.warning(self, "Missing query", "Enter a target name or RA/Dec first.")
            return
        try:
            target = self._resolver(query, source)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Resolve error", str(exc))
            return
        if self._metadata_fetcher and (target.magnitude is None or not target.object_type):
            try:
                self._metadata_fetcher(target)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Metadata enrichment failed for '%s': %s", target.name, exc)

        self._resolved_target = target
        self._resolved_query = query
        self._resolved_source = source
        self.name_edit.setText(target.name)
        self.name_edit.setEnabled(True)
        self.ra_label.setText(Angle(target.ra, u.deg).to_string(unit=u.hourangle, sep=":", pad=True, precision=1))
        self.dec_label.setText(Angle(target.dec, u.deg).to_string(unit=u.deg, sep=":", pad=True, precision=1, alwayssign=True))
        self.mag_label.setText(f"{target.magnitude:.2f}" if target.magnitude is not None else "-")
        self.type_edit.setText(target.object_type or "")
        self.type_edit.setEnabled(True)
        if not self.details_widget.isVisible():
            self.details_widget.setVisible(True)
            self.adjustSize()
        ok_btn = self.button_box.button(QDialogButtonBox.Ok)
        if ok_btn:
            ok_btn.setEnabled(True)

    def build_target(self) -> Target:
        if self._resolved_target is None:
            raise ValueError("Target is not resolved.")
        target = self._resolved_target.model_copy(deep=True)
        edited_name = self.name_edit.text().strip()
        target.name = edited_name or target.name
        target.object_type = self.type_edit.text().strip()
        target.priority = self.priority_spin.value()
        target.notes = self.notes_edit.toPlainText().strip()
        return target

    def selected_source(self) -> str:
        return self._current_source()

# Number of time samples for visibility curve (lower = faster)
TIME_SAMPLES = 240 

class ClockWorker(QThread):
    updated = Signal(dict)

    def __init__(self, site: Site, targets: list[Target], parent=None):
        super().__init__(parent)
        self.setObjectName(self.__class__.__name__)
        self.site = site
        self.targets = targets
        self.running = True
        # Interruptible sleep infrastructure
        self._mutex = QMutex()
        self._wait_cond = QWaitCondition()

    def run(self):
        tz_name = self.site.timezone_name
        tz = pytz.timezone(tz_name)
        logger.info("ClockWorker started for site %s", self.site.name)

        while self.running:
            now_local = datetime.now(tz)
            now_utc = datetime.now(timezone.utc)

            eph_obs = ephem.Observer()
            eph_obs.lat = str(self.site.latitude)
            eph_obs.lon = str(self.site.longitude)
            eph_obs.elevation = self.site.elevation
            eph_obs.date = now_local
            sun = ephem.Sun(eph_obs)
            moon = ephem.Moon(eph_obs)
            sun_alt = sun.alt * 180.0 / math.pi
            moon_alt = moon.alt * 180.0 / math.pi
            moon_coord = SkyCoord(ra=Angle(moon.ra, u.rad), dec=Angle(moon.dec, u.rad))

            obs = Observer(location=self.site.to_earthlocation(), timezone=tz_name)
            eph_obs.date = now_local

            current_alts = []
            current_azs = []
            current_seps = []
            for tgt in self.targets:
                fixed = FixedTarget(name=tgt.name, coord=tgt.skycoord)
                altaz = obs.altaz(Time(now_local), fixed)
                current_alts.append(float(altaz.alt.deg))   # type: ignore[arg-type]
                current_azs.append(float(altaz.az.deg))     # type: ignore[arg-type]
                sep_deg = tgt.skycoord.separation(moon_coord).deg
                current_seps.append(float(np.real(sep_deg)))

            self.updated.emit({
                "now_local": now_local,
                "now_utc": now_utc,
                "sun_alt": sun_alt,
                "moon_alt": moon_alt,
                "alts": current_alts,
                "azs": current_azs,
                "seps": current_seps
            })

            # Sleep until either 1 s passes or stop() wakes us early
            self._mutex.lock()
            self._wait_cond.wait(self._mutex, 1000)
            self._mutex.unlock()

        logger.info("ClockWorker exiting for site %s", self.site.name)

    def stop(self):
        logger.info("ClockWorker stop requested for site %s", self.site.name)
        self.running = False
        self._wait_cond.wakeAll()  # interrupt the 1‑second wait
        self.quit()
        self.wait()


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


class FinderChartWorker(QThread):
    """Render finder chart in background thread to keep UI responsive."""

    completed = Signal(int, str, bytes, str)

    def __init__(
        self,
        request_id: int,
        key: str,
        name: str,
        ra_deg: float,
        dec_deg: float,
        survey_key: str,
        fov_arcmin: int,
        width_px: int,
        height_px: int,
        parent=None,
    ):
        super().__init__(parent)
        self.setObjectName(self.__class__.__name__)
        self.request_id = int(request_id)
        self.key = key
        self.name = name
        self.ra_deg = float(ra_deg)
        self.dec_deg = float(dec_deg)
        self.survey_key = survey_key
        self.fov_arcmin = int(fov_arcmin)
        self.width_px = int(width_px)
        self.height_px = int(height_px)

    def run(self):
        if self.isInterruptionRequested():
            self.completed.emit(self.request_id, self.key, b"", "cancelled")
            return
        try:
            payload = _render_finder_chart_png_bytes(
                name=self.name,
                ra_deg=self.ra_deg,
                dec_deg=self.dec_deg,
                survey_key=self.survey_key,
                fov_arcmin=self.fov_arcmin,
                width_px=self.width_px,
                height_px=self.height_px,
            )
            if self.isInterruptionRequested():
                self.completed.emit(self.request_id, self.key, b"", "cancelled")
                return
            self.completed.emit(self.request_id, self.key, payload, "")
        except Exception as exc:  # noqa: BLE001
            self.completed.emit(self.request_id, self.key, b"", str(exc))

# --------------------------------------------------
# --- Models (Pydantic) -----------------------------
# --------------------------------------------------
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
        return cls(name=name, ra=coord.ra.deg, dec=coord.dec.deg)    # type: ignore[arg-type]

    @property
    def skycoord(self) -> SkyCoord:  # noqa: D401
        return SkyCoord(ra=self.ra * u.deg, dec=self.dec * u.deg)


class Site(BaseModel):
    """Observation site."""

    name: str = "Custom"
    latitude: float = Field(..., description="Latitude in °")
    longitude: float = Field(..., description="Longitude in °")
    elevation: float = Field(0.0, description="Elevation in m")

    def to_earthlocation(self) -> EarthLocation:  # noqa: D401
        return EarthLocation(lat=self.latitude * u.deg, lon=self.longitude * u.deg, height=self.elevation * u.m)

    @property
    def timezone_name(self) -> str:  # noqa: D401
        return _TZ_FINDER.timezone_at(lng=self.longitude, lat=self.latitude) or "UTC"


class SessionSettings(BaseModel):
    """Per‑night settings handed to the worker thread."""

    date: QDate
    site: Site
    limit_altitude: float = 35.0  # deg
    time_samples: int = 240       # user-configurable resolution

    # <‑‑ make Pydantic happy with Qt types
    model_config = ConfigDict(arbitrary_types_allowed=True)


def _targets_match(left: Target, right: Target, max_sep_deg: float = 0.05) -> bool:
    left_source = _normalize_catalog_token(left.source_object_id)
    right_source = _normalize_catalog_token(right.source_object_id)
    if left_source and right_source and left_source == right_source:
        return True
    if _normalize_catalog_token(left.name) == _normalize_catalog_token(right.name):
        return True
    try:
        return float(left.skycoord.separation(right.skycoord).deg) < max_sep_deg
    except Exception:
        return False


class SuggestionTableModel(QAbstractTableModel):
    COL_NAME = 0
    COL_TYPE = 1
    COL_MAG = 2
    COL_PRIORITY = 3
    COL_IMPORTANCE = 4
    COL_SCORE = 5
    COL_AIRMASS = 6
    COL_HOURS = 7
    COL_WINDOW = 8
    COL_MOON_SEP = 9
    COL_ACTION = 10

    headers = [
        "Name",
        "Type",
        "Last Mag",
        "Pri",
        "Importance",
        "Score",
        "Min Airmass",
        "Over Lim (h)",
        "Best Window",
        "Moon Sep",
        "Add",
    ]

    def __init__(
        self,
        suggestions: list[dict[str, object]],
        moon_sep_threshold: float,
        parent=None,
    ):
        super().__init__(parent)
        self._all_rows = suggestions
        self._visible_rows: list[dict[str, object]] = []
        self._min_importance = BHTOM_SUGGESTION_MIN_IMPORTANCE
        self._min_score = 0.0
        self._min_hours = 0.0
        self._min_moon_sep = 0.0
        self._max_airmass = 99.0
        self._max_magnitude = 99.0
        self._moon_sep_threshold = float(moon_sep_threshold)
        self._sort_column = self.COL_IMPORTANCE
        self._sort_order = Qt.DescendingOrder
        self._hover_row: Optional[int] = None
        self._hover_name_row: Optional[int] = None
        self._hover_action_row: Optional[int] = None
        self._rebuild_rows()

    @staticmethod
    def _target(item: dict[str, object]) -> Target:
        target = item["target"]
        assert isinstance(target, Target)
        return target

    @staticmethod
    def _metrics(item: dict[str, object]) -> TargetNightMetrics:
        metrics = item["metrics"]
        assert isinstance(metrics, TargetNightMetrics)
        return metrics

    @staticmethod
    def _window_start(item: dict[str, object]) -> datetime:
        value = item["window_start"]
        assert isinstance(value, datetime)
        return value

    @staticmethod
    def _window_end(item: dict[str, object]) -> datetime:
        value = item["window_end"]
        assert isinstance(value, datetime)
        return value

    @staticmethod
    def _optional_float_key(value: object, descending: bool) -> tuple[int, float]:
        number = _safe_float(value)
        if number is None or not math.isfinite(number):
            return (1, 0.0)
        return (0, -float(number) if descending else float(number))

    @staticmethod
    def _numeric_key(value: object, descending: bool) -> float:
        number = _safe_float(value)
        if number is None or not math.isfinite(number):
            number = 0.0
        return -float(number) if descending else float(number)

    def _passes_filters(self, item: dict[str, object]) -> bool:
        target = self._target(item)
        metrics = self._metrics(item)
        importance = float(item.get("importance", 0.0) or 0.0)
        if importance < self._min_importance:
            return False
        if metrics.score < self._min_score:
            return False
        if metrics.hours_above_limit < self._min_hours:
            return False
        min_window_moon_sep = _safe_float(item.get("min_window_moon_sep"))
        if self._min_moon_sep > 0.0:
            if min_window_moon_sep is None or not math.isfinite(min_window_moon_sep) or min_window_moon_sep < self._min_moon_sep:
                return False
        best_airmass = _safe_float(item.get("best_airmass"))
        if self._max_airmass < 99.0:
            if best_airmass is None or not math.isfinite(best_airmass) or best_airmass > self._max_airmass:
                return False
        if self._max_magnitude < 99.0:
            if target.magnitude is None or target.magnitude > self._max_magnitude:
                return False
        return True

    def _sort_rows(self, rows: list[dict[str, object]]) -> None:
        descending = self._sort_order == Qt.DescendingOrder
        col = self._sort_column
        if col == self.COL_NAME:
            rows.sort(key=lambda item: self._target(item).name.lower(), reverse=descending)
        elif col == self.COL_TYPE:
            rows.sort(key=lambda item: (self._target(item).object_type or "").lower(), reverse=descending)
        elif col == self.COL_MAG:
            rows.sort(key=lambda item: self._optional_float_key(self._target(item).magnitude, descending))
        elif col == self.COL_PRIORITY:
            rows.sort(key=lambda item: self._numeric_key(self._target(item).priority, descending))
        elif col == self.COL_IMPORTANCE:
            rows.sort(key=lambda item: self._numeric_key(item.get("importance"), descending))
        elif col == self.COL_SCORE:
            rows.sort(key=lambda item: self._numeric_key(self._metrics(item).score, descending))
        elif col == self.COL_AIRMASS:
            rows.sort(key=lambda item: self._optional_float_key(item.get("best_airmass"), descending))
        elif col == self.COL_HOURS:
            rows.sort(key=lambda item: self._numeric_key(self._metrics(item).hours_above_limit, descending))
        elif col == self.COL_WINDOW:
            rows.sort(
                key=lambda item: self._window_start(item).timestamp() * (-1.0 if descending else 1.0)
            )
        elif col == self.COL_MOON_SEP:
            rows.sort(key=lambda item: self._optional_float_key(item.get("min_window_moon_sep"), descending))
        elif col == self.COL_ACTION:
            rows.sort(
                key=lambda item: (
                    bool(item.get("added_to_plan")),
                    self._target(item).name.lower(),
                )
            )
            if descending:
                rows.reverse()
        else:
            rows.sort(
                key=lambda item: (
                    -float(item.get("importance", 0.0) or 0.0),
                    -float(self._metrics(item).score),
                    self._target(item).name.lower(),
                )
            )

    def _rebuild_rows(self) -> None:
        self.beginResetModel()
        filtered = [item for item in self._all_rows if self._passes_filters(item)]
        self._sort_rows(filtered)
        self._visible_rows = filtered
        self._hover_row = None
        self._hover_name_row = None
        self._hover_action_row = None
        self.endResetModel()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return len(self._visible_rows)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return len(self.headers)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):  # noqa: N802
        if not index.isValid() or not (0 <= index.row() < len(self._visible_rows)):
            return None

        item = self._visible_rows[index.row()]
        target = self._target(item)
        metrics = self._metrics(item)
        row = index.row()
        col = index.column()

        if role == Qt.TextAlignmentRole:
            if col in {self.COL_NAME, self.COL_TYPE, self.COL_WINDOW}:
                return Qt.AlignLeft | Qt.AlignVCenter
            return Qt.AlignCenter | Qt.AlignVCenter

        if role == Qt.FontRole and col in {self.COL_NAME, self.COL_ACTION}:
            font = QFont()
            font.setBold(True)
            if col == self.COL_NAME and row == self._hover_name_row:
                font.setUnderline(True)
            return font

        if role == Qt.ToolTipRole:
            if col == self.COL_NAME:
                return f"Open in BHTOM: {target.name}"
            if col == self.COL_WINDOW:
                return (
                    f"{self._window_start(item).strftime('%Y-%m-%d %H:%M')} -> "
                    f"{self._window_end(item).strftime('%Y-%m-%d %H:%M')}"
                )
            if col == self.COL_MOON_SEP and item.get("moon_sep_warning"):
                min_sep = _safe_float(item.get("min_window_moon_sep"))
                if min_sep is not None:
                    return (
                        f"Moon separation drops to {min_sep:.1f} deg in the best window "
                        f"(threshold {self._moon_sep_threshold:.0f} deg)."
                    )
            if col == self.COL_ACTION and item.get("added_to_plan"):
                return "This target has already been added to the current plan."

        if role == Qt.BackgroundRole:
            if row == self._hover_row:
                return QBrush(QColor("#ff3d78"))
            if col == self.COL_MOON_SEP and item.get("moon_sep_warning"):
                return QBrush(QColor("#fff1a8"))
            if col == self.COL_ACTION:
                if item.get("added_to_plan"):
                    return QBrush(QColor("#e6f4ea"))
                if row == self._hover_action_row:
                    return QBrush(QColor("#d8e2ef"))
                return QBrush(QColor("#c7d1dc"))

        if role == Qt.ForegroundRole:
            if row == self._hover_row:
                return QBrush(QColor("#ffcb3a"))
            if col == self.COL_MOON_SEP and item.get("moon_sep_warning"):
                return QBrush(QColor("#3f3200"))
            if col == self.COL_ACTION:
                if item.get("added_to_plan"):
                    return QBrush(QColor("#4f6f52"))
                if row == self._hover_action_row:
                    return QBrush(QColor("#17212b"))
                return QBrush(QColor("#243241"))

        if role not in (Qt.DisplayRole, Qt.EditRole):
            return None

        if col == self.COL_NAME:
            return target.name
        if col == self.COL_TYPE:
            return target.object_type or "-"
        if col == self.COL_MAG:
            return f"{target.magnitude:.2f}" if target.magnitude is not None else "-"
        if col == self.COL_PRIORITY:
            return str(target.priority)
        if col == self.COL_IMPORTANCE:
            return f"{float(item.get('importance', 0.0) or 0.0):.1f}"
        if col == self.COL_SCORE:
            return f"{metrics.score:.1f}"
        if col == self.COL_AIRMASS:
            best_airmass = _safe_float(item.get("best_airmass"))
            return f"{best_airmass:.2f}" if best_airmass is not None else "-"
        if col == self.COL_HOURS:
            return f"{metrics.hours_above_limit:.1f}"
        if col == self.COL_WINDOW:
            return f"{self._window_start(item).strftime('%H:%M')} - {self._window_end(item).strftime('%H:%M')}"
        if col == self.COL_MOON_SEP:
            min_sep = _safe_float(item.get("min_window_moon_sep"))
            return f"{min_sep:.1f}" if min_sep is not None else "-"
        if col == self.COL_ACTION:
            return "Added" if item.get("added_to_plan") else "Add"
        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole):  # noqa: N802
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self.headers[section]
        return None

    def flags(self, index: QModelIndex):  # noqa: N802
        if not index.isValid():
            return Qt.ItemIsEnabled
        return Qt.ItemIsSelectable | Qt.ItemIsEnabled

    def sort(self, column: int, order: Qt.SortOrder = Qt.AscendingOrder) -> None:
        self._sort_column = int(column)
        self._sort_order = order
        self._rebuild_rows()

    def set_filters(
        self,
        min_importance: float,
        min_score: float,
        min_hours: float,
        min_moon_sep: float,
        max_airmass: float,
        max_magnitude: float,
    ) -> None:
        self._min_importance = float(min_importance)
        self._min_score = float(min_score)
        self._min_hours = float(min_hours)
        self._min_moon_sep = float(min_moon_sep)
        self._max_airmass = float(max_airmass)
        self._max_magnitude = float(max_magnitude)
        self._rebuild_rows()

    def filtered_count(self) -> int:
        return sum(1 for item in self._all_rows if self._passes_filters(item))

    def total_count(self) -> int:
        return len(self._all_rows)

    def suggestion_at(self, row: int) -> dict[str, object]:
        return self._visible_rows[row]

    def mark_added(self, target: Target) -> None:
        changed = False
        for item in self._all_rows:
            item_target = item.get("target")
            if isinstance(item_target, Target) and _targets_match(item_target, target):
                item["added_to_plan"] = True
                changed = True
        if changed:
            self._rebuild_rows()

    def replace_suggestions(self, suggestions: list[dict[str, object]]) -> None:
        self.beginResetModel()
        self._all_rows = suggestions
        self._visible_rows = []
        self._hover_row = None
        self._hover_name_row = None
        self._hover_action_row = None
        self.endResetModel()
        self._rebuild_rows()

    def set_action_hover_row(self, row: Optional[int]) -> None:
        new_row = row if row is not None and 0 <= row < len(self._visible_rows) else None
        if new_row == self._hover_action_row:
            return
        old_row = self._hover_action_row
        self._hover_action_row = new_row
        for changed_row in (old_row, new_row):
            if changed_row is None:
                continue
            idx = self.index(changed_row, self.COL_ACTION)
            self.dataChanged.emit(idx, idx, [Qt.BackgroundRole, Qt.ForegroundRole])

    def set_hover_row(self, row: Optional[int]) -> None:
        new_row = row if row is not None and 0 <= row < len(self._visible_rows) else None
        if new_row == self._hover_row:
            return
        old_row = self._hover_row
        self._hover_row = new_row
        for changed_row in (old_row, new_row):
            if changed_row is None:
                continue
            left = self.index(changed_row, 0)
            right = self.index(changed_row, self.columnCount() - 1)
            self.dataChanged.emit(
                left,
                right,
                [Qt.BackgroundRole, Qt.ForegroundRole, Qt.FontRole],
            )

    def set_name_hover_row(self, row: Optional[int]) -> None:
        new_row = row if row is not None and 0 <= row < len(self._visible_rows) else None
        if new_row == self._hover_name_row:
            return
        old_row = self._hover_name_row
        self._hover_name_row = new_row
        for changed_row in (old_row, new_row):
            if changed_row is None:
                continue
            idx = self.index(changed_row, self.COL_NAME)
            self.dataChanged.emit(idx, idx, [Qt.BackgroundRole, Qt.ForegroundRole, Qt.FontRole])


class SuggestedTargetsDialog(QDialog):
    def __init__(
        self,
        suggestions: list[dict[str, object]],
        notes: list[str],
        moon_sep_threshold: float,
        initial_score_filter: float,
        bhtom_base_url: str,
        add_callback: Callable[[Target], bool],
        reload_callback: Optional[Callable[[], tuple[list[dict[str, object]], list[str]]]] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setObjectName(self.__class__.__name__)
        self.setWindowTitle("Suggested Targets")
        self.resize(1280, 680)
        self._add_callback = add_callback
        self._reload_callback = reload_callback
        self._bhtom_base_url = bhtom_base_url.rstrip("/")
        self._notes = notes
        self._settings = (
            parent.settings
            if parent is not None and hasattr(parent, "settings")
            else QSettings("YourCompany", "AstroPlanner")
        )
        self._filter_defaults = {
            "importance": float(BHTOM_SUGGESTION_MIN_IMPORTANCE),
            "score": float(initial_score_filter),
            "hours": 0.0,
            "moon_sep": float(moon_sep_threshold),
            "airmass": 99.0,
            "magnitude": 99.0,
        }

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self.summary_label = QLabel(self)
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        filters_row = QHBoxLayout()
        filters_row.setSpacing(8)
        filters_row.addWidget(QLabel("Importance ≥", self))
        self.importance_spin = QDoubleSpinBox(self)
        self.importance_spin.setRange(0.0, 10.0)
        self.importance_spin.setDecimals(1)
        self.importance_spin.setSingleStep(0.5)
        self.importance_spin.setValue(BHTOM_SUGGESTION_MIN_IMPORTANCE)
        self.importance_spin.setToolTip("Minimum BHTOM importance.")
        filters_row.addWidget(self.importance_spin)

        filters_row.addWidget(QLabel("Score ≥", self))
        self.score_spin = QDoubleSpinBox(self)
        self.score_spin.setRange(0.0, 100.0)
        self.score_spin.setDecimals(1)
        self.score_spin.setSingleStep(1.0)
        self.score_spin.setValue(float(initial_score_filter))
        filters_row.addWidget(self.score_spin)

        filters_row.addWidget(QLabel("Over Lim ≥", self))
        self.hours_spin = QDoubleSpinBox(self)
        self.hours_spin.setRange(0.0, 24.0)
        self.hours_spin.setDecimals(1)
        self.hours_spin.setSingleStep(0.5)
        self.hours_spin.setValue(0.0)
        self.hours_spin.setToolTip("Minimum hours above the altitude limit in the observing window.")
        filters_row.addWidget(self.hours_spin)

        filters_row.addWidget(QLabel("Moon Sep ≥", self))
        self.moon_sep_spin = QDoubleSpinBox(self)
        self.moon_sep_spin.setRange(0.0, 180.0)
        self.moon_sep_spin.setDecimals(1)
        self.moon_sep_spin.setSingleStep(1.0)
        self.moon_sep_spin.setValue(float(moon_sep_threshold))
        self.moon_sep_spin.setToolTip("Minimum Moon separation in the best observing window.")
        filters_row.addWidget(self.moon_sep_spin)

        filters_row.addWidget(QLabel("Min Airmass ≤", self))
        self.airmass_spin = QDoubleSpinBox(self)
        self.airmass_spin.setRange(1.0, 99.0)
        self.airmass_spin.setDecimals(2)
        self.airmass_spin.setSingleStep(0.1)
        self.airmass_spin.setValue(99.0)
        self.airmass_spin.setToolTip("Maximum minimum airmass in the best observing window. Leave at 99 to disable this filter.")
        filters_row.addWidget(self.airmass_spin)

        filters_row.addWidget(QLabel("Mag ≤", self))
        self.magnitude_spin = QDoubleSpinBox(self)
        self.magnitude_spin.setRange(-5.0, 99.0)
        self.magnitude_spin.setDecimals(1)
        self.magnitude_spin.setSingleStep(0.5)
        self.magnitude_spin.setValue(99.0)
        self.magnitude_spin.setToolTip("Maximum magnitude. Leave at 99 to disable this filter.")
        filters_row.addWidget(self.magnitude_spin)

        filters_row.addStretch(1)
        self.reset_filters_btn = QPushButton("Reset", self)
        self.reset_filters_btn.setToolTip("Restore the default suggested-target filters.")
        filters_row.addWidget(self.reset_filters_btn)
        self.reload_btn = QPushButton("Reload", self)
        self.reload_btn.setToolTip("Fetch a fresh BHTOM target list and rebuild suggestions.")
        self.reload_btn.setEnabled(self._reload_callback is not None)
        filters_row.addWidget(self.reload_btn)
        layout.addLayout(filters_row)

        self._restore_filter_settings()

        self.table_model = SuggestionTableModel(suggestions, moon_sep_threshold, self)
        self.table_view = QTableView(self)
        self.table_view.setModel(self.table_model)
        self.table_view.setSelectionBehavior(QTableView.SelectRows)
        self.table_view.setSelectionMode(QTableView.SingleSelection)
        self.table_view.setSortingEnabled(True)
        self.table_view.setEditTriggers(QTableView.NoEditTriggers)
        self.table_view.setHorizontalScrollMode(QTableView.ScrollPerPixel)
        self.table_view.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.table_view.setMouseTracking(True)
        self.table_view.verticalHeader().setVisible(False)
        self.table_view.setShowGrid(False)
        self.table_view.viewport().installEventFilter(self)
        header = self.table_view.horizontalHeader()
        header.setSectionsClickable(True)
        header.setStretchLastSection(False)
        header.setSectionResizeMode(QHeaderView.Interactive)
        self.table_view.setColumnWidth(
            SuggestionTableModel.COL_NAME,
            self._default_name_column_width(suggestions),
        )
        self.table_view.setColumnWidth(SuggestionTableModel.COL_TYPE, 128)
        self.table_view.setColumnWidth(SuggestionTableModel.COL_MAG, 64)
        self.table_view.setColumnWidth(SuggestionTableModel.COL_PRIORITY, 42)
        self.table_view.setColumnWidth(SuggestionTableModel.COL_IMPORTANCE, 88)
        self.table_view.setColumnWidth(SuggestionTableModel.COL_SCORE, 76)
        self.table_view.setColumnWidth(SuggestionTableModel.COL_AIRMASS, 96)
        self.table_view.setColumnWidth(SuggestionTableModel.COL_HOURS, 92)
        self.table_view.setColumnWidth(SuggestionTableModel.COL_WINDOW, 140)
        self.table_view.setColumnWidth(SuggestionTableModel.COL_MOON_SEP, 90)
        self.table_view.setColumnWidth(SuggestionTableModel.COL_ACTION, 74)
        layout.addWidget(self.table_view, 1)

        self.notes_label = QLabel(self)
        self.notes_label.setWordWrap(True)
        self.notes_label.setVisible(bool(notes))
        layout.addWidget(self.notes_label)

        buttons = QDialogButtonBox(QDialogButtonBox.Close, self)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)

        self.importance_spin.valueChanged.connect(self._apply_filters)
        self.score_spin.valueChanged.connect(self._apply_filters)
        self.hours_spin.valueChanged.connect(self._apply_filters)
        self.moon_sep_spin.valueChanged.connect(self._apply_filters)
        self.airmass_spin.valueChanged.connect(self._apply_filters)
        self.magnitude_spin.valueChanged.connect(self._apply_filters)
        self.reset_filters_btn.clicked.connect(self._reset_filters_to_defaults)
        self.reload_btn.clicked.connect(self._reload_suggestions)
        self.table_model.modelReset.connect(self._refresh_dialog_state)
        self.table_view.clicked.connect(self._on_table_clicked)
        self.table_view.entered.connect(self._on_table_entered)

        self.table_view.sortByColumn(SuggestionTableModel.COL_IMPORTANCE, Qt.DescendingOrder)
        self._apply_filters()

    def _default_name_column_width(self, suggestions: list[dict[str, object]]) -> int:
        name_font = QFont(self.table_view.font())
        name_font.setBold(True)
        name_metrics = QFontMetrics(name_font)
        header_metrics = QFontMetrics(self.table_view.horizontalHeader().font())
        max_width = header_metrics.horizontalAdvance(SuggestionTableModel.headers[SuggestionTableModel.COL_NAME])
        for item in suggestions:
            target = item.get("target")
            if isinstance(target, Target):
                max_width = max(max_width, name_metrics.horizontalAdvance(target.name))
        return max_width + 24

    @Slot()
    def _apply_filters(self) -> None:
        self._save_filter_settings()
        self.table_model.set_filters(
            self.importance_spin.value(),
            self.score_spin.value(),
            self.hours_spin.value(),
            self.moon_sep_spin.value(),
            self.airmass_spin.value(),
            self.magnitude_spin.value(),
        )

    def _restore_filter_settings(self) -> None:
        self.importance_spin.setValue(
            self._settings.value(
                "suggestions/minImportance",
                self._filter_defaults["importance"],
                type=float,
            )
        )
        self.score_spin.setValue(
            self._settings.value(
                "suggestions/minScore",
                self._filter_defaults["score"],
                type=float,
            )
        )
        self.hours_spin.setValue(
            self._settings.value(
                "suggestions/minHours",
                self._filter_defaults["hours"],
                type=float,
            )
        )
        self.moon_sep_spin.setValue(
            self._settings.value(
                "suggestions/minMoonSep",
                self._filter_defaults["moon_sep"],
                type=float,
            )
        )
        self.airmass_spin.setValue(
            self._settings.value(
                "suggestions/maxAirmass",
                self._filter_defaults["airmass"],
                type=float,
            )
        )
        self.magnitude_spin.setValue(
            self._settings.value(
                "suggestions/maxMagnitude",
                self._filter_defaults["magnitude"],
                type=float,
            )
        )

    def _save_filter_settings(self) -> None:
        self._settings.setValue("suggestions/minImportance", self.importance_spin.value())
        self._settings.setValue("suggestions/minScore", self.score_spin.value())
        self._settings.setValue("suggestions/minHours", self.hours_spin.value())
        self._settings.setValue("suggestions/minMoonSep", self.moon_sep_spin.value())
        self._settings.setValue("suggestions/maxAirmass", self.airmass_spin.value())
        self._settings.setValue("suggestions/maxMagnitude", self.magnitude_spin.value())

    @Slot()
    def _reset_filters_to_defaults(self) -> None:
        blockers = [
            QSignalBlocker(self.importance_spin),
            QSignalBlocker(self.score_spin),
            QSignalBlocker(self.hours_spin),
            QSignalBlocker(self.moon_sep_spin),
            QSignalBlocker(self.airmass_spin),
            QSignalBlocker(self.magnitude_spin),
        ]
        self.importance_spin.setValue(self._filter_defaults["importance"])
        self.score_spin.setValue(self._filter_defaults["score"])
        self.hours_spin.setValue(self._filter_defaults["hours"])
        self.moon_sep_spin.setValue(self._filter_defaults["moon_sep"])
        self.airmass_spin.setValue(self._filter_defaults["airmass"])
        self.magnitude_spin.setValue(self._filter_defaults["magnitude"])
        del blockers
        self._apply_filters()

    @Slot()
    def _reload_suggestions(self) -> None:
        if self._reload_callback is None:
            return
        self.reload_btn.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            suggestions, notes = self._reload_callback()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Reload Suggestions", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()
            self.reload_btn.setEnabled(True)

        self._notes = notes
        self.table_model.replace_suggestions(suggestions)
        self.table_view.setColumnWidth(
            SuggestionTableModel.COL_NAME,
            self._default_name_column_width(suggestions),
        )
        self._refresh_dialog_state()

    @Slot()
    def _refresh_dialog_state(self) -> None:
        filtered = self.table_model.filtered_count()
        total = self.table_model.total_count()
        self.summary_label.setText(
            f"Showing {filtered} matching BHTOM targets "
            f"(loaded {total}, base importance ≥ {BHTOM_SUGGESTION_MIN_IMPORTANCE:.1f})."
        )
        if self._notes:
            self.notes_label.setText("Notes: " + " | ".join(self._notes))
            self.notes_label.setVisible(True)
        else:
            self.notes_label.clear()
            self.notes_label.setVisible(False)

    @Slot(QModelIndex)
    def _on_table_clicked(self, index: QModelIndex) -> None:
        if not index.isValid():
            return
        if index.column() == SuggestionTableModel.COL_NAME:
            self._open_bhtom_target(index.row())
            return
        if index.column() != SuggestionTableModel.COL_ACTION:
            return
        item = self.table_model.suggestion_at(index.row())
        if bool(item.get("added_to_plan")):
            return
        self._add_row_to_plan(index.row())

    @Slot(QModelIndex)
    def _on_table_entered(self, index: QModelIndex) -> None:
        hover_row: Optional[int] = index.row() if index.isValid() else None
        name_hover_row: Optional[int] = None
        action_hover_row: Optional[int] = None
        if index.isValid():
            if index.column() == SuggestionTableModel.COL_NAME:
                name_hover_row = index.row()
            elif index.column() == SuggestionTableModel.COL_ACTION:
                item = self.table_model.suggestion_at(index.row())
                if not bool(item.get("added_to_plan")):
                    action_hover_row = index.row()
        self.table_model.set_hover_row(hover_row)
        self.table_model.set_name_hover_row(name_hover_row)
        self.table_model.set_action_hover_row(action_hover_row)
        self.table_view.viewport().setCursor(
            Qt.CursorShape.PointingHandCursor
            if (action_hover_row is not None or name_hover_row is not None)
            else Qt.CursorShape.ArrowCursor
        )

    def eventFilter(self, watched, event):  # noqa: D401
        if watched is self.table_view.viewport() and event.type() == QEvent.Type.Leave:
            self.table_model.set_hover_row(None)
            self.table_model.set_name_hover_row(None)
            self.table_model.set_action_hover_row(None)
            self.table_view.viewport().setCursor(Qt.CursorShape.ArrowCursor)
        return super().eventFilter(watched, event)

    def _open_bhtom_target(self, row: int) -> None:
        item = self.table_model.suggestion_at(row)
        target = item.get("target")
        if not isinstance(target, Target):
            return
        slug = quote(target.name.strip(), safe="")
        if not slug:
            return
        QDesktopServices.openUrl(QUrl(f"{self._bhtom_base_url}/targets/{slug}/"))

    def _add_row_to_plan(self, row: int) -> None:
        item = self.table_model.suggestion_at(row)
        target = item.get("target")
        if not isinstance(target, Target):
            return
        if self._add_callback(target):
            self.table_model.mark_added(target)

# --------------------------------------------------
# --- Astronomy Worker (runs in separate QThread) ---
# --------------------------------------------------
class AstronomyWorker(QThread):
    """Runs astroplan calculations off the GUI thread."""

    finished: Signal = Signal(dict)  # payload with curves & events

    _cache: dict = {}

    def __init__(self, targets: List[Target], settings: SessionSettings, parent=None):
        super().__init__(parent)
        self.setObjectName(self.__class__.__name__)
        self.targets = targets
        self.settings = settings

    # heavy lifting happens here
    def run(self) -> None:  # noqa: D401
        obs_date = self.settings.date
        site = self.settings.site
        observer = Observer(location=site.to_earthlocation(), timezone=site.timezone_name)

        # Caching key: site coords + elevation + calendar date + sample count
        key = (
            site.latitude, site.longitude, site.elevation,
            obs_date.toString("yyyy-MM-dd"),
            self.settings.time_samples
        )
        cache = AstronomyWorker._cache

        # determine local midnight following the chosen observation date’s evening
        tz = pytz.timezone(site.timezone_name)
        next_mid = datetime(obs_date.year(), obs_date.month(), obs_date.day(), 0, 0) + timedelta(days=1)
        local_mid_dt = tz.localize(next_mid)
        midnight = Time(local_mid_dt)

        if (
            key in cache
            and "moon_ra" in cache[key]["events"]
            and "moon_dec" in cache[key]["events"]
        ):
            cached = cache[key]
            times = cached["times"]
            jd = times.plot_date
            events = cached["events"]
        else:
            # compute astronomical dusk/dawn only if it occurs
            with warnings.catch_warnings():
                warnings.filterwarnings('error', category=TargetAlwaysUpWarning)
                try:
                    dusk = observer.twilight_evening_astronomical(midnight, which="nearest")
                    dawn = observer.twilight_morning_astronomical(midnight, which="next")
                    astro_ok = True
                except (TargetAlwaysUpWarning, Exception):
                    astro_ok = False
            # Build a ±12-hour grid around midnight (lower resolution for speed)
            ts = self.settings.time_samples
            times = midnight + np.linspace(-12, 12, ts) * u.hour
            jd = times.plot_date

            # Precompute all solar/lunar/twilight event floats
            dusk_naut = observer.twilight_evening_nautical(midnight, which="nearest")
            dawn_naut = observer.twilight_morning_nautical(midnight, which="next")

            events = {"times": jd}
            if astro_ok:
                events["dusk"] = dusk.plot_date
                events["dawn"] = dawn.plot_date
            events.update({
                "dusk_naut": dusk_naut.plot_date,
                "dawn_naut": dawn_naut.plot_date,
                "dusk_civ": observer.twilight_evening_civil(midnight, which="nearest").plot_date,
                "dawn_civ": observer.twilight_morning_civil(midnight, which="next").plot_date,
                "sunset": observer.sun_set_time(midnight, which="nearest").plot_date,
                "sunrise": observer.sun_rise_time(midnight, which="next").plot_date,
                "moonrise": observer.moon_rise_time(midnight, which="nearest").plot_date,
                "moonset": observer.moon_set_time(midnight, which="next").plot_date,
                "midnight": midnight.plot_date,
            })
            # Compute sun and moon altitudes via PyEphem (fast)
            eph_observer = ephem.Observer()
            eph_observer.lat = str(site.latitude)
            eph_observer.lon = str(site.longitude)
            eph_observer.elevation = site.elevation
            sun = ephem.Sun()
            moon = ephem.Moon()

            sun_alts = []
            sun_azs = []
            moon_alts = []
            moon_azs = []
            moon_ras = []
            moon_decs = []
            for t in times.datetime:
                # PyEphem expects UTC datetime
                eph_observer.date = t
                sun.compute(eph_observer)
                moon.compute(eph_observer)
                # convert radians to degrees
                sun_alts.append(sun.alt * 180.0 / math.pi)
                sun_azs.append(sun.az * 180.0 / math.pi)
                moon_alts.append(moon.alt * 180.0 / math.pi)
                moon_azs.append(moon.az * 180.0 / math.pi)
                moon_ras.append(moon.ra * 180.0 / math.pi)
                moon_decs.append(moon.dec * 180.0 / math.pi)

            events["sun_alt"] = np.array(sun_alts)
            events["sun_az"] = np.array(sun_azs)
            events["moon_alt"] = np.array(moon_alts)
            events["moon_az"] = np.array(moon_azs)
            events["moon_ra"] = np.array(moon_ras)
            events["moon_dec"] = np.array(moon_decs)
            # Moon phase from PyEphem (0–100%)
            events["moon_phase"] = moon.phase
            cache[key] = {"times": times, "events": events}

        # Start payload with cached/global events
        payload: dict[str, object] = {k: v for k, v in {"times": jd, **events}.items()}
        moon_coords = SkyCoord(
            ra=np.array(events["moon_ra"]) * u.deg,
            dec=np.array(events["moon_dec"]) * u.deg,
        )
        for tgt in self.targets:
            fixed = FixedTarget(name=tgt.name, coord=tgt.skycoord)
            altaz = observer.altaz(times, fixed)
            moon_sep = tgt.skycoord.separation(moon_coords).deg
            payload[tgt.name] = {
                "altitude": altaz.alt.deg,    # type: ignore[arg-type]
                "azimuth": altaz.az.deg,  # type: ignore[arg-type]
                "moon_sep": moon_sep,  # type: ignore[arg-type]
            }
        # Tell the GUI which timezone we used
        payload["tz"] = site.timezone_name   
        logger.info("AstronomyWorker finished (%d targets, date %s)",
            len(self.targets),
            obs_date.toString("yyyy-MM-dd"))                   # type: ignore[arg-type]
        self.finished.emit(payload)


# --------------------------------------------------
# --- Local LLM integration ------------------------
# --------------------------------------------------
class LLMConfig:
    """Configuration for a local OpenAI-compatible inference server."""

    DEFAULT_URL = "http://localhost:8080"
    DEFAULT_MODEL = "bitnet-b1.58-3b"
    DEFAULT_TIMEOUT_S = 45

    def __init__(
        self,
        url: str = DEFAULT_URL,
        model: str = DEFAULT_MODEL,
        timeout_s: int = DEFAULT_TIMEOUT_S,
    ) -> None:
        normalized_url = str(url or self.DEFAULT_URL).strip().rstrip("/")
        normalized_model = str(model or self.DEFAULT_MODEL).strip()
        self.url = normalized_url or self.DEFAULT_URL
        self.model = normalized_model or self.DEFAULT_MODEL
        self.timeout_s = max(5, int(timeout_s))


class LLMWorker(QThread):
    """Send a single prompt to a local OpenAI-compatible server."""

    responseReady = Signal(str, str)  # (tag, text)
    responseChunk = Signal(str, str)  # (tag, delta)
    errorOccurred = Signal(str)  # message

    def __init__(
        self,
        config: LLMConfig,
        prompt: str,
        system_prompt: str = "",
        tag: str = "chat",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName(self.__class__.__name__)
        self.config = config
        self.prompt = prompt
        self.system_prompt = system_prompt
        self.tag = tag

    @staticmethod
    def _extract_content(data: object) -> str:
        if not isinstance(data, dict):
            return ""
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict):
                content = message.get("content", "")
                if isinstance(content, str):
                    return content.strip()
                if isinstance(content, list):
                    parts: list[str] = []
                    for item in content:
                        if not isinstance(item, dict):
                            continue
                        if item.get("type") == "text":
                            txt = item.get("text", "")
                            if isinstance(txt, str) and txt.strip():
                                parts.append(txt.strip())
                    if parts:
                        return "\n".join(parts)
            text = first.get("text")
            if isinstance(text, str):
                return text.strip()
        return ""

    @staticmethod
    def _extract_delta_content(data: object) -> str:
        if not isinstance(data, dict):
            return ""
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""
        first = choices[0]
        if not isinstance(first, dict):
            return ""
        delta = first.get("delta")
        if not isinstance(delta, dict):
            return ""
        content = delta.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text":
                    txt = item.get("text", "")
                    if isinstance(txt, str) and txt:
                        parts.append(txt)
            if parts:
                return "".join(parts)
        return ""

    def _consume_sse_stream(self, response) -> str:
        chunks: list[str] = []
        event_lines: list[str] = []

        def _flush_event() -> bool:
            if not event_lines:
                return False
            payload_text = "\n".join(event_lines).strip()
            event_lines.clear()
            if not payload_text:
                return False
            if payload_text == "[DONE]":
                return True
            try:
                decoded = json.loads(payload_text)
            except json.JSONDecodeError:
                return False
            delta = self._extract_delta_content(decoded)
            if delta:
                chunks.append(delta)
                self.responseChunk.emit(self.tag, delta)
                return False
            if not chunks:
                fallback = self._extract_content(decoded)
                if fallback:
                    chunks.append(fallback)
                    self.responseChunk.emit(self.tag, fallback)
            return False

        for raw_line in response:
            if self.isInterruptionRequested():
                break
            line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
            if not line:
                if _flush_event():
                    break
                continue
            if line.startswith("data:"):
                event_lines.append(line[5:].lstrip())

        if not self.isInterruptionRequested() and event_lines:
            _flush_event()
        return "".join(chunks)

    def run(self) -> None:  # noqa: D401
        if self.isInterruptionRequested():
            return
        messages: list[dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": self.prompt})
        payload = json.dumps(
            {
                "model": self.config.model,
                "messages": messages,
                "max_tokens": 1024,
                "temperature": 0.7,
                "stream": True,
            }
        ).encode("utf-8")
        request = Request(
            f"{self.config.url}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.config.timeout_s) as response:
                content_type = response.headers.get("Content-Type", "")
                if "text/event-stream" in content_type:
                    text = self._consume_sse_stream(response)
                else:
                    body = response.read().decode("utf-8", errors="replace")
                    decoded = json.loads(body)
                    text = self._extract_content(decoded)
            if not text:
                raise RuntimeError("LLM response does not contain message content.")
            self.responseReady.emit(self.tag, text)
        except HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="ignore").strip()
            except Exception:
                detail = ""
            if detail:
                self.errorOccurred.emit(f"LLM HTTP {exc.code}: {detail[:280]}")
            else:
                self.errorOccurred.emit(f"LLM HTTP {exc.code}.")
        except URLError as exc:
            reason = getattr(exc, "reason", str(exc))
            self.errorOccurred.emit(
                f"Cannot reach LLM server at {self.config.url}: {reason}\n"
                "Start your local server and verify URL/model in Settings -> General Settings."
            )
        except Exception as exc:  # noqa: BLE001
            self.errorOccurred.emit(f"LLM request failed ({type(exc).__name__}): {exc}")


# --------------------------------------------------
# --- Qt Table Model for the targets list ----------
# --------------------------------------------------
class TargetTableModel(QAbstractTableModel):
    COL_ORDER = 0
    COL_NAME = 1
    COL_RA = 2
    COL_HA = 3
    COL_DEC = 4
    COL_ALT = 5
    COL_AZ = 6
    COL_MOON_SEP = 7
    COL_SCORE = 8
    COL_HOURS = 9
    COL_PRIORITY = 10
    COL_OBSERVED = 11
    COL_ACTIONS = 12
    headers = [
        "Order",
        "Name",
        "RA (°)",
        "HA (h)",
        "Dec (°)",
        "Alt (°)",
        "Az (°)",
        "Moon Sep (°)",
        "Score",
        "Over Lim (h)",
        "Pri",
        "Obs",
        "Actions",
    ]

    def __init__(self, targets: List[Target], site: Optional[Site] = None):
        super().__init__()
        self.setObjectName(self.__class__.__name__)
        self._targets = targets
        self.site = site
        self.limit: float | None = None
        # Cached current values for table display
        self.order_values: list[int] = []
        self.current_alts: list[float] = []
        self.current_azs: list[float] = []
        self.current_seps: list[float] = []
        self.scores: list[float] = []
        self.hours_above_limit: list[float] = []
        self.row_enabled: list[bool] = []
        self.color_map: dict[str, QColor] = {}

    def _ensure_cache_lengths(self) -> None:
        n = len(self._targets)
        if len(self.order_values) < n:
            self.order_values.extend([0] * (n - len(self.order_values)))
        if len(self.current_alts) < n:
            self.current_alts.extend([float("nan")] * (n - len(self.current_alts)))
        if len(self.current_azs) < n:
            self.current_azs.extend([float("nan")] * (n - len(self.current_azs)))
        if len(self.current_seps) < n:
            self.current_seps.extend([float("nan")] * (n - len(self.current_seps)))
        if len(self.scores) < n:
            self.scores.extend([0.0] * (n - len(self.scores)))
        if len(self.hours_above_limit) < n:
            self.hours_above_limit.extend([0.0] * (n - len(self.hours_above_limit)))
        if len(self.row_enabled) < n:
            self.row_enabled.extend([True] * (n - len(self.row_enabled)))

    def reset_targets(self, targets: list[Target]) -> None:
        self.beginResetModel()
        self._targets[:] = targets
        self.order_values = []
        self.current_alts = []
        self.current_azs = []
        self.current_seps = []
        self.scores = []
        self.hours_above_limit = []
        self.row_enabled = []
        self.endResetModel()

    def append_target(self, target: Target) -> None:
        row = len(self._targets)
        self.beginInsertRows(QModelIndex(), row, row)
        self._targets.append(target)
        self.order_values.append(0)
        self.endInsertRows()

    def remove_rows(self, rows: list[int]) -> list[Target]:
        removed: list[Target] = []
        for row in sorted(rows, reverse=True):
            if not (0 <= row < len(self._targets)):
                continue
            self.beginRemoveRows(QModelIndex(), row, row)
            removed.append(self._targets.pop(row))
            if row < len(self.order_values):
                self.order_values.pop(row)
            if row < len(self.current_alts):
                self.current_alts.pop(row)
            if row < len(self.current_azs):
                self.current_azs.pop(row)
            if row < len(self.current_seps):
                self.current_seps.pop(row)
            if row < len(self.scores):
                self.scores.pop(row)
            if row < len(self.hours_above_limit):
                self.hours_above_limit.pop(row)
            if row < len(self.row_enabled):
                self.row_enabled.pop(row)
            self.endRemoveRows()
        return removed

    # basic model plumbing
    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return len(self._targets)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return len(self.headers)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):  # noqa: N802
        if not index.isValid():
            return None
        self._ensure_cache_lengths()
        tgt = self._targets[index.row()]
        col = index.column()
        row = index.row()
        row_is_enabled = self.row_enabled[row] if row < len(self.row_enabled) else True

        # Center-align all cell text except left-align names
        if role == Qt.TextAlignmentRole:
            # Left-align names, center others
            if col == self.COL_NAME:
                return Qt.AlignLeft | Qt.AlignVCenter
            return Qt.AlignCenter | Qt.AlignVCenter

        # Column 2: Hour Angle (hours, sexagesimal)
        if role in (Qt.DisplayRole, Qt.EditRole) and col == self.COL_HA and self.site:
            now = Time.now()
            loc = self.site.to_earthlocation()
            lst = now.sidereal_time('apparent', loc.lon).hour    # LST in hours
            ra_h = self._targets[row].ra / 15.0                   # RA in hours
            ha = (lst - ra_h + 24) % 24
            # Hour Angle in sexagesimal
            ha_angle = Angle(ha, u.hour)
            return ha_angle.to_string(unit=u.hour, sep=":", pad=True, precision=0)

        if role == Qt.BackgroundRole and not row_is_enabled:
            return QBrush(QColor("#ececec"))

        # Combined row background and per-cell highlights
        if role == Qt.BackgroundRole and self.site and self.limit is not None:
            alt = self.current_alts[row] if row < len(self.current_alts) else float("nan")
            if math.isnan(alt):
                return None

            # Plot line color for this target (custom per object or palette fallback)
            colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
            fallback_css = colors[row % len(colors)] if colors else "#4da3ff"
            plot_css = _normalized_css_color(tgt.plot_color) or fallback_css
            plot_color = QColor(plot_css)

            # Highlight in Alt column using user-configured colors
            if col == self.COL_ALT:
                hc = getattr(self, "highlight_colors", {})
                if alt < 0:
                    return QBrush(hc.get("below", QColor("#ff8080")))
                if alt < self.limit:
                    return QBrush(hc.get("limit", QColor("#ffff80")))
                return QBrush(hc.get("above", QColor("#b3ffb3")))

            # Name cell colored by plot color
            if col == self.COL_NAME:
                # Use stored color_map for consistent colors across sort
                brush_color = self.color_map.get(tgt.name)
                if brush_color:
                    return QBrush(brush_color)
                # Fallback to default by-row color
                return QBrush(plot_color)

            # 3) Otherwise, soft row background
            if alt >= self.limit:
                return QBrush(QColor("#d4ffd4"))
            if alt > 0:
                return QBrush(QColor("#fff5d4"))
            return QBrush(QColor("#ffd4d4"))

        if role == Qt.ForegroundRole:
            if not row_is_enabled:
                return QBrush(QColor("#777777"))
            return QBrush(QColor("#000000"))

        if role == Qt.ToolTipRole and col == self.COL_NAME:
            display_type = tgt.object_type
            parent_widget = self.parent()
            if _object_type_is_unknown(display_type) and parent_widget is not None and hasattr(parent_widget, "_bhtom_type_for_target"):
                try:
                    fallback_type = parent_widget._bhtom_type_for_target(tgt)
                except Exception:
                    fallback_type = ""
                if fallback_type:
                    display_type = fallback_type
            extras: list[str] = []
            if display_type and not _object_type_is_unknown(display_type):
                extras.append(f"Type: {display_type}")
            if tgt.magnitude is not None:
                extras.append(f"{_target_magnitude_label(tgt)}: {tgt.magnitude:.1f}")
            if tgt.size_arcmin is not None:
                extras.append(f"Size: {tgt.size_arcmin:.1f}'")
            extras.append(f"Priority: {tgt.priority}")
            extras.append(f"Observed: {'Yes' if tgt.observed else 'No'}")
            if tgt.notes:
                extras.append(f"Notes: {tgt.notes}")
            return "\n".join([tgt.name, *extras])

        if role == Qt.ToolTipRole and col == self.COL_ORDER:
            order_value = self.order_values[row] if row < len(self.order_values) else 0
            if order_value > 0:
                return f"Recommended observing order: {order_value}"
            return "No deterministic observing order is available for this row yet."

        # Tooltip for altitude status in altitude column
        if role == Qt.ToolTipRole and col == self.COL_ALT and self.site:
            alt = self.current_alts[row] if row < len(self.current_alts) else None
            limit = self.limit or 0
            if alt is None or math.isnan(alt):
                return ""
            if alt < 0:
                return "Below horizon"
            if alt < limit:
                return "Below limit altitude"
            return "Above limit altitude"

        if role not in (Qt.DisplayRole, Qt.EditRole):
            return None
        if col == self.COL_ORDER:
            order_value = self.order_values[row] if row < len(self.order_values) else 0
            return str(order_value) if order_value > 0 else ""
        if col == self.COL_NAME:
            return tgt.name
        if col == self.COL_RA:
            # Right Ascension in sexagesimal (hours)
            ra_angle = Angle(tgt.ra, u.degree)
            return ra_angle.to_string(unit='hourangle', sep=":", pad=True, precision=0)
        if col == self.COL_DEC:
            # Declination in sexagesimal (degrees)
            dec_angle = Angle(tgt.dec, u.degree)
            return dec_angle.to_string(unit='deg', sep=":", alwayssign=True, pad=True, precision=0)
        if col == self.COL_ALT:
            return f"{self.current_alts[row]:.1f}" if row < len(self.current_alts) and not math.isnan(self.current_alts[row]) else ""
        if col == self.COL_AZ:
            return f"{self.current_azs[row]:.1f}" if row < len(self.current_azs) and not math.isnan(self.current_azs[row]) else ""
        if col == self.COL_MOON_SEP:
            return f"{self.current_seps[row]:.1f}" if row < len(self.current_seps) and not math.isnan(self.current_seps[row]) else ""
        if col == self.COL_SCORE:
            return f"{self.scores[row]:.1f}" if row < len(self.scores) else "0.0"
        if col == self.COL_HOURS:
            return f"{self.hours_above_limit[row]:.2f}" if row < len(self.hours_above_limit) else "0.00"
        if col == self.COL_PRIORITY:
            return str(tgt.priority)
        if col == self.COL_OBSERVED:
            return "Yes" if tgt.observed else "No"
        if col == self.COL_ACTIONS:
            return ""
        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole):  # noqa: N802
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self.headers[section]
        return None

    def sort(self, column: int, order: Qt.SortOrder = Qt.AscendingOrder) -> None:
        """Sort the targets by the given column index."""
        self.layoutAboutToBeChanged.emit()
        reverse = (order == Qt.DescendingOrder)
        self._ensure_cache_lengths()
        n = len(self._targets)
        rows = []
        for idx, tgt in enumerate(self._targets):
            rows.append({
                "target": tgt,
                "order": self.order_values[idx] if idx < len(self.order_values) else 0,
                "alt": self.current_alts[idx] if idx < len(self.current_alts) else float("-inf"),
                "az": self.current_azs[idx] if idx < len(self.current_azs) else float("-inf"),
                "sep": self.current_seps[idx] if idx < len(self.current_seps) else float("-inf"),
                "score": self.scores[idx] if idx < len(self.scores) else 0.0,
                "hours": self.hours_above_limit[idx] if idx < len(self.hours_above_limit) else 0.0,
                "enabled": self.row_enabled[idx] if idx < len(self.row_enabled) else True,
            })

        if column == self.COL_ORDER:
            rows.sort(
                key=lambda r: (r["order"] <= 0, int(r["order"]) if r["order"] > 0 else 10**9),
                reverse=reverse,
            )
        elif column == self.COL_NAME:
            rows.sort(key=lambda r: r["target"].name.lower(), reverse=reverse)
        elif column == self.COL_RA:
            rows.sort(key=lambda r: r["target"].ra, reverse=reverse)
        elif column == self.COL_HA and self.site:
            now = Time.now()
            lon = self.site.to_earthlocation().lon
            lst = now.sidereal_time('apparent', lon).hour
            rows.sort(
                key=lambda r: (float(lst) - (r["target"].ra / 15.0) + 24.0) % 24.0,  # type: ignore[arg-type]
                reverse=reverse,
            )
        elif column == self.COL_DEC:
            rows.sort(key=lambda r: r["target"].dec, reverse=reverse)
        elif column == self.COL_ALT:
            rows.sort(key=lambda r: r["alt"], reverse=reverse)
        elif column == self.COL_AZ:
            rows.sort(key=lambda r: r["az"], reverse=reverse)
        elif column == self.COL_MOON_SEP:
            rows.sort(key=lambda r: r["sep"], reverse=reverse)
        elif column == self.COL_SCORE:
            rows.sort(key=lambda r: r["score"], reverse=reverse)
        elif column == self.COL_HOURS:
            rows.sort(key=lambda r: r["hours"], reverse=reverse)
        elif column == self.COL_PRIORITY:
            rows.sort(key=lambda r: r["target"].priority, reverse=reverse)
        elif column == self.COL_OBSERVED:
            rows.sort(key=lambda r: r["target"].observed, reverse=reverse)

        if rows:
            self._targets[:] = [row["target"] for row in rows]
            self.order_values = [int(row["order"]) for row in rows]
            self.current_alts = [float(row["alt"]) for row in rows]
            self.current_azs = [float(row["az"]) for row in rows]
            self.current_seps = [float(row["sep"]) for row in rows]
            self.scores = [float(row["score"]) for row in rows]
            self.hours_above_limit = [float(row["hours"]) for row in rows]
            self.row_enabled = [bool(row["enabled"]) for row in rows]
        elif n == 0:
            self.order_values = []
            self.current_alts = []
            self.current_azs = []
            self.current_seps = []
            self.scores = []
            self.hours_above_limit = []
            self.row_enabled = []

        self.layoutChanged.emit()

    # minimal drag‑reorder support
    def flags(self, index: QModelIndex):  # noqa: N802
        if not index.isValid():
            return Qt.ItemIsEnabled
        return Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsDragEnabled | Qt.ItemIsDropEnabled

    def supportedDropActions(self):  # noqa: N802
        return Qt.MoveAction

    def mimeTypes(self):  # noqa: N802
        return ["application/x-target-row"]

    def mimeData(self, indexes):  # noqa: N802
        mimedata = super().mimeData(indexes)
        rows = sorted({idx.row() for idx in indexes})
        mimedata.setData("application/x-target-row", json.dumps(rows).encode())
        return mimedata

    def dropMimeData(self, mimedata, action, row, column, parent):  # noqa: N802
        if action == Qt.IgnoreAction or not mimedata.hasFormat("application/x-target-row"):
            return False
        self._ensure_cache_lengths()
        rows = json.loads(bytes(mimedata.data("application/x-target-row")).decode())
        rows = sorted(set(int(r) for r in rows))
        if not rows:
            return False

        insert_row = row if row >= 0 else (parent.row() if parent.isValid() else len(self._targets))
        insert_row = min(max(insert_row, 0), len(self._targets))
        for r in rows:
            if r < insert_row:
                insert_row -= 1

        bundles = []
        for r in rows:
            bundles.append((
                self._targets[r],
                self.order_values[r] if r < len(self.order_values) else 0,
                self.current_alts[r] if r < len(self.current_alts) else float("nan"),
                self.current_azs[r] if r < len(self.current_azs) else float("nan"),
                self.current_seps[r] if r < len(self.current_seps) else float("nan"),
                self.scores[r] if r < len(self.scores) else 0.0,
                self.hours_above_limit[r] if r < len(self.hours_above_limit) else 0.0,
                self.row_enabled[r] if r < len(self.row_enabled) else True,
            ))

        for r in reversed(rows):
            self._targets.pop(r)
            if r < len(self.order_values):
                self.order_values.pop(r)
            if r < len(self.current_alts):
                self.current_alts.pop(r)
            if r < len(self.current_azs):
                self.current_azs.pop(r)
            if r < len(self.current_seps):
                self.current_seps.pop(r)
            if r < len(self.scores):
                self.scores.pop(r)
            if r < len(self.hours_above_limit):
                self.hours_above_limit.pop(r)
            if r < len(self.row_enabled):
                self.row_enabled.pop(r)

        for offset, bundle in enumerate(bundles):
            pos = insert_row + offset
            tgt, order_value, alt, az, sep, score, hours, enabled = bundle
            self._targets.insert(pos, tgt)
            self.order_values.insert(pos, int(order_value))
            self.current_alts.insert(pos, alt)
            self.current_azs.insert(pos, az)
            self.current_seps.insert(pos, sep)
            self.scores.insert(pos, score)
            self.hours_above_limit.insert(pos, hours)
            self.row_enabled.insert(pos, enabled)
        self.layoutChanged.emit()
        return True


# --------------------------------------------------
# --- Main Window ----------------------------------
# --------------------------------------------------
class MainWindow(QMainWindow):
    darkModeChanged = Signal(bool)

    def _update_night_details_constraints(self):
        if not hasattr(self, "info_widget") or not hasattr(self, "info_card"):
            return
        self.info_widget.adjustSize()
        content_hint = self.info_widget.minimumSizeHint()
        content_h = max(self.info_widget.sizeHint().height(), content_hint.height())
        content_w = max(self.info_widget.sizeHint().width(), content_hint.width())
        title_h = self.info_title_label.sizeHint().height() if hasattr(self, "info_title_label") else 20
        card_min_h = max(220, content_h + title_h + 30)
        card_min_w = max(330, content_w + 26)
        self.info_card.setMinimumHeight(card_min_h)
        self.info_card.setMinimumWidth(card_min_w)
        if hasattr(self, "info_scroll"):
            self.info_scroll.setMinimumHeight(content_h + 6)
            self.info_scroll.setMinimumWidth(content_w + 8)
        if hasattr(self, "right_dashboard"):
            self.right_dashboard.setMinimumWidth(max(460, card_min_w + 22))

    def _pick_font_family(self, candidates: list[str]) -> str:
        available = set(QFontDatabase.families())
        for family in candidates:
            if family in available:
                return family
        return QApplication.font().family()

    def _pick_matplotlib_font_family(self, candidates: list[str]) -> str:
        available = {f.name for f in mpl_font_manager.fontManager.ttflist}
        for family in candidates:
            if family in available:
                return family
        return "DejaVu Sans"

    def _set_dark_mode_enabled(self, enabled: bool, persist: bool = True):
        enabled = bool(enabled)
        if enabled == getattr(self, "_dark_enabled", False):
            if persist:
                self.settings.setValue("general/darkMode", enabled)
            return
        self._dark_enabled = enabled
        if persist:
            self.settings.setValue("general/darkMode", self._dark_enabled)
        self._apply_styles()
        self.darkModeChanged.emit(self._dark_enabled)

    def _apply_styles(self):
        """Apply a custom stylesheet, fonts, and default icon sizes."""
        body_family = self._pick_font_family(
            ["SF Pro Text", "Avenir Next", "Inter", "Segoe UI", "Noto Sans", "Helvetica Neue", "Arial"]
        )
        display_family = self._pick_font_family(
            ["SF Pro Display", "Avenir Next", "Inter", "Segoe UI Semibold", body_family]
        )
        font_size = max(9, min(16, int(getattr(self, "_ui_font_size", 11))))
        app_font = QFont(body_family)
        app_font.setPointSize(font_size)
        QApplication.setFont(app_font)
        mpl_family = self._pick_matplotlib_font_family(
            [body_family, display_family, "DejaVu Sans", "Arial", "Helvetica"]
        )
        plt.rcParams["font.family"] = [mpl_family, "DejaVu Sans", "Arial", "Helvetica"]
        self._theme_name = normalize_theme_key(getattr(self, "_theme_name", DEFAULT_UI_THEME))
        self.setStyleSheet(
            build_stylesheet(
                self._theme_name,
                dark_enabled=getattr(self, "_dark_enabled", False),
                ui_font_size=font_size,
                font_family=body_family,
                display_font_family=display_family,
            )
        )
        self._refresh_date_nav_icons()
        self._refresh_plot_mode_switch()
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
            base = QColor("#d8e6f5")
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

    def _line_palette(self) -> list[str]:
        return COLORBLIND_LINE_COLORS if self.color_blind_mode else DEFAULT_LINE_COLORS

    def _target_plot_color_css(
        self,
        target: Target,
        index: int,
        palette: Optional[list[str]] = None,
    ) -> str:
        custom_css = _normalized_css_color(target.plot_color)
        if custom_css:
            return custom_css
        use_palette = palette if palette is not None else self._line_palette()
        if not use_palette:
            return "#4da3ff"
        return str(use_palette[index % len(use_palette)])

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

    def _configure_main_plot_y_axis(self) -> None:
        if not self._plot_airmass:
            self.ax_alt.set_ylabel("Altitude (°)")
            self.ax_alt.set_ylim(0, 90)
            self.ax_alt.set_yticks([0, 15, 30, 45, 60, 75, 90])
            return

        limit_airmass = self._plot_limit_value()
        max_airmass = max(3.0, min(8.0, float(math.ceil(limit_airmass + 1.0))))
        ticks = [1.0, 1.1, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]
        visible_ticks = [tick for tick in ticks if tick <= max_airmass]
        if not visible_ticks or visible_ticks[0] != 1.0:
            visible_ticks.insert(0, 1.0)
        self.ax_alt.set_ylabel("Airmass")
        self.ax_alt.set_ylim(max_airmass, 1.0)
        self.ax_alt.set_yticks(visible_ticks)
        self.ax_alt.set_yticklabels([f"{tick:.1f}" for tick in visible_ticks])

    def _refresh_plot_mode_switch(self) -> None:
        if not hasattr(self, "plot_mode_alt_label") or not hasattr(self, "plot_mode_airmass_label"):
            return
        base = self.palette().color(QPalette.ColorRole.WindowText)
        active_color = f"rgba({base.red()}, {base.green()}, {base.blue()}, 255)"
        inactive_color = f"rgba({base.red()}, {base.green()}, {base.blue()}, 150)"
        self.plot_mode_alt_label.setStyleSheet(
            f"font-weight: {'700' if not self._plot_airmass else '500'}; color: {active_color if not self._plot_airmass else inactive_color};"
        )
        self.plot_mode_airmass_label.setStyleSheet(
            f"font-weight: {'700' if self._plot_airmass else '500'}; color: {active_color if self._plot_airmass else inactive_color};"
        )

    def _reset_plot_navigation_home(self) -> None:
        if not hasattr(self, "plot_toolbar"):
            return
        try:
            self.plot_toolbar.update()
            self.plot_toolbar.push_current()
        except Exception:
            pass

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

    def _cutout_fov_text(self, width_px: int, height_px: int) -> str:
        fov_x, fov_y = self._cutout_fov_axes_arcmin(width_px, height_px)
        return f"{fov_x:.1f}x{fov_y:.1f} arcmin"

    def _cutout_key_for_target(self, target: Target, width_px: int, height_px: int) -> str:
        return (
            f"{self._cutout_survey_key}:{self._cutout_fov_arcmin}:{width_px}x{height_px}:"
            f"{target.ra:.6f},{target.dec:.6f}"
        )

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
        sig = self._cutout_resize_signature_for_target(target)
        if sig is None or sig == self._cutout_last_resize_signature:
            return
        self._cutout_last_resize_signature = sig
        self._update_cutout_preview_for_target(target)

    def _set_cutout_placeholder(self, text: str):
        if not hasattr(self, "aladin_image_label"):
            return
        for label in (self.aladin_image_label, getattr(self, "finder_image_label", None)):
            if label is None:
                continue
            label.setPixmap(QPixmap())
            label.setText(text)

    def _build_aladin_overlay_pixmap(self, source: QPixmap) -> QPixmap:
        if source.isNull():
            return source
        view = source.copy()
        painter = QPainter(view)
        painter.setRenderHint(QPainter.Antialiasing, True)
        w = view.width()
        h = view.height()
        cx = w // 2
        cy = h // 2
        radius = max(12, int(min(w, h) * 0.14))

        # Crosshair and center ring for quick visual centering.
        pen_cross = QPen(QColor(255, 84, 84, 225))
        pen_cross.setWidth(max(1, int(min(w, h) * 0.01)))
        painter.setPen(pen_cross)
        span = max(18, int(min(w, h) * 0.18))
        painter.drawLine(cx - span, cy, cx + span, cy)
        painter.drawLine(cx, cy - span, cx, cy + span)
        pen_ring = QPen(QColor(250, 250, 250, 210))
        pen_ring.setWidth(max(1, int(min(w, h) * 0.008)))
        painter.setPen(pen_ring)
        painter.drawEllipse(cx - radius, cy - radius, radius * 2, radius * 2)

        # Keep survey/FOV visible directly on Aladin image.
        strip_h = max(16, int(h * 0.10))
        painter.fillRect(0, h - strip_h, w, strip_h, QColor(9, 14, 22, 165))
        painter.setPen(QColor(238, 244, 252, 240))
        meta_txt = f"{_cutout_survey_label(self._cutout_survey_key)} | {self._cutout_fov_text(w, h)}"
        painter.drawText(8, h - strip_h, w - 16, strip_h, Qt.AlignVCenter | Qt.AlignLeft, meta_txt)
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
        pen_cross = QPen(QColor(255, 84, 84, 225))
        pen_cross.setWidth(max(1, int(min(w, h) * 0.010)))
        painter.setPen(pen_cross)
        span = max(16, int(min(w, h) * 0.10))
        gap = max(6, int(span * 0.34))
        painter.drawLine(cx - span, cy, cx - gap, cy)
        painter.drawLine(cx + gap, cy, cx + span, cy)
        painter.drawLine(cx, cy - span, cx, cy - gap)
        painter.drawLine(cx, cy + gap, cx, cy + span)

        strip_h = max(16, int(h * 0.10))
        painter.fillRect(0, h - strip_h, w, strip_h, QColor(9, 14, 22, 165))
        painter.setPen(QColor(238, 244, 252, 240))
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

    def _cache_cutout_pixmap(self, key: str, pixmap: QPixmap):
        self._cutout_cache[key] = pixmap
        if key in self._cutout_cache_order:
            self._cutout_cache_order.remove(key)
        self._cutout_cache_order.append(key)
        while len(self._cutout_cache_order) > CUTOUT_CACHE_MAX:
            stale = self._cutout_cache_order.pop(0)
            self._cutout_cache.pop(stale, None)

    def _cache_finder_pixmap(self, key: str, pixmap: QPixmap):
        self._finder_cache[key] = pixmap
        if key in self._finder_cache_order:
            self._finder_cache_order.remove(key)
        self._finder_cache_order.append(key)
        while len(self._finder_cache_order) > CUTOUT_CACHE_MAX:
            stale = self._finder_cache_order.pop(0)
            self._finder_cache.pop(stale, None)

    def _show_finder_aladin_fallback(self, key: str, text_if_missing: str) -> bool:
        if not hasattr(self, "finder_image_label"):
            return False
        fallback = self._cutout_cache.get(key)
        if fallback is not None and not fallback.isNull():
            self.finder_image_label.setText("")
            self.finder_image_label.setPixmap(self._build_aladin_overlay_pixmap(fallback))
            return True
        self.finder_image_label.setPixmap(QPixmap())
        self.finder_image_label.setText(text_if_missing)
        return False

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
                        stopped = worker.wait(150)
                    except Exception:
                        stopped = False
                    if not stopped:
                        try:
                            worker.terminate()
                            worker.wait(250)
                        except Exception:
                            pass

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
        self._finder_pending_key = ""

        if err or not payload:
            status_text = "Finder: unavailable"
            if err and err.lower() == "cancelled":
                status_text = "Finder: cancelled"
            self._set_finder_status(status_text, busy=False)
            if err and err.lower() != "cancelled":
                logger.warning("Finder chart generation failed for key '%s': %s", key, err)
                self._finder_retry_after[key] = perf_counter() + FINDER_RETRY_COOLDOWN_S
            self._show_finder_aladin_fallback(key, "Finder chart unavailable")
            return

        image = QImage.fromData(payload, "PNG")
        if image.isNull():
            self._set_finder_status("Finder: decode failed", busy=False)
            self._finder_retry_after[key] = perf_counter() + FINDER_RETRY_COOLDOWN_S
            self._show_finder_aladin_fallback(key, "Finder chart decode failed")
            return
        pix = QPixmap.fromImage(image)
        if pix.isNull():
            self._set_finder_status("Finder: decode failed", busy=False)
            self._finder_retry_after[key] = perf_counter() + FINDER_RETRY_COOLDOWN_S
            self._show_finder_aladin_fallback(key, "Finder chart decode failed")
            return
        pix_with_overlay = self._build_finder_overlay_pixmap(pix)
        self._cache_finder_pixmap(key, pix_with_overlay)
        self._finder_retry_after.pop(key, None)
        self._set_finder_status("Finder: ready", busy=False)
        if hasattr(self, "finder_image_label"):
            self.finder_image_label.setText("")
            self.finder_image_label.setPixmap(pix_with_overlay)

    def _on_finder_chart_worker_finished(self, worker: FinderChartWorker):
        workers = getattr(self, "_finder_workers", None)
        if isinstance(workers, list) and worker in workers:
            workers.remove(worker)
        if self._finder_worker is worker:
            self._finder_worker = None

    @Slot()
    def _on_finder_chart_timeout(self):
        key = self._finder_pending_key
        if not key:
            return
        self._finder_pending_key = ""
        self._finder_request_id += 1
        self._finder_retry_after[key] = perf_counter() + FINDER_RETRY_COOLDOWN_S
        self._stop_finder_workers(aggressive=True)
        self._set_finder_status("Finder: timeout", busy=False)
        self._show_finder_aladin_fallback(key, "Finder chart timeout")
        logger.warning("Finder chart timed out for key '%s'", key)

    def _update_finder_chart_for_target(self, target: Target, key: str):
        if not hasattr(self, "finder_image_label"):
            return
        cached = self._finder_cache.get(key)
        if cached is not None and not cached.isNull():
            if hasattr(self, "_finder_timeout_timer"):
                self._finder_timeout_timer.stop()
            self.finder_image_label.setText("")
            self.finder_image_label.setPixmap(cached)
            self._set_finder_status("Finder: cached", busy=False)
            return
        if self._finder_pending_key == key:
            self._set_finder_status("Finder: loading...", busy=True)
            return

        retry_after = float(self._finder_retry_after.get(key, 0.0))
        now = perf_counter()
        if retry_after > now:
            secs = max(1, int(round(retry_after - now)))
            self._set_finder_status(f"Finder: retry in {secs}s", busy=False)
            self._show_finder_aladin_fallback(key, f"Finder chart unavailable ({secs}s)")
            return

        self.finder_image_label.setPixmap(QPixmap())
        self.finder_image_label.setText("Loading finder chart...")

        self._finder_request_id += 1
        req_id = self._finder_request_id
        self._finder_pending_key = key
        render_w, render_h = self._cutout_render_dimensions_px(getattr(self, "finder_image_label", None))
        old_worker = self._finder_worker
        if old_worker is not None:
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
        self._set_finder_status("Finder: loading...", busy=True)
        worker.start()
        if hasattr(self, "_finder_timeout_timer"):
            self._finder_timeout_timer.start()

    def _update_cutout_preview_for_target(self, target: Optional[Target]):
        if not hasattr(self, "cutout_image_label"):
            return

        if target is None:
            if self._cutout_reply is not None and not self._cutout_reply.isFinished():
                self._cutout_reply.abort()
            self._cutout_reply = None
            self._cutout_pending_key = ""
            self._cutout_pending_name = ""
            self._cutout_displayed_key = ""
            self._set_cutout_placeholder("Select a target")
            self._set_finder_status("Finder: idle", busy=False)
            return

        render_w, render_h = self._cutout_render_dimensions_px(getattr(self, "aladin_image_label", None))
        key = self._cutout_key_for_target(target, render_w, render_h)
        self._cutout_last_resize_signature = self._cutout_resize_signature_for_target(target)
        show_finder = getattr(self, "_cutout_view_key", "aladin") == "finderchart"
        if show_finder:
            self._update_finder_chart_for_target(target, key)

        if key in self._cutout_cache:
            aladin_pix = self._cutout_cache[key]
            self.aladin_image_label.setText("")
            self.aladin_image_label.setPixmap(self._build_aladin_overlay_pixmap(aladin_pix))
            self._cutout_displayed_key = key
            return

        if key == self._cutout_pending_key and self._cutout_reply is not None and not self._cutout_reply.isFinished():
            return

        if self._cutout_reply is not None and not self._cutout_reply.isFinished():
            self._cutout_reply.abort()
        self._cutout_reply = None

        self._set_cutout_placeholder("Loading...")
        self._cutout_request_id += 1
        self._cutout_pending_key = key
        self._cutout_pending_name = target.name

        query = QUrlQuery()
        query.addQueryItem("hips", _cutout_survey_hips(self._cutout_survey_key))
        query.addQueryItem("ra", f"{target.ra:.8f}")
        query.addQueryItem("dec", f"{target.dec:.8f}")
        fov_x_arcmin, _ = self._cutout_fov_axes_arcmin(render_w, render_h)
        query.addQueryItem("fov", f"{(fov_x_arcmin / 60.0):.6f}")
        query.addQueryItem("width", str(render_w))
        query.addQueryItem("height", str(render_h))
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
        self._cutout_reply = reply

    @Slot(QNetworkReply)
    def _on_cutout_reply(self, reply: QNetworkReply):
        req_id = int(reply.property("cutout_request_id") or 0)
        key = str(reply.property("cutout_key") or "")
        target_name = str(reply.property("cutout_name") or "").strip()

        # Ignore stale or unrelated replies.
        if req_id != self._cutout_request_id or key != self._cutout_pending_key:
            reply.deleteLater()
            return

        self._cutout_reply = None
        self._cutout_pending_key = ""
        self._cutout_pending_name = ""

        if reply.error() != QNetworkReply.NoError:
            err = reply.error()
            if err != QNetworkReply.OperationCanceledError:
                logger.warning("Cutout fetch failed for '%s': %s", target_name or key, reply.errorString())
                self._set_cutout_placeholder("Preview unavailable")
            reply.deleteLater()
            return

        payload = bytes(reply.readAll())
        reply.deleteLater()
        pixmap = QPixmap()
        if not payload or not pixmap.loadFromData(payload):
            self._set_cutout_placeholder("Preview decode failed")
            return

        self._cache_cutout_pixmap(key, pixmap)
        self.aladin_image_label.setText("")
        self.aladin_image_label.setPixmap(self._build_aladin_overlay_pixmap(pixmap))
        target_lc = target_name.strip().lower()
        target = next((t for t in self.targets if t.name.strip().lower() == target_lc), None)
        if target is not None and hasattr(self, "finder_image_label"):
            should_render_finder = getattr(self, "_cutout_view_key", "aladin") == "finderchart"
            if should_render_finder:
                self._update_finder_chart_for_target(target, key)
        self._cutout_displayed_key = key

    def _clear_cutout_cache(self):
        if self._cutout_reply is not None and not self._cutout_reply.isFinished():
            self._cutout_reply.abort()
        self._cancel_finder_chart_worker()
        self._cutout_reply = None
        self._cutout_cache.clear()
        self._cutout_cache_order.clear()
        self._finder_cache.clear()
        self._finder_cache_order.clear()
        self._finder_retry_after.clear()
        self._cutout_last_resize_signature = None
        self._cutout_displayed_key = ""
        self._cutout_pending_key = ""
        self._cutout_pending_name = ""

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

    @Slot(str)
    def _on_obs_change(self, name: str):
        logger.info("Observatory switched to %s", name)
        """Populate site fields when an observatory is selected."""
        site = self.observatories.get(name)
        if site is None:
            return
        self.lat_edit.setText(f"{site.latitude}")
        self.lon_edit.setText(f"{site.longitude}")
        self.elev_edit.setText(f"{site.elevation}")
        # Update the table model and replot with debounce
        self.table_model.site = site
        self._clear_table_dynamic_cache()
        self.target_metrics.clear()
        self.target_windows.clear()
        # Reset color mapping until new calculation assigns fresh colors
        self.table_model.color_map.clear()
        self.table_model.layoutChanged.emit()
        self._validate_site_inputs()
        if hasattr(self, "progress"):
            self._replot_timer.start()
        # Restart clock worker to update real-time altitudes for the new site
        self._start_clock_worker()

    def _observatories_config_path(self) -> Path:
        return Path(__file__).resolve().parent / "config" / "observatories.json"

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

    def _parse_custom_observatories_payload(self, payload: object) -> dict[str, Site]:
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
        for item in items:
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            try:
                loaded[name] = Site(
                    name=name,
                    latitude=float(item.get("latitude", 0.0)),
                    longitude=float(item.get("longitude", 0.0)),
                    elevation=float(item.get("elevation", 0.0)),
                )
            except Exception as exc:
                logger.warning("Skipping invalid saved observatory %r: %s", name, exc)
        return loaded

    def _load_custom_observatories(self) -> dict[str, Site]:
        cfg_path = self._observatories_config_path()
        loaded: dict[str, Site] = {}
        if cfg_path.exists():
            try:
                payload = json.loads(cfg_path.read_text(encoding="utf-8"))
                loaded = self._parse_custom_observatories_payload(payload)
            except Exception as exc:
                logger.warning("Failed to parse %s: %s", cfg_path, exc)

        # One-time migration from older QSettings location.
        if loaded:
            return loaded
        raw = self.settings.value("general/customObservatories", "", type=str)
        if not raw:
            return {}
        try:
            migrated = self._parse_custom_observatories_payload(json.loads(raw))
        except Exception:
            logger.warning("Failed to parse legacy custom observatories from settings.")
            return {}
        if migrated:
            loaded = migrated
            self._save_custom_observatories(migrated)
            self.settings.remove("general/customObservatories")
            logger.info("Migrated %d observatories to %s", len(migrated), cfg_path)
        return loaded

    def _save_custom_observatories(self, custom_sites: Optional[dict[str, Site]] = None):
        if custom_sites is None:
            source_sites = self.observatories if hasattr(self, "observatories") else {}
        else:
            source_sites = custom_sites
        observatory_items: list[dict[str, object]] = []
        for name in sorted(source_sites.keys(), key=str.lower):
            site = source_sites[name]
            observatory_items.append(
                {
                    "name": name,
                    "latitude": float(site.latitude),
                    "longitude": float(site.longitude),
                    "elevation": float(site.elevation),
                }
            )
        cfg_path = self._observatories_config_path()
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "observatories": observatory_items,
        }
        cfg_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _refresh_observatory_combo(self, selected_name: Optional[str] = None):
        current = selected_name or self.obs_combo.currentText()
        self.obs_combo.blockSignals(True)
        self.obs_combo.clear()
        self.obs_combo.addItems(self.observatories.keys())
        self.obs_combo.blockSignals(False)
        self._update_obs_combo_widths()
        if current in self.observatories:
            self.obs_combo.setCurrentText(current)
        elif self.obs_combo.count() > 0:
            self.obs_combo.setCurrentIndex(0)

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
    def _open_add_observatory_dialog(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Add Observatory")
        dlg.setObjectName("AddObservatoryDialog")
        layout = QFormLayout(dlg)

        name_edit = QLineEdit(dlg)
        name_edit.setPlaceholderText("Name")
        lookup_btn = QPushButton("Lookup", dlg)
        lookup_btn.setToolTip("Find coordinates from place name")
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
        lookup_info = QLabel("", dlg)
        lookup_info.setWordWrap(True)
        lookup_info.setStyleSheet("color: #6e8cab;")

        layout.addRow("Name:", name_row)
        layout.addRow("Lat:", lat_edit)
        layout.addRow("Lon:", lon_edit)
        layout.addRow("Elev:", elev_edit)
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
        layout.addWidget(buttons)
        dlg.setMinimumWidth(max(380, dlg.sizeHint().width()))

        while dlg.exec() == QDialog.Accepted:
            name = name_edit.text().strip()
            if not name:
                QMessageBox.warning(self, "Invalid Observatory", "Name cannot be empty.")
                continue
            if name in self.observatories:
                QMessageBox.warning(self, "Invalid Observatory", f"Observatory '{name}' already exists.")
                continue
            try:
                site = Site(
                    name=name,
                    latitude=float(lat_edit.text()),
                    longitude=float(lon_edit.text()),
                    elevation=float(elev_edit.text()),
                )
            except Exception:
                QMessageBox.warning(self, "Invalid Coordinates", "Please enter valid latitude, longitude and elevation.")
                continue

            self.observatories[name] = site
            self._save_custom_observatories()
            self._refresh_observatory_combo(selected_name=name)
            return
    def __init__(self):
        super().__init__()
        self.setObjectName(self.__class__.__name__)
        fallback_family = QApplication.font().family() or "Arial"
        QFont.insertSubstitution("Sans Serif", fallback_family)
        QFont.insertSubstitution("SF Pro Text", fallback_family)
        QFont.insertSubstitution("SF Pro Display", fallback_family)
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
        self.settings = QSettings("YourCompany", "AstroPlanner")
        self._migrate_table_settings_schema()
        self._dark_enabled = self.settings.value("general/darkMode", False, type=bool)
        self._theme_name = normalize_theme_key(self.settings.value("general/uiTheme", DEFAULT_UI_THEME, type=str))
        self._ui_font_size = max(9, min(16, self.settings.value("general/uiFontSize", 11, type=int)))

        # ── Polar-path visibility flags (persisted) ───────────────────────────
        self.show_sun_path  = self.settings.value("general/showSunPath",  True, type=bool)
        self.show_moon_path = self.settings.value("general/showMoonPath", True, type=bool)
        self.show_obj_path  = self.settings.value("general/showObjPath",  True, type=bool)
        
        # state holders
        self.targets: List[Target] = []
        self.worker: Optional[AstronomyWorker] = None  # keep reference!
        self.target_metrics: dict[str, TargetNightMetrics] = {}
        self.target_windows: dict[str, tuple[datetime, datetime]] = {}
        self._simbad_meta_cache: dict[str, tuple[Optional[float], str]] = {}
        self._simbad_compact_cache: dict[str, dict[str, object]] = {}
        self._gaia_alerts_cache: dict[str, dict[str, str]] = {}
        self._gaia_alerts_cache_loaded_at = 0.0
        self._meta_worker: Optional[MetadataLookupWorker] = None
        self._meta_request_id = 0
        self._calc_started_at = 0.0
        self._last_calc_stats = CalcRunStats(duration_s=0.0, visible_targets=0, total_targets=0)
        self._clock_polar_tick = 0
        self.color_blind_mode = self.settings.value("general/colorBlindMode", False, type=bool)
        self._plot_airmass = self.settings.value("general/plotAirmass", False, type=bool)
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
        self.llm_config = LLMConfig(
            url=self.settings.value("llm/serverUrl", LLMConfig.DEFAULT_URL, type=str),
            model=self.settings.value("llm/model", LLMConfig.DEFAULT_MODEL, type=str),
            timeout_s=LLMConfig.DEFAULT_TIMEOUT_S,
        )
        self._llm_worker: Optional[LLMWorker] = None
        self._ai_messages: list[dict[str, str]] = []
        self._ai_stream_message_index: Optional[int] = None
        self._cutout_manager = QNetworkAccessManager(self)
        self._cutout_manager.finished.connect(self._on_cutout_reply)
        self._cutout_request_id = 0
        self._cutout_pending_key = ""
        self._cutout_pending_name = ""
        self._cutout_displayed_key = ""
        self._cutout_reply: Optional[QNetworkReply] = None
        self._cutout_cache: dict[str, QPixmap] = {}
        self._cutout_cache_order: list[str] = []
        self._finder_cache: dict[str, QPixmap] = {}
        self._finder_cache_order: list[str] = []
        self._bhtom_candidate_cache_key: Optional[tuple[str, str]] = None
        self._bhtom_candidate_cache: Optional[list[dict[str, object]]] = None
        self._bhtom_candidate_cache_loaded_at = 0.0
        self._finder_workers: list[FinderChartWorker] = []
        self._finder_worker: Optional[FinderChartWorker] = None
        self._finder_request_id = 0
        self._finder_pending_key = ""
        self._finder_retry_after: dict[str, float] = {}
        self._finder_timeout_timer = QTimer(self)
        self._finder_timeout_timer.setSingleShot(True)
        self._finder_timeout_timer.setInterval(FINDER_WORKER_TIMEOUT_MS)
        self._finder_timeout_timer.timeout.connect(self._on_finder_chart_timeout)
        self._cutout_resize_timer = QTimer(self)
        self._cutout_resize_timer.setSingleShot(True)
        self._cutout_resize_timer.setInterval(260)
        self._cutout_resize_timer.timeout.connect(self._on_cutout_resize_timeout)
        self._cutout_last_resize_signature: Optional[tuple] = None

        # UI ------------------------------------------------
        self.table_model = TargetTableModel(self.targets, site=None)
        self.table_view = TargetTableView()
        self.table_view.setSelectionBehavior(QTableView.SelectRows)
        self.table_view.deleteRequested.connect(self._delete_selected_targets)
        # Use custom delegate for Name column to preserve its background on selection
        self.table_view.setItemDelegateForColumn(TargetTableModel.COL_NAME, NoSelectBackgroundDelegate(self.table_view))
        # Make columns only as wide as their contents
        header = self.table_view.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        self.table_view.verticalHeader().setVisible(False)
        self.table_view.setShowGrid(False)
        self.table_view.setModel(self.table_model)
        self.table_model.layoutChanged.connect(self._apply_table_row_visibility)
        self.table_model.modelReset.connect(self._apply_table_row_visibility)
        # NOTE: Reapply table settings and default sort are now handled explicitly after layoutChanged.emit()
        # # Apply saved settings now that table_view exists
        # self._load_settings()
        # Enable click‐to‐sort
        self.table_view.setSortingEnabled(True)
        self.table_view.horizontalHeader().setSectionsClickable(True)
        # Do not override user default sort here; let _load_settings() apply the saved sort
        self.table_view.setDragDropMode(QTableView.InternalMove)
        self.table_view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table_view.customContextMenuRequested.connect(self._open_table_context_menu)
        self.table_view.doubleClicked.connect(self._on_table_double_click)

        # Polar plot for alt-az projection
        self.polar_canvas = FigureCanvas(Figure(figsize=(4, 4)))
        self.polar_ax = self.polar_canvas.figure.add_subplot(projection='polar')
        self.polar_ax.set_theta_zero_location('N')
        self.polar_ax.set_theta_direction(-1)
        # Plot placeholders: targets, selected target, sun, moon
        self.polar_scatter = self.polar_ax.scatter([], [], c='blue', marker='x', s=20, label='Targets', alpha=0.5, picker=True)
        self.selected_scatter = self.polar_ax.scatter([], [], c='red', marker='x', s=40, alpha=1, label='Selected')
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
        # Connect pick event for polar scatter
        self.polar_canvas.mpl_connect('pick_event', self._on_polar_pick)
        self.polar_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.polar_canvas.setMinimumWidth(250)
        self.polar_canvas.setMinimumHeight(220)
        self.polar_canvas.setMaximumWidth(340)
        # Keep enough edge room so N/S labels are not clipped.
        self.polar_canvas.figure.subplots_adjust(left=0.06, right=0.94, bottom=0.12, top=0.90)

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
            "OCM": Site(name="OCM", latitude=-24.59, longitude=-70.19, elevation=2800),
            "Białków": Site(name="Białków", latitude=51.474248, longitude=16.657821, elevation=128),
            "Roque de los Muchachos": Site(name="Roque de los Muchachos", latitude=28.4522, longitude=-17.5330, elevation=2426),
        }
        loaded_observatories = self._load_custom_observatories()
        self.observatories = dict(self._builtin_observatories)
        self.observatories.update(loaded_observatories)
        # Keep repo config in sync with current full observatory list.
        self._save_custom_observatories(self.observatories)
        self.obs_combo = QComboBox()
        self.obs_combo.addItems(self.observatories.keys())
        init_site = self.settings.value("general/defaultSite", "OCM", type=str)
        if init_site not in self.observatories:
            init_site = "OCM" if "OCM" in self.observatories else (next(iter(self.observatories.keys()), ""))
        self.obs_combo.blockSignals(True)
        self.obs_combo.setCurrentText(init_site)
        self.obs_combo.blockSignals(False)
        self.obs_combo.currentTextChanged.connect(self._on_obs_change)
        # Ensure site fields & clock worker start immediately for final startup site.
        if init_site:
            self._on_obs_change(init_site)
        self.obs_combo.setMinimumContentsLength(26)
        self.obs_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.obs_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._update_obs_combo_widths()
        self.add_obs_btn = QToolButton()
        self.add_obs_btn.setText("+")
        self.add_obs_btn.setToolTip("Add custom observatory")
        self.add_obs_btn.setFixedSize(24, 24)
        self.add_obs_btn.clicked.connect(self._open_add_observatory_dialog)

        # Debounced connections for date and limit spin
        self.date_edit.dateChanged.connect(lambda _: self._replot_timer.start())
        self.limit_spin.valueChanged.connect(self._update_limit)
        self.sun_alt_limit_spin.valueChanged.connect(self._on_sun_alt_limit_changed)
        # Re-plot when latitude, longitude, or elevation is changed
        self.lat_edit.editingFinished.connect(self._on_site_inputs_changed)
        self.lon_edit.editingFinished.connect(self._on_site_inputs_changed)
        self.elev_edit.editingFinished.connect(self._on_site_inputs_changed)

        self.plot_canvas = FigureCanvas(Figure(figsize=(6, 4), tight_layout=True))
        self.plot_canvas.setContentsMargins(0, 0, 0, 0)
        self.ax_alt = self.plot_canvas.figure.subplots()
        self.ax_alt.margins(x=0, y=0)

        # Matplotlib toolbar
        self.plot_toolbar = NavigationToolbar(self.plot_canvas, self)
        # Minimize toolbar padding and height
        self.plot_toolbar.setIconSize(QSize(16, 16))
        self.plot_toolbar.layout().setContentsMargins(0, 0, 0, 0)
        self.plot_toolbar.layout().setSpacing(0)
        self.plot_mode_widget = QWidget(self.plot_toolbar)
        self.plot_mode_widget.setObjectName("PlotModeWidget")
        plot_mode_layout = QHBoxLayout(self.plot_mode_widget)
        plot_mode_layout.setContentsMargins(4, 0, 4, 0)
        plot_mode_layout.setSpacing(6)
        self.plot_mode_alt_label = QLabel("Altitude", self.plot_mode_widget)
        self.plot_mode_airmass_label = QLabel("Airmass", self.plot_mode_widget)
        self.airmass_toggle_btn = QSlider(Qt.Orientation.Horizontal, self.plot_mode_widget)
        self.airmass_toggle_btn.setObjectName("PlotModeSwitch")
        self.airmass_toggle_btn.setToolTip("Switch the main plot Y-axis between altitude and airmass")
        self.airmass_toggle_btn.setRange(0, 1)
        self.airmass_toggle_btn.setSingleStep(1)
        self.airmass_toggle_btn.setPageStep(1)
        self.airmass_toggle_btn.setFixedWidth(34)
        self.airmass_toggle_btn.setFixedHeight(18)
        self.airmass_toggle_btn.setValue(1 if self._plot_airmass else 0)
        self.airmass_toggle_btn.valueChanged.connect(self._on_plot_mode_switch_changed)
        self.airmass_toggle_btn.setStyleSheet(
            """
            QSlider::groove:horizontal {
                height: 18px;
                border-radius: 9px;
                background: rgba(120, 120, 120, 0.30);
                border: 1px solid rgba(120, 120, 120, 0.55);
            }
            QSlider::sub-page:horizontal {
                background: rgba(67, 145, 214, 0.48);
                border-radius: 9px;
            }
            QSlider::add-page:horizontal {
                background: transparent;
                border-radius: 9px;
            }
            QSlider::handle:horizontal {
                width: 14px;
                margin: 1px;
                border-radius: 7px;
                background: rgb(244, 247, 250);
                border: 1px solid rgba(30, 41, 59, 0.22);
            }
            """
        )
        plot_mode_layout.addWidget(self.plot_mode_alt_label)
        plot_mode_layout.addWidget(self.airmass_toggle_btn)
        plot_mode_layout.addWidget(self.plot_mode_airmass_label)
        self._refresh_plot_mode_switch()
        self.plot_toolbar.addSeparator()
        self.plot_toolbar.addWidget(self.plot_mode_widget)
        self.plot_toolbar.setMaximumHeight(self.plot_toolbar.sizeHint().height())

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
        self.coord_error_label.setStyleSheet("color: #cc2f2f;")
        self.coord_error_label.setWordWrap(True)

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

        add_btn = QPushButton("Add Target…")
        add_btn.setMinimumHeight(28)
        add_btn.setIcon(self.style().standardIcon(QStyle.SP_FileDialogNewFolder))
        add_btn.clicked.connect(self._add_target_dialog)

        toggle_obs_btn = QPushButton("Toggle Observed")
        toggle_obs_btn.setMinimumHeight(28)
        toggle_obs_btn.clicked.connect(self._toggle_observed_selected)
        self.edit_object_btn = QPushButton("Edit Object…")
        self.edit_object_btn.setMinimumHeight(28)
        self.edit_object_btn.setIcon(self.style().standardIcon(QStyle.SP_FileDialogDetailedView))
        self.edit_object_btn.clicked.connect(self._edit_object_for_selected)
        self.edit_object_btn.setEnabled(False)

        load_plan_btn = QPushButton("Load Plan…")
        load_plan_btn.setMinimumHeight(28)
        load_plan_btn.setIcon(self.style().standardIcon(QStyle.SP_DialogOpenButton))
        load_plan_btn.clicked.connect(self._load_plan)
        save_btn = QPushButton("Save Plan…")
        save_btn.setMinimumHeight(28)
        save_btn.setIcon(self.style().standardIcon(QStyle.SP_DialogSaveButton))
        save_btn.clicked.connect(self._export_plan)
        settings_btn = QPushButton("Settings…")
        settings_btn.setMinimumHeight(28)
        settings_btn.setIcon(self.style().standardIcon(QStyle.SP_FileDialogDetailedView))
        settings_btn.clicked.connect(self.open_general_settings)
        table_settings_btn = QPushButton("Table Settings…")
        table_settings_btn.setMinimumHeight(28)
        table_settings_btn.setIcon(self.style().standardIcon(QStyle.SP_FileDialogListView))
        table_settings_btn.clicked.connect(self.open_table_settings)
        suggest_targets_btn = QPushButton("Suggest Targets")
        suggest_targets_btn.setMinimumHeight(28)
        suggest_targets_btn.clicked.connect(self._ai_suggest_targets)
        self.ai_toggle_btn = QPushButton("AI Assistant")
        self.ai_toggle_btn.setMinimumHeight(28)
        self.ai_toggle_btn.setCheckable(True)
        self.ai_toggle_btn.setChecked(False)
        self.ai_toggle_btn.setToolTip("Toggle the local AI assistant panel")
        self.ai_toggle_btn.toggled.connect(self._toggle_ai_panel)

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

        actions_bar = QFrame()
        actions_bar.setObjectName("ActionsBar")
        actions_l = QHBoxLayout()
        actions_l.setContentsMargins(10, 6, 10, 6)
        actions_l.setSpacing(8)
        actions_l.addWidget(add_btn)
        actions_l.addWidget(suggest_targets_btn)
        actions_l.addWidget(toggle_obs_btn)
        actions_l.addWidget(self.edit_object_btn)
        actions_l.addWidget(load_plan_btn)
        actions_l.addWidget(save_btn)
        actions_l.addWidget(settings_btn)
        actions_l.addWidget(table_settings_btn)
        actions_l.addWidget(self.ai_toggle_btn)
        actions_l.addStretch(1)
        actions_bar.setLayout(actions_l)
        actions_bar.setMinimumHeight(46)
        actions_bar.setMaximumHeight(58)

        plot_card = QFrame()
        plot_card.setObjectName("PlotCard")
        plot_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        plot_l = QVBoxLayout()
        plot_l.setContentsMargins(2, 2, 2, 2)
        plot_l.setSpacing(2)
        plot_l.addWidget(self.plot_toolbar)
        plot_l.addWidget(self.plot_canvas)
        plot_card.setLayout(plot_l)
        self.plot_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

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
        self.sun_alt_label = QLabel("-")
        self.moon_alt_label = QLabel("-")
        self.sidereal_label = QLabel("-")

        # Selected target details
        self.sel_name_label = QLabel("-")
        self.sel_type_label = QLabel("-")
        self.sel_score_label = QLabel("-")
        self.sel_window_label = QLabel("-")
        self.sel_notes_label = QLabel("-")
        self.sel_notes_label.setWordWrap(True)

        details_row = QWidget()
        details_layout = QVBoxLayout()
        details_layout.setContentsMargins(0, 0, 0, 0)
        details_layout.setSpacing(2)
        details_layout.addWidget(self.sel_notes_label)
        details_row.setLayout(details_layout)

        def _add_info_row(form: QFormLayout, row_idx: int, title: str, value_widget: QWidget):
            title_label = QLabel(title)
            title_label.setFont(self.info_label_font)
            title_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            if isinstance(value_widget, QLabel):
                value_widget.setFont(self.info_value_font)
                value_widget.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            form.insertRow(row_idx, title_label, value_widget)
            return title_label

        _add_info_row(left_info_form, 0, "🕑 Local time:", self.localtime_label)
        _add_info_row(left_info_form, 1, "🌐 UTC time:", self.utctime_label)
        _add_info_row(left_info_form, 2, "⭐ Sidereal time:", self.sidereal_label)
        _add_info_row(left_info_form, 3, "🌇 Sunset:", self.sunset_label)
        _add_info_row(left_info_form, 4, "🌅 Sunrise:", self.sunrise_label)
        _add_info_row(left_info_form, 5, "🌕 Moon phase:", self.moonphase_bar)
        _add_info_row(left_info_form, 6, "🌙 Moonrise:", self.moonrise_label)
        _add_info_row(left_info_form, 7, "🌙 Moonset:", self.moonset_label)

        _add_info_row(right_info_form, 0, "☀️ Sun altitude:", self.sun_alt_label)
        _add_info_row(right_info_form, 1, "🌙 Moon altitude:", self.moon_alt_label)
        _add_info_row(right_info_form, 2, "🎯 Selected target:", self.sel_name_label)
        self.sel_type_title_label = _add_info_row(right_info_form, 3, "Type / Mag:", self.sel_type_label)
        _add_info_row(right_info_form, 4, "Score:", self.sel_score_label)
        _add_info_row(right_info_form, 5, "Best window:", self.sel_window_label)
        _add_info_row(right_info_form, 6, "Notes:", details_row)

        self.info_widget.setContentsMargins(0, 0, 0, 0)

        table_card = QFrame()
        table_card.setObjectName("TableCard")
        table_card.setMinimumHeight(180)
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
        polar_l = QVBoxLayout()
        polar_l.setContentsMargins(2, 2, 2, 2)
        polar_l.setSpacing(1)
        polar_title = QLabel("Sky View")
        polar_title.setObjectName("SectionTitle")
        polar_l.addWidget(polar_title)

        cutout_frame = QFrame()
        cutout_frame.setObjectName("CutoutFrame")
        cutout_frame.setMinimumWidth(220)
        cutout_frame.setMaximumWidth(560)
        cutout_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        cutout_l = QVBoxLayout()
        cutout_l.setContentsMargins(0, 0, 0, 0)
        cutout_l.setSpacing(0)
        self.cutout_tabs = QTabWidget(self)
        self.cutout_tabs.setDocumentMode(True)
        self.cutout_tabs.setTabPosition(QTabWidget.North)
        self.cutout_tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.aladin_image_label = CoverImageLabel("Select a target")
        self.aladin_image_label.setObjectName("CutoutImage")
        self.aladin_image_label.setAlignment(Qt.AlignCenter)
        self.aladin_image_label.setWordWrap(True)
        self.aladin_image_label.setScaledContents(False)
        self.aladin_image_label.setMinimumSize(1, 1)
        self.aladin_image_label.setMaximumSize(16777215, 16777215)
        self.aladin_image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.aladin_image_label.resized.connect(self._schedule_cutout_resize_refresh)
        # Backward-compatible alias used by existing cutout helpers.
        self.cutout_image_label = self.aladin_image_label

        self.finder_image_label = CoverImageLabel("Select a target")
        self.finder_image_label.setObjectName("CutoutImage")
        self.finder_image_label.setAlignment(Qt.AlignCenter)
        self.finder_image_label.setWordWrap(True)
        self.finder_image_label.setScaledContents(False)
        self.finder_image_label.setMinimumSize(1, 1)
        self.finder_image_label.setMaximumSize(16777215, 16777215)
        self.finder_image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.finder_image_label.resized.connect(self._schedule_cutout_resize_refresh)

        aladin_tab = QWidget(self.cutout_tabs)
        aladin_tab_l = QVBoxLayout(aladin_tab)
        aladin_tab_l.setContentsMargins(0, 0, 0, 0)
        aladin_tab_l.addWidget(self.aladin_image_label, 1)
        finder_tab = QWidget(self.cutout_tabs)
        finder_tab_l = QVBoxLayout(finder_tab)
        finder_tab_l.setContentsMargins(0, 0, 0, 0)
        finder_tab_l.addWidget(self.finder_image_label, 1)
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

        right_dashboard = QSplitter(Qt.Vertical)
        right_dashboard.addWidget(polar_card)
        right_dashboard.addWidget(info_card)
        right_dashboard.setHandleWidth(1)
        right_dashboard.setStretchFactor(0, 3)
        right_dashboard.setStretchFactor(1, 2)
        right_dashboard.setSizes([390, 300])
        right_dashboard.setMinimumWidth(460)
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
        self.ai_panel = self._build_ai_panel()
        self.ai_panel.setVisible(False)

        container = QWidget()
        container.setObjectName("RootContainer")
        container_l = QVBoxLayout()
        container_l.setContentsMargins(0, 0, 0, 0)
        container_l.setSpacing(2)
        container_l.addWidget(top_controls)
        container_l.addWidget(main_area, 1)
        container_l.addWidget(self.ai_panel)
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
        self.status_finder_label = QLabel("Finder: idle")
        self.status_finder_progress = QProgressBar(self)
        self.status_finder_progress.setMinimumWidth(96)
        self.status_finder_progress.setMaximumWidth(140)
        self.status_finder_progress.setTextVisible(False)
        self.status_finder_progress.setRange(0, 0)
        self.status_finder_progress.hide()
        self.status_calc_label = QLabel("Last calc: -")
        self.statusBar().addPermanentWidget(self.status_filters_label)
        self.statusBar().addPermanentWidget(self.status_finder_label)
        self.statusBar().addPermanentWidget(self.status_finder_progress)
        self.statusBar().addPermanentWidget(self.status_calc_label)
        # Ensure threads are cleaned up on application exit
        app = QApplication.instance()
        if app:
            app.aboutToQuit.connect(self._cleanup_threads)

        # Progress indicator for calculations
        self.progress = QProgressDialog("Calculating visibility...", "", 0, 0, self)
        self.progress.setWindowTitle("Please wait")
        self.progress.setWindowModality(Qt.WindowModal)
        self.progress.setCancelButton(None)
        self.progress.setAutoClose(False)
        self.progress.setAutoReset(False)
        self.progress.hide()

        # Start real‑time clock updates for time labels (but avoid double‑launch)
        self.clock_worker = getattr(self, "clock_worker", None)
        if self.table_model.site and self.clock_worker is None:
            self._start_clock_worker()
        self._clock_timer = QTimer(self)
        self._clock_timer.timeout.connect(self._update_clock)
        self._clock_timer.start(1000)
        self._update_status_bar()
        self._update_selected_details()
        self._validate_site_inputs()
        self._replot_timer.start()
    def _start_clock_worker(self):
        # Don’t create new workers while exiting
        if getattr(self, "_shutting_down", False):
            return
        # Stop any existing clock worker
        if self.clock_worker:
            self.clock_worker.stop()
        # Determine site, fallback to default if none
        site = self.table_model.site or Site(name="Default", latitude=0.0, longitude=0.0, elevation=0.0)
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
        worker.finished.connect(worker.deleteLater)

    # --------------------------------------------------
    # Qt close handler – ensure threads are stopped
    # --------------------------------------------------
    def closeEvent(self, event):
        """Stop background threads before the window closes."""
        try:
            self._cleanup_threads()
        except Exception as exc:
            logger.exception("Error during thread cleanup: %s", exc)
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

        self.load_plan_act = QAction("Load plan…", self, shortcut=QKeySequence("Ctrl+Shift+L"))
        self.load_plan_act.triggered.connect(self._load_plan)

        self.add_act = QAction("Add target…", self, shortcut=QKeySequence("Ctrl+N"))
        self.add_act.triggered.connect(self._add_target_dialog)

        self.toggle_obs_act = QAction("Toggle observed", self, shortcut=QKeySequence("Ctrl+Shift+O"))
        self.toggle_obs_act.triggered.connect(self._toggle_observed_selected)

        self.exp_act = QAction("Export plan…", self, shortcut=QKeySequence("Ctrl+E"))
        self.exp_act.triggered.connect(self._export_plan)

        self.dark_act = QAction("Toggle dark mode", self, shortcut=QKeySequence("Ctrl+D"))
        self.dark_act.setShortcutContext(Qt.ApplicationShortcut)
        self.dark_act.triggered.connect(self._toggle_dark)
        self.ai_describe_act = QAction("Describe selected object", self, shortcut=QKeySequence("Ctrl+I"))
        self.ai_describe_act.triggered.connect(self._ai_describe_target)
        self.ai_suggest_act = QAction("Suggest targets for tonight", self, shortcut=QKeySequence("Ctrl+Shift+I"))
        self.ai_suggest_act.triggered.connect(self._ai_suggest_targets)
        self.ai_toggle_panel_act = QAction("Toggle AI panel", self)
        self.ai_toggle_panel_act.triggered.connect(
            lambda: self.ai_toggle_btn.setChecked(not self.ai_toggle_btn.isChecked())
        )

        self.view_obs_preset_act = QAction("Observation Columns", self, checkable=True)
        self.view_obs_preset_act.triggered.connect(lambda: self._apply_column_preset("observation"))
        self.view_full_preset_act = QAction("Full Columns", self, checkable=True)
        self.view_full_preset_act.triggered.connect(lambda: self._apply_column_preset("full"))
        preset_group = QActionGroup(self)
        preset_group.setExclusive(True)
        preset_group.addAction(self.view_obs_preset_act)
        preset_group.addAction(self.view_full_preset_act)
        current_preset = self.settings.value("table/viewPreset", "observation", type=str)
        if current_preset == "full":
            self.view_full_preset_act.setChecked(True)
        else:
            self.view_obs_preset_act.setChecked(True)

        # Make shortcuts work even if the action isn't in a visible menu
        for act in (
            self.load_act,
            self.load_plan_act,
            self.add_act,
            self.toggle_obs_act,
            self.exp_act,
            self.dark_act,
            self.ai_describe_act,
            self.ai_suggest_act,
        ):
            self.addAction(act)

        # ----- Menu bar -----
        menubar = self.menuBar()
        if sys.platform == "darwin":
            menubar.setNativeMenuBar(True)
        menubar.setVisible(True)
        file_menu = menubar.addMenu("&File")
        file_menu.addAction(self.load_act)
        file_menu.addAction(self.load_plan_act)
        file_menu.addAction(self.exp_act)
        file_menu.addSeparator()
        file_menu.addAction(self.dark_act)
        file_menu.addSeparator()
        file_menu.addAction("E&xit", self.close)

        target_menu = menubar.addMenu("&Targets")
        target_menu.addAction(self.add_act)
        target_menu.addAction(self.ai_suggest_act)
        target_menu.addAction(self.toggle_obs_act)

        view_menu = menubar.addMenu("&View")
        view_menu.addAction(self.view_obs_preset_act)
        view_menu.addAction(self.view_full_preset_act)

        ai_menu = menubar.addMenu("&AI")
        ai_menu.addAction(self.ai_describe_act)
        ai_menu.addSeparator()
        ai_menu.addAction(self.ai_toggle_panel_act)

        # Settings menu
        settings_menu = menubar.addMenu("&Settings")
        self.gen_settings_act = QAction("General Settings…", self)
        self.gen_settings_act.setShortcuts([QKeySequence("Ctrl+,"), QKeySequence("Ctrl+;")])
        self.gen_settings_act.setShortcutContext(Qt.ApplicationShortcut)
        self.gen_settings_act.triggered.connect(self.open_general_settings)
        settings_menu.addAction(self.gen_settings_act)
        self.tbl_settings_act = QAction("Table Settings…", self)
        self.tbl_settings_act.triggered.connect(self.open_table_settings)
        settings_menu.addAction(self.tbl_settings_act)

        self.addAction(self.gen_settings_act)
        self.addAction(self.tbl_settings_act)

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
        if version >= 2:
            return

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

        self.settings.setValue("table/columnSchemaVersion", 2)

    def _recompute_recommended_order_cache(self) -> None:
        order_values = [0] * len(self.targets)
        ordered, _ = self._build_deterministic_observation_order()
        for rank, item in enumerate(ordered, start=1):
            row_index = int(item.get("row_index", -1))
            if 0 <= row_index < len(order_values):
                order_values[row_index] = rank
        self.table_model.order_values = order_values

    def _apply_table_settings(self):
        """Apply table row height and table column widths."""
        row_h = self.settings.value("table/rowHeight", 24, type=int)
        self.table_view.verticalHeader().setDefaultSectionSize(row_h)
        # Lock row height and apply to all existing rows
        for r in range(self.table_model.rowCount()):
            self.table_view.setRowHeight(r, row_h)
        name_col_w = self.settings.value("table/firstColumnWidth", 100, type=int)
        self.table_view.setColumnWidth(TargetTableModel.COL_ORDER, 56)
        self.table_view.setColumnWidth(TargetTableModel.COL_NAME, name_col_w)
        # Lock Order and Name columns so Stretch mode doesn’t override them
        header = self.table_view.horizontalHeader()
        header.setSectionResizeMode(TargetTableModel.COL_ORDER, QHeaderView.Fixed)
        header.setSectionResizeMode(TargetTableModel.COL_NAME, QHeaderView.Fixed)
        # Font size
        fs = self.settings.value("table/fontSize", 11, type=int)
        fnt = self.table_view.font()
        fnt.setPointSize(fs)
        self.table_view.setFont(fnt)
        # Column visibility
        for col in range(self.table_model.columnCount()):
            show = self.settings.value(f"table/col{col}", True, type=bool)
            self.table_view.setColumnHidden(col, not show)
        # Highlight colors in model
        palette = COLORBLIND_HIGHLIGHT if self.color_blind_mode else DEFAULT_HIGHLIGHT
        default_colors = {"below": palette.below, "limit": palette.limit, "above": palette.above}
        self.table_model.highlight_colors = {
            k: QColor(self.settings.value(f"table/color/{k}", default_colors[k]))
            for k in default_colors
        }
        self._apply_column_preset(self.settings.value("table/viewPreset", "observation", type=str), save=False)

    def _apply_column_preset(self, preset: str, save: bool = True):
        if preset not in {"observation", "full"}:
            preset = "observation"
        obs_visible = {
            TargetTableModel.COL_ORDER,
            TargetTableModel.COL_NAME,
            TargetTableModel.COL_ALT,
            TargetTableModel.COL_AZ,
            TargetTableModel.COL_MOON_SEP,
            TargetTableModel.COL_SCORE,
            TargetTableModel.COL_HOURS,
            TargetTableModel.COL_PRIORITY,
            TargetTableModel.COL_OBSERVED,
        }
        for col in range(self.table_model.columnCount()):
            if col == TargetTableModel.COL_ACTIONS:
                self.table_view.setColumnHidden(col, True)
                continue
            if preset == "observation":
                self.table_view.setColumnHidden(col, col not in obs_visible)
            else:
                show = self.settings.value(f"table/col{col}", True, type=bool)
                self.table_view.setColumnHidden(col, not show)
        if save:
            self.settings.setValue("table/viewPreset", preset)
        if hasattr(self, "view_obs_preset_act"):
            self.view_obs_preset_act.setChecked(preset == "observation")
        if hasattr(self, "view_full_preset_act"):
            self.view_full_preset_act.setChecked(preset == "full")

    def open_table_settings(self):
        """Open the Table Settings dialog."""
        dlg = TableSettingsDialog(self)
        dlg.exec()

    def _apply_general_settings(self):
        """Apply default site."""
        s = self.settings
        self.show_sun_path  = self.settings.value("general/showSunPath",  True, type=bool)
        self.show_moon_path = self.settings.value("general/showMoonPath", True, type=bool)
        self.show_obj_path  = self.settings.value("general/showObjPath",  True, type=bool)
        self._dark_enabled = self.settings.value("general/darkMode", False, type=bool)
        self._theme_name = normalize_theme_key(self.settings.value("general/uiTheme", DEFAULT_UI_THEME, type=str))
        self._ui_font_size = max(9, min(16, self.settings.value("general/uiFontSize", 11, type=int)))
        self.color_blind_mode = self.settings.value("general/colorBlindMode", False, type=bool)
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

        # If a payload is already cached, refresh the polar plot right away
        if getattr(self, "last_payload", None):
            self._update_polar_positions(self.last_payload)
        ds = s.value("general/defaultSite", type=str)
        if ds in self.observatories and self.obs_combo.currentText() != ds:
            self.obs_combo.setCurrentText(ds)
        self._apply_table_settings()
        self._apply_styles()
        self._update_status_bar()

    def open_general_settings(self):
        dlg = GeneralSettingsDialog(self)
        dlg.exec()

    def _sun_alt_limit(self) -> float:
        if not hasattr(self, "sun_alt_limit_spin"):
            return -10.0
        return float(self.sun_alt_limit_spin.value())

    @Slot(int)
    def _on_sun_alt_limit_changed(self, value: int):
        self.settings.setValue("general/sunAltLimit", int(value))
        self._update_status_bar()
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

    def _update_status_bar(self):
        if not hasattr(self, "status_filters_label") or not hasattr(self, "status_calc_label"):
            return
        parts: list[str] = []
        if hasattr(self, "min_score_spin") and self.min_score_spin.value() > 0:
            parts.append(f"score≥{self.min_score_spin.value()}")
        if hasattr(self, "min_moon_sep_spin") and self.min_moon_sep_spin.value() > 0:
            parts.append(f"moon≥{self.min_moon_sep_spin.value()}°")
        if hasattr(self, "hide_observed_chk") and self.hide_observed_chk.isChecked():
            parts.append("hide observed")
        if hasattr(self, "sun_alt_limit_spin"):
            parts.append(f"sun≤{self._sun_alt_limit():.0f}°")

        total = len(self.targets)
        visible = sum(1 for flag in self.table_model.row_enabled if flag) if self.table_model.row_enabled else total
        active_filters = ", ".join(parts) if parts else "none"
        self.status_filters_label.setText(f"Filters: {active_filters} | Visible: {visible}/{total}")

        if self.worker is not None and self.worker.isRunning():
            self.status_calc_label.setText("Last calc: running...")
            return
        stats = self._last_calc_stats
        if stats.total_targets > 0:
            self.status_calc_label.setText(
                f"Last calc: {stats.duration_s:.2f}s | {stats.visible_targets}/{stats.total_targets}"
            )
        else:
            self.status_calc_label.setText("Last calc: -")

    @Slot(object, object)
    def _update_selected_details(self, *_):
        rows = self._selected_rows()
        if not rows:
            self.sel_name_label.setText("-")
            self.sel_type_title_label.setText("Type / Mag:")
            self.sel_type_label.setText("-")
            self.sel_score_label.setText("-")
            self.sel_window_label.setText("-")
            self.sel_notes_label.setText("-")
            self.edit_object_btn.setEnabled(False)
            if hasattr(self, "ai_describe_act"):
                self.ai_describe_act.setEnabled(False)
            self._update_cutout_preview_for_target(None)
            self._update_night_details_constraints()
            self._update_status_bar()
            return

        row = rows[0]
        if not (0 <= row < len(self.targets)):
            self._update_cutout_preview_for_target(None)
            return
        tgt = self.targets[row]
        self._ensure_known_target_type(tgt)
        self.sel_name_label.setText(tgt.name)
        self.sel_type_title_label.setText(
            "Type / Last Mag:" if _target_magnitude_label(tgt) == "Last Mag" else "Type / Mag:"
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
        self._update_cutout_preview_for_target(tgt)
        self._update_night_details_constraints()
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
        layout.addWidget(buttons)
        dlg.setMinimumWidth(max(520, dlg.sizeHint().width()))
        if dlg.exec() != QDialog.Accepted:
            return
        tgt.notes = notes_edit.toPlainText().strip()
        idx = self.table_model.index(row, TargetTableModel.COL_NAME)
        self.table_model.dataChanged.emit(idx, idx, [Qt.ToolTipRole, Qt.DisplayRole])
        self._update_selected_details()

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
            return self._target_plot_color_css(tgt, row, line_palette)

        def _refresh_color_preview():
            color_css = _normalized_css_color(color_edit.text())
            if color_css:
                sample = QColor(color_css)
                fg = "#111111" if sample.lightness() > 145 else "#f5f7fb"
                color_preview.setText(color_css.upper())
                color_preview.setStyleSheet(
                    f"background:{color_css}; color:{fg}; border:1px solid #50627a; border-radius:4px; padding:2px 6px;"
                )
                return
            fallback = _default_color_for_row()
            sample = QColor(fallback)
            fg = "#111111" if sample.lightness() > 145 else "#f5f7fb"
            color_preview.setText("Auto")
            color_preview.setStyleSheet(
                f"background:{fallback}; color:{fg}; border:1px dashed #50627a; border-radius:4px; padding:2px 6px;"
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
        form.addRow(buttons)
        dlg.setMinimumWidth(max(540, dlg.sizeHint().width()))

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
            if edit in invalid:
                edit.setStyleSheet("border: 1px solid #cc2f2f;")
            else:
                edit.setStyleSheet("")
        return not invalid

    @Slot()
    def _on_site_inputs_changed(self):
        if not self._validate_site_inputs():
            return
        try:
            site = Site(
                name=self.obs_combo.currentText(),
                latitude=self._read_site_float(self.lat_edit),
                longitude=self._read_site_float(self.lon_edit),
                elevation=self._read_site_float(self.elev_edit),
            )
        except (ValidationError, ValueError):
            return
        self.table_model.site = site
        self._start_clock_worker()
        self._replot_timer.start()

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
        self._stop_finder_workers(aggressive=True)
        self._finder_worker = None
        self._finder_pending_key = ""
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

    @Slot()
    def _load_plan(self):
        """Load a saved plan (JSON targets)."""
        fn, _ = QFileDialog.getOpenFileName(
            self, "Load plan JSON", str(Path.cwd()), "JSON files (*.json)"
        )
        if not fn:
            return
        try:
            with open(fn, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
            loaded_targets: list[Target] = []
            self.target_metrics.clear()
            self.target_windows.clear()
            for entry in data:
                loaded_targets.append(Target(**entry))
            self.table_model.reset_targets(loaded_targets)
            # Reapply settings & default sort after layout change
            self._apply_table_settings()
            self._apply_default_sort()
            self._fetch_missing_magnitudes_async()
            # Automatically redraw the visibility plot after loading a plan
            self._run_plan()
        except Exception as e:
            QMessageBox.critical(self, "Load plan error", str(e))

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

    def _append_target_to_plan(self, target: Target) -> bool:
        if self._plan_contains_target(target):
            QMessageBox.information(self, "Already in plan", f"{target.name} is already present in the current plan.")
            return False

        target_copy = Target(**target.model_dump())
        self._ensure_known_target_type(target_copy)
        self.table_model.append_target(target_copy)
        self._recompute_recommended_order_cache()
        self._apply_table_settings()
        self._apply_default_sort()
        self._fetch_missing_magnitudes_async()
        self._replot_timer.start()

        for row_idx, existing in enumerate(self.targets):
            if existing is target_copy or _targets_match(existing, target_copy):
                self.table_view.selectRow(row_idx)
                self.table_view.scrollTo(self.table_model.index(row_idx, TargetTableModel.COL_NAME))
                break
        self._update_selected_details()
        return True

    @Slot()
    def _run_plan(self):
        """Kick off the worker thread unless one is already running."""
        logger.info("Starting new visibility calculation …")
        if self.worker and self.worker.isRunning():
            # Stop existing calculation and start a new one
            self.worker.quit()
            self.worker.wait()
        if not self._validate_site_inputs():
            return
        try:
            site = Site(
                latitude=self._read_site_float(self.lat_edit),
                longitude=self._read_site_float(self.lon_edit),
                elevation=self._read_site_float(self.elev_edit),
            )
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
        # Update the observation limit for table coloring
        self.table_model.limit = settings.limit_altitude
        self._emit_table_data_changed()
        self._calc_started_at = perf_counter()
        self._update_status_bar()

        # Show busy indicator
        self.progress.show()

        self.worker = AstronomyWorker(self.targets, settings, parent=self)
        self.worker.finished.connect(self._update_plot)
        self.worker.finished.connect(lambda _: self.progress.hide())
        self.worker.finished.connect(lambda _: self.plot_canvas.setEnabled(True))
        # disable canvas during computation
        self.plot_canvas.setEnabled(False)
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
        self._replot_timer.start()

    def _toggle_observed_row(self, row: int):
        if 0 <= row < len(self.targets):
            self.targets[row].observed = not self.targets[row].observed
            self._emit_table_data_changed()
            self._update_selected_details()
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
                self.targets[row].observed = not self.targets[row].observed
        self._emit_table_data_changed()
        self._update_selected_details()
        if self.last_payload is not None:
            self._update_plot(self.last_payload)
        else:
            self._replot_timer.start()

    def _emit_table_data_changed(self):
        rows = self.table_model.rowCount()
        cols = self.table_model.columnCount()
        if rows <= 0 or cols <= 0:
            return
        top_left = self.table_model.index(0, 0)
        bottom_right = self.table_model.index(rows - 1, cols - 1)
        self.table_model.dataChanged.emit(
            top_left,
            bottom_right,
            [Qt.DisplayRole, Qt.BackgroundRole, Qt.ForegroundRole, Qt.ToolTipRole, Qt.CheckStateRole],
        )

    def _apply_table_row_visibility(self):
        if not hasattr(self, "table_view") or not hasattr(self, "table_model"):
            return
        rows = self.table_model.rowCount()
        for row in range(rows):
            # Keep all rows visible in the table; filtered/observed rows are
            # represented by disabled (greyed) styling via row_enabled.
            self.table_view.setRowHidden(row, False)

    def _clear_table_dynamic_cache(self):
        self.table_model.order_values = []
        self.table_model.current_alts = []
        self.table_model.current_azs = []
        self.table_model.current_seps = []
        self.table_model.scores = []
        self.table_model.hours_above_limit = []
        self.table_model.row_enabled = []
        self._apply_table_row_visibility()

    def _refresh_table_buttons(self):
        # Row widgets were replaced by a context menu to keep the table fast.
        return

    @Slot(dict)
    def _update_plot(self, data: dict):
        """Redraw the altitude plot with new data from the worker."""
        logger.info("Altitude plot refresh (%d targets)", len(self.targets))
        self.last_payload = data
        # Keep full visibility data around for polar path plotting
        self.full_payload = data
        # Reset stored visibility lines for this redraw
        self.vis_lines.clear()
        self.ax_alt.clear()

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

        current_alts = []
        current_azs = []
        current_seps = []
        for tgt in self.targets:
            # alt/az via astroplan
            fixed = FixedTarget(name=tgt.name, coord=tgt.skycoord)
            altaz_now = observer_now.altaz(Time(now_dt), fixed)
            current_alts.append(float(altaz_now.alt.deg))                                       # type: ignore[arg-type]
            current_azs.append(float(altaz_now.az.deg))                                         # type: ignore[arg-type]
            # sep via PyEphem
            current_seps.append(float(tgt.skycoord.separation(moon_coord).deg))                 # type: ignore[arg-type]

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
        civil_col = "#FFF2CC"
        naut_col = "#CCE5FF"
        astro_col = "#D9D9D9"
        # Build a dict of available event datetimes
        ev = {}
        for key in ("sunset", "dusk_civ", "dusk_naut", "dusk",
                    "dawn", "dawn_naut", "dawn_civ", "sunrise",
                    "moonrise", "moonset"):
            try:
                dt = mdates.num2date(data[key]).astimezone(tz)
                ev[key] = dt
            except Exception:
                continue

        # Always display a fixed local-time window: 12:00 -> 12:00 next day.
        obs_date = self.date_edit.date()
        start_noon_naive = datetime(obs_date.year(), obs_date.month(), obs_date.day(), 12, 0)
        next_date = obs_date.addDays(1)
        end_noon_naive = datetime(next_date.year(), next_date.month(), next_date.day(), 12, 0)
        try:
            start_dt = tz.localize(start_noon_naive)
            end_dt = tz.localize(end_noon_naive)
        except Exception:
            # Fallback to cached midnight window if timezone localization fails.
            center_dt = mdates.num2date(data["midnight"]).astimezone(tz)
            start_dt = center_dt - timedelta(hours=12)
            end_dt = center_dt + timedelta(hours=12)
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
                self.ax_alt.axvline(dt, color="#BBBBBB", linestyle="--", alpha=0.15)

        # ------------------------------------------------------------------
        # Red limiting‑altitude line
        # ------------------------------------------------------------------
        limit_line_value = self._plot_limit_value()
        limit_line_label = "Limit Airmass" if self._plot_airmass else "Limit Altitude"
        self.ax_alt.axhline(
            limit_line_value,
            color="red",
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
                color="orange", linewidth=1.2, linestyle='-',
                alpha=0.8, label="Sun"
            )
            self.sun_line.set_visible(self.sun_check.isChecked())

        # Moon altitude curve (always plot, visibility controlled)
        if "moon_alt" in data:
            moon_plot_values = self._plot_y_values(data["moon_alt"])
            self.moon_line, = self.ax_alt.plot(
                times, moon_plot_values,
                color="silver", linewidth=1.2, linestyle='-',
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
        # self.ax_alt.legend(loc="upper right")
        # Hour labels in the observer's local timezone
        self.ax_alt.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=tz))
        # Display selected observation date
        date_str = self.date_edit.date().toString("yyyy-MM-dd")
        self.ax_alt.set_title(f"Date: {date_str}")
        # Current time indicator
        now = datetime.now(tz)
        data["now_local"] = now
        self.now_line = self.ax_alt.axvline(float(mdates.date2num(now)), color="magenta", linestyle=":", linewidth=1.2, label="Now")

        # Update time labels
        # Local time
        self.localtime_label.setText(now.strftime("%Y-%m-%d %H:%M:%S"))
        # UTC time (with globe icon)
        now_utc = datetime.now(timezone.utc)
        self.utctime_label.setText(f"{now_utc.strftime('%Y-%m-%d %H:%M:%S')}")

        # Apply default alpha and width based on altitude limit
        for name, line, is_over in self.vis_lines:
            line.set_linewidth(1.4)
            if is_over:
                line.set_alpha(0.7)
            else:
                line.set_alpha(0.3)
        # Highlight selected targets over limit
        sel_rows = [i.row() for i in self.table_view.selectionModel().selectedRows()]
        sel_names = [self.targets[i].name for i in sel_rows]
        for name, line, is_over in self.vis_lines:
            if name in sel_names and is_over:
                line.set_alpha(1.0)
                line.set_linewidth(2.5)
        visible_targets = sum(1 for flag in row_enabled if flag)
        if self._calc_started_at > 0:
            self._last_calc_stats = CalcRunStats(
                duration_s=max(0.0, perf_counter() - self._calc_started_at),
                visible_targets=visible_targets,
                total_targets=len(self.targets),
            )
            self._calc_started_at = 0.0
        self._update_selected_details()
        self._update_status_bar()
        self._reset_plot_navigation_home()
        self.plot_canvas.draw_idle()
        self._update_polar_positions(data)

    @Slot()
    def _toggle_visibility(self):
        """Show or hide sun and moon lines without recalculation."""
        if hasattr(self, 'sun_line') and self.sun_line:
            self.sun_line.set_visible(self.sun_check.isChecked())
        if hasattr(self, 'moon_line') and self.moon_line:
            self.moon_line.set_visible(self.moon_check.isChecked())
        self.plot_canvas.draw_idle()

    @Slot(int)
    def _on_plot_mode_switch_changed(self, value: int):
        checked = bool(value)
        if self._plot_airmass == checked:
            self._refresh_plot_mode_switch()
            return
        self._plot_airmass = checked
        self._refresh_plot_mode_switch()
        self.settings.setValue("general/plotAirmass", self._plot_airmass)
        if isinstance(self.last_payload, dict):
            self._update_plot(self.last_payload)


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
            self._update_polar_positions(data, dynamic_only=True)


    @Slot()
    def _update_polar_selection(self, selected, deselected):
        """Update highlight for selected targets on polar plot."""
        # Gather selected rows
        sel_rows = [idx.row() for idx in self.table_view.selectionModel().selectedRows()]
        # Prepare coordinates for selected targets
        sel_coords = []
        for i, tgt in enumerate(self.targets):
            if i in sel_rows:
                if i < len(self.table_model.row_enabled) and not self.table_model.row_enabled[i]:
                    continue
                alt = self.table_model.current_alts[i] if i < len(self.table_model.current_alts) else None
                az = self.table_model.current_azs[i] if i < len(self.table_model.current_azs) else None
                if alt is not None and az is not None and alt > 0:
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
                color='green', linewidth=0.8, linestyle=':', alpha=0.7, zorder=1
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
        sel_names = [self.targets[i].name for i in sel_rows]
        for name, line, is_over in self.vis_lines:
            if name in sel_names and is_over:
                line.set_alpha(1.0)
                line.set_linewidth(2.5)
            else:
                line.set_linewidth(1.4)
                if is_over:
                    line.set_alpha(0.7)
                else:
                    line.set_alpha(0.3)
        self.plot_canvas.draw_idle()

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
        self.polar_scatter.set_offsets(np.array(tgt_coords) if tgt_coords else np.empty((0, 2)))

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
                sun_alt_series = np.array(data["sun_alt"])
                sun_az_series = np.array(data["sun_az"])
                mask = sun_alt_series > 0
                if mask.any():
                    theta = np.deg2rad(sun_az_series[mask])
                    r = 90 - sun_alt_series[mask]
                    wrap_pts = np.where(np.abs(np.diff(theta)) > np.pi)[0] + 1
                    theta = np.insert(theta, wrap_pts, np.nan)
                    r = np.insert(r, wrap_pts, np.nan)
                    self.sun_path_line, = self.polar_ax.plot(
                        theta, r, color="gold", linewidth=0.9, linestyle="--", alpha=0.7, zorder=1
                    )

            if self.show_moon_path and has_moon_path:
                moon_alt_series = np.array(data["moon_alt"])
                moon_az_series = np.array(data["moon_az"])
                mask = moon_alt_series > 0
                if mask.any():
                    theta = np.deg2rad(moon_az_series[mask])
                    r = 90 - moon_alt_series[mask]
                    wrap_pts = np.where(np.abs(np.diff(theta)) > np.pi)[0] + 1
                    theta = np.insert(theta, wrap_pts, np.nan)
                    r = np.insert(r, wrap_pts, np.nan)
                    self.moon_path_line, = self.polar_ax.plot(
                        theta, r, color="silver", linewidth=0.9, linestyle="--", alpha=0.7, zorder=1
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
                    facecolors='none', edgecolors='purple', marker='o', s=80, linewidths=1.5, zorder=3, alpha=0.3
                )
                dot = self.polar_ax.scatter(
                    [theta_pol], [r_pol], c='purple', marker='.', s=30, zorder=4, alpha=0.3
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
                    theta_full, r_full, color='red', linestyle='-', linewidth=0.5, alpha=0.4
                )
                self._polar_static_signature = signature

        self._clock_polar_tick += 1
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

    def _build_ai_panel(self) -> QWidget:
        panel = QFrame(self)
        panel.setObjectName("AIAssistantPanel")
        panel.setMinimumHeight(170)
        panel.setMaximumHeight(260)

        layout = QHBoxLayout(panel)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        btn_col = QVBoxLayout()
        btn_col.setSpacing(6)

        describe_btn = QPushButton("Describe Object")
        describe_btn.clicked.connect(self._ai_describe_target)
        btn_col.addWidget(describe_btn)

        btn_col.addStretch(1)
        btn_widget = QWidget(panel)
        btn_widget.setLayout(btn_col)
        btn_widget.setFixedWidth(145)
        layout.addWidget(btn_widget)

        self.ai_output = QTextEdit(panel)
        self.ai_output.setReadOnly(True)
        self.ai_output.setPlaceholderText(
            "AI responses will appear here.\n"
            "Configure your local LLM in Settings -> General Settings."
        )
        layout.addWidget(self.ai_output, 1)

        input_col = QVBoxLayout()
        input_col.setSpacing(6)

        self.ai_input = QLineEdit(panel)
        self.ai_input.setPlaceholderText("Ask about tonight's observing plan...")
        self.ai_input.returnPressed.connect(self._send_ai_query)
        input_col.addWidget(self.ai_input)

        send_btn = QPushButton("Send")
        send_btn.clicked.connect(self._send_ai_query)
        input_col.addWidget(send_btn)

        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._clear_ai_messages)
        input_col.addWidget(clear_btn)

        self.ai_status_label = QLabel("Ready", panel)
        self.ai_status_label.setObjectName("SectionHint")
        input_col.addWidget(self.ai_status_label)
        input_col.addStretch(1)

        input_widget = QWidget(panel)
        input_widget.setLayout(input_col)
        input_widget.setFixedWidth(210)
        layout.addWidget(input_widget)

        return panel

    @Slot(bool)
    def _toggle_ai_panel(self, checked: bool) -> None:
        if hasattr(self, "ai_panel"):
            self.ai_panel.setVisible(bool(checked))
        if hasattr(self, "ai_toggle_btn"):
            self.ai_toggle_btn.setText("Hide AI" if checked else "AI Assistant")

    def _build_session_context(self, *, include_current_snapshot: bool = True) -> str:
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
                "Events: "
                f"sunset {_fmt_event('sunset')}, sunrise {_fmt_event('sunrise')}, "
                f"dusk {_fmt_event('dusk')}, dawn {_fmt_event('dawn')}"
            )
            moon_phase = payload.get("moon_phase")
            if moon_phase is not None:
                try:
                    parts.append(f"Moon phase: {float(moon_phase):.1f}%")
                except Exception:
                    pass

        if not self.targets:
            parts.append("Targets: none")
            return "\n".join(parts)

        rows: list[str] = []
        max_items = 40
        for idx, target in enumerate(self.targets):
            if idx >= max_items:
                rows.append(f"  - ... and {len(self.targets) - max_items} more")
                break
            details: list[str] = [f"priority {target.priority}"]
            if target.observed:
                details.append("observed yes")

            metrics = self.target_metrics.get(target.name)
            if metrics is not None:
                details.extend(
                    [
                        f"score {metrics.score:.1f}",
                        f"hours_above_limit {metrics.hours_above_limit:.2f} h",
                        f"max_altitude {metrics.max_altitude_deg:.1f} deg",
                        f"peak_moon_sep {metrics.peak_moon_sep_deg:.1f} deg",
                    ]
                )

            window = self.target_windows.get(target.name)
            if window is not None:
                try:
                    start_txt = window[0].astimezone(tz).strftime("%Y-%m-%d %H:%M")
                    end_txt = window[1].astimezone(tz).strftime("%Y-%m-%d %H:%M")
                except Exception:
                    start_txt = window[0].strftime("%Y-%m-%d %H:%M")
                    end_txt = window[1].strftime("%Y-%m-%d %H:%M")
                details.append(f"best_window_local {start_txt} -> {end_txt}")
            else:
                details.append("best_window_local none under current constraints")

            if include_current_snapshot:
                if idx < len(self.table_model.current_alts):
                    alt_now = self.table_model.current_alts[idx]
                    if math.isfinite(alt_now):
                        details.append(f"current_alt_snapshot {alt_now:.1f} deg")
                if idx < len(self.table_model.current_azs):
                    az_now = self.table_model.current_azs[idx]
                    if math.isfinite(az_now):
                        details.append(f"current_az_snapshot {az_now:.1f} deg")
                if idx < len(self.table_model.current_seps):
                    moon_sep_now = self.table_model.current_seps[idx]
                    if math.isfinite(moon_sep_now):
                        details.append(f"current_moon_sep_snapshot {moon_sep_now:.1f} deg")

            rows.append(
                f"  - {target.name}: RA {target.ra:.3f} deg, Dec {target.dec:.3f} deg; "
                + "; ".join(details)
            )
        parts.append("Current targets:\n" + "\n".join(rows))
        return "\n".join(parts)

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
        "When suggesting objects, prefer common catalog names (Messier, NGC, IC, etc.). "
        "Use the provided session context and prioritize objects practical for the current site and night."
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
        primary_key = cache_keys[0]
        if primary_key:
            primary_cached = self._simbad_compact_cache.get(primary_key)
            if primary_cached is not None:
                return dict(primary_cached)
        secondary_key = cache_keys[1]
        if secondary_key and secondary_key != primary_key:
            secondary_cached = self._simbad_compact_cache.get(secondary_key)
            if secondary_cached is not None:
                secondary_status = str(secondary_cached.get("_simbad_status", "")).strip().lower()
                if secondary_status == "matched":
                    return dict(secondary_cached)

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
            return {"_simbad_status": "lookup_failed"}

        if not _simbad_has_row(result):
            compact_data = {"_simbad_status": "not_found"}
            for key in cache_keys:
                if key:
                    self._simbad_compact_cache[key] = dict(compact_data)
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

    def _bhtom_api_token(self) -> str:
        token = (
            self.settings.value("general/bhtomApiToken", "", type=str)
            or os.getenv("BHTOM_API_TOKEN", "")
        ).strip()
        if not token:
            raise RuntimeError("Suggest Targets requires a BHTOM API token in Settings -> General Settings.")
        return token

    def _fetch_bhtom_target_page(self, page: int, token: Optional[str] = None) -> object:
        auth_token = token or self._bhtom_api_token()
        endpoint = f"{self._bhtom_api_base_url()}{BHTOM_TARGET_LIST_PATH}"
        body = json.dumps(
            {
                "page": page,
                "type": "SIDEREAL",
                "importanceMin": BHTOM_SUGGESTION_MIN_IMPORTANCE,
            }
        ).encode("utf-8")
        req = Request(
            endpoint,
            data=body,
            headers={
                "Authorization": f"Token {auth_token}",
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
                raise RuntimeError("BHTOM unauthorized (401/403). Check BHTOM_API_TOKEN.") from exc
            if detail:
                raise RuntimeError(f"BHTOM request failed ({exc.code}): {detail[:240]}") from exc
            raise RuntimeError(f"BHTOM request failed ({exc.code}).") from exc
        except Exception as exc:
            raise RuntimeError(f"BHTOM lookup failed: {exc}") from exc

        try:
            return json.loads(raw)
        except Exception as exc:
            raise RuntimeError("BHTOM returned non-JSON response.") from exc

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def _pick_first_present(sources: list[dict[str, Any]], *keys: str) -> object:
        for source in sources:
            for key in keys:
                if key in source and source[key] not in (None, ""):
                    return source[key]
        return None

    def _build_bhtom_candidate(self, item: dict[str, Any]) -> Optional[dict[str, object]]:
        nested_sources = [item]
        for key in ("target", "target_data", "data", "object", "coordinates"):
            nested = item.get(key)
            if isinstance(nested, dict):
                nested_sources.append(nested)

        name_raw = self._pick_first_present(nested_sources, "name", "target_name", "display_name", "identifier")
        name = _normalize_catalog_display_name(name_raw)
        if not name:
            return None

        ra_deg = _safe_float(self._pick_first_present(nested_sources, "ra", "raDeg", "ra_deg", "rightAscension"))
        dec_deg = _safe_float(self._pick_first_present(nested_sources, "dec", "decDeg", "dec_deg", "declination"))
        if ra_deg is None or dec_deg is None:
            return None

        classification = str(
            self._pick_first_present(
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
            self._pick_first_present(
                nested_sources,
                "mag_last",
                "last_mag",
                "lastMagnitude",
                "magnitude",
                "mag",
            )
        )
        importance = _safe_float(self._pick_first_present(nested_sources, "importance", "importance_value")) or 0.0
        bhtom_priority = _safe_int(self._pick_first_present(nested_sources, "priority", "observing_priority")) or 0
        target_priority = max(1, min(5, int(bhtom_priority))) if bhtom_priority > 0 else 3
        sun_separation = _safe_float(
            self._pick_first_present(nested_sources, "sun_separation", "sunSeparation", "sun")
        )
        source_id_raw = self._pick_first_present(nested_sources, "id", "pk", "target_id", "targetId", "name")
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

    def _fetch_bhtom_target_candidates(self) -> list[dict[str, object]]:
        token = self._bhtom_api_token()
        cache_key = (self._bhtom_api_base_url(), token)
        cache_age = perf_counter() - self._bhtom_candidate_cache_loaded_at
        if (
            self._bhtom_candidate_cache_key == cache_key
            and self._bhtom_candidate_cache is not None
            and cache_age < BHTOM_SUGGESTION_CACHE_TTL_S
        ):
            return list(self._bhtom_candidate_cache)

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

    def _reload_local_target_suggestions(self) -> tuple[list[dict[str, object]], list[str]]:
        self._bhtom_candidate_cache_key = None
        self._bhtom_candidate_cache = None
        self._bhtom_candidate_cache_loaded_at = 0.0
        return self._build_local_target_suggestions()

    def _build_local_target_suggestions(self) -> tuple[list[dict[str, object]], list[str]]:
        payload = self.full_payload if isinstance(getattr(self, "full_payload", None), dict) else None
        if not payload:
            return [], ["Run a visibility calculation first so suggestions use tonight's sky."]

        if not self.table_model.site:
            return [], ["Set a valid observing site before requesting suggestions."]

        try:
            tz_name = str(payload.get("tz", self.table_model.site.timezone_name))
            tz = pytz.timezone(tz_name)
        except Exception:
            tz_name = self.table_model.site.timezone_name
            tz = pytz.UTC

        try:
            time_datetimes = [t.astimezone(tz) for t in mdates.num2date(payload["times"])]
        except Exception:
            return [], ["Visibility samples are unavailable in the current plot state."]
        if not time_datetimes:
            return [], ["Visibility samples are unavailable in the current plot state."]

        site = Site(
            name=self.table_model.site.name,
            latitude=self._read_site_float(self.lat_edit),
            longitude=self._read_site_float(self.lon_edit),
            elevation=self._read_site_float(self.elev_edit),
        )
        observer = Observer(location=site.to_earthlocation(), timezone=tz_name)
        time_samples = Time(time_datetimes)
        moon_ra = np.array(payload.get("moon_ra", np.full(len(time_datetimes), np.nan)), dtype=float)
        moon_dec = np.array(payload.get("moon_dec", np.full(len(time_datetimes), np.nan)), dtype=float)
        if moon_ra.shape[0] != len(time_datetimes) or moon_dec.shape[0] != len(time_datetimes):
            return [], ["Moon position samples are unavailable in the current plot state."]
        moon_coords = SkyCoord(ra=moon_ra * u.deg, dec=moon_dec * u.deg)

        sun_alt_series = np.array(payload.get("sun_alt", np.full(len(time_datetimes), np.nan)), dtype=float)
        sun_alt_limit = self._sun_alt_limit()
        obs_sun_mask = np.isfinite(sun_alt_series) & (sun_alt_series <= sun_alt_limit)
        limit = float(self.limit_spin.value())
        min_moon_sep = float(self.min_moon_sep_spin.value()) if hasattr(self, "min_moon_sep_spin") else 0.0
        sample_hours = 24.0 / max(len(time_datetimes) - 1, 1)

        current_names = {_normalize_catalog_token(target.name) for target in self.targets}
        current_source_ids = {_normalize_catalog_token(target.source_object_id) for target in self.targets if target.source_object_id}
        current_coords = [target.skycoord for target in self.targets]

        candidates = self._fetch_bhtom_target_candidates()
        ranked: list[dict[str, object]] = []
        skipped_notes: list[str] = []

        for candidate_info in candidates:
            target = candidate_info["target"]
            assert isinstance(target, Target)
            normalized_name = _normalize_catalog_token(target.name)
            normalized_source = _normalize_catalog_token(target.source_object_id)
            if normalized_name in current_names or (normalized_source and normalized_source in current_source_ids):
                continue
            if any(float(target.skycoord.separation(coord).deg) < 0.05 for coord in current_coords):
                continue

            fixed = FixedTarget(name=target.name, coord=target.skycoord)
            altaz = observer.altaz(time_samples, fixed)
            altitude = np.array(altaz.alt.deg, dtype=float)  # type: ignore[arg-type]
            moon_sep = np.array(target.skycoord.separation(moon_coords).deg, dtype=float)
            metrics = compute_target_metrics(
                altitude_deg=altitude,
                moon_sep_deg=moon_sep,
                limit_altitude=limit,
                sample_hours=sample_hours,
                priority=max(1, min(5, int(candidate_info.get("bhtom_priority", 3) or 3))),
                observed=False,
                valid_mask=obs_sun_mask,
            )

            valid_mask = np.isfinite(altitude) & (altitude >= limit) & obs_sun_mask
            if not valid_mask.any():
                continue

            valid_indices = np.where(valid_mask)[0]
            runs = np.split(valid_indices, np.where(np.diff(valid_indices) != 1)[0] + 1)
            best_run = max(runs, key=len)
            start_idx = int(best_run[0])
            end_idx = min(int(best_run[-1]) + 1, len(time_datetimes) - 1)
            window_start = time_datetimes[start_idx]
            window_end = time_datetimes[end_idx]
            best_window_airmass = self._airmass_from_altitude(altitude[best_run])
            finite_window_airmass = best_window_airmass[np.isfinite(best_window_airmass)]
            best_airmass = float(np.min(finite_window_airmass)) if finite_window_airmass.size > 0 else None
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
                        min_moon_sep > 0.0
                        and min_window_moon_sep is not None
                        and round(float(min_window_moon_sep), 1) < min_moon_sep
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
    def _send_ai_query(self) -> None:
        text = self.ai_input.text().strip() if hasattr(self, "ai_input") else ""
        if not text:
            return
        self.ai_input.clear()
        context = self._build_session_context()
        prompt = f"Current session context:\n{context}\n\nUser question: {text}"
        self._dispatch_llm(prompt, tag="chat", label=f"You: {text}")

    @Slot()
    def _ai_describe_target(self) -> None:
        target = self._selected_target_or_none()
        if target is None:
            QMessageBox.information(self, "No selection", "Select one target first.")
            return
        if hasattr(self, "ai_toggle_btn") and not self.ai_toggle_btn.isChecked():
            self.ai_toggle_btn.setChecked(True)
        self._append_ai_message(f"Describe: {target.name}", is_user=True)
        self._append_ai_message(self._build_compact_target_description(target), is_ai=True)
        worker = self._llm_worker
        if hasattr(self, "ai_status_label") and (worker is None or not worker.isRunning()):
            self.ai_status_label.setText("Ready")

    @Slot()
    def _ai_suggest_targets(self) -> None:
        if hasattr(self, "ai_status_label"):
            self.ai_status_label.setText("Loading suggestions...")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            suggestions, notes = self._build_local_target_suggestions()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Local target suggestion failed: %s", exc)
            QMessageBox.warning(self, "Suggest Targets", str(exc))
            if hasattr(self, "ai_status_label"):
                self.ai_status_label.setText("Ready")
            QApplication.restoreOverrideCursor()
            return
        QApplication.restoreOverrideCursor()

        if not suggestions:
            QMessageBox.information(
                self,
                "Suggest Targets",
                "\n".join(notes) if notes else "No BHTOM targets matched the current night window.",
            )
            if hasattr(self, "ai_status_label"):
                self.ai_status_label.setText("Ready")
            return

        dlg = SuggestedTargetsDialog(
            suggestions=suggestions,
            notes=notes,
            moon_sep_threshold=float(self.min_moon_sep_spin.value()) if hasattr(self, "min_moon_sep_spin") else 0.0,
            initial_score_filter=float(self.min_score_spin.value()) if hasattr(self, "min_score_spin") else 0.0,
            bhtom_base_url=self._bhtom_api_base_url(),
            add_callback=self._append_target_to_plan,
            reload_callback=self._reload_local_target_suggestions,
            parent=self,
        )
        dlg.exec()

        worker = self._llm_worker
        if hasattr(self, "ai_status_label") and (worker is None or not worker.isRunning()):
            self.ai_status_label.setText("Ready")

    def _dispatch_llm(self, prompt: str, tag: str, label: str) -> None:
        worker = self._llm_worker
        if worker is not None and worker.isRunning():
            self._append_ai_message(
                "The AI assistant is still processing the previous request.",
                is_error=True,
            )
            return
        if hasattr(self, "ai_toggle_btn") and not self.ai_toggle_btn.isChecked():
            self.ai_toggle_btn.setChecked(True)
        self._append_ai_message(label, is_user=True)
        self._start_ai_response_message()
        if hasattr(self, "ai_status_label"):
            self.ai_status_label.setText("Thinking...")
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
        worker.start()

    @Slot(str, str)
    def _on_ai_chunk(self, tag: str, text: str) -> None:
        if not text:
            return
        if hasattr(self, "ai_status_label"):
            self.ai_status_label.setText("Streaming...")
        self._append_ai_stream_chunk(text)

    @Slot(str, str)
    def _on_ai_response(self, tag: str, text: str) -> None:
        logger.info("LLM response received (tag=%s, length=%d)", tag, len(text))
        self._finalize_ai_response(text)

    @Slot(str)
    def _on_ai_error(self, message: str) -> None:
        logger.warning("LLM error: %s", message)
        self._fail_ai_response(message)

    @Slot()
    def _on_ai_worker_finished(self) -> None:
        if hasattr(self, "ai_status_label"):
            self.ai_status_label.setText("Ready")
        idx = self._ai_stream_message_index
        if idx is not None and 0 <= idx < len(self._ai_messages):
            if not self._ai_messages[idx].get("text", "").strip():
                self._ai_messages.pop(idx)
                self._render_ai_messages()
        self._ai_stream_message_index = None
        sender = self.sender()
        if sender is self._llm_worker:
            self._llm_worker = None

    def _clear_ai_messages(self) -> None:
        self._ai_messages.clear()
        self._ai_stream_message_index = None
        if hasattr(self, "ai_output"):
            self.ai_output.clear()

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

    def _render_ai_messages(self) -> None:
        if not hasattr(self, "ai_output"):
            return
        palette = self.ai_output.palette()
        base = palette.color(QPalette.Base)
        alt_base = palette.color(QPalette.AlternateBase)
        text_color = palette.color(QPalette.Text)
        highlight = palette.color(QPalette.Highlight)
        accent_red = QColor("#dc2626")

        user_border = self._mix_qcolors(highlight, base, 0.45)
        ai_border = self._mix_qcolors(text_color, base, 0.14)
        error_border = self._mix_qcolors(accent_red, base, 0.38)
        muted_text = self._mix_qcolors(text_color, base, 0.42)

        parts: list[str] = []
        for message in self._ai_messages:
            kind = message.get("kind", "ai")
            raw_text = str(message.get("text", ""))
            escaped = html_module.escape(raw_text).replace("\n", "<br>")
            if kind == "user":
                html = (
                    '<div style="margin:6px 0;text-align:right">'
                    '<span style="display:inline-block;max-width:95%;padding:6px 9px;'
                    'background:transparent;'
                    f'border:1px solid {self._qcolor_css(user_border)};'
                    f'border-radius:10px;color:{self._qcolor_css(text_color)}">'
                    f"<b>{escaped}</b></span></div>"
                )
            elif kind == "error":
                html = (
                    '<div style="margin:6px 0;text-align:left">'
                    '<span style="display:inline-block;max-width:95%;padding:6px 9px;'
                    'background:transparent;'
                    f'border:1px solid {self._qcolor_css(error_border)};'
                    f'border-radius:10px;color:{self._qcolor_css(text_color)}">'
                    f"{escaped}</span></div>"
                )
            else:
                body = escaped or (
                    f'<span style="color:{self._qcolor_css(muted_text)}"><i>...</i></span>'
                )
                html = (
                    '<div style="margin:6px 0;text-align:left">'
                    '<span style="display:inline-block;max-width:95%;padding:6px 9px;'
                    'background:transparent;'
                    f'border:1px solid {self._qcolor_css(ai_border)};'
                    f'border-radius:10px;color:{self._qcolor_css(text_color)}">'
                    f"{body}</span></div>"
                )
            parts.append(html)
        self.ai_output.setHtml("".join(parts))
        scroll = self.ai_output.verticalScrollBar()
        scroll.setValue(scroll.maximum())

    def _start_ai_response_message(self) -> None:
        self._ai_stream_message_index = self._append_ai_message("", is_ai=True)

    def _append_ai_stream_chunk(self, text: str) -> None:
        if not text:
            return
        idx = self._ai_stream_message_index
        if idx is None or not (0 <= idx < len(self._ai_messages)):
            idx = self._append_ai_message("", is_ai=True)
            self._ai_stream_message_index = idx
        self._ai_messages[idx]["text"] = self._ai_messages[idx].get("text", "") + text
        self._render_ai_messages()

    def _finalize_ai_response(self, text: str) -> None:
        idx = self._ai_stream_message_index
        if idx is not None and 0 <= idx < len(self._ai_messages):
            self._ai_messages[idx]["kind"] = "ai"
            self._ai_messages[idx]["text"] = text
            self._render_ai_messages()
        else:
            self._append_ai_message(text, is_ai=True)
        self._ai_stream_message_index = None

    def _fail_ai_response(self, message: str) -> None:
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

    def _append_ai_message(
        self,
        text: str,
        *,
        is_user: bool = False,
        is_ai: bool = False,
        is_error: bool = False,
    ) -> int:
        kind = "user" if is_user else "error" if is_error else "ai" if is_ai else "info"
        self._ai_messages.append({"kind": kind, "text": str(text)})
        self._render_ai_messages()
        return len(self._ai_messages) - 1

    @Slot(int)
    def _change_date(self, offset_days: int):
        """Shift the selected date by the given number of days and re-plot."""
        new_date = self.date_edit.date().addDays(offset_days)
        self.date_edit.setDate(new_date)
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
        self._replot_timer.start()

    @Slot()
    def _update_limit(self):
        """Update limit altitude, refresh table warnings, and replot."""
        # Update table model limit so coloring reflects the new threshold
        new_limit = float(self.limit_spin.value())
        self.table_model.limit = new_limit
        self._emit_table_data_changed()
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

    # If a plan file is specified, load targets and immediately plot
    if args.plan:
        try:
            with open(args.plan, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
            # Populate the existing targets list so the model sees it
            win.targets.clear()
            win.target_metrics.clear()
            win.target_windows.clear()
            for entry in data:
                win.targets.append(Target(**entry))
            win._clear_table_dynamic_cache()
            win.table_model.layoutChanged.emit()
            win._apply_table_settings()
            win._apply_default_sort()
            win._fetch_missing_magnitudes_async()
            # Now run the plot (also sets the site and refreshes buttons)
            win._run_plan()
        except Exception as e:
            QMessageBox.critical(None, "Startup Load Error", f"Failed to load plan '{args.plan}': {e}")

    win.show()
    sys.exit(app.exec())
