from __future__ import annotations

import io
import logging
import threading
import warnings
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.coordinates import Angle, SkyCoord
from astroplan import FixedTarget
from astroplan.plots import plot_finder_image
from astroquery.simbad import Simbad
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QThread, Signal, Slot
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from astroplanner.i18n import current_language, localize_widget_tree
from astroplanner.models import Target
from astroplanner.ui.common import _fit_dialog_to_screen
from astroplanner.ui.theme_utils import (
    _set_button_icon_kind,
    _set_button_variant,
    _style_dialog_button_box,
)

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

FINDER_HTTP_TIMEOUT_S = 5.0
_FINDER_PATCH_LOCK = threading.Lock()


def _normalize_catalog_token(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _decode_simbad_value(value: object) -> str:
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="ignore").strip()
    return str(value).strip()


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
        if raw is None or np.ma.is_masked(raw):
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


def _finder_survey_candidates(key: object) -> list[str]:
    norm = str(key or "").strip().lower()
    if norm in {"decals", "cds/p/decals/dr5/color"}:
        norm = "2mass"
    elif norm == "cds/p/dss2/color":
        norm = "dss2"
    elif norm == "cds/p/panstarrs/dr1/color-z-zg-g":
        norm = "panstarrs"
    elif norm == "cds/p/2mass/color":
        norm = "2mass"
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
                raise LookupError(f"SkyView returned no image for survey '{survey}'") from exc
            except TypeError as exc:
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


class AddTargetDialog(QDialog):
    """Two-step target add dialog with lazy-expanded metadata details."""

    def __init__(
        self,
        resolver: Callable[[str, str], Target],
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
        _set_button_variant(self.resolve_btn, "secondary")
        _set_button_icon_kind(self.resolve_btn, "resolve")
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
        _style_dialog_button_box(self.button_box)
        ok_btn = self.button_box.button(QDialogButtonBox.Ok)
        if ok_btn:
            ok_btn.setEnabled(False)
        layout.addWidget(self.button_box)
        _fit_dialog_to_screen(
            self,
            preferred_width=760,
            preferred_height=620,
            min_width=560,
            min_height=420,
        )

        self.query_edit.returnPressed.connect(self._resolve_target)
        self.query_edit.textChanged.connect(self._on_query_changed)
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        localize_widget_tree(self, current_language())

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


__all__ = [
    "AddTargetDialog",
    "FinderChartWorker",
    "MetadataLookupWorker",
    "TARGET_SEARCH_SOURCES",
]
