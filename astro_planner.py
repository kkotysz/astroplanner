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
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

# Third-party imports
import numpy as np
import pytz
import ephem
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord, Angle
from astropy.time import Time
from astroplan import FixedTarget, Observer

from astroquery.simbad import Simbad
from pydantic import BaseModel, Field, ValidationError, ConfigDict
from timezonefinder import TimezoneFinder
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import warnings
from astroplan.observer import TargetAlwaysUpWarning
from typing import Any
from matplotlib.figure import Figure

# GUI imports (PySide6)
from PySide6.QtCore import (
    QAbstractTableModel,
    QDate,
    QModelIndex,
    Qt,
    QThread,
    Signal,
    Slot,
    QSize,
    QTimer,
    QItemSelectionModel,
    QSettings,
)
from PySide6.QtGui import (
    QAction,
    QBrush,
    QColor,
    QFont,
    QFontDatabase,
    QIcon,
    QKeySequence,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QFormLayout,
    QHeaderView,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressDialog,
    QSpinBox,
    QStyle,
    QSizePolicy,
    QTableView,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QInputDialog,
    QColorDialog,
    QDialog,
    QDialogButtonBox,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt import NavigationToolbar2QT as NavigationToolbar

# --- Custom delegate to preserve model background for column 0 even when selected ---
from PySide6.QtWidgets import QStyledItemDelegate, QStyle


class NoSelectBackgroundDelegate(QStyledItemDelegate):
    """Delegate that preserves model background for column 0 even when selected."""
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
        self.setWindowTitle("Table Settings")
        layout = QFormLayout(self)
        # Row height
        self.row_height_spin = QSpinBox(self)
        self.row_height_spin.setRange(10, 100)
        init_h = parent.settings.value("table/rowHeight", 24, type=int)
        self.row_height_spin.setValue(init_h)
        layout.addRow("Row height:", self.row_height_spin)
        # First-column width
        self.first_col_width_spin = QSpinBox(self)
        self.first_col_width_spin.setRange(50, 500)
        init_w = parent.settings.value("table/firstColumnWidth", 100, type=int)
        self.first_col_width_spin.setValue(init_w)
        layout.addRow("First-column width:", self.first_col_width_spin)

        # Font size
        self.font_spin = QSpinBox(self)
        self.font_spin.setRange(8, 16)
        init_fs = parent.settings.value("table/fontSize", 11, type=int)
        self.font_spin.setValue(init_fs)
        layout.addRow("Font size:", self.font_spin)

        # Column visibility
        self.col_checks = {}
        for idx, lbl in enumerate(parent.table_model.headers[:-1]):
            chk = QCheckBox(lbl, self)
            val = parent.settings.value(f"table/col{idx}", True, type=bool)
            chk.setChecked(val)
            layout.addRow(f"Show {lbl}:", chk)
            self.col_checks[idx] = chk

        # Highlight colors
        default_colors = {"below":"#ff8080","limit":"#ffff80","above":"#b3ffb3"}
        def _pick_color(key, btn):
            col = QColorDialog.getColor(QColor(parent.settings.value(f"table/color/{key}", default_colors[key])), self, f"Pick {key} color")
            if col.isValid():
                btn.setStyleSheet(f"background:{col.name()}")
        for key in ("below","limit","above"):
            btn = QPushButton(f"{key.capitalize()} highlight", self)
            init = parent.settings.value(f"table/color/{key}", default_colors[key])
            btn.setStyleSheet(f"background:{init}")
            btn.clicked.connect(lambda _,k=key,b=btn: _pick_color(k,b))
            layout.addRow(f"{key.capitalize()} color:", btn)
            setattr(self, f"{key}_btn", btn)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    def accept(self):
        s = self.parent().settings
        s.setValue("table/rowHeight", self.row_height_spin.value())
        s.setValue("table/firstColumnWidth", self.first_col_width_spin.value())
        s.setValue("table/fontSize", self.font_spin.value())
        for idx, chk in self.col_checks.items():
            s.setValue(f"table/col{idx}", chk.isChecked())
        for key in ("below","limit","above"):
            btn = getattr(self, f"{key}_btn")
            col = btn.palette().color(btn.backgroundRole()).name()
            s.setValue(f"table/color/{key}", col)
        self.parent()._apply_table_settings()
        super().accept()

# --- General Settings Dialog ---
class GeneralSettingsDialog(QDialog):
    """Configure default site, date, samples & clock refresh."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("General Settings")
        layout = QFormLayout(self)

        # Default Observatory
        self.site_combo = QComboBox(self)
        self.site_combo.addItems(parent.observatories.keys())
        init_site = parent.settings.value("general/defaultSite", parent.obs_combo.currentText(), type=str)
        self.site_combo.setCurrentText(init_site)
        layout.addRow("Default Observatory:", self.site_combo)

        # Visibility samples
        self.ts_spin = QSpinBox(self)
        self.ts_spin.setRange(50, 1000)
        init_ts = parent.settings.value("general/timeSamples", 240, type=int)
        self.ts_spin.setValue(init_ts)
        layout.addRow("Visibility samples:", self.ts_spin)

        # OK / Cancel
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def accept(self):
        s = self.parent().settings
        s.setValue("general/defaultSite", self.site_combo.currentText())
        s.setValue("general/timeSamples", self.ts_spin.value())
        self.parent()._apply_general_settings()
        # Immediately re-run the plan so samples take effect
        self.parent()._replot_timer.start()
        super().accept()

# Number of time samples for visibility curve (lower = faster)
TIME_SAMPLES = 240 

class ClockWorker(QThread):
    updated = Signal(dict)

    def __init__(self, site: Site, targets: list[Target], parent=None):
        super().__init__(parent)
        self.site = site
        self.targets = targets
        self.running = True

    def run(self):
        tz_name = self.site.timezone_name
        tz = pytz.timezone(tz_name)

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
                moon = ephem.Moon(eph_obs)
                moon_coord = SkyCoord(ra=Angle(moon.ra, u.rad), dec=Angle(moon.dec, u.rad))
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

            self.msleep(1000)

    def stop(self):
        self.running = False
        self.quit()
        self.wait()

# --------------------------------------------------
# --- Models (Pydantic) -----------------------------
# --------------------------------------------------
class Target(BaseModel):
    """A celestial target."""

    name: str = Field(..., description="Display name")
    ra: float = Field(..., description="Right Ascension in degrees")
    dec: float = Field(..., description="Declination in degrees")

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
        return TimezoneFinder().timezone_at(lng=self.longitude, lat=self.latitude) or "UTC"


class SessionSettings(BaseModel):
    """Per‑night settings handed to the worker thread."""

    date: QDate
    site: Site
    limit_altitude: float = 35.0  # deg
    time_samples: int = 240       # user-configurable resolution

    # <‑‑ make Pydantic happy with Qt types
    model_config = ConfigDict(arbitrary_types_allowed=True)


# --------------------------------------------------
# --- Astronomy Worker (runs in separate QThread) ---
# --------------------------------------------------
class AstronomyWorker(QThread):
    """Runs astroplan calculations off the GUI thread."""

    finished: Signal = Signal(dict)  # payload with curves & events

    _cache: dict = {}

    def __init__(self, targets: List[Target], settings: SessionSettings, parent=None):
        super().__init__(parent)
        self.targets = targets
        self.settings = settings

    # heavy lifting happens here
    def run(self) -> None:  # noqa: D401
        obs_date = self.settings.date
        site = self.settings.site
        observer = Observer(location=site.to_earthlocation(), timezone=site.timezone_name)

        # Caching key: site coords + elevation + calendar date
        key = (
            site.latitude, site.longitude, site.elevation,
            obs_date.toString("yyyy-MM-dd")
        )
        cache = AstronomyWorker._cache

        # determine local midnight following the chosen observation date’s evening
        tz = pytz.timezone(site.timezone_name)
        next_mid = datetime(obs_date.year(), obs_date.month(), obs_date.day(), 0, 0) + timedelta(days=1)
        local_mid_dt = tz.localize(next_mid)
        midnight = Time(local_mid_dt)

        if key in cache:
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
            moon_alts = []
            moon_azs = []
            for t in times.datetime:
                # PyEphem expects UTC datetime
                eph_observer.date = t
                sun.compute(eph_observer)
                moon.compute(eph_observer)
                # convert radians to degrees
                sun_alts.append(sun.alt * 180.0 / math.pi)
                moon_alts.append(moon.alt * 180.0 / math.pi)
                moon_azs.append(moon.az * 180.0 / math.pi)

            events["sun_alt"] = np.array(sun_alts)
            events["moon_alt"] = np.array(moon_alts)
            events["moon_az"] = np.array(moon_azs)
            # Moon phase from PyEphem (0–100%)
            events["moon_phase"] = moon.phase
            cache[key] = {"times": times, "events": events}

        # Start payload with cached/global events
        payload: dict[str, object] = {k: v for k, v in {"times": jd, **events}.items()}
        for tgt in self.targets:
            fixed = FixedTarget(name=tgt.name, coord=tgt.skycoord)
            altaz = observer.altaz(times, fixed)
            payload[tgt.name] = {
                "altitude": altaz.alt.deg,    # type: ignore[arg-type]
                "azimuth": altaz.az.deg  # type: ignore[arg-type]
            }
        # Tell the GUI which timezone we used
        payload["tz"] = site.timezone_name                      # type: ignore[arg-type]
        self.finished.emit(payload)


# --------------------------------------------------
# --- Qt Table Model for the targets list ----------
# --------------------------------------------------
class TargetTableModel(QAbstractTableModel):
    headers = ["Name", "RA (°)", "HA (h)", "Dec (°)", "Alt (°)", "Az (°)", "Moon Sep (°)", "Actions"]

    def __init__(self, targets: List[Target], site: Optional[Site] = None):
        super().__init__()
        self.targets = targets
        self.site = site
        self.limit: float | None = None
        # Cached current values for table display
        self.current_alts: list[float] = []
        self.current_azs: list[float] = []
        self.current_seps: list[float] = []
        self.color_map: dict[str, QColor] = {}

    # basic model plumbing
    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return len(self.targets)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return len(self.headers)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):  # noqa: N802
        if not index.isValid():
            return None
        tgt = self.targets[index.row()]
        col = index.column()

        # Center-align all cell text except left-align names
        if role == Qt.TextAlignmentRole:
            # Left-align names, center others
            if index.column() == 0:
                return Qt.AlignLeft | Qt.AlignVCenter
            return Qt.AlignCenter | Qt.AlignVCenter
        
        # Column 2: Hour Angle (hours, sexagesimal)
        if role in (Qt.DisplayRole, Qt.EditRole) and index.column() == 2 and self.site:
            now = Time.now()
            loc = self.site.to_earthlocation()
            lst = now.sidereal_time('apparent', loc.lon).hour    # LST in hours
            ra_h = self.targets[index.row()].ra / 15.0           # RA in hours
            ha = (lst - ra_h + 24) % 24
            # Hour Angle in sexagesimal
            ha_angle = Angle(ha, u.hour)
            return ha_angle.to_string(unit=u.hour, sep=":", pad=True, precision=0)

        # Combined row background and per-cell highlights
        if role == Qt.BackgroundRole and self.site and self.limit is not None:
            row = index.row()
            col = index.column()
            alt = (self.current_alts[row]
                   if row < len(self.current_alts) else None)
            if alt is None:
                return None

            # Plot line color for this target
            colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            plot_css = colors[row % len(colors)]
            plot_color = QColor(plot_css)

            # 1) Cause highlight in Alt column (col 3)
            if col == 3:
                if alt < 0:
                    return QBrush(QColor("#ff8080"))  # red
                if alt < self.limit:
                    return QBrush(QColor("#ffff80"))  # yellow
                return QBrush(QColor("#b3ffb3"))      # green

            # 2) Name cell colored by plot color (col 0)
            if col == 0:
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

        # Always render text in black for all columns
        if role == Qt.ForegroundRole:
            return QBrush(QColor("#000000"))

        # FontRole: Use condensed bold font for Name, monospace for numeric columns
        if role == Qt.FontRole:
            if index.column() == 0:
                # Condensed bold font for Name column
                font = QFont("Arial Narrow", 13, QFont.Thin)
                font.setStretch(95)
                return font
            # Monospace font for numeric columns to align digits
            if index.column() in [1, 2, 3, 4, 5, 6]:
                # Use the system fixed-width font
                font = QFontDatabase.systemFont(QFontDatabase.FixedFont)
                return font

        # Tooltip for full name in Name column
        if role == Qt.ToolTipRole and index.column() == 0:
            return tgt.name

        # Tooltip for altitude status in column 3
        if role == Qt.ToolTipRole and col == 3 and self.site:
            alt = self.current_alts[index.row()] if len(self.current_alts) > index.row() else None
            limit = self.limit or 0
            if alt is None:
                return ""
            if alt < 0:
                return "Below horizon"
            if alt < limit:
                return "Below limit altitude"
            return "Above limit altitude"

        if role not in (Qt.DisplayRole, Qt.EditRole):
            return None
        if col == 0:
            return tgt.name
        if col == 1:
            # Right Ascension in sexagesimal (hours)
            ra_angle = Angle(tgt.ra, u.degree)
            return ra_angle.to_string(unit='hourangle', sep=":", pad=True, precision=0)
        if col == 3:
            # Declination in sexagesimal (degrees)
            dec_angle = Angle(tgt.dec, u.degree)
            return dec_angle.to_string(unit='deg', sep=":", alwayssign=True, pad=True, precision=0)
        # Column 4: current altitude
        if col == 4:
            return f"{self.current_alts[index.row()]:.1f}" if len(self.current_alts) > index.row() else ""
        # Column 5: current azimuth
        if col == 5:
            return f"{self.current_azs[index.row()]:.1f}" if len(self.current_azs) > index.row() else ""
        # Column 6: current separation from the Moon
        if col == 6:
            return f"{self.current_seps[index.row()]:.1f}" if len(self.current_seps) > index.row() else ""
        # Column 7: placeholder for button
        if col == 7:
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

        # Mapping dynamic values if available
        alts = getattr(self, "current_alts", [])
        azs = getattr(self, "current_azs", [])
        seps = getattr(self, "current_seps", [])

        if column == 0:
            self.targets.sort(key=lambda t: t.name, reverse=reverse)
        elif column == 1:
            self.targets.sort(key=lambda t: t.ra, reverse=reverse)
        elif column == 2 and self.site:
            # Hour Angle
            now = Time.now()
            lon = self.site.to_earthlocation().lon
            lst = now.sidereal_time('apparent', lon).hour
            def ha(tgt: Target) -> float:
                return (float(lst) - (tgt.ra / 15.0) + 24.0) % 24.0                             # type: ignore[arg-type]
            self.targets.sort(key=ha, reverse=reverse)
        elif column == 3:
            self.targets.sort(key=lambda t: t.dec, reverse=reverse)
        elif column == 4 and len(alts) == len(self.targets):
            # Map by target name to avoid unhashable Target instances
            mapping = dict(zip((t.name for t in self.targets), alts))
            self.targets.sort(key=lambda t: mapping.get(t.name, float("-inf")), reverse=reverse)
        elif column == 5 and len(azs) == len(self.targets):
            # Map by target name to avoid unhashable Target instances
            mapping = dict(zip((t.name for t in self.targets), azs))
            self.targets.sort(key=lambda t: mapping.get(t.name, float("-inf")), reverse=reverse)
        elif column == 6 and len(seps) == len(self.targets):
            # Map by target name to avoid unhashable Target instances
            mapping = dict(zip((t.name for t in self.targets), seps))
            self.targets.sort(key=lambda t: mapping.get(t.name, float("-inf")), reverse=reverse)
        # Else: do nothing for Actions column

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
        rows = json.loads(bytes(mimedata.data("application/x-target-row")).decode())
        rows.sort(reverse=True)
        insert_row = parent.row() if parent.isValid() else len(self.targets)
        moving = [self.targets.pop(r) for r in rows]
        for tgt in reversed(moving):
            self.targets.insert(insert_row, tgt)
        self.layoutChanged.emit()
        return True


# --------------------------------------------------
# --- Main Window ----------------------------------
# --------------------------------------------------
class MainWindow(QMainWindow):
    def _apply_styles(self):
        """Apply a custom stylesheet, fonts, and default icon sizes."""
        # Load a custom sans-serif font if available
        QFontDatabase.addApplicationFont(":/fonts/Roboto-Regular.ttf")  # optional resource
        self.setStyleSheet("""
            QWidget {
                font-family: Arial, sans-serif;
            }
            QPushButton {
                background-color: #467fcf;
                color: white;
                border-radius: 4px;
                padding: 4px 8px;
            }
            QPushButton:hover {
                background-color: #356ab9;
            }
            QToolButton {
                background: transparent;
                padding: 2px;
            }
            QComboBox, QSpinBox, QDateEdit, QLineEdit {
                border: 1px solid #ccc;
                border-radius: 4px;
                padding: 2px 4px;
                min-height: 22px;
                min-width: 120px;
            }
            QTableView {
                /* Preserve custom cell backgrounds by disabling selection background */
                selection-background-color: transparent;
            }
            QTableView::item:selected {
                /* Ensure no override of model-set backgrounds on selection */
                background: transparent;
            }
            QHeaderView::section {
                background-color: #444444;
                color: #e0e0e0;
                padding: 4px;
                border: none;
            }
            QCheckBox {
                spacing: 6px;
            }
        """)
        # Apply icon size globally
        self.setIconSize(QSize(20, 20))
    @Slot(str)
    def _on_obs_change(self, name: str):
        """Populate site fields when an observatory is selected."""
        site = self.observatories[name]
        self.lat_edit.setText(f"{site.latitude}")
        self.lon_edit.setText(f"{site.longitude}")
        self.elev_edit.setText(f"{site.elevation}")
        # Update the table model and replot with debounce
        self.table_model.site = site
        self.table_model.layoutChanged.emit()
        self._replot_timer.start()
        # Restart clock worker to update real-time altitudes for the new site
        self._start_clock_worker()
    def __init__(self):
        super().__init__()
        self.last_payload = None
        self.setWindowTitle("Astronomical Observation Planner")
        self.resize(1100, 680)

        # Persistent user settings
        self.settings = QSettings("YourCompany", "AstroPlanner")

        # state holders
        self.targets: List[Target] = []
        self.worker: Optional[AstronomyWorker] = None  # keep reference!

        # UI ------------------------------------------------
        self.table_model = TargetTableModel(self.targets, site=None)
        self.table_view = QTableView(selectionBehavior=QTableView.SelectRows)
        # Use custom delegate for Name column to preserve its background on selection
        self.table_view.setItemDelegateForColumn(0, NoSelectBackgroundDelegate(self.table_view))
        # Make columns only as wide as their contents
        header = self.table_view.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        self.table_view.verticalHeader().setVisible(False)
        self.table_view.setShowGrid(False)
        self.table_view.setModel(self.table_model)
        # # Apply saved settings now that table_view exists
        # self._load_settings()
        # Enable click‐to‐sort and default‐sort by HA (column 2)
        self.table_view.setSortingEnabled(True)
        self.table_view.horizontalHeader().setSectionsClickable(True)
        # Defer initial sort by HA to after the UI is shown
        QTimer.singleShot(0, lambda: self.table_view.sortByColumn(2, Qt.AscendingOrder))
        self.table_view.setDragDropMode(QTableView.InternalMove)

        # Polar plot for alt-az projection
        self.polar_canvas = FigureCanvas(Figure(figsize=(4, 4), tight_layout=True))
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
        self.polar_ax.set_rlim(0, 90)
        self.polar_ax.set_rlabel_position(135)
        # Will map scatter points to target indices for picking
        self.polar_indices: list[int] = []
        # Label cardinal directions on the polar plot
        self.polar_ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
        self.polar_ax.set_xticklabels(['N', 'E', 'S', 'W'])
        # Draw cardinal labels in white for visibility on dark background
        self.polar_ax.tick_params(axis='x', colors='white')
        # Radial ticks for altitudes 20°, 40°, 60°, and 80° (r = 90 - altitude), labels hidden
        alt_ticks = [20, 40, 60, 80]
        r_ticks = [90 - a for a in alt_ticks]
        self.polar_ax.set_yticks(r_ticks)
        # Hide radial tick labels for a cleaner look
        self.polar_ax.set_yticklabels([])
        self.polar_ax.tick_params(pad=1)
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
        # Holder for visibility line artists per target, with over-limit flag
        self.vis_lines: list[tuple[str, Any, bool]] = []
        # Connect pick event for polar scatter
        self.polar_canvas.mpl_connect('pick_event', self._on_polar_pick)
        # Make polar plot as narrow as its minimum size
        self.polar_canvas.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        # Cap maximum width so plot stays narrow
        self.polar_canvas.setMaximumWidth(180)
        # Tiny margins so E/W labels remain visible yet keep plot compact
        self.polar_canvas.figure.subplots_adjust(left=0.02, right=0.98, bottom=0.06, top=0.94)
        # Minimal internal padding
        self.polar_canvas.figure.tight_layout(pad=0)

        # Debounce frequent requests for plotting
        self._replot_timer = QTimer(self)
        self._replot_timer.setSingleShot(True)
        self._replot_timer.setInterval(300)  # ms
        self._replot_timer.timeout.connect(self._run_plan)

        self.date_edit = QDateEdit()
        # Initialize to current observing night
        self._change_to_today()
        self.date_edit.setCalendarPopup(True)
        self.date_edit.setMaximumWidth(100)

        # Observatory selection
        self.observatories = {
            "OCM": Site(name="OCM", latitude=-24.59, longitude=-70.19, elevation=2800),
            "Białków": Site(name="Białków", latitude=51.474248, longitude=16.657821, elevation=128),
        }
        self.obs_combo = QComboBox()
        self.obs_combo.addItems(self.observatories.keys())
        self.obs_combo.currentTextChanged.connect(self._on_obs_change)
        self.obs_combo.setCurrentText("OCM")
        # Ensure observatory names are fully visible
        self.obs_combo.setMinimumWidth(140)
        self.obs_combo.setMinimumContentsLength(8)
        # Now that observatories and date widget exist, load settings
        self._load_settings()

        self.lat_edit = QLineEdit("-24.59")
        self.lat_edit.setMaximumWidth(90)
        # site/date widgets
        self.lat_edit = QLineEdit("-24.59")
        self.lat_edit.setMaximumWidth(90)
        self.lon_edit = QLineEdit("-70.19")
        self.lon_edit.setMaximumWidth(90)
        self.elev_edit = QLineEdit("2800")
        self.elev_edit.setMaximumWidth(90)
        self.limit_spin = QSpinBox(minimum=0, maximum=90, value=35)
        self.limit_spin.setMaximumWidth(80)

        # Debounced connections for date and limit spin
        self.date_edit.dateChanged.connect(lambda _: self._replot_timer.start())
        self.limit_spin.valueChanged.connect(self._update_limit)
        # Re-plot when latitude, longitude, or elevation is changed
        self.lat_edit.editingFinished.connect(lambda: self._replot_timer.start())
        self.lon_edit.editingFinished.connect(lambda: self._replot_timer.start())
        self.elev_edit.editingFinished.connect(lambda: self._replot_timer.start())

        self.plot_canvas = FigureCanvas(Figure(figsize=(6, 4), tight_layout=True))
        self.plot_canvas.setContentsMargins(0, 0, 0, 0)
        self.ax_alt = self.plot_canvas.figure.subplots()
        self.ax_alt.margins(x=0, y=0)

        # Matplotlib toolbar
        self.plot_toolbar = NavigationToolbar(self.plot_canvas, self)
        # Minimize toolbar padding and height
        self.plot_toolbar.setIconSize(QSize(16, 16))
        self.plot_toolbar.setMaximumHeight(self.plot_toolbar.sizeHint().height())
        self.plot_toolbar.layout().setContentsMargins(0, 0, 0, 0)
        self.plot_toolbar.layout().setSpacing(0)

        # left pane (controls)
        form = QFormLayout()
        # Minimize form padding
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(2)
        form.setVerticalSpacing(2)
        # Observatory dropdown
        form.addRow("Observatory", self.obs_combo)
        # Date selector with previous/next day buttons
        prev_day_btn = QToolButton()
        prev_day_btn.setIcon(self.style().standardIcon(QStyle.SP_ArrowLeft))
        prev_day_btn.setToolTip("Previous day")
        prev_day_btn.clicked.connect(lambda: self._change_date(-1))

        today_btn = QToolButton()
        today_btn.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
        today_btn.setToolTip("Today")
        today_btn.clicked.connect(lambda: self._change_to_today())

        next_day_btn = QToolButton()
        next_day_btn.setIcon(self.style().standardIcon(QStyle.SP_ArrowRight))
        next_day_btn.setToolTip("Next day")
        next_day_btn.clicked.connect(lambda: self._change_date(1))

        date_widget = QWidget()
        date_layout = QHBoxLayout()
        date_layout.setContentsMargins(0, 0, 0, 0)
        date_layout.setSpacing(2)
        date_layout.addWidget(prev_day_btn)
        date_layout.addWidget(today_btn)
        date_layout.addWidget(self.date_edit)
        date_layout.addWidget(next_day_btn)
        date_widget.setLayout(date_layout)

        form.addRow("Date", date_widget)
        form.addRow("Latitude", self.lat_edit)
        form.addRow("Longitude", self.lon_edit)
        form.addRow("Elevation (m)", self.elev_edit)
        form.addRow("Lim. Altitude (°)", self.limit_spin)

        # Sun/Moon visibility toggles
        self.sun_check = QCheckBox("Show Sun")
        self.sun_check.setChecked(False)
        form.addRow("", self.sun_check)
        self.sun_check.stateChanged.connect(self._toggle_visibility)

        self.moon_check = QCheckBox("Show Moon")
        self.moon_check.setChecked(True)
        form.addRow("", self.moon_check)
        self.moon_check.stateChanged.connect(self._toggle_visibility)

        add_btn = QPushButton("Add Target…")
        add_btn.setMaximumWidth(100)
        add_btn.setIcon(self.style().standardIcon(QStyle.SP_FileDialogNewFolder))
        add_btn.clicked.connect(self._add_target_dialog)

        load_plan_btn = QPushButton("Load Plan…")
        load_plan_btn.setMaximumWidth(100)
        load_plan_btn.setIcon(self.style().standardIcon(QStyle.SP_DialogOpenButton))
        load_plan_btn.clicked.connect(self._load_plan)


        left = QVBoxLayout()
        # Minimize left pane padding
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(4)
        left.addLayout(form)
        left.addWidget(add_btn)
        left.addWidget(load_plan_btn)
        save_btn = QPushButton("Save Plan…")
        save_btn.setMaximumWidth(100)
        save_btn.setIcon(self.style().standardIcon(QStyle.SP_DialogSaveButton))
        save_btn.clicked.connect(self._export_plan)
        left.addWidget(save_btn)
        left_w = QWidget()
        left_w.setLayout(left)
        # Remove default widget margins
        left_w.layout().setContentsMargins(0, 0, 0, 0)
        # Fix left pane width so controls (date, coords, buttons) stay compact
        left_w.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        left_w.setMaximumWidth(300)

        # right pane (plot + toolbar)
        right = QVBoxLayout()
        # Minimize right pane padding
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(0)
        right.addWidget(self.plot_toolbar)
        right.addWidget(self.plot_canvas)
        right_w = QWidget()
        right_w.setLayout(right)
        # Remove default widget margins
        right_w.layout().setContentsMargins(0, 0, 0, 0)

        # Wrap toolbar + canvas into a single plot widget
        plot_w = QWidget()
        plot_l = QVBoxLayout()
        plot_l.setContentsMargins(0, 0, 0, 0)
        plot_l.setSpacing(0)
        plot_l.addWidget(self.plot_toolbar)
        plot_l.addWidget(self.plot_canvas)
        plot_w.setLayout(plot_l)

        # Ensure table expands in both directions
        self.table_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Two-column row: controls on left, table on right, info panel
        bottom_h = QHBoxLayout()
        bottom_h.setContentsMargins(0, 0, 0, 0)
        bottom_h.setSpacing(4)
        bottom_h.addWidget(left_w)
        bottom_h.addWidget(self.table_view)
        bottom_h.addWidget(self.polar_canvas)
        bottom_h.setStretchFactor(left_w, 0)
        bottom_h.setStretchFactor(self.table_view, 1)

        # Info panel for sun/moon events
        self.info_widget = QWidget()
        info_form = QFormLayout()
        info_form.setContentsMargins(0, 0, 0, 0)
        info_form.setHorizontalSpacing(4)
        info_form.setVerticalSpacing(2)
        # Align labels (text before “:”) to the left
        info_form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.sunrise_label = QLabel("-")
        self.sunset_label = QLabel("-")
        self.moonrise_label = QLabel("-")
        self.moonset_label = QLabel("-")
        self.moonphase_label = QLabel("-")
        # Current time labels
        self.localtime_label = QLabel("-")
        self.utctime_label = QLabel("-")
        # Fonts for info panel: labels bold, values italic
        label_font = QFont(self.font())
        label_font.setPointSize(14)
        label_font.setBold(True)
        value_font = QFont(self.font())
        value_font.setPointSize(14)
        value_font.setItalic(True)
        # Save for dynamic reordering
        self.info_form = info_form
        self.info_label_font = label_font
        self.info_value_font = value_font
        self.info_widget.setLayout(info_form)
        # Static info panel rows in fixed order
        # Local and UTC time display
        lbl = QLabel("🕑 Local time:")
        lbl.setFont(self.info_label_font)
        lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.localtime_label.setFont(self.info_value_font)
        self.localtime_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        info_form.insertRow(0, lbl, self.localtime_label)

        lbl = QLabel("🌐 UTC time:")
        lbl.setFont(self.info_label_font)
        lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.utctime_label.setFont(self.info_value_font)
        self.utctime_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        info_form.insertRow(1, lbl, self.utctime_label)

        # Sidereal time display
        lbl = QLabel("⭐ Sidereal time:")
        lbl.setFont(self.info_label_font)
        lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.sidereal_label = QLabel("-")
        self.sidereal_label.setFont(self.info_value_font)
        self.sidereal_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        info_form.insertRow(2, lbl, self.sidereal_label)

        # Sunset
        lbl = QLabel("🌇 Sunset:")
        lbl.setFont(self.info_label_font)
        lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.sunset_label.setFont(self.info_value_font)
        self.sunset_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        info_form.insertRow(3, lbl, self.sunset_label)

        # Sunrise
        lbl = QLabel("🌅 Sunrise:")
        lbl.setFont(self.info_label_font)
        lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.sunrise_label.setFont(self.info_value_font)
        self.sunrise_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        info_form.insertRow(4, lbl, self.sunrise_label)

        # Moon phase
        lbl = QLabel("🌕 Moon phase:")
        lbl.setFont(self.info_label_font)
        lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.moonphase_label.setFont(self.info_value_font)
        self.moonphase_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        info_form.insertRow(5, lbl, self.moonphase_label)

        # Moonrise
        lbl = QLabel("🌙 Moonrise:")
        lbl.setFont(self.info_label_font)
        lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.moonrise_label.setFont(self.info_value_font)
        self.moonrise_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        info_form.insertRow(6, lbl, self.moonrise_label)

        # Moonset
        lbl = QLabel("🌙 Moonset:")
        lbl.setFont(self.info_label_font)
        lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.moonset_label.setFont(self.info_value_font)
        self.moonset_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        info_form.insertRow(7, lbl, self.moonset_label)

        # Current sun altitude
        self.sun_alt_label = QLabel("-")
        self.sun_alt_label.setFont(self.info_value_font)
        self.sun_alt_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        lbl = QLabel("☀️ Sun alt:")
        lbl.setFont(self.info_label_font)
        lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        info_form.insertRow(8, lbl, self.sun_alt_label)

        # Current moon altitude
        self.moon_alt_label = QLabel("-")
        self.moon_alt_label.setFont(self.info_value_font)
        self.moon_alt_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        lbl = QLabel("🌙 Moon alt:")
        lbl.setFont(self.info_label_font)
        lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        info_form.insertRow(9, lbl, self.moon_alt_label)

        # Add right margin to info panel to prevent text clipping
        self.info_widget.setContentsMargins(0, 0, 10, 0)
        # Ensure info panel labels use the larger fonts
        self.info_widget.setStyleSheet("QLabel { font-size: 14pt; }")
        # Add it to the bottom row on the right
        bottom_h.addWidget(self.info_widget)

        # Main vertical layout: plot on top, bottom row beneath
        main_l = QVBoxLayout()
        main_l.setContentsMargins(0, 0, 0, 0)
        main_l.setSpacing(4)
        main_l.addWidget(plot_w, stretch=3)
        main_l.addLayout(bottom_h, stretch=1)

        container = QWidget()
        container.setLayout(main_l)
        self.setCentralWidget(container)

        # Apply custom GUI styles
        self._apply_styles()

        # Progress indicator for calculations
        self.progress = QProgressDialog("Calculating visibility...", "", 0, 0, self)
        self.progress.setWindowTitle("Please wait")
        self.progress.setWindowModality(Qt.WindowModal)
        self.progress.setCancelButton(None)
        self.progress.setAutoClose(False)
        self.progress.setAutoReset(False)
        self.progress.hide()

        # Start real-time clock updates for time labels
        self.clock_worker = None
        if self.table_model.site:
            self._start_clock_worker()
        self._clock_timer = QTimer(self)
        self._clock_timer.timeout.connect(self._update_clock)
        self._clock_timer.start(1000)
    def _start_clock_worker(self):
        if self.clock_worker:
            self.clock_worker.stop()
        site = self.table_model.site
        if site is None:
            # Provide a default site if none is set
            site = Site(name="Default", latitude=0.0, longitude=0.0, elevation=0.0)
        self.clock_worker = ClockWorker(site, self.targets, self)
        self.clock_worker.updated.connect(self._handle_clock_update)
        self.clock_worker.start()

        # actions / shortcuts
        self._build_actions()

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

        self.exp_act = QAction("Export plan…", self, shortcut=QKeySequence("Ctrl+E"))
        self.exp_act.triggered.connect(self._export_plan)

        self.dark_act = QAction("Toggle dark mode", self, shortcut=QKeySequence("Ctrl+D"))
        self.dark_act.triggered.connect(self._toggle_dark)

        # Make shortcuts work even if the action isn't in a visible menu
        for act in (self.load_act, self.load_plan_act, self.add_act, self.exp_act, self.dark_act):
            self.addAction(act)

        # ----- Menu bar -----
        menubar = self.menuBar()
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

        # Settings menu
        settings_menu = menubar.addMenu("&Settings")
        gen_act = QAction("General Settings…", self)
        gen_act.triggered.connect(self.open_general_settings)
        settings_menu.addAction(gen_act)
        tbl_act = QAction("Table Settings…", self)
        tbl_act.triggered.connect(self.open_table_settings)
        settings_menu.addAction(tbl_act)

    def _load_settings(self):
        """Load saved settings and apply both Table and General."""
        self._apply_table_settings()
        self._apply_general_settings()

    def _apply_table_settings(self):
        """Apply table row height and first-column width."""
        row_h = self.settings.value("table/rowHeight", 24, type=int)
        self.table_view.verticalHeader().setDefaultSectionSize(row_h)
        col_w = self.settings.value("table/firstColumnWidth", 100, type=int)
        self.table_view.setColumnWidth(0, col_w)
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
        default_colors = {"below":"#ff8080","limit":"#ffff80","above":"#b3ffb3"}
        self.table_model.highlight_colors = {
            k: QColor(self.settings.value(f"table/color/{k}", default_colors[k]))
            for k in default_colors
        }

    def open_table_settings(self):
        """Open the Table Settings dialog."""
        dlg = TableSettingsDialog(self)
        dlg.exec()

    def _apply_general_settings(self):
        """Apply default site."""
        s = self.settings
        ds = s.value("general/defaultSite", type=str)
        if ds in self.observatories:
            self.obs_combo.setCurrentText(ds)

    def open_general_settings(self):
        dlg = GeneralSettingsDialog(self)
        dlg.exec()

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
            self.targets.clear()
            for entry in data:
                self.targets.append(Target(**entry))
            self.table_model.layoutChanged.emit()
            self._refresh_table_buttons()
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
            self.targets.clear()
            with open(fn, newline="", encoding="utf-8") as fh:
                dialect = csv.Sniffer().sniff(fh.read(2048))
                fh.seek(0)
                rdr = csv.DictReader(fh, dialect=dialect)
                for row in rdr:
                    self.targets.append(
                        Target(
                            name=row["name"],
                            ra=self._parse_angle(row["ra"]),
                            dec=self._parse_angle(row["dec"]),
                        )
                    )
        except (FileNotFoundError, KeyError, ValueError, ValidationError, csv.Error) as exc:
            QMessageBox.critical(self, "Load error", str(exc))
            return

        # Refresh table view after successful load
        self.table_model.layoutChanged.emit()
        self._refresh_table_buttons()

    @Slot()
    def _add_target_dialog(self):
        """Prompt user for a target name or coordinate string and append it."""
        text, ok = QInputDialog.getText(
            self, "Add target", "Name or RA Dec (sexagesimal or decimal):"
        )
        if not ok or not text.strip():
            return
        try:
            target = self._resolve_target(text.strip())
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Resolve error", str(exc))
            return
        self.targets.append(target)
        self.table_model.layoutChanged.emit()
        self._refresh_table_buttons()
        # Debounce and update plot after adding a target
        self._replot_timer.start()

    @Slot()
    def _run_plan(self):
        """Kick off the worker thread unless one is already running."""
        if self.worker and self.worker.isRunning():
            # Stop existing calculation and start a new one
            self.worker.quit()
            self.worker.wait()
        try:
            site = Site(
                latitude=float(self.lat_edit.text()),
                longitude=float(self.lon_edit.text()),
                elevation=float(self.elev_edit.text()),
            )
            settings = SessionSettings(
                date=self.date_edit.date(),
                site=site,
                limit_altitude=float(self.limit_spin.value()),
                time_samples=self.settings.value("general/timeSamples", 240, type=int),
            )
        except ValidationError as exc:
            QMessageBox.critical(self, "Invalid input", str(exc))
            return

        self.table_model.site = site
        # Update the observation limit for table coloring
        self.table_model.limit = settings.limit_altitude
        self.table_model.layoutChanged.emit()
        self._refresh_table_buttons()

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
        del self.targets[row]
        self.table_model.layoutChanged.emit()
        self._refresh_table_buttons()
        # Debounce and update plot after removing a target
        self._replot_timer.start()

    def _refresh_table_buttons(self):
        # Place a Remove button in the 'Actions' column for each row
        for row in range(self.table_model.rowCount()):
            idx = self.table_model.index(row, 7)
            btn = QToolButton()
            btn.setIcon(self.style().standardIcon(QStyle.SP_TrashIcon))
            btn.setToolTip("Remove")
            btn.setAutoRaise(True)
            btn.clicked.connect(lambda _, r=row: self._remove_target(r))
            self.table_view.setIndexWidget(idx, btn)

    @Slot(dict)
    def _update_plot(self, data: dict):
        """Redraw the altitude plot with new data from the worker."""
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
        # Generate a distinct color for each target using the tab20 colormap
        num_targets = len(self.targets)
        cmap = plt.get_cmap('tab20', num_targets)
        colors = [mcolors.to_hex(cmap(i)) for i in range(num_targets)]
        # Update table model color_map so table colors align with plot lines
        self.table_model.color_map.clear()
        for idx, tgt in enumerate(self.targets):
            color_css = colors[idx % len(colors)]
            self.table_model.color_map[tgt.name] = QColor(color_css)
        for idx, tgt in enumerate(self.targets):
            alt = np.array(data[tgt.name]["altitude"])
            times_nums = mdates.date2num(times)
            color = colors[idx % len(colors)]
            limit = self.limit_spin.value()

            # Points above horizon
            vis_mask = alt > 0
            if not vis_mask.any():
                continue
            # Dashed base path for full visible range
            times_vis = times_nums[vis_mask]
            alt_vis = alt[vis_mask]
            base_line, = self.ax_alt.plot(
                times_vis, alt_vis,
                color=color, linewidth=1.4,
                linestyle="--", alpha=0.3, zorder=1
            )
            self.vis_lines.append((tgt.name, base_line, False))
            # Solid overlay for portions above limit
            high_mask = alt >= limit
            if high_mask.any():
                times_high = times_nums[high_mask]
                alt_high = alt[high_mask]
                solid_line, = self.ax_alt.plot(
                    times_high, alt_high,
                    color=color, linewidth=1.4,
                    linestyle="-", alpha=1.0, zorder=2
                )
                self.vis_lines.append((tgt.name, solid_line, True))

        # ------------------------------------------------------------------
        # Compute and cache current alt, az, sep for each target for the table
        # Prepare timezone and observer only once
        tz_name = data.get("tz", "UTC")
        site = Site(
            name="",
            latitude=float(self.lat_edit.text()),
            longitude=float(self.lon_edit.text()),
            elevation=float(self.elev_edit.text()),
        )
        observer_now = Observer(location=site.to_earthlocation(), timezone=tz_name)
        now_dt = datetime.now(pytz.timezone(tz_name))
        # Prepare ephem observer for separation
        eph_obs = ephem.Observer()
        eph_obs.lat = str(float(self.lat_edit.text()))
        eph_obs.lon = str(float(self.lon_edit.text()))
        eph_obs.elevation = float(self.elev_edit.text())
        eph_obs.date = now_dt

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
            moon = ephem.Moon(eph_obs)
            moon_coord = SkyCoord(ra=Angle(moon.ra, u.rad), dec=Angle(moon.dec, u.rad))
            current_seps.append(float(tgt.skycoord.separation(moon_coord).deg))                 # type: ignore[arg-type]

        # Assign to model and refresh table
        self.table_model.current_alts = current_alts
        self.table_model.current_azs = current_azs
        self.table_model.current_seps = current_seps
        self.table_model.layoutChanged.emit()

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

        # Center plot on middle of night (DST-aware)
        if "sunset" in ev and "sunrise" in ev:
            center_dt = ev["sunset"] + (ev["sunrise"] - ev["sunset"]) / 2
        else:
            center_dt = mdates.num2date(data["midnight"]).astimezone(tz)
        self.ax_alt.set_xlim(
            center_dt - timedelta(hours=12),
            center_dt + timedelta(hours=12),
        )
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
        self.ax_alt.axhline(self.limit_spin.value(), color="red", linestyle="-", linewidth=0.5, alpha=0.4, label="Limit Altitude")

        # Reset line references
        self.sun_line = None
        self.moon_line = None

        # Sun altitude curve (always plot, visibility controlled)
        if "sun_alt" in data:
            self.sun_line, = self.ax_alt.plot(
                times, data["sun_alt"],
                color="orange", linewidth=1.2, linestyle='-',
                alpha=0.8, label="Sun"
            )
            self.sun_line.set_visible(self.sun_check.isChecked())

        # Moon altitude curve (always plot, visibility controlled)
        if "moon_alt" in data:
            self.moon_line, = self.ax_alt.plot(
                times, data["moon_alt"],
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
        phase_pct = data.get("moon_phase", 0.0)
        self.moonphase_label.setText(f"{phase_pct:.0f}%")

        self.ax_alt.set_ylabel("Altitude (°)")
        self.ax_alt.set_xlabel("Time (local)")
        # self.ax_alt.legend(loc="upper right")
        self.ax_alt.set_ylim(0, 90)
        # Hour labels in the observer's local timezone
        self.ax_alt.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=tz))
        # Display selected observation date
        date_str = self.date_edit.date().toString("yyyy-MM-dd")
        self.ax_alt.set_title(f"Date: {date_str}")
        # Current time indicator
        now = datetime.now(tz)
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
        self.plot_canvas.draw_idle()

    @Slot()
    def _toggle_visibility(self):
        """Show or hide sun and moon lines without recalculation."""
        if hasattr(self, 'sun_line') and self.sun_line:
            self.sun_line.set_visible(self.sun_check.isChecked())
        if hasattr(self, 'moon_line') and self.moon_line:
            self.moon_line.set_visible(self.moon_check.isChecked())
        self.plot_canvas.draw_idle()


    @Slot()
    def _update_clock(self):
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
        self.table_model.layoutChanged.emit()

        # Update sidereal time based on local time and site longitude
        if hasattr(self, 'last_payload') and self.last_payload:
            tz = pytz.timezone(self.last_payload.get("tz", "UTC"))
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
            self._update_polar_positions(data)


    @Slot()
    def _update_polar_selection(self, selected, deselected):
        """Update highlight for selected targets on polar plot."""
        # Gather selected rows
        sel_rows = [idx.row() for idx in self.table_view.selectionModel().selectedRows()]
        # Prepare coordinates for selected targets
        sel_coords = []
        for i, tgt in enumerate(self.targets):
            if i in sel_rows:
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
            alt_arr = np.array(self.full_payload[name]["altitude"])
            az_arr  = np.array(self.full_payload[name]["azimuth"])
            # Only points above horizon
            mask = alt_arr > 0
            vis_idx = np.where(mask)[0]
            if vis_idx.size == 0:
                self.selected_trace_line = None
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
    def _update_polar_positions(self, data):
        """Update all markers on the polar plot based on latest alt-az data."""
        # Build coordinate list and track corresponding target indices
        tgt_coords: list[tuple[float, float]] = []
        self.polar_indices = []
        for i, tgt in enumerate(self.targets):
            alt = data.get('alts', [])[i] if i < len(data.get('alts', [])) else None
            az = data.get('azs', [])[i] if i < len(data.get('azs', [])) else None
            if alt is not None and az is not None and alt > 0:
                theta = np.deg2rad(az)
                r = 90 - alt
                tgt_coords.append((theta, r))
                self.polar_indices.append(i)
        if tgt_coords:
            arr = np.array(tgt_coords)
            self.polar_scatter.set_offsets(arr)
        else:
            self.polar_scatter.set_offsets(np.empty((0, 2)))

        # Sun position (via PyEphem)
        eph_obs = ephem.Observer()
        site = self.table_model.site
        if site is not None:
            eph_obs.lat = str(site.latitude)
            eph_obs.lon = str(site.longitude)
            eph_obs.elevation = site.elevation
            eph_obs.date = data["now_local"]
            sun = ephem.Sun(eph_obs)
            sun_alt = sun.alt * 180.0 / math.pi  # type: ignore[arg-type]
            sun_az = sun.az * 180.0 / math.pi    # type: ignore[arg-type]
            if sun_alt > 0:
                theta_sun = np.deg2rad(sun_az)
                r_sun = 90 - sun_alt
                self.sun_marker.set_offsets(np.array([[theta_sun, r_sun]]))
            else:
                self.sun_marker.set_offsets(np.empty((0, 2)))

            # Moon position (via PyEphem)
            eph_obs.date = data["now_local"]
            moon = ephem.Moon(eph_obs)
            moon_alt = moon.alt * 180.0 / math.pi  # type: ignore[arg-type]
            moon_az = moon.az * 180.0 / math.pi    # type: ignore[arg-type]
            if moon_alt > 0:
                theta_moon = np.deg2rad(moon_az)
                r_moon = 90 - moon_alt
                self.moon_marker.set_offsets(np.array([[theta_moon, r_moon]]))
            else:
                self.moon_marker.set_offsets(np.empty((0, 2)))
            # Plot moon path on the polar plot
            if hasattr(self, 'moon_path_line'):
                try:
                    self.moon_path_line.remove()
                except Exception:
                    pass
            if hasattr(self, 'full_payload') and 'moon_alt' in self.full_payload and 'moon_az' in self.full_payload:
                alt_arr = np.array(self.full_payload['moon_alt'])
                az_arr = np.array(self.full_payload['moon_az'])
                mask = alt_arr > 0
                # Build continuous segments without wrapping artifacts
                vis_indices = np.where(mask)[0]
                theta_full = np.deg2rad(az_arr[mask])
                r_full = 90 - alt_arr[mask]
                # Insert NaNs at wrap discontinuities greater than π
                dtheta = np.abs(np.diff(theta_full))
                wrap_points = np.where(dtheta > np.pi)[0] + 1
                for wp in reversed(wrap_points):
                    theta_full = np.insert(theta_full, wp, np.nan)
                    r_full = np.insert(r_full, wp, np.nan)
                # Plot moon path on polar plot without straight wrap lines
                self.moon_path_line, = self.polar_ax.plot(
                    theta_full, r_full,
                    color='silver', linestyle=':', linewidth=0.8, alpha=0.7, zorder=1
                )
        else:
            self.sun_marker.set_offsets(np.empty((0, 2)))
            self.moon_marker.set_offsets(np.empty((0, 2)))

        # Draw celestial pole marker (north or south pole if above horizon)
        if self.pole_marker:
            try:
                # Remove previous marker(s)
                if isinstance(self.pole_marker, (list, tuple)):
                    for art in self.pole_marker:
                        art.remove()
                else:
                    self.pole_marker.remove()
            except Exception:
                pass
        site = self.table_model.site
        if site:
            lat = site.latitude
            if lat >= 0:
                pole_alt = lat
                pole_az = 0.0
            else:
                pole_alt = -lat
                pole_az = 180.0
            r_pol = 90 - pole_alt
            theta_pol = np.deg2rad(pole_az)
            # Plot a purple circle with a dot inside for the celestial pole
            circle = self.polar_ax.scatter(
                [theta_pol], [r_pol],
                facecolors='none', edgecolors='purple',
                marker='o', s=80, linewidths=1.5, zorder=3, alpha=0.3
            )
            dot = self.polar_ax.scatter(
                [theta_pol], [r_pol],
                c='purple', marker='.', s=30, zorder=4, alpha=0.3
            )
            self.pole_marker = (circle, dot)
        else:
            self.pole_marker = None

        # Draw or update altitude limit circle
        if self.limit_circle:
            try:
                self.limit_circle.remove()
            except Exception:
                pass
        lim = self.limit_spin.value()
        r_lim = 90 - lim
        theta_full = np.linspace(0, 2 * math.pi, 200)
        r_full = np.full_like(theta_full, r_lim)
        circle_line, = self.polar_ax.plot(
            theta_full, r_full,
            color='red', linestyle='-', linewidth=0.5, alpha=0.4
        )
        self.limit_circle = circle_line

        self.polar_canvas.draw_idle()

    @Slot()
    def _toggle_dark(self):
        """Toggle a very basic dark stylesheet."""
        dark_qss = "QWidget { background:#2e2e2e; color:#e0e0e0; }"
        self.setStyleSheet("" if self.styleSheet() else dark_qss)

    @Slot()
    def _export_plan(self):
        """Write targets JSON and plot PNG to a directory the user chooses."""
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
        QMessageBox.information(self, "Export complete", f"Wrote files to {out_path}")

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _parse_angle(self, text: str) -> float:
        """Interpret sexagesimal or decimal RA/Dec and return degrees."""
        t = text.strip()
        if any(c in t for c in " :"):
            # Sexagesimal
            try:
                coord = SkyCoord(t, unit=(u.hourangle, u.deg))
            except ValueError:
                coord = SkyCoord(t, unit=(u.deg, u.deg))
            return coord.ra.deg if "h" in t.lower() else coord.dec.deg                  # type: ignore[arg-type]
        return float(t)

    def _resolve_target(self, query: str) -> Target:
        """Resolve an object by name via Simbad/Sesame, or parse an RA-Dec string.

        Resolution order:
        1. Astroquery Simbad (fast and explicit)
        2. Astropy SkyCoord.from_name (Sesame)
        3. Coordinate string fallback
        """

        # --- 1) Try astroquery Simbad first --------------------------------
        try:
            custom = Simbad()
            # 'ra' and 'dec' now return decimal degrees, superseding the old 'ra(d)' / 'dec(d)'
            custom.add_votable_fields("ra", "dec")
            result = custom.query_object(query)
            if result is not None:
                ra_deg = float(result["ra"][0])
                dec_deg = float(result["dec"][0])
                # Use Simbad’s official main identifier as the target name
                raw_name = result["MAIN_ID"][0]
                name_res = (
                    raw_name.decode("utf-8")
                    if isinstance(raw_name, (bytes, bytearray))
                    else str(raw_name)
                )
                return Target(name=name_res, ra=ra_deg, dec=dec_deg)
        except Exception as e:
            print(f"Failed to resolve '{query}' with Simbad. Falling back…")
            print(f"Error: {e}")  # noqa: T201
        
        # --- 2) Try Astropy's built‑in Sesame resolver ----------------------
        try:
            coord = SkyCoord.from_name(query)
            return Target.from_skycoord(query, coord)
        except Exception as e:  # noqa: BLE001
            print(f"Failed to resolve '{query}' with Astropy. Falling back…")
            print(f"Error: {e}")  # noqa: T201

        # --- 3) Treat the input as an RA/Dec string -------------------------
        parts = query.replace(",", " ").split()
        if len(parts) < 2:
            raise ValueError("Unrecognised name and no RA/Dec pair provided.")

        ra_str = " ".join(parts[: len(parts) // 2])
        dec_str = " ".join(parts[len(parts) // 2 :])
        ra_deg = self._parse_angle(ra_str)
        dec_deg = self._parse_angle(dec_str)
        return Target(name=query, ra=ra_deg, dec=dec_deg)

    # Ensure thread finishes before app closes
    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.quit()
            self.worker.wait()
        if self.clock_worker:
            self.clock_worker.stop()
        super().closeEvent(event)

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
        if now_local.hour < 12:
            new_date = QDate.currentDate().addDays(-1)
        else:
            new_date = QDate.currentDate()
        self.date_edit.setDate(new_date)
        self._replot_timer.start()

    @Slot()
    def _update_limit(self):
        """Update limit altitude, refresh table warnings, and replot."""
        # Update table model limit so coloring reflects the new threshold
        new_limit = float(self.limit_spin.value())
        self.table_model.limit = new_limit
        self.table_model.layoutChanged.emit()
        # Replot visibility with updated limit
        if self.last_payload is not None:
            self._update_plot(self.last_payload)
            # Update only the polar limit circle when limit altitude changes
            lim = self.limit_spin.value()
            r_lim = 90 - lim
            # Remove old limit circle
            if self.limit_circle:
                try:
                    self.limit_circle.remove()
                except Exception:
                    pass
            theta_full = np.linspace(0, 2 * math.pi, 200)
            r_full = np.full_like(theta_full, r_lim)
            circle_line, = self.polar_ax.plot(
                theta_full, r_full,
                color='red', linestyle='-', linewidth=0.5, alpha=0.4
            )
            self.limit_circle = circle_line

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
            for entry in data:
                win.targets.append(Target(**entry))
            win.table_model.layoutChanged.emit()
            win._refresh_table_buttons()
            # Now run the plot (also sets the site and refreshes buttons)
            win._run_plan()
        except Exception as e:
            QMessageBox.critical(None, "Startup Load Error", f"Failed to load plan '{args.plan}': {e}")

    win.show()
sys.exit(app.exec())
