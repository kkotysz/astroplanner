from __future__ import annotations

import logging
import math
import warnings
from datetime import datetime, timedelta, timezone

import ephem
import numpy as np
import pytz
from astropy import units as u
from astropy.coordinates import Angle, SkyCoord
from astropy.time import Time
from astroplan import FixedTarget, Observer
from astroplan.observer import TargetAlwaysUpWarning
from PySide6.QtCore import QMutex, QThread, QWaitCondition, Signal

from astroplanner.models import SessionSettings, Site, Target


logger = logging.getLogger(__name__)


class ClockWorker(QThread):
    updated = Signal(dict)

    def __init__(self, site: Site, targets: list[Target], parent=None):
        super().__init__(parent)
        self.setObjectName(self.__class__.__name__)
        self.site = site
        self.targets = targets
        self.running = True
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

            current_alts: list[float] = []
            current_azs: list[float] = []
            current_seps: list[float] = []
            for tgt in self.targets:
                fixed = FixedTarget(name=tgt.name, coord=tgt.skycoord)
                altaz = obs.altaz(Time(now_local), fixed)
                current_alts.append(float(altaz.alt.deg))  # type: ignore[arg-type]
                current_azs.append(float(altaz.az.deg))  # type: ignore[arg-type]
                sep_deg = tgt.skycoord.separation(moon_coord).deg
                current_seps.append(float(np.real(sep_deg)))

            self.updated.emit(
                {
                    "now_local": now_local,
                    "now_utc": now_utc,
                    "sun_alt": sun_alt,
                    "moon_alt": moon_alt,
                    "alts": current_alts,
                    "azs": current_azs,
                    "seps": current_seps,
                }
            )

            self._mutex.lock()
            self._wait_cond.wait(self._mutex, 1000)
            self._mutex.unlock()

        logger.info("ClockWorker exiting for site %s", self.site.name)

    def request_stop(self):
        self.running = False
        self._wait_cond.wakeAll()
        self.quit()

    def stop(self):
        logger.info("ClockWorker stop requested for site %s", self.site.name)
        self.request_stop()
        self.wait()


class AstronomyWorker(QThread):
    """Runs astroplan calculations off the GUI thread."""

    finished: Signal = Signal(dict)
    aborted: Signal = Signal()

    _cache: dict = {}

    def __init__(self, targets: list[Target], settings: SessionSettings, parent=None):
        super().__init__(parent)
        self.setObjectName(self.__class__.__name__)
        self.targets = targets
        self.settings = settings

    def run(self) -> None:  # noqa: D401
        obs_date = self.settings.date
        site = self.settings.site
        site_key = (
            f"{site.name}|{site.latitude:.6f}|{site.longitude:.6f}|{site.elevation:.1f}|"
            f"{obs_date.toString('yyyy-MM-dd')}|{self.settings.time_samples}|{self.settings.limit_altitude:.1f}"
        )
        observer = Observer(location=site.to_earthlocation(), timezone=site.timezone_name)

        key = (
            site.latitude,
            site.longitude,
            site.elevation,
            obs_date.toString("yyyy-MM-dd"),
            self.settings.time_samples,
        )
        cache = AstronomyWorker._cache

        tz = pytz.timezone(site.timezone_name)
        next_mid = datetime(obs_date.year(), obs_date.month(), obs_date.day(), 0, 0) + timedelta(days=1)
        local_mid_dt = tz.localize(next_mid)
        midnight = Time(local_mid_dt)

        if self.isInterruptionRequested():
            self.aborted.emit()
            return

        if key in cache and "moon_ra" in cache[key]["events"] and "moon_dec" in cache[key]["events"]:
            cached = cache[key]
            times = cached["times"]
            jd = times.plot_date
            events = cached["events"]
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=TargetAlwaysUpWarning)
                try:
                    dusk = observer.twilight_evening_astronomical(midnight, which="nearest")
                    dawn = observer.twilight_morning_astronomical(midnight, which="next")
                    astro_ok = True
                except (TargetAlwaysUpWarning, Exception):
                    astro_ok = False

            ts = self.settings.time_samples
            times = midnight + np.linspace(-12, 12, ts) * u.hour
            jd = times.plot_date

            dusk_naut = observer.twilight_evening_nautical(midnight, which="nearest")
            dawn_naut = observer.twilight_morning_nautical(midnight, which="next")

            events = {"times": jd}
            if astro_ok:
                events["dusk"] = dusk.plot_date
                events["dawn"] = dawn.plot_date
            events.update(
                {
                    "dusk_naut": dusk_naut.plot_date,
                    "dawn_naut": dawn_naut.plot_date,
                    "dusk_civ": observer.twilight_evening_civil(midnight, which="nearest").plot_date,
                    "dawn_civ": observer.twilight_morning_civil(midnight, which="next").plot_date,
                    "sunset": observer.sun_set_time(midnight, which="nearest").plot_date,
                    "sunrise": observer.sun_rise_time(midnight, which="next").plot_date,
                    "moonrise": observer.moon_rise_time(midnight, which="nearest").plot_date,
                    "moonset": observer.moon_set_time(midnight, which="next").plot_date,
                    "midnight": midnight.plot_date,
                }
            )

            eph_observer = ephem.Observer()
            eph_observer.lat = str(site.latitude)
            eph_observer.lon = str(site.longitude)
            eph_observer.elevation = site.elevation
            sun = ephem.Sun()
            moon = ephem.Moon()

            sun_alts: list[float] = []
            sun_azs: list[float] = []
            moon_alts: list[float] = []
            moon_azs: list[float] = []
            moon_ras: list[float] = []
            moon_decs: list[float] = []
            for t in times.datetime:
                if self.isInterruptionRequested():
                    self.aborted.emit()
                    return
                eph_observer.date = t
                sun.compute(eph_observer)
                moon.compute(eph_observer)
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
            events["moon_phase"] = moon.phase
            cache[key] = {"times": times, "events": events}

        payload: dict[str, object] = {k: v for k, v in {"times": jd, **events}.items()}
        payload["site_key"] = site_key
        moon_coords = SkyCoord(
            ra=np.array(events["moon_ra"]) * u.deg,
            dec=np.array(events["moon_dec"]) * u.deg,
        )
        target_count = len(self.targets)
        sample_count = len(times)

        def _normalize_target_time_grid(values: object) -> np.ndarray:
            arr = np.array(values, dtype=float)
            if arr.ndim == 1:
                if target_count == 1 and arr.shape[0] == sample_count:
                    return arr.reshape(1, sample_count)
                raise ValueError(f"Unexpected 1D visibility grid shape: {arr.shape}")
            if arr.shape == (target_count, sample_count):
                return arr
            if arr.shape == (sample_count, target_count):
                return arr.T
            raise ValueError(f"Unexpected visibility grid shape: {arr.shape}")

        if target_count > 0:
            try:
                target_coords = SkyCoord(
                    ra=np.array([float(t.ra) for t in self.targets], dtype=float) * u.deg,
                    dec=np.array([float(t.dec) for t in self.targets], dtype=float) * u.deg,
                )
                altaz_grid = observer.altaz(times, target_coords, grid_times_targets=True)
                alt_grid = _normalize_target_time_grid(altaz_grid.alt.deg)
                az_grid = _normalize_target_time_grid(altaz_grid.az.deg)
                moon_sep_grid = _normalize_target_time_grid(
                    target_coords[:, np.newaxis].separation(moon_coords[np.newaxis, :]).deg
                )
                for idx, tgt in enumerate(self.targets):
                    if self.isInterruptionRequested():
                        self.aborted.emit()
                        return
                    payload[tgt.name] = {
                        "altitude": alt_grid[idx],
                        "azimuth": az_grid[idx],
                        "moon_sep": moon_sep_grid[idx],
                    }
            except Exception:
                for tgt in self.targets:
                    if self.isInterruptionRequested():
                        self.aborted.emit()
                        return
                    fixed = FixedTarget(name=tgt.name, coord=tgt.skycoord)
                    altaz = observer.altaz(times, fixed)
                    moon_sep = tgt.skycoord.separation(moon_coords).deg
                    payload[tgt.name] = {
                        "altitude": altaz.alt.deg,  # type: ignore[arg-type]
                        "azimuth": altaz.az.deg,  # type: ignore[arg-type]
                        "moon_sep": moon_sep,  # type: ignore[arg-type]
                    }

        payload["tz"] = site.timezone_name
        logger.info(
            "AstronomyWorker finished (%d targets, date %s)",
            len(self.targets),
            obs_date.toString("yyyy-MM-dd"),
        )
        self.finished.emit(payload)


__all__ = ["AstronomyWorker", "ClockWorker"]
