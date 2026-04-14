from __future__ import annotations

from astropy.utils import iers
from PySide6.QtCore import QDate
from PySide6.QtWidgets import QApplication

from astroplanner.astronomy import AstronomyWorker
from astroplanner.models import SessionSettings, Site, Target


iers.conf.auto_download = False


def test_astronomy_worker_run_emits_payload_for_single_target() -> None:
    app = QApplication.instance() or QApplication([])
    assert app is not None

    site = Site(name="WRO", latitude=51.1, longitude=17.0, elevation=120.0)
    settings = SessionSettings(
        date=QDate(2026, 4, 14),
        site=site,
        limit_altitude=20.0,
        time_samples=24,
    )
    worker = AstronomyWorker(
        [Target(name="M31", ra=10.684, dec=41.269)],
        settings,
    )

    payloads: list[dict] = []
    worker.finished.connect(payloads.append)
    worker.run()

    assert payloads
    payload = payloads[-1]
    assert payload["tz"] == site.timezone_name
    assert "M31" in payload
    assert payload["site_key"].startswith("WRO|")
    assert len(payload["times"]) == 24
    assert len(payload["M31"]["altitude"]) == 24
    assert len(payload["M31"]["moon_sep"]) == 24
