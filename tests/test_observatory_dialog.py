from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from astroplanner.models import Site
from astroplanner.ui.observatory import AddObservatoryDialog, AddObservatoryValidationError


def _app() -> QApplication:
    app = QApplication.instance() or QApplication([])
    assert app is not None
    return app


def test_add_observatory_dialog_builds_site_with_template_metadata() -> None:
    _app()
    template = Site(
        name="Template",
        latitude=51.1,
        longitude=17.0,
        elevation=120.0,
        limiting_magnitude=19.5,
        custom_conditions_url="",
        telescope_diameter_mm=80.0,
        focal_length_mm=400.0,
        pixel_size_um=2.9,
        detector_width_px=1920,
        detector_height_px=1080,
    )
    dialog = AddObservatoryDialog(
        existing_names={"Template"},
        template_site=template,
        default_limiting_magnitude=18.2,
    )

    dialog.name_edit.setText("Backyard")
    dialog.lat_edit.setText("50.061")
    dialog.lon_edit.setText("19.938")
    dialog.elev_edit.setText("220")
    dialog.limiting_mag_spin.setValue(20.1)
    dialog.custom_conditions_url_edit.setText(" https://example.com/station.json ")

    site = dialog.build_site()

    assert site.name == "Backyard"
    assert site.latitude == pytest.approx(50.061)
    assert site.longitude == pytest.approx(19.938)
    assert site.elevation == pytest.approx(220.0)
    assert site.limiting_magnitude == pytest.approx(20.1)
    assert site.custom_conditions_url == "https://example.com/station.json"
    assert site.telescope_diameter_mm == pytest.approx(80.0)
    assert site.focal_length_mm == pytest.approx(400.0)
    assert site.pixel_size_um == pytest.approx(2.9)
    assert site.detector_width_px == 1920
    assert site.detector_height_px == 1080


def test_add_observatory_dialog_rejects_empty_and_duplicate_names() -> None:
    _app()
    dialog = AddObservatoryDialog(
        existing_names={"WRO"},
        template_site=None,
        default_limiting_magnitude=19.0,
    )

    with pytest.raises(AddObservatoryValidationError) as empty_exc:
        dialog.build_site()

    assert empty_exc.value.title == "Invalid Observatory"
    assert empty_exc.value.message == "Name cannot be empty."

    dialog.name_edit.setText("WRO")

    with pytest.raises(AddObservatoryValidationError) as duplicate_exc:
        dialog.build_site()

    assert duplicate_exc.value.title == "Invalid Observatory"
    assert duplicate_exc.value.message == "Observatory 'WRO' already exists."


def test_add_observatory_dialog_lookup_fills_coordinates() -> None:
    _app()

    def resolver(query: str) -> tuple[float, float, float, str]:
        assert query == "Krakow"
        return 50.06143, 19.93658, 219.0, "Krakow, Poland"

    dialog = AddObservatoryDialog(
        existing_names=set(),
        template_site=None,
        default_limiting_magnitude=19.0,
        lookup_resolver=resolver,
    )
    dialog.name_edit.setText("Krakow")

    dialog._lookup_coords()

    assert dialog.lat_edit.text() == "50.061430"
    assert dialog.lon_edit.text() == "19.936580"
    assert dialog.elev_edit.text() == "219.0"
    assert dialog.lookup_info.text() == "Krakow, Poland"
