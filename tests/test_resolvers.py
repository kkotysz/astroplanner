from __future__ import annotations

import pytest
from astropy.table import Table

from astroplanner.resolvers import (
    TargetResolver,
    _extract_simbad_metadata,
    _extract_simbad_name,
    _normalize_tns_endpoint_key,
    _simbad_best_row_index,
    _simbad_row_coord,
)


def test_simbad_metadata_helpers_parse_basic_payload() -> None:
    result = Table(
        rows=[
            ("M 31", "00 42 44.3", "+41 16 09", 3.44, "Galaxy"),
        ],
        names=("MAIN_ID", "RA", "DEC", "V", "OTYPE"),
    )

    magnitude, object_type = _extract_simbad_metadata(result)

    assert magnitude == pytest.approx(3.44)
    assert object_type == "Galaxy"
    assert _extract_simbad_name(result, "fallback") == "M 31"


def test_simbad_best_row_index_uses_reference_coordinate() -> None:
    result = Table(
        rows=[
            ("far", 20.0, 20.0),
            ("near", 10.0, 10.0),
        ],
        names=("MAIN_ID", "RA_d", "DEC_d"),
    )
    reference = Table(rows=[("ref", 10.01, 10.01)], names=("MAIN_ID", "RA_d", "DEC_d"))

    row_idx = _simbad_best_row_index(result, reference_coord=_simbad_row_coord(reference))

    assert row_idx == 1


class SimplePlanner:
    pass


def test_target_resolver_parses_coordinate_query_without_network() -> None:
    target = TargetResolver(SimplePlanner())._resolve_target_from_coordinates("10.684 +41.269")

    assert target is not None
    assert target.ra == pytest.approx(10.684)
    assert target.dec == pytest.approx(41.269)
    assert target.source_catalog == "coordinates"


def test_tns_endpoint_normalization_accepts_sandbox_alias() -> None:
    assert _normalize_tns_endpoint_key("sandbox") == "sandbox"
    assert _normalize_tns_endpoint_key("Sandbox TNS") == "sandbox"
    assert _normalize_tns_endpoint_key("unknown") == "production"
