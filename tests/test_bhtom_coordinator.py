from __future__ import annotations

from astroplanner.bhtom_coordinator import BhtomCoordinator
from astroplanner.models import Site, Target


class _DummyPlanner:
    pass


def test_bhtom_coordinator_candidate_round_trip() -> None:
    coordinator = BhtomCoordinator(_DummyPlanner())  # type: ignore[arg-type]
    target = Target(
        name="SN 2026abc",
        ra=10.0,
        dec=20.0,
        source_catalog="bhtom",
        source_object_id="bhtom-123",
        magnitude=17.2,
    )

    payload = coordinator._serialize_bhtom_candidates(
        [
            {
                "target": target,
                "importance": 4.5,
                "bhtom_priority": 3,
                "sun_separation": "92.5",
            }
        ]
    )
    restored = coordinator._deserialize_bhtom_candidates(payload)

    assert len(restored) == 1
    assert restored[0]["target"] == target
    assert restored[0]["importance"] == 4.5
    assert restored[0]["bhtom_priority"] == 3
    assert restored[0]["sun_separation"] == 92.5


def test_bhtom_coordinator_storage_key_uses_normalized_base_url() -> None:
    coordinator = BhtomCoordinator(_DummyPlanner())  # type: ignore[arg-type]

    assert coordinator._bhtom_storage_cache_key(token="secret", base_url="https://example.test///").startswith(
        "https://example.test::"
    )
    assert len(coordinator._bhtom_token_hash("secret")) == 64


def test_bhtom_coordinator_clones_observatory_presets() -> None:
    coordinator = BhtomCoordinator(_DummyPlanner())  # type: ignore[arg-type]
    site = Site(name="Obs", latitude=1.0, longitude=2.0, elevation=3.0)
    cloned = coordinator._clone_bhtom_observatory_presets(
        [{"key": "obs:1", "label": "Obs", "source": "bhtom", "site": site}]
    )

    assert len(cloned) == 1
    assert cloned[0]["site"] == site
    assert cloned[0]["site"] is not site
