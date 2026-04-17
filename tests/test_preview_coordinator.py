from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import QApplication

from astroplanner.models import Target
from astroplanner.preview_coordinator import CUTOUT_CACHE_MAX, PreviewCoordinator


def _pixmap() -> QPixmap:
    app = QApplication.instance() or QApplication([])
    assert app is not None
    pixmap = QPixmap(4, 4)
    pixmap.fill(QColor("#123456"))
    return pixmap


def test_preview_coordinator_caps_cutout_cache_without_storage() -> None:
    planner = SimpleNamespace(
        _cutout_cache={},
        _cutout_cache_order=[],
        targets=[],
        app_storage=None,
    )
    coordinator = PreviewCoordinator(planner)

    for idx in range(CUTOUT_CACHE_MAX + 6):
        coordinator._cache_cutout_pixmap(f"key-{idx}", _pixmap())

    assert len(planner._cutout_cache_order) == CUTOUT_CACHE_MAX
    assert "key-0" not in planner._cutout_cache
    assert f"key-{CUTOUT_CACHE_MAX + 5}" in planner._cutout_cache


def test_preview_coordinator_builds_stable_cutout_key() -> None:
    planner = SimpleNamespace(
        _cutout_survey_key="dss2",
        _cutout_fov_arcmin=15,
        _cutout_size_px=280,
        _aladin_context_factor=1.0,
    )
    coordinator = PreviewCoordinator(planner)
    target = Target(name="M31", ra=10.684, dec=41.269)

    key = coordinator._cutout_key_for_target(target, 280, 280)

    assert key.startswith("dss2:15:280x280:")
    assert "ctx1.00" in key
    assert "none:" in key
    assert key.endswith("10.684000,41.269000")


def test_preview_coordinator_cancel_finder_clears_pending_state() -> None:
    class _Timer:
        def __init__(self) -> None:
            self.stopped = False

        def stop(self) -> None:
            self.stopped = True

    timer = _Timer()
    planner = SimpleNamespace(
        _finder_timeout_timer=timer,
        _finder_workers=[],
        _finder_worker=None,
        _finder_pending_key="pending",
        _finder_pending_name="M31",
        _finder_pending_background=True,
        _finder_prefetch_queue=[(Target(name="M31", ra=10.684, dec=41.269), "key")],
        _finder_prefetch_enqueued_keys={"key"},
        _finder_prefetch_total=1,
        _finder_prefetch_completed=0,
        _finder_prefetch_cached=0,
        _finder_prefetch_active=True,
        _finder_request_id=3,
    )
    coordinator = PreviewCoordinator(planner)

    coordinator._cancel_finder_chart_worker()

    assert timer.stopped is True
    assert planner._finder_pending_key == ""
    assert planner._finder_pending_name == ""
    assert planner._finder_pending_background is False
    assert planner._finder_prefetch_queue == []
    assert planner._finder_prefetch_enqueued_keys == set()
    assert planner._finder_prefetch_active is False
    assert planner._finder_request_id == 4
