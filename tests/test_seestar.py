from __future__ import annotations

from datetime import datetime, timedelta
from io import BytesIO
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image

from astroplanner.seestar import (
    SEESTAR_ALP_AF_MODE_OFF,
    SEESTAR_ALP_AF_MODE_PER_RUN,
    SEESTAR_ALP_AF_MODE_PER_TARGET,
    SEESTAR_ALP_LP_FILTER_AUTO,
    NightQueueCandidate,
    NightQueueSiteSnapshot,
    SEESTAR_ALP_LP_FILTER_OFF,
    SEESTAR_ALP_LP_FILTER_ON,
    SEESTAR_CAMPAIGN_PRESET_BL_LAC,
    SEESTAR_TEMPLATE_SCOPE_SINGLE_TARGET,
    SeestarAlpAdapter,
    SeestarAlpBackendStatus,
    SeestarAlpClient,
    SeestarAlpConfig,
    SeestarGuidedAdapter,
    SeestarRawFlatMeasurement,
    SeestarSessionTemplate,
    SeestarSmartFlatsConfig,
    SeestarTargetSessionItem,
    build_alp_debug_sample_queue,
    build_alp_imager_video_url,
    build_alp_raw_stats_url,
    build_alp_schedule_item,
    build_alp_schedule_items,
    build_alp_web_ui_url,
    build_night_queue,
    build_repeated_target_queue,
    build_session_queue,
    builtin_seestar_session_templates,
    dump_user_seestar_session_templates,
    filter_alp_schedule_items_for_immediate_start,
    is_supported_dso_type,
    load_user_seestar_session_templates,
    analyze_raw_linearity,
    measure_preview_brightness,
    recommended_filter_for_object_type,
    render_alp_backend_status_text,
    render_alp_dialog_text,
    run_raw_adu_smart_flats,
    run_smart_sky_flats,
)


TZ = ZoneInfo("Europe/Warsaw")


def _dt(hour: int, minute: int = 0, *, day: int = 4) -> datetime:
    return datetime(2026, 4, day, hour, minute, tzinfo=TZ)


def _site() -> NightQueueSiteSnapshot:
    return NightQueueSiteSnapshot(
        name="Seestar S50 [WRO]",
        latitude=51.1,
        longitude=17.0,
        elevation=120.0,
        timezone="Europe/Warsaw",
        limiting_magnitude=18.5,
        telescope_diameter_mm=50.0,
        focal_length_mm=250.0,
        pixel_size_um=2.9,
        detector_width_px=1920,
        detector_height_px=1080,
    )


def _candidate(
    name: str,
    *,
    score: float,
    start: datetime,
    end: datetime,
    object_type: str = "galaxy",
    max_altitude_deg: float = 60.0,
    hours_above_limit: float = 2.5,
) -> NightQueueCandidate:
    return NightQueueCandidate(
        target_name=name,
        ra_deg=12.34,
        dec_deg=56.78,
        object_type=object_type,
        score=score,
        hours_above_limit=hours_above_limit,
        max_altitude_deg=max_altitude_deg,
        window_start_local=start,
        window_end_local=end,
        notes=f"Notes for {name}",
    )


def test_build_night_queue_prefers_score_and_handles_gaps() -> None:
    queue = build_night_queue(
        [
            _candidate("M42", score=92, start=_dt(20, 0), end=_dt(22, 0), object_type="emission nebula"),
            _candidate("M31", score=70, start=_dt(20, 15), end=_dt(23, 0)),
            _candidate("NGC 7000", score=88, start=_dt(23, 30), end=_dt(1, 30, day=5), object_type="nebula"),
        ],
        site_snapshot=_site(),
        night_start_local=_dt(19, 45),
        night_end_local=_dt(2, 0, day=5),
        timezone="Europe/Warsaw",
        default_block_minutes=60,
        min_block_minutes=20,
        max_targets_per_night=3,
    )

    assert [block.target_name for block in queue.blocks] == ["M42", "NGC 7000"]
    assert queue.blocks[0].start_local == _dt(20, 0)
    assert queue.blocks[0].end_local == _dt(21, 0)
    assert queue.blocks[1].start_local == _dt(23, 30)
    assert queue.blocks[1].end_local == _dt(0, 30, day=5)


def test_build_night_queue_tie_breaks_by_window_end_then_altitude() -> None:
    queue = build_night_queue(
        [
            _candidate("Earlier End", score=80, start=_dt(20, 0), end=_dt(21, 20), max_altitude_deg=55.0),
            _candidate("Later End", score=80, start=_dt(20, 0), end=_dt(22, 0), max_altitude_deg=80.0),
        ],
        site_snapshot=_site(),
        night_start_local=_dt(19, 30),
        night_end_local=_dt(1, 0, day=5),
        timezone="Europe/Warsaw",
        default_block_minutes=45,
        min_block_minutes=20,
        max_targets_per_night=2,
    )

    assert [block.target_name for block in queue.blocks] == ["Earlier End", "Later End"]
    assert queue.blocks[0].start_local == _dt(20, 0)
    assert queue.blocks[0].end_local == _dt(20, 45)


def test_build_night_queue_skips_unsupported_short_windows_and_respects_limit() -> None:
    queue = build_night_queue(
        [
            _candidate("The Moon", score=99, start=_dt(20, 0), end=_dt(22, 0), object_type="moon"),
            _candidate("Too Short", score=95, start=_dt(21, 0), end=_dt(21, 10), object_type="galaxy"),
            _candidate("M51", score=78, start=_dt(21, 0), end=_dt(23, 0), object_type="galaxy"),
            _candidate("M13", score=77, start=_dt(23, 0), end=_dt(1, 30, day=5), object_type="globular cluster"),
            _candidate("M27", score=76, start=_dt(1, 0, day=5), end=_dt(2, 0, day=5), object_type="planetary nebula"),
        ],
        site_snapshot=_site(),
        night_start_local=_dt(20, 0),
        night_end_local=_dt(2, 0, day=5),
        timezone="Europe/Warsaw",
        default_block_minutes=60,
        min_block_minutes=20,
        max_targets_per_night=2,
    )

    assert [block.target_name for block in queue.blocks] == ["M51", "M13"]
    assert len(queue.blocks) == 2
    assert all("Moon" not in block.target_name for block in queue.blocks)


def test_guided_adapter_bundle_is_consistent() -> None:
    queue = build_night_queue(
        [
            _candidate("NGC 7000", score=88, start=_dt(22, 0), end=_dt(1, 0, day=5), object_type="nebula"),
            _candidate("M31", score=82, start=_dt(20, 30), end=_dt(23, 30), object_type="galaxy"),
        ],
        site_snapshot=_site(),
        night_start_local=_dt(20, 0),
        night_end_local=_dt(2, 0, day=5),
        timezone="Europe/Warsaw",
        default_block_minutes=50,
        min_block_minutes=20,
        max_targets_per_night=2,
    )
    adapter = SeestarGuidedAdapter()
    bundle = adapter.build_handoff_bundle(queue)

    assert bundle.session_payload["timezone"] == "Europe/Warsaw"
    assert len(bundle.csv_rows) == len(queue.blocks)
    assert bundle.csv_rows[0]["alias"] == queue.blocks[0].alias
    assert bundle.csv_rows[0]["recommended_filter"] == queue.blocks[0].recommended_filter
    assert "Custom Celestial Objects" in bundle.dialog_text
    assert "Plan Mode" in bundle.checklist_markdown
    assert len({row["alias"] for row in bundle.csv_rows}) == len(bundle.csv_rows)


def test_alp_schedule_item_and_bundle_are_consistent() -> None:
    queue = build_night_queue(
        [
            _candidate("NGC 7000", score=88, start=_dt(22, 0), end=_dt(23, 15), object_type="nebula"),
        ],
        site_snapshot=_site(),
        night_start_local=_dt(20, 0),
        night_end_local=_dt(2, 0, day=5),
        timezone="Europe/Warsaw",
        default_block_minutes=75,
        min_block_minutes=20,
        max_targets_per_night=1,
    )
    block = queue.blocks[0]
    item = build_alp_schedule_item(block)

    assert item["action"] == "start_mosaic"
    assert item["params"]["target_name"] == block.alias
    assert item["params"]["ra"] == block.ra_hms
    assert item["params"]["dec"] == block.dec_dms
    assert item["params"]["panel_time_sec"] == 75 * 60
    assert item["params"]["is_use_lp_filter"] is True

    adapter = SeestarAlpAdapter(SeestarAlpConfig(base_url="http://localhost:5555", device_num=1, client_id=1, timeout_s=5.0))
    bundle = adapter.build_handoff_bundle(queue)
    assert bundle.session_payload["alp_service"]["base_url"] == "http://localhost:5555"
    schedule_items = bundle.session_payload["schedule_items"]
    assert [item["action"] for item in schedule_items] == ["wait_until", "set_wheel_position", "start_mosaic"]
    assert bundle.csv_rows[0]["recommended_filter"] == "wait_until"
    assert bundle.csv_rows[-1]["alias"] == block.alias
    assert "Push + Start" in bundle.dialog_text
    assert "ALP service" in bundle.checklist_markdown


def test_alp_schedule_item_uses_configurable_params() -> None:
    queue = build_night_queue(
        [
            _candidate("M51", score=81, start=_dt(22, 0), end=_dt(23, 15), object_type="galaxy"),
        ],
        site_snapshot=_site(),
        night_start_local=_dt(20, 0),
        night_end_local=_dt(2, 0, day=5),
        timezone="Europe/Warsaw",
        default_block_minutes=75,
        min_block_minutes=20,
        max_targets_per_night=1,
    )
    block = queue.blocks[0]
    config = SeestarAlpConfig(
        base_url="http://localhost:5555",
        device_num=1,
        client_id=1,
        timeout_s=5.0,
        gain=95,
        panel_overlap_percent=12,
        use_autofocus=True,
        num_tries=3,
        retry_wait_s=45,
        target_integration_override_min=30,
        stack_exposure_ms=5000,
    )

    item = build_alp_schedule_item(block, config)

    assert item["params"]["panel_time_sec"] == 30 * 60
    assert item["params"]["gain"] == 95
    assert item["params"]["panel_overlap_percent"] == 12
    assert item["params"]["is_use_autofocus"] is True
    assert item["params"]["num_tries"] == 3
    assert item["params"]["retry_wait_s"] == 45

    adapter = SeestarAlpAdapter(config)
    bundle = adapter.build_handoff_bundle(queue)
    assert bundle.session_payload["alp_service"]["gain"] == 95
    assert bundle.session_payload["alp_service"]["target_integration_override_min"] == 30
    assert [item["action"] for item in bundle.session_payload["schedule_items"]] == [
        "action_set_exposure",
        "wait_until",
        "set_wheel_position",
        "start_mosaic",
    ]
    assert "Gain: 95" in bundle.dialog_text
    assert "Stack exposure: `5000 ms`" in bundle.checklist_markdown


def test_alp_schedule_item_respects_lp_filter_override_modes() -> None:
    queue = build_night_queue(
        [
            _candidate("M51", score=81, start=_dt(22, 0), end=_dt(23, 15), object_type="galaxy"),
        ],
        site_snapshot=_site(),
        night_start_local=_dt(20, 0),
        night_end_local=_dt(2, 0, day=5),
        timezone="Europe/Warsaw",
        default_block_minutes=75,
        min_block_minutes=20,
        max_targets_per_night=1,
    )
    block = queue.blocks[0]

    auto_item = build_alp_schedule_item(block, SeestarAlpConfig())
    forced_off = build_alp_schedule_item(block, SeestarAlpConfig(lp_filter_mode=SEESTAR_ALP_LP_FILTER_OFF))
    forced_on = build_alp_schedule_item(block, SeestarAlpConfig(lp_filter_mode=SEESTAR_ALP_LP_FILTER_ON))

    assert auto_item["params"]["is_use_lp_filter"] is False
    assert forced_off["params"]["is_use_lp_filter"] is False
    assert forced_on["params"]["is_use_lp_filter"] is True


def test_alp_client_push_queue_applies_stack_exposure_before_schedule_upload() -> None:
    queue = build_night_queue(
        [
            _candidate("M31", score=82, start=_dt(20, 30), end=_dt(21, 30), object_type="galaxy"),
        ],
        site_snapshot=_site(),
        night_start_local=_dt(20, 0),
        night_end_local=_dt(2, 0, day=5),
        timezone="Europe/Warsaw",
        default_block_minutes=60,
        min_block_minutes=20,
        max_targets_per_night=1,
    )

    class _StubClient(SeestarAlpClient):
        def __init__(self, config: SeestarAlpConfig):
            super().__init__(config)
            self.calls: list[tuple[str, dict[str, object]]] = []
            self._schedule_state = "stopped"
            self._schedule_id = ""
            self._schedule_id = ""

        def get_connected(self) -> bool:
            return True

        def put_action(self, action: str, params: dict[str, object]) -> dict[str, object]:
            self.calls.append((action, params))
            if action == "create_schedule":
                self._schedule_id = str(params.get("schedule_id", "") or "")
            if action == "start_scheduler":
                self._schedule_state = "working"
            return {"Value": {"ok": True, "state": "stopped"}}

        def get_schedule(self) -> dict[str, object]:
            return {"schedule_id": self._schedule_id, "state": self._schedule_state, "list": []}

    client = _StubClient(
        SeestarAlpConfig(
            base_url="http://localhost:5555",
            device_num=1,
            client_id=1,
            timeout_s=5.0,
            stack_exposure_ms=7000,
        )
    )
    result = client.push_queue(queue, start_immediately=False)

    assert result["uploaded_items"] == 1
    assert client.calls[0][0] == "method_sync"
    assert client.calls[0][1]["method"] == "set_setting"
    assert client.calls[0][1]["params"] == {"exp_ms": {"stack_l": 7000}}
    assert client.calls[1][0] == "create_schedule"
    assert client.calls[2][0] == "add_schedule_item"


def test_build_alp_schedule_items_support_optional_automation_steps() -> None:
    queue = build_night_queue(
        [
            _candidate("BL Lac", score=91, start=_dt(22, 0), end=_dt(23, 0), object_type="blazar"),
        ],
        site_snapshot=_site(),
        night_start_local=_dt(20, 0),
        night_end_local=_dt(2, 0, day=5),
        timezone="Europe/Warsaw",
        default_block_minutes=60,
        min_block_minutes=20,
        max_targets_per_night=1,
    )

    items = build_alp_schedule_items(
        queue,
        SeestarAlpConfig(
            wait_until_local_time="21:15",
            startup_enabled=True,
            startup_polar_align=True,
            startup_auto_focus=True,
            capture_flats_before_session=True,
            flats_wait_s=90,
            stack_exposure_ms=10000,
            lp_filter_mode=SEESTAR_ALP_LP_FILTER_OFF,
            schedule_autofocus_before_each_target=True,
            schedule_autofocus_try_count=2,
            dew_heater_value=35,
            park_after_session=True,
        ),
    )

    assert [item["action"] for item in items] == [
        "wait_until",
        "start_up_sequence",
        "action_set_dew_heater",
        "set_wheel_position",
        "start_create_calib_frame",
        "wait_for",
        "action_set_exposure",
        "auto_focus",
        "start_mosaic",
        "scope_park",
    ]
    assert items[0]["params"] == {"local_time": "21:15"}
    assert items[1]["params"] == {"auto_focus": True, "3ppa": True, "dark_frames": False}
    assert items[4]["action"] == "start_create_calib_frame"
    assert items[5]["params"] == {"timer_sec": 90}


def test_render_alp_dialog_marks_stale_wait_until_for_immediate_start() -> None:
    queue = build_session_queue(
        [
            SeestarTargetSessionItem(
                order=1,
                target_name="Current target",
                ra_deg=42.0,
                dec_deg=21.0,
                object_type="blazar",
                repeat_count=1,
                segment_minutes=10,
            )
        ],
        session_template=SeestarSessionTemplate(
            name="Current settings",
            repeat_count=1,
            minutes_per_run=10,
        ),
        site_snapshot=_site(),
        night_start_local=_dt(20, 0),
        night_end_local=_dt(2, 0, day=5),
        timezone="Europe/Warsaw",
        start_cursor_local=_dt(21, 17),
        generated_at=_dt(21, 17),
    )
    adapter = SeestarAlpAdapter(
        SeestarAlpConfig(base_url="http://localhost:5555", device_num=1, client_id=1, timeout_s=5.0)
    )
    bundle = adapter.build_handoff_bundle(queue)

    text = render_alp_dialog_text(bundle, adapter.config)

    assert "will be skipped on Push + Start" in text


def test_render_alp_dialog_hides_science_checklist_when_not_required() -> None:
    queue = build_session_queue(
        [
            SeestarTargetSessionItem(
                order=1,
                target_name="No checklist target",
                ra_deg=42.0,
                dec_deg=21.0,
                object_type="blazar",
            )
        ],
        session_template=SeestarSessionTemplate(
            name="Current settings",
            repeat_count=1,
            minutes_per_run=10,
            require_science_checklist=False,
            science_checklist_items=["Should not be shown"],
            template_notes="Also hidden with checklist off",
        ),
        site_snapshot=_site(),
        night_start_local=_dt(20, 0),
        night_end_local=_dt(2, 0, day=5),
        timezone="Europe/Warsaw",
    )
    adapter = SeestarAlpAdapter(
        SeestarAlpConfig(base_url="http://localhost:5555", device_num=1, client_id=1, timeout_s=5.0)
    )
    bundle = adapter.build_handoff_bundle(queue)

    assert "Science Checklist" not in bundle.dialog_text
    assert "Should not be shown" not in bundle.dialog_text


def test_build_alp_schedule_items_inserts_wait_steps_between_segmented_blocks() -> None:
    queue = build_repeated_target_queue(
        _candidate("BL Lac", score=91, start=_dt(22, 0), end=_dt(23, 0), object_type="blazar"),
        site_snapshot=_site(),
        night_start_local=_dt(21, 30),
        night_end_local=_dt(0, 0, day=5),
        timezone="Europe/Warsaw",
        repeat_count=2,
        segment_minutes=10,
        gap_seconds=60,
    )

    items = build_alp_schedule_items(queue, SeestarAlpConfig())

    assert [item["action"] for item in items] == [
        "wait_until",
        "set_wheel_position",
        "start_mosaic",
        "wait_for",
        "start_mosaic",
    ]
    assert items[0]["params"] == {"local_time": "22:00"}
    assert items[3]["params"] == {"timer_sec": 60}


def test_push_queue_start_immediately_skips_expired_wait_until_upload() -> None:
    queue = build_session_queue(
        [
            SeestarTargetSessionItem(
                order=1,
                target_name="Current target",
                ra_deg=42.0,
                dec_deg=21.0,
                object_type="blazar",
                repeat_count=1,
                segment_minutes=10,
            )
        ],
        session_template=SeestarSessionTemplate(
            name="Current settings",
            repeat_count=1,
            minutes_per_run=10,
        ),
        site_snapshot=_site(),
        night_start_local=_dt(20, 0),
        night_end_local=_dt(2, 0, day=5),
        timezone="Europe/Warsaw",
        start_cursor_local=_dt(21, 17),
    )

    class _StubClient(SeestarAlpClient):
        def __init__(self, config: SeestarAlpConfig):
            super().__init__(config)
            self.calls: list[tuple[str, dict[str, object]]] = []
            self._schedule_state = "stopped"

        def get_connected(self) -> bool:
            return True

        def put_action(self, action: str, params: dict[str, object]) -> dict[str, object]:
            self.calls.append((action, params))
            if action == "create_schedule":
                self._schedule_id = str(params.get("schedule_id", "") or "")
            if action == "start_scheduler":
                self._schedule_state = "working"
            return {"Value": {"ok": True, "state": "stopped"}}

        def get_schedule(self) -> dict[str, object]:
            return {"schedule_id": self._schedule_id, "state": self._schedule_state, "list": []}

    client = _StubClient(
        SeestarAlpConfig(
            base_url="http://localhost:5555",
            device_num=1,
            client_id=1,
            timeout_s=5.0,
        )
    )
    schedule_items = build_alp_schedule_items(queue, client.config)
    result = client.push_queue(
        queue,
        schedule_items=schedule_items,
        start_immediately=True,
        now_local=_dt(21, 17) + timedelta(seconds=45),
    )

    uploaded_actions = [params["action"] for action, params in client.calls if action == "add_schedule_item"]
    assert uploaded_actions == ["set_wheel_position", "start_mosaic"]
    assert result["started"] is True


def test_push_queue_tolerates_non_dict_create_schedule_value() -> None:
    queue = build_session_queue(
        [
            SeestarTargetSessionItem(
                order=1,
                target_name="Current target",
                ra_deg=42.0,
                dec_deg=21.0,
                object_type="blazar",
            )
        ],
        session_template=SeestarSessionTemplate(
            name="Current settings",
            repeat_count=1,
            minutes_per_run=10,
        ),
        site_snapshot=_site(),
        night_start_local=_dt(20, 0),
        night_end_local=_dt(2, 0, day=5),
        timezone="Europe/Warsaw",
    )

    class _StubClient(SeestarAlpClient):
        def __init__(self, config: SeestarAlpConfig):
            super().__init__(config)
            self.calls: list[tuple[str, dict[str, object]]] = []

        def get_connected(self) -> bool:
            return True

        def put_action(self, action: str, params: dict[str, object]) -> dict[str, object]:
            self.calls.append((action, params))
            if action == "create_schedule":
                return {"Value": "ok"}
            return {"Value": {"ok": True, "state": "stopped"}}

        def get_schedule(self) -> dict[str, object]:
            return {"schedule_id": "abc", "state": "stopped", "list": []}

    client = _StubClient(
        SeestarAlpConfig(
            base_url="http://localhost:5555",
            device_num=1,
            client_id=1,
            timeout_s=5.0,
        )
    )
    result = client.push_queue(queue, start_immediately=False)

    assert result["uploaded_items"] == 1
    assert client.calls[0][0] == "create_schedule"
    assert client.calls[1][0] == "add_schedule_item"


def test_push_queue_start_immediately_uses_backend_schedule_id() -> None:
    queue = build_session_queue(
        [
            SeestarTargetSessionItem(
                order=1,
                target_name="Current target",
                ra_deg=42.0,
                dec_deg=21.0,
                object_type="blazar",
            )
        ],
        session_template=SeestarSessionTemplate(
            name="Current settings",
            repeat_count=1,
            minutes_per_run=10,
        ),
        site_snapshot=_site(),
        night_start_local=_dt(20, 0),
        night_end_local=_dt(2, 0, day=5),
        timezone="Europe/Warsaw",
    )

    class _StubClient(SeestarAlpClient):
        def __init__(self, config: SeestarAlpConfig):
            super().__init__(config)
            self.calls: list[tuple[str, dict[str, object]]] = []

        def get_connected(self) -> bool:
            return True

        def put_action(self, action: str, params: dict[str, object]) -> dict[str, object]:
            self.calls.append((action, params))
            if action == "create_schedule":
                return {"Value": {"schedule_id": "backend-id", "state": "stopped", "list": []}}
            if action == "add_schedule_item":
                return {"Value": {"schedule_id": "backend-id", "state": "stopped", "list": []}}
            if action == "start_scheduler":
                return {"Value": {"code": 0, "result": "ok"}}
            return {"Value": {"ok": True}}

        def get_schedule(self) -> dict[str, object]:
            return {"schedule_id": "backend-id", "state": "working", "list": [{}]}

    client = _StubClient(
        SeestarAlpConfig(
            base_url="http://localhost:5555",
            device_num=1,
            client_id=1,
            timeout_s=5.0,
        )
    )
    result = client.push_queue(queue, start_immediately=True, now_local=_dt(20, 0))

    start_calls = [params for action, params in client.calls if action == "start_scheduler"]
    assert start_calls == [{}]
    assert result["started"] is True
    assert result["schedule_id"] == "backend-id"


def test_immediate_start_filters_expired_wait_until() -> None:
    queue = build_session_queue(
        [
            SeestarTargetSessionItem(
                order=1,
                target_name="Current target",
                ra_deg=42.0,
                dec_deg=21.0,
                object_type="blazar",
                repeat_count=1,
                segment_minutes=10,
            )
        ],
        session_template=SeestarSessionTemplate(
            name="Current settings",
            repeat_count=1,
            minutes_per_run=10,
        ),
        site_snapshot=_site(),
        night_start_local=_dt(20, 0),
        night_end_local=_dt(2, 0, day=5),
        timezone="Europe/Warsaw",
        start_cursor_local=_dt(21, 17),
    )

    items = build_alp_schedule_items(queue, SeestarAlpConfig())
    filtered = filter_alp_schedule_items_for_immediate_start(
        items,
        queue=queue,
        now_local=_dt(21, 17) + timedelta(seconds=45),
    )

    assert [item["action"] for item in items][:3] == ["wait_until", "set_wheel_position", "start_mosaic"]
    assert [item["action"] for item in filtered] == ["set_wheel_position", "start_mosaic"]


def test_immediate_start_skips_wait_until_even_if_it_is_future_across_midnight() -> None:
    queue = build_session_queue(
        [
            SeestarTargetSessionItem(
                order=1,
                target_name="Night target",
                ra_deg=42.0,
                dec_deg=21.0,
                object_type="blazar",
                repeat_count=1,
                segment_minutes=10,
            )
        ],
        session_template=SeestarSessionTemplate(
            name="Current settings",
            repeat_count=1,
            minutes_per_run=10,
            wait_until_local_time="00:30",
            honor_queue_times=False,
        ),
        site_snapshot=_site(),
        night_start_local=_dt(20, 0),
        night_end_local=_dt(5, 0, day=5),
        timezone="Europe/Warsaw",
        start_cursor_local=_dt(22, 0),
    )

    items = build_alp_schedule_items(
        queue,
        SeestarAlpConfig(wait_until_local_time="00:30", honor_queue_times=False),
    )
    filtered = filter_alp_schedule_items_for_immediate_start(
        items,
        queue=queue,
        now_local=_dt(22, 0),
    )

    assert [item["action"] for item in filtered][:2] == ["set_wheel_position", "start_mosaic"]
def test_build_alp_schedule_items_honor_per_block_exposure_lp_and_autofocus() -> None:
    queue = build_session_queue(
        [
            SeestarTargetSessionItem(
                order=1,
                target_name="Science 1",
                ra_deg=12.0,
                dec_deg=22.0,
                object_type="galaxy",
                stack_exposure_ms=5000,
                lp_filter_mode=SEESTAR_ALP_LP_FILTER_OFF,
                autofocus=False,
            ),
            SeestarTargetSessionItem(
                order=2,
                target_name="Science 2",
                ra_deg=24.0,
                dec_deg=12.0,
                object_type="nebula",
                stack_exposure_ms=10000,
                lp_filter_mode=SEESTAR_ALP_LP_FILTER_ON,
                autofocus=True,
            ),
        ],
        session_template=SeestarSessionTemplate(
            repeat_count=1,
            minutes_per_run=10,
            stack_exposure_ms=0,
            lp_filter_mode=SEESTAR_ALP_LP_FILTER_AUTO,
            use_autofocus=False,
        ),
        site_snapshot=_site(),
        night_start_local=_dt(22, 0),
        night_end_local=_dt(1, 0, day=5),
        timezone="Europe/Warsaw",
    )

    items = build_alp_schedule_items(queue, SeestarAlpConfig(stack_exposure_ms=0, lp_filter_mode=SEESTAR_ALP_LP_FILTER_AUTO))

    assert [item["action"] for item in items] == [
        "wait_until",
        "set_wheel_position",
        "action_set_exposure",
        "start_mosaic",
        "set_wheel_position",
        "action_set_exposure",
        "start_mosaic",
    ]
    assert items[2]["params"] == {"exp": 5000}
    assert items[3]["params"]["is_use_autofocus"] is False
    assert items[5]["params"] == {"exp": 10000}
    assert items[6]["params"]["is_use_lp_filter"] is True
    assert items[6]["params"]["is_use_autofocus"] is True


def test_build_repeated_target_queue_splits_target_into_segmented_sessions() -> None:
    queue = build_repeated_target_queue(
        _candidate("BL Lac", score=91, start=_dt(22, 0), end=_dt(23, 15), object_type="blazar"),
        site_snapshot=_site(),
        night_start_local=_dt(21, 30),
        night_end_local=_dt(0, 0, day=5),
        timezone="Europe/Warsaw",
        repeat_count=6,
        segment_minutes=10,
        gap_seconds=60,
    )

    assert len(queue.blocks) == 6
    assert queue.blocks[0].start_local == _dt(22, 0)
    assert queue.blocks[-1].end_local == _dt(23, 5)
    assert queue.blocks[0].alias.endswith("-s01")
    assert queue.blocks[-1].alias.endswith("-s06")
    assert "session 1/6" in queue.blocks[0].notes
    assert "session 6/6" in queue.blocks[-1].notes


def test_builtin_bl_lac_template_roundtrips_as_user_template_payload() -> None:
    template = builtin_seestar_session_templates()[SEESTAR_CAMPAIGN_PRESET_BL_LAC]
    payload = dump_user_seestar_session_templates({"bl_lac_copy": template})
    loaded = load_user_seestar_session_templates(payload)

    assert template.scope == SEESTAR_TEMPLATE_SCOPE_SINGLE_TARGET
    assert template.repeat_count == 6
    assert template.minutes_per_run == 10
    assert template.require_science_checklist is True
    assert template.park_after_session is True
    assert "bl-lac-copy" in loaded
    assert loaded["bl-lac-copy"].name == template.name
    assert loaded["bl-lac-copy"].stack_exposure_ms == 10000
    assert "single BL Lac target" in loaded["bl-lac-copy"].template_notes


def test_build_session_queue_keeps_target_table_order_and_row_overrides() -> None:
    template = SeestarSessionTemplate(
        name="Current settings",
        repeat_count=2,
        minutes_per_run=15,
        gap_seconds=30,
        lp_filter_mode=SEESTAR_ALP_LP_FILTER_OFF,
        stack_exposure_ms=10000,
    )

    queue = build_session_queue(
        [
            SeestarTargetSessionItem(
                order=1,
                target_name="Target B",
                ra_deg=20.0,
                dec_deg=10.0,
                object_type="galaxy",
                repeat_count=1,
                segment_minutes=12,
                lp_filter_mode=SEESTAR_ALP_LP_FILTER_ON,
            ),
            SeestarTargetSessionItem(
                order=2,
                target_name="Target A",
                ra_deg=10.0,
                dec_deg=-5.0,
                object_type="planetary nebula",
                repeat_count=3,
                segment_minutes=8,
                gap_seconds=45,
                stack_exposure_ms=5000,
                autofocus=True,
            ),
        ],
        session_template=template,
        site_snapshot=_site(),
        night_start_local=_dt(20, 0),
        night_end_local=_dt(2, 0, day=5),
        timezone="Europe/Warsaw",
    )

    assert [block.target_name for block in queue.blocks] == ["Target B", "Target A", "Target A", "Target A"]
    assert queue.blocks[0].start_local == _dt(20, 0)
    assert queue.blocks[0].end_local == _dt(20, 12)
    assert queue.blocks[1].start_local == _dt(20, 12) + timedelta(seconds=30)
    assert queue.blocks[2].start_local == _dt(20, 20) + timedelta(seconds=75)
    assert queue.blocks[3].start_local == _dt(20, 29) + timedelta(seconds=60)
    assert queue.blocks[0].lp_filter_mode == SEESTAR_ALP_LP_FILTER_ON
    assert queue.blocks[1].stack_exposure_ms == 5000
    assert queue.blocks[1].use_autofocus is True
    assert queue.campaign_name == "Current settings"


def test_build_alp_schedule_item_uses_session_block_duration_not_integration_override() -> None:
    queue = build_session_queue(
        [
            SeestarTargetSessionItem(
                order=1,
                target_name="Science",
                ra_deg=12.0,
                dec_deg=22.0,
                object_type="galaxy",
                repeat_count=1,
                segment_minutes=10,
            )
        ],
        session_template=SeestarSessionTemplate(
            name="Current settings",
            repeat_count=1,
            minutes_per_run=10,
            target_integration_override_min=30,
        ),
        site_snapshot=_site(),
        night_start_local=_dt(21, 0),
        night_end_local=_dt(2, 0, day=5),
        timezone="Europe/Warsaw",
    )

    item = build_alp_schedule_item(
        queue.blocks[0],
        SeestarAlpConfig(target_integration_override_min=30),
    )

    assert item["action"] == "start_mosaic"
    assert item["params"]["panel_time_sec"] == 600


def test_build_alp_schedule_items_insert_wait_between_session_targets() -> None:
    queue = build_session_queue(
        [
            SeestarTargetSessionItem(
                order=1,
                target_name="Target B",
                ra_deg=20.0,
                dec_deg=10.0,
                object_type="galaxy",
                repeat_count=1,
                segment_minutes=10,
            ),
            SeestarTargetSessionItem(
                order=2,
                target_name="Target A",
                ra_deg=10.0,
                dec_deg=-5.0,
                object_type="planetary nebula",
                repeat_count=1,
                segment_minutes=10,
            ),
        ],
        session_template=SeestarSessionTemplate(
            name="Current settings",
            repeat_count=1,
            minutes_per_run=10,
            gap_seconds=60,
            lp_filter_mode=SEESTAR_ALP_LP_FILTER_OFF,
        ),
        site_snapshot=_site(),
        night_start_local=_dt(20, 0),
        night_end_local=_dt(2, 0, day=5),
        timezone="Europe/Warsaw",
    )

    items = build_alp_schedule_items(queue, SeestarAlpConfig())

    assert [item["action"] for item in items] == [
        "wait_until",
        "set_wheel_position",
        "start_mosaic",
        "wait_for",
        "start_mosaic",
    ]
    assert items[3]["params"] == {"timer_sec": 60}


def test_build_alp_schedule_items_can_run_schedule_autofocus_per_run() -> None:
    queue = build_session_queue(
        [
            SeestarTargetSessionItem(
                order=1,
                target_name="Target A",
                ra_deg=20.0,
                dec_deg=10.0,
                object_type="qso",
                repeat_count=2,
                segment_minutes=10,
            ),
        ],
        session_template=SeestarSessionTemplate(
            name="Current settings",
            repeat_count=2,
            minutes_per_run=10,
            gap_seconds=30,
        ),
        site_snapshot=_site(),
        night_start_local=_dt(20, 0),
        night_end_local=_dt(2, 0, day=5),
        timezone="Europe/Warsaw",
    )

    items = build_alp_schedule_items(
        queue,
        SeestarAlpConfig(
            schedule_autofocus_mode=SEESTAR_ALP_AF_MODE_PER_RUN,
            schedule_autofocus_try_count=2,
        ),
    )

    assert [item["action"] for item in items] == [
        "wait_until",
        "set_wheel_position",
        "auto_focus",
        "start_mosaic",
        "wait_for",
        "auto_focus",
        "start_mosaic",
    ]
    assert items[2]["params"] == {"try_count": 2}
    assert items[5]["params"] == {"try_count": 2}


def test_build_alp_schedule_items_can_run_schedule_autofocus_once_per_target() -> None:
    queue = build_session_queue(
        [
            SeestarTargetSessionItem(
                order=1,
                target_name="Target A",
                ra_deg=20.0,
                dec_deg=10.0,
                object_type="qso",
                repeat_count=2,
                segment_minutes=10,
            ),
            SeestarTargetSessionItem(
                order=2,
                target_name="Target B",
                ra_deg=30.0,
                dec_deg=15.0,
                object_type="galaxy",
                repeat_count=2,
                segment_minutes=10,
            ),
        ],
        session_template=SeestarSessionTemplate(
            name="Current settings",
            repeat_count=1,
            minutes_per_run=10,
            gap_seconds=30,
        ),
        site_snapshot=_site(),
        night_start_local=_dt(20, 0),
        night_end_local=_dt(2, 0, day=5),
        timezone="Europe/Warsaw",
    )

    items = build_alp_schedule_items(
        queue,
        SeestarAlpConfig(
            schedule_autofocus_mode=SEESTAR_ALP_AF_MODE_PER_TARGET,
            schedule_autofocus_try_count=3,
        ),
    )

    actions = [item["action"] for item in items]
    assert actions == [
        "wait_until",
        "set_wheel_position",
        "auto_focus",
        "start_mosaic",
        "wait_for",
        "start_mosaic",
        "wait_for",
        "auto_focus",
        "start_mosaic",
        "wait_for",
        "start_mosaic",
    ]
    autofocus_items = [item for item in items if item["action"] == "auto_focus"]
    assert len(autofocus_items) == 2
    assert all(item["params"] == {"try_count": 3} for item in autofocus_items)


def test_build_session_queue_single_target_list_only_generates_that_target() -> None:
    queue = build_session_queue(
        [
            SeestarTargetSessionItem(
                order=1,
                target_name="Only target",
                ra_deg=42.0,
                dec_deg=21.0,
                object_type="blazar",
            )
        ],
        session_template=SeestarSessionTemplate(
            name="BL Lac campaign",
            scope=SEESTAR_TEMPLATE_SCOPE_SINGLE_TARGET,
            repeat_count=1,
            minutes_per_run=10,
        ),
        site_snapshot=_site(),
        night_start_local=_dt(21, 0),
        night_end_local=_dt(2, 0, day=5),
        timezone="Europe/Warsaw",
    )

    assert len(queue.blocks) == 1
    assert queue.blocks[0].target_name == "Only target"


def test_build_session_queue_can_start_from_current_moment() -> None:
    queue = build_session_queue(
        [
            SeestarTargetSessionItem(
                order=1,
                target_name="Current target",
                ra_deg=42.0,
                dec_deg=21.0,
                object_type="blazar",
                repeat_count=1,
                segment_minutes=10,
            )
        ],
        session_template=SeestarSessionTemplate(
            name="Current settings",
            repeat_count=1,
            minutes_per_run=10,
        ),
        site_snapshot=_site(),
        night_start_local=_dt(20, 0),
        night_end_local=_dt(2, 0, day=5),
        timezone="Europe/Warsaw",
        start_cursor_local=_dt(21, 17),
    )

    assert len(queue.blocks) == 1
    assert queue.blocks[0].start_local == _dt(21, 17)
    assert queue.blocks[0].end_local == _dt(21, 27)
    assert queue.night_start_local == _dt(21, 17)


def test_type_helpers_cover_expected_filters() -> None:
    assert is_supported_dso_type("reflection nebula")
    assert not is_supported_dso_type("moon")
    assert recommended_filter_for_object_type("planetary nebula") == "duoband"
    assert recommended_filter_for_object_type("open cluster") == "uv_ir"
    assert recommended_filter_for_object_type("qso") == "uv_ir"
    assert recommended_filter_for_object_type("") == "auto"


def test_build_alp_debug_sample_queue_shortens_blocks() -> None:
    queue = build_night_queue(
        [
            _candidate("M31", score=82, start=_dt(20, 30), end=_dt(23, 30), object_type="galaxy"),
            _candidate("NGC 7000", score=88, start=_dt(22, 0), end=_dt(1, 0, day=5), object_type="nebula"),
            _candidate("M27", score=70, start=_dt(1, 0, day=5), end=_dt(2, 0, day=5), object_type="planetary nebula"),
        ],
        site_snapshot=_site(),
        night_start_local=_dt(20, 0),
        night_end_local=_dt(2, 0, day=5),
        timezone="Europe/Warsaw",
        default_block_minutes=50,
        min_block_minutes=20,
        max_targets_per_night=3,
    )

    sample_queue = build_alp_debug_sample_queue(queue, max_blocks=2, sample_block_seconds=60, gap_seconds=15)

    assert len(sample_queue.blocks) == 2
    assert all(block.alias.endswith("-sample") for block in sample_queue.blocks)
    assert all(int((block.end_local - block.start_local).total_seconds()) == 60 for block in sample_queue.blocks)
    assert sample_queue.source_plan_version.endswith("-sample")


def test_render_alp_backend_status_text_reports_state_and_error() -> None:
    status = SeestarAlpBackendStatus(
        base_url="http://localhost:5555",
        device_num=1,
        connected=True,
        device_name="Seestar Smart Telescope",
        schedule_state="working",
        schedule_id="abc",
        queued_items=2,
        current_item_id="item-1",
        supports_raw_flats=True,
    )

    text = render_alp_backend_status_text(status, last_error="timeout")

    assert "connected" in text
    assert "schedule: working" in text
    assert "items: 2" in text
    assert "raw flats: yes" in text
    assert "last error: timeout" in text


def test_build_alp_web_ui_url_uses_same_host_with_ui_port() -> None:
    assert build_alp_web_ui_url("http://localhost:5555") == "http://localhost:5432/"
    assert build_alp_web_ui_url("192.168.1.50:7000/api") == "http://192.168.1.50:5432/"


def test_build_alp_imager_video_url_uses_same_host_with_imager_port() -> None:
    assert build_alp_imager_video_url("http://localhost:5555", 1) == "http://localhost:7556/1/vid"
    assert build_alp_imager_video_url("192.168.1.50:7000/api", 2) == "http://192.168.1.50:7556/2/vid"


def test_build_alp_raw_stats_url_uses_same_host_with_imager_port() -> None:
    assert build_alp_raw_stats_url("http://localhost:5555", 1) == "http://localhost:7556/1/raw-stats"
    assert build_alp_raw_stats_url("192.168.1.50:7000/api", 2) == "http://192.168.1.50:7556/2/raw-stats"


def test_measure_preview_brightness_reads_center_crop_fraction() -> None:
    image = np.zeros((100, 100), dtype=np.uint8)
    image[20:80, 20:80] = 128
    pil_image = Image.fromarray(image, mode="L")
    buf = BytesIO()
    pil_image.save(buf, format="JPEG", quality=100)

    stats = measure_preview_brightness(buf.getvalue(), crop_fraction=0.5)

    assert 0.45 <= stats["mean_fraction"] <= 0.55
    assert 110.0 <= stats["estimated_mean_8bit"] <= 140.0


def test_run_smart_sky_flats_converges_and_can_trigger_flats() -> None:
    class _FakeClient:
        def __init__(self) -> None:
            self.live_exposure_ms = 10
            self.actions: list[tuple[str, object]] = []

        def put_action(self, action: str, params: object) -> dict[str, object]:
            self.actions.append((action, params))
            return {"Value": {"ok": True}}

        def begin_streaming(self) -> dict[str, object]:
            self.actions.append(("begin_streaming", {}))
            return {"ok": True}

        def stop_streaming(self) -> dict[str, object]:
            self.actions.append(("stop_streaming", {}))
            return {"ok": True}

        def set_live_exposure_ms(self, exposure_ms: int) -> dict[str, object]:
            self.live_exposure_ms = exposure_ms
            self.actions.append(("set_live_exposure_ms", exposure_ms))
            return {"ok": True}

        def get_live_exposure_ms(self) -> int:
            return self.live_exposure_ms

        def start_create_calib_frame(self) -> dict[str, object]:
            self.actions.append(("start_create_calib_frame", {}))
            return {"ok": True}

    class _FakeImager:
        def __init__(self, client: _FakeClient) -> None:
            self.client = client

        def measure_frame_brightness(self, *, crop_fraction: float) -> dict[str, float]:
            del crop_fraction
            fraction = min(1.0, self.client.live_exposure_ms / 100.0)
            level = fraction * 255.0
            return {
                "mean_fraction": fraction,
                "median_fraction": fraction,
                "estimated_mean_8bit": level,
                "estimated_median_8bit": level,
            }

    client = _FakeClient()
    imager = _FakeImager(client)
    report = run_smart_sky_flats(
        client,
        imager,
        SeestarSmartFlatsConfig(
            target_fraction=0.5,
            tolerance_fraction=0.05,
            min_exposure_ms=5,
            max_exposure_ms=200,
            starting_exposure_ms=20,
            samples_per_step=1,
            max_iterations=5,
            settle_s=0.0,
            linearity_tolerance_fraction=0.10,
            trigger_flat_capture_when_ready=True,
        ),
        lp_filter_mode=SEESTAR_ALP_LP_FILTER_OFF,
        sleep_fn=lambda _seconds: None,
    )

    assert report.success is True
    assert report.ready_for_flats is True
    assert report.auto_triggered is True
    assert 45 <= report.final_exposure_ms <= 55
    assert any(action == "start_create_calib_frame" for action, _ in client.actions)


def test_analyze_raw_linearity_returns_zero_for_linear_samples() -> None:
    measurements = [
        SeestarRawFlatMeasurement(
            exposure_ms=20,
            mean_adu=4000.0,
            median_adu=4000.0,
            p95_adu=4300.0,
            p99_adu=4400.0,
        ),
        SeestarRawFlatMeasurement(
            exposure_ms=40,
            mean_adu=8000.0,
            median_adu=8000.0,
            p95_adu=8300.0,
            p99_adu=8400.0,
        ),
        SeestarRawFlatMeasurement(
            exposure_ms=80,
            mean_adu=16000.0,
            median_adu=16000.0,
            p95_adu=16300.0,
            p99_adu=16400.0,
        ),
    ]

    assert analyze_raw_linearity(measurements) == 0.0


def test_run_raw_adu_smart_flats_converges_and_can_trigger_flats() -> None:
    class _FakeClient:
        def __init__(self) -> None:
            self.live_exposure_ms = 10
            self.actions: list[tuple[str, object]] = []

        def put_action(self, action: str, params: object) -> dict[str, object]:
            self.actions.append((action, params))
            return {"Value": {"ok": True}}

        def begin_streaming(self) -> dict[str, object]:
            self.actions.append(("begin_streaming", {}))
            return {"ok": True}

        def stop_streaming(self) -> dict[str, object]:
            self.actions.append(("stop_streaming", {}))
            return {"ok": True}

        def set_live_exposure_ms(self, exposure_ms: int) -> dict[str, object]:
            self.live_exposure_ms = exposure_ms
            self.actions.append(("set_live_exposure_ms", exposure_ms))
            return {"ok": True}

        def get_live_exposure_ms(self) -> int:
            return self.live_exposure_ms

        def start_create_calib_frame(self) -> dict[str, object]:
            self.actions.append(("start_create_calib_frame", {}))
            return {"ok": True}

    class _FakeRawImager:
        def __init__(self, client: _FakeClient) -> None:
            self.client = client

        def get_raw_stats(self, *, crop_fraction: float, sample_count: int, mode: str) -> SeestarRawFlatMeasurement:
            del crop_fraction, sample_count, mode
            exp = float(self.client.live_exposure_ms)
            median_adu = exp * 200.0
            return SeestarRawFlatMeasurement(
                exposure_ms=int(exp),
                bit_depth=16,
                max_adu=40000,
                mean_adu=median_adu,
                median_adu=median_adu,
                p95_adu=median_adu * 1.05,
                p99_adu=median_adu * 1.10,
                saturated_fraction=0.0,
                quadrant_medians=[median_adu] * 4,
                quadrant_spread_fraction=0.02,
                gain=80,
                filter_position=1,
                frame_width=1920,
                frame_height=1080,
                sample_count=1,
            )

    client = _FakeClient()
    imager = _FakeRawImager(client)
    report = run_raw_adu_smart_flats(
        client,
        imager,
        SeestarSmartFlatsConfig(
            target_fraction=0.5,
            tolerance_fraction=0.05,
            min_exposure_ms=5,
            max_exposure_ms=200,
            starting_exposure_ms=20,
            samples_per_step=1,
            max_iterations=5,
            settle_s=0.0,
            linearity_tolerance_fraction=0.20,
            saturation_tolerance_fraction=0.001,
            quadrant_tolerance_fraction=0.12,
            trigger_flat_capture_when_ready=True,
        ),
        lp_filter_mode=SEESTAR_ALP_LP_FILTER_OFF,
        sleep_fn=lambda _seconds: None,
    )

    assert report.success is True
    assert report.ready_for_flats is True
    assert report.auto_triggered is True
    assert 95 <= report.final_exposure_ms <= 105
    assert abs(report.final_percent_of_full_scale - 0.5) <= 0.05
    assert any(action == "start_create_calib_frame" for action, _ in client.actions)


def test_run_raw_adu_smart_flats_rejects_saturation() -> None:
    class _FakeClient:
        def __init__(self) -> None:
            self.live_exposure_ms = 100

        def put_action(self, _action: str, _params: object) -> dict[str, object]:
            return {"Value": {"ok": True}}

        def begin_streaming(self) -> dict[str, object]:
            return {"ok": True}

        def stop_streaming(self) -> dict[str, object]:
            return {"ok": True}

        def set_live_exposure_ms(self, exposure_ms: int) -> dict[str, object]:
            self.live_exposure_ms = exposure_ms
            return {"ok": True}

        def get_live_exposure_ms(self) -> int:
            return self.live_exposure_ms

        def start_create_calib_frame(self) -> dict[str, object]:
            return {"ok": True}

    class _SaturatedImager:
        def __init__(self, client: _FakeClient) -> None:
            self.client = client

        def get_raw_stats(self, *, crop_fraction: float, sample_count: int, mode: str) -> SeestarRawFlatMeasurement:
            del crop_fraction, sample_count, mode
            return SeestarRawFlatMeasurement(
                exposure_ms=int(self.client.live_exposure_ms),
                bit_depth=16,
                max_adu=40000,
                mean_adu=20000.0,
                median_adu=20000.0,
                p95_adu=39500.0,
                p99_adu=39900.0,
                saturated_fraction=0.02,
                quadrant_medians=[20000.0] * 4,
                quadrant_spread_fraction=0.02,
            )

    report = run_raw_adu_smart_flats(
        _FakeClient(),
        _SaturatedImager(_FakeClient()),
        SeestarSmartFlatsConfig(
            target_fraction=0.5,
            tolerance_fraction=0.05,
            min_exposure_ms=5,
            max_exposure_ms=200,
            starting_exposure_ms=100,
            samples_per_step=1,
            max_iterations=1,
            settle_s=0.0,
            linearity_tolerance_fraction=0.20,
            saturation_tolerance_fraction=0.001,
            quadrant_tolerance_fraction=0.12,
        ),
        sleep_fn=lambda _seconds: None,
    )

    assert report.ready_for_flats is False
    assert any("Saturated fraction" in warning for warning in report.warnings)
