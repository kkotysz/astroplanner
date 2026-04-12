from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from io import BytesIO
import json
import math
import os
import re
import time
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from astroplanner.i18n import current_language, localize_widget_tree, translate_text
from astroplanner.qt_helpers import configure_tab_widget


SEESTAR_METHOD_GUIDED = "guided"
SEESTAR_METHOD_ALP = "alp_service"
SEESTAR_DEFAULT_BLOCK_MINUTES = 60
SEESTAR_MIN_BLOCK_MINUTES = 20
SEESTAR_MAX_TARGETS_PER_NIGHT = 4
SEESTAR_MODE_DSO = "dso"
SEESTAR_DEVICE_PROFILE = "seestar_s50_guided"
SEESTAR_DEVICE_PROFILE_ALP = "seestar_alp_service"
SEESTAR_SOURCE_PLAN_VERSION = "astroplanner-seestar-v1"
SEESTAR_ALP_DEFAULT_BASE_URL = "http://localhost:5555"
SEESTAR_ALP_DEFAULT_DEVICE_NUM = 1
SEESTAR_ALP_DEFAULT_CLIENT_ID = 1
SEESTAR_ALP_DEFAULT_TIMEOUT_S = 8.0
SEESTAR_ALP_DEFAULT_GAIN = 80
SEESTAR_ALP_DEFAULT_PANEL_OVERLAP_PERCENT = 20
SEESTAR_ALP_DEFAULT_USE_AUTOFOCUS = False
SEESTAR_ALP_DEFAULT_NUM_TRIES = 1
SEESTAR_ALP_DEFAULT_RETRY_WAIT_S = 300
SEESTAR_ALP_DEFAULT_TARGET_INTEGRATION_OVERRIDE_MIN = 0
SEESTAR_ALP_DEFAULT_STACK_EXPOSURE_MS = 0
SEESTAR_ALP_DEFAULT_HONOR_QUEUE_TIMES = True
SEESTAR_ALP_DEFAULT_WAIT_UNTIL_LOCAL_TIME = ""
SEESTAR_ALP_IMMEDIATE_START_WAIT_GRACE_S = 60
SEESTAR_ALP_DEFAULT_STARTUP_SEQUENCE = False
SEESTAR_ALP_DEFAULT_STARTUP_POLAR_ALIGN = False
SEESTAR_ALP_DEFAULT_STARTUP_AUTO_FOCUS = False
SEESTAR_ALP_DEFAULT_STARTUP_DARK_FRAMES = False
SEESTAR_ALP_DEFAULT_CAPTURE_FLATS = False
SEESTAR_ALP_DEFAULT_FLATS_WAIT_S = 180
SEESTAR_ALP_DEFAULT_SCHEDULE_AUTOFOCUS = False
SEESTAR_ALP_DEFAULT_AUTOFOCUS_TRY_COUNT = 1
SEESTAR_ALP_AF_MODE_OFF = "off"
SEESTAR_ALP_AF_MODE_PER_RUN = "per_run"
SEESTAR_ALP_AF_MODE_PER_TARGET = "per_target"
SEESTAR_ALP_DEFAULT_SCHEDULE_AUTOFOCUS_MODE = SEESTAR_ALP_AF_MODE_OFF
SEESTAR_ALP_DEFAULT_DEW_HEATER_VALUE = -1
SEESTAR_ALP_DEFAULT_PARK_AFTER_SESSION = False
SEESTAR_ALP_DEFAULT_SHUTDOWN_AFTER_SESSION = False
SEESTAR_SMART_FLATS_DEFAULT_TARGET_FRACTION = 0.50
SEESTAR_SMART_FLATS_DEFAULT_TOLERANCE_FRACTION = 0.08
SEESTAR_SMART_FLATS_DEFAULT_CROP_FRACTION = 0.70
SEESTAR_SMART_FLATS_DEFAULT_MIN_EXPOSURE_MS = 5
SEESTAR_SMART_FLATS_DEFAULT_MAX_EXPOSURE_MS = 200
SEESTAR_SMART_FLATS_DEFAULT_STARTING_EXPOSURE_MS = 30
SEESTAR_SMART_FLATS_DEFAULT_SETTLE_S = 1.0
SEESTAR_SMART_FLATS_DEFAULT_SAMPLES_PER_STEP = 2
SEESTAR_SMART_FLATS_DEFAULT_MAX_ITERATIONS = 6
SEESTAR_SMART_FLATS_DEFAULT_LINEARITY_TOLERANCE = 0.25
SEESTAR_SMART_FLATS_DEFAULT_RAW_SATURATION_TOLERANCE = 0.001
SEESTAR_SMART_FLATS_DEFAULT_RAW_QUADRANT_TOLERANCE = 0.12
SEESTAR_ALP_LP_FILTER_AUTO = "auto"
SEESTAR_ALP_LP_FILTER_OFF = "off"
SEESTAR_ALP_LP_FILTER_ON = "on"
SEESTAR_CAMPAIGN_PRESET_BL_LAC = "bl_lac_campaign"
SEESTAR_TEMPLATE_SCOPE_MULTI_TARGET = "multi_target"
SEESTAR_TEMPLATE_SCOPE_SINGLE_TARGET = "single_target"
SEESTAR_DEBUG_ENV = "ASTROPLANNER_SEESTAR_DEBUG"


class NightQueueSiteSnapshot(BaseModel):
    name: str = ""
    latitude: float
    longitude: float
    elevation: float = 0.0
    timezone: str = "UTC"
    limiting_magnitude: float = 0.0
    telescope_diameter_mm: float = 0.0
    focal_length_mm: float = 0.0
    pixel_size_um: float = 0.0
    detector_width_px: int = 0
    detector_height_px: int = 0


class NightQueueBlock(BaseModel):
    alias: str
    target_name: str
    ra_deg: float
    dec_deg: float
    ra_hms: str
    dec_dms: str
    start_local: datetime
    end_local: datetime
    score: float
    hours_above_limit: float
    recommended_filter: str
    window_start_local: datetime
    window_end_local: datetime
    notes: str = ""
    lp_filter_mode: str = SEESTAR_ALP_LP_FILTER_AUTO
    stack_exposure_ms: int = 0
    gain: int = -1
    use_autofocus: Optional[bool] = None
    repeat_index: int = 1
    repeat_count: int = 1
    target_order: int = 0

    model_config = ConfigDict(arbitrary_types_allowed=True)


class NightQueue(BaseModel):
    site_snapshot: NightQueueSiteSnapshot
    night_start_local: datetime
    night_end_local: datetime
    timezone: str
    device_profile: str = SEESTAR_DEVICE_PROFILE
    blocks: list[NightQueueBlock] = Field(default_factory=list)
    generated_at: datetime
    source_plan_version: str = SEESTAR_SOURCE_PLAN_VERSION
    campaign_name: str = ""
    require_science_checklist: bool = False
    science_checklist_items: list[str] = Field(default_factory=list)
    campaign_notes: str = ""

    model_config = ConfigDict(arbitrary_types_allowed=True)


class SeestarHandoffBundle(BaseModel):
    queue: NightQueue
    session_payload: dict[str, Any]
    csv_rows: list[dict[str, Any]]
    dialog_text: str
    checklist_markdown: str


class SeestarAlpConfig(BaseModel):
    base_url: str = SEESTAR_ALP_DEFAULT_BASE_URL
    device_num: int = SEESTAR_ALP_DEFAULT_DEVICE_NUM
    client_id: int = SEESTAR_ALP_DEFAULT_CLIENT_ID
    timeout_s: float = SEESTAR_ALP_DEFAULT_TIMEOUT_S
    gain: int = SEESTAR_ALP_DEFAULT_GAIN
    panel_overlap_percent: int = SEESTAR_ALP_DEFAULT_PANEL_OVERLAP_PERCENT
    use_autofocus: bool = SEESTAR_ALP_DEFAULT_USE_AUTOFOCUS
    num_tries: int = SEESTAR_ALP_DEFAULT_NUM_TRIES
    retry_wait_s: int = SEESTAR_ALP_DEFAULT_RETRY_WAIT_S
    target_integration_override_min: int = SEESTAR_ALP_DEFAULT_TARGET_INTEGRATION_OVERRIDE_MIN
    stack_exposure_ms: int = SEESTAR_ALP_DEFAULT_STACK_EXPOSURE_MS
    lp_filter_mode: str = SEESTAR_ALP_LP_FILTER_AUTO
    honor_queue_times: bool = SEESTAR_ALP_DEFAULT_HONOR_QUEUE_TIMES
    wait_until_local_time: str = SEESTAR_ALP_DEFAULT_WAIT_UNTIL_LOCAL_TIME
    startup_enabled: bool = SEESTAR_ALP_DEFAULT_STARTUP_SEQUENCE
    startup_polar_align: bool = SEESTAR_ALP_DEFAULT_STARTUP_POLAR_ALIGN
    startup_auto_focus: bool = SEESTAR_ALP_DEFAULT_STARTUP_AUTO_FOCUS
    startup_dark_frames: bool = SEESTAR_ALP_DEFAULT_STARTUP_DARK_FRAMES
    capture_flats_before_session: bool = SEESTAR_ALP_DEFAULT_CAPTURE_FLATS
    flats_wait_s: int = SEESTAR_ALP_DEFAULT_FLATS_WAIT_S
    schedule_autofocus_mode: str = SEESTAR_ALP_DEFAULT_SCHEDULE_AUTOFOCUS_MODE
    schedule_autofocus_before_each_target: bool = SEESTAR_ALP_DEFAULT_SCHEDULE_AUTOFOCUS
    schedule_autofocus_try_count: int = SEESTAR_ALP_DEFAULT_AUTOFOCUS_TRY_COUNT
    dew_heater_value: int = SEESTAR_ALP_DEFAULT_DEW_HEATER_VALUE
    park_after_session: bool = SEESTAR_ALP_DEFAULT_PARK_AFTER_SESSION
    shutdown_after_session: bool = SEESTAR_ALP_DEFAULT_SHUTDOWN_AFTER_SESSION


class SeestarSessionTemplate(BaseModel):
    key: str = ""
    name: str = ""
    scope: str = SEESTAR_TEMPLATE_SCOPE_MULTI_TARGET
    repeat_count: int = 1
    minutes_per_run: int = SEESTAR_DEFAULT_BLOCK_MINUTES
    gap_seconds: int = 0
    require_science_checklist: bool = False
    science_checklist_items: list[str] = Field(default_factory=list)
    template_notes: str = ""
    lp_filter_mode: str = SEESTAR_ALP_LP_FILTER_AUTO
    gain: int = SEESTAR_ALP_DEFAULT_GAIN
    panel_overlap_percent: int = SEESTAR_ALP_DEFAULT_PANEL_OVERLAP_PERCENT
    use_autofocus: bool = SEESTAR_ALP_DEFAULT_USE_AUTOFOCUS
    num_tries: int = SEESTAR_ALP_DEFAULT_NUM_TRIES
    retry_wait_s: int = SEESTAR_ALP_DEFAULT_RETRY_WAIT_S
    target_integration_override_min: int = SEESTAR_ALP_DEFAULT_TARGET_INTEGRATION_OVERRIDE_MIN
    stack_exposure_ms: int = SEESTAR_ALP_DEFAULT_STACK_EXPOSURE_MS
    honor_queue_times: bool = SEESTAR_ALP_DEFAULT_HONOR_QUEUE_TIMES
    wait_until_local_time: str = SEESTAR_ALP_DEFAULT_WAIT_UNTIL_LOCAL_TIME
    startup_enabled: bool = SEESTAR_ALP_DEFAULT_STARTUP_SEQUENCE
    startup_polar_align: bool = SEESTAR_ALP_DEFAULT_STARTUP_POLAR_ALIGN
    startup_auto_focus: bool = SEESTAR_ALP_DEFAULT_STARTUP_AUTO_FOCUS
    startup_dark_frames: bool = SEESTAR_ALP_DEFAULT_STARTUP_DARK_FRAMES
    capture_flats_before_session: bool = SEESTAR_ALP_DEFAULT_CAPTURE_FLATS
    flats_wait_s: int = SEESTAR_ALP_DEFAULT_FLATS_WAIT_S
    schedule_autofocus_mode: str = SEESTAR_ALP_DEFAULT_SCHEDULE_AUTOFOCUS_MODE
    schedule_autofocus_before_each_target: bool = SEESTAR_ALP_DEFAULT_SCHEDULE_AUTOFOCUS
    schedule_autofocus_try_count: int = SEESTAR_ALP_DEFAULT_AUTOFOCUS_TRY_COUNT
    dew_heater_value: int = SEESTAR_ALP_DEFAULT_DEW_HEATER_VALUE
    park_after_session: bool = SEESTAR_ALP_DEFAULT_PARK_AFTER_SESSION
    shutdown_after_session: bool = SEESTAR_ALP_DEFAULT_SHUTDOWN_AFTER_SESSION


def normalize_seestar_alp_schedule_autofocus_mode(
    value: object,
    *,
    legacy_enabled: bool = False,
) -> str:
    mode = str(value or "").strip().lower()
    if bool(legacy_enabled) and mode in {"", SEESTAR_ALP_AF_MODE_OFF}:
        return SEESTAR_ALP_AF_MODE_PER_RUN
    if mode in {
        SEESTAR_ALP_AF_MODE_OFF,
        SEESTAR_ALP_AF_MODE_PER_RUN,
        SEESTAR_ALP_AF_MODE_PER_TARGET,
    }:
        return mode
    return SEESTAR_ALP_AF_MODE_PER_RUN if bool(legacy_enabled) else SEESTAR_ALP_AF_MODE_OFF


def seestar_alp_schedule_autofocus_mode_label(value: object, *, short: bool = False) -> str:
    mode = normalize_seestar_alp_schedule_autofocus_mode(value)
    if mode == SEESTAR_ALP_AF_MODE_PER_RUN:
        return "per run" if short else "Before each run"
    if mode == SEESTAR_ALP_AF_MODE_PER_TARGET:
        return "per target" if short else "Once before each target"
    return "off" if short else "Off"

class SeestarAlpBackendStatus(BaseModel):
    base_url: str
    device_num: int
    connected: bool
    device_name: str = ""
    schedule_state: str = "unavailable"
    schedule_id: str = ""
    queued_items: int = 0
    current_item_id: str = ""
    supports_raw_flats: bool = False


class SeestarSmartFlatMeasurement(BaseModel):
    exposure_ms: int
    mean_fraction: float
    median_fraction: float
    estimated_mean_8bit: float
    estimated_median_8bit: float


class SeestarSmartFlatsConfig(BaseModel):
    target_fraction: float = SEESTAR_SMART_FLATS_DEFAULT_TARGET_FRACTION
    tolerance_fraction: float = SEESTAR_SMART_FLATS_DEFAULT_TOLERANCE_FRACTION
    crop_fraction: float = SEESTAR_SMART_FLATS_DEFAULT_CROP_FRACTION
    min_exposure_ms: int = SEESTAR_SMART_FLATS_DEFAULT_MIN_EXPOSURE_MS
    max_exposure_ms: int = SEESTAR_SMART_FLATS_DEFAULT_MAX_EXPOSURE_MS
    starting_exposure_ms: int = SEESTAR_SMART_FLATS_DEFAULT_STARTING_EXPOSURE_MS
    settle_s: float = SEESTAR_SMART_FLATS_DEFAULT_SETTLE_S
    samples_per_step: int = SEESTAR_SMART_FLATS_DEFAULT_SAMPLES_PER_STEP
    max_iterations: int = SEESTAR_SMART_FLATS_DEFAULT_MAX_ITERATIONS
    linearity_tolerance_fraction: float = SEESTAR_SMART_FLATS_DEFAULT_LINEARITY_TOLERANCE
    saturation_tolerance_fraction: float = SEESTAR_SMART_FLATS_DEFAULT_RAW_SATURATION_TOLERANCE
    quadrant_tolerance_fraction: float = SEESTAR_SMART_FLATS_DEFAULT_RAW_QUADRANT_TOLERANCE
    trigger_flat_capture_when_ready: bool = False


class SeestarSmartFlatsReport(BaseModel):
    success: bool
    ready_for_flats: bool
    auto_triggered: bool = False
    final_exposure_ms: int = 0
    final_mean_fraction: float = 0.0
    final_median_fraction: float = 0.0
    linearity_spread_fraction: float = 0.0
    measurements: list[SeestarSmartFlatMeasurement] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class SeestarRawFlatMeasurement(BaseModel):
    exposure_ms: int
    bit_depth: int = 16
    max_adu: int = 65535
    mean_adu: float
    median_adu: float
    p95_adu: float
    p99_adu: float
    stddev_adu: float = 0.0
    saturated_fraction: float = 0.0
    quadrant_medians: list[float] = Field(default_factory=list)
    quadrant_spread_fraction: float = 0.0
    gain: int = 0
    filter_position: int = 0
    timestamp: str = ""
    frame_width: int = 0
    frame_height: int = 0
    sample_count: int = 1


class SeestarRawFlatsReport(BaseModel):
    success: bool
    ready_for_flats: bool
    auto_triggered: bool = False
    final_exposure_ms: int = 0
    final_bit_depth: int = 0
    final_max_adu: int = 0
    final_target_adu: float = 0.0
    final_mean_adu: float = 0.0
    final_median_adu: float = 0.0
    final_percent_of_full_scale: float = 0.0
    final_saturated_fraction: float = 0.0
    final_quadrant_spread_fraction: float = 0.0
    linearity_spread_fraction: float = 0.0
    measurements: list[SeestarRawFlatMeasurement] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class NightQueueCandidate:
    target_name: str
    ra_deg: float
    dec_deg: float
    object_type: str
    score: float
    hours_above_limit: float
    max_altitude_deg: float
    window_start_local: datetime
    window_end_local: datetime
    notes: str = ""


class SeestarTargetSessionItem(BaseModel):
    enabled: bool = True
    order: int = 0
    target_name: str
    ra_deg: float
    dec_deg: float
    object_type: str = ""
    notes: str = ""
    score: float = 0.0
    hours_above_limit: float = 0.0
    max_altitude_deg: float = 0.0
    window_start_local: Optional[datetime] = None
    window_end_local: Optional[datetime] = None
    repeat_count: Optional[int] = None
    segment_minutes: Optional[int] = None
    gap_seconds: Optional[int] = None
    stack_exposure_ms: Optional[int] = None
    lp_filter_mode: Optional[str] = None
    gain: Optional[int] = None
    autofocus: Optional[bool] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


def _normalize_type_token(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"[_/]+", " ", text)
    return " ".join(text.split())


def is_seestar_debug_enabled() -> bool:
    return str(os.environ.get(SEESTAR_DEBUG_ENV, "")).strip().lower() in {"1", "true", "yes", "on"}


def default_science_checklist_items() -> list[str]:
    return [
        "Telescope leveled and calibrated; EQ mode preferred when available.",
        "Flats captured for the current setup and the telescope was not moved roughly after flats.",
        "LP filter set as intended for the campaign before capture starts.",
        "Save each frame is enabled.",
        "No in-app Deep Sky Stack editing or AI Denoise will be applied after capture.",
    ]


def _coerce_session_template_payload(item: dict[str, Any]) -> dict[str, Any]:
    payload = dict(item)
    if "minutes_per_run" not in payload:
        payload["minutes_per_run"] = payload.get("segment_minutes", SEESTAR_DEFAULT_BLOCK_MINUTES)
    if "template_notes" not in payload:
        payload["template_notes"] = payload.get("campaign_notes", "")
    if "scope" not in payload:
        payload["scope"] = SEESTAR_TEMPLATE_SCOPE_MULTI_TARGET
    if "schedule_autofocus_mode" not in payload:
        payload["schedule_autofocus_mode"] = normalize_seestar_alp_schedule_autofocus_mode(
            payload.get("schedule_autofocus_mode"),
            legacy_enabled=bool(payload.get("schedule_autofocus_before_each_target", False)),
        )
    payload.pop("source", None)
    payload.pop("target_name", None)
    payload.pop("target_ra", None)
    payload.pop("target_dec", None)
    payload.pop("target_object_type", None)
    payload.pop("target_notes", None)
    return payload


def builtin_seestar_session_templates() -> dict[str, SeestarSessionTemplate]:
    bl_lac_items = [
        "Level and calibrate the telescope before the session. EQ mode is preferred.",
        "Capture flats in Image Calibration -> Flat Shoot and avoid moving the telescope after flats.",
        "Make sure exactly one BL Lac target is enabled in the session target table.",
        "LP Filter must be OFF for the science run.",
        "Save each frame must be enabled.",
        "Use 10 s stack exposure and keep each session at 10 minutes.",
        "Do not use AI Denoise or in-app stack editing after capture.",
    ]
    preset = SeestarSessionTemplate(
        key=SEESTAR_CAMPAIGN_PRESET_BL_LAC,
        name="BL Lac campaign",
        scope=SEESTAR_TEMPLATE_SCOPE_SINGLE_TARGET,
        repeat_count=6,
        minutes_per_run=10,
        gap_seconds=60,
        require_science_checklist=True,
        science_checklist_items=bl_lac_items,
        template_notes=(
            "Use this template with a single BL Lac target already present in the AstroPlanner Targets list. "
            "It is tuned for six separate 10-minute runs with 10 s sub-exposures."
        ),
        lp_filter_mode=SEESTAR_ALP_LP_FILTER_OFF,
        gain=SEESTAR_ALP_DEFAULT_GAIN,
        panel_overlap_percent=SEESTAR_ALP_DEFAULT_PANEL_OVERLAP_PERCENT,
        use_autofocus=False,
        num_tries=1,
        retry_wait_s=SEESTAR_ALP_DEFAULT_RETRY_WAIT_S,
        target_integration_override_min=0,
        stack_exposure_ms=10000,
        honor_queue_times=True,
        wait_until_local_time="",
        startup_enabled=False,
        startup_polar_align=False,
        startup_auto_focus=False,
        startup_dark_frames=False,
        capture_flats_before_session=False,
        flats_wait_s=SEESTAR_ALP_DEFAULT_FLATS_WAIT_S,
        schedule_autofocus_mode=SEESTAR_ALP_AF_MODE_OFF,
        schedule_autofocus_before_each_target=False,
        schedule_autofocus_try_count=SEESTAR_ALP_DEFAULT_AUTOFOCUS_TRY_COUNT,
        dew_heater_value=SEESTAR_ALP_DEFAULT_DEW_HEATER_VALUE,
        park_after_session=True,
        shutdown_after_session=False,
    )
    return {preset.key: preset}


def load_user_seestar_session_templates(raw_value: object) -> dict[str, SeestarSessionTemplate]:
    if raw_value in (None, "", b""):
        return {}
    payload = raw_value
    if isinstance(raw_value, str):
        try:
            payload = json.loads(raw_value)
        except Exception:
            return {}
    if not isinstance(payload, list):
        return {}
    loaded: dict[str, SeestarSessionTemplate] = {}
    for item in payload:
        if not isinstance(item, dict):
            continue
        try:
            preset = SeestarSessionTemplate.model_validate(_coerce_session_template_payload(item))
        except Exception:
            continue
        key = _slugify(preset.key or preset.name or "preset")
        loaded[key] = preset.model_copy(update={"key": key})
    return loaded


def dump_user_seestar_session_templates(presets: dict[str, SeestarSessionTemplate]) -> str:
    serializable = []
    for key in sorted(presets.keys()):
        preset = presets[key]
        serializable.append(preset.model_copy(update={"key": _slugify(key or preset.key or preset.name)}).model_dump(mode="json"))
    return json.dumps(serializable, ensure_ascii=False, indent=2)


def _slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value or ""))
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_text).strip("-").lower()
    return slug or "target"


def _format_ra_hms(ra_deg: float) -> str:
    total_seconds = ((float(ra_deg) % 360.0) / 15.0) * 3600.0
    hours = int(total_seconds // 3600) % 24
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(round(total_seconds % 60))
    if seconds == 60:
        seconds = 0
        minutes += 1
    if minutes == 60:
        minutes = 0
        hours = (hours + 1) % 24
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _format_dec_dms(dec_deg: float) -> str:
    sign = "+" if float(dec_deg) >= 0 else "-"
    abs_deg = abs(float(dec_deg))
    total_seconds = abs_deg * 3600.0
    degrees = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(round(total_seconds % 60))
    if seconds == 60:
        seconds = 0
        minutes += 1
    if minutes == 60:
        minutes = 0
        degrees += 1
    return f"{sign}{degrees:02d}:{minutes:02d}:{seconds:02d}"


def is_supported_dso_type(object_type: str) -> bool:
    token = _normalize_type_token(object_type)
    if not token:
        return True
    unsupported_phrases = (
        "sun",
        "solar system",
        "moon",
        "planet",
        "mercury",
        "venus",
        "mars",
        "jupiter",
        "saturn",
        "uranus",
        "neptune",
        "pluto",
        "asteroid",
        "comet",
        "satellite",
        "iss",
        "scenery",
        "landscape",
    )
    return not any(re.search(rf"\b{re.escape(part)}\b", token) for part in unsupported_phrases)


def recommended_filter_for_object_type(object_type: str) -> str:
    token = _normalize_type_token(object_type)
    if not token:
        return "auto"
    duoband_tokens = (
        "nebula",
        "planetary nebula",
        "emission nebula",
        "supernova remnant",
        "snr",
        "h ii",
        "hii",
        "h-ii",
    )
    if any(part in token for part in duoband_tokens):
        return "duoband"
    uv_ir_tokens = (
        "galaxy",
        "cluster",
        "globular",
        "open cluster",
        "star",
        "reflection nebula",
        "qso",
        "quasar",
        "blazar",
        "bl lac",
        "agn",
        "active galactic nucleus",
        "binary",
        "eclipsing binary",
        "variable star",
        "cataclysmic",
        "nova",
        "supernova",
    )
    if any(part in token for part in uv_ir_tokens):
        return "uv_ir"
    return "auto"


def build_night_queue(
    candidates: Iterable[NightQueueCandidate],
    *,
    site_snapshot: NightQueueSiteSnapshot,
    night_start_local: datetime,
    night_end_local: datetime,
    timezone: str,
    default_block_minutes: int = SEESTAR_DEFAULT_BLOCK_MINUTES,
    min_block_minutes: int = SEESTAR_MIN_BLOCK_MINUTES,
    max_targets_per_night: int = SEESTAR_MAX_TARGETS_PER_NIGHT,
    device_profile: str = SEESTAR_DEVICE_PROFILE,
    generated_at: Optional[datetime] = None,
    source_plan_version: str = SEESTAR_SOURCE_PLAN_VERSION,
) -> NightQueue:
    if night_end_local <= night_start_local:
        raise ValueError("Night end must be later than night start.")

    default_block_minutes = max(5, int(default_block_minutes))
    min_block_minutes = max(5, min(int(min_block_minutes), default_block_minutes))
    max_targets_per_night = max(1, int(max_targets_per_night))
    min_block_delta = timedelta(minutes=min_block_minutes)
    default_block_delta = timedelta(minutes=default_block_minutes)

    usable_candidates = [
        candidate
        for candidate in candidates
        if candidate.score > 0.0
        and candidate.window_end_local > candidate.window_start_local
        and is_supported_dso_type(candidate.object_type)
    ]

    cursor = night_start_local
    blocks: list[NightQueueBlock] = []
    queue_date = night_start_local.strftime("%Y%m%d")
    pending = list(usable_candidates)

    while pending and len(blocks) < max_targets_per_night and cursor < night_end_local:
        viable: list[tuple[NightQueueCandidate, datetime]] = []
        next_window_start: Optional[datetime] = None

        for candidate in pending:
            candidate_start = max(cursor, night_start_local, candidate.window_start_local)
            candidate_end = min(candidate.window_end_local, night_end_local)
            if candidate_end - candidate_start >= min_block_delta:
                viable.append((candidate, candidate_start))
            elif candidate.window_start_local > cursor:
                if next_window_start is None or candidate.window_start_local < next_window_start:
                    next_window_start = candidate.window_start_local

        if not viable:
            if next_window_start is None or next_window_start >= night_end_local:
                break
            cursor = next_window_start
            continue

        viable.sort(
            key=lambda item: (
                -float(item[0].score),
                item[0].window_end_local,
                -float(item[0].max_altitude_deg),
                item[0].window_start_local,
                item[0].target_name.lower(),
            )
        )
        selected, block_start = viable[0]
        block_end = min(block_start + default_block_delta, selected.window_end_local, night_end_local)
        if block_end - block_start < min_block_delta:
            pending = [candidate for candidate in pending if candidate is not selected]
            continue

        alias = f"AP-{queue_date}-{len(blocks) + 1:02d}-{_slugify(selected.target_name)}"
        blocks.append(
            NightQueueBlock(
                alias=alias,
                target_name=selected.target_name,
                ra_deg=round(float(selected.ra_deg), 6),
                dec_deg=round(float(selected.dec_deg), 6),
                ra_hms=_format_ra_hms(selected.ra_deg),
                dec_dms=_format_dec_dms(selected.dec_deg),
                start_local=block_start,
                end_local=block_end,
                score=round(float(selected.score), 1),
                hours_above_limit=round(float(selected.hours_above_limit), 2),
                recommended_filter=recommended_filter_for_object_type(selected.object_type),
                window_start_local=selected.window_start_local,
                window_end_local=selected.window_end_local,
                notes=str(selected.notes or "").strip(),
            )
        )
        cursor = block_end
        pending = [candidate for candidate in pending if candidate is not selected]

    generated = generated_at or datetime.now(night_start_local.tzinfo)
    return NightQueue(
        site_snapshot=site_snapshot,
        night_start_local=night_start_local,
        night_end_local=night_end_local,
        timezone=str(timezone or site_snapshot.timezone or "UTC"),
        device_profile=device_profile,
        blocks=blocks,
        generated_at=generated,
        source_plan_version=source_plan_version,
    )


def build_repeated_target_queue(
    candidate: NightQueueCandidate,
    *,
    site_snapshot: NightQueueSiteSnapshot,
    night_start_local: datetime,
    night_end_local: datetime,
    timezone: str,
    repeat_count: int,
    segment_minutes: int,
    gap_seconds: int = 0,
    device_profile: str = SEESTAR_DEVICE_PROFILE,
    generated_at: Optional[datetime] = None,
    source_plan_version: str = SEESTAR_SOURCE_PLAN_VERSION,
) -> NightQueue:
    repeat_count = max(1, int(repeat_count))
    segment_minutes = max(1, int(segment_minutes))
    gap_seconds = max(0, int(gap_seconds))
    segment_delta = timedelta(minutes=segment_minutes)
    gap_delta = timedelta(seconds=gap_seconds)
    queue_date = night_start_local.strftime("%Y%m%d")
    cursor = max(night_start_local, candidate.window_start_local)
    hard_end = min(night_end_local, candidate.window_end_local)
    blocks: list[NightQueueBlock] = []

    for idx in range(repeat_count):
        block_start = cursor
        block_end = min(block_start + segment_delta, hard_end)
        if block_end <= block_start:
            break
        alias = f"AP-{queue_date}-{idx + 1:02d}-{_slugify(candidate.target_name)}-s{idx + 1:02d}"
        note_parts = [str(candidate.notes or "").strip(), f"session {idx + 1}/{repeat_count}"]
        blocks.append(
            NightQueueBlock(
                alias=alias,
                target_name=candidate.target_name,
                ra_deg=round(float(candidate.ra_deg), 6),
                dec_deg=round(float(candidate.dec_deg), 6),
                ra_hms=_format_ra_hms(candidate.ra_deg),
                dec_dms=_format_dec_dms(candidate.dec_deg),
                start_local=block_start,
                end_local=block_end,
                score=round(float(candidate.score), 1),
                hours_above_limit=round(float(candidate.hours_above_limit), 2),
                recommended_filter=recommended_filter_for_object_type(candidate.object_type),
                window_start_local=candidate.window_start_local,
                window_end_local=candidate.window_end_local,
                notes=" | ".join(part for part in note_parts if part),
            )
        )
        cursor = block_end + gap_delta
        if cursor >= hard_end:
            break

    generated = generated_at or datetime.now(night_start_local.tzinfo)
    effective_start = blocks[0].start_local if blocks else cursor
    effective_end = blocks[-1].end_local if blocks else min(cursor + segment_delta, hard_end)
    return NightQueue(
        site_snapshot=site_snapshot,
        night_start_local=effective_start,
        night_end_local=effective_end,
        timezone=str(timezone or site_snapshot.timezone or "UTC"),
        device_profile=device_profile,
        blocks=blocks,
        generated_at=generated,
        source_plan_version=source_plan_version,
    )


def _resolved_session_item_int(override_value: Optional[int], default_value: int, *, minimum: int = 0) -> int:
    if override_value is None:
        return max(minimum, int(default_value))
    return max(minimum, int(override_value))


def _resolved_session_item_bool(override_value: Optional[bool], default_value: bool) -> bool:
    if override_value is None:
        return bool(default_value)
    return bool(override_value)


def _resolved_session_item_text(override_value: Optional[str], default_value: str) -> str:
    text = str(override_value or "").strip().lower()
    if not text:
        return str(default_value or "").strip().lower()
    return text


def build_session_queue(
    session_items: Iterable[SeestarTargetSessionItem],
    *,
    session_template: SeestarSessionTemplate,
    site_snapshot: NightQueueSiteSnapshot,
    night_start_local: datetime,
    night_end_local: datetime,
    timezone: str,
    device_profile: str = SEESTAR_DEVICE_PROFILE,
    start_cursor_local: Optional[datetime] = None,
    generated_at: Optional[datetime] = None,
    source_plan_version: str = SEESTAR_SOURCE_PLAN_VERSION,
) -> NightQueue:
    if night_end_local <= night_start_local:
        raise ValueError("Night end must be later than night start.")

    enabled_items = [item for item in session_items if bool(item.enabled)]
    blocks: list[NightQueueBlock] = []
    queue_date = night_start_local.strftime("%Y%m%d")
    cursor = night_start_local
    if start_cursor_local is not None:
        cursor = max(cursor, start_cursor_local)
    if cursor > night_end_local:
        cursor = night_end_local

    for session_index, item in enumerate(enabled_items, start=1):
        repeat_count = _resolved_session_item_int(item.repeat_count, session_template.repeat_count, minimum=1)
        segment_minutes = _resolved_session_item_int(
            item.segment_minutes,
            session_template.minutes_per_run,
            minimum=1,
        )
        gap_seconds = _resolved_session_item_int(item.gap_seconds, session_template.gap_seconds, minimum=0)
        lp_filter_mode = _resolved_session_item_text(item.lp_filter_mode, session_template.lp_filter_mode)
        block_gain = _resolved_session_item_int(item.gain, session_template.gain, minimum=0)
        block_stack_exposure = _resolved_session_item_int(
            item.stack_exposure_ms,
            session_template.stack_exposure_ms,
            minimum=0,
        )
        block_use_autofocus = _resolved_session_item_bool(item.autofocus, session_template.use_autofocus)
        block_delta = timedelta(minutes=segment_minutes)
        gap_delta = timedelta(seconds=gap_seconds)
        window_start = item.window_start_local or night_start_local
        window_end = item.window_end_local or max(cursor + block_delta, night_end_local)

        for repeat_index in range(1, repeat_count + 1):
            block_start = cursor
            block_end = block_start + block_delta
            alias_suffix = f"-r{repeat_index:02d}" if repeat_count > 1 else ""
            alias = f"AP-{queue_date}-{len(blocks) + 1:02d}-{_slugify(item.target_name)}{alias_suffix}"
            note_parts = [str(item.notes or "").strip()]
            if repeat_count > 1:
                note_parts.append(f"run {repeat_index}/{repeat_count}")
            blocks.append(
                NightQueueBlock(
                    alias=alias,
                    target_name=item.target_name,
                    ra_deg=round(float(item.ra_deg), 6),
                    dec_deg=round(float(item.dec_deg), 6),
                    ra_hms=_format_ra_hms(item.ra_deg),
                    dec_dms=_format_dec_dms(item.dec_deg),
                    start_local=block_start,
                    end_local=block_end,
                    score=round(float(item.score), 1),
                    hours_above_limit=round(float(item.hours_above_limit), 2),
                    recommended_filter=recommended_filter_for_object_type(item.object_type),
                    window_start_local=window_start,
                    window_end_local=window_end,
                    notes=" | ".join(part for part in note_parts if part),
                    lp_filter_mode=lp_filter_mode or SEESTAR_ALP_LP_FILTER_AUTO,
                    stack_exposure_ms=block_stack_exposure,
                    gain=block_gain,
                    use_autofocus=block_use_autofocus,
                    repeat_index=repeat_index,
                    repeat_count=repeat_count,
                    target_order=session_index,
                )
            )
            cursor = block_end
            if gap_seconds > 0:
                cursor += gap_delta

    generated = generated_at or datetime.now(night_start_local.tzinfo)
    effective_start = blocks[0].start_local if blocks else cursor
    effective_end = blocks[-1].end_local if blocks else night_end_local
    return NightQueue(
        site_snapshot=site_snapshot,
        night_start_local=effective_start,
        night_end_local=effective_end,
        timezone=str(timezone or site_snapshot.timezone or "UTC"),
        device_profile=device_profile,
        blocks=blocks,
        generated_at=generated,
        source_plan_version=source_plan_version,
        campaign_name=str(session_template.name or "").strip(),
        require_science_checklist=bool(session_template.require_science_checklist),
        science_checklist_items=[str(item).strip() for item in session_template.science_checklist_items if str(item).strip()],
        campaign_notes=str(session_template.template_notes or "").strip(),
    )


def _filter_display_name(
    recommended_filter: str,
    *,
    lp_mode: str = SEESTAR_ALP_LP_FILTER_AUTO,
) -> str:
    mode = str(lp_mode or SEESTAR_ALP_LP_FILTER_AUTO).strip().lower()
    if mode == SEESTAR_ALP_LP_FILTER_OFF:
        return "IR-cut"
    if mode == SEESTAR_ALP_LP_FILTER_ON:
        return "LP filter"
    token = str(recommended_filter or "").strip().lower()
    if token == "duoband":
        return "LP filter"
    if token == "uv_ir":
        return "IR-cut"
    return "Auto"


def _block_filter_label(block: NightQueueBlock, config: SeestarAlpConfig | None = None) -> str:
    if config is not None:
        position = _alp_filter_position_for_block(block, config)
        return {1: "IR-cut", 2: "LP filter"}.get(int(position), "Auto")
    lp_mode = str(getattr(block, "lp_filter_mode", "") or "").strip().lower()
    return _filter_display_name(
        str(getattr(block, "recommended_filter", "") or ""),
        lp_mode=lp_mode or SEESTAR_ALP_LP_FILTER_AUTO,
    )


def _queue_summary_text(queue: NightQueue) -> str:
    duration_hours = max(0.0, (queue.night_end_local - queue.night_start_local).total_seconds() / 3600.0)
    summary = (
        f"Seestar guided session for {queue.site_snapshot.name or 'current observatory'}\n"
        f"Night: {queue.night_start_local.strftime('%Y-%m-%d %H:%M')} -> "
        f"{queue.night_end_local.strftime('%Y-%m-%d %H:%M')} ({queue.timezone}, {duration_hours:.1f} h)\n"
        f"Blocks: {len(queue.blocks)} | Device: {queue.device_profile}"
    )
    if str(queue.campaign_name or "").strip():
        summary += f"\nTemplate: {queue.campaign_name.strip()}"
    return summary


def _append_science_checklist(lines: list[str], queue: NightQueue) -> None:
    if not bool(getattr(queue, "require_science_checklist", False)):
        return
    items = [str(item).strip() for item in queue.science_checklist_items if str(item).strip()]
    if not items:
        return
    lines.extend(["", "Science Checklist", "-" * 72])
    for idx, item in enumerate(items, start=1):
        lines.append(f"{idx}. {item}")
    if str(queue.campaign_notes or "").strip():
        lines.append(f"Notes: {queue.campaign_notes.strip()}")


def render_handoff_dialog_text(bundle: SeestarHandoffBundle) -> str:
    queue = bundle.queue
    lines = [_queue_summary_text(queue), ""]
    if not queue.blocks:
        lines.append("No Seestar queue blocks were generated from the current plan.")
        return "\n".join(lines)

    _append_science_checklist(lines, queue)

    lines.append("Queue")
    lines.append("-" * 72)
    for idx, block in enumerate(queue.blocks, start=1):
        lines.append(
            f"{idx}. {block.alias} | {block.target_name} | "
            f"{block.start_local.strftime('%H:%M')} -> {block.end_local.strftime('%H:%M')} | "
            f"filter={_block_filter_label(block)}"
        )
        lines.append(
            f"   RA/Dec {block.ra_hms} {block.dec_dms} | "
            f"window {block.window_start_local.strftime('%H:%M')} -> {block.window_end_local.strftime('%H:%M')} | "
            f"score {block.score:.1f}"
        )
        if block.notes:
            lines.append(f"   Notes: {block.notes}")

    lines.extend(
        [
            "",
            "Custom Celestial Objects",
            "-" * 72,
            "V1 assumes each queued item should be added in Seestar via Custom Celestial Objects",
            "to avoid catalog-name mismatches. Use the alias as the display name.",
        ]
    )
    for block in queue.blocks:
        lines.append(f"- {block.alias}: {block.ra_hms} {block.dec_dms}")

    lines.extend(
        [
            "",
            "Checklist",
            "-" * 72,
            "1. Open the Seestar app and connect to the telescope.",
            "2. Use local control or the official Telescope Network path; the queue order stays the same.",
            "3. Add each queued object through Custom Celestial Objects using the alias and RA/Dec above.",
            "4. Create Plan Mode entries in the exact order shown in Queue.",
            "5. Set each block start/stop time and keep the suggested filter unless you intentionally override it.",
            "6. Start the first plan block and monitor normal Seestar safety checks in the official app.",
        ]
    )
    return "\n".join(lines)


def render_handoff_checklist_markdown(queue: NightQueue) -> str:
    lines = [
        "# Seestar Session Checklist",
        "",
        f"- Observatory: `{queue.site_snapshot.name or 'current observatory'}`",
        f"- Night: `{queue.night_start_local.strftime('%Y-%m-%d %H:%M')}` -> `{queue.night_end_local.strftime('%Y-%m-%d %H:%M')}`",
        f"- Timezone: `{queue.timezone}`",
        f"- Device profile: `{queue.device_profile}`",
    ]
    if str(queue.campaign_name or "").strip():
        lines.append(f"- Template: `{queue.campaign_name.strip()}`")
    if bool(queue.require_science_checklist) and queue.science_checklist_items:
        lines.extend(["", "## Science Checklist", ""])
        for item in queue.science_checklist_items:
            text = str(item).strip()
            if text:
                lines.append(f"- {text}")
        if str(queue.campaign_notes or "").strip():
            lines.append(f"- Notes: {queue.campaign_notes.strip()}")
    lines.extend(
        [
            "",
            "## Queue",
            "",
            "| # | Alias | Target | Start | End | Filter | RA | Dec | Score |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for idx, block in enumerate(queue.blocks, start=1):
        lines.append(
            f"| {idx} | {block.alias} | {block.target_name} | "
            f"{block.start_local.strftime('%H:%M')} | {block.end_local.strftime('%H:%M')} | "
            f"{_block_filter_label(block)} | {block.ra_hms} | {block.dec_dms} | {block.score:.1f} |"
        )

    lines.extend(
        [
            "",
            "## Custom Celestial Objects",
            "",
            "V1 assumes every queued target should be added via **Custom Celestial Objects**.",
            "",
        ]
    )
    for block in queue.blocks:
        lines.append(f"- `{block.alias}` -> `{block.ra_hms}` / `{block.dec_dms}`")

    lines.extend(
        [
            "",
            "## Launch Checklist",
            "",
            "1. Open the Seestar app and connect to the telescope.",
            "2. Choose local control or the official Telescope Network path.",
            "3. Add every object above through **Custom Celestial Objects** using the alias as the display name.",
            "4. Recreate the queue in **Plan Mode** in the exact order shown above.",
            "5. For each block, use the listed start/end time and keep the suggested filter unless you intentionally change it.",
            "6. Start the session from the official Seestar app and monitor regular safety/connection prompts there.",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def _block_uses_session_run_length(block: NightQueueBlock) -> bool:
    return int(getattr(block, "target_order", 0) or 0) > 0


def _resolved_alp_block_duration_seconds(
    block: NightQueueBlock,
    config: SeestarAlpConfig,
) -> int:
    queue_duration_s = max(60, int(round((block.end_local - block.start_local).total_seconds())))
    if _block_uses_session_run_length(block):
        return queue_duration_s
    override_min = max(0, int(config.target_integration_override_min))
    if override_min > 0:
        return max(60, override_min * 60)
    return queue_duration_s


def build_alp_schedule_item(
    block: NightQueueBlock,
    config: SeestarAlpConfig | None = None,
) -> dict[str, Any]:
    cfg = config or SeestarAlpConfig()
    duration_s = _resolved_alp_block_duration_seconds(block, cfg)
    lp_mode = _resolved_block_lp_mode(block, cfg)
    is_use_lp_filter = block.recommended_filter == "duoband"
    if lp_mode == SEESTAR_ALP_LP_FILTER_ON:
        is_use_lp_filter = True
    elif lp_mode == SEESTAR_ALP_LP_FILTER_OFF:
        is_use_lp_filter = False
    raw_block_gain = getattr(block, "gain", -1)
    try:
        block_gain = int(raw_block_gain)
    except Exception:
        block_gain = -1
    if block_gain < 0:
        block_gain = max(0, int(cfg.gain))
    use_autofocus = getattr(block, "use_autofocus", None)
    if use_autofocus is None:
        use_autofocus = bool(cfg.use_autofocus)
    params = {
        "target_name": block.alias,
        "ra": block.ra_hms,
        "dec": block.dec_dms,
        "is_j2000": False,
        "is_use_lp_filter": is_use_lp_filter,
        "panel_time_sec": duration_s,
        "ra_num": 1,
        "dec_num": 1,
        "panel_overlap_percent": max(0, int(cfg.panel_overlap_percent)),
        "gain": block_gain,
        "is_use_autofocus": bool(use_autofocus),
        "selected_panels": "",
        "num_tries": max(1, int(cfg.num_tries)),
        "retry_wait_s": max(0, int(cfg.retry_wait_s)),
    }
    return {"action": "start_mosaic", "params": params}


def _normalize_wait_until_local_time(value: object) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    if "T" in raw:
        raw = raw.split("T", 1)[1]
    raw = raw.split()[0]
    raw = raw.replace(".", ":")
    match = re.fullmatch(r"(\d{1,2}):(\d{2})(?::\d{2})?", raw)
    if not match:
        return ""
    hour = int(match.group(1))
    minute = int(match.group(2))
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        return ""
    return f"{hour:02d}:{minute:02d}"


def _alp_filter_position_for_block(block: NightQueueBlock, config: SeestarAlpConfig) -> int:
    lp_mode = _resolved_block_lp_mode(block, config)
    if lp_mode == SEESTAR_ALP_LP_FILTER_ON:
        return 2
    if lp_mode == SEESTAR_ALP_LP_FILTER_OFF:
        return 1
    return 2 if block.recommended_filter == "duoband" else 1


def _resolved_block_lp_mode(block: NightQueueBlock, config: SeestarAlpConfig) -> str:
    block_mode = str(getattr(block, "lp_filter_mode", "") or "").strip().lower()
    if block_mode and block_mode != SEESTAR_ALP_LP_FILTER_AUTO:
        return block_mode
    return str(config.lp_filter_mode or SEESTAR_ALP_LP_FILTER_AUTO).strip().lower() or SEESTAR_ALP_LP_FILTER_AUTO


def _alp_wait_until_item(local_time: str) -> dict[str, Any]:
    return {"action": "wait_until", "params": {"local_time": local_time}}


def _alp_wait_for_item(seconds: int) -> dict[str, Any]:
    return {"action": "wait_for", "params": {"timer_sec": max(1, int(seconds))}}


def _alp_auto_focus_item(try_count: int) -> dict[str, Any]:
    return {"action": "auto_focus", "params": {"try_count": max(1, int(try_count))}}


def _alp_startup_item(config: SeestarAlpConfig) -> dict[str, Any]:
    return {
        "action": "start_up_sequence",
        "params": {
            "auto_focus": bool(config.startup_auto_focus),
            "3ppa": bool(config.startup_polar_align),
            "dark_frames": bool(config.startup_dark_frames),
        },
    }


def _alp_flats_item() -> dict[str, Any]:
    return {"action": "start_create_calib_frame", "params": {}}


def _alp_filter_item(position: int) -> dict[str, Any]:
    return {"action": "set_wheel_position", "params": [int(position)]}


def _alp_dew_heater_item(value: int) -> dict[str, Any]:
    return {"action": "action_set_dew_heater", "params": {"heater": max(0, int(value))}}


def _alp_exposure_item(exposure_ms: int) -> dict[str, Any]:
    return {"action": "action_set_exposure", "params": {"exp": max(1, int(exposure_ms))}}


def _alp_park_item() -> dict[str, Any]:
    return {"action": "scope_park", "params": {}}


def _alp_shutdown_item() -> dict[str, Any]:
    return {"action": "shutdown", "params": {}}


def _resolve_alp_wait_until_datetime(
    local_time: str,
    *,
    queue: NightQueue,
    now_local: datetime,
) -> Optional[datetime]:
    normalized = _normalize_wait_until_local_time(local_time)
    if not normalized:
        return None
    hour_str, minute_str = normalized.split(":", 1)
    hour = int(hour_str)
    minute = int(minute_str)
    tzinfo = now_local.tzinfo or queue.night_start_local.tzinfo or queue.generated_at.tzinfo
    candidate_dates = {
        now_local.date(),
        (now_local + timedelta(days=1)).date(),
        queue.night_start_local.date(),
        queue.night_end_local.date(),
    }
    candidates = [
        datetime(day.year, day.month, day.day, hour, minute, tzinfo=tzinfo)
        for day in candidate_dates
    ]
    in_window = [dt for dt in candidates if queue.night_start_local <= dt <= queue.night_end_local]
    if in_window:
        upcoming = [dt for dt in in_window if dt >= now_local]
        return min(upcoming) if upcoming else max(in_window)
    upcoming = [dt for dt in candidates if dt >= now_local]
    return min(upcoming) if upcoming else max(candidates)


def filter_alp_schedule_items_for_immediate_start(
    schedule_items: Iterable[dict[str, Any]],
    *,
    queue: NightQueue,
    now_local: Optional[datetime] = None,
    wait_grace_s: int = SEESTAR_ALP_IMMEDIATE_START_WAIT_GRACE_S,
) -> list[dict[str, Any]]:
    current = now_local or datetime.now(queue.night_start_local.tzinfo or queue.generated_at.tzinfo)
    grace_delta = timedelta(seconds=max(0, int(wait_grace_s)))
    filtered: list[dict[str, Any]] = []
    for item in schedule_items:
        action = str(item.get("action", "") or "").strip()
        if action != "wait_until":
            filtered.append(item)
            continue
        # Push + Start means "start now", so wait_until steps are never uploaded
        # in the immediate-start path. The stale-time check is kept only for
        # preview/explanation logic elsewhere.
        continue
        params = item.get("params", {})
        local_time = str(params.get("local_time", "") or "") if isinstance(params, dict) else ""
        target_dt = _resolve_alp_wait_until_datetime(local_time, queue=queue, now_local=current)
        if target_dt is None or target_dt <= current + grace_delta:
            continue
        filtered.append(item)
    return filtered


def build_alp_schedule_items(queue: NightQueue, config: SeestarAlpConfig) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    explicit_wait_until = _normalize_wait_until_local_time(config.wait_until_local_time)
    current_filter_position: Optional[int] = None
    current_exposure_ms: Optional[int] = None
    fixed_filter_position: Optional[int] = None
    config_lp_mode = str(config.lp_filter_mode or SEESTAR_ALP_LP_FILTER_AUTO).strip().lower()
    if config_lp_mode in {SEESTAR_ALP_LP_FILTER_OFF, SEESTAR_ALP_LP_FILTER_ON} and all(
        str(getattr(block, "lp_filter_mode", "") or "").strip().lower() in {"", SEESTAR_ALP_LP_FILTER_AUTO}
        for block in queue.blocks
    ):
        fixed_filter_position = 1 if config_lp_mode == SEESTAR_ALP_LP_FILTER_OFF else 2

    if explicit_wait_until:
        items.append(_alp_wait_until_item(explicit_wait_until))
    if bool(config.startup_enabled):
        items.append(_alp_startup_item(config))
    if int(config.dew_heater_value) >= 0:
        items.append(_alp_dew_heater_item(int(config.dew_heater_value)))
    if fixed_filter_position is not None:
        items.append(_alp_filter_item(fixed_filter_position))
        current_filter_position = fixed_filter_position
    if bool(config.capture_flats_before_session):
        items.append(_alp_flats_item())
        if int(config.flats_wait_s) > 0:
            items.append(_alp_wait_for_item(int(config.flats_wait_s)))
    if int(config.stack_exposure_ms) > 0:
        current_exposure_ms = int(config.stack_exposure_ms)
        items.append(_alp_exposure_item(current_exposure_ms))
    if bool(config.honor_queue_times) and not explicit_wait_until and queue.blocks:
        items.append(_alp_wait_until_item(queue.blocks[0].start_local.strftime("%H:%M")))

    previous_end: Optional[datetime] = None
    autofocus_mode = normalize_seestar_alp_schedule_autofocus_mode(
        getattr(config, "schedule_autofocus_mode", ""),
        legacy_enabled=bool(getattr(config, "schedule_autofocus_before_each_target", False)),
    )
    previous_target_key: Optional[tuple[int, str, float, float]] = None
    for index, block in enumerate(queue.blocks):
        if index > 0 and bool(config.honor_queue_times) and previous_end is not None:
            gap_seconds = int(round((block.start_local - previous_end).total_seconds()))
            if gap_seconds > 0:
                items.append(_alp_wait_for_item(gap_seconds))

        desired_filter_position = fixed_filter_position if fixed_filter_position is not None else _alp_filter_position_for_block(block, config)
        if desired_filter_position is not None and desired_filter_position != current_filter_position:
            items.append(_alp_filter_item(desired_filter_position))
            current_filter_position = desired_filter_position

        block_exposure_ms = max(0, int(getattr(block, "stack_exposure_ms", 0) or 0))
        if block_exposure_ms > 0 and block_exposure_ms != current_exposure_ms:
            items.append(_alp_exposure_item(block_exposure_ms))
            current_exposure_ms = block_exposure_ms
        elif block_exposure_ms <= 0 and int(config.stack_exposure_ms) > 0 and current_exposure_ms != int(config.stack_exposure_ms):
            current_exposure_ms = int(config.stack_exposure_ms)
            items.append(_alp_exposure_item(current_exposure_ms))

        target_key = (
            int(getattr(block, "target_order", 0) or 0),
            str(getattr(block, "target_name", "") or ""),
            float(getattr(block, "ra_deg", 0.0) or 0.0),
            float(getattr(block, "dec_deg", 0.0) or 0.0),
        )
        should_run_schedule_af = False
        if autofocus_mode == SEESTAR_ALP_AF_MODE_PER_RUN:
            should_run_schedule_af = True
        elif autofocus_mode == SEESTAR_ALP_AF_MODE_PER_TARGET and target_key != previous_target_key:
            should_run_schedule_af = True
        if should_run_schedule_af:
            items.append(_alp_auto_focus_item(int(config.schedule_autofocus_try_count)))
        items.append(build_alp_schedule_item(block, config))
        previous_end = block.end_local
        previous_target_key = target_key

    if bool(config.shutdown_after_session):
        items.append(_alp_shutdown_item())
    elif bool(config.park_after_session):
        items.append(_alp_park_item())
    return items


def _alp_schedule_item_detail(item: dict[str, Any]) -> str:
    action = str(item.get("action", "") or "").strip()
    params = item.get("params", {})
    if action == "wait_until":
        local_time = str(params.get("local_time", "--")) if isinstance(params, dict) else "--"
        return f"wait_until {local_time}"
    if action == "wait_for":
        seconds = int(params.get("timer_sec", 0)) if isinstance(params, dict) else 0
        return f"wait_for {seconds} s"
    if action == "auto_focus":
        rounds = int(params.get("try_count", 0)) if isinstance(params, dict) else 0
        return f"auto_focus {rounds} round(s)"
    if action == "start_up_sequence":
        if isinstance(params, dict):
            bits = []
            if params.get("3ppa"):
                bits.append("polar-align")
            if params.get("auto_focus"):
                bits.append("startup AF")
            if params.get("dark_frames"):
                bits.append("dark frames")
            return "start_up_sequence" + (f" ({', '.join(bits)})" if bits else "")
        return "start_up_sequence"
    if action == "start_create_calib_frame":
        return "start_create_calib_frame (blind flats trigger)"
    if action == "action_set_dew_heater":
        heater = int(params.get("heater", 0)) if isinstance(params, dict) else 0
        return f"set dew heater {heater}"
    if action == "action_set_exposure":
        exp = int(params.get("exp", 0)) if isinstance(params, dict) else 0
        return f"set exposure {exp} ms"
    if action == "set_wheel_position":
        position = None
        if isinstance(params, list) and params:
            try:
                position = int(params[0])
            except Exception:
                position = None
        filter_name = {0: "dark", 1: "IR-cut / LP off", 2: "LP filter on"}.get(position, f"wheel {position}")
        return f"set filter {filter_name}"
    if action == "start_mosaic":
        if not isinstance(params, dict):
            return "capture target"
        duration_s = int(params.get("panel_time_sec", 0) or 0)
        duration_min = max(1, int(round(duration_s / 60.0)))
        lp_txt = "LP on" if params.get("is_use_lp_filter") else "LP off"
        return (
            f"capture target {params.get('target_name', '-')}"
            f" | {duration_min} min | {lp_txt} | {params.get('ra', '-')} {params.get('dec', '-')}"
        )
    if action == "scope_park":
        return "park scope"
    if action == "shutdown":
        return "shutdown (parks first)"
    try:
        compact = json.dumps(params, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        compact = str(params)
    return f"{action} {compact}".strip()


def _alp_wait_until_will_be_skipped_for_immediate_start(
    item: dict[str, Any],
    *,
    queue: NightQueue,
    now_local: Optional[datetime] = None,
    wait_grace_s: int = SEESTAR_ALP_IMMEDIATE_START_WAIT_GRACE_S,
) -> bool:
    action = str(item.get("action", "") or "").strip()
    if action != "wait_until":
        return False
    return True


def _alp_integration_summary_text(
    config: SeestarAlpConfig,
    queue: NightQueue | None = None,
) -> str:
    if queue is not None and any(_block_uses_session_run_length(block) for block in queue.blocks):
        return "use session run length"
    if int(config.target_integration_override_min) > 0:
        return f"{int(config.target_integration_override_min)} min/item override"
    return "use queue block duration"


def _alp_item_config_summary(config: SeestarAlpConfig, queue: NightQueue | None = None) -> list[str]:
    integration_txt = _alp_integration_summary_text(config, queue)
    lp_filter_txt = _filter_display_name("", lp_mode=str(config.lp_filter_mode or SEESTAR_ALP_LP_FILTER_AUTO))
    stack_exposure_txt = (
        f"{int(config.stack_exposure_ms)} ms"
        if int(config.stack_exposure_ms) > 0
        else "ALP current/default"
    )
    return [
        f"LP filter: {lp_filter_txt}",
        f"Gain: {int(config.gain)}",
        f"Capture-job AF: {'on' if config.use_autofocus else 'off'}",
        f"Retries: {int(config.num_tries)}",
        f"Retry wait: {int(config.retry_wait_s)} s",
        f"Integration: {integration_txt}",
        f"Stack exposure: {stack_exposure_txt}",
        f"Honor queue times: {'on' if config.honor_queue_times else 'off'}",
        f"Manual wait-until: {_normalize_wait_until_local_time(config.wait_until_local_time) or '-'}",
        f"Startup: {'on' if config.startup_enabled else 'off'}",
        f"Flats: {'on' if config.capture_flats_before_session else 'off'}",
        f"Schedule AF: {seestar_alp_schedule_autofocus_mode_label(config.schedule_autofocus_mode, short=True)}",
        f"Dew heater: {int(config.dew_heater_value) if int(config.dew_heater_value) >= 0 else 'skip'}",
        f"Park after: {'on' if config.park_after_session else 'off'}",
        f"Shutdown after: {'on' if config.shutdown_after_session else 'off'}",
    ]


def _alp_schedule_action_label(item: dict[str, Any]) -> str:
    action = str(item.get("action", "") or "").strip()
    return {
        "wait_until": "Wait until",
        "wait_for": "Wait",
        "auto_focus": "Autofocus",
        "start_up_sequence": "Startup",
        "start_create_calib_frame": "Flats",
        "action_set_dew_heater": "Dew heater",
        "action_set_exposure": "Exposure",
        "set_wheel_position": "Filter",
        "start_mosaic": "Capture",
        "scope_park": "Park",
        "shutdown": "Shutdown",
    }.get(action, action or "-")


def build_alp_schedule_preview_rows(
    queue: NightQueue,
    schedule_items: Iterable[dict[str, Any]],
    *,
    now_local: Optional[datetime] = None,
) -> list[dict[str, str]]:
    current = now_local or datetime.now(queue.night_start_local.tzinfo or queue.generated_at.tzinfo)
    rows: list[dict[str, str]] = []
    for idx, item in enumerate(schedule_items, start=1):
        skip_on_start = _alp_wait_until_will_be_skipped_for_immediate_start(
            item,
            queue=queue,
            now_local=current,
        )
        detail = _alp_schedule_item_detail(item)
        if skip_on_start:
            detail += " | skips on Push + Start"
        rows.append(
            {
                "order": str(idx),
                "action": _alp_schedule_action_label(item),
                "detail": detail,
            }
        )
    return rows


def build_queue_block_preview_rows(
    queue: NightQueue,
    config: SeestarAlpConfig | None = None,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for idx, block in enumerate(queue.blocks, start=1):
        run_label = ""
        if int(getattr(block, "repeat_count", 1) or 1) > 1:
            run_label = f"{int(getattr(block, 'repeat_index', 1) or 1)}/{int(getattr(block, 'repeat_count', 1) or 1)}"
        rows.append(
            {
                "order": str(idx),
                "alias": str(block.alias or ""),
                "target": str(block.target_name or ""),
                "start": block.start_local.strftime("%H:%M"),
                "end": block.end_local.strftime("%H:%M"),
                "run": run_label,
                "filter": _block_filter_label(block, config),
                "score": f"{float(block.score):.1f}",
                "notes": str(block.notes or ""),
            }
        )
    return rows


def _extract_alp_schedule_id(value: Any) -> str:
    if isinstance(value, dict):
        schedule_id = str(value.get("schedule_id", "") or "").strip()
        if schedule_id:
            return schedule_id
        result = value.get("result")
        if isinstance(result, dict):
            return str(result.get("schedule_id", "") or "").strip()
    return ""


def _extract_alp_response_message(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        result = value.get("result")
        if isinstance(result, str):
            return result.strip()
        error = value.get("error")
        if isinstance(error, str):
            return error.strip()
    return ""


def render_alp_dialog_text(bundle: SeestarHandoffBundle, config: SeestarAlpConfig) -> str:
    queue = bundle.queue
    schedule_items = list(bundle.session_payload.get("schedule_items", []))
    has_wait_until = any(str(item.get("action", "") or "").strip() == "wait_until" for item in schedule_items)
    preview_now = datetime.now(queue.night_start_local.tzinfo or queue.generated_at.tzinfo)
    lines = [
        _queue_summary_text(queue),
        "",
        f"ALP service: {config.base_url.rstrip('/')}",
        f"Device number: {config.device_num}",
        "ALP item settings: " + " | ".join(_alp_item_config_summary(config, queue)),
        "",
    ]
    if not schedule_items:
        lines.append("No ALP schedule items are available to push from the current session.")
        return "\n".join(lines)

    _append_science_checklist(lines, queue)
    if bool(config.capture_flats_before_session):
        lines.extend(
            [
                "",
                "Flat Warning",
                "-" * 72,
                "Current AstroPlanner ALP flats step is a blind trigger for start_create_calib_frame.",
                "It does not yet measure ADU, model twilight brightness, or iterate exposure toward mid-well.",
                "Use it only when you intentionally want ALP to launch the flat routine, not as validated smart sky-flats automation.",
            ]
        )

    lines.append("ALP Schedule")
    lines.append("-" * 72)
    for idx, item in enumerate(schedule_items, start=1):
        detail = _alp_schedule_item_detail(item)
        if _alp_wait_until_will_be_skipped_for_immediate_start(item, queue=queue, now_local=preview_now):
            detail += " (will be skipped on Push + Start)"
        lines.append(f"{idx}. {detail}")

    lines.extend(["", "Queue Blocks", "-" * 72])
    for idx, block in enumerate(queue.blocks, start=1):
        lines.append(
            f"{idx}. {block.alias} | {block.target_name} | "
            f"{block.start_local.strftime('%H:%M')} -> {block.end_local.strftime('%H:%M')} | "
            f"filter={_block_filter_label(block, config)} | score {block.score:.1f}"
        )
        if block.notes:
            lines.append(f"   Notes: {block.notes}")

    lines.extend(
        [
            "",
            "ALP Flow",
            "-" * 72,
            "Push Queue uploads the exact item sequence shown above.",
            "Push + Start uploads the queue and immediately calls start_scheduler.",
            "The official Seestar app should not control the same device at the same time.",
        ]
    )
    if has_wait_until:
        lines.insert(
            -1,
            "Push + Start removes wait_until steps and starts the ALP scheduler now.",
        )
    return "\n".join(lines)


def render_alp_checklist_markdown(queue: NightQueue, config: SeestarAlpConfig) -> str:
    schedule_items = build_alp_schedule_items(queue, config)
    integration_txt = _alp_integration_summary_text(config, queue)
    lines = [
        "# Seestar ALP Session",
        "",
        f"- ALP service: `{config.base_url.rstrip('/')}`",
        f"- Device number: `{config.device_num}`",
        f"- LP filter: `{_filter_display_name('', lp_mode=str(config.lp_filter_mode or SEESTAR_ALP_LP_FILTER_AUTO))}`",
        f"- Gain: `{int(config.gain)}`",
        f"- Autofocus: `{'on' if config.use_autofocus else 'off'}`",
        f"- Retries: `{int(config.num_tries)}`",
        f"- Retry wait: `{int(config.retry_wait_s)} s`",
        f"- Integration: `{integration_txt}`",
        f"- Stack exposure: `{'%d ms' % int(config.stack_exposure_ms) if int(config.stack_exposure_ms) > 0 else 'ALP current/default'}`",
        f"- Honor queue times: `{'on' if config.honor_queue_times else 'off'}`",
        f"- Manual wait-until: `{_normalize_wait_until_local_time(config.wait_until_local_time) or '-'}`",
        f"- Startup sequence: `{'on' if config.startup_enabled else 'off'}`",
        f"- Flats before session: `{'on' if config.capture_flats_before_session else 'off'}`",
        f"- Schedule autofocus: `{seestar_alp_schedule_autofocus_mode_label(config.schedule_autofocus_mode, short=True)}`",
        f"- Dew heater: `{int(config.dew_heater_value) if int(config.dew_heater_value) >= 0 else 'skip'}`",
        f"- Park after session: `{'on' if config.park_after_session else 'off'}`",
        f"- Shutdown after session: `{'on' if config.shutdown_after_session else 'off'}`",
        f"- Night: `{queue.night_start_local.strftime('%Y-%m-%d %H:%M')}` -> `{queue.night_end_local.strftime('%Y-%m-%d %H:%M')}`",
        f"- Timezone: `{queue.timezone}`",
    ]
    if str(queue.campaign_name or "").strip():
        lines.append(f"- Template: `{queue.campaign_name.strip()}`")
    if queue.science_checklist_items:
        lines.extend(["", "## Science Checklist", ""])
        for item in queue.science_checklist_items:
            text = str(item).strip()
            if text:
                lines.append(f"- {text}")
        if str(queue.campaign_notes or "").strip():
            lines.append(f"- Notes: {queue.campaign_notes.strip()}")
    if bool(config.capture_flats_before_session):
        lines.extend(
            [
                "",
                "## Flat Warning",
                "",
                "- Current AstroPlanner ALP flats automation is only a blind `start_create_calib_frame` trigger.",
                "- It does not read ADU, estimate sky brightness, or tune exposure toward mid-dynamic-range flats.",
                "- Use it only as an expert shortcut when you already trust the physical flat conditions.",
            ]
        )
    lines.extend(
        [
            "",
            "## ALP Schedule",
            "",
            "| # | Action | Detail |",
            "|---|---|---|",
        ]
    )
    for idx, item in enumerate(schedule_items, start=1):
        lines.append(
            f"| {idx} | `{str(item.get('action', ''))}` | {_alp_schedule_item_detail(item)} |"
        )

    lines.extend(
        [
            "",
            "## Queue Blocks",
            "",
            "| # | Alias | Target | Start | End | Filter | RA | Dec | Score |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for idx, block in enumerate(queue.blocks, start=1):
        lines.append(
            f"| {idx} | {block.alias} | {block.target_name} | "
            f"{block.start_local.strftime('%H:%M')} | {block.end_local.strftime('%H:%M')} | "
            f"{_block_filter_label(block, config)} | {block.ra_hms} | {block.dec_dms} | {block.score:.1f} |"
        )

    lines.extend(
        [
            "",
            "## Push Sequence",
            "",
            "1. Ensure the `seestar_alp` service is running and owns the device.",
            "2. Test the ALP connection from AstroPlanner.",
            "3. Review the generated ALP schedule items and science checklist.",
            "4. Push the queue to create a fresh ALP schedule.",
            "5. Optionally start the scheduler immediately from AstroPlanner.",
            "6. Do not control the same Seestar from the official app while ALP is active.",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def build_alp_debug_sample_queue(
    queue: NightQueue,
    *,
    max_blocks: int = 2,
    sample_block_seconds: int = 60,
    gap_seconds: int = 30,
) -> NightQueue:
    sample_block_seconds = max(60, int(sample_block_seconds))
    gap_seconds = max(0, int(gap_seconds))
    sample_blocks: list[NightQueueBlock] = []
    tzinfo = queue.night_start_local.tzinfo or queue.generated_at.tzinfo
    cursor = datetime.now(tzinfo) if tzinfo is not None else datetime.now()

    for idx, block in enumerate(queue.blocks[: max(1, int(max_blocks))], start=1):
        block_start = cursor
        block_end = block_start + timedelta(seconds=sample_block_seconds)
        sample_blocks.append(
            NightQueueBlock(
                alias=f"{block.alias}-sample",
                target_name=block.target_name,
                ra_deg=block.ra_deg,
                dec_deg=block.dec_deg,
                ra_hms=block.ra_hms,
                dec_dms=block.dec_dms,
                start_local=block_start,
                end_local=block_end,
                score=block.score,
                hours_above_limit=block.hours_above_limit,
                recommended_filter=block.recommended_filter,
                window_start_local=block_start,
                window_end_local=block_end,
                notes=f"{block.notes} [sample]" if block.notes else "sample queue item",
                lp_filter_mode=block.lp_filter_mode,
                stack_exposure_ms=block.stack_exposure_ms,
                gain=block.gain,
                use_autofocus=block.use_autofocus,
                repeat_index=block.repeat_index,
                repeat_count=block.repeat_count,
                target_order=block.target_order,
            )
        )
        cursor = block_end + timedelta(seconds=gap_seconds)

    if sample_blocks:
        night_start = sample_blocks[0].start_local
        night_end = sample_blocks[-1].end_local
    else:
        night_start = cursor
        night_end = cursor + timedelta(seconds=sample_block_seconds)

    return NightQueue(
        site_snapshot=queue.site_snapshot,
        night_start_local=night_start,
        night_end_local=night_end,
        timezone=queue.timezone,
        device_profile=queue.device_profile,
        blocks=sample_blocks,
        generated_at=night_start,
        source_plan_version=f"{queue.source_plan_version}-sample",
    )


def render_alp_backend_status_text(
    status: SeestarAlpBackendStatus | dict[str, Any] | None,
    *,
    last_error: str = "",
    base_url: str = "",
    device_num: int | None = None,
) -> str:
    if isinstance(status, dict):
        status_obj = SeestarAlpBackendStatus.model_validate(status)
    else:
        status_obj = status
    if status_obj is None:
        location = base_url.rstrip("/") if base_url else "unavailable"
        device_txt = f"device {device_num}" if device_num is not None else "device ?"
        return f"ALP status: unavailable at {location} ({device_txt}) | Last error: {last_error or '-'}"

    connected_txt = "connected" if status_obj.connected else "service reachable, device not connected"
    device_name = status_obj.device_name or "unknown device"
    current_item = status_obj.current_item_id or "-"
    raw_txt = "yes" if bool(status_obj.supports_raw_flats) else "no"
    return (
        f"ALP status: {connected_txt} | {device_name} | schedule: {status_obj.schedule_state} | "
        f"items: {status_obj.queued_items} | current item: {current_item} | "
        f"raw flats: {raw_txt} | last error: {last_error or '-'}"
    )


def build_alp_web_ui_url(base_url: str) -> str:
    raw = str(base_url or SEESTAR_ALP_DEFAULT_BASE_URL).strip()
    if "://" not in raw:
        raw = f"http://{raw}"
    parsed = urlparse(raw)
    scheme = parsed.scheme or "http"
    host = parsed.hostname or "localhost"
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    auth = ""
    if parsed.username:
        auth = parsed.username
        if parsed.password:
            auth += f":{parsed.password}"
        auth += "@"
    netloc = f"{auth}{host}:5432"
    return urlunparse((scheme, netloc, "/", "", "", ""))


def build_alp_imager_base_url(base_url: str) -> str:
    raw = str(base_url or SEESTAR_ALP_DEFAULT_BASE_URL).strip()
    if "://" not in raw:
        raw = f"http://{raw}"
    parsed = urlparse(raw)
    scheme = parsed.scheme or "http"
    host = parsed.hostname or "localhost"
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    auth = ""
    if parsed.username:
        auth = parsed.username
        if parsed.password:
            auth += f":{parsed.password}"
        auth += "@"
    netloc = f"{auth}{host}:7556"
    return urlunparse((scheme, netloc, "/", "", "", "")).rstrip("/")


def build_alp_imager_video_url(base_url: str, device_num: int) -> str:
    return f"{build_alp_imager_base_url(base_url)}/{int(device_num)}/vid"


def build_alp_raw_stats_url(base_url: str, device_num: int) -> str:
    return f"{build_alp_imager_base_url(base_url)}/{int(device_num)}/raw-stats"


def build_alp_raw_frame_url(base_url: str, device_num: int) -> str:
    return f"{build_alp_imager_base_url(base_url)}/{int(device_num)}/raw-frame.npy"


def measure_preview_brightness(
    jpeg_bytes: bytes,
    *,
    crop_fraction: float = SEESTAR_SMART_FLATS_DEFAULT_CROP_FRACTION,
) -> dict[str, float]:
    if not jpeg_bytes:
        raise RuntimeError("Preview frame is empty.")
    with Image.open(BytesIO(jpeg_bytes)) as image:
        gray = image.convert("L")
        arr = np.asarray(gray, dtype=np.float32)
    if arr.size == 0:
        raise RuntimeError("Preview frame could not be decoded.")
    crop_fraction = float(crop_fraction)
    crop_fraction = min(1.0, max(0.1, crop_fraction))
    height, width = arr.shape[:2]
    crop_h = max(1, int(round(height * crop_fraction)))
    crop_w = max(1, int(round(width * crop_fraction)))
    top = max(0, (height - crop_h) // 2)
    left = max(0, (width - crop_w) // 2)
    cropped = arr[top : top + crop_h, left : left + crop_w]
    mean_8bit = float(np.mean(cropped))
    median_8bit = float(np.median(cropped))
    return {
        "mean_fraction": mean_8bit / 255.0,
        "median_fraction": median_8bit / 255.0,
        "estimated_mean_8bit": mean_8bit,
        "estimated_median_8bit": median_8bit,
    }


def analyze_preview_linearity(measurements: list[SeestarSmartFlatMeasurement]) -> float:
    valid_ratios = [
        float(item.mean_fraction) / max(1.0, float(item.exposure_ms))
        for item in measurements
        if float(item.mean_fraction) > 0.0 and int(item.exposure_ms) > 0
    ]
    if len(valid_ratios) < 2:
        return 0.0
    mean_ratio = float(sum(valid_ratios) / len(valid_ratios))
    if mean_ratio <= 0.0:
        return math.inf
    return max(abs(ratio - mean_ratio) for ratio in valid_ratios) / mean_ratio


def analyze_raw_linearity(measurements: list[SeestarRawFlatMeasurement]) -> float:
    valid_ratios = [
        float(item.median_adu) / max(1.0, float(item.exposure_ms))
        for item in measurements
        if float(item.median_adu) > 0.0 and int(item.exposure_ms) > 0
    ]
    if len(valid_ratios) < 2:
        return 0.0
    mean_ratio = float(sum(valid_ratios) / len(valid_ratios))
    if mean_ratio <= 0.0:
        return math.inf
    return max(abs(ratio - mean_ratio) for ratio in valid_ratios) / mean_ratio


def render_smart_flats_report_text(report: SeestarSmartFlatsReport) -> str:
    lines = [
        "Smart Sky Flats",
        "-" * 72,
        f"Success: {'yes' if report.success else 'no'}",
        f"Ready for flats: {'yes' if report.ready_for_flats else 'no'}",
        f"Auto-triggered flat capture: {'yes' if report.auto_triggered else 'no'}",
        f"Final preview exposure: {int(report.final_exposure_ms)} ms",
        f"Final mean brightness: {float(report.final_mean_fraction) * 100.0:.1f}% "
        f"(estimated {float(report.final_mean_fraction) * 255.0:.1f}/255)",
        f"Final median brightness: {float(report.final_median_fraction) * 100.0:.1f}% "
        f"(estimated {float(report.final_median_fraction) * 255.0:.1f}/255)",
        f"Linearity spread: {float(report.linearity_spread_fraction) * 100.0:.1f}%",
    ]
    if report.measurements:
        lines.extend(["", "Measurements", "-" * 72])
        for idx, item in enumerate(report.measurements, start=1):
            lines.append(
                f"{idx}. {int(item.exposure_ms)} ms | mean {float(item.mean_fraction) * 100.0:.1f}% "
                f"| median {float(item.median_fraction) * 100.0:.1f}% "
                f"| est mean {float(item.estimated_mean_8bit):.1f}/255"
            )
    if report.warnings:
        lines.extend(["", "Warnings", "-" * 72])
        for warning in report.warnings:
            lines.append(f"- {warning}")
    lines.extend(
        [
            "",
            "This routine uses preview-frame luminance, not sensor-true ADU from raw flats.",
            "Treat it as a practical twilight guardrail for ALP, not a calibrated photometric ADU meter.",
        ]
    )
    return "\n".join(lines)


def render_raw_flats_report_text(report: SeestarRawFlatsReport) -> str:
    lines = [
        "Smart Sky Flats (Raw ADU)",
        "-" * 72,
        f"Success: {'yes' if report.success else 'no'}",
        f"Ready for flats: {'yes' if report.ready_for_flats else 'no'}",
        f"Auto-triggered flat capture: {'yes' if report.auto_triggered else 'no'}",
        f"Final preview exposure: {int(report.final_exposure_ms)} ms",
        f"Final median ADU: {float(report.final_median_adu):.1f} / {int(report.final_max_adu)} "
        f"({float(report.final_percent_of_full_scale) * 100.0:.1f}% full scale)",
        f"Target median ADU: {float(report.final_target_adu):.1f}",
        f"Final mean ADU: {float(report.final_mean_adu):.1f}",
        f"Saturated fraction: {float(report.final_saturated_fraction) * 100.0:.3f}%",
        f"Quadrant spread: {float(report.final_quadrant_spread_fraction) * 100.0:.1f}%",
        f"Linearity spread: {float(report.linearity_spread_fraction) * 100.0:.1f}%",
    ]
    if report.measurements:
        lines.extend(["", "Measurements", "-" * 72])
        for idx, item in enumerate(report.measurements, start=1):
            lines.append(
                f"{idx}. {int(item.exposure_ms)} ms | median {float(item.median_adu):.1f} | "
                f"mean {float(item.mean_adu):.1f} | p95 {float(item.p95_adu):.1f} | "
                f"sat {float(item.saturated_fraction) * 100.0:.3f}% | "
                f"quad {float(item.quadrant_spread_fraction) * 100.0:.1f}%"
            )
    if report.warnings:
        lines.extend(["", "Warnings", "-" * 72])
        for warning in report.warnings:
            lines.append(f"- {warning}")
    lines.extend(
        [
            "",
            "This routine uses raw-like pre-JPEG frame statistics reported by Seestar ALP.",
            "It is a practical automation aid, but sky conditions can still invalidate flats if the field is uneven.",
        ]
    )
    return "\n".join(lines)


class SeestarAlpImagerClient:
    def __init__(self, config: SeestarAlpConfig):
        self.config = config

    @property
    def video_url(self) -> str:
        return build_alp_imager_video_url(self.config.base_url, self.config.device_num)

    @property
    def raw_stats_url(self) -> str:
        return build_alp_raw_stats_url(self.config.base_url, self.config.device_num)

    @property
    def raw_frame_url(self) -> str:
        return build_alp_raw_frame_url(self.config.base_url, self.config.device_num)

    def _http_get_json(self, url: str, *, timeout_s: float | None = None) -> dict[str, Any]:
        request = Request(url, method="GET")
        effective_timeout = float(timeout_s or self.config.timeout_s or 5.0)
        try:
            with urlopen(request, timeout=effective_timeout) as response:
                payload = response.read()
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore").strip()
            raise RuntimeError(f"ALP imager GET failed ({exc.code}): {detail or exc.reason}") from exc
        except URLError as exc:
            raise RuntimeError(f"ALP imager GET failed: {exc.reason}") from exc

        try:
            parsed = json.loads(payload.decode("utf-8"))
        except Exception as exc:
            raise RuntimeError("ALP imager returned invalid JSON.") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError("ALP imager returned an unexpected payload.")
        if parsed.get("error"):
            raise RuntimeError(str(parsed.get("error")))
        return parsed

    def fetch_frame_jpeg(self, *, timeout_s: float | None = None) -> bytes:
        request = Request(self.video_url, method="GET")
        effective_timeout = float(timeout_s or self.config.timeout_s or 5.0)
        try:
            with urlopen(request, timeout=effective_timeout) as response:
                buffer = bytearray()
                while True:
                    chunk = response.read(4096)
                    if not chunk:
                        break
                    buffer.extend(chunk)
                    start = buffer.find(b"\xff\xd8")
                    if start < 0:
                        if len(buffer) > 1_000_000:
                            del buffer[:-2]
                        continue
                    end = buffer.find(b"\xff\xd9", start + 2)
                    if end >= 0:
                        return bytes(buffer[start : end + 2])
                    if start > 0:
                        del buffer[:start]
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore").strip()
            raise RuntimeError(f"ALP imager GET failed ({exc.code}): {detail or exc.reason}") from exc
        except URLError as exc:
            raise RuntimeError(f"ALP imager GET failed: {exc.reason}") from exc
        raise RuntimeError("Could not extract a preview JPEG frame from the ALP imager stream.")

    def measure_frame_brightness(self, *, crop_fraction: float) -> dict[str, float]:
        return measure_preview_brightness(self.fetch_frame_jpeg(), crop_fraction=crop_fraction)

    def get_raw_stats(
        self,
        *,
        crop_fraction: float,
        sample_count: int,
        mode: str = "preview",
    ) -> SeestarRawFlatMeasurement:
        url = (
            f"{self.raw_stats_url}?"
            + urlencode(
                {
                    "crop_fraction": float(crop_fraction),
                    "sample_count": max(1, int(sample_count)),
                    "mode": str(mode or "preview"),
                }
            )
        )
        payload = self._http_get_json(url)
        return SeestarRawFlatMeasurement.model_validate(payload)

    def supports_raw_flats(self) -> bool:
        url = (
            f"{self.raw_stats_url}?"
            + urlencode(
                {
                    "crop_fraction": SEESTAR_SMART_FLATS_DEFAULT_CROP_FRACTION,
                    "sample_count": 1,
                    "mode": "preview",
                }
            )
        )
        request = Request(url, method="GET")
        try:
            with urlopen(request, timeout=max(1.0, float(self.config.timeout_s or 5.0) / 2.0)):
                return True
        except HTTPError as exc:
            if exc.code == 404:
                return False
            return True
        except URLError:
            return False


class SeestarAlpClient:
    def __init__(self, config: SeestarAlpConfig):
        self.config = config
        self._transaction_id = 1

    @property
    def _base_url(self) -> str:
        return str(self.config.base_url or SEESTAR_ALP_DEFAULT_BASE_URL).strip().rstrip("/")

    @property
    def _device_base_url(self) -> str:
        return f"{self._base_url}/api/v1/telescope/{int(self.config.device_num)}"

    def _next_transaction_id(self) -> int:
        current = self._transaction_id
        self._transaction_id += 1
        return current

    def _parse_json_response(self, payload: bytes) -> dict[str, Any]:
        try:
            parsed = json.loads(payload.decode("utf-8"))
        except Exception as exc:
            raise RuntimeError("ALP service returned invalid JSON.") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError("ALP service returned an unexpected payload.")
        error_number = parsed.get("ErrorNumber", 0)
        try:
            error_number = int(error_number)
        except Exception:
            error_number = 0
        if error_number != 0:
            message = str(parsed.get("ErrorMessage", "") or f"ALP error {error_number}")
            raise RuntimeError(message)
        return parsed

    def _http_get(self, path: str) -> dict[str, Any]:
        query = urlencode(
            {
                "ClientID": int(self.config.client_id),
                "ClientTransactionID": self._next_transaction_id(),
            }
        )
        url = f"{self._device_base_url}/{path}?{query}"
        request = Request(url, method="GET")
        try:
            with urlopen(request, timeout=float(self.config.timeout_s)) as response:
                return self._parse_json_response(response.read())
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore").strip()
            raise RuntimeError(f"ALP GET {path} failed ({exc.code}): {detail or exc.reason}") from exc
        except URLError as exc:
            raise RuntimeError(f"ALP GET {path} failed: {exc.reason}") from exc

    def _http_put_form(self, path: str, form_data: dict[str, object]) -> dict[str, Any]:
        encoded = urlencode(
            {
                "ClientID": int(self.config.client_id),
                "ClientTransactionID": self._next_transaction_id(),
                **form_data,
            }
        ).encode("utf-8")
        request = Request(
            f"{self._device_base_url}/{path}",
            data=encoded,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="PUT",
        )
        try:
            with urlopen(request, timeout=float(self.config.timeout_s)) as response:
                return self._parse_json_response(response.read())
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore").strip()
            raise RuntimeError(f"ALP PUT {path} failed ({exc.code}): {detail or exc.reason}") from exc
        except URLError as exc:
            raise RuntimeError(f"ALP PUT {path} failed: {exc.reason}") from exc

    def get_connected(self) -> bool:
        payload = self._http_get("connected")
        value = payload.get("Value", False)
        return bool(value)

    def get_name(self) -> str:
        payload = self._http_get("name")
        return str(payload.get("Value", "") or "")

    def set_connected(self, connected: bool) -> dict[str, Any]:
        return self._http_put_form("connected", {"Connected": bool(connected)})

    def put_action(self, action: str, params: Any) -> dict[str, Any]:
        return self._http_put_form("action", {"Action": action, "Parameters": json.dumps(params, separators=(",", ":"))})

    def set_stack_exposure_ms(self, exposure_ms: int) -> dict[str, Any]:
        return self.put_action(
            "method_sync",
            {
                "method": "set_setting",
                "params": {"exp_ms": {"stack_l": max(0, int(exposure_ms))}},
            },
        )

    def get_setting(self) -> dict[str, Any]:
        payload = self.put_action("method_sync", {"method": "get_setting"})
        value = payload.get("Value", {})
        if not isinstance(value, dict):
            raise RuntimeError("ALP get_setting returned an unexpected payload.")
        result = value.get("result", {})
        return result if isinstance(result, dict) else {}

    def get_live_exposure_ms(self) -> int:
        setting = self.get_setting()
        try:
            value = int(setting.get("isp_exp_ms", SEESTAR_SMART_FLATS_DEFAULT_STARTING_EXPOSURE_MS))
        except Exception:
            value = SEESTAR_SMART_FLATS_DEFAULT_STARTING_EXPOSURE_MS
        return max(1, value)

    def set_live_exposure_ms(self, exposure_ms: int) -> dict[str, Any]:
        exposure_ms = max(1, min(200, int(exposure_ms)))
        return self.put_action(
            "method_sync",
            {"method": "set_setting", "params": {"isp_exp_ms": exposure_ms}},
        )

    def begin_streaming(self) -> dict[str, Any]:
        return self.put_action("method_sync", {"method": "begin_streaming"})

    def stop_streaming(self) -> dict[str, Any]:
        return self.put_action("method_sync", {"method": "stop_streaming"})

    def start_create_calib_frame(self) -> dict[str, Any]:
        return self.put_action("method_sync", {"method": "start_create_calib_frame"})

    def get_schedule(self) -> dict[str, Any]:
        payload = self.put_action("get_schedule", {})
        value = payload.get("Value", {})
        if not isinstance(value, dict):
            raise RuntimeError("ALP get_schedule returned an unexpected payload.")
        return value

    def get_backend_status(self) -> SeestarAlpBackendStatus:
        connected = self.get_connected()
        device_name = ""
        try:
            device_name = self.get_name()
        except Exception:
            device_name = ""

        supports_raw_flats = False
        try:
            supports_raw_flats = SeestarAlpImagerClient(self.config).supports_raw_flats()
        except Exception:
            supports_raw_flats = False

        schedule_state = "unavailable"
        schedule_id = ""
        queued_items = 0
        current_item_id = ""
        if connected:
            try:
                schedule = self.get_schedule()
            except Exception:
                schedule = {}
            if isinstance(schedule, dict):
                schedule_state = str(schedule.get("state", "unknown") or "unknown")
                schedule_id = str(schedule.get("schedule_id", "") or "")
                current_item_id = str(schedule.get("current_item_id", "") or "")
                queue_items = schedule.get("list", [])
                if isinstance(queue_items, (list, tuple)):
                    queued_items = len(queue_items)
                else:
                    try:
                        queued_items = len(queue_items)
                    except Exception:
                        queued_items = 0

        return SeestarAlpBackendStatus(
            base_url=self._base_url,
            device_num=int(self.config.device_num),
            connected=connected,
            device_name=device_name,
            schedule_state=schedule_state,
            schedule_id=schedule_id,
            queued_items=queued_items,
            current_item_id=current_item_id,
            supports_raw_flats=supports_raw_flats,
        )

    def test_connection(self) -> dict[str, Any]:
        return self.get_backend_status().model_dump(mode="json")

    def push_queue(
        self,
        queue: NightQueue,
        *,
        schedule_items: Optional[list[dict[str, Any]]] = None,
        start_immediately: bool = False,
        now_local: Optional[datetime] = None,
    ) -> dict[str, Any]:
        requested_schedule_id = queue.generated_at.strftime("AP-%Y%m%d-%H%M%S")
        connected = self.get_connected()
        if not connected:
            self.set_connected(True)
            connected = self.get_connected()
        if not connected:
            raise RuntimeError("ALP service is reachable, but the target device is not connected.")

        upload_items = list(schedule_items) if schedule_items is not None else []
        if not upload_items and int(self.config.stack_exposure_ms) > 0:
            self.set_stack_exposure_ms(int(self.config.stack_exposure_ms))
        if not upload_items:
            upload_items = [build_alp_schedule_item(block, self.config) for block in queue.blocks]
        if start_immediately:
            upload_items = filter_alp_schedule_items_for_immediate_start(
                upload_items,
                queue=queue,
                now_local=now_local,
            )

        create_response = self.put_action("create_schedule", {"schedule_id": requested_schedule_id})
        schedule = create_response.get("Value")
        if isinstance(schedule, str):
            message = schedule.strip()
            if message:
                lowered = message.lower()
                if any(token in lowered for token in ("active", "error", "failed", "invalid")):
                    raise RuntimeError(f"ALP create_schedule failed: {message}")
            schedule = {}
        if schedule is None:
            schedule = {}
        if not isinstance(schedule, dict):
            schedule = {}
        if not schedule:
            try:
                schedule = self.get_schedule()
            except Exception:
                schedule = {}
        active_schedule_id = _extract_alp_schedule_id(schedule) or requested_schedule_id

        uploaded_items = 0
        for item in upload_items:
            schedule = self.put_action("add_schedule_item", item).get("Value")
            active_schedule_id = _extract_alp_schedule_id(schedule) or active_schedule_id
            uploaded_items += 1

        started = False
        skipped_waits = max(
            0,
            sum(1 for item in (schedule_items or []) if str(item.get("action", "") or "").strip() == "wait_until")
            - sum(1 for item in upload_items if str(item.get("action", "") or "").strip() == "wait_until"),
        )
        if start_immediately:
            start_messages: list[str] = []
            post_start_schedule: dict[str, Any] = {}
            for attempt in range(2):
                start_response = self.put_action("start_scheduler", {}).get("Value")
                start_message = _extract_alp_response_message(start_response)
                if start_message:
                    start_messages.append(start_message)
                if isinstance(start_response, dict):
                    schedule = start_response
                for _poll in range(8):
                    try:
                        post_start_schedule = self.get_schedule()
                    except Exception:
                        post_start_schedule = {}
                    post_start_id = _extract_alp_schedule_id(post_start_schedule)
                    post_start_state = (
                        str(post_start_schedule.get("state", "") or "").strip().lower()
                        if isinstance(post_start_schedule, dict)
                        else ""
                    )
                    if post_start_id and post_start_id != active_schedule_id and active_schedule_id:
                        raise RuntimeError(
                            f"ALP start_scheduler started a different schedule ({post_start_id}) than the uploaded one ({active_schedule_id})."
                        )
                    if post_start_state == "working":
                        schedule = post_start_schedule or schedule
                        started = True
                        break
                    time.sleep(0.35)
                if started:
                    break
            if not started:
                detail = " | ".join(part for part in start_messages if part) or "backend did not enter a running state"
                raise RuntimeError(f"ALP start_scheduler did not start the uploaded schedule: {detail}")

        return {
            "schedule_id": active_schedule_id,
            "uploaded_items": uploaded_items,
            "started": started,
            "schedule": schedule,
            "connected": connected,
            "skipped_waits": skipped_waits,
        }


def run_smart_sky_flats(
    client: Any,
    imager: Any,
    config: SeestarSmartFlatsConfig,
    *,
    lp_filter_mode: str = SEESTAR_ALP_LP_FILTER_AUTO,
    sleep_fn: Any = time.sleep,
) -> SeestarSmartFlatsReport:
    cfg = SeestarSmartFlatsConfig.model_validate(config)
    warnings: list[str] = []
    measurements: list[SeestarSmartFlatMeasurement] = []
    auto_triggered = False
    started_streaming = False

    try:
        if str(lp_filter_mode or SEESTAR_ALP_LP_FILTER_AUTO).strip().lower() == SEESTAR_ALP_LP_FILTER_OFF:
            client.put_action("method_sync", {"method": "set_wheel_position", "params": [1]})
        elif str(lp_filter_mode or SEESTAR_ALP_LP_FILTER_AUTO).strip().lower() == SEESTAR_ALP_LP_FILTER_ON:
            client.put_action("method_sync", {"method": "set_wheel_position", "params": [2]})
    except Exception as exc:
        warnings.append(f"Could not apply requested filter wheel state before flats: {exc}")

    try:
        client.begin_streaming()
        started_streaming = True
    except Exception as exc:
        warnings.append(f"Could not start preview streaming automatically: {exc}")

    try:
        exposure_ms = max(
            int(cfg.min_exposure_ms),
            min(int(cfg.max_exposure_ms), int(cfg.starting_exposure_ms or 0)),
        )
        if exposure_ms <= 0:
            exposure_ms = max(int(cfg.min_exposure_ms), min(int(cfg.max_exposure_ms), int(client.get_live_exposure_ms())))
    except Exception:
        exposure_ms = max(int(cfg.min_exposure_ms), min(int(cfg.max_exposure_ms), SEESTAR_SMART_FLATS_DEFAULT_STARTING_EXPOSURE_MS))

    last_mean = 0.0
    last_median = 0.0
    for _ in range(max(1, int(cfg.max_iterations))):
        client.set_live_exposure_ms(exposure_ms)
        settle_s = max(0.0, float(cfg.settle_s))
        if settle_s > 0.0:
            sleep_fn(settle_s)

        mean_values: list[float] = []
        median_values: list[float] = []
        est_mean_values: list[float] = []
        est_median_values: list[float] = []
        for _sample_idx in range(max(1, int(cfg.samples_per_step))):
            sample = imager.measure_frame_brightness(crop_fraction=float(cfg.crop_fraction))
            mean_values.append(float(sample["mean_fraction"]))
            median_values.append(float(sample["median_fraction"]))
            est_mean_values.append(float(sample["estimated_mean_8bit"]))
            est_median_values.append(float(sample["estimated_median_8bit"]))

        last_mean = float(sum(mean_values) / len(mean_values))
        last_median = float(sum(median_values) / len(median_values))
        measurements.append(
            SeestarSmartFlatMeasurement(
                exposure_ms=int(exposure_ms),
                mean_fraction=last_mean,
                median_fraction=last_median,
                estimated_mean_8bit=float(sum(est_mean_values) / len(est_mean_values)),
                estimated_median_8bit=float(sum(est_median_values) / len(est_median_values)),
            )
        )

        if abs(last_mean - float(cfg.target_fraction)) <= float(cfg.tolerance_fraction):
            break

        if last_mean <= 1e-6:
            next_exposure = max(int(cfg.min_exposure_ms), min(int(cfg.max_exposure_ms), exposure_ms * 2))
        else:
            scale = float(cfg.target_fraction) / last_mean
            next_exposure = int(round(exposure_ms * scale))
            next_exposure = max(int(cfg.min_exposure_ms), min(int(cfg.max_exposure_ms), next_exposure))
        if next_exposure == exposure_ms:
            if last_mean < float(cfg.target_fraction):
                next_exposure = min(int(cfg.max_exposure_ms), exposure_ms + 1)
            elif last_mean > float(cfg.target_fraction):
                next_exposure = max(int(cfg.min_exposure_ms), exposure_ms - 1)
        if next_exposure == exposure_ms:
            break
        exposure_ms = next_exposure

    linearity_spread = analyze_preview_linearity(measurements)
    ready_for_flats = bool(measurements) and abs(last_mean - float(cfg.target_fraction)) <= float(cfg.tolerance_fraction)
    if not ready_for_flats:
        warnings.append(
            "Preview brightness did not converge close enough to the requested mid-range target."
        )
    if len(measurements) >= 2 and not math.isfinite(linearity_spread):
        warnings.append("Linearity check failed because brightness ratios were not finite.")
    elif len(measurements) >= 2 and linearity_spread > float(cfg.linearity_tolerance_fraction):
        warnings.append(
            f"Preview brightness vs exposure was not linear enough (spread {linearity_spread * 100.0:.1f}%)."
        )
        ready_for_flats = False

    if ready_for_flats and bool(cfg.trigger_flat_capture_when_ready):
        try:
            client.start_create_calib_frame()
            auto_triggered = True
        except Exception as exc:
            warnings.append(f"Flats looked ready but ALP flat capture trigger failed: {exc}")

    if started_streaming:
        try:
            client.stop_streaming()
        except Exception as exc:
            warnings.append(f"Could not stop preview streaming cleanly: {exc}")

    return SeestarSmartFlatsReport(
        success=bool(measurements),
        ready_for_flats=ready_for_flats,
        auto_triggered=auto_triggered,
        final_exposure_ms=int(measurements[-1].exposure_ms if measurements else exposure_ms),
        final_mean_fraction=float(last_mean),
        final_median_fraction=float(last_median),
        linearity_spread_fraction=float(linearity_spread if math.isfinite(linearity_spread) else 0.0),
        measurements=measurements,
        warnings=warnings,
    )


def run_raw_adu_smart_flats(
    client: Any,
    imager: Any,
    config: SeestarSmartFlatsConfig,
    *,
    lp_filter_mode: str = SEESTAR_ALP_LP_FILTER_AUTO,
    sleep_fn: Any = time.sleep,
) -> SeestarRawFlatsReport:
    cfg = SeestarSmartFlatsConfig.model_validate(config)
    warnings: list[str] = []
    measurements: list[SeestarRawFlatMeasurement] = []
    auto_triggered = False
    started_streaming = False

    try:
        if str(lp_filter_mode or SEESTAR_ALP_LP_FILTER_AUTO).strip().lower() == SEESTAR_ALP_LP_FILTER_OFF:
            client.put_action("method_sync", {"method": "set_wheel_position", "params": [1]})
        elif str(lp_filter_mode or SEESTAR_ALP_LP_FILTER_AUTO).strip().lower() == SEESTAR_ALP_LP_FILTER_ON:
            client.put_action("method_sync", {"method": "set_wheel_position", "params": [2]})
    except Exception as exc:
        warnings.append(f"Could not apply requested filter wheel state before flats: {exc}")

    try:
        client.begin_streaming()
        started_streaming = True
    except Exception as exc:
        warnings.append(f"Could not start preview streaming automatically: {exc}")

    try:
        exposure_ms = max(
            int(cfg.min_exposure_ms),
            min(int(cfg.max_exposure_ms), int(cfg.starting_exposure_ms or 0)),
        )
        if exposure_ms <= 0:
            exposure_ms = max(
                int(cfg.min_exposure_ms),
                min(int(cfg.max_exposure_ms), int(client.get_live_exposure_ms())),
            )
    except Exception:
        exposure_ms = max(
            int(cfg.min_exposure_ms),
            min(int(cfg.max_exposure_ms), SEESTAR_SMART_FLATS_DEFAULT_STARTING_EXPOSURE_MS),
        )

    final_measurement: SeestarRawFlatMeasurement | None = None
    for _ in range(max(1, int(cfg.max_iterations))):
        client.set_live_exposure_ms(exposure_ms)
        settle_s = max(0.0, float(cfg.settle_s))
        if settle_s > 0.0:
            sleep_fn(settle_s)

        measurement = imager.get_raw_stats(
            crop_fraction=float(cfg.crop_fraction),
            sample_count=max(1, int(cfg.samples_per_step)),
            mode="preview",
        )
        measurements.append(measurement)
        final_measurement = measurement

        target_adu = float(measurement.max_adu) * float(cfg.target_fraction)
        tolerance_adu = float(measurement.max_adu) * float(cfg.tolerance_fraction)
        if (
            abs(float(measurement.median_adu) - target_adu) <= tolerance_adu
            and float(measurement.saturated_fraction) <= float(cfg.saturation_tolerance_fraction)
            and float(measurement.quadrant_spread_fraction) <= float(cfg.quadrant_tolerance_fraction)
        ):
            break

        if float(measurement.median_adu) <= 1e-6:
            next_exposure = min(int(cfg.max_exposure_ms), max(int(cfg.min_exposure_ms), exposure_ms * 2))
        else:
            scale = target_adu / float(measurement.median_adu)
            next_exposure = int(round(exposure_ms * scale))
            next_exposure = max(int(cfg.min_exposure_ms), min(int(cfg.max_exposure_ms), next_exposure))
        if next_exposure == exposure_ms:
            if float(measurement.median_adu) < target_adu:
                next_exposure = min(int(cfg.max_exposure_ms), exposure_ms + 1)
            elif float(measurement.median_adu) > target_adu:
                next_exposure = max(int(cfg.min_exposure_ms), exposure_ms - 1)
        if next_exposure == exposure_ms:
            break
        exposure_ms = next_exposure

    if final_measurement is None:
        if started_streaming:
            try:
                client.stop_streaming()
            except Exception:
                pass
        return SeestarRawFlatsReport(success=False, ready_for_flats=False, warnings=["No raw-stat samples were collected."])

    linearity_measurements: list[SeestarRawFlatMeasurement] = []
    by_exposure: dict[int, SeestarRawFlatMeasurement] = {}
    for item in measurements:
        by_exposure[int(item.exposure_ms)] = item
    candidate_exposures: list[int] = []
    for candidate in (
        int(round(final_measurement.exposure_ms * 0.7)),
        int(final_measurement.exposure_ms),
        int(round(final_measurement.exposure_ms * 1.3)),
    ):
        candidate = max(int(cfg.min_exposure_ms), min(int(cfg.max_exposure_ms), candidate))
        if candidate not in candidate_exposures:
            candidate_exposures.append(candidate)

    for candidate in candidate_exposures:
        existing = by_exposure.get(candidate)
        if existing is not None:
            linearity_measurements.append(existing)
            continue
        client.set_live_exposure_ms(candidate)
        settle_s = max(0.0, float(cfg.settle_s))
        if settle_s > 0.0:
            sleep_fn(settle_s)
        measurement = imager.get_raw_stats(
            crop_fraction=float(cfg.crop_fraction),
            sample_count=max(1, int(cfg.samples_per_step)),
            mode="preview",
        )
        measurements.append(measurement)
        by_exposure[int(measurement.exposure_ms)] = measurement
        linearity_measurements.append(measurement)

    linearity_spread = analyze_raw_linearity(linearity_measurements)
    target_adu = float(final_measurement.max_adu) * float(cfg.target_fraction)
    tolerance_adu = float(final_measurement.max_adu) * float(cfg.tolerance_fraction)
    ready_for_flats = (
        abs(float(final_measurement.median_adu) - target_adu) <= tolerance_adu
        and float(final_measurement.saturated_fraction) <= float(cfg.saturation_tolerance_fraction)
        and float(final_measurement.quadrant_spread_fraction) <= float(cfg.quadrant_tolerance_fraction)
    )
    if not ready_for_flats:
        warnings.append("Raw median ADU did not converge close enough to the requested mid-range target.")
    if float(final_measurement.saturated_fraction) > float(cfg.saturation_tolerance_fraction):
        warnings.append(
            f"Saturated fraction is too high ({float(final_measurement.saturated_fraction) * 100.0:.3f}%)."
        )
    if float(final_measurement.quadrant_spread_fraction) > float(cfg.quadrant_tolerance_fraction):
        warnings.append(
            f"Sky gradient is too uneven across quadrants ({float(final_measurement.quadrant_spread_fraction) * 100.0:.1f}%)."
        )
    if len(linearity_measurements) >= 2 and not math.isfinite(linearity_spread):
        warnings.append("Linearity check failed because raw ADU/exposure ratios were not finite.")
        ready_for_flats = False
    elif len(linearity_measurements) >= 2 and linearity_spread > float(cfg.linearity_tolerance_fraction):
        warnings.append(
            f"Raw ADU vs exposure was not linear enough (spread {linearity_spread * 100.0:.1f}%)."
        )
        ready_for_flats = False

    try:
        client.set_live_exposure_ms(int(final_measurement.exposure_ms))
    except Exception as exc:
        warnings.append(f"Could not restore final preview exposure cleanly: {exc}")

    if ready_for_flats and bool(cfg.trigger_flat_capture_when_ready):
        try:
            client.start_create_calib_frame()
            auto_triggered = True
        except Exception as exc:
            warnings.append(f"Flats looked ready but ALP flat capture trigger failed: {exc}")

    if started_streaming:
        try:
            client.stop_streaming()
        except Exception as exc:
            warnings.append(f"Could not stop preview streaming cleanly: {exc}")

    return SeestarRawFlatsReport(
        success=bool(measurements),
        ready_for_flats=bool(ready_for_flats),
        auto_triggered=auto_triggered,
        final_exposure_ms=int(final_measurement.exposure_ms),
        final_bit_depth=int(final_measurement.bit_depth),
        final_max_adu=int(final_measurement.max_adu),
        final_target_adu=float(target_adu),
        final_mean_adu=float(final_measurement.mean_adu),
        final_median_adu=float(final_measurement.median_adu),
        final_percent_of_full_scale=(
            float(final_measurement.median_adu) / max(1.0, float(final_measurement.max_adu))
        ),
        final_saturated_fraction=float(final_measurement.saturated_fraction),
        final_quadrant_spread_fraction=float(final_measurement.quadrant_spread_fraction),
        linearity_spread_fraction=float(linearity_spread if math.isfinite(linearity_spread) else 0.0),
        measurements=measurements,
        warnings=warnings,
    )


class DeviceAdapter(ABC):
    @abstractmethod
    def build_handoff_bundle(self, queue: NightQueue) -> SeestarHandoffBundle:
        raise NotImplementedError

    @abstractmethod
    def open_handoff_dialog(self, bundle: SeestarHandoffBundle, parent: object | None = None) -> object:
        raise NotImplementedError


class SeestarGuidedAdapter(DeviceAdapter):
    def build_handoff_bundle(self, queue: NightQueue) -> SeestarHandoffBundle:
        csv_rows: list[dict[str, Any]] = []
        for idx, block in enumerate(queue.blocks, start=1):
            csv_rows.append(
                {
                    "order": idx,
                    "alias": block.alias,
                    "target_name": block.target_name,
                    "requires_custom_object": "yes",
                    "recommended_filter": block.recommended_filter,
                    "start_local": block.start_local.strftime("%Y-%m-%d %H:%M"),
                    "end_local": block.end_local.strftime("%Y-%m-%d %H:%M"),
                    "window_start_local": block.window_start_local.strftime("%Y-%m-%d %H:%M"),
                    "window_end_local": block.window_end_local.strftime("%Y-%m-%d %H:%M"),
                    "score": f"{block.score:.1f}",
                    "hours_above_limit": f"{block.hours_above_limit:.2f}",
                    "ra_deg": f"{block.ra_deg:.6f}",
                    "dec_deg": f"{block.dec_deg:.6f}",
                    "ra_hms": block.ra_hms,
                    "dec_dms": block.dec_dms,
                    "notes": block.notes,
                }
            )

        session_payload = queue.model_dump(mode="json")
        checklist_markdown = render_handoff_checklist_markdown(queue)
        bundle = SeestarHandoffBundle(
            queue=queue,
            session_payload=session_payload,
            csv_rows=csv_rows,
            dialog_text="",
            checklist_markdown=checklist_markdown,
        )
        bundle.dialog_text = render_handoff_dialog_text(bundle)
        return bundle

    def open_handoff_dialog(self, bundle: SeestarHandoffBundle, parent: object | None = None) -> object:
        from PySide6.QtCore import QTimer, Qt, QUrl
        from PySide6.QtGui import QDesktopServices
        from PySide6.QtWidgets import (
            QApplication,
            QAbstractItemView,
            QCheckBox,
            QDoubleSpinBox,
            QFileDialog,
            QDialog,
            QDialogButtonBox,
            QFormLayout,
            QHeaderView,
            QHBoxLayout,
            QLabel,
            QMessageBox,
            QPushButton,
            QRadioButton,
            QSplitter,
            QSpinBox,
            QTabWidget,
            QTableWidget,
            QTableWidgetItem,
            QTextEdit,
            QVBoxLayout,
            QWidget,
        )

        from astroplanner.exporters import (
            export_seestar_checklist_markdown,
            export_seestar_handoff_csv,
            export_seestar_session_json,
        )
        from astroplanner.qt_helpers import configure_tab_widget

        def _fit_dialog_to_screen(
            dialog: QDialog,
            *,
            preferred_width: int,
            preferred_height: int,
            min_width: int,
            min_height: int,
        ) -> None:
            screen = dialog.screen()
            if screen is None and dialog.parentWidget() is not None:
                screen = dialog.parentWidget().screen()
            if screen is None:
                screen = QApplication.primaryScreen()
            if screen is None:
                dialog.resize(preferred_width, preferred_height)
                dialog.setMinimumSize(min_width, min_height)
                return
            available = screen.availableGeometry()
            target_w = max(min_width, min(preferred_width, available.width() - 16))
            target_h = max(min_height, min(preferred_height, available.height() - 16))
            dialog.setMinimumSize(
                min(max(min_width, 320), target_w),
                min(max(min_height, 240), target_h),
            )
            dialog.resize(target_w, target_h)

        class _SeestarHandoffDialog(QDialog):
            def __init__(self, handoff_bundle: SeestarHandoffBundle, dialog_parent: object | None = None):
                super().__init__(dialog_parent)
                self._bundle = handoff_bundle
                self.setWindowTitle("Seestar Session")
                self.setModal(True)

                root = QVBoxLayout(self)
                root.setContentsMargins(10, 10, 10, 10)
                root.setSpacing(8)

                header = QLabel(_queue_summary_text(self._bundle.queue), self)
                header.setWordWrap(True)
                header.setTextInteractionFlags(Qt.TextSelectableByMouse)
                root.addWidget(header)

                body = QTextEdit(self)
                body.setReadOnly(True)
                body.setPlainText(self._bundle.dialog_text)
                root.addWidget(body, 1)

                footer = QWidget(self)
                footer_l = QHBoxLayout(footer)
                footer_l.setContentsMargins(0, 0, 0, 0)
                footer_l.setSpacing(8)
                footer_hint = QLabel(
                    "Export writes seestar_session.json, seestar_handoff.csv and seestar_checklist.md.",
                    footer,
                )
                footer_hint.setWordWrap(True)
                footer_l.addWidget(footer_hint, 1)
                export_btn = QPushButton("Export Bundle…", footer)
                export_btn.clicked.connect(self._export_bundle)
                footer_l.addWidget(export_btn, 0)
                root.addWidget(footer)

                buttons = QDialogButtonBox(QDialogButtonBox.Close, self)
                buttons.rejected.connect(self.reject)
                buttons.accepted.connect(self.accept)
                root.addWidget(buttons)
                _fit_dialog_to_screen(
                    self,
                    preferred_width=1240,
                    preferred_height=860,
                    min_width=880,
                    min_height=620,
                )
                localize_widget_tree(self, current_language())

            def _export_bundle(self) -> None:
                out_dir = QFileDialog.getExistingDirectory(self, "Select Seestar export directory", str(Path.cwd()))
                if not out_dir:
                    return
                out_path = Path(out_dir)
                export_seestar_session_json(out_path / "seestar_session.json", self._bundle.session_payload)
                export_seestar_handoff_csv(out_path / "seestar_handoff.csv", self._bundle.csv_rows)
                export_seestar_checklist_markdown(out_path / "seestar_checklist.md", self._bundle.checklist_markdown)
                QMessageBox.information(self, "Seestar Export", f"Wrote Seestar bundle to {out_path}")

        dialog = _SeestarHandoffDialog(bundle, dialog_parent=parent)
        dialog.exec()
        return dialog


class SeestarAlpAdapter(DeviceAdapter):
    def __init__(self, config: SeestarAlpConfig):
        self.config = config

    def build_handoff_bundle(self, queue: NightQueue) -> SeestarHandoffBundle:
        schedule_items = build_alp_schedule_items(queue, self.config)
        csv_rows: list[dict[str, Any]] = []
        for idx, item in enumerate(schedule_items, start=1):
            params = item.get("params", {})
            target_name = ""
            ra_hms = ""
            dec_dms = ""
            notes = ""
            if isinstance(params, dict):
                target_name = str(params.get("target_name", "") or "")
                ra_hms = str(params.get("ra", "") or "")
                dec_dms = str(params.get("dec", "") or "")
                try:
                    notes = json.dumps(params, ensure_ascii=False, separators=(",", ":"))
                except Exception:
                    notes = str(params)
            elif isinstance(params, list):
                notes = json.dumps(params, ensure_ascii=False, separators=(",", ":"))
            else:
                notes = str(params)
            csv_rows.append(
                {
                    "order": idx,
                    "alias": target_name,
                    "target_name": target_name,
                    "requires_custom_object": "no",
                    "recommended_filter": str(item.get("action", "") or ""),
                    "start_local": "",
                    "end_local": "",
                    "window_start_local": "",
                    "window_end_local": "",
                    "score": "",
                    "hours_above_limit": "",
                    "ra_deg": "",
                    "dec_deg": "",
                    "ra_hms": ra_hms,
                    "dec_dms": dec_dms,
                    "notes": notes,
                }
            )

        session_payload = {
            "queue": queue.model_dump(mode="json"),
            "alp_service": self.config.model_dump(mode="json"),
            "schedule_items": schedule_items,
        }
        bundle = SeestarHandoffBundle(
            queue=queue,
            session_payload=session_payload,
            csv_rows=csv_rows,
            dialog_text="",
            checklist_markdown=render_alp_checklist_markdown(queue, self.config),
        )
        bundle.dialog_text = render_alp_dialog_text(bundle, self.config)
        return bundle

    def open_handoff_dialog(self, bundle: SeestarHandoffBundle, parent: object | None = None) -> object:
        from PySide6.QtCore import QTimer, Qt, QUrl
        from PySide6.QtGui import QDesktopServices
        from PySide6.QtWidgets import (
            QApplication,
            QAbstractItemView,
            QCheckBox,
            QDoubleSpinBox,
            QFileDialog,
            QDialog,
            QDialogButtonBox,
            QFormLayout,
            QHeaderView,
            QHBoxLayout,
            QLabel,
            QMessageBox,
            QPushButton,
            QRadioButton,
            QSplitter,
            QSpinBox,
            QTabWidget,
            QTableWidget,
            QTableWidgetItem,
            QTextEdit,
            QVBoxLayout,
            QWidget,
        )

        from astroplanner.exporters import (
            export_seestar_checklist_markdown,
            export_seestar_handoff_csv,
            export_seestar_session_json,
        )

        adapter = self

        def _fit_dialog_to_screen(
            dialog: QDialog,
            *,
            preferred_width: int,
            preferred_height: int,
            min_width: int,
            min_height: int,
        ) -> None:
            screen = dialog.screen()
            if screen is None and dialog.parentWidget() is not None:
                screen = dialog.parentWidget().screen()
            if screen is None:
                screen = QApplication.primaryScreen()
            if screen is None:
                dialog.resize(preferred_width, preferred_height)
                dialog.setMinimumSize(min_width, min_height)
                return
            available = screen.availableGeometry()
            target_w = max(min_width, min(preferred_width, available.width() - 16))
            target_h = max(min_height, min(preferred_height, available.height() - 16))
            dialog.setMinimumSize(
                min(max(min_width, 320), target_w),
                min(max(min_height, 240), target_h),
            )
            dialog.resize(target_w, target_h)

        class _SeestarAlpDialog(QDialog):
            def __init__(self, handoff_bundle: SeestarHandoffBundle, dialog_parent: object | None = None):
                super().__init__(dialog_parent)
                self._bundle = handoff_bundle
                self._client = SeestarAlpClient(adapter.config)
                self._last_error_text = ""
                self._science_checklist_confirmed = False
                self.setWindowTitle("Seestar ALP Session")
                self.setModal(True)

                root = QVBoxLayout(self)
                root.setContentsMargins(10, 10, 10, 10)
                root.setSpacing(8)

                header = QLabel(_queue_summary_text(self._bundle.queue), self)
                header.setWordWrap(True)
                header.setTextInteractionFlags(Qt.TextSelectableByMouse)
                root.addWidget(header)

                self._status_label = QLabel(
                    render_alp_backend_status_text(
                        None,
                        base_url=adapter.config.base_url,
                        device_num=adapter.config.device_num,
                    ),
                    self,
                )
                self._status_label.setObjectName("SectionHint")
                self._status_label.setWordWrap(True)
                self._status_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
                root.addWidget(self._status_label)

                tabs = configure_tab_widget(QTabWidget(self))
                tabs.setAttribute(Qt.WA_StyledBackground, True)
                root.addWidget(tabs, 1)

                plan_tab = QWidget(tabs)
                plan_tab.setAttribute(Qt.WA_StyledBackground, True)
                plan_tab_l = QVBoxLayout(plan_tab)
                plan_tab_l.setContentsMargins(8, 8, 8, 8)
                plan_tab_l.setSpacing(8)

                plan_hint = QLabel(
                    "Push Queue uploads the exact ALP item list. Stale wait-until rows are annotated inline when Push + Start would skip them.",
                    plan_tab,
                )
                plan_hint.setObjectName("SectionHint")
                plan_hint.setWordWrap(True)
                plan_tab_l.addWidget(plan_hint)

                plan_splitter = QSplitter(Qt.Horizontal, plan_tab)
                plan_splitter.setChildrenCollapsible(False)
                plan_tab_l.addWidget(plan_splitter, 1)

                schedule_panel = QWidget(plan_splitter)
                schedule_panel_l = QVBoxLayout(schedule_panel)
                schedule_panel_l.setContentsMargins(0, 0, 0, 0)
                schedule_panel_l.setSpacing(6)
                schedule_title = QLabel("ALP Schedule", schedule_panel)
                schedule_title.setObjectName("SectionTitle")
                schedule_panel_l.addWidget(schedule_title)
                self._schedule_table = QTableWidget(schedule_panel)
                self._schedule_table.setColumnCount(3)
                self._schedule_table.setHorizontalHeaderLabels(
                    [translate_text(label, current_language()) for label in ["#", "Action", "Detail"]]
                )
                self._schedule_table.verticalHeader().setVisible(False)
                self._schedule_table.setAlternatingRowColors(True)
                self._schedule_table.setSelectionMode(QAbstractItemView.NoSelection)
                self._schedule_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
                schedule_header = self._schedule_table.horizontalHeader()
                schedule_header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
                schedule_header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
                schedule_header.setSectionResizeMode(2, QHeaderView.Stretch)
                schedule_panel_l.addWidget(self._schedule_table, 1)
                plan_splitter.addWidget(schedule_panel)

                blocks_panel = QWidget(plan_splitter)
                blocks_panel_l = QVBoxLayout(blocks_panel)
                blocks_panel_l.setContentsMargins(0, 0, 0, 0)
                blocks_panel_l.setSpacing(6)
                blocks_title = QLabel("Queue Blocks", blocks_panel)
                blocks_title.setObjectName("SectionTitle")
                blocks_panel_l.addWidget(blocks_title)
                self._blocks_table = QTableWidget(blocks_panel)
                self._blocks_table.setColumnCount(6)
                self._blocks_table.setHorizontalHeaderLabels(
                    [translate_text(label, current_language()) for label in ["Target", "Start", "End", "Run", "Filter", "Notes"]]
                )
                self._blocks_table.verticalHeader().setVisible(False)
                self._blocks_table.setAlternatingRowColors(True)
                self._blocks_table.setSelectionMode(QAbstractItemView.NoSelection)
                self._blocks_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
                blocks_header = self._blocks_table.horizontalHeader()
                blocks_header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
                for column in (1, 2, 3, 4):
                    blocks_header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
                blocks_header.setSectionResizeMode(5, QHeaderView.Stretch)
                blocks_panel_l.addWidget(self._blocks_table, 1)
                plan_splitter.addWidget(blocks_panel)
                plan_splitter.setStretchFactor(0, 1)
                plan_splitter.setStretchFactor(1, 1)
                plan_splitter.setSizes([620, 520])

                tabs.addTab(plan_tab, "Plan")

                notes_tab = QWidget(tabs)
                notes_tab.setAttribute(Qt.WA_StyledBackground, True)
                notes_tab_l = QVBoxLayout(notes_tab)
                notes_tab_l.setContentsMargins(8, 8, 8, 8)
                notes_tab_l.setSpacing(8)
                notes_splitter = QSplitter(Qt.Horizontal, notes_tab)
                notes_splitter.setChildrenCollapsible(False)
                notes_tab_l.addWidget(notes_splitter, 1)

                notes_panel = QWidget(notes_splitter)
                notes_panel_l = QVBoxLayout(notes_panel)
                notes_panel_l.setContentsMargins(0, 0, 0, 0)
                notes_panel_l.setSpacing(6)
                notes_title = QLabel("Session Notes", notes_panel)
                notes_title.setObjectName("SectionTitle")
                notes_panel_l.addWidget(notes_title)
                self._notes_body = QTextEdit(notes_panel)
                self._notes_body.setReadOnly(True)
                self._notes_body.setPlainText(str(self._bundle.queue.campaign_notes or "").strip() or "No session notes.")
                notes_panel_l.addWidget(self._notes_body, 1)
                notes_splitter.addWidget(notes_panel)

                checklist_panel = QWidget(notes_splitter)
                checklist_panel_l = QVBoxLayout(checklist_panel)
                checklist_panel_l.setContentsMargins(0, 0, 0, 0)
                checklist_panel_l.setSpacing(6)
                checklist_title = QLabel("Science Checklist", checklist_panel)
                checklist_title.setObjectName("SectionTitle")
                checklist_panel_l.addWidget(checklist_title)
                self._checklist_body = QTextEdit(checklist_panel)
                self._checklist_body.setReadOnly(True)
                checklist_items = [
                    str(item).strip()
                    for item in self._bundle.queue.science_checklist_items
                    if str(item).strip()
                ]
                checklist_text = (
                    "\n".join(f"{idx}. {item}" for idx, item in enumerate(checklist_items, start=1))
                    if bool(getattr(self._bundle.queue, "require_science_checklist", False)) and checklist_items
                    else "Checklist confirmation is not required for this session."
                )
                self._checklist_body.setPlainText(checklist_text)
                checklist_panel_l.addWidget(self._checklist_body, 1)
                notes_splitter.addWidget(checklist_panel)
                notes_splitter.setStretchFactor(0, 1)
                notes_splitter.setStretchFactor(1, 1)
                notes_splitter.setSizes([520, 520])

                tabs.addTab(notes_tab, "Notes")

                advanced_tab = QWidget(tabs)
                advanced_tab.setAttribute(Qt.WA_StyledBackground, True)
                advanced_tab_l = QVBoxLayout(advanced_tab)
                advanced_tab_l.setContentsMargins(8, 8, 8, 8)
                advanced_tab_l.setSpacing(8)
                advanced_hint = QLabel(
                    "Advanced text preview of the same session. Useful for copy/paste and export verification.",
                    advanced_tab,
                )
                advanced_hint.setObjectName("SectionHint")
                advanced_hint.setWordWrap(True)
                advanced_tab_l.addWidget(advanced_hint)
                self._detail_body = QTextEdit(advanced_tab)
                self._detail_body.setReadOnly(True)
                advanced_tab_l.addWidget(self._detail_body, 1)
                tabs.addTab(advanced_tab, "Advanced")

                controls = QWidget(self)
                controls_l = QHBoxLayout(controls)
                controls_l.setContentsMargins(0, 0, 0, 0)
                controls_l.setSpacing(8)
                hint = QLabel(
                    f"Target ALP service: {adapter.config.base_url.rstrip('/')} | device {adapter.config.device_num}",
                    controls,
                )
                hint.setWordWrap(True)
                controls_l.addWidget(hint, 1)
                refresh_btn = QPushButton("Refresh Status", controls)
                refresh_btn.clicked.connect(self._refresh_status)
                controls_l.addWidget(refresh_btn, 0)
                test_btn = QPushButton("Test ALP", controls)
                test_btn.clicked.connect(self._test_connection)
                controls_l.addWidget(test_btn, 0)
                open_ui_btn = QPushButton("Open ALP Web UI", controls)
                open_ui_btn.clicked.connect(self._open_web_ui)
                controls_l.addWidget(open_ui_btn, 0)
                smart_flats_btn = QPushButton("Run Smart Sky Flats…", controls)
                smart_flats_btn.clicked.connect(self._open_smart_flats_dialog)
                controls_l.addWidget(smart_flats_btn, 0)
                push_btn = QPushButton("Push Queue", controls)
                push_btn.clicked.connect(lambda: self._push_queue(start_immediately=False))
                controls_l.addWidget(push_btn, 0)
                start_btn = QPushButton("Push + Start", controls)
                start_btn.clicked.connect(lambda: self._push_queue(start_immediately=True))
                controls_l.addWidget(start_btn, 0)
                if is_seestar_debug_enabled():
                    sample_btn = QPushButton("Test ALP + Push sample", controls)
                    sample_btn.clicked.connect(self._push_sample_queue)
                    controls_l.addWidget(sample_btn, 0)
                export_btn = QPushButton("Export Bundle…", controls)
                export_btn.clicked.connect(self._export_bundle)
                controls_l.addWidget(export_btn, 0)
                root.addWidget(controls)

                buttons = QDialogButtonBox(QDialogButtonBox.Close, self)
                buttons.rejected.connect(self.reject)
                buttons.accepted.connect(self.accept)
                root.addWidget(buttons)
                _fit_dialog_to_screen(
                    self,
                    preferred_width=1660,
                    preferred_height=980,
                    min_width=1240,
                    min_height=760,
                )
                localize_widget_tree(self, current_language())

                self._refresh_preview_content()
                QTimer.singleShot(0, self._refresh_status)

            def _populate_readonly_table(self, table: QTableWidget, rows: list[dict[str, str]], columns: list[str]) -> None:
                table.setRowCount(len(rows))
                for row_idx, row in enumerate(rows):
                    for col_idx, key in enumerate(columns):
                        item = QTableWidgetItem(str(row.get(key, "") or ""))
                        item.setFlags(item.flags() & ~Qt.ItemIsEditable & ~Qt.ItemIsSelectable)
                        table.setItem(row_idx, col_idx, item)
                table.resizeRowsToContents()

            def _refresh_preview_content(self) -> None:
                schedule_items = list(self._bundle.session_payload.get("schedule_items", []))
                schedule_rows = build_alp_schedule_preview_rows(self._bundle.queue, schedule_items)
                self._populate_readonly_table(
                    self._schedule_table,
                    schedule_rows,
                    ["order", "action", "detail"],
                )
                block_rows = build_queue_block_preview_rows(self._bundle.queue, adapter.config)
                self._populate_readonly_table(
                    self._blocks_table,
                    block_rows,
                    ["target", "start", "end", "run", "filter", "notes"],
                )
                self._detail_body.setPlainText(render_alp_dialog_text(self._bundle, adapter.config))

            def _update_status_label(self, status: SeestarAlpBackendStatus | None) -> None:
                self._status_label.setText(
                    render_alp_backend_status_text(
                        status,
                        last_error=self._last_error_text,
                        base_url=adapter.config.base_url,
                        device_num=adapter.config.device_num,
                    )
                )

            def _refresh_status(self, *, show_message: bool = False) -> None:
                self._refresh_preview_content()
                try:
                    status = self._client.get_backend_status()
                    self._last_error_text = ""
                    self._update_status_label(status)
                except Exception as exc:
                    self._last_error_text = str(exc)
                    self._update_status_label(None)
                    if show_message:
                        QMessageBox.warning(self, "Seestar ALP", self._last_error_text)
                    return
                if show_message:
                    QMessageBox.information(self, "Seestar ALP", self._status_label.text())

            def _test_connection(self) -> None:
                self._refresh_status(show_message=True)

            def _open_web_ui(self) -> None:
                from PySide6.QtCore import QUrl
                from PySide6.QtGui import QDesktopServices

                url = build_alp_web_ui_url(adapter.config.base_url)
                if not QDesktopServices.openUrl(QUrl(url)):
                    QMessageBox.warning(self, "Seestar ALP", f"Unable to open URL:\n{url}")

            def _ensure_science_checklist_confirmed(self) -> bool:
                if not bool(getattr(self._bundle.queue, "require_science_checklist", False)):
                    return True
                items = [str(item).strip() for item in self._bundle.queue.science_checklist_items if str(item).strip()]
                if not items or self._science_checklist_confirmed:
                    return True

                dialog = QDialog(self)
                dialog.setWindowTitle("Science Checklist")
                dialog.setModal(True)
                root = QVBoxLayout(dialog)
                root.setContentsMargins(10, 10, 10, 10)
                root.setSpacing(8)

                intro = QLabel(
                    "Confirm every science preflight item before uploading this queue to Seestar ALP.",
                    dialog,
                )
                intro.setWordWrap(True)
                root.addWidget(intro)

                if str(self._bundle.queue.campaign_name or "").strip():
                    template_label = QLabel(f"Template: {self._bundle.queue.campaign_name.strip()}", dialog)
                    template_label.setWordWrap(True)
                    template_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
                    root.addWidget(template_label)

                if str(self._bundle.queue.campaign_notes or "").strip():
                    notes_label = QLabel(f"Notes: {self._bundle.queue.campaign_notes.strip()}", dialog)
                    notes_label.setWordWrap(True)
                    notes_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
                    root.addWidget(notes_label)

                checkboxes: list[QCheckBox] = []
                for item in items:
                    checkbox = QCheckBox(item, dialog)
                    root.addWidget(checkbox)
                    checkboxes.append(checkbox)

                buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dialog)
                ok_btn = buttons.button(QDialogButtonBox.Ok)
                if ok_btn is not None:
                    ok_btn.setText("Confirm Checklist")
                    ok_btn.setEnabled(False)

                def _sync_ok() -> None:
                    if ok_btn is not None:
                        ok_btn.setEnabled(all(box.isChecked() for box in checkboxes))

                for checkbox in checkboxes:
                    checkbox.toggled.connect(_sync_ok)
                buttons.accepted.connect(dialog.accept)
                buttons.rejected.connect(dialog.reject)
                root.addWidget(buttons)
                _sync_ok()
                _fit_dialog_to_screen(
                    dialog,
                    preferred_width=760,
                    preferred_height=560,
                    min_width=560,
                    min_height=420,
                )

                if dialog.exec() != int(QDialog.Accepted):
                    return False
                self._science_checklist_confirmed = True
                return True

            def _open_smart_flats_dialog(self) -> None:
                class _SmartSkyFlatsDialog(QDialog):
                    def __init__(self, dialog_parent: object | None = None):
                        super().__init__(dialog_parent)
                        self._imager_client = SeestarAlpImagerClient(adapter.config)
                        self._raw_supported = False
                        self._raw_support_reason = ""
                        try:
                            self._raw_supported = self._imager_client.supports_raw_flats()
                        except Exception as exc:
                            self._raw_support_reason = str(exc)
                            self._raw_supported = False
                        if not self._raw_supported and not self._raw_support_reason:
                            self._raw_support_reason = (
                                "This ALP backend does not expose the /raw-stats endpoint yet."
                            )

                        self.setWindowTitle("Smart Sky Flats")
                        self.setModal(True)

                        root = QVBoxLayout(self)
                        root.setContentsMargins(10, 10, 10, 10)
                        root.setSpacing(8)

                        intro = QLabel(
                            "Choose a smart-flats probe mode. Raw ADU is preferred when the backend exposes "
                            "raw-statistics support. Preview brightness remains available as a fallback/debug path.",
                            self,
                        )
                        intro.setWordWrap(True)
                        root.addWidget(intro)

                        mode_row = QWidget(self)
                        mode_row_l = QHBoxLayout(mode_row)
                        mode_row_l.setContentsMargins(0, 0, 0, 0)
                        mode_row_l.setSpacing(12)
                        self.raw_radio = QRadioButton("Raw ADU (Recommended)", mode_row)
                        self.preview_radio = QRadioButton("Preview Brightness (Fallback)", mode_row)
                        if self._raw_supported:
                            self.raw_radio.setChecked(True)
                        else:
                            self.raw_radio.setEnabled(False)
                            self.preview_radio.setChecked(True)
                        mode_row_l.addWidget(self.raw_radio, 0)
                        mode_row_l.addWidget(self.preview_radio, 0)
                        mode_row_l.addStretch(1)
                        root.addWidget(mode_row)

                        self.mode_hint = QLabel("", self)
                        self.mode_hint.setWordWrap(True)
                        self.mode_hint.setObjectName("SectionHint")
                        root.addWidget(self.mode_hint)

                        form_widget = QWidget(self)
                        form = QFormLayout(form_widget)
                        form.setContentsMargins(0, 0, 0, 0)
                        form.setSpacing(8)

                        self.target_spin = QDoubleSpinBox(self)
                        self.target_spin.setRange(10.0, 90.0)
                        self.target_spin.setDecimals(1)
                        self.target_spin.setSuffix(" %")
                        self.target_spin.setValue(SEESTAR_SMART_FLATS_DEFAULT_TARGET_FRACTION * 100.0)
                        form.addRow("Target brightness:", self.target_spin)

                        self.tolerance_spin = QDoubleSpinBox(self)
                        self.tolerance_spin.setRange(1.0, 30.0)
                        self.tolerance_spin.setDecimals(1)
                        self.tolerance_spin.setSuffix(" %")
                        self.tolerance_spin.setValue(SEESTAR_SMART_FLATS_DEFAULT_TOLERANCE_FRACTION * 100.0)
                        form.addRow("Tolerance:", self.tolerance_spin)

                        self.crop_spin = QDoubleSpinBox(self)
                        self.crop_spin.setRange(0.3, 1.0)
                        self.crop_spin.setDecimals(2)
                        self.crop_spin.setSingleStep(0.05)
                        self.crop_spin.setValue(SEESTAR_SMART_FLATS_DEFAULT_CROP_FRACTION)
                        form.addRow("Center crop:", self.crop_spin)

                        self.min_exp_spin = QSpinBox(self)
                        self.min_exp_spin.setRange(1, 200)
                        self.min_exp_spin.setSuffix(" ms")
                        self.min_exp_spin.setValue(SEESTAR_SMART_FLATS_DEFAULT_MIN_EXPOSURE_MS)
                        form.addRow("Min preview exp:", self.min_exp_spin)

                        self.max_exp_spin = QSpinBox(self)
                        self.max_exp_spin.setRange(1, 200)
                        self.max_exp_spin.setSuffix(" ms")
                        self.max_exp_spin.setValue(SEESTAR_SMART_FLATS_DEFAULT_MAX_EXPOSURE_MS)
                        form.addRow("Max preview exp:", self.max_exp_spin)

                        self.start_exp_spin = QSpinBox(self)
                        self.start_exp_spin.setRange(1, 200)
                        self.start_exp_spin.setSuffix(" ms")
                        self.start_exp_spin.setValue(SEESTAR_SMART_FLATS_DEFAULT_STARTING_EXPOSURE_MS)
                        form.addRow("Starting exp:", self.start_exp_spin)

                        self.samples_spin = QSpinBox(self)
                        self.samples_spin.setRange(1, 5)
                        self.samples_spin.setValue(SEESTAR_SMART_FLATS_DEFAULT_SAMPLES_PER_STEP)
                        form.addRow("Samples/step:", self.samples_spin)

                        self.iter_spin = QSpinBox(self)
                        self.iter_spin.setRange(1, 12)
                        self.iter_spin.setValue(SEESTAR_SMART_FLATS_DEFAULT_MAX_ITERATIONS)
                        form.addRow("Max iterations:", self.iter_spin)

                        self.settle_spin = QDoubleSpinBox(self)
                        self.settle_spin.setRange(0.0, 10.0)
                        self.settle_spin.setDecimals(1)
                        self.settle_spin.setSingleStep(0.5)
                        self.settle_spin.setSuffix(" s")
                        self.settle_spin.setValue(SEESTAR_SMART_FLATS_DEFAULT_SETTLE_S)
                        form.addRow("Settle time:", self.settle_spin)

                        self.linearity_spin = QDoubleSpinBox(self)
                        self.linearity_spin.setRange(5.0, 100.0)
                        self.linearity_spin.setDecimals(1)
                        self.linearity_spin.setSuffix(" %")
                        self.linearity_spin.setValue(SEESTAR_SMART_FLATS_DEFAULT_LINEARITY_TOLERANCE * 100.0)
                        form.addRow("Linearity spread:", self.linearity_spin)

                        self.saturation_spin = QDoubleSpinBox(self)
                        self.saturation_spin.setRange(0.001, 5.0)
                        self.saturation_spin.setDecimals(3)
                        self.saturation_spin.setSuffix(" %")
                        self.saturation_spin.setValue(
                            SEESTAR_SMART_FLATS_DEFAULT_RAW_SATURATION_TOLERANCE * 100.0
                        )
                        form.addRow("Saturated pixels:", self.saturation_spin)

                        self.quadrant_spin = QDoubleSpinBox(self)
                        self.quadrant_spin.setRange(1.0, 50.0)
                        self.quadrant_spin.setDecimals(1)
                        self.quadrant_spin.setSuffix(" %")
                        self.quadrant_spin.setValue(
                            SEESTAR_SMART_FLATS_DEFAULT_RAW_QUADRANT_TOLERANCE * 100.0
                        )
                        form.addRow("Quadrant spread:", self.quadrant_spin)

                        self.auto_trigger_chk = QCheckBox("Trigger ALP flat capture automatically when ready", self)
                        form.addRow(self.auto_trigger_chk)

                        root.addWidget(form_widget)

                        self.report_body = QTextEdit(self)
                        self.report_body.setReadOnly(True)
                        self.report_body.setPlainText("")
                        root.addWidget(self.report_body, 1)

                        controls = QWidget(self)
                        controls_l = QHBoxLayout(controls)
                        controls_l.setContentsMargins(0, 0, 0, 0)
                        controls_l.setSpacing(8)
                        run_btn = QPushButton("Run", controls)
                        run_btn.clicked.connect(self._run_probe)
                        controls_l.addWidget(run_btn, 0)
                        run_trigger_btn = QPushButton("Run + Trigger Flats", controls)
                        run_trigger_btn.clicked.connect(lambda: self._run_probe(trigger=True))
                        controls_l.addWidget(run_trigger_btn, 0)
                        close_btn = QPushButton("Close", controls)
                        close_btn.clicked.connect(self.accept)
                        controls_l.addWidget(close_btn, 0)
                        controls_l.addStretch(1)
                        root.addWidget(controls)

                        self.raw_radio.toggled.connect(self._sync_mode_text)
                        self.preview_radio.toggled.connect(self._sync_mode_text)
                        self._sync_mode_text()
                        _fit_dialog_to_screen(
                            self,
                            preferred_width=940,
                            preferred_height=760,
                            min_width=680,
                            min_height=560,
                        )
                        localize_widget_tree(self, current_language())

                    def _config(self, *, trigger: bool) -> SeestarSmartFlatsConfig:
                        min_exp = min(int(self.min_exp_spin.value()), int(self.max_exp_spin.value()))
                        max_exp = max(int(self.min_exp_spin.value()), int(self.max_exp_spin.value()))
                        start_exp = min(max(int(self.start_exp_spin.value()), min_exp), max_exp)
                        return SeestarSmartFlatsConfig(
                            target_fraction=float(self.target_spin.value()) / 100.0,
                            tolerance_fraction=float(self.tolerance_spin.value()) / 100.0,
                            crop_fraction=float(self.crop_spin.value()),
                            min_exposure_ms=min_exp,
                            max_exposure_ms=max_exp,
                            starting_exposure_ms=start_exp,
                            settle_s=float(self.settle_spin.value()),
                            samples_per_step=max(1, int(self.samples_spin.value())),
                            max_iterations=max(1, int(self.iter_spin.value())),
                            linearity_tolerance_fraction=float(self.linearity_spin.value()) / 100.0,
                            saturation_tolerance_fraction=float(self.saturation_spin.value()) / 100.0,
                            quadrant_tolerance_fraction=float(self.quadrant_spin.value()) / 100.0,
                            trigger_flat_capture_when_ready=bool(trigger or self.auto_trigger_chk.isChecked()),
                        )

                    def _sync_mode_text(self) -> None:
                        if self.raw_radio.isChecked():
                            self.mode_hint.setText(
                                "Raw ADU mode uses `/raw-stats` from Seestar ALP to target roughly half-scale median ADU, "
                                "then checks saturation, gradient and linearity before optionally triggering flat capture."
                            )
                            self.report_body.setPlaceholderText(
                                "Press Run to probe raw ADU. Results will appear here."
                            )
                        else:
                            suffix = ""
                            if not self._raw_supported:
                                suffix = f" Raw ADU unavailable: {self._raw_support_reason}"
                            self.mode_hint.setText(
                                "Preview fallback estimates luminance from preview JPEG frames. "
                                "Use it only when raw-stat support is unavailable or for quick debugging."
                                + suffix
                            )
                            self.report_body.setPlaceholderText(
                                "Press Run to probe preview brightness. Results will appear here."
                            )

                    def _run_probe(self, *, trigger: bool = False) -> None:
                        cfg = self._config(trigger=trigger)
                        client = SeestarAlpClient(adapter.config)
                        try:
                            if self.raw_radio.isChecked():
                                report = run_raw_adu_smart_flats(
                                    client,
                                    self._imager_client,
                                    cfg,
                                    lp_filter_mode=adapter.config.lp_filter_mode,
                                )
                                report_text = render_raw_flats_report_text(report)
                                ready_title = "Raw ADU looks suitable for twilight flats."
                                fail_title = (
                                    "Raw ADU did not meet the requested readiness thresholds. "
                                    "Review the report and adjust twilight timing or pointing."
                                )
                            else:
                                report = run_smart_sky_flats(
                                    client,
                                    self._imager_client,
                                    cfg,
                                    lp_filter_mode=adapter.config.lp_filter_mode,
                                )
                                report_text = render_smart_flats_report_text(report)
                                ready_title = "Preview brightness looks suitable for twilight flats."
                                fail_title = (
                                    "Preview brightness was not close enough to the requested mid-range target. "
                                    "Review the report and adjust twilight timing or pointing."
                                )
                        except Exception as exc:
                            QMessageBox.warning(
                                self,
                                "Smart Sky Flats",
                                f"Smart flats run failed:\n{exc}",
                            )
                            return
                        self.report_body.setPlainText(report_text)
                        if report.ready_for_flats:
                            QMessageBox.information(
                                self,
                                "Smart Sky Flats",
                                ready_title + (" ALP flat capture was triggered." if report.auto_triggered else ""),
                            )
                        else:
                            QMessageBox.warning(
                                self,
                                "Smart Sky Flats",
                                fail_title,
                            )

                dialog = _SmartSkyFlatsDialog(self)
                dialog.exec()

            def _push_queue(self, *, start_immediately: bool) -> None:
                if not self._ensure_science_checklist_confirmed():
                    return
                schedule_items = list(self._bundle.session_payload.get("schedule_items", []))
                now_local = datetime.now(self._bundle.queue.night_start_local.tzinfo or self._bundle.queue.generated_at.tzinfo)
                try:
                    result = self._client.push_queue(
                        self._bundle.queue,
                        schedule_items=schedule_items,
                        start_immediately=start_immediately,
                        now_local=now_local,
                    )
                except Exception as exc:
                    self._last_error_text = str(exc)
                    self._update_status_label(None)
                    QMessageBox.warning(self, "Seestar ALP", self._last_error_text)
                    return
                self._last_error_text = ""
                self._refresh_status(show_message=False)
                action_txt = "started" if result["started"] else "uploaded"
                skipped_waits = max(0, int(result.get("skipped_waits", 0) or 0))
                QMessageBox.information(
                    self,
                    "Seestar ALP",
                    (
                        f"ALP schedule {result['schedule_id']} {action_txt}. Uploaded {result['uploaded_items']} item(s)."
                        + (
                            f" Skipped {skipped_waits} wait_until item(s) for immediate start."
                            if start_immediately and skipped_waits > 0
                            else ""
                        )
                    ),
                )

            def _push_sample_queue(self) -> None:
                sample_queue = build_alp_debug_sample_queue(self._bundle.queue)
                if not sample_queue.blocks:
                    QMessageBox.information(self, "Seestar ALP", "No queue blocks are available for sample push.")
                    return
                sample_items = build_alp_schedule_items(sample_queue, adapter.config)
                try:
                    self._client.test_connection()
                    result = self._client.push_queue(
                        sample_queue,
                        schedule_items=sample_items,
                        start_immediately=False,
                    )
                except Exception as exc:
                    self._last_error_text = str(exc)
                    self._update_status_label(None)
                    QMessageBox.warning(self, "Seestar ALP", self._last_error_text)
                    return
                self._last_error_text = ""
                self._refresh_status(show_message=False)
                QMessageBox.information(
                    self,
                    "Seestar ALP",
                    f"ALP sample schedule {result['schedule_id']} uploaded with {result['uploaded_items']} short item(s).",
                )

            def _export_bundle(self) -> None:
                out_dir = QFileDialog.getExistingDirectory(self, "Select Seestar export directory", str(Path.cwd()))
                if not out_dir:
                    return
                out_path = Path(out_dir)
                export_seestar_session_json(out_path / "seestar_session.json", self._bundle.session_payload)
                export_seestar_handoff_csv(out_path / "seestar_handoff.csv", self._bundle.csv_rows)
                export_seestar_checklist_markdown(out_path / "seestar_checklist.md", self._bundle.checklist_markdown)
                QMessageBox.information(self, "Seestar Export", f"Wrote Seestar bundle to {out_path}")

        dialog = _SeestarAlpDialog(bundle, dialog_parent=parent)
        dialog.exec()
        return dialog
