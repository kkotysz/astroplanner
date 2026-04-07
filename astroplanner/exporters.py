from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


def export_metrics_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    fieldnames = [
        "name",
        "ra_deg",
        "dec_deg",
        "priority",
        "observed",
        "score",
        "hours_above_limit",
        "max_altitude_deg",
        "peak_moon_sep_deg",
        "window_start_local",
        "window_end_local",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _ics_escape(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace(",", "\\,")
        .replace(";", "\\;")
        .replace("\n", "\\n")
    )


def _to_ics_local(dt: datetime) -> str:
    return dt.strftime("%Y%m%dT%H%M%S")


def export_observation_ics(
    path: Path,
    tz_name: str,
    events: Iterable[dict[str, object]],
) -> None:
    now = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//AstroPlanner//Observation Planner//EN",
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH",
    ]

    for idx, event in enumerate(events):
        start = event.get("start")
        end = event.get("end")
        if not isinstance(start, datetime) or not isinstance(end, datetime):
            continue
        if end <= start:
            continue
        title = _ics_escape(str(event.get("title", "Observation window")))
        desc = _ics_escape(str(event.get("description", "")))
        uid = f"astroplanner-{idx}-{_to_ics_local(start)}"

        lines.extend(
            [
                "BEGIN:VEVENT",
                f"UID:{uid}",
                f"DTSTAMP:{now}",
                f"DTSTART;TZID={tz_name}:{_to_ics_local(start)}",
                f"DTEND;TZID={tz_name}:{_to_ics_local(end)}",
                f"SUMMARY:{title}",
                f"DESCRIPTION:{desc}",
                "END:VEVENT",
            ]
        )

    lines.append("END:VCALENDAR")
    path.write_text("\r\n".join(lines) + "\r\n", encoding="utf-8")


def export_seestar_session_json(path: Path, payload: Any) -> None:
    data = payload.model_dump(mode="json") if hasattr(payload, "model_dump") else payload
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def export_seestar_handoff_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    fieldnames = [
        "order",
        "alias",
        "target_name",
        "requires_custom_object",
        "recommended_filter",
        "start_local",
        "end_local",
        "window_start_local",
        "window_end_local",
        "score",
        "hours_above_limit",
        "ra_deg",
        "dec_deg",
        "ra_hms",
        "dec_dms",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def export_seestar_checklist_markdown(path: Path, markdown: str) -> None:
    path.write_text(str(markdown), encoding="utf-8")
