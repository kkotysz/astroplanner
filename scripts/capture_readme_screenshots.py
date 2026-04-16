#!/usr/bin/env python3
"""Capture README screenshots from live Qt widgets.

This script generates fresh base screenshots used by README:
- dashboard-main.png
- observatory-manager.png
- suggest-targets.png
- ai-assistant.png
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence

from PySide6.QtCore import QDate, QEventLoop, QTimer, Qt
from PySide6.QtWidgets import QApplication, QWidget

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from astro_planner import (
    BHTOM_API_BASE_URL,
    MainWindow,
    ObservatoryManagerDialog,
    SuggestedTargetsDialog,
    Target,
)
from astroplanner.scoring import TargetNightMetrics


DEFAULT_VIEWS = ("dashboard", "observatory", "suggest", "ai")
VIEW_TO_FILE = {
    "dashboard": "dashboard-main.png",
    "observatory": "observatory-manager.png",
    "suggest": "suggest-targets.png",
    "ai": "ai-assistant.png",
}


@dataclass(frozen=True)
class DemoSuggestion:
    name: str
    ra: float
    dec: float
    magnitude: float
    obj_type: str
    priority: int
    importance: float
    score: float
    min_airmass: float
    min_moon_sep: float
    hours_above_limit: float
    max_altitude: float


def _wait(ms: int) -> None:
    loop = QEventLoop()
    QTimer.singleShot(max(0, int(ms)), loop.quit)
    loop.exec()


def _save_widget_grab(widget: QWidget, output_path: Path) -> None:
    QApplication.processEvents()
    pixmap = widget.grab()
    dpr = float(pixmap.devicePixelRatio())
    if dpr > 1.01:
        logical_w = max(1, int(round(pixmap.width() / dpr)))
        logical_h = max(1, int(round(pixmap.height() / dpr)))
        pixmap = pixmap.scaled(
            logical_w,
            logical_h,
            Qt.IgnoreAspectRatio,
            Qt.SmoothTransformation,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not pixmap.save(str(output_path)):
        raise RuntimeError(f"Failed to save screenshot: {output_path}")


def _load_plan_targets(plan_path: Path) -> list[Target]:
    with plan_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    targets: list[Target] = []
    for item in payload:
        targets.append(Target(**item))
    return targets


def _apply_targets(win: MainWindow, targets: Sequence[Target]) -> None:
    win.targets.clear()
    win.target_metrics.clear()
    win.target_windows.clear()
    for target in targets:
        win.targets.append(target)
    win._clear_table_dynamic_cache()
    win.table_model.layoutChanged.emit()
    win._apply_table_settings()
    win._apply_default_sort()
    win._fetch_missing_magnitudes_async()


def _wait_for_visibility_calc(win: MainWindow, timeout_s: float = 24.0) -> None:
    deadline = time.time() + max(3.0, float(timeout_s))
    while time.time() < deadline:
        QApplication.processEvents()
        worker = getattr(win, "worker", None)
        if worker is None or not worker.isRunning():
            _wait(500)
            return
        _wait(120)
    raise TimeoutError("Visibility calculation did not finish before timeout.")


def _demo_suggestions() -> list[dict[str, object]]:
    base_time = datetime(2026, 4, 8, 20, 0, 0)
    sample_rows = [
        DemoSuggestion("SN 2026fvx", 183.74, 63.47, 19.8, "SN", 5, 4.5, 89.2, 1.08, 91.0, 3.9, 57.4),
        DemoSuggestion("Mrk 817", 219.09, 58.79, 14.1, "Seyfert", 5, 3.6, 89.1, 1.14, 80.8, 4.6, 51.4),
        DemoSuggestion("Gaia23cgg", 253.19, 51.41, 15.6, "Transient", 4, 4.1, 83.0, 1.18, 73.0, 6.7, 46.0),
        DemoSuggestion("SDSS J101353", 153.47, 49.46, 13.3, "QSO", 3, 3.9, 79.9, 1.19, 94.1, 3.2, 87.6),
        DemoSuggestion("8C0716_714", 109.06, 71.34, 13.4, "QSO", 3, 4.2, 78.2, 1.21, 117.9, 2.5, 62.8),
        DemoSuggestion("ATO J223.4251+52.7158", 223.43, 52.72, 12.2, "AGN", 3, 3.2, 77.8, 1.25, 74.7, 6.0, 46.8),
        DemoSuggestion("NGC5683-Seyfert", 223.73, 48.65, 15.7, "Seyfert", 3, 2.9, 77.5, 1.31, 70.7, 5.3, 47.4),
        DemoSuggestion("RZ_LMi", 148.22, 34.12, 15.6, "DN", 3, 2.2, 76.6, 1.33, 90.6, 8.8, 73.2),
        DemoSuggestion("AM_Her", 274.05, 49.87, 13.0, "CV", 5, 2.8, 71.2, 1.42, 84.9, 7.4, 21.1),
        DemoSuggestion("1H1936+541", 293.22, 53.97, 11.0, "BLLac", 3, 2.4, 71.2, 1.39, 96.8, 8.1, 19.0),
    ]
    rows: list[dict[str, object]] = []
    for idx, row in enumerate(sample_rows):
        target = Target(
            name=row.name,
            ra=row.ra,
            dec=row.dec,
            source_catalog="bhtom",
            source_object_id=row.name,
            object_type=row.obj_type,
            magnitude=row.magnitude,
            priority=row.priority,
            observed=False,
        )
        start = base_time + timedelta(minutes=35 * idx)
        end = start + timedelta(hours=2, minutes=30)
        rows.append(
            {
                "target": target,
                "metrics": TargetNightMetrics(
                    hours_above_limit=row.hours_above_limit,
                    max_altitude_deg=row.max_altitude,
                    peak_moon_sep_deg=row.min_moon_sep,
                    score=row.score,
                ),
                "window_start": start,
                "window_end": end,
                "best_airmass": row.min_airmass,
                "min_window_moon_sep": row.min_moon_sep,
                "moon_sep_warning": bool(row.min_moon_sep < 30.0),
                "added_to_plan": False,
                "importance": row.importance,
            }
        )
    return rows


def _capture_dashboard(win: MainWindow, output_path: Path) -> None:
    _save_widget_grab(win, output_path)


def _capture_observatory_manager(win: MainWindow, output_path: Path) -> None:
    dialog = ObservatoryManagerDialog(
        win.observatories,
        win,
        preset_keys=getattr(win, "_observatory_preset_keys", {}),
    )
    dialog.resize(1280, 860)
    dialog.show()
    dialog.raise_()
    dialog.activateWindow()
    _wait(1100)
    _save_widget_grab(dialog, output_path)
    dialog.close()
    _wait(120)


def _capture_suggest_dialog(win: MainWindow, output_path: Path) -> None:
    suggestions = _demo_suggestions()
    dialog = SuggestedTargetsDialog(
        suggestions=suggestions,
        notes=["Synthetic screenshot data for README visuals."],
        moon_sep_threshold=20.0,
        mag_warning_threshold=19.0,
        initial_score_filter=0.0,
        bhtom_base_url=BHTOM_API_BASE_URL,
        add_callback=lambda _target: False,
        reload_callback=None,
        parent=win,
    )
    dialog.resize(1470, 860)
    dialog.show()
    dialog.raise_()
    dialog.activateWindow()
    _wait(700)
    _save_widget_grab(dialog, output_path)
    dialog.close()
    _wait(120)


def _capture_ai_window(win: MainWindow, output_path: Path) -> None:
    ai_window = win.ai_window
    ai_window.resize(1480, 860)
    ai_window.show()
    ai_window.raise_()
    ai_window.activateWindow()
    win._clear_ai_messages()
    win._append_ai_message(
        "Which object is best for the next 2 hours at this observatory?",
        is_user=True,
    )
    win._append_ai_message(
        "Top candidate: 8C0716_714. It has score 78.2, over-limit window 20:38-04:52 and moon separation 117.9°.",
        is_ai=True,
    )
    win._append_ai_message(
        "Second option: SDSS J101353.45+492758.1 with score 79.9 and strong altitude margin.",
        is_ai=True,
    )
    win.ai_input.setText("Compare expected cloud risk and suggest exposure strategy.")
    win._set_ai_status("Ready", tone="success")
    _wait(600)
    _save_widget_grab(ai_window, output_path)
    ai_window.hide()
    _wait(120)


def _normalize_requested_views(values: Sequence[str]) -> list[str]:
    requested = [str(value).strip().lower() for value in values]
    if not requested or "all" in requested:
        return list(DEFAULT_VIEWS)
    unknown = [item for item in requested if item not in VIEW_TO_FILE]
    if unknown:
        raise ValueError(f"Unsupported views: {', '.join(sorted(unknown))}")
    seen: set[str] = set()
    ordered: list[str] = []
    for item in requested:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture fresh README screenshots.")
    parser.add_argument(
        "--views",
        nargs="+",
        default=["all"],
        help="Views to capture: dashboard observatory suggest ai (or all).",
    )
    parser.add_argument(
        "--output-dir",
        default="docs/screenshots",
        help="Directory for screenshot PNG files.",
    )
    parser.add_argument(
        "--plan",
        default="examples/plan_targets.json",
        help="JSON plan used to seed dashboard state.",
    )
    parser.add_argument(
        "--date",
        default="2026-04-07",
        help="Session date used for deterministic screenshots (YYYY-MM-DD).",
    )
    args = parser.parse_args()

    views = _normalize_requested_views(args.views)
    output_dir = Path(args.output_dir).resolve()
    plan_path = Path(args.plan).resolve()
    targets = _load_plan_targets(plan_path)

    date_parts = str(args.date).split("-")
    if len(date_parts) != 3:
        raise ValueError("Date must be in YYYY-MM-DD format.")
    year, month, day = (int(part) for part in date_parts)

    app = QApplication([])
    win = MainWindow()
    win.resize(1470, 860)
    win.show()
    win.date_edit.setDate(QDate(year, month, day))
    _apply_targets(win, targets)
    win._run_plan()
    _wait_for_visibility_calc(win)
    _wait(1200)

    if "dashboard" in views:
        _capture_dashboard(win, output_dir / VIEW_TO_FILE["dashboard"])
    if "suggest" in views:
        _capture_suggest_dialog(win, output_dir / VIEW_TO_FILE["suggest"])
    if "ai" in views:
        _capture_ai_window(win, output_dir / VIEW_TO_FILE["ai"])
    if "observatory" in views:
        _capture_observatory_manager(win, output_dir / VIEW_TO_FILE["observatory"])

    win.close()
    _wait(120)
    app.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
