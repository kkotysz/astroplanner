from __future__ import annotations

import logging
import math
import re
import warnings
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

import matplotlib.dates as mdates
import numpy as np
import pytz
from astropy import units as u
from astropy.coordinates import Angle
from astroquery.simbad import Simbad
from PySide6.QtWidgets import QMessageBox

try:
    from astroquery.exceptions import NoResultsWarning
except Exception:  # pragma: no cover - fallback only for older astroquery variants
    class NoResultsWarning(Warning):
        pass

from astroplanner.ai import (
    AIIntent,
    ClassQuerySpec,
    CompareQuerySpec,
    KNOWLEDGE_DIR,
    KnowledgeNote,
    ObjectQuerySpec,
    _format_knowledge_note_snippet,
    _knowledge_note_family,
    _load_knowledge_note,
    _looks_like_object_class_query,
    _looks_like_object_scoped_query,
    _looks_like_observing_guidance_query,
    _normalize_knowledge_tag,
    _question_action_flags,
    _question_bhtom_type_markers,
    _requested_marker_family,
    _requested_marker_label,
    _requested_object_class_marker,
    _truncate_ai_memory_text,
    _type_label_class_family,
    _type_matches_requested_class,
)
from astroplanner.models import Target, targets_match as _targets_match
from astroplanner.resolvers import (
    SIMBAD_COMPACT_CACHE_TTL_S,
    SIMBAD_COMPACT_NEGATIVE_CACHE_TTL_S,
    _extract_simbad_compact_measurements,
    _extract_simbad_name,
    _normalize_catalog_display_name,
    _normalize_catalog_token,
    _object_type_is_unknown,
    _safe_float,
    _safe_int,
    _simbad_best_row_index,
    _simbad_has_row,
    _simbad_row_coord,
    _target_magnitude_label,
    _target_source_label,
)
from astroplanner.scoring import TargetNightMetrics

if TYPE_CHECKING:
    from astro_planner import MainWindow


logger = logging.getLogger(__name__)


class AIContextCoordinator:
    """Own deterministic AI context, local answers, and prompt construction."""

    def __init__(self, planner: "MainWindow") -> None:
        object.__setattr__(self, "_planner", planner)

    def __getattr__(self, name: str):
        return getattr(self._planner, name)

    def __setattr__(self, name: str, value: object) -> None:
        if name == "_planner":
            object.__setattr__(self, name, value)
            return
        setattr(self._planner, name, value)

    def _selected_target_row_index(self) -> Optional[int]:
        rows = self._selected_rows() if hasattr(self, "table_view") else []
        if rows and 0 <= rows[0] < len(self.targets):
            return int(rows[0])
        return None

    def _build_llm_target_summary_line(
        self,
        row_index: int,
        target: Target,
        *,
        include_current_snapshot: bool = True,
    ) -> str:
        self._ensure_known_target_type(target)
        details: list[str] = []
        order_values = getattr(self.table_model, "order_values", [])
        if row_index < len(order_values):
            order_value = _safe_int(order_values[row_index])
            if isinstance(order_value, int) and order_value > 0:
                details.append(f"order {order_value}")
        details.append(f"pri {target.priority}")
        if target.observed:
            details.append("observed")
        if target.object_type and not _object_type_is_unknown(target.object_type):
            details.append(f"type {target.object_type}")
        class_family = self._target_class_family(target)
        if class_family:
            details.append(f"family {class_family}")

        metrics = self.target_metrics.get(target.name)
        if metrics is not None:
            details.extend(
                [
                    f"score {metrics.score:.1f}",
                    f"best {self._format_target_best_window_compact(target) or 'none'}",
                    f"over {metrics.hours_above_limit:.1f} h",
                    f"max alt {metrics.max_altitude_deg:.0f} deg",
                ]
            )
        else:
            details.append("visibility not calculated")

        if include_current_snapshot:
            if row_index < len(self.table_model.current_alts):
                alt_now = self.table_model.current_alts[row_index]
                if math.isfinite(alt_now):
                    details.append(f"now alt {alt_now:.1f} deg")
            if row_index < len(self.table_model.current_seps):
                moon_sep_now = self.table_model.current_seps[row_index]
                if math.isfinite(moon_sep_now):
                    details.append(f"now moon sep {moon_sep_now:.1f} deg")

        return f"  - {target.name}: " + ", ".join(details)

    def _session_context_target_indices(self, *, max_items: int = 8) -> tuple[list[int], int]:
        if not self.targets:
            return [], 0

        row_enabled = list(getattr(self.table_model, "row_enabled", []))
        visible_indices = [idx for idx, enabled in enumerate(row_enabled) if enabled] if row_enabled else []
        candidate_indices = visible_indices or list(range(len(self.targets)))
        selected_row = self._selected_target_row_index()

        def _sort_key(idx: int) -> tuple[object, ...]:
            order_values = getattr(self.table_model, "order_values", [])
            raw_order = order_values[idx] if idx < len(order_values) else 0
            order_value = _safe_int(raw_order) or 0
            metrics = self.target_metrics.get(self.targets[idx].name)
            score = float(metrics.score) if metrics is not None else -1.0
            hours_above = float(metrics.hours_above_limit) if metrics is not None else -1.0
            return (
                0 if order_value > 0 else 1,
                order_value if order_value > 0 else 10**9,
                -score,
                -hours_above,
                _normalize_catalog_display_name(self.targets[idx].name).lower(),
            )

        ranked = sorted(candidate_indices, key=_sort_key)
        summary_indices: list[int] = []
        for idx in ranked:
            if idx == selected_row:
                continue
            summary_indices.append(idx)
            if len(summary_indices) >= max_items:
                break
        omitted_count = max(0, len(candidate_indices) - len(summary_indices))
        return summary_indices, omitted_count

    def _build_session_context(
        self,
        *,
        include_current_snapshot: bool = True,
        user_question: str = "",
    ) -> str:
        parts: list[str] = []
        date_str = self.date_edit.date().toString("yyyy-MM-dd")
        parts.append(f"Observation date: {date_str}")
        site = self.table_model.site
        if site:
            parts.append(
                f"Site: {site.name} (lat {site.latitude:.3f}, lon {site.longitude:.3f}, "
                f"elev {site.elevation:.0f} m, timezone {site.timezone_name})"
            )
        parts.append(f"Altitude limit: {self.limit_spin.value()} deg")
        parts.append(f"Sun altitude threshold: {self._sun_alt_limit():.0f} deg")
        if hasattr(self, "min_moon_sep_spin"):
            parts.append(f"Moon separation threshold: {float(self.min_moon_sep_spin.value()):.0f} deg")
        if hasattr(self, "min_score_spin"):
            parts.append(f"Score threshold: {float(self.min_score_spin.value()):.1f}")

        payload = self.last_payload if isinstance(self.last_payload, dict) else None
        tz_name = site.timezone_name if site else "UTC"
        try:
            tz = pytz.timezone(tz_name)
        except Exception:
            tz = pytz.UTC
        now_local = None
        if payload:
            tz_name = str(payload.get("tz", tz_name or "UTC"))
            try:
                tz = pytz.timezone(tz_name)
            except Exception:
                tz = pytz.UTC
            now_local = payload.get("now_local")

            if isinstance(now_local, datetime):
                try:
                    now_local = now_local.astimezone(tz)
                except Exception:
                    pass

            def _fmt_event(key: str) -> str:
                raw = payload.get(key)
                if raw is None:
                    return "N/A"
                try:
                    dt = mdates.num2date(float(raw)).astimezone(tz)
                    return dt.strftime("%Y-%m-%d %H:%M")
                except Exception:
                    return "N/A"

            if isinstance(now_local, datetime):
                parts.append(f"Current local time: {now_local.strftime('%Y-%m-%d %H:%M:%S')}")
            parts.append(
                "Night events: "
                f"sunset {_fmt_event('sunset')}, sunrise {_fmt_event('sunrise')}, "
                f"astronomical night starts {_fmt_event('dusk')}, "
                f"astronomical night ends {_fmt_event('dawn')}"
            )
            moon_phase = payload.get("moon_phase")
            if moon_phase is not None:
                try:
                    parts.append(f"Moon phase: {float(moon_phase):.1f}%")
                except Exception:
                    pass

        if not self.targets:
            parts.append("Targets in plan: none")
            suggestion_context = self._build_bhtom_suggestion_shortlist_context(
                user_question=user_question,
                max_items=5,
            )
            if suggestion_context:
                parts.append(suggestion_context)
            return "\n".join(parts)

        visible_count = sum(bool(enabled) for enabled in getattr(self.table_model, "row_enabled", []))
        if visible_count > 0:
            parts.append(f"Targets in plan: {len(self.targets)} total, {visible_count} visible under current filters")
        else:
            parts.append(f"Targets in plan: {len(self.targets)}")

        class_query = self._parse_class_query_spec(user_question)
        requested_marker = class_query.requested_class if class_query is not None else ""
        selected_row = self._selected_target_row_index()
        if selected_row is not None and 0 <= selected_row < len(self.targets):
            selected_target = self.targets[selected_row]
            self._ensure_known_target_type(selected_target)
            if not requested_marker or _type_matches_requested_class(selected_target.object_type, requested_marker):
                parts.append(
                    "Selected target:\n"
                    + self._build_llm_target_summary_line(
                        selected_row,
                        selected_target,
                        include_current_snapshot=include_current_snapshot,
                    )
                )

        summary_indices, omitted_count = self._session_context_target_indices(
            max_items=8,
        )
        if requested_marker:
            filtered_indices: list[int] = []
            for idx in summary_indices:
                self._ensure_known_target_type(self.targets[idx])
                if _type_matches_requested_class(self.targets[idx].object_type, requested_marker):
                    filtered_indices.append(idx)
            omitted_count += max(0, len(summary_indices) - len(filtered_indices))
            summary_indices = filtered_indices
        if summary_indices:
            rows = [
                self._build_llm_target_summary_line(
                    idx,
                    self.targets[idx],
                    include_current_snapshot=include_current_snapshot,
                )
                for idx in summary_indices
            ]
            parts.append("Target shortlist:\n" + "\n".join(rows))
            if omitted_count > 0:
                parts.append(f"Additional visible targets omitted from prompt: {omitted_count}")

        suggestion_context = self._build_bhtom_suggestion_shortlist_context(
            user_question=user_question,
            max_items=5,
        )
        if suggestion_context:
            parts.append(suggestion_context)
        return "\n".join(parts)

    def _build_bhtom_suggestion_shortlist_context(
        self,
        *,
        user_question: str = "",
        max_items: int = 5,
    ) -> str:
        suggestions = list(getattr(self, "_bhtom_ranked_suggestions_cache", []) or [])
        if not suggestions:
            return ""

        type_markers = list(_question_bhtom_type_markers(user_question))
        class_query = self._parse_class_query_spec(user_question)
        if class_query is not None:
            marker = _normalize_knowledge_tag(class_query.requested_class)
            if marker and marker not in type_markers:
                type_markers.append(marker)
        shortlist = suggestions
        shortlist_label = "Cached BHTOM suggestion shortlist (not yet in plan)"
        max_rows = max_items

        if type_markers:
            filtered: list[dict[str, object]] = []
            for item in suggestions:
                target = item.get("target")
                if not isinstance(target, Target):
                    continue
                haystack = " ".join(
                    [
                        _normalize_catalog_display_name(target.name).lower(),
                        _normalize_catalog_display_name(target.source_object_id).lower(),
                        _normalize_catalog_display_name(target.object_type).lower(),
                    ]
                )
                if any(marker in haystack for marker in type_markers):
                    filtered.append(item)
            if filtered:
                shortlist = filtered
                max_rows = max(max_items, 10)
                shortlist_label = (
                    "Cached BHTOM suggestion shortlist matching this question "
                    "(not yet in plan)"
                )

        rows: list[str] = []
        for item in shortlist:
            target = item.get("target")
            metrics = item.get("metrics")
            window_start = item.get("window_start")
            window_end = item.get("window_end")
            if not isinstance(target, Target) or not isinstance(metrics, TargetNightMetrics):
                continue
            if not isinstance(window_start, datetime) or not isinstance(window_end, datetime):
                continue

            details: list[str] = []
            if target.object_type and not _object_type_is_unknown(target.object_type):
                details.append(f"type {target.object_type}")
            importance = _safe_float(item.get("importance"))
            if importance is not None and math.isfinite(importance):
                details.append(f"importance {importance:.1f}")
            if target.magnitude is not None and math.isfinite(float(target.magnitude)):
                details.append(f"{_target_magnitude_label(target).lower()} {float(target.magnitude):.2f}")
            details.append(f"score {metrics.score:.1f}")
            details.append(f"best {window_start.strftime('%H:%M')}-{window_end.strftime('%H:%M')}")
            best_airmass = _safe_float(item.get("best_airmass"))
            if best_airmass is not None and math.isfinite(best_airmass):
                details.append(f"min airmass {best_airmass:.2f}")
            rows.append(f"  - {target.name}: " + ", ".join(details))
            if len(rows) >= max_rows:
                break

        if not rows:
            return ""
        omitted = max(0, len(shortlist) - len(rows))
        summary = shortlist_label
        if omitted > 0:
            summary = f"{summary} (+{omitted} more)"
        return summary + ":\n" + "\n".join(rows)

    def _target_class_family(self, target: Target) -> str:
        self._ensure_known_target_type(target)
        return _type_label_class_family(target.object_type)

    def _find_referenced_target_in_question(self, text: str) -> Optional[Target]:
        raw_text = str(text or "").strip()
        if not raw_text:
            return None

        candidates: list[tuple[int, Target]] = [(0, target) for target in self.targets]
        suggestions = list(getattr(self, "_bhtom_ranked_suggestions_cache", []) or [])
        for item in suggestions:
            target = item.get("target")
            if isinstance(target, Target):
                candidates.append((1, target))

        matched: list[tuple[int, int, int, Target]] = []
        seen: set[str] = set()
        for source_rank, target in candidates:
            dedupe_key = _normalize_catalog_token(target.source_object_id or target.name)
            if not dedupe_key or dedupe_key in seen:
                continue
            positions: list[tuple[int, int]] = []
            for candidate_name in {target.name.strip(), str(target.source_object_id or "").strip()}:
                if not candidate_name:
                    continue
                pos = self._find_normalized_text_position(raw_text, candidate_name)
                if pos is None:
                    continue
                match_len = len(re.sub(r"[^a-z0-9]+", "", _normalize_catalog_display_name(candidate_name).lower()))
                positions.append((pos, match_len))
            if not positions:
                continue
            seen.add(dedupe_key)
            best_pos, best_len = min(positions, key=lambda item: (item[0], -item[1]))
            matched.append((best_pos, -best_len, source_rank, target))

        if not matched:
            return None
        matched.sort(key=lambda item: (item[0], item[1], item[2], item[3].name.lower()))
        return matched[0][3]

    def _find_referenced_targets_in_question(self, text: str, *, max_targets: int = 6) -> list[Target]:
        raw_text = str(text or "").strip()
        if not raw_text:
            return []

        candidates: list[tuple[int, Target]] = [(0, target) for target in self.targets]
        suggestions = list(getattr(self, "_bhtom_ranked_suggestions_cache", []) or [])
        for item in suggestions:
            target = item.get("target")
            if isinstance(target, Target):
                candidates.append((1, target))

        matched: list[tuple[int, int, int, str, Target]] = []
        seen: set[str] = set()
        for source_rank, target in candidates:
            dedupe_key = _normalize_catalog_token(target.source_object_id or target.name)
            if not dedupe_key or dedupe_key in seen:
                continue
            positions: list[tuple[int, int]] = []
            for candidate_name in {target.name.strip(), str(target.source_object_id or "").strip()}:
                if not candidate_name:
                    continue
                pos = self._find_normalized_text_position(raw_text, candidate_name)
                if pos is None:
                    continue
                match_len = len(re.sub(r"[^a-z0-9]+", "", _normalize_catalog_display_name(candidate_name).lower()))
                positions.append((pos, match_len))
            if not positions:
                continue
            seen.add(dedupe_key)
            best_pos, best_len = min(positions, key=lambda item: (item[0], -item[1]))
            matched.append((best_pos, -best_len, source_rank, target.name.lower(), target))

        matched.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
        return [item[4] for item in matched[: max(1, int(max_targets))]]

    def _plan_row_index_for_target(self, target: Target) -> Optional[int]:
        for idx, existing in enumerate(self.targets):
            if _targets_match(existing, target):
                return idx
        return None

    def _lookup_target_observing_candidate(self, target: Target) -> Optional[dict[str, object]]:
        row_index = self._plan_row_index_for_target(target)
        if row_index is not None and 0 <= row_index < len(self.targets):
            plan_target = self.targets[row_index]
            self._ensure_known_target_type(plan_target)
            metrics = self.target_metrics.get(plan_target.name)
            if metrics is not None:
                current_alt = None
                if row_index < len(self.table_model.current_alts):
                    alt_now = self.table_model.current_alts[row_index]
                    if math.isfinite(alt_now):
                        current_alt = float(alt_now)
                moon_sep = None
                if row_index < len(self.table_model.current_seps):
                    sep_now = self.table_model.current_seps[row_index]
                    if math.isfinite(sep_now):
                        moon_sep = float(sep_now)
                return {
                    "target": plan_target,
                    "metrics": metrics,
                    "best_window": self._format_target_best_window_compact(plan_target),
                    "current_alt": current_alt,
                    "moon_sep": moon_sep,
                    "source": "plan",
                }

        suggestions = list(getattr(self, "_bhtom_ranked_suggestions_cache", []) or [])
        for item in suggestions:
            suggestion_target = item.get("target")
            metrics = item.get("metrics")
            if not isinstance(suggestion_target, Target) or not isinstance(metrics, TargetNightMetrics):
                continue
            if not _targets_match(suggestion_target, target):
                continue
            self._ensure_known_target_type(suggestion_target)
            best_window = ""
            window_start = item.get("window_start")
            window_end = item.get("window_end")
            if isinstance(window_start, datetime) and isinstance(window_end, datetime):
                best_window = f"{window_start.strftime('%H:%M')}-{window_end.strftime('%H:%M')}"
            return {
                "target": suggestion_target,
                "metrics": metrics,
                "best_window": best_window,
                "current_alt": None,
                "moon_sep": _safe_float(item.get("min_window_moon_sep")),
                "source": "bhtom",
                "best_airmass": _safe_float(item.get("best_airmass")),
                "importance": _safe_float(item.get("importance")),
            }

        self._ensure_known_target_type(target)
        metrics = self.target_metrics.get(target.name)
        if metrics is None:
            return None
        return {
            "target": target,
            "metrics": metrics,
            "best_window": self._format_target_best_window_compact(target),
            "current_alt": None,
            "moon_sep": None,
            "source": _normalize_catalog_token(target.source_catalog) or "target",
        }

    def _load_knowledge_notes(self) -> list[KnowledgeNote]:
        cached = getattr(self, "_knowledge_notes_cache", None)
        if cached is not None:
            return cached

        notes: list[KnowledgeNote] = []
        try:
            paths = sorted(
                path
                for path in KNOWLEDGE_DIR.rglob("*.md")
                if path.is_file()
                and "_templates" not in path.parts
                and path.name != "_index.md"
            )
        except Exception:
            paths = []

        for path in paths:
            note = _load_knowledge_note(path)
            if note is not None:
                notes.append(note)
        self._knowledge_notes_cache = notes
        return notes

    def _select_knowledge_notes(
        self,
        *,
        question: str,
        target: Optional[Target] = None,
        max_notes: int = 3,
        max_chars: int = 1600,
    ) -> list[KnowledgeNote]:
        notes = self._load_knowledge_notes()
        if not notes:
            return []

        request_tags = self._knowledge_request_tags(question, target=target)
        if not request_tags:
            return []

        requested_family = _requested_marker_family(_requested_object_class_marker(question))
        if not requested_family and target is not None:
            requested_family = self._target_class_family(target)

        if target is None and not requested_family:
            global_note_tags = {
                "bhtom",
                "last-mag-vs-mag",
                "simbad",
                "tns",
                "gaia-alerts",
                "best-window",
                "moonlight",
                "choosing-between-similar-targets",
                "small-scope-practicality",
            }
            if not (request_tags & global_note_tags):
                return []

        ranked: list[tuple[int, KnowledgeNote]] = []
        for note in notes:
            note_family = _knowledge_note_family(note)
            if requested_family and "object-classes" in note.path.parts:
                if note_family != requested_family:
                    continue
            score = self._knowledge_note_score(note, request_tags=request_tags, question=question, target=target)
            if score <= 0:
                continue
            ranked.append((score, note))
        if not ranked:
            return []

        ranked.sort(key=lambda item: (-item[0], item[1].path.as_posix()))
        selected: list[KnowledgeNote] = []
        total_chars = 0
        for _, note in ranked[:max_notes]:
            snippet = _format_knowledge_note_snippet(note)
            next_total = total_chars + len(snippet) + (2 if selected else 0)
            if selected and next_total > max_chars:
                break
            selected.append(note)
            total_chars = next_total
        return selected

    def _knowledge_request_tags(self, question: str, target: Optional[Target] = None) -> set[str]:
        tags: set[str] = set()
        for marker in _question_bhtom_type_markers(question):
            normalized = _normalize_knowledge_tag(marker)
            if normalized:
                tags.add(normalized)

        requested_marker = _requested_object_class_marker(question)
        if requested_marker:
            tags.add(_normalize_knowledge_tag(requested_marker))

        normalized_question = _normalize_catalog_display_name(question).lower()
        keyword_map = {
            "moonlight": (
                "moon", "moonlight", "moon sep", "moon separation", "ksiezyc", "księżyc",
                "lun", "bright moon",
            ),
            "bhtom": ("bhtom", "importance", "last mag", "last magnitude"),
            "last-mag-vs-mag": ("last mag", "last magnitude", "catalog mag", "mag vs last mag"),
            "simbad": ("simbad",),
            "tns": ("tns", "transient name server"),
            "gaia-alerts": ("gaia alert", "gaia alerts", "gaia-alerts"),
            "best-window": (
                "best window", "window", "over limit", "score", "order", "kolejn", "airmass",
                "altitude", "wysok", "horizon", "horyzont", "when observe", "kiedy obserw",
            ),
            "practical-observing": ("observe", "obserw", "how should i", "jak powinienem"),
            "choosing-between-similar-targets": (
                "which is best", "which one is best", "ktory najlepiej", "która najlepiej",
                "porown", "porówn", "compare", "comparison", "vs", "better target",
            ),
            "small-scope-practicality": (
                "small scope", "small telescope", "small setup", "seestar",
                "easy target", "practical", "realistically", "ma sens",
            ),
        }
        for tag, markers in keyword_map.items():
            if any(marker in normalized_question for marker in markers):
                tags.add(tag)

        if target is not None:
            self._ensure_known_target_type(target)
            family = self._target_class_family(target)
            if family == "AGN":
                tags.add("agn")
                type_norm = _normalize_catalog_display_name(target.object_type).lower()
                if "qso" in type_norm or "quasar" in type_norm:
                    tags.add("qso")
            elif family == "Supernova":
                tags.update({"supernova", "sn"})
            elif family == "Nova":
                tags.add("nova")
            elif family == "Cataclysmic variable":
                tags.add("cv")
            elif family == "Galaxy":
                tags.add("galaxy")
            elif family == "Star":
                tags.add("star")
                type_norm = _normalize_catalog_display_name(target.object_type).lower()
                if "variable" in type_norm:
                    tags.add("variable-star")
            if "cluster" in _normalize_catalog_display_name(target.object_type).lower():
                tags.add("open-cluster")
            source_tag = _normalize_knowledge_tag(target.source_catalog)
            if source_tag:
                tags.add(source_tag)
            if _normalize_catalog_token(target.source_catalog) == "bhtom":
                tags.add("bhtom")

        return {tag for tag in tags if tag}

    def _knowledge_note_score(
        self,
        note: KnowledgeNote,
        *,
        request_tags: set[str],
        question: str,
        target: Optional[Target],
    ) -> int:
        score = 0
        note_tags = set(note.tags)
        overlap = request_tags & note_tags
        score += len(overlap) * 4

        path_tokens = {
            _normalize_knowledge_tag(note.path.stem),
            _normalize_knowledge_tag(note.path.parent.name),
        }
        score += len(request_tags & path_tokens) * 3

        normalized_question = _normalize_catalog_display_name(question).lower()
        if note.summary and any(tag.replace("-", " ") in normalized_question for tag in note_tags):
            score += 2

        if target is not None:
            if "bhtom" in note_tags and _normalize_catalog_token(target.source_catalog) == "bhtom":
                score += 2
            if "agn" in note_tags and self._target_class_family(target) == "AGN":
                score += 2
            if {"supernova", "sn"} & note_tags and self._target_class_family(target) == "Supernova":
                score += 2

        return score

    def _build_knowledge_context(
        self,
        *,
        question: str,
        target: Optional[Target] = None,
        max_notes: int = 3,
        max_chars: int = 1600,
    ) -> str:
        notes = self._select_knowledge_notes(
            question=question,
            target=target,
            max_notes=max_notes,
            max_chars=max_chars,
        )
        if not notes:
            return ""
        snippets = [_format_knowledge_note_snippet(note) for note in notes]
        return "Local knowledge notes:\n" + "\n".join(snippets)

    def _build_local_object_fact_answer(self, question: str, *, target: Optional[Target] = None) -> Optional[str]:
        target = target or self._resolve_object_query_target(question, selected_target=self._selected_target_or_none())
        if target is None:
            return None

        type_label = self._ensure_known_target_type(target).strip()
        family = self._target_class_family(target)
        if not type_label and not family:
            return None

        normalized = _normalize_catalog_display_name(question).lower()
        requested_marker = _requested_object_class_marker(question)
        class_label_map = {
            "agn": "AGN",
            "qso": "QSO",
            "seyfert": "Seyfert",
            "blazar": "blazar",
            "supernova": "supernova",
            "nova": "nova",
            "xrb": "X-ray binary",
            "cv": "cataclysmic variable",
            "galaxy": "galaxy",
            "star": "star",
        }
        if requested_marker and _looks_like_object_class_query(normalized):
            matches = _type_matches_requested_class(type_label, requested_marker)
            class_label = class_label_map.get(requested_marker, requested_marker)
            if matches:
                detail = type_label or family or class_label
                answer = f"Yes. {target.name} is classified as {detail}."
                if family and family != detail:
                    answer += f" Broad class: {family}."
                return answer
            detail = type_label or family or "unknown"
            answer = f"No. Current metadata classifies {target.name} as {detail}."
            if family and family != detail:
                answer += f" Broad class: {family}."
            return answer

        type_markers = (
            "what type",
            "type of object",
            "what is",
            "what kind",
            "jaki typ",
            "jaki to typ",
            "jaki to obiekt",
            "co to za obiekt",
            "czym jest",
            "co to jest",
        )
        if any(marker in normalized for marker in type_markers):
            detail = type_label or family
            if not detail:
                return None
            answer = f"{target.name} is classified as {detail}."
            if family and family != detail:
                answer += f" Broad class: {family}."
            return answer

        return None

    def _parse_class_query_spec(self, question: str) -> Optional[ClassQuerySpec]:
        normalized = _normalize_catalog_display_name(question).lower()
        if not normalized:
            return None

        explicit_requested_class = _requested_object_class_marker(normalized)
        filter_flags = _question_action_flags(
            normalized,
            {
                "bhtom_only": (
                    "bhtom only",
                    "only bhtom",
                    "only from bhtom",
                    "from bhtom only",
                    "tylko z bhtom",
                    "tylko bhtom",
                ),
                "exclude_observed": (
                    "exclude observed",
                    "without observed",
                    "hide observed",
                    "not observed",
                    "unobserved",
                    "nieobserw",
                    "nie obserw",
                    "bez obserw",
                ),
                "prefer_brighter": (
                    "brighter",
                    "brightest",
                    "jaśniejs",
                    "jasniejs",
                    "jaśniejsze",
                    "jasniejsze",
                    "najjaś",
                    "najas",
                ),
            },
        )
        flags = _question_action_flags(
            normalized,
            {
                "observe": (
                    "which",
                    "best",
                    "recommend",
                    "observe",
                    "obserw",
                    "moge",
                    "mogę",
                    "today",
                    "tonight",
                    "dzis",
                    "dziś",
                    "któr",
                    "jaki",
                ),
                "list": (
                    "other",
                    "others",
                    "more",
                    "list",
                    "show",
                    "give me",
                    "top ",
                    "jakie",
                    "jakie sa",
                    "jakie są",
                    "inne",
                    "wymien",
                    "wymień",
                    "pokaz",
                    "pokaż",
                    "lista",
                    "podaj",
                ),
                "more": (
                    "other",
                    "others",
                    "more",
                    "another",
                    "inne",
                    "kolejne",
                    "więcej",
                    "wiecej",
                    "next",
                ),
                "choice": (
                    "which one",
                    "which is best",
                    "which is better",
                    "ktory",
                    "która",
                    "ktora",
                    "który",
                ),
            },
        )
        has_action_marker = bool(flags["observe"])
        wants_list = bool(flags["list"]) or bool(re.search(r"\b([2-9]|10)\b", normalized))
        wants_more = bool(flags["more"])
        choice_followup = bool(flags["choice"])
        prefer_bhtom_only = bool(filter_flags["bhtom_only"])
        exclude_observed = bool(filter_flags["exclude_observed"])
        prefer_brighter = bool(filter_flags["prefer_brighter"])
        semantic_followup = prefer_bhtom_only or exclude_observed or prefer_brighter
        requested_class = explicit_requested_class
        if not requested_class and (wants_list or wants_more or choice_followup or semantic_followup):
            requested_class = str(self._recent_ai_conversation_state().get("requested_class", "") or "").strip()

        if not requested_class:
            return None
        if not explicit_requested_class and not (wants_list or wants_more or choice_followup or semantic_followup):
            return None
        if explicit_requested_class and not (
            has_action_marker or wants_list or wants_more or choice_followup or semantic_followup
        ):
            return None

        if semantic_followup and not wants_list and not choice_followup:
            wants_list = True
        count = self._class_query_requested_count(question, default_count=5 if wants_list else 3)
        return ClassQuerySpec(
            requested_class=requested_class,
            count=count,
            wants_list=wants_list,
            wants_more=wants_more,
            choice_followup=choice_followup,
            prefer_bhtom_only=prefer_bhtom_only,
            exclude_observed=exclude_observed,
            exclude_previous_results=wants_more,
            prefer_brighter=prefer_brighter,
        )

    def _looks_like_class_observing_query(self, text: str) -> bool:
        return self._parse_class_query_spec(text) is not None

    def _collect_class_observing_candidates(self, class_query: ClassQuerySpec) -> list[dict[str, object]]:
        marker = _normalize_knowledge_tag(class_query.requested_class)
        if not marker:
            return []

        candidates: list[dict[str, object]] = []
        seen: set[str] = set()
        for row_index, target in enumerate(self.targets):
            self._ensure_known_target_type(target)
            if not _type_matches_requested_class(target.object_type, marker):
                continue
            if class_query.prefer_bhtom_only and _normalize_catalog_token(target.source_catalog) != "bhtom":
                continue
            if class_query.exclude_observed and bool(target.observed):
                continue
            metrics = self.target_metrics.get(target.name)
            if metrics is None:
                continue
            current_alt = None
            if row_index < len(self.table_model.current_alts):
                alt_now = self.table_model.current_alts[row_index]
                if math.isfinite(alt_now):
                    current_alt = float(alt_now)
            moon_sep = None
            if row_index < len(self.table_model.current_seps):
                sep_now = self.table_model.current_seps[row_index]
                if math.isfinite(sep_now):
                    moon_sep = float(sep_now)
            dedupe_key = _normalize_catalog_token(target.source_object_id or target.name)
            if not dedupe_key or dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            candidates.append(
                {
                    "target": target,
                    "metrics": metrics,
                    "best_window": self._format_target_best_window_compact(target),
                    "current_alt": current_alt,
                    "moon_sep": moon_sep,
                    "source": "plan",
                }
            )

        suggestions = list(getattr(self, "_bhtom_ranked_suggestions_cache", []) or [])
        for item in suggestions:
            target = item.get("target")
            metrics = item.get("metrics")
            if not isinstance(target, Target) or not isinstance(metrics, TargetNightMetrics):
                continue
            self._ensure_known_target_type(target)
            if not _type_matches_requested_class(target.object_type, marker):
                continue
            if class_query.exclude_observed and bool(target.observed):
                continue
            dedupe_key = _normalize_catalog_token(target.source_object_id or target.name)
            if not dedupe_key or dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            best_window = ""
            window_start = item.get("window_start")
            window_end = item.get("window_end")
            if isinstance(window_start, datetime) and isinstance(window_end, datetime):
                best_window = f"{window_start.strftime('%H:%M')}-{window_end.strftime('%H:%M')}"
            candidates.append(
                {
                    "target": target,
                    "metrics": metrics,
                    "best_window": best_window,
                    "current_alt": None,
                    "moon_sep": _safe_float(item.get("min_window_moon_sep")),
                    "source": "bhtom",
                }
            )
        return candidates

    def _recent_ai_conversation_state(self, *, max_messages: int = 8) -> dict[str, Any]:
        requested_class = ""
        primary_target: Optional[Target] = None
        suggested_targets: list[Target] = []

        considered = 0
        for message in reversed(self._ai_messages):
            kind = str(message.get("kind", "") or "").strip().lower()
            if kind not in {"user", "ai"}:
                continue
            considered += 1
            if not requested_class:
                requested_class = str(message.get("requested_class", "") or "").strip()
            if primary_target is None:
                candidate_target = message.get("primary_target")
                if isinstance(candidate_target, Target):
                    primary_target = candidate_target
            if not suggested_targets:
                candidate_targets = message.get("suggested_targets")
                if isinstance(candidate_targets, list):
                    suggested_targets = [target for target in candidate_targets if isinstance(target, Target)]
            if requested_class and primary_target is not None and suggested_targets:
                break
            if considered >= max_messages:
                break

        if primary_target is None and suggested_targets:
            primary_target = suggested_targets[0]

        return {
            "requested_class": requested_class,
            "primary_target": primary_target,
            "suggested_targets": suggested_targets,
        }

    def _resolve_recent_class_marker(self, *, max_messages: int = 6) -> str:
        recent_state = self._recent_ai_conversation_state(max_messages=max_messages)
        requested_class = str(recent_state.get("requested_class", "") or "").strip()
        if requested_class:
            return requested_class

        recent_user_texts: list[str] = []
        for message in reversed(self._ai_messages):
            kind = str(message.get("kind", "") or "").strip().lower()
            if kind != "user":
                continue
            text = str(message.get("text", "") or "").strip()
            if not text:
                continue
            recent_user_texts.append(text)
            if len(recent_user_texts) >= max_messages:
                break

        for text in recent_user_texts:
            marker = _requested_object_class_marker(text)
            if marker:
                return marker
        return ""

    @staticmethod
    def _class_query_requested_count(question: str, *, default_count: int = 3, max_count: int = 10) -> int:
        normalized = _normalize_catalog_display_name(question).lower()
        if not normalized:
            return default_count

        digit_match = re.search(r"\b([1-9]|10)\b", normalized)
        if digit_match:
            return max(1, min(max_count, int(digit_match.group(1))))

        word_map = {
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10,
            "jeden": 1,
            "dwa": 2,
            "trzy": 3,
            "cztery": 4,
            "piec": 5,
            "pięć": 5,
            "szesc": 6,
            "sześć": 6,
            "siedem": 7,
            "osiem": 8,
            "dziewiec": 9,
            "dziewięć": 9,
            "dziesiec": 10,
            "dziesięć": 10,
        }
        for token, value in word_map.items():
            if re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", normalized):
                return max(1, min(max_count, value))
        return default_count

    def _build_local_class_observing_response(self, question: str) -> Optional[dict[str, Any]]:
        class_query = self._parse_class_query_spec(question)
        if class_query is None:
            return None

        recent_state = self._recent_ai_conversation_state()
        requested_marker = class_query.requested_class
        if not requested_marker:
            return None

        candidates = self._collect_class_observing_candidates(class_query)
        label = _requested_marker_label(requested_marker)
        is_polish = any(
            token in _normalize_catalog_display_name(question).lower()
            for token in ("jak ", "obserw", "dzis", "dziś", "któr", "jaki", "mogę", "moge")
        )
        if not candidates:
            if is_polish:
                return {
                    "text": f"Brak pasujących celów klasy {label} w planie ani w cache BHTOM.",
                    "requested_class": requested_marker,
                    "primary_target": None,
                    "suggested_targets": [],
                }
            return {
                "text": f"No matching {label} targets are available in the plan or cached BHTOM suggestions.",
                "requested_class": requested_marker,
                "primary_target": None,
                "suggested_targets": [],
            }

        wants_list = bool(class_query.wants_list)
        wants_more = bool(class_query.wants_more)
        choice_followup = bool(class_query.choice_followup)
        requested_count = int(class_query.count)
        excluded_targets: list[Target] = []
        if class_query.exclude_previous_results:
            recent_suggested = list(recent_state.get("suggested_targets", []) or [])
            for recent_target in recent_suggested:
                if not isinstance(recent_target, Target):
                    continue
                self._ensure_known_target_type(recent_target)
                if _type_matches_requested_class(recent_target.object_type, requested_marker):
                    excluded_targets.append(recent_target)
            if not excluded_targets:
                recent_target = recent_state.get("primary_target")
                if isinstance(recent_target, Target):
                    self._ensure_known_target_type(recent_target)
                    if _type_matches_requested_class(recent_target.object_type, requested_marker):
                        excluded_targets.append(recent_target)

        def _sort_key(item: dict[str, object]) -> tuple[object, ...]:
            metrics = item.get("metrics")
            score = float(metrics.score) if isinstance(metrics, TargetNightMetrics) else -1.0
            hours_above = float(metrics.hours_above_limit) if isinstance(metrics, TargetNightMetrics) else -1.0
            current_alt = _safe_float(item.get("current_alt"))
            current_alt_sort = float(current_alt) if current_alt is not None else -1.0
            target = item.get("target")
            magnitude = float(target.magnitude) if isinstance(target, Target) and target.magnitude is not None and math.isfinite(float(target.magnitude)) else float("inf")
            source_rank = 0 if str(item.get("source") or "") == "plan" else 1
            name = target.name if isinstance(target, Target) else ""
            if class_query.prefer_brighter:
                return (magnitude, -score, -hours_above, -current_alt_sort, source_rank, name.lower())
            return (-score, -hours_above, -current_alt_sort, magnitude, source_rank, name.lower())

        candidates.sort(key=_sort_key)
        if choice_followup:
            recent_keys = {
                _normalize_catalog_token(target.source_object_id or target.name)
                for target in list(recent_state.get("suggested_targets", []) or [])
                if isinstance(target, Target)
            }
            if recent_keys:
                filtered_candidates = []
                for item in candidates:
                    item_target = item.get("target")
                    if not isinstance(item_target, Target):
                        continue
                    item_key = _normalize_catalog_token(item_target.source_object_id or item_target.name)
                    if item_key and item_key in recent_keys:
                        filtered_candidates.append(item)
                if filtered_candidates:
                    candidates = filtered_candidates
        if excluded_targets:
            excluded_keys = {
                _normalize_catalog_token(target.source_object_id or target.name)
                for target in excluded_targets
                if isinstance(target, Target)
            }
            filtered_candidates = []
            for item in candidates:
                item_target = item.get("target")
                if not isinstance(item_target, Target):
                    continue
                item_key = _normalize_catalog_token(item_target.source_object_id or item_target.name)
                if item_key and item_key in excluded_keys:
                    continue
                filtered_candidates.append(item)
            if filtered_candidates:
                candidates = filtered_candidates
            elif wants_more:
                if is_polish:
                    return {
                        "text": f"Nie mam już więcej celów klasy {label} poza tymi, które już pokazałem.",
                        "requested_class": requested_marker,
                        "primary_target": excluded_targets[0] if excluded_targets else None,
                        "suggested_targets": list(excluded_targets),
                    }
                return {
                    "text": f"No more {label} targets are available beyond the ones already shown.",
                    "requested_class": requested_marker,
                    "primary_target": excluded_targets[0] if excluded_targets else None,
                    "suggested_targets": list(excluded_targets),
                }

        top = candidates[0]
        target = top.get("target")
        metrics = top.get("metrics")
        if not isinstance(target, Target) or not isinstance(metrics, TargetNightMetrics):
            return None

        displayed_targets: list[Target] = []
        if wants_list:
            rows: list[str] = []
            for item in candidates[: max(1, requested_count)]:
                item_target = item.get("target")
                item_metrics = item.get("metrics")
                if not isinstance(item_target, Target) or not isinstance(item_metrics, TargetNightMetrics):
                    continue
                displayed_targets.append(item_target)
                item_window = str(item.get("best_window") or "").strip()
                item_alt = _safe_float(item.get("current_alt"))
                item_source = str(item.get("source") or "plan")
                parts = [f"score {item_metrics.score:.1f}"]
                if item_window:
                    parts.append(f"best {item_window}")
                if item_alt is not None and math.isfinite(item_alt):
                    parts.append(f"now alt {item_alt:.1f}°")
                if item_source == "bhtom":
                    parts.append("BHTOM")
                rows.append(f"- {item_target.name}: " + ", ".join(parts))
            if is_polish:
                return {
                    "text": f"{label} dostępne teraz:\n" + "\n".join(rows),
                    "requested_class": requested_marker,
                    "primary_target": displayed_targets[0] if displayed_targets else target,
                    "suggested_targets": displayed_targets,
                }
            return {
                "text": f"{label} options now:\n" + "\n".join(rows),
                "requested_class": requested_marker,
                "primary_target": displayed_targets[0] if displayed_targets else target,
                "suggested_targets": displayed_targets,
            }

        best_window = str(top.get("best_window") or "").strip()
        current_alt = _safe_float(top.get("current_alt"))
        source = str(top.get("source") or "plan")
        details = [f"score {metrics.score:.1f}"]
        if best_window:
            details.append(f"best {best_window}")
        details.append(f"max alt {metrics.max_altitude_deg:.0f}°")
        if current_alt is not None and math.isfinite(current_alt):
            details.append(f"now alt {current_alt:.1f}°")
        if source == "bhtom":
            details.append("from BHTOM cache")

        backups: list[str] = []
        displayed_targets.append(target)
        for item in candidates[1:3]:
            backup_target = item.get("target")
            backup_metrics = item.get("metrics")
            if not isinstance(backup_target, Target) or not isinstance(backup_metrics, TargetNightMetrics):
                continue
            backups.append(f"{backup_target.name} ({backup_metrics.score:.1f})")
            displayed_targets.append(backup_target)

        if is_polish:
            answer = f"Najlepszy {label} teraz: {target.name} — {', '.join(details)}."
            if backups:
                answer += " Rezerwa: " + ", ".join(backups) + "."
            return {
                "text": answer,
                "requested_class": requested_marker,
                "primary_target": target,
                "suggested_targets": displayed_targets,
            }

        answer = f"Best {label} now: {target.name} — {', '.join(details)}."
        if backups:
            answer += " Backups: " + ", ".join(backups) + "."
        return {
            "text": answer,
            "requested_class": requested_marker,
            "primary_target": target,
            "suggested_targets": displayed_targets,
        }

    def _build_local_class_observing_answer(self, question: str) -> Optional[str]:
        response = self._build_local_class_observing_response(question)
        if not response:
            return None
        text = str(response.get("text", "") or "").strip()
        return text or None

    def _build_local_object_observing_answer(self, question: str, *, target: Optional[Target] = None) -> Optional[str]:
        if not _looks_like_observing_guidance_query(question):
            return None

        target = target or self._resolve_object_query_target(question, selected_target=self._selected_target_or_none())
        if target is None:
            return None

        self._ensure_known_target_type(target)
        family = self._target_class_family(target)
        metrics = self.target_metrics.get(target.name)
        best_window = self._format_target_best_window_compact(target)

        row_index: Optional[int] = None
        for idx, existing in enumerate(self.targets):
            if _targets_match(existing, target):
                row_index = idx
                break

        moon_sep_now: Optional[float] = None
        if row_index is not None and row_index < len(self.table_model.current_seps):
            candidate = self.table_model.current_seps[row_index]
            if math.isfinite(candidate):
                moon_sep_now = float(candidate)

        moon_sep_threshold = float(self.min_moon_sep_spin.value()) if hasattr(self, "min_moon_sep_spin") else 0.0
        magnitude_label = _target_magnitude_label(target)
        magnitude_text = (
            f"{magnitude_label} {float(target.magnitude):.2f}"
            if target.magnitude is not None and math.isfinite(float(target.magnitude))
            else ""
        )

        is_polish = any(
            token in _normalize_catalog_display_name(question).lower()
            for token in ("jak ", "obserw", "dzis", "dziś", "księ", "ksie", "powinienem")
        )

        if is_polish:
            lines: list[str] = []
            lead = f"{target.name}: "
            lead_parts: list[str] = []
            if best_window:
                lead_parts.append(f"najlepsze okno {best_window}")
            if metrics is not None:
                lead_parts.append(f"max alt {metrics.max_altitude_deg:.0f}°")
                lead_parts.append(f"ponad limit {metrics.hours_above_limit:.1f} h")
            if magnitude_text:
                lead_parts.append(magnitude_text)
            if lead_parts:
                lines.append(lead + ", ".join(lead_parts) + ".")
            else:
                lines.append(f"{target.name}: obserwuj go w najwyższym segmencie dostępnego okna.")

            if family in {"Supernova", "Nova", "Cataclysmic variable", "AGN"}:
                lines.append("To kompaktowy cel punktowy: priorytet to niski airmass, stabilne prowadzenie i dłuższa ciągła integracja.")
            elif family == "Galaxy":
                lines.append("To cel rozmyty: ważniejsze są ciemniejsze niebo i najwyższy fragment okna niż sam score.")
            else:
                lines.append("Priorytetem jest najwyższy fragment okna i możliwie niski airmass, a nie krótkie przeskoki między celami.")

            if moon_sep_now is not None and moon_sep_threshold > 0:
                if moon_sep_now >= moon_sep_threshold:
                    lines.append(f"Separacja od Księżyca {moon_sep_now:.1f}° jest bezpieczna względem progu {moon_sep_threshold:.0f}°.")
                else:
                    lines.append(f"Separacja od Księżyca {moon_sep_now:.1f}° jest poniżej progu {moon_sep_threshold:.0f}°; kontrast może być gorszy.")

            return "\n".join(f"- {line}" for line in lines[:3])

        lines_en: list[str] = []
        lead_parts_en: list[str] = []
        if best_window:
            lead_parts_en.append(f"best window {best_window}")
        if metrics is not None:
            lead_parts_en.append(f"max alt {metrics.max_altitude_deg:.0f}°")
            lead_parts_en.append(f"over limit {metrics.hours_above_limit:.1f} h")
        if magnitude_text:
            lead_parts_en.append(magnitude_text)
        if lead_parts_en:
            lines_en.append(f"{target.name}: " + ", ".join(lead_parts_en) + ".")
        else:
            lines_en.append(f"{target.name}: observe it during the highest-altitude segment of the available window.")

        if family in {"Supernova", "Nova", "Cataclysmic variable", "AGN"}:
            lines_en.append("Treat it as a compact point source: prioritize low airmass, stable tracking, and longer continuous integration.")
        elif family == "Galaxy":
            lines_en.append("Treat it as a diffuse target: dark sky and the highest clean segment matter more than score alone.")
        else:
            lines_en.append("Prioritize the highest segment of the window and lower airmass rather than hopping between short looks.")

        if moon_sep_now is not None and moon_sep_threshold > 0:
            if moon_sep_now >= moon_sep_threshold:
                lines_en.append(f"Moon separation {moon_sep_now:.1f}° is safely above the {moon_sep_threshold:.0f}° threshold.")
            else:
                lines_en.append(f"Moon separation {moon_sep_now:.1f}° is below the {moon_sep_threshold:.0f}° threshold, so contrast may suffer.")

        return "\n".join(f"- {line}" for line in lines_en[:3])

    @staticmethod
    def _find_normalized_text_position(text: str, candidate: str) -> Optional[int]:
        normalized_text = _normalize_catalog_display_name(text).lower()
        normalized_candidate = _normalize_catalog_display_name(candidate).lower()
        if not normalized_text or not normalized_candidate:
            return None
        direct_pos = normalized_text.find(normalized_candidate)
        if direct_pos >= 0:
            return direct_pos
        compact_text = re.sub(r"[^a-z0-9]+", "", normalized_text)
        compact_candidate = re.sub(r"[^a-z0-9]+", "", normalized_candidate)
        if len(compact_candidate) < 5:
            return None
        compact_pos = compact_text.find(compact_candidate)
        if compact_pos >= 0:
            return compact_pos
        return None

    def _extract_addable_bhtom_targets_from_ai_text(self, text: str, *, max_items: int = 4) -> list[Target]:
        raw_text = str(text or "").strip()
        if not raw_text:
            return []
        suggestions = list(getattr(self, "_bhtom_ranked_suggestions_cache", []) or [])
        if not suggestions:
            return []

        matched: list[tuple[int, int, Target]] = []
        seen: set[str] = set()
        for rank, item in enumerate(suggestions):
            target = item.get("target")
            if not isinstance(target, Target):
                continue
            if self._plan_contains_target(target):
                continue
            positions: list[int] = []
            for candidate_name in {target.name.strip(), str(target.source_object_id or "").strip()}:
                if not candidate_name:
                    continue
                pos = self._find_normalized_text_position(raw_text, candidate_name)
                if pos is not None:
                    positions.append(pos)
            if not positions:
                continue
            dedupe_key = _normalize_catalog_token(target.source_object_id or target.name)
            if not dedupe_key or dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            matched.append((min(positions), rank, target))

        matched.sort(key=lambda item: (item[0], item[1], item[2].name.lower()))
        return [target for _, _, target in matched[: max(1, int(max_items))]]

    def _resolve_recent_chat_target_reference(self, *, max_messages: int = 6) -> Optional[Target]:
        recent_state = self._recent_ai_conversation_state(max_messages=max_messages)
        primary_target = recent_state.get("primary_target")
        if isinstance(primary_target, Target):
            return primary_target
        suggested_targets = recent_state.get("suggested_targets")
        if isinstance(suggested_targets, list):
            for target in suggested_targets:
                if isinstance(target, Target):
                    return target
        if not bool(getattr(self.llm_config, "enable_chat_memory", False)):
            return None

        recent_user_texts: list[str] = []
        recent_ai_texts: list[str] = []
        for message in reversed(self._ai_messages):
            kind = str(message.get("kind", "") or "").strip().lower()
            if kind not in {"user", "ai"}:
                continue
            text = str(message.get("text", "") or "").strip()
            if not text:
                continue
            if kind == "user":
                recent_user_texts.append(text)
            else:
                recent_ai_texts.append(text)
            if len(recent_user_texts) + len(recent_ai_texts) >= max_messages:
                break

        for text in recent_user_texts:
            target = self._find_referenced_target_in_question(text)
            if target is not None:
                return target
        for text in recent_ai_texts:
            target = self._find_referenced_target_in_question(text)
            if target is not None:
                return target
        return None

    def _parse_object_query_spec(
        self,
        question: str,
        *,
        selected_target: Optional[Target] = None,
    ) -> Optional[ObjectQuerySpec]:
        text = str(question or "").strip()
        if not text:
            return None
        resolved_target = self._resolve_object_query_target(question, selected_target=selected_target)
        object_scoped = _looks_like_object_scoped_query(question)
        wants_guidance = _looks_like_observing_guidance_query(question)
        wants_fact = _looks_like_object_class_query(question) or any(
            marker in _normalize_catalog_display_name(question).lower()
            for marker in (
                "what type",
                "type of object",
                "what is",
                "what kind",
                "jaki typ",
                "jaki to typ",
                "jaki to obiekt",
                "co to za obiekt",
                "czym jest",
                "co to jest",
            )
        )
        wants_selected_llm = self._should_auto_route_selected_target_query(question, resolved_target)
        blocked_no_selection = bool(object_scoped and resolved_target is None)
        if not (object_scoped or wants_guidance or wants_fact or wants_selected_llm or blocked_no_selection):
            return None
        return ObjectQuerySpec(
            target=resolved_target,
            object_scoped=object_scoped,
            wants_guidance=wants_guidance,
            wants_fact=wants_fact,
            wants_selected_llm=wants_selected_llm,
            blocked_no_selection=blocked_no_selection,
        )

    def _parse_compare_query_spec(
        self,
        question: str,
        *,
        selected_target: Optional[Target] = None,
    ) -> Optional[CompareQuerySpec]:
        normalized = _normalize_catalog_display_name(question).lower()
        if not normalized:
            return None

        flags = _question_action_flags(
            normalized,
            {
                "compare": (
                    "compare",
                    "comparison",
                    "vs",
                    "versus",
                    "between",
                    "porown",
                    "porówn",
                    "better",
                    "lepszy",
                    "lepsza",
                ),
                "choose": (
                    "which one",
                    "which is better",
                    "which is best",
                    "which target",
                    "which object",
                    "ktory",
                    "który",
                    "ktora",
                    "która",
                    "best of",
                    "best between",
                ),
                "brightness": (
                    "brighter",
                    "brightest",
                    "jaśniejs",
                    "jasniejs",
                    "jaśniejsze",
                    "jasniejsze",
                    "najjaś",
                    "najas",
                ),
                "reason": (
                    "why",
                    "reason",
                    "justify",
                    "uzasad",
                    "dlaczego",
                    "czemu",
                ),
            },
        )

        explicit_targets = self._find_referenced_targets_in_question(question, max_targets=6)
        if len(explicit_targets) == 1 and isinstance(selected_target, Target) and not _targets_match(explicit_targets[0], selected_target):
            explicit_targets.append(selected_target)

        if len(explicit_targets) < 2 and (flags["compare"] or flags["choose"]):
            recent_targets = [
                target
                for target in list(self._recent_ai_conversation_state().get("suggested_targets", []) or [])
                if isinstance(target, Target)
            ]
            if len(recent_targets) >= 2:
                explicit_targets = recent_targets[: min(5, len(recent_targets))]

        if len(explicit_targets) < 2:
            return None
        if not (flags["compare"] or flags["choose"]):
            return None

        deduped: list[Target] = []
        seen: set[str] = set()
        for target in explicit_targets:
            dedupe_key = _normalize_catalog_token(target.source_object_id or target.name)
            if not dedupe_key or dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            deduped.append(target)

        if len(deduped) < 2:
            return None

        return CompareQuerySpec(
            targets=tuple(deduped),
            criterion="brightness" if flags["brightness"] else "overall",
            return_best_only=bool(flags["choose"]),
            include_reason=bool(flags["reason"] or flags["choose"]),
        )

    def _build_local_compare_response(
        self,
        question: str,
        *,
        compare_query: CompareQuerySpec,
    ) -> Optional[dict[str, Any]]:
        items: list[dict[str, object]] = []
        seen: set[str] = set()
        for target in compare_query.targets:
            entry = self._lookup_target_observing_candidate(target)
            if not isinstance(entry, dict):
                continue
            candidate_target = entry.get("target")
            metrics = entry.get("metrics")
            if not isinstance(candidate_target, Target) or not isinstance(metrics, TargetNightMetrics):
                continue
            dedupe_key = _normalize_catalog_token(candidate_target.source_object_id or candidate_target.name)
            if not dedupe_key or dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            items.append(entry)

        if len(items) < 2:
            return None

        is_polish = any(
            token in _normalize_catalog_display_name(question).lower()
            for token in ("porown", "porówn", "któr", "ktora", "która", "lepszy", "lepsza")
        )

        def _sort_key(item: dict[str, object]) -> tuple[object, ...]:
            target = item["target"]
            metrics = item["metrics"]
            assert isinstance(target, Target)
            assert isinstance(metrics, TargetNightMetrics)
            magnitude = (
                float(target.magnitude)
                if target.magnitude is not None and math.isfinite(float(target.magnitude))
                else float("inf")
            )
            current_alt = _safe_float(item.get("current_alt"))
            current_alt_sort = float(current_alt) if current_alt is not None else -1.0
            if compare_query.criterion == "brightness":
                return (magnitude, -float(metrics.score), -current_alt_sort, target.name.lower())
            return (-float(metrics.score), -float(metrics.hours_above_limit), -current_alt_sort, magnitude, target.name.lower())

        items.sort(key=_sort_key)
        best_item = items[0]
        best_target = best_item["target"]
        best_metrics = best_item["metrics"]
        assert isinstance(best_target, Target)
        assert isinstance(best_metrics, TargetNightMetrics)

        def _format_item_summary(item: dict[str, object]) -> str:
            target = item["target"]
            metrics = item["metrics"]
            assert isinstance(target, Target)
            assert isinstance(metrics, TargetNightMetrics)
            bits = [f"score {metrics.score:.1f}"]
            magnitude_label = _target_magnitude_label(target)
            if target.magnitude is not None and math.isfinite(float(target.magnitude)):
                bits.append(f"{magnitude_label} {float(target.magnitude):.2f}")
            best_window = str(item.get("best_window") or "").strip()
            if best_window:
                bits.append(f"best {best_window}")
            current_alt = _safe_float(item.get("current_alt"))
            if current_alt is not None and math.isfinite(current_alt):
                bits.append(f"now alt {current_alt:.1f}°")
            best_airmass = _safe_float(item.get("best_airmass"))
            if best_airmass is not None and math.isfinite(best_airmass):
                bits.append(f"min airmass {best_airmass:.2f}")
            return f"{target.name} ({', '.join(bits)})"

        compared_targets = [item["target"] for item in items if isinstance(item.get("target"), Target)]
        compared_names = ", ".join(target.name for target in compared_targets[:5])
        action_targets = [target for target in compared_targets if not self._plan_contains_target(target)]

        if compare_query.return_best_only:
            reasons: list[str] = []
            if compare_query.criterion == "brightness":
                if best_target.magnitude is not None and math.isfinite(float(best_target.magnitude)):
                    reasons.append(f"brightest with {_target_magnitude_label(best_target).lower()} {float(best_target.magnitude):.2f}")
            else:
                reasons.append(f"highest score {best_metrics.score:.1f}")
                if best_metrics.hours_above_limit > 0:
                    reasons.append(f"over limit {best_metrics.hours_above_limit:.1f} h")
            best_window = str(best_item.get("best_window") or "").strip()
            if best_window:
                reasons.append(f"best {best_window}")
            if is_polish:
                reason_text = ", ".join(reasons[:3]) if reasons else "najlepszy łączny profil obserwacyjny"
                text = f"Najlepszy wybór między {compared_names}: {best_target.name} — {reason_text}."
                return {
                    "text": text,
                    "primary_target": best_target,
                    "suggested_targets": compared_targets,
                    "action_targets": action_targets,
                }
            reason_text = ", ".join(reasons[:3]) if reasons else "best combined observing profile"
            text = f"Best choice between {compared_names}: {best_target.name} — {reason_text}."
            return {
                "text": text,
                "primary_target": best_target,
                "suggested_targets": compared_targets,
                "action_targets": action_targets,
            }

        rows = [_format_item_summary(item) for item in items]
        if is_polish:
            text = "Porównanie:\n" + "\n".join(f"- {row}" for row in rows)
        else:
            text = "Comparison:\n" + "\n".join(f"- {row}" for row in rows)
        return {
            "text": text,
            "primary_target": best_target,
            "suggested_targets": compared_targets,
            "action_targets": action_targets,
        }

    def _resolve_object_query_target(self, question: str, *, selected_target: Optional[Target]) -> Optional[Target]:
        explicit_target = self._find_referenced_target_in_question(question)
        if explicit_target is not None:
            return explicit_target

        if _looks_like_object_scoped_query(question):
            recent_target = self._resolve_recent_chat_target_reference()
            if recent_target is not None:
                return recent_target
            return selected_target

        if selected_target is not None and self._should_auto_route_selected_target_query(question, selected_target):
            return selected_target
        return None

    def _should_auto_route_selected_target_query(self, text: str, target: Optional[Target]) -> bool:
        if target is None:
            return False
        normalized = _normalize_catalog_display_name(text).lower()
        if not normalized:
            return False
        if _looks_like_object_scoped_query(normalized):
            return True

        selected_tokens = [
            _normalize_catalog_display_name(target.name).lower(),
            _normalize_catalog_display_name(target.source_object_id).lower(),
        ]
        mentions_selected_target = any(token and token in normalized for token in selected_tokens)
        if not mentions_selected_target:
            return False

        object_markers = (
            "describe",
            "details",
            "detail",
            "summary",
            "summarize",
            "tell me about",
            "what is",
            "what are",
            "info",
            "information",
            "object",
            "target",
            "obiekt",
            "opisz",
            "szczegoly",
            "szczegóły",
            "informacje",
        )
        session_markers = (
            "tonight",
            "plan",
            "schedule",
            "compare",
            "comparison",
            "order",
            "rank",
            "ranking",
            "which target",
            "which object",
            "best target",
            "other targets",
            "lista",
            "porown",
            "porówn",
            "kolejn",
            "harmonogram",
        )
        return any(marker in normalized for marker in object_markers) and not any(
            marker in normalized for marker in session_markers
        )

    def _dispatch_selected_target_llm_question(self, target: Target, question: str, *, label: str) -> None:
        prompt = self._build_selected_target_llm_prompt(target, question)
        self._dispatch_llm(
            prompt,
            tag="chat_selected",
            label=label,
            primary_target=target,
        )

    def _build_deterministic_observation_order(self) -> tuple[list[dict[str, object]], list[str]]:
        payload = self.full_payload if isinstance(getattr(self, "full_payload", None), dict) else None
        if not payload:
            return [], ["Run a visibility calculation first so night windows are available."]

        try:
            tz = pytz.timezone(str(payload.get("tz", "UTC")))
        except Exception:
            tz = pytz.UTC

        try:
            times = [t.astimezone(tz) for t in mdates.num2date(payload["times"])]
        except Exception:
            return [], ["Visibility samples are unavailable in the current plot state."]

        if not times:
            return [], ["Visibility samples are unavailable in the current plot state."]

        limit = float(self.limit_spin.value())
        sun_alt_limit = self._sun_alt_limit()
        sun_alt_series = np.array(payload.get("sun_alt", np.full(len(times), np.nan)), dtype=float)
        obs_sun_mask = np.isfinite(sun_alt_series) & (sun_alt_series <= sun_alt_limit)

        considered_rows = set(range(len(self.targets)))
        if self.table_model.row_enabled:
            visible_rows = {idx for idx, enabled in enumerate(self.table_model.row_enabled) if enabled}
            if visible_rows:
                considered_rows = visible_rows

        valid_items: list[dict[str, object]] = []
        invalid_notes: list[str] = []

        for idx, target in enumerate(self.targets):
            if idx not in considered_rows:
                continue

            target_payload = payload.get(target.name)
            if not isinstance(target_payload, dict):
                invalid_notes.append(f"{target.name}: missing altitude series in current plot data.")
                continue

            alt = np.array(target_payload.get("altitude", []), dtype=float)
            if alt.shape[0] != len(times):
                invalid_notes.append(f"{target.name}: incomplete altitude series in current plot data.")
                continue

            valid_mask = np.isfinite(alt) & (alt >= limit) & obs_sun_mask
            metrics = self.target_metrics.get(target.name)
            if not valid_mask.any():
                invalid_notes.append(f"{target.name}: no valid observing window under current constraints.")
                continue

            valid_indices = np.where(valid_mask)[0]
            runs = np.split(valid_indices, np.where(np.diff(valid_indices) != 1)[0] + 1)
            run_candidates: list[dict[str, object]] = []
            for run in runs:
                if len(run) == 0:
                    continue
                start_idx = int(run[0])
                end_idx = min(int(run[-1]) + 1, len(times) - 1)
                peak_idx = int(run[np.argmax(alt[run])])
                start_dt = times[start_idx]
                end_dt = times[end_idx]
                duration_h = max(0.0, (end_dt - start_dt).total_seconds() / 3600.0)
                still_rising = peak_idx >= int(run[-1])
                run_candidates.append(
                    {
                        "window_start": start_dt,
                        "window_end": end_dt,
                        "peak_time": times[peak_idx],
                        "window_hours": duration_h,
                        "still_rising": still_rising,
                    }
                )

            if not run_candidates:
                invalid_notes.append(f"{target.name}: no valid observing window under current constraints.")
                continue

            selected_run = min(
                run_candidates,
                key=lambda item: (
                    int(bool(item["still_rising"])),
                    item["window_end"] if bool(item["still_rising"]) else item["window_start"],
                    item["window_start"] if bool(item["still_rising"]) else item["peak_time"],
                    item["window_hours"],
                    item["peak_time"],
                ),
            )

            valid_items.append(
                {
                    "row_index": idx,
                    "name": target.name,
                    "priority": int(target.priority),
                    "score": float(metrics.score) if metrics else 0.0,
                    "hours_above_limit": float(metrics.hours_above_limit) if metrics else 0.0,
                    "max_altitude_deg": float(metrics.max_altitude_deg) if metrics else 0.0,
                    "peak_moon_sep_deg": float(metrics.peak_moon_sep_deg) if metrics else 0.0,
                    "window_start": selected_run["window_start"],
                    "window_end": selected_run["window_end"],
                    "peak_time": selected_run["peak_time"],
                    "window_hours": selected_run["window_hours"],
                    "still_rising": selected_run["still_rising"],
                }
            )

        valid_items.sort(
            key=lambda item: (
                int(bool(item["still_rising"])),
                item["window_end"] if bool(item["still_rising"]) else item["window_start"],
                item["window_start"] if bool(item["still_rising"]) else item["peak_time"],
                item["window_hours"],
                item["peak_time"],
                item["window_end"],
                item["window_start"],
                -int(item["priority"]),
                -float(item["score"]),
                str(item["name"]).lower(),
            )
        )
        return valid_items, invalid_notes

    def _format_target_coords_compact(self, target: Target) -> str:
        ra_txt = Angle(target.ra, u.deg).to_string(unit=u.hourangle, sep=":", pad=True, precision=0)
        dec_txt = Angle(target.dec, u.deg).to_string(unit=u.deg, sep=":", alwayssign=True, pad=True, precision=0)
        return f"{ra_txt} {dec_txt}"

    def _format_target_best_window_compact(self, target: Target) -> str:
        window = self.target_windows.get(target.name)
        if window is None:
            return ""
        tz_name = "UTC"
        if isinstance(getattr(self, "last_payload", None), dict):
            tz_name = str(self.last_payload.get("tz", "UTC"))
        try:
            tz = pytz.timezone(tz_name)
        except Exception:
            tz = pytz.UTC
        try:
            start = window[0].astimezone(tz).strftime("%H:%M")
            end = window[1].astimezone(tz).strftime("%H:%M")
        except Exception:
            start = window[0].strftime("%H:%M")
            end = window[1].strftime("%H:%M")
        return f"{start}-{end}"

    def _build_fast_target_llm_context(self, target: Target) -> str:
        self._ensure_known_target_type(target)
        lines: list[str] = [f"Name: {target.name}", f"Source: {_target_source_label(target.source_catalog)}"]

        source_id = target.source_object_id.strip()
        if source_id and _normalize_catalog_token(source_id) != _normalize_catalog_token(target.name):
            lines.append(f"Catalog ID: {source_id}")
        if target.object_type and not _object_type_is_unknown(target.object_type):
            lines.append(f"Type: {target.object_type}")
        class_family = self._target_class_family(target)
        if class_family:
            lines.append(f"Class family: {class_family}")

        bhtom_importance = self._bhtom_importance_for_target(target)
        if bhtom_importance is not None:
            lines.append(f"BHTOM importance: {bhtom_importance:.1f}")

        if target.magnitude is not None:
            lines.append(f"{_target_magnitude_label(target)}: {target.magnitude:.2f}")
        lines.append(f"Coords: {self._format_target_coords_compact(target)}")

        row_index: Optional[int] = None
        for idx, existing in enumerate(self.targets):
            if _targets_match(existing, target):
                row_index = idx
                break

        metrics = self.target_metrics.get(target.name)
        tonight_parts: list[str] = []
        best_window = self._format_target_best_window_compact(target)
        if best_window:
            tonight_parts.append(f"best {best_window}")
        if metrics is not None:
            tonight_parts.append(f"max alt {metrics.max_altitude_deg:.0f} deg")
            tonight_parts.append(f"over limit {metrics.hours_above_limit:.1f} h")
            tonight_parts.append(f"score {metrics.score:.1f}")
        if row_index is not None:
            if row_index < len(self.table_model.current_alts):
                alt_now = self.table_model.current_alts[row_index]
                if math.isfinite(alt_now):
                    tonight_parts.append(f"now alt {alt_now:.1f} deg")
            if row_index < len(self.table_model.current_seps):
                moon_sep_now = self.table_model.current_seps[row_index]
                if math.isfinite(moon_sep_now):
                    tonight_parts.append(f"now moon sep {moon_sep_now:.1f} deg")
        if tonight_parts:
            lines.append("Tonight: " + ", ".join(tonight_parts))

        return "\n".join(lines)

    @staticmethod
    def _format_compact_number(value: float, *, decimals_small: int = 3) -> str:
        abs_value = abs(float(value))
        if abs_value >= 100:
            return f"{value:.0f}"
        if abs_value >= 10:
            return f"{value:.1f}"
        if abs_value >= 1:
            return f"{value:.2f}"
        return f"{value:.{decimals_small}f}"

    @staticmethod
    def _format_signed_value(value: float, decimals: int = 2) -> str:
        return f"{float(value):+.{decimals}f}"

    def _is_star_like_target(self, target: Target, details: dict[str, object]) -> bool:
        object_type = _normalize_catalog_token(target.object_type)
        if "*" in object_type:
            return True
        sp_type = str(details.get("sp_type", "")).strip()
        if sp_type:
            return True
        return any(marker in object_type for marker in ("star", "nova", "binary", "cv"))

    def _format_distance_text(self, details: dict[str, object]) -> str:
        distance_value = details.get("distance_value")
        unit = str(details.get("distance_unit", "")).strip()
        if not isinstance(distance_value, (int, float)) or not unit:
            return ""
        value = float(distance_value)
        plus_err = details.get("distance_plus_err")
        minus_err = details.get("distance_minus_err")
        value_txt = self._format_compact_number(value, decimals_small=3)
        if isinstance(plus_err, (int, float)) and isinstance(minus_err, (int, float)):
            plus_abs = abs(float(plus_err))
            minus_abs = abs(float(minus_err))
            if math.isfinite(plus_abs) and math.isfinite(minus_abs):
                if abs(plus_abs - minus_abs) <= max(0.1, 0.05 * max(plus_abs, minus_abs, 1.0)):
                    err_txt = self._format_compact_number(max(plus_abs, minus_abs), decimals_small=3)
                    return f"{value_txt} +/- {err_txt} {unit}"
                plus_txt = self._format_compact_number(plus_abs, decimals_small=3)
                minus_txt = self._format_compact_number(minus_abs, decimals_small=3)
                return f"{value_txt} +{plus_txt}/-{minus_txt} {unit}"
        return f"{value_txt} {unit}"

    def _format_size_text(self, details: dict[str, object]) -> str:
        major = details.get("size_major_arcmin")
        minor = details.get("size_minor_arcmin")
        if not isinstance(major, (int, float)):
            return ""
        major_txt = self._format_compact_number(float(major), decimals_small=2)
        if isinstance(minor, (int, float)) and math.isfinite(float(minor)):
            minor_val = float(minor)
            if abs(float(major) - minor_val) > 0.05:
                minor_txt = self._format_compact_number(minor_val, decimals_small=2)
                return f"{major_txt} x {minor_txt} arcmin"
        return f"{major_txt} arcmin"

    def _format_kinematics_text(self, details: dict[str, object]) -> str:
        parts: list[str] = []
        radial_velocity = details.get("radial_velocity_kms")
        if isinstance(radial_velocity, (int, float)):
            rv_txt = self._format_compact_number(float(radial_velocity), decimals_small=2)
            parts.append(f"rv {rv_txt} km/s")
        redshift = details.get("redshift")
        if isinstance(redshift, (int, float)) and math.isfinite(float(redshift)):
            parts.append(f"z {float(redshift):+.5f}")
        return ", ".join(parts)

    def _get_simbad_compact_data(self, target: Target) -> dict[str, object]:
        cache_keys = [
            target.name.strip().lower(),
            target.source_object_id.strip().lower(),
        ]
        storage = getattr(self, "app_storage", None)
        primary_key = cache_keys[0]
        if primary_key:
            primary_cached = self._simbad_compact_cache.get(primary_key)
            if primary_cached is not None:
                return dict(primary_cached)
            if storage is not None:
                try:
                    persisted = storage.cache.get_json("simbad_compact", primary_key)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to read SIMBAD compact cache for '%s': %s", target.name, exc)
                else:
                    if isinstance(persisted, dict):
                        self._simbad_compact_cache[primary_key] = dict(persisted)
                        return dict(persisted)
        secondary_key = cache_keys[1]
        if secondary_key and secondary_key != primary_key:
            secondary_cached = self._simbad_compact_cache.get(secondary_key)
            if secondary_cached is not None:
                secondary_status = str(secondary_cached.get("_simbad_status", "")).strip().lower()
                if secondary_status == "matched":
                    return dict(secondary_cached)
            if storage is not None:
                try:
                    persisted = storage.cache.get_json("simbad_compact", secondary_key)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to read SIMBAD compact cache for '%s': %s", target.name, exc)
                else:
                    if isinstance(persisted, dict):
                        self._simbad_compact_cache[secondary_key] = dict(persisted)
                        if str(persisted.get("_simbad_status", "")).strip().lower() == "matched":
                            return dict(persisted)

        query_candidates: list[str] = []
        for candidate in (target.name, target.source_object_id):
            query = candidate.strip()
            if query and query.lower() not in {item.lower() for item in query_candidates}:
                query_candidates.append(query)

        try:
            custom = Simbad()
            custom.add_votable_fields(
                "V",
                "R",
                "B",
                "sp",
                "parallax",
                "mesdistance",
                "mesfe_h",
                "velocity",
                "galdim_majaxis",
                "galdim_minaxis",
            )
            result = None
            row_idx = 0
            match_mode = ""
            for query in query_candidates:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=NoResultsWarning)
                    result = custom.query_object(query)
                if _simbad_has_row(result):
                    match_mode = "name"
                    break
            if not _simbad_has_row(result):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=NoResultsWarning)
                    result = custom.query_region(target.skycoord, radius=120 * u.arcsec)
                row_idx = _simbad_best_row_index(result, reference_coord=target.skycoord)
                if _simbad_has_row(result):
                    match_mode = "coordinates"
        except Exception as exc:  # noqa: BLE001
            logger.warning("SIMBAD compact lookup failed for '%s': %s", target.name, exc)
            payload = {"_simbad_status": "lookup_failed"}
            if storage is not None:
                for key in cache_keys:
                    if not key:
                        continue
                    try:
                        storage.cache.set_json("simbad_compact", key, payload, ttl_s=SIMBAD_COMPACT_NEGATIVE_CACHE_TTL_S)
                    except Exception:
                        pass
            return payload

        if not _simbad_has_row(result):
            compact_data = {"_simbad_status": "not_found"}
            for key in cache_keys:
                if key:
                    self._simbad_compact_cache[key] = dict(compact_data)
                    if storage is not None:
                        try:
                            storage.cache.set_json("simbad_compact", key, compact_data, ttl_s=SIMBAD_COMPACT_NEGATIVE_CACHE_TTL_S)
                        except Exception:
                            pass
            return compact_data

        compact_data = _extract_simbad_compact_measurements(result, row_idx=row_idx)
        compact_data["_simbad_status"] = "matched"
        if match_mode:
            compact_data["_simbad_match_mode"] = match_mode
        if match_mode == "coordinates":
            row_coord = _simbad_row_coord(result, row_idx=row_idx)
            if row_coord is not None:
                try:
                    compact_data["_simbad_sep_arcsec"] = float(row_coord.separation(target.skycoord).arcsec)
                except Exception:
                    pass
        main_id = _extract_simbad_name(result, target.name, row_idx=row_idx).strip().lower()
        for key in (*cache_keys, main_id):
            if key:
                self._simbad_compact_cache[key] = dict(compact_data)
                if storage is not None:
                    try:
                        storage.cache.set_json("simbad_compact", key, compact_data, ttl_s=SIMBAD_COMPACT_CACHE_TTL_S)
                    except Exception:
                        pass
        return compact_data

    def _build_compact_target_description(self, target: Target) -> str:
        self._ensure_known_target_type(target)
        lines = [f"Source: {_target_source_label(target.source_catalog)}"]
        bhtom_importance = self._bhtom_importance_for_target(target)

        source_id = target.source_object_id.strip()
        if source_id and _normalize_catalog_token(source_id) != _normalize_catalog_token(target.name):
            lines.append(f"Catalog ID: {source_id}")
        if target.object_type:
            lines.append(f"Type: {target.object_type}")
        class_family = self._target_class_family(target)
        if class_family:
            lines.append(f"Class family: {class_family}")
        if bhtom_importance is not None:
            lines.append(f"BHTOM importance: {bhtom_importance:.1f}")
        details = self._get_simbad_compact_data(target)
        simbad_status = str(details.get("_simbad_status", "")).strip().lower()
        simbad_match_mode = str(details.get("_simbad_match_mode", "")).strip().lower()
        if simbad_status == "matched":
            if simbad_match_mode == "coordinates":
                simbad_sep_arcsec = _safe_float(details.get("_simbad_sep_arcsec"))
                if simbad_sep_arcsec is not None and math.isfinite(simbad_sep_arcsec):
                    lines.append(f"SIMBAD: coordinate match ({simbad_sep_arcsec:.1f}\")")
                else:
                    lines.append("SIMBAD: coordinate match")
            else:
                lines.append("SIMBAD: name match")
        elif simbad_status == "not_found":
            lines.append("SIMBAD: not found")
        elif simbad_status == "lookup_failed":
            lines.append("SIMBAD: unavailable")
        photometry = details.get("photometry", {})
        if photometry:
            phot_txt = ", ".join(
                f"{band} {float(photometry[band]):.2f}" for band in ("B", "V", "R")
                if isinstance(photometry, dict) and band in photometry
            )
            if phot_txt:
                lines.append(f"Photometry: {phot_txt}")
        elif target.magnitude is not None:
            lines.append(f"{_target_magnitude_label(target)}: {target.magnitude:.2f}")
        lines.append(f"Coords: {self._format_target_coords_compact(target)}")

        is_star_like = self._is_star_like_target(target, details)
        if is_star_like:
            stellar_parts: list[str] = []
            sp_type = str(details.get("sp_type", "")).strip()
            if sp_type:
                stellar_parts.append(f"SpT {sp_type}")
            parallax = details.get("parallax_mas")
            if isinstance(parallax, (int, float)):
                parallax_txt = f"{float(parallax):.3f} mas"
                parallax_err = details.get("parallax_err_mas")
                if isinstance(parallax_err, (int, float)) and math.isfinite(float(parallax_err)):
                    parallax_txt = f"{float(parallax):.3f} +/- {float(parallax_err):.3f} mas"
                stellar_parts.append(f"parallax {parallax_txt}")
            if stellar_parts:
                lines.append("Stellar: " + ", ".join(stellar_parts))

            distance_txt = self._format_distance_text(details)
            if distance_txt:
                lines.append(f"Distance: {distance_txt}")
        else:
            size_txt = self._format_size_text(details)
            if size_txt:
                lines.append(f"Angular size: {size_txt}")
            kinematics_txt = self._format_kinematics_text(details)
            if kinematics_txt:
                lines.append(f"Kinematics: {kinematics_txt}")

        physical_parts: list[str] = []
        teff_k = details.get("teff_k")
        if is_star_like and isinstance(teff_k, (int, float)):
            physical_parts.append(f"Teff {float(teff_k):.0f} K")
        fe_h = details.get("fe_h")
        if isinstance(fe_h, (int, float)):
            physical_parts.append(f"[Fe/H] {self._format_signed_value(float(fe_h), decimals=2)}")
        if physical_parts:
            lines.append(("Atmosphere: " if is_star_like else "Physical: ") + ", ".join(physical_parts))

        metrics = self.target_metrics.get(target.name)
        best_window = self._format_target_best_window_compact(target)
        tonight_parts: list[str] = []
        if best_window:
            tonight_parts.append(f"best {best_window}")
        if metrics is not None:
            tonight_parts.append(f"max alt {metrics.max_altitude_deg:.0f} deg")
            tonight_parts.append(f"over limit {metrics.hours_above_limit:.1f} h")
            tonight_parts.append(f"score {metrics.score:.1f}")
        elif self.targets:
            tonight_parts.append("run visibility calculation for tonight's window")
        if tonight_parts:
            lines.append("Tonight: " + ", ".join(tonight_parts))
        return "\n".join(lines)

    def _resolve_ai_intent(self, question: str) -> AIIntent:
        text = str(question or "").strip()
        if not text:
            return AIIntent(kind="empty", question="", label="")

        selected_target = self._selected_target_or_none()
        class_query = self._parse_class_query_spec(text)
        object_query = self._parse_object_query_spec(text, selected_target=selected_target)
        compare_query = self._parse_compare_query_spec(text, selected_target=selected_target)
        if compare_query is not None:
            local_compare_response = self._build_local_compare_response(text, compare_query=compare_query)
            if local_compare_response is not None:
                primary_target = local_compare_response.get("primary_target")
                suggested_targets = tuple(
                    target
                    for target in (local_compare_response.get("suggested_targets") or [])
                    if isinstance(target, Target)
                )
                action_targets = tuple(
                    target
                    for target in (local_compare_response.get("action_targets") or [])
                    if isinstance(target, Target)
                )
                return AIIntent(
                    kind="local_compare",
                    question=text,
                    label=text,
                    target=primary_target if isinstance(primary_target, Target) else None,
                    requested_class=class_query.requested_class if class_query is not None else "",
                    class_query=class_query,
                    object_query=object_query,
                    compare_query=compare_query,
                    local_text=str(local_compare_response.get("text", "") or "").strip(),
                    suggested_targets=suggested_targets,
                    action_targets=action_targets,
                )

        local_class_response = self._build_local_class_observing_response(text)
        if local_class_response is not None:
            requested_class = str(local_class_response.get("requested_class", "") or "").strip()
            primary_target = local_class_response.get("primary_target")
            suggested_targets = tuple(
                target
                for target in (local_class_response.get("suggested_targets") or [])
                if isinstance(target, Target)
            )
            action_targets = tuple(
                target for target in suggested_targets if not self._plan_contains_target(target)
            )
            return AIIntent(
                kind="local_class",
                question=text,
                label=text,
                target=primary_target if isinstance(primary_target, Target) else None,
                requested_class=requested_class,
                class_query=class_query,
                object_query=object_query,
                compare_query=compare_query,
                local_text=str(local_class_response.get("text", "") or "").strip(),
                suggested_targets=suggested_targets,
                action_targets=action_targets,
            )

        resolved_object_target = object_query.target if object_query is not None else None
        local_observing_answer = self._build_local_object_observing_answer(text, target=resolved_object_target)
        if local_observing_answer is not None:
            return AIIntent(
                kind="local_object_guidance",
                question=text,
                label=text,
                target=resolved_object_target if isinstance(resolved_object_target, Target) else None,
                object_query=object_query,
                compare_query=compare_query,
                local_text=local_observing_answer,
            )

        local_fact_answer = self._build_local_object_fact_answer(text, target=resolved_object_target)
        if local_fact_answer is not None:
            return AIIntent(
                kind="local_object_fact",
                question=text,
                label=text,
                target=resolved_object_target if isinstance(resolved_object_target, Target) else None,
                object_query=object_query,
                compare_query=compare_query,
                local_text=local_fact_answer,
            )

        if object_query is not None and object_query.blocked_no_selection:
            return AIIntent(kind="blocked_no_selection", question=text, label=text)

        if object_query is not None and object_query.wants_selected_llm and isinstance(resolved_object_target, Target):
            return AIIntent(
                kind="llm_selected",
                question=text,
                label=text,
                target=resolved_object_target if isinstance(resolved_object_target, Target) else None,
                object_query=object_query,
                compare_query=compare_query,
            )

        return AIIntent(
            kind="llm_session",
            question=text,
            label=text,
            target=resolved_object_target if isinstance(resolved_object_target, Target) else None,
            requested_class=class_query.requested_class if class_query is not None else "",
            class_query=class_query,
            object_query=object_query,
            compare_query=compare_query,
        )

    def _execute_ai_intent(self, intent: AIIntent) -> None:
        if intent.kind == "empty":
            return
        if intent.kind == "blocked_no_selection":
            QMessageBox.information(
                self._planner,
                "No selection",
                "Select one target first, or use a session-wide question.",
            )
            return

        worker = self._llm_worker
        if worker is not None and worker.isRunning():
            self._append_ai_message(
                "The AI assistant is still processing the previous request.",
                is_error=True,
            )
            return

        if hasattr(self, "ai_input"):
            self.ai_input.clear()
        if hasattr(self, "ai_toggle_btn") and not self.ai_toggle_btn.isChecked():
            self.ai_toggle_btn.setChecked(True)

        if intent.kind == "local_class":
            self._refresh_ai_knowledge_hint([])
            self._append_ai_message(
                intent.label,
                is_user=True,
                requested_class=intent.requested_class,
                primary_target=intent.target,
            )
            self._append_ai_message(
                intent.local_text,
                is_ai=True,
                requested_class=intent.requested_class,
                primary_target=intent.target,
                suggested_targets=list(intent.suggested_targets),
                action_targets=list(intent.action_targets),
            )
            self._set_ai_status("Ready", tone="info")
            return

        if intent.kind == "local_compare":
            self._refresh_ai_knowledge_hint([])
            self._append_ai_message(
                intent.label,
                is_user=True,
                requested_class=intent.requested_class,
                primary_target=intent.target,
            )
            self._append_ai_message(
                intent.local_text,
                is_ai=True,
                requested_class=intent.requested_class,
                primary_target=intent.target,
                suggested_targets=list(intent.suggested_targets),
                action_targets=list(intent.action_targets),
            )
            self._set_ai_status("Ready", tone="info")
            return

        if intent.kind == "local_object_guidance":
            self._refresh_ai_knowledge_hint([])
            self._append_ai_message(intent.label, is_user=True, primary_target=intent.target)
            self._append_ai_message(intent.local_text, is_ai=True, primary_target=intent.target)
            self._set_ai_status("Ready", tone="info")
            return

        if intent.kind == "local_object_fact":
            self._refresh_ai_knowledge_hint([])
            self._append_ai_message(intent.label, is_user=True, primary_target=intent.target)
            self._append_ai_message(intent.local_text, is_ai=True, primary_target=intent.target)
            self._set_ai_status("Ready", tone="info")
            return

        if intent.kind == "llm_selected" and isinstance(intent.target, Target):
            prompt = self._build_selected_target_llm_prompt(intent.target, intent.question)
            self._dispatch_llm(
                prompt,
                tag="chat_selected",
                label=intent.label,
                requested_class=intent.requested_class,
                primary_target=intent.target,
            )
            return

        if intent.kind != "llm_session":
            return

        knowledge_target = (
            intent.target
            if isinstance(intent.target, Target) and _looks_like_object_scoped_query(intent.question)
            else intent.target or self._find_referenced_target_in_question(intent.question)
        )
        context = self._build_session_context(user_question=intent.question)
        recent_memory = self._build_recent_chat_memory_block()
        knowledge_notes = self._select_knowledge_notes(question=intent.question, target=knowledge_target)
        knowledge_context = (
            "Local knowledge notes:\n"
            + "\n".join(_format_knowledge_note_snippet(note) for note in knowledge_notes)
            if knowledge_notes
            else ""
        )
        self._refresh_ai_knowledge_hint([note.title for note in knowledge_notes])
        prompt_sections = []
        if recent_memory:
            prompt_sections.append(recent_memory)
        if knowledge_context:
            prompt_sections.append(knowledge_context)
        prompt_sections.append(
            f"Current session context:\n{context}\n\nUser question: {intent.question}"
        )
        prompt = "\n\n".join(prompt_sections)
        self._dispatch_llm(
            prompt,
            tag="chat",
            label=intent.label,
            requested_class=intent.requested_class,
            primary_target=intent.target,
        )

    def _send_ai_query(self) -> None:
        text = self.ai_input.text().strip() if hasattr(self, "ai_input") else ""
        if not text:
            return
        intent = self._resolve_ai_intent(text)
        self._execute_ai_intent(intent)

    def _build_selected_target_llm_prompt(self, target: Target, question: str) -> str:
        compact_description = self._build_fast_target_llm_context(target)
        recent_memory = self._build_recent_chat_memory_block()
        knowledge_notes = self._select_knowledge_notes(question=question, target=target)
        knowledge_context = (
            "Local knowledge notes:\n"
            + "\n".join(_format_knowledge_note_snippet(note) for note in knowledge_notes)
            if knowledge_notes
            else ""
        )
        self._refresh_ai_knowledge_hint([note.title for note in knowledge_notes])
        prompt_sections: list[str] = []
        if recent_memory:
            prompt_sections.append(recent_memory)
        if knowledge_context:
            prompt_sections.append(knowledge_context)
        prompt_sections.append(
            f"Selected object context:\n"
            f"{compact_description}\n\n"
            f"User question about the selected object: {question}\n\n"
            "Answer in at most 3 short sentences or 3 short bullets and stay grounded in the selected object context. "
            "Treat the provided Type and Class family fields as authoritative for classification questions. "
            "Do not switch to a different object. Do not recommend other targets unless the user explicitly asks. "
            "Do not repeat the same metric, sentence, or phrase."
        )
        return "\n\n".join(prompt_sections)

    def _build_fast_general_llm_prompt(self, question: str) -> str:
        recent_memory = self._build_recent_chat_memory_block()
        resolved_object_target = self._resolve_object_query_target(
            question,
            selected_target=self._selected_target_or_none(),
        )
        knowledge_target = (
            resolved_object_target
            if _looks_like_object_scoped_query(question)
            else self._find_referenced_target_in_question(question)
        )
        knowledge_notes = self._select_knowledge_notes(question=question, target=knowledge_target)
        knowledge_context = (
            "Local knowledge notes:\n"
            + "\n".join(_format_knowledge_note_snippet(note) for note in knowledge_notes)
            if knowledge_notes
            else ""
        )
        self._refresh_ai_knowledge_hint([note.title for note in knowledge_notes])
        prompt_sections: list[str] = []
        if recent_memory:
            prompt_sections.append(recent_memory)
        if knowledge_context:
            prompt_sections.append(knowledge_context)
        prompt_sections.append(
            f"User question: {question}\n\n"
            "Answer concisely in no more than 4 short sentences. "
            "Do not assume details about any selected object unless the question explicitly asks about it."
        )
        return "\n\n".join(prompt_sections)

    def _build_recent_chat_memory_block(self, *, max_messages: int = 4) -> str:
        if not bool(getattr(self.llm_config, "enable_chat_memory", False)):
            return ""

        recent_sections: list[str] = []
        for message in reversed(self._ai_messages):
            kind = str(message.get("kind", "") or "").strip().lower()
            if kind not in {"user", "ai"}:
                continue
            text = _truncate_ai_memory_text(str(message.get("text", "") or "").strip())
            if not text:
                continue
            role = "User" if kind == "user" else "LLM"
            recent_sections.append(f"{role}: {text}")
            if len(recent_sections) >= max_messages:
                break

        if not recent_sections:
            return ""
        recent_sections.reverse()
        return "Recent chat turns:\n" + "\n".join(recent_sections)

    def _send_ai_selected_target_query(self) -> None:
        target = self._selected_target_or_none()

        typed_text = self.ai_input.text().strip() if hasattr(self, "ai_input") else ""
        question = typed_text or "Give a concise summary of this selected object for tonight's observing."
        if not typed_text and target is None:
            QMessageBox.information(self._planner, "No selection", "Select one target first.")
            return
        if typed_text and hasattr(self, "ai_input"):
            self.ai_input.clear()

        if target is not None and (not typed_text or self._should_auto_route_selected_target_query(question, target)):
            label = typed_text or f"Summarize {target.name}"
            self._dispatch_selected_target_llm_question(target, question, label=label)
            return

        prompt = self._build_fast_general_llm_prompt(question)
        self._dispatch_llm(prompt, tag="chat_fast", label=question)

    def _ai_describe_target(self) -> None:
        target = self._selected_target_or_none()
        if target is None:
            QMessageBox.information(self._planner, "No selection", "Select one target first.")
            return
        if hasattr(self, "ai_toggle_btn") and not self.ai_toggle_btn.isChecked():
            self.ai_toggle_btn.setChecked(True)
        self._refresh_ai_knowledge_hint([])
        self._append_ai_message(f"Describe {target.name}", is_user=True)
        self._append_ai_message(self._build_compact_target_description(target), is_ai=True)
        worker = self._llm_worker
        if worker is None or not worker.isRunning():
            self._set_ai_status("Ready", tone="info")
