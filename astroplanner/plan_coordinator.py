from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import QDate, Qt
from PySide6.QtWidgets import QDialog, QFileDialog, QInputDialog, QMessageBox

from astroplanner.models import Site, Target
from astroplanner.ui.plans import ObservationHistoryDialog, OpenSavedPlanDialog
from astroplanner.ui.targets import TargetTableModel

if TYPE_CHECKING:
    from astro_planner import MainWindow


logger = logging.getLogger(__name__)


def _normalize_catalog_token(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _site_runtime_signature(site: Site | None) -> tuple[str, float, float, float, str] | None:
    if not isinstance(site, Site):
        return None
    return (
        str(site.name or "").strip(),
        round(float(site.latitude), 6),
        round(float(site.longitude), 6),
        round(float(site.elevation), 2),
        str(site.timezone_name or "").strip(),
    )


def _same_runtime_site(first: Site | None, second: Site | None) -> bool:
    first_sig = _site_runtime_signature(first)
    second_sig = _site_runtime_signature(second)
    return first_sig is not None and first_sig == second_sig


def _should_apply_observatory_change(
    current_name: str,
    current_site: Site | None,
    next_name: str,
    next_site: Site | None,
) -> bool:
    normalized_next = str(next_name or "").strip()
    if not normalized_next or not isinstance(next_site, Site):
        return False
    if str(current_name or "").strip() != normalized_next:
        return True
    return not _same_runtime_site(current_site, next_site)


class PlanCoordinator:
    """Own plan snapshots, workspace autosave, named plans, and per-plan AI history."""

    def __init__(self, planner: "MainWindow") -> None:
        self._planner = planner

    def plan_target_key(self, target: Target) -> str:
        source_catalog = str(target.source_catalog or "").strip().lower()
        source_object_id = str(target.source_object_id or "").strip().lower()
        if source_catalog and source_object_id:
            return f"{source_catalog}:{source_object_id}"
        return _normalize_catalog_token(target.name) or f"{target.ra:.6f}:{target.dec:.6f}"

    def active_plan_storage_id(self) -> str:
        planner = self._planner
        return str(planner._active_plan_id or planner._workspace_plan_id or "")

    def serialize_current_targets(self) -> list[dict[str, object]]:
        return [target.model_dump(mode="json") for target in self._planner.targets]

    def current_site_snapshot(self) -> dict[str, object]:
        planner = self._planner
        site = getattr(planner.table_model, "site", None)
        if isinstance(site, Site):
            return site.model_dump(mode="json")
        if hasattr(planner, "obs_combo"):
            name = planner.obs_combo.currentText().strip()
            obs = planner.observatories.get(name) if hasattr(planner, "observatories") else None
            if isinstance(obs, Site):
                return obs.model_dump(mode="json")
            try:
                fallback = Site(
                    name=name or "Custom",
                    latitude=planner._read_site_float(planner.lat_edit),
                    longitude=planner._read_site_float(planner.lon_edit),
                    elevation=planner._read_site_float(planner.elev_edit),
                    limiting_magnitude=planner._current_limiting_magnitude(),
                )
            except Exception:
                return {}
            return fallback.model_dump(mode="json")
        return {}

    def current_plan_snapshot(self) -> dict[str, object]:
        planner = self._planner
        header = planner.table_view.horizontalHeader() if hasattr(planner, "table_view") else None
        selected = planner._selected_target_or_none()
        default_sort_column = TargetTableModel.COL_SCORE
        if header is not None:
            default_sort_column = int(header.sortIndicatorSection())
        return {
            "date": planner.date_edit.date().toString(Qt.ISODate) if hasattr(planner, "date_edit") else "",
            "site_name": planner.obs_combo.currentText().strip() if hasattr(planner, "obs_combo") else "",
            "site_snapshot": self.current_site_snapshot(),
            "limit_altitude": float(planner.limit_spin.value()) if hasattr(planner, "limit_spin") else 30.0,
            "sun_alt_limit": float(planner._sun_alt_limit()),
            "min_moon_sep": float(planner.min_moon_sep_spin.value()) if hasattr(planner, "min_moon_sep_spin") else 0.0,
            "min_score": float(planner.min_score_spin.value()) if hasattr(planner, "min_score_spin") else 0.0,
            "hide_observed": bool(planner.hide_observed_chk.isChecked()) if hasattr(planner, "hide_observed_chk") else False,
            "selected_target_name": str(selected.name if isinstance(selected, Target) else ""),
            "view_preset": "observation" if planner._table_matches_observation_preset() else "full",
            "default_sort_column": int(default_sort_column),
        }

    def set_plan_context(self, *, plan_id: str, plan_kind: str, plan_name: str) -> None:
        planner = self._planner
        planner._active_plan_id = str(plan_id or "")
        planner._active_plan_kind = str(plan_kind or "workspace")
        planner._active_plan_name = str(plan_name or "Workspace")
        if planner._active_plan_kind == "workspace":
            planner._workspace_plan_id = planner._active_plan_id
        self.load_ai_messages_for_active_plan()

    def serialize_ai_messages_for_storage(self) -> list[dict[str, object]]:
        planner = self._planner
        serialized: list[dict[str, object]] = []
        for idx, message in enumerate(planner._ai_messages):
            payload: dict[str, object] = {
                "kind": str(message.get("kind", "info") or "info"),
                "text": str(message.get("text", "") or ""),
                "created_at": float(
                    message.get("created_at", datetime.now(timezone.utc).timestamp())
                    or datetime.now(timezone.utc).timestamp()
                ),
                "sort_order": idx,
            }
            action_targets = message.get("action_targets")
            if isinstance(action_targets, list):
                payload["action_targets"] = [
                    target.model_dump(mode="json")
                    for target in action_targets
                    if isinstance(target, Target)
                ]
            requested_class = str(message.get("requested_class", "") or "").strip()
            if requested_class:
                payload["requested_class"] = requested_class
            primary_target = message.get("primary_target")
            if isinstance(primary_target, Target):
                payload["primary_target"] = primary_target.model_dump(mode="json")
            suggested_targets = message.get("suggested_targets")
            if isinstance(suggested_targets, list):
                payload["suggested_targets"] = [
                    target.model_dump(mode="json")
                    for target in suggested_targets
                    if isinstance(target, Target)
                ]
            serialized.append(payload)
        return serialized

    def deserialize_ai_messages_from_storage(self, rows: list[dict[str, object]]) -> list[dict[str, Any]]:
        restored: list[dict[str, Any]] = []
        for row in rows:
            message: dict[str, Any] = {
                "kind": str(row.get("kind", "info") or "info"),
                "text": str(row.get("text", "") or ""),
                "created_at": float(row.get("created_at", 0.0) or 0.0),
            }
            action_targets_payload = row.get("action_targets")
            if isinstance(action_targets_payload, list):
                action_targets: list[Target] = []
                for target_payload in action_targets_payload:
                    if not isinstance(target_payload, dict):
                        continue
                    try:
                        action_targets.append(Target(**target_payload))
                    except Exception:
                        continue
                if action_targets:
                    message["action_targets"] = action_targets
            requested_class = str(row.get("requested_class", "") or "").strip()
            if requested_class:
                message["requested_class"] = requested_class
            primary_target_payload = row.get("primary_target")
            if isinstance(primary_target_payload, dict):
                try:
                    message["primary_target"] = Target(**primary_target_payload)
                except Exception:
                    pass
            suggested_targets_payload = row.get("suggested_targets")
            if isinstance(suggested_targets_payload, list):
                suggested_targets: list[Target] = []
                for target_payload in suggested_targets_payload:
                    if not isinstance(target_payload, dict):
                        continue
                    try:
                        suggested_targets.append(Target(**target_payload))
                    except Exception:
                        continue
                if suggested_targets:
                    message["suggested_targets"] = suggested_targets
            restored.append(message)
        return restored

    def load_ai_messages_for_active_plan(self) -> None:
        planner = self._planner
        storage = getattr(planner, "app_storage", None)
        plan_id = self.active_plan_storage_id()
        if storage is None or not plan_id:
            planner._ai_messages = []
            planner._ai_message_widget_refs = []
            if hasattr(planner, "ai_output_layout"):
                planner._render_ai_messages()
            planner._refresh_ai_panel_action_buttons()
            return
        try:
            rows = storage.chat_history.list_messages(plan_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load AI history for %s: %s", plan_id, exc)
            rows = []
        planner._ai_messages = self.deserialize_ai_messages_from_storage(rows)
        planner._ai_message_widget_refs = []
        if hasattr(planner, "ai_output_layout"):
            planner._render_ai_messages()
        planner._refresh_ai_panel_action_buttons()

    def persist_ai_messages_to_storage(self, *, allow_empty_clear: bool = False) -> None:
        planner = self._planner
        storage = getattr(planner, "app_storage", None)
        plan_id = self.active_plan_storage_id()
        if storage is None or not plan_id:
            return
        try:
            serialized = self.serialize_ai_messages_for_storage()
            if serialized:
                storage.chat_history.replace_messages(plan_id, serialized)
            elif allow_empty_clear:
                storage.chat_history.clear(plan_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to persist AI history for %s: %s", plan_id, exc)

    def persist_workspace_now(self) -> None:
        planner = self._planner
        storage = getattr(planner, "app_storage", None)
        if storage is None or planner._suspend_plan_autosave:
            return
        try:
            record = storage.plans.save_workspace(self.current_plan_snapshot(), self.serialize_current_targets())
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to persist workspace: %s", exc)
            return
        workspace_id = str(record.get("id", "") or "")
        if workspace_id:
            planner._workspace_plan_id = workspace_id
            if planner._active_plan_kind == "workspace" or not planner._active_plan_id:
                planner._active_plan_id = workspace_id
                planner._active_plan_kind = "workspace"
                planner._active_plan_name = str(record.get("name", "Workspace") or "Workspace")

    def persist_active_plan_now(self) -> None:
        planner = self._planner
        if planner._suspend_plan_autosave:
            return
        self.persist_workspace_now()
        storage = getattr(planner, "app_storage", None)
        if storage is None or planner._active_plan_kind != "saved" or not planner._active_plan_id:
            self.persist_ai_messages_to_storage()
            return
        try:
            record = storage.plans.save_named(
                planner._active_plan_name,
                self.current_plan_snapshot(),
                self.serialize_current_targets(),
                plan_id=planner._active_plan_id,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to autosave active plan '%s': %s", planner._active_plan_name, exc)
        else:
            planner._active_plan_id = str(record.get("id", planner._active_plan_id) or planner._active_plan_id)
            planner._active_plan_name = str(record.get("name", planner._active_plan_name) or planner._active_plan_name)
        self.persist_ai_messages_to_storage()

    def schedule_plan_autosave(self) -> None:
        planner = self._planner
        if planner._suspend_plan_autosave or getattr(planner, "app_storage", None) is None:
            return
        planner._plan_autosave_timer.start(500)

    def flush_plan_autosave(self) -> None:
        self.persist_active_plan_now()

    def ensure_named_site_available(self, site_snapshot: dict[str, object], *, preferred_name: str) -> None:
        planner = self._planner
        if not site_snapshot:
            return
        site_name = str(preferred_name or site_snapshot.get("name") or "Custom").strip() or "Custom"
        site_snapshot = dict(site_snapshot)
        site_snapshot["name"] = site_name
        try:
            site = Site(**site_snapshot)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping invalid stored site snapshot for '%s': %s", site_name, exc)
            return
        if site_name not in planner.observatories:
            planner.observatories[site_name] = site
            planner._observatory_preset_keys[site_name] = "custom"
            planner._save_custom_observatories()
            planner._refresh_observatory_combo(selected_name=site_name)

    def apply_plan_payload(
        self,
        snapshot: dict[str, object],
        target_payloads: list[dict[str, object]],
        *,
        plan_id: str,
        plan_kind: str,
        plan_name: str,
        defer_visual_refresh: bool = False,
        apply_snapshot_date: bool = True,
    ) -> None:
        planner = self._planner
        loaded_targets: list[Target] = []
        for payload in target_payloads:
            try:
                loaded_targets.append(Target(**payload))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping invalid stored target payload: %s", exc)
        site_snapshot = snapshot.get("site_snapshot")
        site_name = str(snapshot.get("site_name", "") or "")
        if isinstance(site_snapshot, dict):
            self.ensure_named_site_available(site_snapshot, preferred_name=site_name)

        selected_target_name = str(snapshot.get("selected_target_name", "") or "")
        current_site_name = planner.obs_combo.currentText().strip() if hasattr(planner, "obs_combo") else ""
        current_site = planner.table_model.site if isinstance(getattr(planner.table_model, "site", None), Site) else None
        target_site = planner.observatories.get(site_name) if site_name else None
        planner.obs_combo.blockSignals(True)
        try:
            if site_name and site_name in planner.observatories:
                planner.obs_combo.setCurrentText(site_name)
        finally:
            planner.obs_combo.blockSignals(False)
        if _should_apply_observatory_change(current_site_name, current_site, site_name, target_site):
            planner._on_obs_change(site_name, defer_replot=defer_visual_refresh)
        elif isinstance(site_snapshot, dict):
            try:
                restored_site = Site(**site_snapshot)
            except Exception:
                restored_site = None
            if isinstance(restored_site, Site):
                planner.lat_edit.setText(f"{restored_site.latitude}")
                planner.lon_edit.setText(f"{restored_site.longitude}")
                planner.elev_edit.setText(f"{restored_site.elevation}")
                planner.table_model.site = restored_site

        if apply_snapshot_date:
            iso_date = str(snapshot.get("date", "") or "").strip()
            if iso_date:
                qdate = QDate.fromString(iso_date, Qt.ISODate)
                if qdate.isValid():
                    planner.date_edit.setDate(qdate)
        else:
            planner.date_edit.setDate(QDate.currentDate())
        if "limit_altitude" in snapshot:
            planner.limit_spin.setValue(
                int(round(float(snapshot.get("limit_altitude", planner.limit_spin.value()) or planner.limit_spin.value())))
            )
        if "sun_alt_limit" in snapshot:
            planner.sun_alt_limit_spin.setValue(
                int(round(float(snapshot.get("sun_alt_limit", planner.sun_alt_limit_spin.value()) or planner.sun_alt_limit_spin.value())))
            )
        if "min_moon_sep" in snapshot:
            planner.min_moon_sep_spin.setValue(
                int(round(float(snapshot.get("min_moon_sep", planner.min_moon_sep_spin.value()) or planner.min_moon_sep_spin.value())))
            )
        if "min_score" in snapshot:
            planner.min_score_spin.setValue(
                int(round(float(snapshot.get("min_score", planner.min_score_spin.value()) or planner.min_score_spin.value())))
            )
        if "hide_observed" in snapshot:
            planner.hide_observed_chk.setChecked(bool(snapshot.get("hide_observed")))

        planner.target_metrics.clear()
        planner.target_windows.clear()
        planner.table_model.reset_targets(loaded_targets)
        planner._recompute_recommended_order_cache()
        planner._apply_table_settings()
        default_preset = str(planner.settings.value("table/viewPreset", "full", type=str) or "full")
        preset = str(snapshot.get("view_preset", default_preset) or default_preset)
        planner._apply_column_preset(preset, save=False)
        sort_col_raw = snapshot.get("default_sort_column", TargetTableModel.COL_SCORE)
        try:
            sort_col = int(sort_col_raw)
        except Exception:
            sort_col = TargetTableModel.COL_SCORE
        if 0 <= sort_col < planner.table_model.columnCount():
            planner.table_view.sortByColumn(sort_col, Qt.AscendingOrder)
        else:
            planner._apply_default_sort()
        planner._fetch_missing_magnitudes_async()
        self.set_plan_context(plan_id=plan_id, plan_kind=plan_kind, plan_name=plan_name)
        if selected_target_name:
            for row_idx, target in enumerate(planner.targets):
                if _normalize_catalog_token(target.name) == _normalize_catalog_token(selected_target_name):
                    planner.table_view.selectRow(row_idx)
                    break
        if not defer_visual_refresh:
            planner._update_selected_details()
        planner._refresh_target_color_map()
        planner._emit_table_data_changed()
        planner._run_plan()

    def restore_plan_record(
        self,
        record: dict[str, object],
        *,
        defer_visual_refresh: bool = False,
        apply_snapshot_date: bool = False,
    ) -> None:
        planner = self._planner
        snapshot = dict(record.get("snapshot") or {})
        target_payloads = [
            dict(payload)
            for payload in record.get("targets", [])
            if isinstance(payload, dict)
        ]
        planner._suspend_plan_autosave = True
        try:
            self.apply_plan_payload(
                snapshot,
                target_payloads,
                plan_id=str(record.get("id", "") or ""),
                plan_kind=str(record.get("plan_kind", "workspace") or "workspace"),
                plan_name=str(record.get("name", "Workspace") or "Workspace"),
                defer_visual_refresh=defer_visual_refresh,
                apply_snapshot_date=apply_snapshot_date,
            )
        finally:
            planner._suspend_plan_autosave = False
        self.persist_workspace_now()

    def restore_workspace_on_startup(self, *, defer_visual_refresh: bool = False) -> bool:
        planner = self._planner
        storage = getattr(planner, "app_storage", None)
        if storage is None:
            return False
        try:
            workspace = storage.plans.load_workspace()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to restore workspace from storage: %s", exc)
            workspace = None
        if workspace:
            self.restore_plan_record(
                workspace,
                defer_visual_refresh=defer_visual_refresh,
                apply_snapshot_date=False,
            )
            if str(workspace.get("id", "")).strip():
                planner._workspace_plan_id = str(workspace.get("id"))
            return True
        self.persist_workspace_now()
        self.load_ai_messages_for_active_plan()
        return False

    def load_plan_from_json_path(self, file_path: str, *, persist_workspace: bool = True) -> None:
        planner = self._planner
        try:
            with open(file_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            target_payloads = [dict(entry) for entry in data if isinstance(entry, dict)]
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(planner, "Load plan error", str(exc))
            return
        planner._suspend_plan_autosave = True
        try:
            self.apply_plan_payload(
                {},
                target_payloads,
                plan_id=str(planner._workspace_plan_id or ""),
                plan_kind="workspace",
                plan_name="Workspace",
            )
        finally:
            planner._suspend_plan_autosave = False
        if persist_workspace:
            self.persist_workspace_now()
            self.set_plan_context(
                plan_id=str(planner._workspace_plan_id or planner._active_plan_id),
                plan_kind="workspace",
                plan_name="Workspace",
            )
            self.persist_ai_messages_to_storage()

    def import_plan_json(self) -> None:
        planner = self._planner
        fn, _ = QFileDialog.getOpenFileName(
            planner, "Import plan JSON", str(Path.cwd()), "JSON files (*.json)"
        )
        if not fn:
            return
        self.load_plan_from_json_path(fn, persist_workspace=True)

    def load_plan(self) -> None:
        self.open_saved_plan()

    def save_plan_as(self) -> None:
        planner = self._planner
        storage = getattr(planner, "app_storage", None)
        if storage is None:
            QMessageBox.information(planner, "Save Plan", "SQLite storage is not available.")
            return
        default_name = planner._active_plan_name if planner._active_plan_kind == "saved" else ""
        name, ok = QInputDialog.getText(planner, "Save Plan As", "Plan name:", text=default_name)
        if not ok:
            return
        normalized_name = str(name or "").strip()
        if not normalized_name:
            QMessageBox.information(planner, "Save Plan", "Plan name cannot be empty.")
            return
        try:
            record = storage.plans.save_named(
                normalized_name,
                self.current_plan_snapshot(),
                self.serialize_current_targets(),
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(planner, "Save Plan", f"Could not save the plan:\n{exc}")
            return
        self.persist_workspace_now()
        self.set_plan_context(
            plan_id=str(record.get("id", "") or ""),
            plan_kind="saved",
            plan_name=str(record.get("name", normalized_name) or normalized_name),
        )
        self.persist_ai_messages_to_storage()
        planner.statusBar().showMessage(f"Saved plan '{planner._active_plan_name}'.", 4000)

    def open_saved_plan(self) -> None:
        planner = self._planner
        storage = getattr(planner, "app_storage", None)
        if storage is None:
            QMessageBox.information(planner, "Open Saved Plan", "SQLite storage is not available.")
            return
        try:
            plans = storage.plans.list_saved()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(planner, "Open Saved Plan", f"Could not read saved plans:\n{exc}")
            return
        if not plans:
            QMessageBox.information(planner, "Open Saved Plan", "No saved plans were found.")
            return

        dlg = OpenSavedPlanDialog(plans, delete_plan=storage.plans.delete_plan, parent=planner)
        if dlg.exec() != QDialog.Accepted:
            return
        plan_id = dlg.selected_plan_id()
        if not plan_id:
            return
        try:
            record = storage.plans.load_plan(plan_id)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(planner, "Open Saved Plan", f"Could not load the plan:\n{exc}")
            return
        if not record:
            QMessageBox.information(planner, "Open Saved Plan", "The selected plan is no longer available.")
            return
        self.restore_plan_record(record)
        planner.statusBar().showMessage(f"Opened '{record.get('name', 'saved plan')}'.", 4000)

    def show_observation_history(self) -> None:
        planner = self._planner
        storage = getattr(planner, "app_storage", None)
        if storage is None:
            QMessageBox.information(planner, "Observation History", "SQLite storage is not available.")
            return

        dlg = ObservationHistoryDialog(
            lambda search: storage.observation_log.list_entries(search=search),
            parent=planner,
        )
        dlg.exported.connect(
            lambda file_name: planner.statusBar().showMessage(
                f"Exported observation history to {file_name}.",
                4000,
            )
        )
        dlg.exec()

    def record_observation_if_needed(self, target: Target, *, was_observed: bool, source: str) -> None:
        planner = self._planner
        if was_observed or not target.observed:
            return
        storage = getattr(planner, "app_storage", None)
        if storage is None:
            return
        site_payload = self.current_site_snapshot()
        site_name = str(
            site_payload.get("name", planner.obs_combo.currentText() if hasattr(planner, "obs_combo") else "") or ""
        )
        try:
            storage.observation_log.append(
                target_name=target.name,
                target_key=self.plan_target_key(target),
                target_payload=target.model_dump(mode="json"),
                site_name=site_name,
                site_payload=site_payload,
                notes=str(target.notes or ""),
                source=source,
                plan_id=self.active_plan_storage_id() or None,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to persist observation log entry for %s: %s", target.name, exc)
