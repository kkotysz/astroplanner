from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from astroplanner.ui.theme_utils import _set_button_variant


class OpenSavedPlanDialog(QDialog):
    """Filter, open, and delete named plans from the plan repository."""

    def __init__(
        self,
        plans: list[dict[str, object]],
        *,
        delete_plan: Callable[[str], bool],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName(self.__class__.__name__)
        self.setWindowTitle("Open Saved Plan")
        self.resize(520, 420)
        self._plans = list(plans)
        self._delete_plan = delete_plan
        self._current_plan_map: dict[str, dict[str, object]] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self.search_edit = QLineEdit(self)
        self.search_edit.setPlaceholderText("Filter saved plans...")
        self.plan_list = QListWidget(self)
        self.info_label = QLabel(self)
        self.info_label.setObjectName("SectionHint")
        self.delete_btn = QPushButton("Delete", self)
        _set_button_variant(self.delete_btn, "neutral")
        self.buttons = QDialogButtonBox(QDialogButtonBox.Open | QDialogButtonBox.Cancel, parent=self)
        self.open_btn = self.buttons.button(QDialogButtonBox.Open)

        controls_row = QHBoxLayout()
        controls_row.addWidget(self.delete_btn)
        controls_row.addStretch(1)

        layout.addWidget(self.search_edit)
        layout.addWidget(self.plan_list, 1)
        layout.addWidget(self.info_label)
        layout.addLayout(controls_row)
        layout.addWidget(self.buttons)

        self.search_edit.textChanged.connect(lambda _text: (self.refresh_items(), self.update_info()))
        self.plan_list.currentItemChanged.connect(lambda *_args: self.update_info())
        self.plan_list.itemDoubleClicked.connect(lambda *_args: self.accept())
        self.delete_btn.clicked.connect(self.delete_selected)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        self.refresh_items()
        self.update_info()

    def selected_plan_id(self) -> str:
        current_item = self.plan_list.currentItem()
        if current_item is None:
            return ""
        return str(current_item.data(Qt.UserRole) or "")

    def refresh_items(self) -> None:
        self._current_plan_map.clear()
        self.plan_list.clear()
        query = str(self.search_edit.text() or "").strip().lower()
        for item in self._plans:
            plan_name = str(item.get("name", "") or "")
            if query and query not in plan_name.lower():
                continue
            label = f"{plan_name} ({int(item.get('target_count', 0) or 0)} targets)"
            widget_item = QListWidgetItem(label, self.plan_list)
            plan_id = str(item.get("id", "") or "")
            widget_item.setData(Qt.UserRole, plan_id)
            self._current_plan_map[plan_id] = item
        has_selection = self.plan_list.count() > 0
        if has_selection:
            self.plan_list.setCurrentRow(0)
        self.open_btn.setEnabled(has_selection)
        self.delete_btn.setEnabled(has_selection)

    def update_info(self) -> None:
        current_item = self.plan_list.currentItem()
        if current_item is None:
            self.info_label.setText("No saved plan selected.")
            return
        plan_id = str(current_item.data(Qt.UserRole) or "")
        plan = self._current_plan_map.get(plan_id, {})
        updated_at = float(plan.get("updated_at", 0.0) or 0.0)
        if updated_at > 0:
            stamp = datetime.fromtimestamp(updated_at, tz=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M")
        else:
            stamp = "unknown"
        self.info_label.setText(
            f"Targets: {int(plan.get('target_count', 0) or 0)} | Last update: {stamp}"
        )

    def delete_selected(self) -> None:
        plan_id = self.selected_plan_id()
        if not plan_id:
            return
        plan = self._current_plan_map.get(plan_id, {})
        plan_name = str(plan.get("name", "this plan") or "this plan")
        answer = QMessageBox.question(
            self,
            "Delete Saved Plan",
            f"Delete '{plan_name}' from SQLite storage?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        try:
            deleted = bool(self._delete_plan(plan_id))
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Delete Saved Plan", f"Could not delete the plan:\n{exc}")
            return
        if not deleted:
            QMessageBox.information(self, "Delete Saved Plan", "The selected plan no longer exists.")
            return
        self._plans = [item for item in self._plans if str(item.get("id", "")) != plan_id]
        self.refresh_items()
        self.update_info()


class ObservationHistoryDialog(QDialog):
    """Display and export persisted observation log entries."""

    exported = Signal(str)

    def __init__(
        self,
        list_entries: Callable[[str], list[dict[str, object]]],
        *,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName(self.__class__.__name__)
        self.setWindowTitle("Observation History")
        self.resize(880, 460)
        self._list_entries = list_entries
        self._current_rows: list[dict[str, object]] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self.search_edit = QLineEdit(self)
        self.search_edit.setPlaceholderText("Search by target or site...")
        self.table = QTableWidget(self)
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Observed At", "Target", "Site", "Source", "Notes"])
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.Stretch)

        self.export_btn = QPushButton("Export CSV", self)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Close, parent=self)

        controls = QHBoxLayout()
        controls.addWidget(self.export_btn)
        controls.addStretch(1)

        layout.addWidget(self.search_edit)
        layout.addWidget(self.table, 1)
        layout.addLayout(controls)
        layout.addWidget(self.buttons)

        self.search_edit.textChanged.connect(lambda _text: self.refresh_table())
        self.export_btn.clicked.connect(self.export_csv)
        self.buttons.rejected.connect(self.reject)

        self.refresh_table()

    def _load_rows(self) -> list[dict[str, object]]:
        try:
            return self._list_entries(str(self.search_edit.text() or ""))
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Observation History", f"Could not load history:\n{exc}")
            return []

    def refresh_table(self) -> None:
        rows = self._load_rows()
        self._current_rows = rows
        self.table.setRowCount(len(rows))
        for row_idx, row in enumerate(rows):
            observed_at = float(row.get("observed_at", 0.0) or 0.0)
            if observed_at > 0:
                stamp = datetime.fromtimestamp(observed_at, tz=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M")
            else:
                stamp = "-"
            values = [
                stamp,
                str(row.get("target_name", "") or ""),
                str(row.get("site_name", "") or ""),
                str(row.get("source", "") or ""),
                str(row.get("notes", "") or ""),
            ]
            for col_idx, value in enumerate(values):
                self.table.setItem(row_idx, col_idx, QTableWidgetItem(value))
        self.table.resizeRowsToContents()

    def export_csv(self) -> None:
        if not self._current_rows:
            QMessageBox.information(self, "Observation History", "No observation entries to export.")
            return
        default_name = f"astroplanner-observation-history-{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Observation History",
            str(Path.home() / default_name),
            "CSV files (*.csv)",
        )
        if not file_path:
            return
        try:
            with open(file_path, "w", encoding="utf-8", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(["observed_at", "target_name", "target_key", "site_name", "source", "notes"])
                for row in self._current_rows:
                    writer.writerow(
                        [
                            float(row.get("observed_at", 0.0) or 0.0),
                            str(row.get("target_name", "") or ""),
                            str(row.get("target_key", "") or ""),
                            str(row.get("site_name", "") or ""),
                            str(row.get("source", "") or ""),
                            str(row.get("notes", "") or ""),
                        ]
                    )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Observation History", f"Could not export CSV:\n{exc}")
            return
        self.exported.emit(Path(file_path).name)
