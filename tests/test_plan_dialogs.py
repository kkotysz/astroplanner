from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from astroplanner.ui.plans import ObservationHistoryDialog, OpenSavedPlanDialog


def test_open_saved_plan_dialog_filters_and_selects_plan() -> None:
    app = QApplication.instance() or QApplication([])
    assert app is not None

    dialog = OpenSavedPlanDialog(
        [
            {"id": "plan-1", "name": "Moon Night", "target_count": 2, "updated_at": 0.0},
            {"id": "plan-2", "name": "Deep Sky", "target_count": 5, "updated_at": 0.0},
        ],
        delete_plan=lambda _plan_id: True,
    )

    assert dialog.selected_plan_id() == "plan-1"
    dialog.search_edit.setText("deep")

    assert dialog.plan_list.count() == 1
    assert dialog.selected_plan_id() == "plan-2"


def test_observation_history_dialog_refreshes_rows_from_callback() -> None:
    app = QApplication.instance() or QApplication([])
    assert app is not None

    searches: list[str] = []

    def list_entries(search: str) -> list[dict[str, object]]:
        searches.append(search)
        return [
            {
                "observed_at": 0.0,
                "target_name": "M31",
                "target_key": "m31",
                "site_name": "WRO",
                "source": "manual",
                "notes": search,
            }
        ]

    dialog = ObservationHistoryDialog(list_entries)

    assert dialog.table.rowCount() == 1
    dialog.search_edit.setText("m31")

    assert searches[-1] == "m31"
    assert dialog.table.item(0, 1).text() == "M31"
