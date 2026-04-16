from __future__ import annotations

import os
from time import perf_counter

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication, QWidget

from astroplanner.ai import LLMConfig
from astroplanner.ai_panel_coordinator import AIPanelCoordinator


class _DummyAIPlanner(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.llm_config = LLMConfig(url="http://localhost:11434", model="gemma4:e4b")
        self._llm_last_warmup_key = ("", "")
        self._llm_last_warmup_at = 0.0
        self._llm_active_tag = ""
        self._llm_worker = None
        self._llm_warmup_silent = False
        self._llm_startup_warmup_attempted = False
        self._llm_active_requested_class = ""
        self._llm_active_primary_target = None
        self._ai_runtime_status = ""
        self._ai_runtime_status_tone = "info"
        self._ai_status_error_clear_s = 0
        self._ai_status_clear_timer = QTimer(self)
        self._ai_last_knowledge_titles: list[str] = []
        self._ai_messages: list[dict[str, object]] = []
        self._apply_font_calls = 0
        self._refresh_action_calls = 0

    def _theme_qcolor(self, _key: str, fallback: str) -> QColor:
        return QColor(fallback)

    def _apply_ai_chat_font(self) -> None:
        self._apply_font_calls += 1

    def _refresh_ai_panel_action_buttons(self) -> None:
        self._refresh_action_calls += 1

    def _ai_describe_target(self) -> None:
        return

    def _copy_last_ai_response(self) -> None:
        return

    def _export_ai_chat(self) -> None:
        return

    def _reset_ai_chat_appearance(self) -> None:
        return

    def _open_ai_settings(self) -> None:
        return

    def _warmup_llm_manual(self) -> None:
        return

    def _clear_ai_messages(self) -> None:
        return

    def _send_ai_query(self) -> None:
        return


def test_ai_panel_coordinator_smoke() -> None:
    app = QApplication.instance() or QApplication([])
    assert app is not None

    planner = _DummyAIPlanner()
    coordinator = AIPanelCoordinator(planner)
    panel = coordinator.build_panel()

    assert panel.objectName() == "AIAssistantPanel"
    assert planner.ai_input.placeholderText()
    assert planner._apply_font_calls == 1
    assert planner._refresh_action_calls >= 1

    coordinator.set_status("Ready", tone="info")
    assert planner._ai_runtime_status == "Ready"
    assert planner._ai_runtime_status_tone == "info"

    planner._llm_last_warmup_key = coordinator.warmup_cache_key()
    planner._llm_last_warmup_at = perf_counter()
    assert coordinator.is_warm() is True
