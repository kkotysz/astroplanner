from __future__ import annotations

import logging
from time import perf_counter
from typing import TYPE_CHECKING, Optional

from PySide6.QtCore import QObject, QSignalBlocker, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from astroplanner.ai import LLMConfig, LLMWorker
from astroplanner.i18n import (
    current_language,
    localize_widget_tree,
    set_translated_text,
    set_translated_tooltip,
)
from astroplanner.models import Target
from astroplanner.ui.common import _fit_dialog_to_screen
from astroplanner.ui.theme_utils import (
    _set_button_icon_kind,
    _set_button_variant,
    _set_dynamic_property,
    _set_label_tone,
)

if TYPE_CHECKING:
    from astro_planner import MainWindow


logger = logging.getLogger(__name__)


class AIPanelCoordinator(QObject):
    def __init__(self, planner: "MainWindow") -> None:
        super().__init__(planner)
        self._planner = planner

    def open_settings(self) -> None:
        self._planner.open_general_settings(initial_tab="AI")

    def refresh_context_hint(self) -> None:
        planner = self._planner
        memory_enabled = bool(getattr(planner.llm_config, "enable_chat_memory", False))
        if memory_enabled:
            hint_text = (
                "Short chat memory enabled.\n"
                "Last 1-2 turns may be reused."
            )
            hint_tooltip = (
                "Examples: 'Którą z tych SN najlepiej dziś obserwować?' or "
                "'How should I observe 8C0716_714 tonight?'\n\n"
                "Short chat memory can help follow-up questions, but requests may be slower."
            )
            input_tooltip = (
                "The AI chat is saved per active plan/workspace. "
                "Prompts may reuse the last 1-2 user/LLM turns for follow-up questions. "
                "Use complete questions when the context could be ambiguous."
            )
        else:
            hint_text = (
                "No chat memory.\n"
                "Ask complete questions with the object or class name."
            )
            hint_tooltip = (
                "Examples: 'Którą SN z BHTOM najlepiej dziś obserwować?' or "
                "'How should I observe 8C0716_714 tonight?'"
            )
            input_tooltip = (
                "The AI chat is saved per active plan/workspace. "
                "Prompts do not reuse previous turns, so use complete questions with the object or class name, "
                "e.g. 'Która SN z BHTOM jest dziś najlepsza?'"
            )

        if hasattr(planner, "ai_context_hint"):
            set_translated_text(planner.ai_context_hint, hint_text, current_language())
            set_translated_tooltip(planner.ai_context_hint, hint_tooltip, current_language())
        if hasattr(planner, "ai_input"):
            set_translated_tooltip(planner.ai_input, input_tooltip, current_language())

    def refresh_knowledge_hint(self, titles: Optional[list[str]] = None) -> None:
        planner = self._planner
        if titles is not None:
            planner._ai_last_knowledge_titles = [str(title).strip() for title in titles if str(title).strip()]
        current_titles = list(getattr(planner, "_ai_last_knowledge_titles", []) or [])
        if current_titles:
            if len(current_titles) > 3:
                visible = ", ".join(current_titles[:3]) + f" (+{len(current_titles) - 3} more)"
            else:
                visible = ", ".join(current_titles)
            text = f"Using local knowledge: {visible}"
            tooltip = "Local knowledge notes used for the most recent AI prompt:\n- " + "\n- ".join(current_titles)
        else:
            text = "Using local knowledge: none"
            tooltip = "No local knowledge notes were added to the most recent AI prompt."
        if hasattr(planner, "ai_knowledge_hint"):
            set_translated_text(planner.ai_knowledge_hint, text, current_language())
            set_translated_tooltip(planner.ai_knowledge_hint, tooltip, current_language())

    def focus_input(self) -> None:
        planner = self._planner
        ai_window = self.ensure_window()
        if hasattr(planner, "ai_toggle_btn") and not planner.ai_toggle_btn.isChecked():
            planner.ai_toggle_btn.setChecked(True)
        if isinstance(ai_window, QDialog):
            ai_window.show()
            ai_window.raise_()
            ai_window.activateWindow()
        if hasattr(planner, "ai_input"):
            planner.ai_input.setFocus(Qt.OtherFocusReason)
            planner.ai_input.selectAll()

    def build_panel(self, parent: Optional[QWidget] = None) -> QWidget:
        planner = self._planner
        host = parent if parent is not None else planner
        panel = QFrame(host)
        panel.setObjectName("AIAssistantPanel")
        _set_dynamic_property(panel, "accented", True)

        layout = QHBoxLayout(panel)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        btn_col = QVBoxLayout()
        btn_col.setSpacing(6)

        describe_btn = QPushButton("Describe Object")
        describe_btn.clicked.connect(planner._ai_describe_target)
        btn_col.addWidget(describe_btn)
        _set_button_variant(describe_btn, "secondary")
        _set_button_icon_kind(describe_btn, "describe")
        describe_btn.ensurePolished()

        describe_hint = QLabel("Uses local metadata only.\nNo LLM request.", panel)
        describe_hint.setObjectName("SectionHint")
        describe_hint.setWordWrap(True)
        btn_col.addWidget(describe_hint)

        planner.ai_copy_last_btn = QPushButton("Copy Last Reply")
        planner.ai_copy_last_btn.clicked.connect(planner._copy_last_ai_response)
        _set_button_variant(planner.ai_copy_last_btn, "ghost")
        _set_button_icon_kind(planner.ai_copy_last_btn, "describe")
        btn_col.addWidget(planner.ai_copy_last_btn)

        planner.ai_export_chat_btn = QPushButton("Export Chat")
        planner.ai_export_chat_btn.clicked.connect(planner._export_ai_chat)
        _set_button_variant(planner.ai_export_chat_btn, "ghost")
        _set_button_icon_kind(planner.ai_export_chat_btn, "save")
        btn_col.addWidget(planner.ai_export_chat_btn)

        planner.ai_reset_appearance_btn = QPushButton("Reset Appearance")
        planner.ai_reset_appearance_btn.clicked.connect(planner._reset_ai_chat_appearance)
        _set_button_variant(planner.ai_reset_appearance_btn, "ghost")
        _set_button_icon_kind(planner.ai_reset_appearance_btn, "refresh")
        btn_col.addWidget(planner.ai_reset_appearance_btn)

        planner.ai_settings_btn = QPushButton("AI Settings")
        planner.ai_settings_btn.clicked.connect(planner._open_ai_settings)
        _set_button_variant(planner.ai_settings_btn, "ghost")
        _set_button_icon_kind(planner.ai_settings_btn, "edit")
        btn_col.addWidget(planner.ai_settings_btn)

        planner.ai_context_hint = QLabel("", panel)
        planner.ai_context_hint.setObjectName("SectionHint")
        planner.ai_context_hint.setWordWrap(True)
        btn_col.addWidget(planner.ai_context_hint)

        planner.ai_knowledge_hint = QLabel("Using local knowledge: none", panel)
        planner.ai_knowledge_hint.setObjectName("SectionHint")
        planner.ai_knowledge_hint.setWordWrap(True)
        planner.ai_knowledge_hint.setToolTip(
            "Shows which local knowledge notes were added to the most recent AI prompt."
        )
        btn_col.addWidget(planner.ai_knowledge_hint)

        warmup_btn = QPushButton("Warm Up LLM", panel)
        warmup_btn.clicked.connect(planner._warmup_llm_manual)
        _set_button_variant(warmup_btn, "ghost")
        _set_button_icon_kind(warmup_btn, "refresh")
        warmup_btn.setToolTip("Send a lightweight request to warm the current LLM model.")
        btn_col.addWidget(warmup_btn)

        clear_btn = QPushButton("Clear", panel)
        clear_btn.clicked.connect(planner._clear_ai_messages)
        _set_button_variant(clear_btn, "ghost")
        _set_button_icon_kind(clear_btn, "clear")
        btn_col.addWidget(clear_btn)

        status_row = QWidget(panel)
        status_row_l = QHBoxLayout(status_row)
        status_row_l.setContentsMargins(0, 0, 0, 0)
        status_row_l.setSpacing(0)

        planner.ai_warm_badge_label = QLabel(f"● cold {planner.llm_config.model}", status_row)
        planner.ai_warm_badge_label.setObjectName("SectionHint")
        planner.ai_warm_badge_label.setToolTip(planner.llm_config.model)
        status_row_l.addWidget(planner.ai_warm_badge_label, 1)
        _set_label_tone(planner.ai_warm_badge_label, "muted")
        btn_col.addWidget(status_row)

        btn_col.addStretch(1)
        btn_widget = QWidget(panel)
        btn_widget.setLayout(btn_col)
        describe_width = max(
            210,
            describe_btn.sizeHint().width() + 24,
            planner.ai_copy_last_btn.sizeHint().width() + 24,
            planner.ai_export_chat_btn.sizeHint().width() + 24,
            planner.ai_reset_appearance_btn.sizeHint().width() + 24,
            planner.ai_settings_btn.sizeHint().width() + 24,
            warmup_btn.sizeHint().width() + 24,
            clear_btn.sizeHint().width() + 24,
            244,
        )
        describe_btn.setMinimumWidth(describe_width - 12)
        describe_hint.setFixedWidth(describe_width - 12)
        planner.ai_copy_last_btn.setMinimumWidth(describe_width - 12)
        planner.ai_export_chat_btn.setMinimumWidth(describe_width - 12)
        planner.ai_reset_appearance_btn.setMinimumWidth(describe_width - 12)
        planner.ai_settings_btn.setMinimumWidth(describe_width - 12)
        planner.ai_context_hint.setFixedWidth(describe_width - 12)
        planner.ai_knowledge_hint.setFixedWidth(describe_width - 12)
        warmup_btn.setMinimumWidth(describe_width - 12)
        clear_btn.setMinimumWidth(describe_width - 12)
        btn_widget.setFixedWidth(describe_width)
        layout.addWidget(btn_widget)

        planner._ai_output_placeholder_text = (
            "AI responses will appear here.\n"
            "Configure Ollama or another local LLM in Settings -> General Settings."
        )
        planner.ai_output = QScrollArea(panel)
        planner.ai_output.setWidgetResizable(True)
        planner.ai_output.setFrameShape(QFrame.NoFrame)
        planner.ai_output.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        planner.ai_output.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        planner.ai_output.setObjectName("AIChatScroll")
        planner.ai_output.installEventFilter(planner)
        planner._ai_output_last_viewport_width = 0
        planner.ai_output_content = QWidget(planner.ai_output)
        planner.ai_output_content.setObjectName("AIChatContent")
        planner.ai_output_layout = QVBoxLayout(planner.ai_output_content)
        planner.ai_output_layout.setContentsMargins(12, 12, 12, 12)
        planner.ai_output_layout.setSpacing(0)
        planner.ai_output.setWidget(planner.ai_output_content)
        planner._apply_ai_chat_font()
        center_widget = QWidget(panel)
        center_col = QVBoxLayout(center_widget)
        center_col.setContentsMargins(0, 0, 0, 0)
        center_col.setSpacing(8)
        center_col.addWidget(planner.ai_output, 1)

        composer_row = QHBoxLayout()
        composer_row.setContentsMargins(0, 0, 0, 0)
        composer_row.setSpacing(8)

        planner.ai_input = QLineEdit(center_widget)
        planner.ai_input.setPlaceholderText("Ask about tonight or the selected object...")
        planner.ai_input.returnPressed.connect(planner._send_ai_query)
        composer_row.addWidget(planner.ai_input, 1)

        send_btn = QPushButton("Send", center_widget)
        send_btn.clicked.connect(planner._send_ai_query)
        _set_button_variant(send_btn, "primary")
        _set_button_icon_kind(send_btn, "send")
        send_btn.setToolTip(
            "Send the current question to the local LLM. Object-specific questions are auto-routed to a lighter selected-target prompt."
        )
        send_btn.setMinimumWidth(max(132, send_btn.sizeHint().width() + 16))
        composer_row.addWidget(send_btn)
        center_col.addLayout(composer_row)
        layout.addWidget(center_widget, 1)
        self.refresh_context_hint()
        self.refresh_knowledge_hint()
        planner._refresh_ai_panel_action_buttons()
        self.refresh_warm_indicator()

        return panel

    def build_window(self) -> QDialog:
        planner = self._planner
        window = QDialog(planner)
        window.setObjectName("AIAssistantWindow")
        window.setWindowTitle("AI Assistant")
        window.setModal(False)
        _fit_dialog_to_screen(
            window,
            preferred_width=1180,
            preferred_height=620,
            min_width=860,
            min_height=360,
        )

        window_l = QVBoxLayout(window)
        window_l.setContentsMargins(8, 8, 8, 8)
        window_l.setSpacing(6)
        panel = self.build_panel(window)
        window_l.addWidget(panel, 1)
        window.finished.connect(self.on_window_finished)
        localize_widget_tree(window, current_language())
        return window

    def ensure_window(self) -> QDialog:
        planner = self._planner
        ai_window = getattr(planner, "ai_window", None)
        if isinstance(ai_window, QDialog):
            return ai_window
        ai_window = self.build_window()
        ai_window.setStyleSheet(planner.styleSheet())
        planner.ai_window = ai_window
        if getattr(planner, "_ai_messages", None):
            planner._render_ai_messages()
        planner._refresh_ai_panel_action_buttons()
        self.refresh_context_hint()
        self.refresh_knowledge_hint()
        self.refresh_warm_indicator()
        return ai_window

    def toggle_panel(self, checked: bool) -> None:
        planner = self._planner
        ai_window = self.ensure_window() if checked else getattr(planner, "ai_window", None)
        if isinstance(ai_window, QDialog):
            if checked:
                ai_window.show()
                ai_window.raise_()
                ai_window.activateWindow()
                self.warmup_if_needed()
            else:
                ai_window.hide()
        if hasattr(planner, "ai_toggle_btn"):
            set_translated_text(planner.ai_toggle_btn, "Hide AI" if checked else "AI", current_language())

    def on_window_finished(self, _result: int) -> None:
        planner = self._planner
        if not hasattr(planner, "ai_toggle_btn"):
            return
        if not planner.ai_toggle_btn.isChecked():
            return
        blocker = QSignalBlocker(planner.ai_toggle_btn)
        planner.ai_toggle_btn.setChecked(False)
        del blocker
        set_translated_text(planner.ai_toggle_btn, "AI", current_language())

    def warmup_cache_key(self) -> tuple[str, str]:
        planner = self._planner
        return (
            str(getattr(planner.llm_config, "url", "") or "").strip().rstrip("/"),
            str(getattr(planner.llm_config, "model", "") or "").strip(),
        )

    def is_warm(self) -> bool:
        planner = self._planner
        cache_key = self.warmup_cache_key()
        if planner._llm_last_warmup_key != cache_key:
            return False
        return (perf_counter() - float(planner._llm_last_warmup_at)) < 600.0

    def start_warmup(
        self,
        *,
        force: bool = False,
        user_initiated: bool = False,
        silent: bool = False,
    ) -> bool:
        planner = self._planner
        worker = planner._llm_worker
        if worker is not None and worker.isRunning():
            if user_initiated:
                self.set_status("AI assistant is busy.", tone="warning")
            return False
        if not force and self.is_warm():
            if not silent:
                self.set_status("Ready", tone="info")
            return False
        planner._llm_warmup_silent = bool(silent)
        planner._llm_active_requested_class = ""
        planner._llm_active_primary_target = None
        if not silent:
            self.set_status("Warming up LLM...", tone="info")
        worker = LLMWorker(
            config=planner.llm_config,
            prompt="Reply with OK.",
            system_prompt="Reply with exactly OK.",
            tag="warmup",
            parent=planner,
        )
        worker.responseChunk.connect(self.on_chunk)
        worker.responseReady.connect(self.on_response)
        worker.errorOccurred.connect(self.on_error)
        worker.finished.connect(self.on_worker_finished)
        planner._llm_worker = worker
        planner._llm_active_tag = "warmup"
        self.refresh_warm_indicator()
        worker.start()
        return True

    def warmup_if_needed(self) -> None:
        self.start_warmup(force=False, user_initiated=False, silent=False)

    def warmup_manual(self) -> None:
        self.start_warmup(force=True, user_initiated=True, silent=False)

    def warmup_on_startup(self) -> None:
        planner = self._planner
        if planner._llm_startup_warmup_attempted or getattr(planner, "_shutting_down", False):
            return
        planner._llm_startup_warmup_attempted = True
        self.start_warmup(force=False, user_initiated=False, silent=True)

    def refresh_warm_indicator(self) -> None:
        planner = self._planner
        configured_model = str(getattr(planner.llm_config, "model", "") or "").strip() or LLMConfig.DEFAULT_MODEL
        is_warm = self.is_warm()
        active_tag = str(getattr(planner, "_llm_active_tag", "") or "").strip().lower()
        runtime_status = str(getattr(planner, "_ai_runtime_status", "") or "").strip().lower()
        model_text = configured_model
        if is_warm and planner._llm_last_warmup_key[1]:
            model_text = planner._llm_last_warmup_key[1]

        badge_text = f"● cold {model_text}"
        badge_tone = "muted"
        badge_glow_color = planner._theme_qcolor("section_hint", "#8fa3b8")

        if runtime_status == "warm-up failed":
            badge_text = f"! warm-up failed {model_text}"
            badge_tone = "warning"
            badge_glow_color = planner._theme_qcolor("state_warning", "#ffcc66")
        elif active_tag == "warmup":
            badge_text = f"◌ warming {model_text}"
            badge_tone = "info"
            badge_glow_color = planner._theme_qcolor("state_info", "#59f3ff")
        elif active_tag and runtime_status.startswith("stream"):
            badge_text = f"✦ streaming {model_text}"
            badge_tone = "info"
            badge_glow_color = planner._theme_qcolor("state_info", "#59f3ff")
        elif active_tag:
            badge_text = f"✦ thinking {model_text}"
            badge_tone = "info"
            badge_glow_color = planner._theme_qcolor("state_info", "#59f3ff")
        elif is_warm:
            badge_text = f"● warm {model_text}"
            badge_tone = "success"
            badge_glow_color = planner._theme_qcolor("state_success", "#67ff9a")

        if hasattr(planner, "ai_warm_badge_label"):
            set_translated_text(planner.ai_warm_badge_label, badge_text, current_language())
            set_translated_tooltip(planner.ai_warm_badge_label, badge_text, current_language())
            _set_label_tone(planner.ai_warm_badge_label, badge_tone)
            self.apply_label_glow_effect(planner.ai_warm_badge_label, badge_glow_color, blur_radius=22.0)

    def set_status(self, text: str, *, tone: str = "info") -> None:
        planner = self._planner
        planner._ai_runtime_status = str(text)
        planner._ai_runtime_status_tone = str(tone)
        if tone in {"warning", "error"} and planner._ai_status_error_clear_s > 0:
            planner._ai_status_clear_timer.start(int(planner._ai_status_error_clear_s * 1000))
        else:
            planner._ai_status_clear_timer.stop()
        self.refresh_warm_indicator()

    def clear_runtime_status(self) -> None:
        planner = self._planner
        if str(getattr(planner, "_llm_active_tag", "") or "").strip():
            return
        planner._ai_runtime_status = ""
        planner._ai_runtime_status_tone = "info"
        self.refresh_warm_indicator()

    @staticmethod
    def apply_label_glow_effect(label: Optional[QLabel], color: QColor, *, blur_radius: float) -> None:
        if label is None:
            return
        current = label.graphicsEffect()
        if not isinstance(current, QGraphicsDropShadowEffect):
            effect = QGraphicsDropShadowEffect(label)
            effect.setOffset(0.0, 0.0)
            label.setGraphicsEffect(effect)
        else:
            effect = current
        glow_color = QColor(color)
        if not glow_color.isValid():
            glow_color = QColor("#59f3ff")
        if glow_color.alpha() < 210:
            glow_color.setAlpha(210)
        effect.setBlurRadius(float(blur_radius))
        effect.setColor(glow_color)

    def dispatch_llm(
        self,
        prompt: str,
        tag: str,
        label: str,
        *,
        requested_class: str = "",
        primary_target: Optional[Target] = None,
    ) -> None:
        planner = self._planner
        worker = planner._llm_worker
        if worker is not None and worker.isRunning():
            planner._append_ai_message(
                "The AI assistant is still processing the previous request.",
                is_error=True,
            )
            return
        if hasattr(planner, "ai_toggle_btn") and not planner.ai_toggle_btn.isChecked():
            planner.ai_toggle_btn.setChecked(True)
        planner._append_ai_message(
            label,
            is_user=True,
            requested_class=requested_class,
            primary_target=primary_target,
        )
        planner._start_ai_response_message()
        self.set_status("Thinking...", tone="info")
        planner._llm_active_requested_class = str(requested_class or "").strip()
        planner._llm_active_primary_target = primary_target if isinstance(primary_target, Target) else None
        worker = LLMWorker(
            config=planner.llm_config,
            prompt=prompt,
            system_prompt=planner._SYSTEM_PROMPT,
            tag=tag,
            parent=planner,
        )
        worker.responseChunk.connect(self.on_chunk)
        worker.responseReady.connect(self.on_response)
        worker.errorOccurred.connect(self.on_error)
        worker.finished.connect(self.on_worker_finished)
        planner._llm_worker = worker
        planner._llm_active_tag = tag
        self.refresh_warm_indicator()
        worker.start()

    def on_chunk(self, tag: str, text: str) -> None:
        planner = self._planner
        if not text:
            return
        if tag == "warmup":
            return
        self.set_status("Streaming...", tone="info")
        planner._append_ai_stream_chunk(text)

    def on_response(self, tag: str, text: str) -> None:
        planner = self._planner
        if tag == "warmup":
            planner._llm_last_warmup_key = self.warmup_cache_key()
            planner._llm_last_warmup_at = perf_counter()
            self.set_status("Ready", tone="info")
            return
        logger.info("LLM response received (tag=%s, length=%d)", tag, len(text))
        planner._finalize_ai_response(text)

    def on_error(self, message: str) -> None:
        planner = self._planner
        if planner._llm_active_tag == "warmup":
            logger.warning("LLM warm-up error: %s", message)
            if planner._llm_warmup_silent:
                planner._ai_runtime_status = ""
                self.refresh_warm_indicator()
            else:
                self.set_status("Warm-up failed", tone="warning")
            return
        logger.warning("LLM error: %s", message)
        planner._fail_ai_response(message)

    def on_worker_finished(self) -> None:
        planner = self._planner
        active_tag = planner._llm_active_tag
        if active_tag != "warmup":
            self.set_status("Ready", tone="info")
        idx = planner._ai_stream_message_index
        if idx is not None and 0 <= idx < len(planner._ai_messages):
            if not planner._ai_messages[idx].get("text", "").strip():
                planner._ai_messages.pop(idx)
                planner._render_ai_messages()
                planner._persist_ai_messages_to_storage()
        planner._ai_stream_message_index = None
        sender = self.sender()
        if sender is planner._llm_worker:
            planner._llm_worker = None
        if active_tag == "warmup":
            planner._llm_warmup_silent = False
        planner._llm_active_requested_class = ""
        planner._llm_active_primary_target = None
        planner._llm_active_tag = ""
        self.refresh_warm_indicator()


__all__ = ["AIPanelCoordinator"]
