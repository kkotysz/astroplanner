from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from PySide6.QtCore import QThread, Signal

from .models import Target

KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "knowledge"


@dataclass(frozen=True)
class KnowledgeNote:
    path: Path
    title: str
    summary: str
    tags: tuple[str, ...]
    sections: dict[str, list[str]]


@dataclass(frozen=True)
class ClassQuerySpec:
    requested_class: str
    count: int
    wants_list: bool
    wants_more: bool
    choice_followup: bool
    prefer_bhtom_only: bool = False
    exclude_observed: bool = False
    exclude_previous_results: bool = False
    prefer_brighter: bool = False


@dataclass(frozen=True)
class ObjectQuerySpec:
    target: Optional[Target]
    object_scoped: bool
    wants_guidance: bool
    wants_fact: bool
    wants_selected_llm: bool
    blocked_no_selection: bool


@dataclass(frozen=True)
class CompareQuerySpec:
    targets: tuple[Target, ...]
    criterion: str
    return_best_only: bool
    include_reason: bool


@dataclass(frozen=True)
class AIIntent:
    kind: str
    question: str
    label: str
    target: Optional[Target] = None
    requested_class: str = ""
    class_query: Optional[ClassQuerySpec] = None
    object_query: Optional[ObjectQuerySpec] = None
    compare_query: Optional[CompareQuerySpec] = None
    local_text: str = ""
    suggested_targets: tuple[Target, ...] = ()
    action_targets: tuple[Target, ...] = ()


def _normalize_catalog_display_name(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return " ".join(text.split())


def _normalize_knowledge_tag(value: object) -> str:
    text = _normalize_catalog_display_name(value).lower()
    if not text:
        return ""
    text = text.replace("_", "-").replace("/", "-")
    text = re.sub(r"[^a-z0-9+\- ]+", " ", text)
    text = "-".join(part for part in text.split() if part)
    return text


def _mentions_supernova_term(normalized: str) -> bool:
    text = str(normalized or "").lower()
    if not text:
        return False
    if "supernova" in text or "super nowa" in text or "supernow" in text:
        return True
    return bool(re.search(r"(?<![a-z0-9])sn(?![a-z0-9])", text))


def _clean_markdown_line(line: str) -> str:
    text = str(line or "").strip()
    if not text:
        return ""
    text = re.sub(r"^[\-\*\u2022]\s+", "", text)
    text = re.sub(r"^\d+\.\s+", "", text)
    text = re.sub(r"^\*\*(.+?)\*\*:\s*", r"\1: ", text)
    text = text.replace("`", "")
    return text.strip()


def _load_knowledge_note(path: Path) -> Optional[KnowledgeNote]:
    try:
        raw_text = path.read_text(encoding="utf-8")
    except Exception:
        return None

    title = ""
    summary = ""
    tags: list[str] = []
    sections: dict[str, list[str]] = {}
    current_section = ""

    for raw_line in raw_text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("# ") and not title:
            title = stripped[2:].strip()
            continue
        if stripped.startswith("## "):
            current_section = stripped[3:].strip().lower()
            sections.setdefault(current_section, [])
            continue

        summary_match = re.match(r"^\*?\*?summary\*?\*?:\s*(.+)$", stripped, flags=re.IGNORECASE)
        if summary_match and not summary:
            summary = summary_match.group(1).strip()
            continue

        tags_match = re.match(r"^\*?\*?tags\*?\*?:\s*(.+)$", stripped, flags=re.IGNORECASE)
        if tags_match and not tags:
            raw_tags = tags_match.group(1)
            parsed = re.split(r"[,\s]+", raw_tags)
            tags = [
                normalized
                for token in parsed
                if (normalized := _normalize_knowledge_tag(str(token).lstrip("#")))
            ]
            continue

        if current_section:
            cleaned = _clean_markdown_line(stripped)
            if cleaned:
                sections.setdefault(current_section, []).append(cleaned)

    if not title:
        title = path.stem.replace("-", " ").title()
    deduped_tags: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        if tag in seen:
            continue
        seen.add(tag)
        deduped_tags.append(tag)
    return KnowledgeNote(
        path=path,
        title=title,
        summary=summary,
        tags=tuple(deduped_tags),
        sections=sections,
    )


def _question_bhtom_type_markers(question: str) -> tuple[str, ...]:
    normalized = _normalize_catalog_display_name(question).lower()
    if not normalized:
        return ()

    markers: list[str] = []

    def _contains_token(token: str) -> bool:
        return bool(re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", normalized))

    if _mentions_supernova_term(normalized):
        markers.extend(["supernova", "sn"])
    if "nova" in normalized:
        markers.append("nova")
    if any(token in normalized for token in ("quasar", "qso")) or _contains_token("qso"):
        markers.extend(["quasar", "qso"])
    if _contains_token("agn") or "seyfert" in normalized:
        markers.extend(["agn", "seyfert"])
    if _contains_token("tde") or "tidal disruption" in normalized:
        markers.append("tde")
    if _contains_token("cv") or "cataclysmic" in normalized:
        markers.extend(["cv", "cataclysmic"])
    if "blazar" in normalized:
        markers.append("blazar")

    deduped: list[str] = []
    seen: set[str] = set()
    for marker in markers:
        if marker in seen:
            continue
        seen.add(marker)
        deduped.append(marker)
    return tuple(deduped)


def _type_label_class_family(type_label: str) -> str:
    normalized = _normalize_catalog_display_name(type_label).lower()
    if not normalized:
        return ""

    def _contains_token(token: str) -> bool:
        return bool(re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", normalized))

    if any(token in normalized for token in ("quasar", "seyfert", "blazar")) or _contains_token("qso") or _contains_token("agn"):
        return "AGN"
    if _mentions_supernova_term(normalized):
        return "Supernova"
    if _contains_token("nova"):
        return "Nova"
    if _contains_token("xrb") or "x ray binary" in normalized:
        return "X-ray binary"
    if _contains_token("cv") or "cataclysmic" in normalized:
        return "Cataclysmic variable"
    if "galaxy" in normalized or "galakty" in normalized:
        return "Galaxy"
    if "*" in normalized or "star" in normalized:
        return "Star"
    return ""


def _requested_object_class_marker(question: str) -> str:
    normalized = _normalize_catalog_display_name(question).lower()
    if not normalized:
        return ""

    def _contains_token(token: str) -> bool:
        return bool(re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", normalized))

    if _contains_token("agn") or "active galactic nucleus" in normalized:
        return "agn"
    if _contains_token("qso") or "quasar" in normalized:
        return "qso"
    if "seyfert" in normalized:
        return "seyfert"
    if "blazar" in normalized:
        return "blazar"
    if _mentions_supernova_term(normalized):
        return "supernova"
    if _contains_token("nova"):
        return "nova"
    if _contains_token("xrb") or "x ray binary" in normalized:
        return "xrb"
    if _contains_token("cv") or "cataclysmic" in normalized:
        return "cv"
    if "galaxy" in normalized:
        return "galaxy"
    if _contains_token("star") or "gwiazd" in normalized:
        return "star"
    return ""


def _question_action_flags(normalized: str, groups: dict[str, tuple[str, ...]]) -> dict[str, bool]:
    if not normalized:
        return {name: False for name in groups}
    return {
        name: any(marker in normalized for marker in markers)
        for name, markers in groups.items()
    }


def _type_matches_requested_class(type_label: str, requested_marker: str) -> bool:
    normalized = _normalize_catalog_display_name(type_label).lower()
    family = _type_label_class_family(type_label)
    if not requested_marker:
        return False
    if requested_marker == "agn":
        return family == "AGN"
    if requested_marker == "supernova":
        return family == "Supernova"
    if requested_marker == "nova":
        return family == "Nova"
    if requested_marker == "xrb":
        return family == "X-ray binary"
    if requested_marker == "cv":
        return family == "Cataclysmic variable"
    if requested_marker == "galaxy":
        return family == "Galaxy"
    if requested_marker == "star":
        return family == "Star"
    if requested_marker == "qso":
        return "qso" in normalized or "quasar" in normalized
    if requested_marker == "seyfert":
        return "seyfert" in normalized
    if requested_marker == "blazar":
        return "blazar" in normalized
    return False


def _requested_marker_family(requested_marker: str) -> str:
    marker = _normalize_knowledge_tag(requested_marker)
    if marker in {"agn", "qso", "seyfert", "blazar"}:
        return "AGN"
    if marker == "supernova":
        return "Supernova"
    if marker == "nova":
        return "Nova"
    if marker == "xrb":
        return "X-ray binary"
    if marker == "cv":
        return "Cataclysmic variable"
    if marker == "galaxy":
        return "Galaxy"
    if marker == "star":
        return "Star"
    return ""


def _knowledge_note_family(note: KnowledgeNote) -> str:
    note_tags = set(note.tags)
    if {"agn", "qso", "seyfert", "blazar"} & note_tags:
        return "AGN"
    if {"supernova", "sn"} & note_tags:
        return "Supernova"
    if "nova" in note_tags:
        return "Nova"
    if "xrb" in note_tags:
        return "X-ray binary"
    if {"cv", "cataclysmic-variable"} & note_tags:
        return "Cataclysmic variable"
    if "galaxy" in note_tags:
        return "Galaxy"
    if {"star", "variable-star"} & note_tags:
        return "Star"
    return ""


def _format_knowledge_note_snippet(note: KnowledgeNote) -> str:
    lines: list[str] = [f"- {note.title}"]
    if note.summary:
        lines.append(f"  Summary: {note.summary}")

    heuristics = list(note.sections.get("key heuristics", []))
    if heuristics:
        lines.append("  Key heuristics:")
        for item in heuristics[:3]:
            lines.append(f"    - {item}")

    caveats = list(note.sections.get("caveats", []))
    if caveats:
        lines.append("  Caveats:")
        for item in caveats[:2]:
            lines.append(f"    - {item}")
    return "\n".join(lines)


def _looks_like_observing_guidance_query(text: str) -> bool:
    normalized = _normalize_catalog_display_name(text).lower()
    if not normalized:
        return False
    phrases = (
        "how to observe",
        "how should i observe",
        "best way to observe",
        "observe it",
        "how do i observe",
        "jak obserwowac",
        "jak obserwować",
        "jak go obserwowac",
        "jak go obserwować",
        "jak powinienem obserwowac",
        "jak powinienem obserwować",
        "jak powinienem go obserwowac",
        "jak powinienem go obserwować",
        "jak obserwowac ten",
        "jak obserwować ten",
    )
    return any(phrase in normalized for phrase in phrases)


def _requested_marker_label(requested_marker: str) -> str:
    marker = _normalize_knowledge_tag(requested_marker)
    mapping = {
        "agn": "AGN",
        "qso": "QSO",
        "seyfert": "Seyfert",
        "blazar": "blazar",
        "supernova": "SN",
        "nova": "nova",
        "xrb": "X-ray binary",
        "cv": "CV",
        "galaxy": "galaxy",
        "star": "star",
    }
    return mapping.get(marker, requested_marker or "target")


def _looks_like_object_scoped_query(text: str) -> bool:
    normalized = _normalize_catalog_display_name(text).lower()
    if not normalized:
        return False
    explicit_phrases = (
        "this object",
        "this target",
        "selected object",
        "selected target",
        "details of this object",
        "details of the object",
        "describe this object",
        "tell me about this object",
        "ten obiekt",
        "tego obiektu",
        "wybrany obiekt",
        "wybranego obiektu",
        "tego targetu",
        "tego celu",
        "opisz ten obiekt",
        "szczegoly tego obiektu",
        "szczegóły tego obiektu",
        "jak powinienem go obserwowac",
        "jak powinienem go obserwować",
        "jak go obserwowac",
        "jak go obserwować",
        "jak obserwowac ten obiekt",
        "jak obserwować ten obiekt",
        "jak obserwowac ten target",
        "jak obserwować ten target",
        "jak obserwowac ten cel",
        "jak obserwować ten cel",
    )
    return any(phrase in normalized for phrase in explicit_phrases)


def _looks_like_object_class_query(text: str) -> bool:
    normalized = _normalize_catalog_display_name(text).lower()
    if not normalized:
        return False
    class_markers = (
        "what type",
        "type of object",
        "what kind",
        "classified as",
        "classification",
        "class of object",
        "what is",
        "czym jest",
        "co to jest",
        "co to za obiekt",
        "jaki typ",
        "jaki to typ",
        "jaki to obiekt",
        "czy to",
        "czy ten",
        "czy ta",
        "czy jest",
        "is it",
        "is this",
        "is that",
        "is the",
    )
    if any(marker in normalized for marker in class_markers):
        return True
    return normalized.startswith("is ")


def _truncate_ai_memory_text(text: str, *, max_chars: int = 280) -> str:
    normalized = " ".join(str(text or "").split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."


class LLMConfig:
    """Configuration for a local OpenAI-compatible inference server."""

    DEFAULT_URL = "http://localhost:11434"
    DEFAULT_MODEL = "gemma4:e4b"
    DEFAULT_TIMEOUT_S = 180
    DEFAULT_MAX_TOKENS = 192
    DEFAULT_WARMUP_MAX_TOKENS = 8
    DEFAULT_TEMPERATURE = 0.2
    DEFAULT_ENABLE_THINKING = False
    DEFAULT_ENABLE_CHAT_MEMORY = False
    DEFAULT_CHAT_FONT_PT = 12
    DEFAULT_CHAT_SPACING = "comfortable"
    DEFAULT_CHAT_TINT_STRENGTH = "medium"
    DEFAULT_CHAT_WIDTH = "normal"
    DEFAULT_STATUS_ERROR_CLEAR_S = 8

    def __init__(
        self,
        url: str = DEFAULT_URL,
        model: str = DEFAULT_MODEL,
        timeout_s: int = DEFAULT_TIMEOUT_S,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        enable_thinking: bool = DEFAULT_ENABLE_THINKING,
        enable_chat_memory: bool = DEFAULT_ENABLE_CHAT_MEMORY,
    ) -> None:
        normalized_url = str(url or self.DEFAULT_URL).strip().rstrip("/")
        normalized_model = str(model or self.DEFAULT_MODEL).strip()
        self.url = normalized_url or self.DEFAULT_URL
        self.model = normalized_model or self.DEFAULT_MODEL
        self.timeout_s = max(5, int(timeout_s))
        self.max_tokens = max(32, int(max_tokens))
        self.enable_thinking = bool(enable_thinking)
        self.enable_chat_memory = bool(enable_chat_memory)


def _llm_models_from_openai_payload(payload: object) -> list[str]:
    if not isinstance(payload, dict):
        return []
    data = payload.get("data")
    if not isinstance(data, list):
        return []
    models: list[str] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        if isinstance(model_id, str) and model_id.strip():
            models.append(model_id.strip())
    return models


def _llm_models_from_ollama_payload(payload: object) -> list[str]:
    if not isinstance(payload, dict):
        return []
    data = payload.get("models")
    if not isinstance(data, list):
        return []
    models: list[str] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if isinstance(name, str) and name.strip():
            models.append(name.strip())
    return models


def _llm_model_endpoints(base_url: str) -> list[tuple[str, str]]:
    normalized = str(base_url or "").strip().rstrip("/")
    if not normalized:
        return []
    candidates: list[tuple[str, str]] = []

    def add(endpoint: str, label: str) -> None:
        if endpoint and (endpoint, label) not in candidates:
            candidates.append((endpoint, label))

    add(f"{normalized}/api/tags", "Ollama")

    if normalized.endswith("/v1/models"):
        add(normalized, "OpenAI-compatible")
    elif normalized.endswith("/v1"):
        add(f"{normalized}/models", "OpenAI-compatible")
    else:
        add(f"{normalized}/v1/models", "OpenAI-compatible")

    return candidates


def _llm_chat_completions_endpoint(base_url: str) -> str:
    normalized = str(base_url or "").strip().rstrip("/")
    if not normalized:
        return ""
    if normalized.endswith("/v1/chat/completions") or normalized.endswith("/chat/completions"):
        return normalized
    if normalized.endswith("/v1"):
        return f"{normalized}/chat/completions"
    return f"{normalized}/v1/chat/completions"


def _fetch_llm_models(base_url: str, *, timeout_s: int = 6) -> tuple[list[str], str, str]:
    for endpoint, backend in _llm_model_endpoints(base_url):
        try:
            req = Request(endpoint, headers={"User-Agent": "AstroPlanner"})
            with urlopen(req, timeout=timeout_s) as resp:
                raw = resp.read()
            payload = json.loads(raw.decode("utf-8", errors="replace"))
        except Exception:
            continue
        models = (
            _llm_models_from_ollama_payload(payload)
            if "ollama" in backend.lower()
            else _llm_models_from_openai_payload(payload)
        )
        if not models and "ollama" not in backend.lower():
            models = _llm_models_from_ollama_payload(payload)
        if models:
            return sorted(set(models)), backend, ""
    return [], "", "No models endpoint responded."


class LLMModelDiscoveryWorker(QThread):
    modelsReady = Signal(list, str)
    failed = Signal(str)

    def __init__(self, url: str, *, timeout_s: int = 6, parent=None) -> None:
        super().__init__(parent)
        self.url = str(url or "").strip()
        self.timeout_s = max(2, int(timeout_s))

    def run(self) -> None:
        if not self.url:
            self.failed.emit("Set the LLM server URL first.")
            return
        models, backend, error = _fetch_llm_models(self.url, timeout_s=self.timeout_s)
        if models:
            self.modelsReady.emit(models, backend or "Detected")
            return
        self.failed.emit(error or "No models found.")


AI_CHAT_SPACING_CHOICES = [
    ("compact", "Compact"),
    ("comfortable", "Comfortable"),
]

AI_CHAT_TINT_CHOICES = [
    ("low", "Low"),
    ("medium", "Medium"),
    ("high", "High"),
]

AI_CHAT_WIDTH_CHOICES = [
    ("narrow", "Narrow"),
    ("normal", "Normal"),
    ("wide", "Wide"),
]


class LLMWorker(QThread):
    """Send a single prompt to a local OpenAI-compatible server."""

    TRUNCATION_NOTE = (
        "[Response truncated by token limit. Increase `LLM max tokens` in Settings -> General Settings -> AI if needed.]"
    )

    responseReady = Signal(str, str)
    responseChunk = Signal(str, str)
    errorOccurred = Signal(str)

    def __init__(
        self,
        config: LLMConfig,
        prompt: str,
        system_prompt: str = "",
        tag: str = "chat",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName(self.__class__.__name__)
        self.config = config
        self.prompt = prompt
        self.system_prompt = system_prompt
        self.tag = tag

    @staticmethod
    def _extract_content(data: object) -> str:
        if not isinstance(data, dict):
            return ""
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict):
                content = message.get("content", "")
                if isinstance(content, str):
                    return content.strip()
                if isinstance(content, list):
                    parts: list[str] = []
                    for item in content:
                        if not isinstance(item, dict):
                            continue
                        if item.get("type") == "text":
                            txt = item.get("text", "")
                            if isinstance(txt, str) and txt.strip():
                                parts.append(txt.strip())
                    if parts:
                        return "\n".join(parts)
            text = first.get("text")
            if isinstance(text, str):
                return text.strip()
        return ""

    @staticmethod
    def _extract_delta_content(data: object) -> str:
        if not isinstance(data, dict):
            return ""
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""
        first = choices[0]
        if not isinstance(first, dict):
            return ""
        delta = first.get("delta")
        if not isinstance(delta, dict):
            return ""
        content = delta.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text":
                    txt = item.get("text", "")
                    if isinstance(txt, str) and txt:
                        parts.append(txt)
            if parts:
                return "".join(parts)
        return ""

    @staticmethod
    def _extract_finish_reason(data: object) -> str:
        if not isinstance(data, dict):
            return ""
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""
        first = choices[0]
        if not isinstance(first, dict):
            return ""
        finish_reason = first.get("finish_reason")
        return finish_reason if isinstance(finish_reason, str) else ""

    @classmethod
    def _append_truncation_note(cls, text: str) -> str:
        normalized = str(text).rstrip()
        if not normalized:
            return cls.TRUNCATION_NOTE
        if cls.TRUNCATION_NOTE in normalized:
            return normalized
        return f"{normalized}\n\n{cls.TRUNCATION_NOTE}"

    def _consume_sse_stream(self, response) -> tuple[str, str]:
        chunks: list[str] = []
        event_lines: list[str] = []
        finish_reason = ""

        def _flush_event() -> bool:
            nonlocal finish_reason
            if not event_lines:
                return False
            payload_text = "\n".join(event_lines).strip()
            event_lines.clear()
            if not payload_text:
                return False
            if payload_text == "[DONE]":
                return True
            try:
                decoded = json.loads(payload_text)
            except json.JSONDecodeError:
                return False
            extracted_finish_reason = self._extract_finish_reason(decoded)
            if extracted_finish_reason:
                finish_reason = extracted_finish_reason
            delta = self._extract_delta_content(decoded)
            if delta:
                chunks.append(delta)
                self.responseChunk.emit(self.tag, delta)
                return False
            if not chunks:
                fallback = self._extract_content(decoded)
                if fallback:
                    chunks.append(fallback)
                    self.responseChunk.emit(self.tag, fallback)
            return False

        for raw_line in response:
            if self.isInterruptionRequested():
                break
            line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
            if not line:
                if _flush_event():
                    break
                continue
            if line.startswith("data:"):
                event_lines.append(line[5:].lstrip())

        if not self.isInterruptionRequested() and event_lines:
            _flush_event()
        return "".join(chunks), finish_reason

    def _request_generation_params(self) -> dict[str, object]:
        if self.tag == "warmup":
            max_tokens = int(self.config.DEFAULT_WARMUP_MAX_TOKENS)
        else:
            max_tokens = int(self.config.max_tokens)
        params: dict[str, object] = {
            "max_tokens": max_tokens,
            "temperature": float(self.config.DEFAULT_TEMPERATURE),
            "stream": True,
        }
        if self._should_manage_gemma4_thinking_via_template():
            params["chat_template_kwargs"] = {"enable_thinking": bool(self.config.enable_thinking)}
        elif not self.config.enable_thinking:
            params["reasoning_effort"] = "none"
        return params

    def _is_docker_model_runner_backend(self) -> bool:
        try:
            parsed = urlparse(self.config.url)
        except Exception:
            return False
        host = (parsed.hostname or "").strip().lower()
        port = parsed.port
        path = (parsed.path or "").strip().lower()
        return (
            port == 12434
            or "/engines" in path
            or host == "model-runner.docker.internal"
        )

    def _should_manage_gemma4_thinking_via_template(self) -> bool:
        model_name = str(self.config.model or "").strip().lower()
        return self._is_docker_model_runner_backend() and "gemma4" in model_name

    def run(self) -> None:
        if self.isInterruptionRequested():
            return
        messages: list[dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": self.prompt})
        generation_params = self._request_generation_params()
        payload = json.dumps(
            {
                "model": self.config.model,
                "messages": messages,
                **generation_params,
            }
        ).encode("utf-8")
        request = Request(
            _llm_chat_completions_endpoint(self.config.url),
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.config.timeout_s) as response:
                content_type = response.headers.get("Content-Type", "")
                if "text/event-stream" in content_type:
                    text, finish_reason = self._consume_sse_stream(response)
                else:
                    body = response.read().decode("utf-8", errors="replace")
                    decoded = json.loads(body)
                    finish_reason = self._extract_finish_reason(decoded)
                    text = self._extract_content(decoded)
            if finish_reason == "length" and self.tag != "warmup":
                text = self._append_truncation_note(text)
            if not text:
                raise RuntimeError("LLM response does not contain message content.")
            self.responseReady.emit(self.tag, text)
        except HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="ignore").strip()
            except Exception:
                detail = ""
            if detail:
                self.errorOccurred.emit(f"LLM HTTP {exc.code}: {detail[:280]}")
            else:
                self.errorOccurred.emit(f"LLM HTTP {exc.code}.")
        except URLError as exc:
            reason = getattr(exc, "reason", str(exc))
            self.errorOccurred.emit(
                f"Cannot reach LLM server at {self.config.url}: {reason}\n"
                "Start Jan, Ollama, Docker Model Runner, or another OpenAI-compatible backend, "
                "then verify URL/model in Settings -> General Settings."
            )
        except Exception as exc:
            self.errorOccurred.emit(f"LLM request failed ({type(exc).__name__}): {exc}")


__all__ = [
    "AIIntent",
    "AI_CHAT_SPACING_CHOICES",
    "AI_CHAT_TINT_CHOICES",
    "AI_CHAT_WIDTH_CHOICES",
    "ClassQuerySpec",
    "CompareQuerySpec",
    "KNOWLEDGE_DIR",
    "KnowledgeNote",
    "LLMConfig",
    "LLMModelDiscoveryWorker",
    "LLMWorker",
    "ObjectQuerySpec",
    "_clean_markdown_line",
    "_fetch_llm_models",
    "_format_knowledge_note_snippet",
    "_knowledge_note_family",
    "_llm_chat_completions_endpoint",
    "_llm_model_endpoints",
    "_llm_models_from_ollama_payload",
    "_llm_models_from_openai_payload",
    "_load_knowledge_note",
    "_looks_like_object_class_query",
    "_looks_like_object_scoped_query",
    "_looks_like_observing_guidance_query",
    "_mentions_supernova_term",
    "_normalize_knowledge_tag",
    "_question_action_flags",
    "_question_bhtom_type_markers",
    "_requested_marker_family",
    "_requested_marker_label",
    "_requested_object_class_marker",
    "_truncate_ai_memory_text",
    "_type_label_class_family",
    "_type_matches_requested_class",
]
