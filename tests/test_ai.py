from __future__ import annotations

from astroplanner.ai import (
    _llm_chat_completions_endpoint,
    _llm_model_endpoints,
    _llm_models_from_ollama_payload,
    _llm_models_from_openai_payload,
    _load_knowledge_note,
    _looks_like_object_scoped_query,
    _looks_like_observing_guidance_query,
    _question_bhtom_type_markers,
    _requested_object_class_marker,
    _type_matches_requested_class,
)


def test_load_knowledge_note_parses_summary_tags_and_sections(tmp_path) -> None:
    note_path = tmp_path / "supernova-practical.md"
    note_path.write_text(
        "\n".join(
            [
                "# Supernova Practical",
                "Summary: Short observing guidance.",
                "Tags: supernova, #sn, supernova",
                "",
                "## Key heuristics",
                "- Observe near transit",
                "1. Keep a low airmass",
                "",
                "## Caveats",
                "**Moon**: contrast drops fast",
            ]
        ),
        encoding="utf-8",
    )

    note = _load_knowledge_note(note_path)

    assert note is not None
    assert note.title == "Supernova Practical"
    assert note.summary == "Short observing guidance."
    assert note.tags == ("supernova", "sn")
    assert note.sections["key heuristics"] == [
        "Observe near transit",
        "Keep a low airmass",
    ]
    assert note.sections["caveats"] == ["Moon: contrast drops fast"]


def test_query_helpers_detect_bhtom_markers_and_object_classes() -> None:
    markers = _question_bhtom_type_markers("Any good SN or quasar targets tonight?")

    assert markers == ("supernova", "sn", "quasar", "qso")
    assert _requested_object_class_marker("Is this a quasar?") == "qso"
    assert _type_matches_requested_class("QSO candidate", "agn") is True
    assert _type_matches_requested_class("spiral galaxy", "supernova") is False


def test_object_scoped_and_guidance_helpers_keep_polish_phrases() -> None:
    assert _looks_like_object_scoped_query("Jak obserwować ten obiekt?") is True
    assert _looks_like_observing_guidance_query("Jak powinienem go obserwować?") is True


def test_llm_payload_and_endpoint_helpers_cover_openai_and_ollama_shapes() -> None:
    assert _llm_models_from_openai_payload({"data": [{"id": "gemma4:e4b"}, {"id": "llama3"}]}) == [
        "gemma4:e4b",
        "llama3",
    ]
    assert _llm_models_from_ollama_payload({"models": [{"name": "gemma4:e4b"}]}) == ["gemma4:e4b"]
    assert _llm_model_endpoints("http://localhost:11434") == [
        ("http://localhost:11434/api/tags", "Ollama"),
        ("http://localhost:11434/v1/models", "OpenAI-compatible"),
    ]
    assert _llm_chat_completions_endpoint("http://localhost:11434") == (
        "http://localhost:11434/v1/chat/completions"
    )
    assert _llm_chat_completions_endpoint("http://localhost:1337/v1") == (
        "http://localhost:1337/v1/chat/completions"
    )
