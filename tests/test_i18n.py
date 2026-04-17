import ast
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from astroplanner.i18n import (
    _PATTERN_TRANSLATIONS,
    _SOURCE_TRANSLATIONS,
    resolve_language_choice,
    translate_text,
)


ROOT = Path(__file__).resolve().parents[1]
UI_SOURCE_FILES = (
    ROOT / "astro_planner.py",
    ROOT / "astroplanner" / "main_window.py",
    ROOT / "astroplanner" / "seestar.py",
)
UI_CALL_NAMES = {
    "setWindowTitle",
    "setPlaceholderText",
    "setText",
    "setToolTip",
    "addTab",
    "addRow",
    "addMenu",
    "setHorizontalHeaderLabels",
    "setStatusTip",
    "setTitle",
}
UI_CTOR_NAMES = {
    "QLabel",
    "QPushButton",
    "QCheckBox",
    "QRadioButton",
    "QAction",
    "QLineEdit",
    "QMenu",
    "QGroupBox",
}
INTENTIONAL_UI_LITERALS = {
    "-",
    "+",
    "−",
    "...",
    "0",
    "-24.59",
    "-70.19",
    "2800",
    "https://example.com/station.json",
}


def _static_str_value(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
                continue
            return None
        return "".join(parts)
    return None


def _extract_static_ui_strings() -> list[tuple[str, int, str]]:
    extracted: list[tuple[str, int, str]] = []
    for path in UI_SOURCE_FILES:
        tree = ast.parse(path.read_text())
        rel_path = str(path.relative_to(ROOT))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else None
            if name in UI_CALL_NAMES:
                for index, arg in enumerate(node.args):
                    if name == "addRow" and index != 0:
                        continue
                    if name == "addTab" and index != 1:
                        continue
                    if name == "addMenu" and index != 0:
                        continue
                    if name == "setHorizontalHeaderLabels" and isinstance(arg, (ast.List, ast.Tuple)):
                        for elt in arg.elts:
                            raw = _static_str_value(elt)
                            if raw is not None:
                                extracted.append((rel_path, node.lineno, raw))
                        continue
                    if index != 0 and name not in {"addRow", "addTab", "addMenu"}:
                        continue
                    raw = _static_str_value(arg)
                    if raw is not None:
                        extracted.append((rel_path, node.lineno, raw))
            elif name in UI_CTOR_NAMES and node.args:
                raw = _static_str_value(node.args[0])
                if raw is not None:
                    extracted.append((rel_path, node.lineno, raw))
    return extracted


def test_resolve_language_choice_uses_supported_system_locale() -> None:
    assert resolve_language_choice("system", system_locale="pl_PL") == "pl"
    assert resolve_language_choice("", system_locale="de-DE") == "de"


def test_resolve_language_choice_falls_back_to_english() -> None:
    assert resolve_language_choice("system", system_locale="xx_YY") == "en"
    assert resolve_language_choice("unknown") == "en"


def test_translate_text_uses_exact_catalog_match() -> None:
    assert translate_text("Weather", "de") == "Wetter"
    assert translate_text("General Settings", "it") == "Impostazioni generali"
    assert translate_text("Apply", "es") == "Aplicar"
    assert translate_text("Cancel", "pl") == "Anuluj"
    assert translate_text("Observation History", "fr") == "Historique d'observation"
    assert translate_text("Today", "cs") == "Dnes"
    assert translate_text("AI Settings", "pt") == "Configurações de IA"


def test_translate_text_uses_dynamic_patterns() -> None:
    assert translate_text("Filters: none | Visible: 3/7", "pl") == "Filtry: brak | Widoczne: 3/7"
    assert translate_text("● warm llama3", "it") == "● caldo llama3"
    assert translate_text("Source EarthEnv", "de") == "Quelle EarthEnv"
    assert translate_text("Last calc: -", "es") == "Último cálculo: -"


def test_translate_text_keeps_unknown_strings() -> None:
    raw = "Unlisted value"
    assert translate_text(raw, "fr") == raw


def test_static_ui_strings_are_translated_or_allowlisted() -> None:
    missing: list[str] = []
    seen: set[str] = set()
    for rel_path, line, raw in _extract_static_ui_strings():
        key = raw.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        if key in INTENTIONAL_UI_LITERALS:
            continue
        if key in _SOURCE_TRANSLATIONS:
            continue
        if any(pattern.match(key) for pattern, _translations in _PATTERN_TRANSLATIONS):
            continue
        missing.append(f"{rel_path}:{line}: {key}")
    assert not missing, "Missing i18n coverage:\n" + "\n".join(missing)
