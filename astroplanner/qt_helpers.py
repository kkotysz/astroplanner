from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QTabWidget


def configure_tab_widget(tab_widget: QTabWidget, *, document_mode: bool = True) -> QTabWidget:
    """Normalize tab widgets so native tab-bar chrome doesn't leak through themed UI."""
    tab_widget.setDocumentMode(document_mode)
    try:
        tab_widget.setUsesScrollButtons(True)
    except Exception:
        pass
    try:
        tab_widget.setElideMode(Qt.TextElideMode.ElideNone)
    except Exception:
        pass
    bar = tab_widget.tabBar()
    if bar is not None:
        bar.setExpanding(False)
        bar.setElideMode(Qt.TextElideMode.ElideNone)
        if hasattr(bar, "setUsesScrollButtons"):
            try:
                bar.setUsesScrollButtons(True)
            except Exception:
                pass
        bar.setDrawBase(False)
    return tab_widget
