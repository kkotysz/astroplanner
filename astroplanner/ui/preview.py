from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QLabel, QSizePolicy, QStackedLayout, QVBoxLayout, QWidget

from astroplanner.ui.common import SkeletonShimmerWidget
from astroplanner.ui.widgets import CoverImageLabel


def build_cutout_loading_placeholder(parent: QWidget, title: str, message: str) -> QWidget:
    """Build the shared loading placeholder used by Aladin/finder previews."""
    widget = QWidget(parent)
    widget.setAttribute(Qt.WA_StyledBackground, True)
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    card = QFrame(widget)
    card.setObjectName("CutoutImage")
    card_layout = QVBoxLayout(card)
    card_layout.setContentsMargins(14, 14, 14, 14)
    card_layout.setSpacing(10)
    title_label = QLabel(title, card)
    title_label.setObjectName("SectionTitle")
    title_label.setAlignment(Qt.AlignCenter)
    skeleton = SkeletonShimmerWidget("image", card)
    skeleton.setMinimumHeight(220)
    hint_label = QLabel(message, card)
    hint_label.setObjectName("SectionHint")
    hint_label.setWordWrap(True)
    hint_label.setAlignment(Qt.AlignCenter)
    card_layout.addWidget(title_label, 0, Qt.AlignHCenter)
    card_layout.addWidget(skeleton, 1)
    card_layout.addWidget(hint_label, 0, Qt.AlignHCenter)
    layout.addWidget(card, 1)
    widget._loading_hint_label = hint_label  # type: ignore[attr-defined]
    return widget


def create_cutout_image_stack(
    owner: object,
    parent: QWidget,
    *,
    kind: str,
    title: str,
) -> tuple[QWidget, CoverImageLabel]:
    host = QWidget(parent)
    host.setAttribute(Qt.WA_StyledBackground, True)
    host.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    stack = QStackedLayout(host)
    stack.setContentsMargins(0, 0, 0, 0)
    placeholder_parent = owner if isinstance(owner, QWidget) else parent
    placeholder = build_cutout_loading_placeholder(placeholder_parent, title, f"Loading {title.lower()}…")
    image_label = CoverImageLabel("Select a target", host)
    image_label.setObjectName("CutoutImage")
    image_label.setProperty("cutout_image", True)
    image_label.setAlignment(Qt.AlignCenter)
    image_label.setWordWrap(True)
    image_label.setScaledContents(False)
    image_label.setMinimumSize(1, 1)
    image_label.setMaximumSize(16777215, 16777215)
    image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    stack.addWidget(placeholder)
    stack.addWidget(image_label)
    stack.setCurrentWidget(image_label)
    owner._cutout_image_stacks[kind] = stack  # type: ignore[attr-defined]
    owner._cutout_image_placeholders[kind] = placeholder  # type: ignore[attr-defined]
    owner._cutout_image_labels[kind] = image_label  # type: ignore[attr-defined]
    return host, image_label


def set_cutout_image_loading(owner: object, kind: str, message: str, *, visible: bool = True) -> None:
    stack = owner._cutout_image_stacks.get(kind)  # type: ignore[attr-defined]
    placeholder = owner._cutout_image_placeholders.get(kind)  # type: ignore[attr-defined]
    image_label = owner._cutout_image_labels.get(kind)  # type: ignore[attr-defined]
    if stack is None or placeholder is None or image_label is None:
        return
    hint_label = getattr(placeholder, "_loading_hint_label", None)
    if isinstance(hint_label, QLabel):
        hint_label.setText(message)
    if visible:
        stack.setCurrentWidget(placeholder)
    else:
        stack.setCurrentWidget(image_label)


__all__ = [
    "build_cutout_loading_placeholder",
    "create_cutout_image_stack",
    "set_cutout_image_loading",
]
