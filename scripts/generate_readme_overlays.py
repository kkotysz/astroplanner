#!/usr/bin/env python3
"""Generate numbered README overlays for screenshots."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


OVERLAY_SPECS: dict[str, dict[str, Any]] = {
    "dashboard": {
        "base": "dashboard-main.png",
        "overlay": "dashboard-main-overlay.png",
        "callouts": [
            {"id": 1, "anchor": (0.36, 0.04), "label": (0.05, 0.14)},
            {"id": 2, "anchor": (0.23, 0.36), "label": (0.04, 0.42)},
            {"id": 3, "anchor": (0.80, 0.23), "label": (0.93, 0.12)},
            {"id": 4, "anchor": (0.81, 0.53), "label": (0.93, 0.45)},
            {"id": 5, "anchor": (0.34, 0.79), "label": (0.08, 0.92)},
        ],
    },
    "suggest": {
        "base": "suggest-targets.png",
        "overlay": "suggest-targets-overlay.png",
        "callouts": [
            {"id": 1, "anchor": (0.18, 0.03), "label": (0.04, 0.10)},
            {"id": 2, "anchor": (0.28, 0.07), "label": (0.04, 0.22)},
            {"id": 3, "anchor": (0.33, 0.33), "label": (0.04, 0.43)},
            {"id": 4, "anchor": (0.65, 0.26), "label": (0.91, 0.24)},
            {"id": 5, "anchor": (0.93, 0.06), "label": (0.91, 0.10)},
        ],
    },
    "ai": {
        "base": "ai-assistant.png",
        "overlay": "ai-assistant-overlay.png",
        "callouts": [
            {"id": 1, "anchor": (0.10, 0.16), "label": (0.03, 0.08)},
            {"id": 2, "anchor": (0.48, 0.21), "label": (0.42, 0.06)},
            {"id": 3, "anchor": (0.61, 0.96), "label": (0.84, 0.86)},
            {"id": 4, "anchor": (0.08, 0.52), "label": (0.22, 0.58)},
        ],
    },
}


def _load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "DejaVuSans-Bold.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _fraction_to_point(size: tuple[int, int], fraction: tuple[float, float]) -> tuple[int, int]:
    width, height = size
    return (
        int(round(width * float(fraction[0]))),
        int(round(height * float(fraction[1]))),
    )


def _draw_callout(
    draw: ImageDraw.ImageDraw,
    image_size: tuple[int, int],
    callout_id: int,
    anchor_frac: tuple[float, float],
    label_frac: tuple[float, float],
    font: ImageFont.ImageFont,
) -> None:
    width, height = image_size
    anchor = _fraction_to_point(image_size, anchor_frac)
    label = _fraction_to_point(image_size, label_frac)

    line_width = max(3, int(round(min(width, height) * 0.003)))
    anchor_radius = max(6, int(round(min(width, height) * 0.010)))
    bubble_radius = max(14, int(round(min(width, height) * 0.022)))

    line_color = (86, 244, 255, 255)
    anchor_fill = (255, 91, 246, 255)
    anchor_outline = (195, 247, 255, 255)
    bubble_fill = (255, 91, 246, 235)
    bubble_outline = (215, 248, 255, 255)
    text_color = (13, 22, 40, 255)

    draw.line([label, anchor], fill=line_color, width=line_width)
    draw.ellipse(
        (
            anchor[0] - anchor_radius,
            anchor[1] - anchor_radius,
            anchor[0] + anchor_radius,
            anchor[1] + anchor_radius,
        ),
        fill=anchor_fill,
        outline=anchor_outline,
        width=max(2, line_width - 1),
    )
    draw.ellipse(
        (
            label[0] - bubble_radius,
            label[1] - bubble_radius,
            label[0] + bubble_radius,
            label[1] + bubble_radius,
        ),
        fill=bubble_fill,
        outline=bubble_outline,
        width=max(2, line_width - 1),
    )
    text = str(callout_id)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    text_x = int(round(label[0] - text_w / 2))
    text_y = int(round(label[1] - text_h / 2 - 1))
    draw.text((text_x, text_y), text, fill=text_color, font=font)


def _build_overlay(base_path: Path, overlay_path: Path, callouts: list[dict[str, Any]]) -> None:
    with Image.open(base_path) as base_img:
        image = base_img.convert("RGBA")
    shade = Image.new("RGBA", image.size, (6, 14, 28, 54))
    image.alpha_composite(shade)

    draw = ImageDraw.Draw(image)
    font_size = max(17, int(round(min(image.size) * 0.034)))
    font = _load_font(font_size)

    for callout in callouts:
        _draw_callout(
            draw=draw,
            image_size=image.size,
            callout_id=int(callout["id"]),
            anchor_frac=tuple(callout["anchor"]),
            label_frac=tuple(callout["label"]),
            font=font,
        )

    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(overlay_path)


def _normalize_views(values: list[str]) -> list[str]:
    if not values:
        return list(OVERLAY_SPECS.keys())
    normalized = [value.strip().lower() for value in values]
    if "all" in normalized:
        return list(OVERLAY_SPECS.keys())
    unknown = [value for value in normalized if value not in OVERLAY_SPECS]
    if unknown:
        raise ValueError(f"Unsupported overlay view(s): {', '.join(sorted(set(unknown)))}")
    deduped: list[str] = []
    seen: set[str] = set()
    for value in normalized:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate README screenshot overlays.")
    parser.add_argument(
        "--views",
        nargs="+",
        default=["all"],
        help="Overlay views: dashboard suggest ai (or all).",
    )
    parser.add_argument(
        "--screenshots-dir",
        default="docs/screenshots",
        help="Directory with base screenshots and output overlays.",
    )
    args = parser.parse_args()

    screenshots_dir = Path(args.screenshots_dir).resolve()
    views = _normalize_views(list(args.views))
    for view in views:
        spec = OVERLAY_SPECS[view]
        base_path = screenshots_dir / str(spec["base"])
        overlay_path = screenshots_dir / str(spec["overlay"])
        if not base_path.exists():
            raise FileNotFoundError(f"Base screenshot not found: {base_path}")
        _build_overlay(base_path=base_path, overlay_path=overlay_path, callouts=list(spec["callouts"]))
        print(f"overlay: {overlay_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
