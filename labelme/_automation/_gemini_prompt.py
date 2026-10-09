from __future__ import annotations

import json
import re
from typing import Any
from typing import Final

import numpy as np
import skimage.draw
from numpy.typing import NDArray

from ._gemini_errors import GeminiResponseError

# Gemini normalizes every coordinate to this range, independent of image size.
_NORMALIZED_MAX: Final[float] = 1000.0
_BOX_LENGTH: Final[int] = 4  # [ymin, xmin, ymax, xmax]
_MIN_POLYGON_POINTS: Final[int] = 3
_POINT_DIMENSIONS: Final[int] = 2
_FENCE_RE: Final = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def build_prompt(*, texts: list[str], want_masks: bool) -> str:
    labels = ", ".join(json.dumps(text) for text in texts)
    if want_masks:
        fields = (
            '"box_2d" as [ymin, xmin, ymax, xmax], "mask" as a polygon '
            '[[x, y], ...] outlining the object, and "label"'
        )
    else:
        fields = '"box_2d" as [ymin, xmin, ymax, xmax] and "label"'
    return (
        f"Detect every instance of these objects in the image: {labels}. "
        f"Return a JSON list where each entry has {fields}. "
        "All coordinates are integers normalized to 0-1000 over the whole image. "
        "Each label must be exactly one of the requested names. "
        "Return an empty list if there are none."
    )


def build_response_schema(*, want_masks: bool) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "box_2d": {"type": "array", "items": {"type": "integer"}},
        "label": {"type": "string"},
    }
    required = ["box_2d", "label"]
    if want_masks:
        properties["mask"] = {
            "type": "array",
            "items": {"type": "array", "items": {"type": "integer"}},
        }
        required.append("mask")
    return {
        "type": "array",
        "items": {"type": "object", "properties": properties, "required": required},
    }


def parse_response(
    *,
    text: str,
    texts: list[str],
    image_size: tuple[int, int],
    want_masks: bool,
) -> tuple[
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.int32],
    list[NDArray[np.bool_]] | None,
]:
    width, height = image_size
    body = text.strip()
    if (match := _FENCE_RE.match(body)) is not None:
        body = match.group(1)
    try:
        entries = json.loads(body)
    except json.JSONDecodeError as e:
        raise GeminiResponseError("Gemini response is not valid JSON") from e
    if not isinstance(entries, list):
        raise GeminiResponseError("Gemini response is not a JSON list")

    label_indices = {name.strip().lower(): i for i, name in enumerate(texts)}
    boxes: list[tuple[int, int, int, int]] = []
    labels: list[int] = []
    masks: list[NDArray[np.bool_]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        label = entry.get("label")
        if not isinstance(label, str):
            continue
        label_index = label_indices.get(label.strip().lower())
        if label_index is None:
            continue
        box = _to_pixel_box(entry.get("box_2d"), width=width, height=height)
        if box is None:
            continue
        boxes.append(box)
        labels.append(label_index)
        if want_masks:
            masks.append(
                _rasterize_mask(entry.get("mask"), box=box, width=width, height=height)
            )

    return (
        np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
        np.ones((len(boxes),), dtype=np.float32),
        np.asarray(labels, dtype=np.int32),
        masks if want_masks else None,
    )


def _to_pixel_box(
    value: object, *, width: int, height: int
) -> tuple[int, int, int, int] | None:
    if not isinstance(value, list):
        return None
    numbers = [
        float(v)
        for v in value
        if isinstance(v, int | float) and not isinstance(v, bool)
    ]
    if len(value) != _BOX_LENGTH or len(numbers) != _BOX_LENGTH:
        return None
    ymin, xmin, ymax, xmax = numbers

    def scale(v: float, size: int) -> int:
        return int(np.clip(round(v / _NORMALIZED_MAX * size), 0, size - 1))

    x1, y1 = scale(xmin, width), scale(ymin, height)
    x2, y2 = scale(xmax, width), scale(ymax, height)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _rasterize_mask(
    value: object, *, box: tuple[int, int, int, int], width: int, height: int
) -> NDArray[np.bool_]:
    # The shape builders expect masks cropped to the integer box (inclusive).
    x1, y1, x2, y2 = box
    shape = (y2 - y1 + 1, x2 - x1 + 1)
    box_mask = np.ones(shape, dtype=bool)
    if not isinstance(value, list) or len(value) < _MIN_POLYGON_POINTS:
        return box_mask
    try:
        points = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return box_mask
    if points.ndim != _POINT_DIMENSIONS or points.shape[1] != _POINT_DIMENSIONS:
        return box_mask
    points = points / _NORMALIZED_MAX * np.array([width, height])
    rows, cols = skimage.draw.polygon(points[:, 1] - y1, points[:, 0] - x1, shape=shape)
    mask = np.zeros(shape, dtype=bool)
    mask[rows, cols] = True
    return mask if mask.any() else box_mask
