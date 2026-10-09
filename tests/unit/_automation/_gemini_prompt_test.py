from __future__ import annotations

import json

import numpy as np
import pytest

from labelme._automation._gemini_errors import GeminiResponseError
from labelme._automation._gemini_prompt import build_prompt
from labelme._automation._gemini_prompt import build_response_schema
from labelme._automation._gemini_prompt import parse_response


def test_build_prompt_lists_every_label_and_asks_masks_only_when_wanted() -> None:
    prompt = build_prompt(texts=["rusty screw", "nut"], want_masks=False)
    assert '"rusty screw"' in prompt
    assert '"nut"' in prompt
    assert "mask" not in prompt
    assert "mask" in build_prompt(texts=["nut"], want_masks=True)


def test_response_schema_requires_mask_only_when_wanted() -> None:
    assert "mask" not in build_response_schema(want_masks=False)["items"]["properties"]
    assert "mask" in build_response_schema(want_masks=True)["items"]["required"]


def test_parse_converts_normalized_yxyx_to_pixel_xyxy() -> None:
    boxes, scores, labels, masks = parse_response(
        text='[{"box_2d": [100, 200, 500, 600], "label": "nut"}]',
        texts=["screw", "nut"],
        image_size=(1000, 500),
        want_masks=False,
    )
    np.testing.assert_array_equal(boxes, [[200, 50, 600, 250]])
    np.testing.assert_array_equal(scores, [1.0])
    np.testing.assert_array_equal(labels, [1])
    assert masks is None


def test_parse_accepts_fenced_json_and_loose_label_case() -> None:
    text = '```json\n[{"box_2d": [0, 0, 1000, 1000], "label": " Nut "}]\n```'
    boxes, _, labels, _ = parse_response(
        text=text, texts=["nut"], image_size=(10, 10), want_masks=False
    )
    np.testing.assert_array_equal(boxes, [[0, 0, 9, 9]])
    np.testing.assert_array_equal(labels, [0])


def test_parse_drops_unknown_labels_and_bad_boxes() -> None:
    text = json.dumps(
        [
            {"box_2d": [0, 0, 500, 500], "label": "cat"},
            {"box_2d": [100, 100, 100, 300], "label": "nut"},
            {"box_2d": [1, 2, 3], "label": "nut"},
            {"box_2d": "x", "label": "nut"},
            {"box_2d": [0, 0, True, 10], "label": "nut"},
            "garbage",
        ]
    )
    boxes, scores, labels, _ = parse_response(
        text=text, texts=["nut"], image_size=(100, 100), want_masks=False
    )
    assert boxes.shape == (0, 4)
    assert scores.shape == (0,)
    assert labels.shape == (0,)


def test_parse_rasterizes_polygon_into_box_cropped_mask() -> None:
    text = json.dumps(
        [
            {
                "box_2d": [0, 0, 1000, 1000],
                "label": "nut",
                "mask": [[0, 0], [1000, 0], [0, 1000]],
            }
        ]
    )
    boxes, _, _, masks = parse_response(
        text=text, texts=["nut"], image_size=(101, 101), want_masks=True
    )
    np.testing.assert_array_equal(boxes, [[0, 0, 100, 100]])
    assert masks is not None
    assert masks[0].shape == (101, 101)
    assert 0.4 < masks[0].mean() < 0.6


def test_parse_falls_back_to_box_mask_when_polygon_missing_or_invalid() -> None:
    text = json.dumps(
        [
            {"box_2d": [0, 0, 500, 500], "label": "nut"},
            {"box_2d": [0, 0, 500, 500], "label": "nut", "mask": [[1, 2]]},
        ]
    )
    _, _, _, masks = parse_response(
        text=text, texts=["nut"], image_size=(100, 100), want_masks=True
    )
    assert masks is not None
    assert len(masks) == 2
    for mask in masks:
        assert mask.shape == (51, 51)
        assert mask.all()


@pytest.mark.parametrize("text", ["not json", '{"box_2d": [0, 0, 1, 1]}'])
def test_parse_rejects_non_list_responses(text: str) -> None:
    with pytest.raises(GeminiResponseError):
        parse_response(text=text, texts=["nut"], image_size=(10, 10), want_masks=False)
