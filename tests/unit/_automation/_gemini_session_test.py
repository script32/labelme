from __future__ import annotations

import base64
import email.message
import io
import urllib.error

import numpy as np
import PIL.Image
import pytest

from labelme._automation._gemini_errors import GeminiAuthError
from labelme._automation._gemini_errors import GeminiError
from labelme._automation._gemini_errors import GeminiNetworkError
from labelme._automation._gemini_errors import GeminiQuotaError
from labelme._automation._gemini_errors import GeminiResponseError
from labelme._automation._gemini_session import GEMINI_ENDPOINT
from labelme._automation._gemini_session import GeminiSession

from ._gemini_fakes import FakePost
from ._gemini_fakes import interaction

_KEY = "secret-key-123"


def _session(post: FakePost) -> GeminiSession:
    return GeminiSession(model="gemini-3.8-flash", api_key=_KEY, post=post)


def test_detect_sends_key_model_prompt_and_image() -> None:
    post = FakePost(
        response=interaction('[{"box_2d": [0, 0, 500, 500], "label": "nut"}]')
    )
    boxes, _, labels, masks = _session(post).detect(
        image=np.zeros((100, 200, 3), dtype=np.uint8), texts=["nut"], want_masks=False
    )
    url, headers, body, timeout = post.calls[0]
    assert url == GEMINI_ENDPOINT
    assert headers["x-goog-api-key"] == _KEY
    assert timeout == 60.0
    assert body["model"] == "gemini-3.8-flash"
    assert body["generation_config"] == {"thinking_level": "minimal"}
    assert body["response_format"]["mime_type"] == "application/json"
    assert body["input"][0]["type"] == "text"
    assert '"nut"' in body["input"][0]["text"]
    assert body["input"][1]["type"] == "image"
    assert body["input"][1]["mime_type"] == "image/jpeg"
    np.testing.assert_array_equal(boxes, [[0, 0, 100, 50]])
    np.testing.assert_array_equal(labels, [0])
    assert masks is None


def test_detect_downscales_large_images_but_keeps_original_coordinates() -> None:
    post = FakePost(
        response=interaction('[{"box_2d": [0, 0, 1000, 1000], "label": "nut"}]')
    )
    boxes, _, _, _ = _session(post).detect(
        image=np.zeros((1000, 3000, 3), dtype=np.uint8), texts=["nut"], want_masks=False
    )
    sent = PIL.Image.open(
        io.BytesIO(base64.b64decode(post.calls[0][2]["input"][1]["data"]))
    )
    assert max(sent.size) == 1536
    np.testing.assert_array_equal(boxes, [[0, 0, 2999, 999]])


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            urllib.error.HTTPError(
                GEMINI_ENDPOINT, 401, "x", email.message.Message(), None
            ),
            GeminiAuthError,
        ),
        (
            urllib.error.HTTPError(
                GEMINI_ENDPOINT, 403, "x", email.message.Message(), None
            ),
            GeminiAuthError,
        ),
        (
            urllib.error.HTTPError(
                GEMINI_ENDPOINT, 429, "x", email.message.Message(), None
            ),
            GeminiQuotaError,
        ),
        (
            urllib.error.HTTPError(
                GEMINI_ENDPOINT, 500, "x", email.message.Message(), None
            ),
            GeminiNetworkError,
        ),
        (urllib.error.URLError("no route"), GeminiNetworkError),
        (TimeoutError(), GeminiNetworkError),
    ],
)
def test_detect_maps_transport_errors(
    error: BaseException, expected: type[GeminiError]
) -> None:
    with pytest.raises(expected) as exc_info:
        _session(FakePost(error=error)).detect(
            image=np.zeros((10, 10, 3), dtype=np.uint8), texts=["nut"], want_masks=False
        )
    assert _KEY not in str(exc_info.value)


@pytest.mark.parametrize("raw", [b"not json", b"[]", b'{"steps": []}'])
def test_detect_rejects_responses_without_text_output(raw: bytes) -> None:
    with pytest.raises(GeminiResponseError):
        _session(FakePost(response=raw)).detect(
            image=np.zeros((10, 10, 3), dtype=np.uint8), texts=["nut"], want_masks=False
        )


def test_repr_hides_api_key() -> None:
    assert _KEY not in repr(_session(FakePost(response=b"")))
    assert _session(FakePost(response=b"")).model_name == "gemini"
