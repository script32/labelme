from __future__ import annotations

import base64
import email.message
import http.client
import io
import urllib.error
from typing import Final

import numpy as np
import PIL.Image
import pytest

from labelme._automation._gemini_errors import GeminiAuthError
from labelme._automation._gemini_errors import GeminiError
from labelme._automation._gemini_errors import GeminiNetworkError
from labelme._automation._gemini_errors import GeminiQuotaError
from labelme._automation._gemini_errors import GeminiRequestError
from labelme._automation._gemini_errors import GeminiResponseError
from labelme._automation._gemini_errors import GeminiUnavailableError
from labelme._automation._gemini_session import GEMINI_ENDPOINT
from labelme._automation._gemini_session import GeminiSession

from ._gemini_fakes import FakePost
from ._gemini_fakes import interaction

_KEY: Final = "secret-key-123"


def _session(post: FakePost, /) -> GeminiSession:
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
    # gemini-3.8-flash rejects thinking_level "minimal" with HTTP 400.
    assert "generation_config" not in body
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
            GeminiUnavailableError,
        ),
        (
            urllib.error.HTTPError(
                GEMINI_ENDPOINT, 503, "x", email.message.Message(), None
            ),
            GeminiUnavailableError,
        ),
        (urllib.error.URLError("no route"), GeminiNetworkError),
        (TimeoutError(), GeminiNetworkError),
    ],
)
def test_detect_maps_transport_errors(
    *, error: BaseException, expected: type[GeminiError]
) -> None:
    with pytest.raises(expected) as exc_info:
        _session(FakePost(error=error)).detect(
            image=np.zeros((10, 10, 3), dtype=np.uint8), texts=["nut"], want_masks=False
        )
    assert _KEY not in str(exc_info.value)


@pytest.mark.parametrize("raw", [b"not json", b"[]", b'{"steps": []}'])
def test_detect_rejects_responses_without_text_output(*, raw: bytes) -> None:
    with pytest.raises(GeminiResponseError):
        _session(FakePost(response=raw)).detect(
            image=np.zeros((10, 10, 3), dtype=np.uint8), texts=["nut"], want_masks=False
        )


def test_repr_hides_api_key() -> None:
    assert _KEY not in repr(_session(FakePost(response=b"")))
    assert _session(FakePost(response=b"")).model_name == "gemini"


@pytest.mark.parametrize(
    "error",
    [
        UnicodeEncodeError("latin-1", "​", 0, 1, "ordinal not in range"),
        http.client.IncompleteRead(b""),
        http.client.BadStatusLine("garbage"),
    ],
)
def test_detect_wraps_unexpected_transport_errors_without_leaking_key(
    *, error: BaseException
) -> None:
    with pytest.raises(GeminiNetworkError) as exc_info:
        _session(FakePost(error=error)).detect(
            image=np.zeros((10, 10, 3), dtype=np.uint8), texts=["nut"], want_masks=False
        )
    # A bare traceback would expose detect()'s locals, including the key header.
    assert exc_info.value.__suppress_context__
    assert _KEY not in str(exc_info.value)


def test_detect_wraps_unparsable_numbers_as_response_error() -> None:
    post = FakePost(
        response=interaction('[{"box_2d": [0, 0, 1e999, 10], "label": "nut"}]')
    )
    with pytest.raises(GeminiResponseError):
        _session(post).detect(
            image=np.zeros((10, 10, 3), dtype=np.uint8), texts=["nut"], want_masks=False
        )


def test_detect_reports_googles_message_for_rejected_requests() -> None:
    body = io.BytesIO(
        b'{"error": {"message": "Thinking level is not supported.", "code": "x"}}'
    )
    error = urllib.error.HTTPError(
        GEMINI_ENDPOINT, 400, "Bad Request", email.message.Message(), body
    )
    with pytest.raises(GeminiRequestError) as exc_info:
        _session(FakePost(error=error)).detect(
            image=np.zeros((10, 10, 3), dtype=np.uint8), texts=["nut"], want_masks=False
        )
    assert str(exc_info.value) == "Thinking level is not supported."
    assert exc_info.value.__suppress_context__


def test_detect_falls_back_to_status_when_rejection_has_no_message() -> None:
    error = urllib.error.HTTPError(
        GEMINI_ENDPOINT,
        404,
        "Not Found",
        email.message.Message(),
        io.BytesIO(b"<html>"),
    )
    with pytest.raises(GeminiRequestError, match="HTTP 404"):
        _session(FakePost(error=error)).detect(
            image=np.zeros((10, 10, 3), dtype=np.uint8), texts=["nut"], want_masks=False
        )
