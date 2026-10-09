from __future__ import annotations

import base64
import io
import json
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Final

import numpy as np
import PIL.Image
from loguru import logger
from numpy.typing import NDArray

from ._gemini_errors import GeminiAuthError
from ._gemini_errors import GeminiNetworkError
from ._gemini_errors import GeminiQuotaError
from ._gemini_errors import GeminiRequestError
from ._gemini_errors import GeminiResponseError
from ._gemini_errors import GeminiUnavailableError
from ._gemini_prompt import build_prompt
from ._gemini_prompt import build_response_schema
from ._gemini_prompt import parse_response

GEMINI_ENDPOINT: Final[str] = (
    "https://generativelanguage.googleapis.com/v1beta/interactions"
)
Post = Callable[[str, dict[str, str], bytes, float], bytes]


def _urllib_post(
    url: str, headers: dict[str, str], body: bytes, timeout: float, /
) -> bytes:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


class GeminiSession:
    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        timeout: float = 60.0,
        post: Post | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._timeout = timeout
        # Resolved at construction so tests can monkeypatch the module function.
        self._post = post if post is not None else _urllib_post

    @property
    def model_name(self) -> str:
        return "gemini"

    def __repr__(self) -> str:
        return f"GeminiSession(model={self._model!r})"

    def detect(
        self, *, image: NDArray[np.uint8], texts: list[str], want_masks: bool
    ) -> tuple[
        NDArray[np.float32],
        NDArray[np.float32],
        NDArray[np.int32],
        list[NDArray[np.bool_]] | None,
    ]:
        height, width = image.shape[:2]
        body = json.dumps(
            {
                "model": self._model,
                "input": [
                    {
                        "type": "text",
                        "text": build_prompt(texts=texts, want_masks=want_masks),
                    },
                    {
                        "type": "image",
                        "data": _encode_jpeg(image),
                        "mime_type": "image/jpeg",
                    },
                ],
                "response_format": {
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": build_response_schema(want_masks=want_masks),
                },
            }
        ).encode()
        headers = {"Content-Type": "application/json", "x-goog-api-key": self._api_key}
        AUTH_ERROR_CODES: Final[frozenset[int]] = frozenset({401, 403})
        QUOTA_ERROR_CODE: Final[int] = 429
        SERVER_ERROR_MIN_CODE: Final[int] = 500
        try:
            raw = self._post(GEMINI_ENDPOINT, headers, body, self._timeout)
        except urllib.error.HTTPError as e:
            if e.code in AUTH_ERROR_CODES:
                raise GeminiAuthError("Gemini rejected the API key") from None
            if e.code == QUOTA_ERROR_CODE:
                raise GeminiQuotaError("Gemini quota exceeded") from None
            if e.code >= SERVER_ERROR_MIN_CODE:
                raise GeminiUnavailableError(f"Gemini returned HTTP {e.code}") from None
            raise GeminiRequestError(_read_error_message(e)) from None
        except Exception:  # noqa: BLE001 -- any transport failure must stay a GeminiError
            # `from None` drops the original traceback: with loguru's diagnose
            # mode, it would print this frame's locals, including the key header.
            raise GeminiNetworkError("Could not reach Gemini") from None
        try:
            text = _extract_output_text(raw)
            logger.debug("Gemini response text: {!r}", text)
            return parse_response(
                text=text,
                texts=texts,
                image_size=(width, height),
                want_masks=want_masks,
            )
        except GeminiResponseError:
            raise
        except Exception:  # noqa: BLE001 -- malformed numbers must stay a GeminiError
            raise GeminiResponseError("Gemini returned unusable values") from None


def _read_error_message(error: urllib.error.HTTPError, /) -> str:
    # Google explains rejected requests in the body; it never echoes the key.
    try:
        message = json.loads(error.read())["error"]["message"]
    except Exception:  # noqa: BLE001 -- any unreadable body falls back to the status
        return f"Gemini returned HTTP {error.code}"
    return message if isinstance(message, str) else f"Gemini returned HTTP {error.code}"


def _encode_jpeg(image: NDArray[np.uint8], /) -> str:
    # Gemini answers in 0-1000 normalized coordinates, so downscaling the upload
    # does not change the boxes it returns; it only cuts upload size and latency.
    MAX_IMAGE_SIDE: Final[int] = 1536
    pil_image = PIL.Image.fromarray(image).convert("RGB")
    pil_image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _extract_output_text(raw: bytes, /) -> str:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as e:
        raise GeminiResponseError("Gemini returned a non-JSON body") from e
    if not isinstance(payload, dict):
        raise GeminiResponseError("Gemini returned an unexpected body")
    parts = [
        block["text"]
        for step in payload.get("steps", [])
        if isinstance(step, dict) and step.get("type") == "model_output"
        for block in step.get("content", [])
        if isinstance(block, dict)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
    ]
    if not parts:
        raise GeminiResponseError("Gemini response has no text output")
    return "".join(parts)
