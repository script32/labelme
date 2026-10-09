from __future__ import annotations

import json
from typing import Any


class FakePost:
    calls: list[tuple[str, dict[str, str], dict[str, Any], float]]

    def __init__(
        self, *, response: bytes | None = None, error: BaseException | None = None
    ) -> None:
        self.calls = []
        self._response = response
        self._error = error

    def __call__(
        self, url: str, headers: dict[str, str], body: bytes, timeout: float, /
    ) -> bytes:
        self.calls.append((url, headers, json.loads(body), timeout))
        if self._error is not None:
            raise self._error
        assert self._response is not None
        return self._response


def interaction(text: str, /) -> bytes:
    return json.dumps(
        {
            "steps": [
                {"type": "user_input"},
                {"type": "model_output", "content": [{"type": "text", "text": text}]},
            ]
        }
    ).encode()


class FakeGet:
    calls: list[tuple[str, dict[str, str], float]]

    def __init__(
        self, *, response: bytes | None = None, error: BaseException | None = None
    ) -> None:
        self.calls = []
        self._response = response
        self._error = error

    def __call__(self, url: str, headers: dict[str, str], timeout: float, /) -> bytes:
        self.calls.append((url, headers, timeout))
        if self._error is not None:
            raise self._error
        assert self._response is not None
        return self._response
