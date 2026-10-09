from __future__ import annotations


class GeminiError(Exception):
    pass


class GeminiAuthError(GeminiError):
    pass


class GeminiQuotaError(GeminiError):
    pass


class GeminiNetworkError(GeminiError):
    pass


class GeminiResponseError(GeminiError):
    pass


class GeminiRequestError(GeminiError):
    pass


class GeminiUnavailableError(GeminiError):
    pass
