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
