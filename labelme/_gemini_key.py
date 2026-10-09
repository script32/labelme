from __future__ import annotations

from typing import Final

from PySide6 import QtCore

# Stored in the per-machine Qt settings, never in the user Config (~/.labelmerc),
# which people often keep in dotfiles or version control.
_API_KEY: Final[str] = "gemini/apiKey"
_PRIVACY_ACCEPTED: Final[str] = "gemini/privacyAccepted"


def get_gemini_api_key(*, settings: QtCore.QSettings) -> str | None:
    value = settings.value(_API_KEY, "", type=str)
    assert isinstance(value, str)
    return value.strip() or None


def set_gemini_api_key(*, settings: QtCore.QSettings, key: str | None) -> None:
    # Keys pasted from web pages can carry zero-width or other invisible
    # characters, which an HTTP header cannot encode.
    if key is not None:
        key = "".join(ch for ch in key if ch.isprintable() and not ch.isspace())
    if key:
        settings.setValue(_API_KEY, key)
    else:
        settings.remove(_API_KEY)
    settings.sync()


def has_accepted_gemini_privacy(*, settings: QtCore.QSettings) -> bool:
    accepted = settings.value(_PRIVACY_ACCEPTED, False, type=bool)  # noqa: FBT003 -- QSettings default is positional
    return bool(accepted)


def accept_gemini_privacy(*, settings: QtCore.QSettings) -> None:
    settings.setValue(_PRIVACY_ACCEPTED, True)  # noqa: FBT003 -- QSettings value is positional
    settings.sync()
