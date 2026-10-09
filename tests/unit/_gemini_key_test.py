from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore

from labelme import _gemini_key


def _settings(tmp_path: Path, /) -> QtCore.QSettings:
    return QtCore.QSettings(str(tmp_path / "s.ini"), QtCore.QSettings.Format.IniFormat)


def test_api_key_round_trip(*, tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    assert _gemini_key.get_gemini_api_key(settings=settings) is None
    _gemini_key.set_gemini_api_key(settings=settings, key="abc")
    assert _gemini_key.get_gemini_api_key(settings=settings) == "abc"
    _gemini_key.set_gemini_api_key(settings=settings, key=None)
    assert _gemini_key.get_gemini_api_key(settings=settings) is None


def test_api_key_is_trimmed_and_blank_removes_it(*, tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _gemini_key.set_gemini_api_key(settings=settings, key="  abc \n")
    assert _gemini_key.get_gemini_api_key(settings=settings) == "abc"
    _gemini_key.set_gemini_api_key(settings=settings, key="   ")
    assert _gemini_key.get_gemini_api_key(settings=settings) is None
    assert "gemini/apiKey" not in settings.allKeys()


def test_privacy_acceptance_persists(*, tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    assert not _gemini_key.has_accepted_gemini_privacy(settings=settings)
    _gemini_key.accept_gemini_privacy(settings=settings)
    assert _gemini_key.has_accepted_gemini_privacy(settings=_settings(tmp_path))


def test_api_key_drops_invisible_characters(*, tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _gemini_key.set_gemini_api_key(settings=settings, key="​abc​ def\n")
    assert _gemini_key.get_gemini_api_key(settings=settings) == "abcdef"
