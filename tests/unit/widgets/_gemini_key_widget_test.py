from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore
from PySide6 import QtWidgets
from pytestqt.qtbot import QtBot

from labelme import _gemini_key
from labelme._widgets._gemini_key_widget import GeminiKeyWidget


def _settings(tmp_path: Path, /) -> QtCore.QSettings:
    return QtCore.QSettings(str(tmp_path / "s.ini"), QtCore.QSettings.Format.IniFormat)


def test_key_field_is_masked_and_prefilled(*, qtbot: QtBot, tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _gemini_key.set_gemini_api_key(settings=settings, key="abc")
    widget = GeminiKeyWidget(settings=settings)
    qtbot.addWidget(widget)
    assert widget._key_edit.echoMode() == QtWidgets.QLineEdit.EchoMode.Password
    assert widget._key_edit.text() == "abc"


def test_typing_saves_and_clear_removes(*, qtbot: QtBot, tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    widget = GeminiKeyWidget(settings=settings)
    qtbot.addWidget(widget)
    widget._key_edit.setText(" new-key ")
    assert _gemini_key.get_gemini_api_key(settings=settings) == "new-key"
    widget._clear_button.click()
    assert widget._key_edit.text() == ""
    assert _gemini_key.get_gemini_api_key(settings=settings) is None
