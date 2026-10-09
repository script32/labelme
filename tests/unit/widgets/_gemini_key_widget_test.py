from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PySide6 import QtCore
from PySide6 import QtWidgets
from pytestqt.qtbot import QtBot

from labelme import _gemini_key
from labelme._automation import GeminiAuthError
from labelme._widgets._gemini_key_widget import GeminiKeyWidget


def _settings(tmp_path: Path, /) -> QtCore.QSettings:
    return QtCore.QSettings(str(tmp_path / "s.ini"), QtCore.QSettings.Format.IniFormat)


def _no_models(_api_key: str, /) -> list[str]:
    raise AssertionError("the model list must not be fetched")


def _widget(
    settings: QtCore.QSettings,
    /,
    *,
    applied: list[str],
    editable: bool,
    list_models: Callable[[str], list[str]],
) -> GeminiKeyWidget:
    def apply_model(name: str, /) -> bool:
        applied.append(name)
        return True

    return GeminiKeyWidget(
        settings=settings,
        model="gemini-3.8-flash",
        apply_model=apply_model,
        model_editable=editable,
        list_models=list_models,
    )


def test_key_field_is_masked_and_prefilled(*, qtbot: QtBot, tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _gemini_key.set_gemini_api_key(settings=settings, key="abc")
    widget = _widget(settings, applied=[], editable=True, list_models=_no_models)
    qtbot.addWidget(widget)
    assert widget._key_edit.echoMode() == QtWidgets.QLineEdit.EchoMode.Password
    assert widget._key_edit.text() == "abc"


def test_typing_saves_and_clear_removes(*, qtbot: QtBot, tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    widget = _widget(settings, applied=[], editable=True, list_models=_no_models)
    qtbot.addWidget(widget)
    widget._key_edit.setText(" new-key ")
    assert _gemini_key.get_gemini_api_key(settings=settings) == "new-key"
    widget._clear_button.click()
    assert widget._key_edit.text() == ""
    assert _gemini_key.get_gemini_api_key(settings=settings) is None


def test_model_field_shows_current_model_and_applies_edits(
    *, qtbot: QtBot, tmp_path: Path
) -> None:
    applied: list[str] = []
    widget = _widget(
        _settings(tmp_path), applied=applied, editable=True, list_models=_no_models
    )
    qtbot.addWidget(widget)
    assert widget._model_combo.currentText() == "gemini-3.8-flash"
    line_edit = widget._model_combo.lineEdit()
    assert line_edit is not None
    widget._model_combo.setEditText("  gemini-3.7-flash ")
    line_edit.editingFinished.emit()
    widget._model_combo.setEditText("")
    line_edit.editingFinished.emit()
    assert applied == ["gemini-3.7-flash"]


def test_model_field_is_locked_when_settings_are_not_editable(
    *, qtbot: QtBot, tmp_path: Path
) -> None:
    widget = _widget(
        _settings(tmp_path), applied=[], editable=False, list_models=_no_models
    )
    qtbot.addWidget(widget)
    assert not widget._model_combo.isEnabled()
    assert widget._key_edit.isEnabled()


def test_load_models_fills_the_list_and_keeps_the_current_model(
    *, qtbot: QtBot, tmp_path: Path
) -> None:
    settings = _settings(tmp_path)
    _gemini_key.set_gemini_api_key(settings=settings, key="abc")
    keys: list[str] = []

    def list_models(api_key: str, /) -> list[str]:
        keys.append(api_key)
        return ["gemini-3.8-flash", "gemini-3.5-flash"]

    widget = _widget(settings, applied=[], editable=True, list_models=list_models)
    qtbot.addWidget(widget)
    widget._load_button.click()
    qtbot.waitUntil(lambda: widget._model_combo.count() == 2, timeout=5000)
    assert keys == ["abc"]
    assert widget._model_combo.currentText() == "gemini-3.8-flash"
    assert widget._load_button.isEnabled()


def test_load_models_without_key_explains_and_fetches_nothing(
    *, qtbot: QtBot, tmp_path: Path
) -> None:
    widget = _widget(
        _settings(tmp_path), applied=[], editable=True, list_models=_no_models
    )
    qtbot.addWidget(widget)
    widget._load_button.click()
    assert widget._model_status.text() == "Set the API key first."


def test_load_models_failure_is_explained(*, qtbot: QtBot, tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _gemini_key.set_gemini_api_key(settings=settings, key="abc")

    def list_models(_api_key: str, /) -> list[str]:
        raise GeminiAuthError("Gemini rejected the API key")

    widget = _widget(settings, applied=[], editable=True, list_models=list_models)
    qtbot.addWidget(widget)
    widget._load_button.click()
    qtbot.waitUntil(lambda: widget._model_status.text() != "", timeout=5000)
    assert widget._model_status.text() == "Gemini rejected the API key."
    assert widget._load_button.isEnabled()
