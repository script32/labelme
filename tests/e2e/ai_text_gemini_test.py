from __future__ import annotations

import email.message
import io
import json
import threading
import urllib.error
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import Final

import pytest
from PySide6 import QtWidgets
from pytestqt.qtbot import QtBot

from labelme import _gemini_key
from labelme._automation import _gemini_session
from labelme._widgets._gemini_key_widget import GeminiKeyWidget

from .conftest import MainWinFactory
from .conftest import show_window_and_wait_for_imagedata

if TYPE_CHECKING:
    from labelme._app import MainWindow


class _FakePost:
    calls: int
    models: list[str]

    def __init__(
        self,
        *,
        entries: list[dict[str, Any]] | None = None,
        error: BaseException | None = None,
        gate: threading.Event | None = None,
    ) -> None:
        self.calls = 0
        self.models = []
        self._entries = entries or []
        self._error = error
        self._gate = gate

    def __call__(
        self,
        _url: str,
        _headers: dict[str, str],
        body: bytes,
        _timeout: float,
        /,
    ) -> bytes:
        self.calls += 1
        self.models.append(json.loads(body)["model"])
        if self._gate is not None:
            self._gate.wait(5)
        if self._error is not None:
            raise self._error
        text = json.dumps(self._entries)
        return json.dumps(
            {
                "steps": [
                    {
                        "type": "model_output",
                        "content": [{"type": "text", "text": text}],
                    }
                ]
            }
        ).encode()


_PERSON: Final[dict[str, Any]] = {
    "box_2d": [100, 100, 600, 500],
    "label": "person",
    "mask": [[100, 100], [500, 100], [500, 600], [100, 600]],
}


def _open(
    main_win: MainWinFactory, qtbot: QtBot, data_path: Path, tmp_path: Path, /
) -> MainWindow:
    win = main_win(
        file_or_dir=str(data_path / "raw/2011_000003.jpg"),
        config_overrides=dict(auto_save=True),
        output_dir=str(tmp_path),
    )
    show_window_and_wait_for_imagedata(qtbot=qtbot, win=win)
    return win


def _configure(win: MainWindow, /, *, key: str | None, privacy: bool) -> None:
    _gemini_key.set_gemini_api_key(settings=win._window_state, key=key)
    if privacy:
        _gemini_key.accept_gemini_privacy(settings=win._window_state)


def _run(win: MainWindow, qtbot: QtBot, /, *, create_mode: str) -> None:
    win._switch_canvas_mode(edit=False, create_mode=create_mode)
    win._ai_text._text_input.setText("person")
    win._ai_text.set_model_name(model_name="gemini")
    win._submit_ai_prompt(False)  # noqa: FBT003
    qtbot.waitUntil(lambda: not win._text_prompt_runner.is_busy, timeout=5000)
    qtbot.wait(50)


@pytest.mark.gui
@pytest.mark.parametrize("create_mode", ["rectangle", "polygon"])
def test_gemini_creates_shapes(
    *,
    main_win: MainWinFactory,
    qtbot: QtBot,
    data_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    create_mode: str,
) -> None:
    fake = _FakePost(entries=[_PERSON])
    monkeypatch.setattr(_gemini_session, "_urllib_post", fake)
    win = _open(main_win, qtbot, data_path, tmp_path)
    _configure(win, key="test-key", privacy=True)
    _run(win, qtbot, create_mode=create_mode)
    shapes = win._canvas_widgets.canvas.shapes
    assert fake.calls == 1
    assert [s.label for s in shapes] == ["person"]
    assert shapes[0].shape_type == create_mode


@pytest.mark.gui
def test_gemini_without_key_asks_for_settings_and_sends_nothing(
    *,
    main_win: MainWinFactory,
    qtbot: QtBot,
    data_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakePost(entries=[_PERSON])
    monkeypatch.setattr(_gemini_session, "_urllib_post", fake)
    prompts: list[str] = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "information",
        lambda _parent, title, *_args, **_kw: prompts.append(title)
        or QtWidgets.QMessageBox.StandardButton.Cancel,
    )
    win = _open(main_win, qtbot, data_path, tmp_path)
    _configure(win, key=None, privacy=True)
    _run(win, qtbot, create_mode="rectangle")
    assert prompts == ["Gemini API Key Required"]
    assert fake.calls == 0
    assert win._canvas_widgets.canvas.shapes == []


@pytest.mark.gui
def test_gemini_privacy_prompt_once_and_decline_sends_nothing(
    *,
    main_win: MainWinFactory,
    qtbot: QtBot,
    data_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakePost(entries=[_PERSON])
    monkeypatch.setattr(_gemini_session, "_urllib_post", fake)
    answers = [
        QtWidgets.QMessageBox.StandardButton.No,
        QtWidgets.QMessageBox.StandardButton.Yes,
    ]
    asked: list[str] = []

    def question(
        _parent: object, title: str, *_args: object, **_kw: object
    ) -> QtWidgets.QMessageBox.StandardButton:
        asked.append(title)
        return answers.pop(0)

    monkeypatch.setattr(QtWidgets.QMessageBox, "question", question)
    win = _open(main_win, qtbot, data_path, tmp_path)
    _configure(win, key="test-key", privacy=False)
    _run(win, qtbot, create_mode="rectangle")
    assert fake.calls == 0
    _run(win, qtbot, create_mode="rectangle")
    assert fake.calls == 1
    _run(win, qtbot, create_mode="rectangle")
    assert asked == ["Send Image to Google?", "Send Image to Google?"]
    assert fake.calls == 2


@pytest.mark.gui
def test_gemini_auth_error_keeps_annotations_and_explains(
    *,
    main_win: MainWinFactory,
    qtbot: QtBot,
    data_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = urllib.error.HTTPError(
        _gemini_session.GEMINI_ENDPOINT, 401, "x", email.message.Message(), None
    )
    monkeypatch.setattr(_gemini_session, "_urllib_post", _FakePost(error=error))
    win = _open(main_win, qtbot, data_path, tmp_path)
    _configure(win, key="test-key", privacy=True)
    _run(win, qtbot, create_mode="rectangle")
    assert win._canvas_widgets.canvas.shapes == []
    assert "Gemini rejected the API key" in win.statusBar().currentMessage()
    assert "test-key" not in win.statusBar().currentMessage()


@pytest.mark.gui
def test_gemini_result_discarded_after_image_switch(
    *,
    main_win: MainWinFactory,
    qtbot: QtBot,
    data_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = threading.Event()
    monkeypatch.setattr(
        _gemini_session, "_urllib_post", _FakePost(entries=[_PERSON], gate=gate)
    )
    win = _open(main_win, qtbot, data_path, tmp_path)
    _configure(win, key="test-key", privacy=True)
    win._switch_canvas_mode(edit=False, create_mode="rectangle")
    win._ai_text._text_input.setText("person")
    win._ai_text.set_model_name(model_name="gemini")
    win._submit_ai_prompt(False)  # noqa: FBT003
    assert win._text_prompt_runner.is_busy
    assert not win._ai_text._run_button.isEnabled()
    assert win._load_file(image_or_label_path=str(data_path / "raw/2011_000006.jpg"))
    gate.set()
    qtbot.waitUntil(lambda: not win._text_prompt_runner.is_busy, timeout=5000)
    qtbot.wait(50)
    assert win._canvas_widgets.canvas.shapes == []
    assert win._ai_text._run_button.isEnabled()


@pytest.mark.gui
def test_gemini_result_ignored_after_window_closes(
    *,
    main_win: MainWinFactory,
    qtbot: QtBot,
    data_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = threading.Event()
    monkeypatch.setattr(
        _gemini_session, "_urllib_post", _FakePost(entries=[_PERSON], gate=gate)
    )
    win = _open(main_win, qtbot, data_path, tmp_path)
    _configure(win, key="test-key", privacy=True)
    win._switch_canvas_mode(edit=False, create_mode="rectangle")
    win._ai_text._text_input.setText("person")
    win._ai_text.set_model_name(model_name="gemini")
    win._submit_ai_prompt(False)  # noqa: FBT003
    gate.set()
    win.close()
    qtbot.wait(100)
    assert win._canvas_widgets.canvas.shapes == []


@pytest.mark.gui
@pytest.mark.parametrize(
    ("code", "body", "expected"),
    [
        (
            400,
            b'{"error": {"message": "Thinking level is not supported."}}',
            "Gemini rejected the request: Thinking level is not supported.",
        ),
        (503, b"{}", "Gemini is busy right now. Try again later."),
    ],
)
def test_gemini_service_errors_explain_the_cause(
    *,
    main_win: MainWinFactory,
    qtbot: QtBot,
    data_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    code: int,
    body: bytes,
    expected: str,
) -> None:
    error = urllib.error.HTTPError(
        _gemini_session.GEMINI_ENDPOINT,
        code,
        "x",
        email.message.Message(),
        io.BytesIO(body),
    )
    monkeypatch.setattr(_gemini_session, "_urllib_post", _FakePost(error=error))
    win = _open(main_win, qtbot, data_path, tmp_path)
    _configure(win, key="test-key", privacy=True)
    _run(win, qtbot, create_mode="rectangle")
    assert expected in win.statusBar().currentMessage()


@pytest.mark.gui
def test_gemini_model_chosen_in_settings_is_used_and_saved(
    *,
    main_win: MainWinFactory,
    qtbot: QtBot,
    data_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakePost(entries=[_PERSON])
    monkeypatch.setattr(_gemini_session, "_urllib_post", fake)
    config_file = tmp_path / "config.yaml"
    config_file.write_text("auto_save: true\n")
    win = main_win(
        file_or_dir=str(data_path / "raw/2011_000003.jpg"),
        config_file=config_file,
        output_dir=str(tmp_path),
    )
    show_window_and_wait_for_imagedata(qtbot=qtbot, win=win)
    _configure(win, key="test-key", privacy=True)

    win._open_gemini_settings()
    assert win._settings_dialog is not None
    widget = win._settings_dialog.findChild(GeminiKeyWidget)
    assert widget is not None
    line_edit = widget._model_combo.lineEdit()
    assert line_edit is not None
    widget._model_combo.setEditText("gemini-3.7-flash")
    line_edit.editingFinished.emit()
    win._settings_dialog.close()

    _run(win, qtbot, create_mode="rectangle")
    assert fake.models == ["gemini-3.7-flash"]
    assert "gemini_model: gemini-3.7-flash" in config_file.read_text()
