from __future__ import annotations

import threading

from PySide6 import QtCore
from pytestqt.qtbot import QtBot

from labelme._text_prompt_runner import TextPromptRunner


# Each test keeps `parent` in a local so the runner is not destroyed mid-test.
def test_runner_reports_result(qtbot: QtBot) -> None:
    parent = QtCore.QObject()
    runner = TextPromptRunner(parent=parent)
    with qtbot.waitSignal(runner.finished) as blocker:
        assert runner.submit(job="job", run=lambda: ["shape"])
    assert blocker.args == ["job", ["shape"]]
    assert not runner.is_busy


def test_runner_reports_exceptions(qtbot: QtBot) -> None:
    parent = QtCore.QObject()
    runner = TextPromptRunner(parent=parent)
    error = RuntimeError("boom")

    def fail() -> object:
        raise error

    with qtbot.waitSignal(runner.finished) as blocker:
        runner.submit(job="job", run=fail)
    assert blocker.args[1] is error


def test_runner_rejects_submit_while_busy(qtbot: QtBot) -> None:
    parent = QtCore.QObject()
    runner = TextPromptRunner(parent=parent)
    release = threading.Event()
    assert runner.submit(job=1, run=lambda: release.wait(5) and [])
    assert runner.is_busy
    assert not runner.submit(job=2, run=list)
    with qtbot.waitSignal(runner.finished) as blocker:
        release.set()
    assert blocker.args[0] == 1
    runner.shutdown()
