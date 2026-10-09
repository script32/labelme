from __future__ import annotations

import functools
from collections.abc import Callable

from PySide6 import QtCore


def _run_and_notify(
    job: object, run: Callable[[], object], completed: QtCore.SignalInstance, /
) -> None:
    try:
        outcome = run()
    except BaseException as error:
        # Every failure must reach the GUI, or the Run button stays disabled.
        completed.emit(job, error)
    else:
        completed.emit(job, outcome)


class TextPromptRunner(QtCore.QObject):
    finished = QtCore.Signal(object, object)
    _completed = QtCore.Signal(object, object)

    def __init__(self, *, parent: QtCore.QObject) -> None:
        super().__init__(parent)
        self._pool = QtCore.QThreadPool(self)
        self._pool.setMaxThreadCount(1)
        self._busy = False
        # Results belong to the GUI thread, even if the job finishes immediately.
        self._completed.connect(
            self._on_completed, QtCore.Qt.ConnectionType.QueuedConnection
        )

    @property
    def is_busy(self) -> bool:
        return self._busy

    def submit(self, *, job: object, run: Callable[[], object]) -> bool:
        if self._busy:
            return False
        self._busy = True
        self._pool.start(functools.partial(_run_and_notify, job, run, self._completed))
        return True

    @QtCore.Slot(object, object)
    def _on_completed(self, job: object, outcome: object, /) -> None:
        self._busy = False
        self.finished.emit(job, outcome)

    def shutdown(self) -> None:
        self._pool.waitForDone()
