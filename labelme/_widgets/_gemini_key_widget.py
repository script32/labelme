from __future__ import annotations

from PySide6 import QtCore
from PySide6 import QtWidgets

from .. import _gemini_key


class GeminiKeyWidget(QtWidgets.QWidget):
    _key_edit: QtWidgets.QLineEdit
    _clear_button: QtWidgets.QPushButton

    def __init__(
        self,
        *,
        settings: QtCore.QSettings,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._settings = settings

        self._key_edit = QtWidgets.QLineEdit()
        self._key_edit.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        self._key_edit.setPlaceholderText(self.tr("Paste your Gemini API key"))
        self._key_edit.setAccessibleName(self.tr("Gemini API key"))
        self._key_edit.setText(_gemini_key.get_gemini_api_key(settings=settings) or "")
        self._key_edit.textChanged.connect(self._save)

        self._clear_button = QtWidgets.QPushButton(self.tr("Clear"))
        self._clear_button.clicked.connect(self._key_edit.clear)

        key_row = QtWidgets.QHBoxLayout()
        key_row.addWidget(self._key_edit, stretch=1)
        key_row.addWidget(self._clear_button)

        note = QtWidgets.QLabel(
            self.tr(
                "Stored only on this computer, outside the labelme config file. "
                "Running Gemini sends the current image to Google."
            )
        )
        note.setWordWrap(True)

        layout = QtWidgets.QFormLayout(self)
        layout.addRow(self.tr("API key"), key_row)
        layout.addRow(note)

    def _save(self, text: str, /) -> None:
        _gemini_key.set_gemini_api_key(settings=self._settings, key=text)
