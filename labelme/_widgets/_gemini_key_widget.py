from __future__ import annotations

import functools
from collections.abc import Callable

from loguru import logger
from PySide6 import QtCore
from PySide6 import QtWidgets

from .. import _automation
from .. import _gemini_key
from .._text_prompt_runner import TextPromptRunner


class GeminiKeyWidget(QtWidgets.QWidget):
    _key_edit: QtWidgets.QLineEdit
    _clear_button: QtWidgets.QPushButton
    _model_combo: QtWidgets.QComboBox
    _load_button: QtWidgets.QPushButton
    _model_status: QtWidgets.QLabel

    def __init__(
        self,
        *,
        settings: QtCore.QSettings,
        model: str,
        apply_model: Callable[[str], bool],
        model_editable: bool,
        list_models: Callable[[str], list[str]],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._settings = settings
        self._model = model
        self._apply_model = apply_model
        self._list_models = list_models

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

        # The model is a regular Config setting (not a secret), so it follows
        # the Config's editability, unlike the key above.
        self._model_combo = QtWidgets.QComboBox()
        self._model_combo.setEditable(True)
        self._model_combo.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.NoInsert)
        self._model_combo.setAccessibleName(self.tr("Gemini model"))
        self._model_combo.setEditText(model)
        self._model_combo.setEnabled(model_editable)
        line_edit = self._model_combo.lineEdit()
        assert line_edit is not None
        line_edit.editingFinished.connect(self._on_model_edited)
        self._model_combo.activated.connect(self._on_model_edited)

        self._load_button = QtWidgets.QPushButton(self.tr("Load models"))
        self._load_button.setEnabled(model_editable)
        self._load_button.clicked.connect(self._load_models)

        model_row = QtWidgets.QHBoxLayout()
        model_row.addWidget(self._model_combo, stretch=1)
        model_row.addWidget(self._load_button)

        self._model_status = QtWidgets.QLabel()
        self._model_status.setWordWrap(True)

        self._runner = TextPromptRunner(parent=self)
        self._runner.finished.connect(self._on_models_loaded)

        layout = QtWidgets.QFormLayout(self)
        layout.addRow(self.tr("API key"), key_row)
        layout.addRow(note)
        layout.addRow(self.tr("Model"), model_row)
        layout.addRow(self._model_status)

    def _save(self, text: str, /) -> None:
        _gemini_key.set_gemini_api_key(settings=self._settings, key=text)

    def _on_model_edited(self, *_args: object) -> None:
        name = self._model_combo.currentText().strip()
        if not name or name == self._model:
            return
        if self._apply_model(name):
            self._model = name
        else:
            self._model_combo.setEditText(self._model)

    def _load_models(self) -> None:
        api_key = _gemini_key.get_gemini_api_key(settings=self._settings)
        if api_key is None:
            self._model_status.setText(self.tr("Set the API key first."))
            return
        self._load_button.setEnabled(False)
        self._model_status.setText("")
        self._runner.submit(job=None, run=functools.partial(self._list_models, api_key))

    def _on_models_loaded(self, _job: object, outcome: object, /) -> None:
        self._load_button.setEnabled(True)
        if isinstance(outcome, _automation.GeminiAuthError):
            self._model_status.setText(self.tr("Gemini rejected the API key."))
            return
        if isinstance(outcome, BaseException) or not isinstance(outcome, list):
            logger.warning("Loading Gemini models failed: {}", type(outcome).__name__)
            self._model_status.setText(self.tr("Could not load the model list."))
            return
        current = self._model_combo.currentText()
        self._model_combo.clear()
        self._model_combo.addItems([str(name) for name in outcome])
        self._model_combo.setEditText(current)
