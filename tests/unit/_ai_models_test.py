from __future__ import annotations

from labelme import _ai_models


def test_gemini_is_offered_as_text_model() -> None:
    assert ("gemini", "Gemini (cloud)") in _ai_models.AI_TEXT_MODEL_OPTIONS


def test_is_cloud_model() -> None:
    assert _ai_models.is_cloud_model(model_name="gemini")
    assert not _ai_models.is_cloud_model(model_name="sam3:latest")


def test_local_text_options_exclude_cloud_models() -> None:
    names = {name for name, _ in _ai_models.LOCAL_AI_TEXT_MODEL_OPTIONS}
    assert "gemini" not in names
    assert {"sam3:latest", "yoloworld:latest"} <= names


def test_gemini_is_not_an_ai_assist_model() -> None:
    assert _ai_models.find_ai_assist_model_option(model_name="gemini") is None
