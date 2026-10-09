# Gemini en AI Text Prompt — plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Añadir "Gemini (cloud)" como modelo de AI Text Prompt, con la API key guardada en QSettings y la inferencia de texto ejecutándose en segundo plano.

**Architecture:** `GeminiSession` (REST con `urllib`) devuelve el mismo contrato `(boxes, scores, labels, masks)` que la ruta de osam, así que `propose_shapes_from_texts` reutiliza NMS, supresión y construcción de formas sin cambios. Un `TextPromptRunner` (patrón de `SaveWriter`: `QThreadPool` + señal en cola) saca la inferencia del hilo de la interfaz para todos los modelos de texto. La clave vive en el `QSettings` de la ventana (`self._window_state`) y se edita en una página "Gemini" del diálogo de Ajustes.

**Tech Stack:** Python ≥ 3.12, PySide6, numpy, scikit-image, Pillow, pytest + pytest-qt, `urllib` de la biblioteca estándar.

**Spec:** `docs/superpowers/specs/2026-10-09-gemini-text-prompt-design.md`

### Ajustes respecto a la spec (decididos al leer el código)

1. **Máscaras recortadas a la caja**, no a toda la imagen. El pipeline actual (`_shape_builders.py:70-105` y el test e2e `_make_response`) espera una máscara de tamaño `(ymax - ymin + 1, xmax - xmin + 1)` con la caja en enteros.
2. **`ai.gemini_model` solo en `~/.labelmerc`**, sin control en Ajustes. El esquema no tiene un `kind` de texto libre, y añadirlo sería otro proyecto. Se edita con "Open config file as text…".
3. **La clave se edita en una página propia "Gemini"** del diálogo de Ajustes, junto a "AI Models" (mismo mecanismo que `models_widget`), y sigue editable con `--config` YAML.
4. **`GeminiSession` no se cachea**: se crea en cada envío para recoger los cambios de clave o de modelo. Es barata y no tiene estado.
5. **El mensaje "no encontró objetos"** se muestra para cualquier modelo de texto, no solo para Gemini.
6. **Formato real de la API** (verificado el 2026-10-09 en https://ai.google.dev/api/interactions):
   - Petición: `{"model", "input": [{"type": "text", "text"}, {"type": "image", "data", "mime_type"}], "generation_config": {"thinking_level"}, "response_format": {"type": "text", "mime_type": "application/json", "schema"}}`.
   - Texto de salida: `steps[type == "model_output"].content[type == "text"].text`.

## Global Constraints

- Sin dependencias nuevas: HTTP con `urllib.request`; rasterizado con `skimage.draw.polygon` (scikit-image ya es dependencia).
- Endpoint: `https://generativelanguage.googleapis.com/v1beta/interactions`; cabecera `x-goog-api-key`.
- Modelo por defecto: `gemini-3.8-flash`. `thinking_level: "minimal"`. Timeout de 60 s. Lado mayor de la imagen enviada ≤ 1536 px, en JPEG.
- La API key **nunca** se escribe en `~/.labelmerc`, en logs, en mensajes de error ni en `repr`.
- Claves de QSettings: `gemini/apiKey` y `gemini/privacyAccepted`.
- Id del modelo en `AI_TEXT_MODEL_OPTIONS`: `"gemini"`; nombre visible: `"Gemini (cloud)"`.
- Textos de la interfaz en inglés con `self.tr(...)`; la traducción al español va en `labelme/translate/es_ES.ts`.
- Los tests e2e se ejecutan con `LC_ALL=C.UTF-8 LANGUAGE=en` (los mensajes se comparan en inglés).
- Prompt Compatibility: Gemini solo aparece en la lista de texto; nunca en `AI_ASSIST_MODEL_OPTIONS`.
- Lint al final de cada tarea: `uv run ruff format --check && uv run ruff check && uv run ty check --no-progress`.

## Review Focus

1. **Gemini devuelve el polígono relativo a la caja y no a la imagen.** Las máscaras saldrían desplazadas. Se verifica a mano en la Tarea 9, con una nota sobre qué cambiar.
2. **Cambio de imagen mientras Gemini responde.** Las formas no deben aparecer en la imagen nueva. Lo cubre el test `test_gemini_result_discarded_after_image_switch` (Tarea 7).
3. **Pulsar "Run" dos veces seguidas.** No debe haber dos peticiones en paralelo. Lo cubren `test_runner_rejects_submit_while_busy` (Tarea 5) y el botón deshabilitado (Tarea 6).
4. **Clave con espacios o vacía al pegarla.** Debe recortarse; una clave vacía equivale a "sin clave". Lo cubre `test_api_key_is_trimmed_and_blank_removes_it` (Tarea 4).
5. **Cerrar la ventana con una petición en curso.** No debe fallar al cerrar. `closeEvent` llama a `TextPromptRunner.shutdown()`, que espera al hilo (Tarea 6).

______________________________________________________________________

### Task 1: Gemini en la lista de modelos de texto (sin tratarlo como descargable)

**Files:**

- Modify: `labelme/_ai_models.py:60-63`
- Modify: `labelme/_model_manager.py:17,190`
- Modify: `tests/e2e/conftest.py:28,61`
- Test: `tests/unit/_ai_models_test.py` (nuevo), `tests/unit/_model_manager_test.py`

**Interfaces:**

- Produces:

  - `AI_TEXT_MODEL_OPTIONS` (incluye `("gemini", "Gemini (cloud)")`)
  - `CLOUD_AI_TEXT_MODEL_NAMES: frozenset[str]`
  - `LOCAL_AI_TEXT_MODEL_OPTIONS: tuple[tuple[str, str], ...]`
  - `is_cloud_model(*, model_name: str) -> bool`

- [ ] **Step 1: Write the failing tests**

`tests/unit/_ai_models_test.py`:

```python
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
```

Añadir a `tests/unit/_model_manager_test.py`, con los imports que ya use el archivo:

```python
def test_manager_excludes_cloud_models(qapp: QtWidgets.QApplication) -> None:  # noqa: ARG001
    manager = ModelManager()
    assert "gemini" not in manager.models
    assert not manager.is_ready("gemini")
```

Si el archivo ya crea `ModelManager` mediante una fixture, usa esa fixture en lugar de `ModelManager()` directo.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/_ai_models_test.py tests/unit/_model_manager_test.py -k "gemini or cloud" -v`
Expected: FAIL (`AttributeError: ... is_cloud_model` / gemini ausente).

- [ ] **Step 3: Implement**

En `labelme/_ai_models.py`, reemplazar el bloque de `AI_TEXT_MODEL_OPTIONS` por:

```python
AI_TEXT_MODEL_OPTIONS: Final[tuple[tuple[str, str], ...]] = (
    ("sam3:latest", "SAM3 (smart)"),
    ("yoloworld:latest", "YOLO-World (fast)"),
    ("gemini", "Gemini (cloud)"),
)

# Cloud models run remotely: nothing to download, but they need an API key.
CLOUD_AI_TEXT_MODEL_NAMES: Final[frozenset[str]] = frozenset({"gemini"})

LOCAL_AI_TEXT_MODEL_OPTIONS: Final[tuple[tuple[str, str], ...]] = tuple(
    option
    for option in AI_TEXT_MODEL_OPTIONS
    if option[0] not in CLOUD_AI_TEXT_MODEL_NAMES
)


def is_cloud_model(*, model_name: str) -> bool:
    return model_name in CLOUD_AI_TEXT_MODEL_NAMES
```

En `labelme/_model_manager.py`: importar `LOCAL_AI_TEXT_MODEL_OPTIONS` en lugar de `AI_TEXT_MODEL_OPTIONS` y usar `+ list(LOCAL_AI_TEXT_MODEL_OPTIONS)` en la línea 190.

En `tests/e2e/conftest.py`: importar `LOCAL_AI_TEXT_MODEL_OPTIONS` y usarlo en la línea 61 (`names.update(name for name, _ in LOCAL_AI_TEXT_MODEL_OPTIONS)`), para que el fixture no intente cachear un archivo de modelo para "gemini".

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/_ai_models_test.py tests/unit/_model_manager_test.py tests/unit/_config -v`
Expected: PASS. `_config` sigue pasando: `ai.text_model: gemini` pasa a ser un valor válido.

- [ ] **Step 5: Commit**

```bash
git add labelme/_ai_models.py labelme/_model_manager.py tests/e2e/conftest.py tests/unit/_ai_models_test.py tests/unit/_model_manager_test.py
git commit -m "feat(ai): list Gemini as a cloud AI Text Prompt model"
```

______________________________________________________________________

### Task 2: Prompt y parseo de la respuesta de Gemini (funciones puras)

**Files:**

- Create: `labelme/_automation/_gemini_errors.py`
- Create: `labelme/_automation/_gemini_prompt.py`
- Test: `tests/unit/_automation/_gemini_prompt_test.py`

**Interfaces:**

- Produces:

  - Errores: `GeminiError(Exception)`, `GeminiAuthError`, `GeminiQuotaError`, `GeminiNetworkError` y `GeminiResponseError`, todos subclases de `GeminiError`.
  - `build_prompt(*, texts: list[str], want_masks: bool) -> str`
  - `build_response_schema(*, want_masks: bool) -> dict[str, Any]`
  - `parse_response(*, text: str, texts: list[str], image_size: tuple[int, int], want_masks: bool) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.int32], list[NDArray[np.bool_]] | None]`. `image_size` es `(width, height)`; las cajas salen en píxeles enteros `[x1, y1, x2, y2]` y las máscaras recortadas a la caja.

- [ ] **Step 1: Write the failing tests**

`tests/unit/_automation/_gemini_prompt_test.py`:

````python
from __future__ import annotations

import json

import numpy as np
import pytest

from labelme._automation._gemini_errors import GeminiResponseError
from labelme._automation._gemini_prompt import build_prompt
from labelme._automation._gemini_prompt import build_response_schema
from labelme._automation._gemini_prompt import parse_response


def test_build_prompt_lists_every_label_and_asks_masks_only_when_wanted() -> None:
    prompt = build_prompt(texts=["rusty screw", "nut"], want_masks=False)
    assert '"rusty screw"' in prompt
    assert '"nut"' in prompt
    assert "mask" not in prompt
    assert "mask" in build_prompt(texts=["nut"], want_masks=True)


def test_response_schema_requires_mask_only_when_wanted() -> None:
    assert "mask" not in build_response_schema(want_masks=False)["items"]["properties"]
    assert "mask" in build_response_schema(want_masks=True)["items"]["required"]


def test_parse_converts_normalized_yxyx_to_pixel_xyxy() -> None:
    boxes, scores, labels, masks = parse_response(
        text='[{"box_2d": [100, 200, 500, 600], "label": "nut"}]',
        texts=["screw", "nut"],
        image_size=(1000, 500),
        want_masks=False,
    )
    np.testing.assert_array_equal(boxes, [[200, 50, 600, 250]])
    np.testing.assert_array_equal(scores, [1.0])
    np.testing.assert_array_equal(labels, [1])
    assert masks is None


def test_parse_accepts_fenced_json_and_loose_label_case() -> None:
    text = '```json\n[{"box_2d": [0, 0, 1000, 1000], "label": " Nut "}]\n```'
    boxes, _, labels, _ = parse_response(
        text=text, texts=["nut"], image_size=(10, 10), want_masks=False
    )
    np.testing.assert_array_equal(boxes, [[0, 0, 9, 9]])
    np.testing.assert_array_equal(labels, [0])


def test_parse_drops_unknown_labels_and_bad_boxes() -> None:
    text = json.dumps(
        [
            {"box_2d": [0, 0, 500, 500], "label": "cat"},
            {"box_2d": [100, 100, 100, 300], "label": "nut"},
            {"box_2d": [1, 2, 3], "label": "nut"},
            {"box_2d": "x", "label": "nut"},
            {"box_2d": [0, 0, True, 10], "label": "nut"},
            "garbage",
        ]
    )
    boxes, scores, labels, _ = parse_response(
        text=text, texts=["nut"], image_size=(100, 100), want_masks=False
    )
    assert boxes.shape == (0, 4)
    assert scores.shape == (0,)
    assert labels.shape == (0,)


def test_parse_rasterizes_polygon_into_box_cropped_mask() -> None:
    text = json.dumps(
        [
            {
                "box_2d": [0, 0, 1000, 1000],
                "label": "nut",
                "mask": [[0, 0], [1000, 0], [0, 1000]],
            }
        ]
    )
    boxes, _, _, masks = parse_response(
        text=text, texts=["nut"], image_size=(101, 101), want_masks=True
    )
    np.testing.assert_array_equal(boxes, [[0, 0, 100, 100]])
    assert masks is not None
    assert masks[0].shape == (101, 101)
    assert 0.4 < masks[0].mean() < 0.6


def test_parse_falls_back_to_box_mask_when_polygon_missing_or_invalid() -> None:
    text = json.dumps(
        [
            {"box_2d": [0, 0, 500, 500], "label": "nut"},
            {"box_2d": [0, 0, 500, 500], "label": "nut", "mask": [[1, 2]]},
        ]
    )
    _, _, _, masks = parse_response(
        text=text, texts=["nut"], image_size=(100, 100), want_masks=True
    )
    assert masks is not None
    assert len(masks) == 2
    for mask in masks:
        assert mask.shape == (51, 51)
        assert mask.all()


@pytest.mark.parametrize("text", ["not json", '{"box_2d": [0, 0, 1, 1]}'])
def test_parse_rejects_non_list_responses(text: str) -> None:
    with pytest.raises(GeminiResponseError):
        parse_response(text=text, texts=["nut"], image_size=(10, 10), want_masks=False)
````

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/_automation/_gemini_prompt_test.py -v`
Expected: FAIL con `ModuleNotFoundError: labelme._automation._gemini_errors`.

- [ ] **Step 3: Implement**

`labelme/_automation/_gemini_errors.py`:

```python
from __future__ import annotations


class GeminiError(Exception):
    pass


class GeminiAuthError(GeminiError):
    pass


class GeminiQuotaError(GeminiError):
    pass


class GeminiNetworkError(GeminiError):
    pass


class GeminiResponseError(GeminiError):
    pass
```

`labelme/_automation/_gemini_prompt.py`:

````python
from __future__ import annotations

import json
import re
from typing import Any
from typing import Final

import numpy as np
import skimage.draw
from numpy.typing import NDArray

from ._gemini_errors import GeminiResponseError

# Gemini normalizes every coordinate to this range, independent of image size.
_NORMALIZED_MAX: Final[float] = 1000.0
_FENCE_RE: Final = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def build_prompt(*, texts: list[str], want_masks: bool) -> str:
    labels = ", ".join(json.dumps(text) for text in texts)
    if want_masks:
        fields = (
            '"box_2d" as [ymin, xmin, ymax, xmax], "mask" as a polygon '
            '[[x, y], ...] outlining the object, and "label"'
        )
    else:
        fields = '"box_2d" as [ymin, xmin, ymax, xmax] and "label"'
    return (
        f"Detect every instance of these objects in the image: {labels}. "
        f"Return a JSON list where each entry has {fields}. "
        "All coordinates are integers normalized to 0-1000 over the whole image. "
        "Each label must be exactly one of the requested names. "
        "Return an empty list if there are none."
    )


def build_response_schema(*, want_masks: bool) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "box_2d": {"type": "array", "items": {"type": "integer"}},
        "label": {"type": "string"},
    }
    required = ["box_2d", "label"]
    if want_masks:
        properties["mask"] = {
            "type": "array",
            "items": {"type": "array", "items": {"type": "integer"}},
        }
        required.append("mask")
    return {
        "type": "array",
        "items": {"type": "object", "properties": properties, "required": required},
    }


def parse_response(
    *,
    text: str,
    texts: list[str],
    image_size: tuple[int, int],
    want_masks: bool,
) -> tuple[
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.int32],
    list[NDArray[np.bool_]] | None,
]:
    width, height = image_size
    body = text.strip()
    if (match := _FENCE_RE.match(body)) is not None:
        body = match.group(1)
    try:
        entries = json.loads(body)
    except json.JSONDecodeError as e:
        raise GeminiResponseError("Gemini response is not valid JSON") from e
    if not isinstance(entries, list):
        raise GeminiResponseError("Gemini response is not a JSON list")

    label_indices = {name.strip().lower(): i for i, name in enumerate(texts)}
    boxes: list[tuple[int, int, int, int]] = []
    labels: list[int] = []
    masks: list[NDArray[np.bool_]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        label = entry.get("label")
        if not isinstance(label, str):
            continue
        label_index = label_indices.get(label.strip().lower())
        if label_index is None:
            continue
        box = _to_pixel_box(entry.get("box_2d"), width=width, height=height)
        if box is None:
            continue
        boxes.append(box)
        labels.append(label_index)
        if want_masks:
            masks.append(
                _rasterize_mask(entry.get("mask"), box=box, width=width, height=height)
            )

    return (
        np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
        np.ones((len(boxes),), dtype=np.float32),
        np.asarray(labels, dtype=np.int32),
        masks if want_masks else None,
    )


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _to_pixel_box(
    value: object, *, width: int, height: int
) -> tuple[int, int, int, int] | None:
    if not (
        isinstance(value, list) and len(value) == 4 and all(map(_is_number, value))
    ):
        return None
    ymin, xmin, ymax, xmax = value

    def scale(v: float, size: int) -> int:
        return int(np.clip(round(v / _NORMALIZED_MAX * size), 0, size - 1))

    x1, y1 = scale(xmin, width), scale(ymin, height)
    x2, y2 = scale(xmax, width), scale(ymax, height)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _rasterize_mask(
    value: object, *, box: tuple[int, int, int, int], width: int, height: int
) -> NDArray[np.bool_]:
    # The shape builders expect masks cropped to the integer box (inclusive).
    x1, y1, x2, y2 = box
    shape = (y2 - y1 + 1, x2 - x1 + 1)
    box_mask = np.ones(shape, dtype=bool)
    if not isinstance(value, list) or len(value) < 3:
        return box_mask
    try:
        points = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return box_mask
    if points.ndim != 2 or points.shape[1] != 2:
        return box_mask
    points = points / _NORMALIZED_MAX * np.array([width, height])
    rows, cols = skimage.draw.polygon(
        points[:, 1] - y1, points[:, 0] - x1, shape=shape
    )
    mask = np.zeros(shape, dtype=bool)
    mask[rows, cols] = True
    return mask if mask.any() else box_mask
````

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/_automation/_gemini_prompt_test.py -v`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add labelme/_automation/_gemini_errors.py labelme/_automation/_gemini_prompt.py tests/unit/_automation/_gemini_prompt_test.py
git commit -m "feat(ai): build Gemini detection prompts and parse their responses"
```

______________________________________________________________________

### Task 3: `GeminiSession` (HTTP, reducción de imagen, errores)

**Files:**

- Create: `labelme/_automation/_gemini_session.py`
- Create: `tests/unit/_automation/_gemini_fakes.py`
- Test: `tests/unit/_automation/_gemini_session_test.py`

**Interfaces:**

- Consumes: Tarea 2 (`build_prompt`, `build_response_schema`, `parse_response` y los errores).

- Produces:

  - `GEMINI_ENDPOINT: Final[str]`
  - `Post = Callable[[str, dict[str, str], bytes, float], bytes]`
  - `_urllib_post(url, headers, body, timeout) -> bytes`. Es un atributo de módulo que los tests e2e reemplazan con `monkeypatch`.
  - `GeminiSession(*, model: str, api_key: str, timeout: float = 60.0, post: Post | None = None)`, con `.model_name == "gemini"` y `.detect(*, image: NDArray[np.uint8], texts: list[str], want_masks: bool)`, que devuelve la misma tupla que `parse_response`.

- [ ] **Step 1: Write the test fakes and failing tests**

`tests/unit/_automation/_gemini_fakes.py`:

```python
from __future__ import annotations

import json
from typing import Any


class FakePost:
    def __init__(
        self, *, response: bytes | None = None, error: BaseException | None = None
    ) -> None:
        self.calls: list[tuple[str, dict[str, str], dict[str, Any], float]] = []
        self._response = response
        self._error = error

    def __call__(
        self, url: str, headers: dict[str, str], body: bytes, timeout: float
    ) -> bytes:
        self.calls.append((url, headers, json.loads(body), timeout))
        if self._error is not None:
            raise self._error
        assert self._response is not None
        return self._response


def interaction(text: str) -> bytes:
    return json.dumps(
        {
            "steps": [
                {"type": "user_input"},
                {"type": "model_output", "content": [{"type": "text", "text": text}]},
            ]
        }
    ).encode()
```

`tests/unit/_automation/_gemini_session_test.py`:

```python
from __future__ import annotations

import base64
import email.message
import io
import urllib.error

import numpy as np
import PIL.Image
import pytest

from labelme._automation._gemini_errors import GeminiAuthError
from labelme._automation._gemini_errors import GeminiError
from labelme._automation._gemini_errors import GeminiNetworkError
from labelme._automation._gemini_errors import GeminiQuotaError
from labelme._automation._gemini_errors import GeminiResponseError
from labelme._automation._gemini_session import GEMINI_ENDPOINT
from labelme._automation._gemini_session import GeminiSession

from ._gemini_fakes import FakePost
from ._gemini_fakes import interaction

_KEY = "secret-key-123"


def _session(post: FakePost) -> GeminiSession:
    return GeminiSession(model="gemini-3.8-flash", api_key=_KEY, post=post)


def test_detect_sends_key_model_prompt_and_image() -> None:
    post = FakePost(response=interaction('[{"box_2d": [0, 0, 500, 500], "label": "nut"}]'))
    boxes, _, labels, masks = _session(post).detect(
        image=np.zeros((100, 200, 3), dtype=np.uint8), texts=["nut"], want_masks=False
    )
    url, headers, body, timeout = post.calls[0]
    assert url == GEMINI_ENDPOINT
    assert headers["x-goog-api-key"] == _KEY
    assert timeout == 60.0
    assert body["model"] == "gemini-3.8-flash"
    assert body["generation_config"] == {"thinking_level": "minimal"}
    assert body["response_format"]["mime_type"] == "application/json"
    assert body["input"][0]["type"] == "text"
    assert '"nut"' in body["input"][0]["text"]
    assert body["input"][1]["type"] == "image"
    assert body["input"][1]["mime_type"] == "image/jpeg"
    np.testing.assert_array_equal(boxes, [[0, 0, 100, 50]])
    np.testing.assert_array_equal(labels, [0])
    assert masks is None


def test_detect_downscales_large_images_but_keeps_original_coordinates() -> None:
    post = FakePost(response=interaction('[{"box_2d": [0, 0, 1000, 1000], "label": "nut"}]'))
    boxes, _, _, _ = _session(post).detect(
        image=np.zeros((1000, 3000, 3), dtype=np.uint8), texts=["nut"], want_masks=False
    )
    sent = PIL.Image.open(io.BytesIO(base64.b64decode(post.calls[0][2]["input"][1]["data"])))
    assert max(sent.size) == 1536
    np.testing.assert_array_equal(boxes, [[0, 0, 2999, 999]])


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (urllib.error.HTTPError(GEMINI_ENDPOINT, 401, "x", email.message.Message(), None), GeminiAuthError),
        (urllib.error.HTTPError(GEMINI_ENDPOINT, 403, "x", email.message.Message(), None), GeminiAuthError),
        (urllib.error.HTTPError(GEMINI_ENDPOINT, 429, "x", email.message.Message(), None), GeminiQuotaError),
        (urllib.error.HTTPError(GEMINI_ENDPOINT, 500, "x", email.message.Message(), None), GeminiNetworkError),
        (urllib.error.URLError("no route"), GeminiNetworkError),
        (TimeoutError(), GeminiNetworkError),
    ],
)
def test_detect_maps_transport_errors(
    error: BaseException, expected: type[GeminiError]
) -> None:
    with pytest.raises(expected) as exc_info:
        _session(FakePost(error=error)).detect(
            image=np.zeros((10, 10, 3), dtype=np.uint8), texts=["nut"], want_masks=False
        )
    assert _KEY not in str(exc_info.value)


@pytest.mark.parametrize("raw", [b"not json", b"[]", b'{"steps": []}'])
def test_detect_rejects_responses_without_text_output(raw: bytes) -> None:
    with pytest.raises(GeminiResponseError):
        _session(FakePost(response=raw)).detect(
            image=np.zeros((10, 10, 3), dtype=np.uint8), texts=["nut"], want_masks=False
        )


def test_repr_hides_api_key() -> None:
    assert _KEY not in repr(_session(FakePost(response=b"")))
    assert _session(FakePost(response=b"")).model_name == "gemini"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/_automation/_gemini_session_test.py -v`
Expected: FAIL con `ModuleNotFoundError: labelme._automation._gemini_session`.

- [ ] **Step 3: Implement**

`labelme/_automation/_gemini_session.py`:

```python
from __future__ import annotations

import base64
import io
import json
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Final

import numpy as np
import PIL.Image
from loguru import logger
from numpy.typing import NDArray

from ._gemini_errors import GeminiAuthError
from ._gemini_errors import GeminiNetworkError
from ._gemini_errors import GeminiQuotaError
from ._gemini_errors import GeminiResponseError
from ._gemini_prompt import build_prompt
from ._gemini_prompt import build_response_schema
from ._gemini_prompt import parse_response

GEMINI_ENDPOINT: Final[str] = (
    "https://generativelanguage.googleapis.com/v1beta/interactions"
)
# Gemini answers in 0-1000 normalized coordinates, so downscaling the upload
# does not change the boxes it returns; it only cuts upload size and latency.
_MAX_IMAGE_SIDE: Final[int] = 1536

Post = Callable[[str, dict[str, str], bytes, float], bytes]


def _urllib_post(url: str, headers: dict[str, str], body: bytes, timeout: float) -> bytes:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


class GeminiSession:
    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        timeout: float = 60.0,
        post: Post | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._timeout = timeout
        # Resolved at construction so tests can monkeypatch the module function.
        self._post = post if post is not None else _urllib_post

    @property
    def model_name(self) -> str:
        return "gemini"

    def __repr__(self) -> str:
        return f"GeminiSession(model={self._model!r})"

    def detect(
        self, *, image: NDArray[np.uint8], texts: list[str], want_masks: bool
    ) -> tuple[
        NDArray[np.float32],
        NDArray[np.float32],
        NDArray[np.int32],
        list[NDArray[np.bool_]] | None,
    ]:
        height, width = image.shape[:2]
        body = json.dumps(
            {
                "model": self._model,
                "input": [
                    {"type": "text", "text": build_prompt(texts=texts, want_masks=want_masks)},
                    {"type": "image", "data": _encode_jpeg(image), "mime_type": "image/jpeg"},
                ],
                "generation_config": {"thinking_level": "minimal"},
                "response_format": {
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": build_response_schema(want_masks=want_masks),
                },
            }
        ).encode()
        headers = {"Content-Type": "application/json", "x-goog-api-key": self._api_key}
        try:
            raw = self._post(GEMINI_ENDPOINT, headers, body, self._timeout)
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                raise GeminiAuthError("Gemini rejected the API key") from None
            if e.code == 429:
                raise GeminiQuotaError("Gemini quota exceeded") from None
            raise GeminiNetworkError(f"Gemini returned HTTP {e.code}") from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise GeminiNetworkError("Could not reach Gemini") from None
        text = _extract_output_text(raw)
        logger.debug("Gemini response text: {!r}", text)
        return parse_response(
            text=text, texts=texts, image_size=(width, height), want_masks=want_masks
        )


def _encode_jpeg(image: NDArray[np.uint8]) -> str:
    pil_image = PIL.Image.fromarray(image).convert("RGB")
    pil_image.thumbnail((_MAX_IMAGE_SIDE, _MAX_IMAGE_SIDE))
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _extract_output_text(raw: bytes) -> str:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as e:
        raise GeminiResponseError("Gemini returned a non-JSON body") from e
    if not isinstance(payload, dict):
        raise GeminiResponseError("Gemini returned an unexpected body")
    parts = [
        block["text"]
        for step in payload.get("steps", [])
        if isinstance(step, dict) and step.get("type") == "model_output"
        for block in step.get("content", [])
        if isinstance(block, dict)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
    ]
    if not parts:
        raise GeminiResponseError("Gemini response has no text output")
    return "".join(parts)
```

Nota: en el test del timeout, `TimeoutError` es subclase de `OSError`, y `HTTPError` lo es de `URLError`; el orden de los `except` de arriba es el correcto.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/_automation/_gemini_session_test.py -v && uv run ruff format labelme/_automation tests/unit/_automation && uv run ruff check labelme/_automation tests/unit/_automation`
Expected: PASS. Ruff reformatea las líneas largas de los tests; acepta su formato.

- [ ] **Step 5: Commit**

```bash
git add labelme/_automation/_gemini_session.py tests/unit/_automation/_gemini_fakes.py tests/unit/_automation/_gemini_session_test.py
git commit -m "feat(ai): add GeminiSession for cloud text-prompt detection"
```

______________________________________________________________________

### Task 4: Integrar `GeminiSession` en `propose_shapes_from_texts` y el helper de la clave

**Files:**

- Modify: `labelme/_automation/_text_detection.py:27-45,123-130`
- Modify: `labelme/_automation/__init__.py`
- Create: `labelme/_gemini_key.py`
- Modify: `labelme/_config/default_config.yaml:40-43`
- Modify: `labelme/_config/__init__.py:79-84`
- Test: `tests/unit/_automation/_text_detection_test.py`, `tests/unit/_gemini_key_test.py` (nuevo), `tests/unit/_config/` (archivo existente de validación)

**Interfaces:**

- Consumes: `GeminiSession.detect` (Tarea 3).

- Produces:

  - `propose_shapes_from_texts(session: OsamSession | GeminiSession, ...)` (misma firma; el tipo de `session` se amplía).
  - Exportados desde `labelme._automation`: `GeminiSession`, `GeminiError`, `GeminiAuthError`, `GeminiQuotaError`, `GeminiNetworkError` y `GeminiResponseError`.
  - `labelme._gemini_key`: `get_gemini_api_key(*, settings) -> str | None`, `set_gemini_api_key(*, settings, key: str | None) -> None`, `has_accepted_gemini_privacy(*, settings) -> bool` y `accept_gemini_privacy(*, settings) -> None`.
  - Config: `config["ai"]["gemini_model"]`, de tipo `str` y con valor por defecto `"gemini-3.8-flash"`.

- [ ] **Step 1: Write the failing tests**

Añadir a `tests/unit/_automation/_text_detection_test.py`:

```python
import json

from labelme._automation import GeminiSession

from ._gemini_fakes import FakePost
from ._gemini_fakes import interaction


@pytest.mark.parametrize(
    ("shape_type", "expect_mask_request"),
    [("rectangle", False), ("polygon", True)],
)
def test_propose_shapes_from_texts_with_gemini_session(
    shape_type: str, expect_mask_request: bool
) -> None:
    post = FakePost(
        response=interaction(
            json.dumps(
                [
                    {
                        "box_2d": [100, 100, 600, 600],
                        "label": "nut",
                        "mask": [[100, 100], [600, 100], [600, 600], [100, 600]],
                    }
                ]
            )
        )
    )
    shapes = propose_shapes_from_texts(
        session=GeminiSession(model="m", api_key="k", post=post),
        image=np.zeros((200, 200, 3), dtype=np.uint8),
        image_id="img",
        texts=["nut"],
        shape_type=shape_type,
        existing_shapes=[],
        iou_threshold=0.5,
        score_threshold=0.1,
        image_size=(200, 200),
        polygon_detail=60,
    )
    assert [shape.label for shape in shapes] == ["nut"]
    assert shapes[0].shape_type == shape_type
    prompt = post.calls[0][2]["input"][0]["text"]
    assert ("mask" in prompt) is expect_mask_request
```

Usa los imports de `pytest`, `numpy as np` y `propose_shapes_from_texts` que ya tenga el archivo; añade solo los que falten.

`tests/unit/_gemini_key_test.py`:

```python
from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore

from labelme import _gemini_key


def _settings(tmp_path: Path) -> QtCore.QSettings:
    return QtCore.QSettings(str(tmp_path / "s.ini"), QtCore.QSettings.Format.IniFormat)


def test_api_key_round_trip(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    assert _gemini_key.get_gemini_api_key(settings=settings) is None
    _gemini_key.set_gemini_api_key(settings=settings, key="abc")
    assert _gemini_key.get_gemini_api_key(settings=settings) == "abc"
    _gemini_key.set_gemini_api_key(settings=settings, key=None)
    assert _gemini_key.get_gemini_api_key(settings=settings) is None


def test_api_key_is_trimmed_and_blank_removes_it(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _gemini_key.set_gemini_api_key(settings=settings, key="  abc \n")
    assert _gemini_key.get_gemini_api_key(settings=settings) == "abc"
    _gemini_key.set_gemini_api_key(settings=settings, key="   ")
    assert _gemini_key.get_gemini_api_key(settings=settings) is None
    assert "gemini/apiKey" not in settings.allKeys()


def test_privacy_acceptance_persists(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    assert not _gemini_key.has_accepted_gemini_privacy(settings=settings)
    _gemini_key.accept_gemini_privacy(settings=settings)
    assert _gemini_key.has_accepted_gemini_privacy(settings=_settings(tmp_path))
```

Validación de la config: añadir a `tests/unit/_config/api_test.py`, junto a `test_load_config_rejects_invalid_polygon_detail` (el archivo ya importa `json`, `Path`, `pytest` y `_config`):

```python
@pytest.mark.parametrize("value", ["", "   ", 3, None])
def test_load_config_rejects_invalid_gemini_model(
    *, tmp_path: Path, value: object
) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(f"ai:\n  gemini_model: {json.dumps(value)}\n")

    with pytest.raises(ValueError, match=r"ai\.gemini_model must be a non-empty"):
        _config.load_config(config_file=config_file, config_overrides={})


def test_load_config_defaults_gemini_model() -> None:
    config = _config.load_config(config_file=None, config_overrides={})
    assert config["ai"]["gemini_model"] == "gemini-3.8-flash"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/_automation/_text_detection_test.py tests/unit/_gemini_key_test.py tests/unit/_config -v -k "gemini"`
Expected: FAIL (`ImportError: cannot import name 'GeminiSession'`, módulo `_gemini_key` inexistente y falta la validación).

- [ ] **Step 3: Implement**

`labelme/_automation/_text_detection.py`:

1. Añadir el import `from ._gemini_session import GeminiSession`.
2. Cambiar el tipo `session: OsamSession` por `session: OsamSession | GeminiSession` en `propose_shapes_from_texts` y en `get_bboxes_from_texts`.
3. En `propose_shapes_from_texts`, cambiar la primera llamada por:

```python
    boxes, scores, labels, masks = get_bboxes_from_texts(
        session=session,
        image=image,
        image_id=image_id,
        texts=texts,
        want_masks=shape_type in MASK_REQUIRED_SHAPE_TYPES,
    )
```

4. En `get_bboxes_from_texts`, añadir el parámetro `want_masks: bool = True` y, como primera instrucción del cuerpo:

```python
    if isinstance(session, GeminiSession):
        # Gemini only returns masks when asked; asking costs latency and tokens.
        return session.detect(image=image, texts=texts, want_masks=want_masks)
```

`labelme/_automation/__init__.py`: añadir los imports y las entradas en `__all__` (orden alfabético, como el resto):

```python
from ._gemini_errors import GeminiAuthError
from ._gemini_errors import GeminiError
from ._gemini_errors import GeminiNetworkError
from ._gemini_errors import GeminiQuotaError
from ._gemini_errors import GeminiResponseError
from ._gemini_session import GeminiSession
```

`labelme/_gemini_key.py`:

```python
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
    if key is not None and key.strip():
        settings.setValue(_API_KEY, key.strip())
    else:
        settings.remove(_API_KEY)
    settings.sync()


def has_accepted_gemini_privacy(*, settings: QtCore.QSettings) -> bool:
    return bool(settings.value(_PRIVACY_ACCEPTED, False, type=bool))


def accept_gemini_privacy(*, settings: QtCore.QSettings) -> None:
    settings.setValue(_PRIVACY_ACCEPTED, True)
    settings.sync()
```

`labelme/_config/default_config.yaml`, en el bloque `ai:`:

```yaml
ai:
  default: Sam2 (balanced)
  text_model: yoloworld:latest
  gemini_model: gemini-3.8-flash
  suppress_existing_shape_matches: false
```

`labelme/_config/__init__.py`, justo después del bloque de `("ai", "text_model")`:

```python
    if key_path == ("ai", "gemini_model") and (
        not isinstance(value, str) or not value.strip()
    ):
        raise ValueError(
            f"ai.gemini_model must be a non-empty model name, but got {value!r}"
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit -q --numprocesses=auto`
Expected: PASS (toda la suite unitaria; los tests de osam siguen por su ruta sin cambios).

- [ ] **Step 5: Commit**

```bash
git add labelme/_automation labelme/_gemini_key.py labelme/_config tests/unit
git commit -m "feat(ai): route Gemini sessions through text detection and store its key in QSettings"
```

______________________________________________________________________

### Task 5: `TextPromptRunner` (inferencia en segundo plano)

**Files:**

- Create: `labelme/_text_prompt_runner.py`
- Test: `tests/unit/_text_prompt_runner_test.py`

**Interfaces:**

- Produces: `TextPromptRunner(QtCore.QObject)`, con:

  - `finished = Signal(object, object)`: emite `(job, outcome)`, donde `outcome` es el valor devuelto o la excepción lanzada;
  - `is_busy: bool`;
  - `submit(*, job: object, run: Callable[[], object]) -> bool`: devuelve `False` si está ocupado;
  - `shutdown() -> None`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/_text_prompt_runner_test.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/_text_prompt_runner_test.py -v`
Expected: FAIL con `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`labelme/_text_prompt_runner.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/_text_prompt_runner_test.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add labelme/_text_prompt_runner.py tests/unit/_text_prompt_runner_test.py
git commit -m "feat(ai): add a background runner for text-prompt inference"
```

______________________________________________________________________

### Task 6: AI Text Prompt en segundo plano en `MainWindow` (todos los modelos)

**Files:**

- Modify: `labelme/_widgets/_ai_text_to_annotation_widget.py:70-76,154`
- Modify: `labelme/_app.py:218,332,1510-1563,2570`
- Modify: `tests/e2e/ai_text_to_annotation_test.py:112-129`
- Test: `tests/unit/widgets/_ai_text_to_annotation_widget_test.py`, `tests/e2e/ai_text_to_annotation_test.py`

**Interfaces:**

- Consumes: `TextPromptRunner` (Tarea 5).

- Produces:

  - `AiTextToAnnotationWidget.set_running(running: bool, /) -> None`;
  - `MainWindow._text_prompt_runner: TextPromptRunner`;
  - `MainWindow._on_text_prompt_finished(job: _TextPromptJob, outcome: object, /) -> None`;
  - dataclass `_TextPromptJob(image_path: str | None, model_display_name: str, shape_type: str)` en `_app.py`.

- [ ] **Step 1: Write the failing tests**

En `tests/unit/widgets/_ai_text_to_annotation_widget_test.py`, siguiendo cómo construye el widget ese archivo:

```python
def test_set_running_disables_only_the_run_button(qtbot: QtBot) -> None:
    widget = AiTextToAnnotationWidget(on_submit=lambda _: None)
    qtbot.addWidget(widget)
    widget.set_running(True)
    assert not widget._run_button.isEnabled()
    assert widget._text_input.isEnabled()
    widget.set_running(False)
    assert widget._run_button.isEnabled()
```

En `tests/e2e/ai_text_to_annotation_test.py`, cambiar el final de `_run_text_prompt`:

```python
    win._submit_ai_prompt(False)  # noqa: FBT003 -- the slot takes the Qt clicked flag positionally
    qtbot.waitUntil(lambda: not win._text_prompt_runner.is_busy, timeout=5000)
    qtbot.wait(50)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `LC_ALL=C.UTF-8 LANGUAGE=en uv run pytest tests/unit/widgets/_ai_text_to_annotation_widget_test.py tests/e2e/ai_text_to_annotation_test.py -v`
Expected: FAIL (`set_running` y `_text_prompt_runner` no existen).

- [ ] **Step 3: Implement**

`_ai_text_to_annotation_widget.py`:

- Declarar `_run_button: QtWidgets.QToolButton` junto a los otros atributos de clase.
- En `_init_ui`, después de crear `run_button`, añadir `self._run_button = run_button`.
- Añadir el método:

```python
    def set_running(self, running: bool, /) -> None:  # noqa: FBT001 -- mirrors setEnabled
        self._run_button.setEnabled(not running)
        self._run_button.setText(self.tr("Running…") if running else self.tr("Run"))
```

`_app.py`:

1. Imports: `import dataclasses` (si no está) y `from ._text_prompt_runner import TextPromptRunner`.
2. Junto a los otros aliases a nivel de módulo (cerca de `_TextToAnnotationCreateMode`, línea 83):

```python
@dataclasses.dataclass(frozen=True)
class _TextPromptJob:
    image_path: str | None
    model_display_name: str
    shape_type: str
```

3. En `__init__`, justo después de `self._save_writer = SaveWriter(parent=self)` (línea 332):

```python
        self._text_prompt_runner = TextPromptRunner(parent=self)
        self._text_prompt_runner.finished.connect(self._on_text_prompt_finished)
```

4. En `closeEvent`, después de `self._save_writer.shutdown()`: `self._text_prompt_runner.shutdown()`.

5. Reemplazar `_submit_ai_prompt` completo (líneas 1510-1563) por:

```python
    def _submit_ai_prompt(self, _: bool, /) -> None:  # noqa: FBT001 -- submit callback receives the Qt clicked flag
        if self._text_prompt_runner.is_busy:
            return
        create_mode = self._canvas_widgets.canvas.create_mode
        shape_type = _resolve_text_annotation_shape_type(
            create_mode=create_mode,
            ai_output_format=self._ai_annotation.output_format,
        )
        if shape_type is None:
            logger.warning("Unsupported create_mode={!r}", create_mode)
            return

        texts = self._ai_text.get_text_prompt().split(",")

        model_name: str = self._ai_text.get_model_name()
        if not self._ensure_ai_model_ready(model_name):
            return
        session = self._get_text_session(model_name)
        if session is None:
            return

        job = _TextPromptJob(
            image_path=self._image_path,
            model_display_name=self._ai_text.get_model_display_name(),
            shape_type=shape_type,
        )
        run = functools.partial(
            _automation.propose_shapes_from_texts,
            session=session,
            image=_utils.img_qt_to_rgb_arr(self._image),
            image_id=str(hash(self._image_path)),
            texts=texts,
            shape_type=shape_type,
            existing_shapes=list(self._canvas_widgets.canvas.shapes),
            iou_threshold=self._ai_text.get_iou_threshold(),
            score_threshold=self._ai_text.get_score_threshold(),
            image_size=(
                None
                if self._config["canvas"]["allow_out_of_bounds_points"]
                else (self._image.width(), self._image.height())
            ),
            polygon_detail=self._config["mask_polygonization"]["detail"],
        )
        self._ai_text.set_running(True)
        self.show_status_message(
            self.tr("Running %s…") % job.model_display_name, delay=0
        )
        self._text_prompt_runner.submit(job=job, run=run)

    def _get_text_session(
        self, model_name: str, /
    ) -> _automation.OsamSession | _automation.GeminiSession | None:
        if (
            self._text_osam_session is None
            or self._text_osam_session.model_name != model_name
        ):
            self._text_osam_session = _automation.OsamSession(model_name=model_name)
        return self._text_osam_session

    def _on_text_prompt_finished(self, job: _TextPromptJob, outcome: object, /) -> None:
        self._ai_text.set_running(False)
        self.statusBar().clearMessage()
        if job.image_path != self._image_path:
            logger.info("Discarding AI text result for an image that is no longer open")
            return
        if isinstance(outcome, _automation.MaskOutputUnavailableError):
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Mask Output Unavailable"),
                self.tr(
                    "%s only detects bounding boxes and cannot create "
                    "'%s' annotations.\n\n"
                    "Switch the AI Text-to-Annotation model to 'SAM3 (smart)', "
                    "or set the output format to 'Rectangle'."
                )
                % (job.model_display_name, job.shape_type),
            )
            return
        if isinstance(outcome, BaseException):
            logger.opt(exception=outcome).error("AI text inference failed")
            self._on_inference_failed(f"{type(outcome).__name__}: {outcome}")
            return
        shapes = cast(list[Shape], outcome)
        if not shapes:
            self.show_status_message(
                self.tr("%s found no objects.") % job.model_display_name, delay=5000
            )
            return
        apply_default_flags(shapes=shapes, label_flags=self._config["label_flags"])
        self._commit_shapes([*self._canvas_widgets.canvas.shapes, *shapes])
```

El texto del `QMessageBox` es idéntico al original, para no romper su traducción.

- [ ] **Step 4: Run tests to verify they pass**

Run: `LC_ALL=C.UTF-8 LANGUAGE=en uv run pytest tests/unit tests/e2e -q --numprocesses=auto`
Expected: PASS en todo. Si algún test e2e que mire la barra de estado esperaba el comportamiento síncrono, añade el mismo `qtbot.waitUntil(lambda: not win._text_prompt_runner.is_busy, ...)` en su helper.

- [ ] **Step 5: Commit**

```bash
git add labelme/_app.py labelme/_widgets/_ai_text_to_annotation_widget.py tests/unit/widgets tests/e2e/ai_text_to_annotation_test.py
git commit -m "feat(ai): run AI Text Prompt inference off the GUI thread"
```

______________________________________________________________________

### Task 7: Gemini en `MainWindow` (clave, privacidad, errores)

**Files:**

- Modify: `labelme/_app.py` (`_refresh_model_availability`, `_ensure_ai_model_ready`, `_get_text_session`, `_on_text_prompt_finished`)
- Test: `tests/e2e/ai_text_gemini_test.py` (nuevo)

**Interfaces:**

- Consumes:

  - `_ai_models.is_cloud_model` y `CLOUD_AI_TEXT_MODEL_NAMES` (Tarea 1);
  - `labelme._gemini_key` (Tarea 4);
  - `_automation.GeminiSession` y los errores (Tareas 3 y 4);
  - `_on_text_prompt_finished` y `_get_text_session` (Tarea 6).

- Produces:

  - `MainWindow._ensure_gemini_ready() -> bool`;
  - `MainWindow._open_gemini_settings() -> None`, que en la Tarea 8 abrirá la página de la clave; aquí se limita a `self._open_settings()`.

- [ ] **Step 1: Write the failing e2e tests**

`tests/e2e/ai_text_gemini_test.py`:

```python
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

import pytest
from PySide6 import QtWidgets
from pytestqt.qtbot import QtBot

from labelme import _gemini_key
from labelme._automation import _gemini_session

from .conftest import MainWinFactory
from .conftest import show_window_and_wait_for_imagedata

if TYPE_CHECKING:
    from labelme._app import MainWindow


class _FakePost:
    def __init__(self, *, entries: list[dict[str, Any]] | None = None, error: BaseException | None = None, gate: threading.Event | None = None) -> None:
        self.calls = 0
        self._entries = entries or []
        self._error = error
        self._gate = gate

    def __call__(self, url: str, headers: dict[str, str], body: bytes, timeout: float) -> bytes:
        self.calls += 1
        if self._gate is not None:
            self._gate.wait(5)
        if self._error is not None:
            raise self._error
        text = json.dumps(self._entries)
        return json.dumps({"steps": [{"type": "model_output", "content": [{"type": "text", "text": text}]}]}).encode()


_PERSON = {
    "box_2d": [100, 100, 600, 500],
    "label": "person",
    "mask": [[100, 100], [500, 100], [500, 600], [100, 600]],
}


def _open(main_win: MainWinFactory, qtbot: QtBot, data_path: Path, tmp_path: Path) -> MainWindow:
    win = main_win(
        file_or_dir=str(data_path / "raw/2011_000003.jpg"),
        config_overrides=dict(auto_save=True),
        output_dir=str(tmp_path),
    )
    show_window_and_wait_for_imagedata(qtbot=qtbot, win=win)
    return win


def _configure(win: MainWindow, *, key: str | None = "test-key", privacy: bool = True) -> None:
    _gemini_key.set_gemini_api_key(settings=win._window_state, key=key)
    if privacy:
        _gemini_key.accept_gemini_privacy(settings=win._window_state)


def _run(win: MainWindow, qtbot: QtBot, *, create_mode: str, text: str = "person") -> None:
    win._switch_canvas_mode(edit=False, create_mode=create_mode)
    win._ai_text._text_input.setText(text)
    win._ai_text.set_model_name(model_name="gemini")
    win._submit_ai_prompt(False)  # noqa: FBT003
    qtbot.waitUntil(lambda: not win._text_prompt_runner.is_busy, timeout=5000)
    qtbot.wait(50)


@pytest.mark.gui
@pytest.mark.parametrize("create_mode", ["rectangle", "polygon"])
def test_gemini_creates_shapes(main_win: MainWinFactory, qtbot: QtBot, data_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, create_mode: str) -> None:
    fake = _FakePost(entries=[_PERSON])
    monkeypatch.setattr(_gemini_session, "_urllib_post", fake)
    win = _open(main_win, qtbot, data_path, tmp_path)
    _configure(win)
    _run(win, qtbot, create_mode=create_mode)
    shapes = win._canvas_widgets.canvas.shapes
    assert fake.calls == 1
    assert [s.label for s in shapes] == ["person"]
    assert shapes[0].shape_type == create_mode


@pytest.mark.gui
def test_gemini_without_key_asks_for_settings_and_sends_nothing(main_win: MainWinFactory, qtbot: QtBot, data_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakePost(entries=[_PERSON])
    monkeypatch.setattr(_gemini_session, "_urllib_post", fake)
    prompts: list[str] = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "information",
        lambda _parent, title, *_args, **_kw: prompts.append(title) or QtWidgets.QMessageBox.StandardButton.Cancel,
    )
    win = _open(main_win, qtbot, data_path, tmp_path)
    _configure(win, key=None)
    _run(win, qtbot, create_mode="rectangle")
    assert prompts == ["Gemini API Key Required"]
    assert fake.calls == 0
    assert win._canvas_widgets.canvas.shapes == []


@pytest.mark.gui
def test_gemini_privacy_prompt_once_and_decline_sends_nothing(main_win: MainWinFactory, qtbot: QtBot, data_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakePost(entries=[_PERSON])
    monkeypatch.setattr(_gemini_session, "_urllib_post", fake)
    answers = [QtWidgets.QMessageBox.StandardButton.No, QtWidgets.QMessageBox.StandardButton.Yes]
    asked: list[str] = []

    def question(_parent: object, title: str, *_args: object, **_kw: object) -> QtWidgets.QMessageBox.StandardButton:
        asked.append(title)
        return answers.pop(0)

    monkeypatch.setattr(QtWidgets.QMessageBox, "question", question)
    win = _open(main_win, qtbot, data_path, tmp_path)
    _configure(win, privacy=False)
    _run(win, qtbot, create_mode="rectangle")
    assert fake.calls == 0
    _run(win, qtbot, create_mode="rectangle")
    assert fake.calls == 1
    _run(win, qtbot, create_mode="rectangle", text="person")
    assert asked == ["Send Image to Google?", "Send Image to Google?"]
    assert fake.calls == 2


@pytest.mark.gui
def test_gemini_auth_error_keeps_annotations_and_explains(main_win: MainWinFactory, qtbot: QtBot, data_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import email.message
    import urllib.error

    error = urllib.error.HTTPError(_gemini_session.GEMINI_ENDPOINT, 401, "x", email.message.Message(), None)
    monkeypatch.setattr(_gemini_session, "_urllib_post", _FakePost(error=error))
    win = _open(main_win, qtbot, data_path, tmp_path)
    _configure(win)
    _run(win, qtbot, create_mode="rectangle")
    assert win._canvas_widgets.canvas.shapes == []
    assert "Gemini rejected the API key" in win.statusBar().currentMessage()
    assert "test-key" not in win.statusBar().currentMessage()


@pytest.mark.gui
def test_gemini_result_discarded_after_image_switch(main_win: MainWinFactory, qtbot: QtBot, data_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    gate = threading.Event()
    monkeypatch.setattr(_gemini_session, "_urllib_post", _FakePost(entries=[_PERSON], gate=gate))
    win = _open(main_win, qtbot, data_path, tmp_path)
    _configure(win)
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
```

Notas:

- En el test de privacidad, la tercera ejecución no vuelve a preguntar. Por eso `asked` tiene dos entradas (No y Sí) y `fake.calls` termina en 2.

- Pasa `uv run ruff format tests/e2e/ai_text_gemini_test.py` para que quede con el estilo del repo.

- [ ] **Step 2: Run tests to verify they fail**

Run: `LC_ALL=C.UTF-8 LANGUAGE=en uv run pytest tests/e2e/ai_text_gemini_test.py -v`
Expected: FAIL. Gemini aparece como no disponible en el selector y se intenta usar `OsamSession("gemini")`.

- [ ] **Step 3: Implement in `labelme/_app.py`**

1. Imports: `from . import _gemini_key`.
2. En `_refresh_model_availability`, cambiar la línea del selector de texto por:

```python
        self._ai_text.set_available_models(
            available=available | _ai_models.CLOUD_AI_TEXT_MODEL_NAMES
        )
```

3. Reemplazar `_ensure_ai_model_ready` por:

```python
    def _ensure_ai_model_ready(self, model_name: str, /) -> bool:
        if _ai_models.is_cloud_model(model_name=model_name):
            return self._ensure_gemini_ready()
        if model_name and self._model_manager.is_ready(model_name):
            return True
        self._open_models()
        return False

    def _ensure_gemini_ready(self) -> bool:
        if _gemini_key.get_gemini_api_key(settings=self._window_state) is None:
            answer = QtWidgets.QMessageBox.information(
                self,
                self.tr("Gemini API Key Required"),
                self.tr("Set your Gemini API key in Settings to use Gemini."),
                QtWidgets.QMessageBox.StandardButton.Open
                | QtWidgets.QMessageBox.StandardButton.Cancel,
            )
            if answer == QtWidgets.QMessageBox.StandardButton.Open:
                self._open_gemini_settings()
            return False
        if not _gemini_key.has_accepted_gemini_privacy(settings=self._window_state):
            answer = QtWidgets.QMessageBox.question(
                self,
                self.tr("Send Image to Google?"),
                self.tr(
                    "Gemini runs in the cloud: the current image and the prompt "
                    "are sent to Google each time you run it. Continue?"
                ),
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No,
            )
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                return False
            _gemini_key.accept_gemini_privacy(settings=self._window_state)
        return True

    def _open_gemini_settings(self) -> None:
        self._open_settings()
```

4. En `_get_text_session`, añadir al principio:

```python
        if _ai_models.is_cloud_model(model_name=model_name):
            api_key = _gemini_key.get_gemini_api_key(settings=self._window_state)
            if api_key is None:
                return None
            return _automation.GeminiSession(
                model=self._config["ai"]["gemini_model"], api_key=api_key
            )
```

5. En `_on_text_prompt_finished`, antes de `if isinstance(outcome, BaseException):`:

```python
        if isinstance(outcome, _automation.GeminiError):
            logger.warning("Gemini request failed: {}", type(outcome).__name__)
            self._on_inference_failed(self._describe_gemini_error(outcome))
            return
```

y añadir el método:

```python
    def _describe_gemini_error(self, error: _automation.GeminiError, /) -> str:
        if isinstance(error, _automation.GeminiAuthError):
            return self.tr("Gemini rejected the API key. Check it in Settings.")
        if isinstance(error, _automation.GeminiQuotaError):
            return self.tr("Gemini quota reached. Try again later.")
        if isinstance(error, _automation.GeminiNetworkError):
            return self.tr("Could not reach Gemini.")
        return self.tr("Gemini returned an unexpected response.")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `LC_ALL=C.UTF-8 LANGUAGE=en uv run pytest tests/e2e -q --numprocesses=auto && uv run pytest tests/unit -q --numprocesses=auto`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add labelme/_app.py tests/e2e/ai_text_gemini_test.py
git commit -m "feat(ai): run AI Text Prompt with Gemini, gated by API key and privacy consent"
```

______________________________________________________________________

### Task 8: Página "Gemini" en Ajustes para la API key

**Files:**

- Create: `labelme/_widgets/_gemini_key_widget.py`
- Modify: `labelme/_widgets/__init__.py` (exportar `GeminiKeyWidget` como se exporta `ModelsWidget`)
- Modify: `labelme/_widgets/_settings_dialog.py:774-884`
- Modify: `labelme/_app.py` (`_open_settings`, `_open_gemini_settings`)
- Test: `tests/unit/widgets/_gemini_key_widget_test.py` (nuevo), `tests/unit/widgets/_settings_dialog_test.py`

**Interfaces:**

- Consumes: `labelme._gemini_key` (Tarea 4).

- Produces:

  - `GeminiKeyWidget(*, settings: QtCore.QSettings, parent: QtWidgets.QWidget | None = None)`;
  - `SettingsDialog(..., gemini_widget: QtWidgets.QWidget | None = None, ...)`;
  - `SettingsDialog.show_gemini() -> None`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/widgets/_gemini_key_widget_test.py`:

```python
from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore
from PySide6 import QtWidgets
from pytestqt.qtbot import QtBot

from labelme import _gemini_key
from labelme._widgets import GeminiKeyWidget


def _settings(tmp_path: Path) -> QtCore.QSettings:
    return QtCore.QSettings(str(tmp_path / "s.ini"), QtCore.QSettings.Format.IniFormat)


def test_key_field_is_masked_and_prefilled(qtbot: QtBot, tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _gemini_key.set_gemini_api_key(settings=settings, key="abc")
    widget = GeminiKeyWidget(settings=settings)
    qtbot.addWidget(widget)
    assert widget._key_edit.echoMode() == QtWidgets.QLineEdit.EchoMode.Password
    assert widget._key_edit.text() == "abc"


def test_typing_saves_and_clear_removes(qtbot: QtBot, tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    widget = GeminiKeyWidget(settings=settings)
    qtbot.addWidget(widget)
    widget._key_edit.setText(" new-key ")
    assert _gemini_key.get_gemini_api_key(settings=settings) == "new-key"
    widget._clear_button.click()
    assert widget._key_edit.text() == ""
    assert _gemini_key.get_gemini_api_key(settings=settings) is None
```

En `tests/unit/widgets/_settings_dialog_test.py` (ya importa `QtWidgets`, `QtBot`, `load_config` y `SettingsDialog`), añadir:

```python
def test_gemini_page_stays_enabled_when_settings_are_not_editable(
    *, qtbot: QtBot
) -> None:
    gemini_widget = QtWidgets.QWidget()
    dialog = SettingsDialog(
        config=load_config(config_file=None, config_overrides={}),
        apply_setting=lambda _key_path, _value: True,
        preview_shape_color=lambda _key_path, _value: None,
        open_as_text=lambda: None,
        gemini_widget=gemini_widget,
        settings_editable=False,
    )
    qtbot.addWidget(dialog)
    dialog.show_gemini()
    assert gemini_widget.isEnabled()
    item = dialog._page._navigation.currentItem()
    assert item is not None
    assert item.text() == "Gemini"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/widgets/_gemini_key_widget_test.py tests/unit/widgets/_settings_dialog_test.py -v -k "gemini or key"`
Expected: FAIL (`ImportError: GeminiKeyWidget` y argumento `gemini_widget` desconocido).

- [ ] **Step 3: Implement**

`labelme/_widgets/_gemini_key_widget.py`:

```python
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
```

`labelme/_widgets/__init__.py`: añadir `from ._gemini_key_widget import GeminiKeyWidget` y la entrada en `__all__`, igual que `ModelsWidget`.

`labelme/_widgets/_settings_dialog.py`:

- Añadir el parámetro `gemini_widget: QtWidgets.QWidget | None = None,` justo después de `models_widget`.
- Después del bloque `if models_widget is not None: ...` (líneas 819-824), añadir:

```python
        self._gemini_index = len(groups)
        if gemini_widget is not None:
            # Like AI Models, the key is not part of the Config, so it stays
            # editable even when command-line Settings lock the Config pages.
            gemini_group = QtWidgets.QGroupBox(self.tr("Gemini"))
            QtWidgets.QVBoxLayout(gemini_group).addWidget(gemini_widget)
            groups.append(
                (self.tr("Gemini"), new_icon("phosphor/sparkle.svg"), gemini_group)
            )
```

- Junto a `show_models`:

```python
    def show_gemini(self) -> None:
        self._page.clear_search()
        self._page._navigation.setCurrentRow(self._gemini_index)
```

`labelme/_app.py`:

- En `_open_settings`, pasar `gemini_widget=GeminiKeyWidget(settings=self._window_state),` justo después de `models_widget=...` e importar `GeminiKeyWidget` desde `._widgets`, como `ModelsWidget`.
- Cambiar `_open_gemini_settings` a:

```python
    def _open_gemini_settings(self) -> None:
        self._open_settings()
        assert self._settings_dialog is not None
        self._settings_dialog.show_gemini()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `LC_ALL=C.UTF-8 LANGUAGE=en uv run pytest tests/unit tests/e2e -q --numprocesses=auto`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add labelme/_widgets labelme/_app.py tests/unit/widgets
git commit -m "feat(settings): add a Gemini page to store the API key outside the config"
```

______________________________________________________________________

### Task 9: Traducciones, lint completo y verificación con Gemini real

**Files:**

- Modify: `labelme/translate/*.ts` y `*.qm` (generados) y `labelme/translate/es_ES.ts` (traducciones al español)

- Create (solo si se abre PR): `changelog.d/<PR>.added.md`

- [ ] **Step 1: Regenerar los catálogos**

Run: `uv run tools/update_translate.py`
Expected: aparecen en todos los `.ts` las cadenas nuevas sin traducir:

- "Running…", "Running %s…", "%s found no objects."

- "Gemini API Key Required", "Set your Gemini API key in Settings to use Gemini."

- "Send Image to Google?" y su texto

- los cuatro mensajes de error de Gemini

- "Paste your Gemini API key", "Gemini API key", "Clear", "API key", la nota de la página y "Gemini"

- [ ] **Step 2: Traducir al español en `labelme/translate/es_ES.ts`**

Para cada `<message>` nuevo, poner la traducción en `<translation>` (sin `type="unfinished"`):

| Source | Translation |
|---|---|
| Running… | Ejecutando… |
| Running %s… | Ejecutando %s… |
| %s found no objects. | %s no encontró objetos. |
| Gemini API Key Required | Se requiere la API key de Gemini |
| Set your Gemini API key in Settings to use Gemini. | Configura tu API key de Gemini en Ajustes para usar Gemini. |
| Send Image to Google? | ¿Enviar la imagen a Google? |
| Gemini runs in the cloud: the current image and the prompt are sent to Google each time you run it. Continue? | Gemini se ejecuta en la nube: la imagen actual y el texto se envían a Google cada vez que lo usas. ¿Continuar? |
| Gemini rejected the API key. Check it in Settings. | Gemini rechazó la API key. Revísala en Ajustes. |
| Gemini quota reached. Try again later. | Se alcanzó el límite de Gemini. Inténtalo más tarde. |
| Could not reach Gemini. | No se pudo contactar con Gemini. |
| Gemini returned an unexpected response. | Gemini devolvió una respuesta inesperada. |
| Paste your Gemini API key | Pega tu API key de Gemini |
| Gemini API key | API key de Gemini |
| Clear | Borrar |
| API key | API key |
| Stored only on this computer, outside the labelme config file. Running Gemini sends the current image to Google. | Se guarda solo en este equipo, fuera del archivo de configuración de labelme. Al usar Gemini, la imagen actual se envía a Google. |
| Gemini | Gemini |

Luego: `uv run tools/update_translate.py` (recompila los `.qm`) y `uv run tools/update_translate.py --check`.
Expected: `--check` sin errores. Si `--check` exige traducciones completas en todos los idiomas, revisa cómo resuelven otros PRs las cadenas nuevas (`git log -p -- labelme/translate/de_DE.ts | head -200`) y haz lo mismo.

- [ ] **Step 3: Lint y suite completos**

Run:

```bash
uv run ruff format --check && uv run ruff check && uv run ty check --no-progress && uv run typos
git ls-files "*.toml" | xargs uv run taplo fmt --check
git ls-files "*.md" | xargs uv run mdformat --check
git ls-files "*.yml" "*.yaml" | xargs uv run yamlfix --check
LC_ALL=C.UTF-8 LANGUAGE=en uv run pytest -v tests/ --numprocesses=auto
```

Expected: todo en verde.

- [ ] **Step 4: Commit**

```bash
git add labelme/translate
git commit -m "i18n: translate the Gemini AI Text Prompt strings"
```

- [ ] **Step 5: Verificación manual con una API key real (la hace el usuario o el agente con su clave)**

1. Si `libxcb-cursor0` no está instalada: `sudo apt install libxcb-cursor0`.
2. `uv run labelme examples/tutorial/apc2016_obj3.jpg`.
3. Ajustes → Gemini → pegar la clave → cerrar.
4. Modo rectángulo; en AI Text-to-Annotation elegir "Gemini (cloud)", escribir un objeto visible de la imagen y pulsar Run. Aceptar el aviso de privacidad. **Esperado:** la interfaz sigue respondiendo mientras dice "Ejecutando Gemini (cloud)…" y luego aparecen rectángulos bien colocados.
5. Repetir en modo polígono. **Esperado:** los polígonos siguen el contorno del objeto.
   - **Si los polígonos salen pequeños y pegados a la esquina superior izquierda**, Gemini está devolviendo la máscara relativa a la caja. En ese caso, en `_rasterize_mask` reemplazar la conversión por `points = points / _NORMALIZED_MAX * np.array([x2 - x1 + 1, y2 - y1 + 1])` y quitar la resta de `x1`/`y1`. Después ajustar `test_parse_rasterizes_polygon_into_box_cropped_mask` y la frase "over the whole image" de `build_prompt`.
6. Con la clave mal escrita, **esperado:** "Gemini rechazó la API key. Revísala en Ajustes." y ninguna anotación cambia.
7. `grep -i gemini ~/.labelmerc`. **Esperado:** no aparece la clave.

- [ ] **Step 6: Changelog (al abrir el PR)**

Cuando exista el número de PR, crear `changelog.d/<PR>.added.md` con una sola línea:

```markdown
Add Gemini (cloud) as an AI Text-to-Annotation model; its API key is stored per machine, outside the config file.
```

y commitearlo en la misma rama.
