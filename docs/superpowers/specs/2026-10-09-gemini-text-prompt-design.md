# Gemini como modelo de AI Text Prompt — diseño

Fecha: 2026-10-09 · Rama: `feat/gemini-text-prompt` · Estado: borrador para revisión

## Contexto y objetivo

labelme ejecuta hoy AI Text Prompt solo con modelos locales vía `osam` (SAM3, YOLO-World). Queremos añadir **Gemini (cloud)** como un modelo más del selector de AI Text Prompt para:

1. Etiquetar objetos o clases que los modelos locales reconocen mal (objetos raros, descripciones complejas como "tornillo oxidado").
2. Servir de base para un futuro **pre-etiquetado por lotes** (proyecto B, con su propio diseño), que reutilizará `GeminiSession`.

### Fuera de alcance

- Claude u otros proveedores. La interfaz común de proveedores se extraerá al añadir el segundo.
- Pre-etiquetado por lotes (proyecto B).
- Usar Gemini en AI Assist con puntos o caja. Ese modo sigue siendo exclusivo de SAM.
- Refinar las cajas de Gemini con SAM.

## Decisiones acordadas

| Tema | Decisión |
|---|---|
| Proveedor | Gemini primero |
| Integración | Gemini como otro modelo de AI Text Prompt (`AI_TEXT_MODEL_OPTIONS`) |
| API key | Se introduce en Ajustes y se guarda en QSettings, **no** en `~/.labelmerc` ni en el repo |
| Formas | Respeta el selector de formato actual: rectángulo → `box_2d`; polígono/máscara → máscara de Gemini |
| HTTP | REST con `urllib` de la biblioteca estándar, sin nuevas dependencias |
| Ejecución | Inferencia en `QThread` (también para los modelos locales) |

## Datos de la API de Gemini

Fuente: https://ai.google.dev/gemini-api/docs/image-understanding (consultada el 2026-10-09).

- Endpoint: `POST https://generativelanguage.googleapis.com/v1beta/interactions`, con la clave en la cabecera `x-goog-api-key`.
- Detección: `box_2d = [ymin, xmin, ymax, xmax]`, normalizado de 0 a 1000.
- Segmentación: lista JSON de `{box_2d, mask, label}`, donde `mask` es un **polígono `[[x, y], ...]` normalizado de 0 a 1000**.
- Para segmentar se recomienda `generation_config.thinking_level = "minimal"`.
- Modelo de los ejemplos: `gemini-3.8-flash` (será el valor por defecto, configurable).
- Gemini no devuelve puntuación de confianza.

El cuerpo exacto de la petición a `interactions` se verificará contra la documentación al implementar `GeminiSession`, con un test que fije el formato.

## Componentes

### Nuevos

**`labelme/_automation/_gemini_prompt.py`** — funciones puras, sin red ni Qt:

- `build_prompt(texts: list[str], want_masks: bool) -> str`: pide JSON estricto `[{"box_2d": [...], "mask": [[x, y], ...], "label": "<una de las etiquetas>"}]`. Solo pide `mask` si `want_masks`.
- `parse_response(text: str, texts: list[str], image_size: tuple[int, int]) -> tuple[boxes, scores, labels, masks | None]`:
  - Acepta JSON puro o envuelto en bloque ```` ```json ````.
  - Convierte `[ymin, xmin, ymax, xmax]` de 0–1000 a `[x1, y1, x2, y2]` en píxeles de la imagen original y lo recorta al tamaño de la imagen.
  - Descarta las entradas con `label` fuera de `texts` (comparación sin distinguir mayúsculas y quitando espacios) y las cajas degeneradas (área 0).
  - Rasteriza `mask` a `NDArray[bool]` del tamaño de la imagen con `skimage.draw.polygon`. Si `mask` falta o es inválida en una entrada, se usa la caja como máscara rectangular.
  - `scores = 1.0` para todas las detecciones.
  - Devuelve el mismo contrato que `get_bboxes_from_texts`: `labels` son índices en `texts`.
  - Lanza `GeminiResponseError` si el JSON es inválido o no es una lista.

**`labelme/_automation/_gemini_session.py`** — `GeminiSession`:

- Constructor: `GeminiSession(*, model: str, api_key: str, timeout: float = 60.0, post=<urllib por defecto>)`. `post` se inyecta para poder probar sin red.
- `model_name` devuelve `"gemini"`, para que `_app.py` reutilice la sesión igual que con `OsamSession`.
- `detect(*, image, image_id, texts, want_masks) -> (boxes, scores, labels, masks)`:
  - Si el lado mayor supera 1536 px, reduce la imagen; luego la codifica en JPEG y en base64.
  - Arma la petición con `build_prompt`, envía el POST y parsea la respuesta con `parse_response` usando el tamaño **original**. Las coordenadas son normalizadas, así que el reescalado no afecta.
- Errores con nombre propio (subclases de `GeminiError`): `GeminiAuthError` (401/403), `GeminiQuotaError` (429), `GeminiNetworkError` (URLError o timeout) y `GeminiResponseError`.
- La API key nunca aparece en logs, mensajes ni `repr`.

**`labelme/_gemini_key.py`** — helper de la clave:

- `get_gemini_api_key() -> str | None` y `set_gemini_api_key(key: str | None) -> None`, sobre `QSettings` (la misma organización y aplicación que el Window State).
- `has_accepted_gemini_privacy() -> bool` y `accept_gemini_privacy() -> None`.

### Modificados

- **`labelme/_ai_models.py`**: añadir `("gemini", "Gemini (cloud)")` a `AI_TEXT_MODEL_OPTIONS` y una función `is_cloud_model(model_name) -> bool`.
- **`labelme/_automation/_text_detection.py`**: `propose_shapes_from_texts` y `get_bboxes_from_texts` aceptan `OsamSession | GeminiSession`. Con `GeminiSession` se llama a `detect(..., want_masks=shape_type in MASK_REQUIRED_SHAPE_TYPES)`. El resto del pipeline (existentes, NMS, supresión, `shapes_from_detections`, grupos) no cambia.
- **`labelme/_app.py`**:
  - `_ensure_ai_model_ready`: para modelos cloud no descarga nada. Comprueba la clave (y si falta, muestra un diálogo con botón "Abrir Ajustes" y devuelve `False`) y el aviso de privacidad (si es la primera vez, muestra el diálogo; si se cancela, devuelve `False`).
  - `_submit_ai_prompt`: crea `GeminiSession` u `OsamSession` según el modelo y ejecuta `propose_shapes_from_texts` en un trabajador `QThread`. Mientras corre, la barra de estado muestra "Consultando %s…" y el botón de enviar queda deshabilitado. Al terminar, en el hilo de la interfaz:
    - si `self._image_path` cambió, descarta el resultado;
    - si no, aplica `apply_default_flags` y llama a `_commit_shapes` como hoy.
  - Se mantienen los mensajes existentes (`MaskOutputUnavailableError` y `_on_inference_failed`).
- **`labelme/_config/_schema.py`** y la configuración por defecto: nuevo ajuste `ai.gemini_model` (texto, por defecto `gemini-3.8-flash`) en el grupo "AI assist".
- **`labelme/_widgets/_settings_dialog.py`**: campo "Gemini API key" en el grupo "AI assist", enmascarado y con botón "Borrar". Lee y escribe con el helper de QSettings y **no** genera Overrides en `~/.labelmerc`. Debe seguir editable aunque los Settings Controls estén deshabilitados por `--config` YAML, igual que AI Models (no edita la Config).
- **Traducciones**: textos nuevos en los catálogos, incluido `es`. Se regeneran con `tools/update_translate.py`.
- **Changelog**: fragmento en `changelog.d/` según `changelog.d/README.md`.

## Flujo de datos

1. El usuario escribe `tornillo oxidado, tuerca`, elige "Gemini (cloud)" y pulsa enviar.
2. `_submit_ai_prompt` resuelve `shape_type`. La Prompt Compatibility ya está garantizada porque Gemini solo figura en la lista de texto.
3. `_ensure_ai_model_ready("gemini")` comprueba la clave y el aviso de privacidad.
4. Se crea el trabajador con una copia RGB de la imagen, `image_path`, `texts`, `shape_type` y los parámetros actuales (umbrales, `polygon_detail`, `image_size`, formas existentes).
5. En el trabajador: `propose_shapes_from_texts` → `GeminiSession.detect` → POST → `parse_response` → NMS, supresión y formas.
6. En el hilo de la interfaz: se comprueba que la imagen no cambió y se ejecutan `apply_default_flags` y `_commit_shapes`, lo que permite deshacer.

Notas:

- `score_threshold` no filtra a Gemini (`score = 1.0`); `iou_threshold` sí se aplica.
- Las formas existentes que se le pasan al trabajador son una copia tomada al enviar.

## Errores

| Situación | Comportamiento |
|---|---|
| Sin API key | Diálogo "Configura tu API key de Gemini en Ajustes" con botón Abrir Ajustes; no hay red |
| Aviso de privacidad cancelado | No se envía nada |
| 401/403 | `_on_inference_failed("API key de Gemini inválida")` |
| 429 | `_on_inference_failed("Límite de Gemini alcanzado, intenta más tarde")` |
| Sin red o timeout | `_on_inference_failed("No se pudo contactar a Gemini")` |
| Respuesta inválida | `_on_inference_failed("Respuesta inesperada de Gemini")`; el texto crudo solo va al log en DEBUG |
| 0 detecciones | Barra de estado: "Gemini no encontró objetos" |

En todos los casos las anotaciones existentes quedan intactas, como fija `test_text_prompt_error_preserves_annotation`.

## Pruebas

Sin red ni clave real: el `post` de `GeminiSession` se reemplaza en los tests y QSettings usa un directorio temporal.

**Unitarias**

- `build_prompt`: incluye todas las etiquetas y pide `mask` solo con `want_masks=True`.
- `parse_response`:
  - conversión de coordenadas y recorte;
  - bloque ```` ```json ````;
  - etiquetas desconocidas descartadas;
  - cajas degeneradas descartadas;
  - polígono → máscara (área esperada) y máscara ausente → rectángulo;
  - JSON inválido → `GeminiResponseError`.
- `GeminiSession`:
  - cabecera `x-goog-api-key`, modelo y `thinking_level`;
  - reducción de imágenes mayores de 1536 px;
  - mapeo de HTTP 401, 403 y 429 y de timeout a sus errores;
  - la clave no aparece en `repr` ni en los mensajes.
- Helper de la clave: lectura y escritura en QSettings; `~/.labelmerc` no cambia.
- `AI_TEXT_MODEL_OPTIONS` incluye Gemini; `is_cloud_model`.

**E2E (Gemini simulado)**

- Texto "dog" + Gemini + formato rectángulo → rectángulos; con polígono → polígonos.
- Sin clave → diálogo y cero llamadas a `post`.
- Error de Gemini → las anotaciones se conservan y se muestra el mensaje.
- Cambio de imagen durante la llamada → resultado descartado.
- Botón deshabilitado durante la inferencia y rehabilitado al terminar.
- El aviso de privacidad aparece una sola vez.
- Los tests existentes de SAM3 y YOLO-World siguen pasando con el trabajador en `QThread`.

**Manual**

- Con una API key real, en `examples/tutorial/apc2016_obj3.jpg`, en rectángulo y en polígono, comparando con SAM3.

## Riesgos

- **Formato de `interactions`**: la API es `v1beta` y puede cambiar. Se mitiga aislando el armado y el parseo en `_gemini_prompt.py` y fijándolos con tests.
- **Precisión de los polígonos de Gemini**: es menor que la de SAM. Es aceptable para este alcance; el refinado con SAM queda fuera.
- **Mover la inferencia a un `QThread`** cambia el orden de los eventos en los tests e2e existentes; habrá que ajustarlos esperando la señal de fin (`qtbot.waitSignal`).
- **Privacidad**: la imagen sale del equipo. Se mitiga con el aviso explícito y recordando la elección.
