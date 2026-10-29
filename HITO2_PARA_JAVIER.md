# Hito 2 — Guía para Javier (y para su Claude)

> **Contexto para Claude:** este archivo lo dejó la "Persona 1" (Tomás) al terminar su parte del Hito 2.
> Explica qué se hizo, qué falta (la parte de la "Persona 2", Javier) y cómo hacerlo. Léelo completo
> antes de tocar código. **Entrega: viernes 30 de octubre, 23:59.** El computador de Javier NO es
> el mismo que el de Tomás (Mac M1 Pro con MPS): revisa la sección 2 para preparar el entorno.

## 0. Qué pide el enunciado (PDF de proyectos, tema 24)

> Hito 2 — Detección con encoder: *Entrenar detección de manos/gestos con las bounding boxes provistas,
> reutilizando el encoder. Evaluar mAP y revisar distancia a cámara, una/dos manos y fondos diversos.*

Métricas esperadas por la pauta general: IoU; precision/recall; AP o mAP (AP50 y mAP@[.5:.95]);
análisis por tamaño, oclusión o dificultad. Informe + código reproducible (ver pauta en el PDF).

## 1. Qué está hecho (Persona 1)

Detector **denso** que sigue el notebook de clase `Clase_Deteccion_Multiple_BCCD.ipynb`
(grilla de celdas, focal loss + GIoU + CE, NMS, AP estilo VOC), cambiando DINOv2 por **nuestro
encoder ResNet18 del Hito 1** (`checkpoints/resnet18_finetune/encoder.pt`).

| Pieza | Archivo |
|---|---|
| Anotaciones de detección (TODAS las manos de cada imagen, mismos splits por usuario del Hito 1) | `scripts/make_det_annotations.py` → `data/splits/det_{train,val,test}.json` |
| Dataset, modelo, pérdida, decodificación/NMS y **métricas compartidas** | `src/det.py` |
| Entrenamiento (selección por mAP@0.5 en val) | `src/det_train.py` |
| Evaluación en test + exportación de predicciones | `src/det_evaluate.py` |
| Configs | `configs/det/*.yaml` |

**Arquitectura:** imagen en letterbox 288×384 → encoder ResNet18 (layer3 + layer4 reescalada, stride 16)
→ grilla 18×24 de 768 canales → cabeza conv (1×1 → 3×3 → 1×1) → por celda `[objectness, tx, ty, tw, th, clases]`.
Encoder congelado (salvo en el experimento `_ft`).

**Experimentos de la Persona 1** (resultados en `results/det/<exp>/metrics_test.json`):

| Experimento | Pregunta |
|---|---|
| `det_gestures_hito1` | Detector directo de gestos (11 clases) con el encoder del Hito 1 congelado — **modelo base** |
| `det_gestures_imagenet` | ¿Reutilizar el encoder del Hito 1 ayuda frente a ResNet18 ImageNet? |
| `det_gestures_hito1_ft` | ¿Ajustar layer3/layer4 del encoder mejora la localización? |

**Modelo elegido: `det_gestures_hito1_ft`** (mAP@0.5 = 0.930 en test). Detalle en la sección 6.

## 2. Preparar el entorno en el computador de Javier

```bash
git clone https://github.com/Melaooo/Proyecto-AI.git
cd Proyecto-AI
python -m venv .venv               # Python 3.11 o 3.12
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

- `torch` de `requirements.txt` instala la versión CPU en Windows/Linux. **Si tiene GPU NVIDIA**, instalar
  torch con CUDA según https://pytorch.org/get-started/locally/ (el código usa `cuda` si existe, si no `mps`, si no `cpu`).
- **Sin GPU** (solo CPU) entrenar el detector toma varias horas: usar **Google Colab con GPU**
  (subir el repo + `data/`, o clonar y descargar los datos ahí con los comandos de abajo).
- En Windows, si el DataLoader falla, poner `num_workers: 0` en el YAML.
- macOS con Python de python.org: si aparece `CERTIFICATE_VERIFY_FAILED`, `export SSL_CERT_FILE=$(python -m certifi)`.

**Datos** (no están en git; se regeneran idénticos porque el subconjunto usa semilla fija):

```bash
# anotaciones oficiales (719 MB)
curl -L -o data/raw/annotations.zip https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/annotations_with_landmarks/annotations.zip
unzip -q data/raw/annotations.zip -d data/raw          # Windows: Expand-Archive
python scripts/make_subset.py --seed 42                 # ya están los CSV en git; vuelve a generar lo mismo
python scripts/download_images.py --workers 48          # ~2.6 GB, ~15 min; descarga solo el subconjunto
python scripts/make_crops.py                            # recortes del Hito 1 (no es obligatorio para el Hito 2)
python scripts/make_det_annotations.py                  # det_*.json (ya están en git)
```

Atajo si Tomás le pasa por Drive la carpeta `data/images/` (2.6 GB): copiarla a `data/images/` y saltarse la descarga.

## 3. Tareas de Javier (Persona 2): cascada + análisis

### 3.1 Detector de manos (1 clase)

```bash
python -m src.det_train --config configs/det/det_hands_hito1.yaml          # prueba rápida: --limit 256 --epochs 1
python -m src.det_evaluate --ckpt checkpoints/det/det_hands_hito1/best.pt
```

Mismo modelo que el mejor directo (encoder del Hito 1 con layer3/layer4 ajustados) pero `mode: hands`
(toda mano es clase 0). Deja `results/det/det_hands_hito1/predictions_test.json`.

**Tiempo:** en el Mac M1 Pro el mismo setup tomó ~27 min (12 épocas × 135 s). Con GPU NVIDIA debería ser
menos; en CPU son horas → usar Colab. Si no alcanza el tiempo, bajar `epochs` a 6 en el YAML
(el directo ya estaba en 0.92 de mAP@0.5 val en la época 6) y decirlo en el informe.

### 3.2 Cascada: detector de manos + clasificador del Hito 1 (crear `src/det_cascade.py`)

Para cada imagen de test:
1. Tomar las cajas del detector de manos (`predictions_test.json`, coordenadas xyxy normalizadas a la imagen original).
2. Recortar cada caja **igual que en el Hito 1** (`scripts/make_crops.py`: cuadrado centrado, lado = 1.3 × max(w, h),
   128×128) y normalizar con `src.data.build_transforms(False, 128)`.
3. Clasificar con el modelo del Hito 1: `checkpoints/resnet18_finetune/best.pt`
   (cargar como en `src/evaluate.py`: `build_model({**cfg, "pretrained": False}, 11)` + `load_state_dict`).
4. Detección final = caja del detector, clase = argmax del clasificador, **score = score_det × p(clase)**.
5. Evaluar con **la misma función** que el directo para que sea comparable:
   `src.det.evaluate_detections(preds, gts, CLASSES)` con `preds` = lista de dicts `{'boxes','scores','labels'}`
   (numpy, xyxy) y `gts` = lista de `(boxes xyxy, labels)`; ambos en coordenadas de la imagen original
   (el IoU no cambia por escalar ejes, así que es válido).
6. Guardar `results/det/cascade/metrics_test.json` y `predictions_test.json` con el mismo formato que `src/det_evaluate.py`.

Pregunta experimental: **¿detectar directamente el gesto o detectar la mano y clasificar el recorte?**
(la cascada usa el encoder dos veces: en el detector y en el clasificador).

### 3.3 Análisis pedido por el enunciado (crear `src/det_analysis.py`)

Usar los `predictions_test.json` de **los dos sistemas** (`det_gestures_hito1_ft`, que es el mejor directo, y
`cascade`) y reportar
mAP50 (y recall a score 0.5) por condición, siempre con `evaluate_detections`:

| Condición | Cómo dividir el test |
|---|---|
| **Distancia a la cámara** | Tamaño de la mano en px de la imagen original: `max(w·W, h·H)`. Tercios como en el Hito 1: `< 83`, `83–117`, `> 117` px (≈ lejos / medio / cerca). Para AP por tamaño: filtrar GT del rango y descartar las predicciones cuyo tamaño cae fuera del rango que no calzan con un GT (criterio COCO). Más simple y aceptable: **recall por rango** + AP por rango con las imágenes cuya(s) mano(s) caen en ese rango. |
| **Una / dos manos** | Imágenes con 1 caja GT vs 2 cajas GT (~20 % tiene 2: la segunda casi siempre es `no_gesture`). mAP en cada subconjunto + cuántas segundas manos se pierden. |
| **Fondos diversos** | HaGRID no trae etiqueta de fondo → proxies por imagen: **brillo medio** (luminancia), **contraste** (desv. estándar) y **complejidad del fondo** (densidad de bordes Sobel/Canny **fuera** de las cajas). Tercios de cada proxy → mAP por tercio. Mostrar 2–3 imágenes de ejemplo de cada tercio. |

Además: figura con aciertos y errores típicos de cada condición (falsos positivos, manos perdidas, clase equivocada).

## 4. Reparto del informe

| Sección | Responsable |
|---|---|
| Introducción y objetivo del Hito 2 | Juntos |
| Dataset de detección (todas las manos, letterbox, splits) | Tomás |
| Arquitectura del detector, asignación, pérdidas, NMS, por qué reutilizar el encoder | Tomás |
| Protocolo (hiperparámetros, selección por val mAP50, métricas: IoU, AP VOC, mAP@[.5:.95]) | Tomás |
| Resultados: encoder Hito 1 vs ImageNet vs fine tuning, curvas, AP por clase | Tomás |
| Cascada vs detector directo | Javier |
| Análisis por distancia, una/dos manos y fondos + ejemplos cualitativos | Javier |
| Conclusiones, limitaciones, recursos externos | Juntos (revisión cruzada) |

## 5. Reglas para que todo sea comparable

- **No** cambiar `src/det.py` (métricas y decodificación compartidas). Si hace falta algo, agregar funciones nuevas.
- Test se usa una sola vez por modelo; elegir umbrales/configuración con `val`.
- Semilla 42; guardar configs y logs en `logs/det/`.
- Mantener el README principal actualizado con los comandos nuevos (sección Hito 2).

## 6. Resultados de la Persona 1 (test, 5 000 imágenes, 6 049 manos, 11 clases)

| Modelo | Encoder | Entrenables | mAP@0.5 | mAP@[.5:.95] | Precision | Recall | F1 | IoU medio TP |
|---|---|---|---|---|---|---|---|---|
| `det_gestures_imagenet` | ResNet18 ImageNet, congelado | 0.79 M | 0.808 | 0.408 | 0.792 | 0.770 | 0.781 | 0.751 |
| `det_gestures_hito1` | Hito 1, congelado | 0.79 M | 0.862 | 0.498 | 0.892 | 0.763 | 0.822 | 0.783 |
| **`det_gestures_hito1_ft`** | Hito 1, layer3/4 ajustados | 11.3 M | **0.930** | **0.617** | 0.919 | 0.919 | **0.919** | **0.830** |

Precision/recall/F1 con el umbral de score que maximiza F1 en validación (0.25–0.30, en `metrics_test.json`).

AP@0.5 por clase del mejor modelo: like 0.933, dislike 0.936, palm 0.941, stop 0.956, fist 0.950, one 0.895,
peace 0.905, ok 0.939, call 0.959, mute 0.963, **no_gesture 0.857** (la más baja en los tres modelos; son manos en
reposo, a menudo al costado del cuerpo, ver `examples_test.png`; conviene revisarla en el análisis de errores).

Lecturas principales:
- **Reutilizar el encoder del Hito 1 ayuda:** con todo lo demás igual, +5.4 puntos de mAP@0.5 y +9.0 de
  mAP@[.5:.95] frente a ImageNet. El encoder ya "sabe" de manos.
- **Ajustar layer3/4 ayuda aún más** (+6.8 mAP@0.5, +11.9 mAP@[.5:.95]): el encoder se entrenó con recortes de
  128 px centrados en la mano y aquí ve imágenes completas donde la mano mide ~60 px (mediana) en un lienzo de 288×384; adaptarlo mejora sobre todo la
  localización (IoU medio de TP 0.78 → 0.83).
- Curvas: `results/det/curvas_detectores.png`. Ejemplos: `results/det/<exp>/examples_test.png`.
