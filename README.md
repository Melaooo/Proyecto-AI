# Proyecto 24 — Reconocimiento de gestos de mano (HaGRIDv2)

Inteligencia Artificial 2026. **Hito 1: clasificación de gestos + encoder reutilizable.**

## 1. Entorno

Probado en macOS (Apple M1 Pro, 16 GB) con Python 3.12 y aceleración MPS (también corre en CUDA o CPU).

```bash
git clone https://github.com/Melaooo/Proyecto-AI.git
cd Proyecto-AI
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> Si se usa el Python de python.org en macOS y aparecen errores `CERTIFICATE_VERIFY_FAILED`,
> ejecutar `export SSL_CERT_FILE=$(python -m certifi)` (lo hace `run_hito1.sh` automáticamente).

## 2. Reproducir todo el Hito 1

```bash
bash run_hito1.sh
```

Descarga datos, genera el subconjunto y los recortes, entrena y evalúa los 4 modelos + baseline,
y genera la tabla comparativa. Tiempo aproximado en M1 Pro: ~15 min de descarga + ~1 h de entrenamiento.

## 3. Datos

Fuente oficial: HaGRIDv2 (https://github.com/hukenovs/hagrid), licencia CC BY-SA 4.0 (variante, ver `license/`).

| Recurso | URL | Tamaño |
| Anotaciones (bboxes, labels, user_id, landmarks) | `https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/annotations_with_landmarks/annotations.zip` | 719 MB |
| Imágenes versión reducida (lado mínimo 512 px) | `https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/hagridv2_512.zip` | 128 GB |

No se descarga el zip completo de imágenes: el servidor admite peticiones HTTP por rango,
así que se lee el índice del zip y se extraen **solo** las imágenes del subconjunto (~2.6 GB).

### Subconjunto (congelado, `seed=42`)

- **Clases (11):** `like, dislike, palm, stop, fist, one, peace, ok, call, mute` + `no_gesture`.
  Se eligieron gestos de una mano, útiles como comandos de interfaz. Se excluyen los gestos de dos manos.
- **Splits oficiales** de HaGRIDv2 (ya separados por `user_id`); se verifica que ningún usuario
  aparece en dos splits.
- Por gesto: 1500 imágenes train / 300 val / 500 test, **máximo 1 imagen por usuario y gesto**
  (más diversidad de personas y fondos; muestreo aleatorio, no se eligen imágenes fáciles).
- `no_gesture`: manos en reposo que aparecen en esas mismas imágenes, submuestreadas al mismo tamaño.
- Total: 16 500 / 3 300 / 5 500 recortes de manos (balanceado, 1500/300/500 por clase).
- Recorte: cuadrado centrado en la bbox oficial, lado = 1.3 × max(ancho, alto), redimensionado a 128×128.

Manifests en `data/splits/`: `images_*.csv` (imágenes), `hands_*.csv` (bboxes), `crops_*.csv` (recortes).

## 4. Comandos por etapa

```bash
python scripts/make_subset.py --seed 42          # anotaciones -> manifests
python scripts/download_images.py --workers 48   # descarga selectiva (reanudable)
python scripts/make_crops.py                     # bboxes -> recortes 128x128

python -m src.baseline_landmarks                 # baseline sin CNN (landmarks + sklearn)
python -m src.train --config configs/resnet18_finetune.yaml
python -m src.evaluate --ckpt checkpoints/resnet18_finetune/best.pt --split test
python -m src.compare                            # tabla + curvas de todos los modelos
```

## 5. Experimentos

| Config | Modelo | Preentrenamiento | Qué se entrena |
|---|---|---|---|
| `baseline_landmarks` | Logistic Regression / Random Forest sobre 21 landmarks | MediaPipe (landmarks de las anotaciones) | clasificador |
| `cnn_residual_scratch` | CNN residual propia (4 bloques, 1.2 M parámetros) | ninguno | todo |
| `resnet18_scratch` | ResNet18 | ninguno | todo |
| `resnet18_linear_probe` | ResNet18 | ImageNet | solo capa lineal (encoder congelado) |
| `resnet18_finetune` | ResNet18 | ImageNet | todo |

Protocolo común: AdamW, cosine LR, label smoothing 0.1, batch 128, aumentos (recorte aleatorio, flip
horizontal, rotación ±15°, color jitter). **Selección por macro-F1 en validación** (early stopping);
test se usa una sola vez al final.

## 6. Salidas

```
checkpoints/<exp>/best.pt      # modelo completo (encoder + cabeza)
checkpoints/<exp>/encoder.pt   # SOLO el encoder -> se reutiliza en Hito 2 y 3
checkpoints/<exp>/history.csv  # curvas por época
results/<exp>/                 # métricas test, matriz de confusión, errores, predicciones
results/comparison.md          # tabla comparativa
results/learning_curves.png
```

Cargar el encoder:

```python
import torch
from src.models import ResNet18Encoder
enc = ResNet18Encoder(pretrained=False)
enc.load_state_dict(torch.load("checkpoints/resnet18_finetune/encoder.pt")["encoder"])
emb = enc(x)  # x: (B, 3, 128, 128) normalizado ImageNet -> (B, 512)
```

---

# Hito 2: detección de manos y gestos reutilizando el encoder

Detector **denso** basado en el notebook de clase *Detección de múltiples objetos con un encoder
preentrenado* (BCCD), con el encoder ResNet18 del Hito 1 en lugar de DINOv2:

```
imagen (letterbox 288x384) -> encoder ResNet18 del Hito 1 (layer3 + layer4, stride 16)
  -> grilla 18x24 x 768 canales -> cabeza conv 1x1 -> 3x3 -> 1x1
  -> por celda [objectness, tx, ty, tw, th, clases] -> score = σ(obj)·p(clase) -> NMS por clase
```

- **Datos:** mismas imágenes y splits por usuario del Hito 1, con **todas** las manos de cada imagen
  (`scripts/make_det_annotations.py` -> `data/splits/det_*.json`; 20 % de las imágenes tiene 2 manos).
- **Pérdida:** asignación del objeto a la celda de su centro; focal loss (objectness) + 1 − GIoU (caja)
  + entropía cruzada (clase), pesos 1 / 2 / 1.
- **Métricas:** AP estilo Pascal VOC por clase -> mAP@0.5 y mAP@[.5:.95]; precision/recall/F1 con el
  umbral de score que maximiza F1 en validación; IoU medio de los verdaderos positivos.
- **Selección:** mejor mAP@0.5 en validación; test se usa una vez.

```bash
python scripts/make_det_annotations.py
python -m src.det_train --config configs/det/det_gestures_hito1.yaml     # prueba: --limit 256 --epochs 1
python -m src.det_evaluate --ckpt checkpoints/det/det_gestures_hito1/best.pt
```

| Config | Qué compara |
|---|---|
| `det_gestures_hito1` | Detector directo de 11 clases, encoder del Hito 1 congelado (base) |
| `det_gestures_imagenet` | Mismo detector con ResNet18 ImageNet congelada |
| `det_gestures_hito1_ft` | Encoder del Hito 1 con layer3/layer4 ajustados |
| `det_hands_hito1` | Detector de manos (1 clase) para la cascada con el clasificador del Hito 1 |

Resultados en test (5 000 imágenes, 6 049 manos; P/R/F1 con el umbral que maximiza F1 en validación):

| Modelo | mAP@0.5 | mAP@[.5:.95] | Precision | Recall | F1 | IoU medio TP |
|---|---|---|---|---|---|---|
| `det_gestures_imagenet` | 0.808 | 0.408 | 0.792 | 0.770 | 0.781 | 0.751 |
| `det_gestures_hito1` | 0.862 | 0.498 | 0.892 | 0.763 | 0.822 | 0.783 |
| **`det_gestures_hito1_ft`** | **0.930** | **0.617** | 0.919 | 0.919 | **0.919** | **0.830** |

Salidas: `checkpoints/det/<exp>/` (best.pt, history.csv) y `results/det/<exp>/`
(metrics_test.json, predictions_test.json en coordenadas de la imagen original, examples_test.png).
El reparto del trabajo y la parte pendiente están en [HITO2_PARA_JAVIER.md](HITO2_PARA_JAVIER.md).

## Hito 2 — Persona 2 (Javier): cascada y análisis por condición

**Idea en simple.** Hay dos formas de detectar gestos:

- **Directo** (Tomás, `det_gestures_hito1_ft`): un solo detector encuentra la mano y dice el gesto.
- **Cascada** (Javier): el detector de manos (`det_hands_hito1`) solo dice *dónde* hay una mano; esa mano se
  recorta igual que en el Hito 1 y el clasificador del Hito 1 (`resnet18_finetune`) dice *qué gesto* es.
  Score final = score del detector × probabilidad del gesto.

Ambos se evalúan con la misma función (`src.det.evaluate_detections`) y el umbral se elige en validación,
así que los números son comparables.

| Archivo | Qué hace |
|---|---|
| `configs/det/det_hands_hito1.yaml` | Detector de manos (1 clase), mismo ajuste que el mejor directo (layer3/4 entrenables) |
| `src/det_cascade.py` | Detector de manos → recorte → clasificador del Hito 1 → métricas en `results/det/cascade/` |
| `src/det_analysis.py` | mAP y recall por distancia a la cámara, 1 vs 2 manos y tipo de fondo → `results/det/analysis/` |
| `src/demo.py` | Foto → manos detectadas → gesto → comando (imágenes en `results/demo/`) |

**Análisis por condición** (`results/det/analysis/resumen.md`):

- *Distancia a la cámara:* tamaño de la mano en píxeles, en tercios (`<83`, `83–117`, `>117` px ≈ lejos / medio / cerca).
- *Una / dos manos:* imágenes con 1 caja real vs 2; además, cuántas segundas manos se pierden.
- *Fondos:* HaGRID no etiqueta el fondo, así que se mide por imagen el brillo, el contraste y la cantidad de
  bordes fuera de las manos (fondo "cargado"); cada medida se parte en tercios.

### Comandos (en orden)

```bash
python -m src.det_train --config configs/det/det_hands_hito1.yaml   # entrena el detector de manos
python -m src.det_evaluate --ckpt checkpoints/det/det_hands_hito1/best.pt
python -m src.det_cascade                                           # necesita det_hands_hito1 entrenado
python -m src.det_analysis                                          # necesita cascade y det_gestures_hito1_ft
python -m src.demo --random 4                                       # opcional: demo con 4 fotos de test
```

### Resultados (test, 5 000 imágenes, 6 049 manos)

| Sistema | mAP@0.5 | mAP@[.5:.95] | Precision | Recall | F1 | IoU medio TP |
|---|---|---|---|---|---|---|
| Detector de manos (1 clase) | 0.967 | 0.634 | 0.956 | 0.953 | 0.955 | 0.837 |
| Directo `det_gestures_hito1_ft` | 0.930 | 0.617 | 0.919 | 0.919 | 0.919 | 0.830 |
| **Cascada** | **0.967** | **0.651** | 0.957 | 0.948 | **0.953** | 0.837 |

La cascada gana sobre todo con manos lejanas (mAP@0.5 0.936 vs 0.822) y en imágenes con 2 manos
(0.953 vs 0.885). En ambos sistemas la mano `no_gesture` es la que más se pierde.

### Correrlo en Windows (con GPU NVIDIA)

En PowerShell, dentro de la carpeta del proyecto (Python 3.12 instalado):

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu132
pip install -r requirements.txt
$env:PYTHONUTF8=1
```

- Si `activate` da error de permisos: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` y repetir.
- `data/images/` (2.6 GB) no está en git: hay que copiarla aparte.
- Si el DataLoader falla, poner `num_workers: 0` en el YAML. Si sale *CUDA out of memory*, bajar `batch_size` a 16.
