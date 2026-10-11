# Proyecto 24 — Reconocimiento de gestos de mano (HaGRIDv2)

Inteligencia Artificial 2026. **Hito 1: clasificación de gestos + encoder reutilizable.**

## 1. Entorno

Probado en macOS (Apple M1 Pro, 16 GB) con Python 3.12 y aceleración MPS (también corre en CUDA o CPU).

```bash
cd proyecto24
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
