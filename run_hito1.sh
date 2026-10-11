#!/usr/bin/env bash
# Reproduce el Hito 1 completo: datos -> baseline -> entrenamiento -> evaluación -> comparación.
# Uso: bash run_hito1.sh   (desde la carpeta proyecto24, con el venv creado)
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
# El Python de python.org en macOS no trae certificados raíz: usar los de certifi
export SSL_CERT_FILE="$($PY -m certifi)"

# 1. Datos (anotaciones 719 MB + ~2.7 GB de imágenes del subconjunto)
if [ ! -d data/raw/annotations ]; then
  mkdir -p data/raw
  curl -L -o data/raw/annotations.zip \
    https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/annotations_with_landmarks/annotations.zip
  unzip -q data/raw/annotations.zip -d data/raw
fi
$PY scripts/make_subset.py --seed 42
$PY scripts/download_images.py --workers 48
$PY scripts/make_crops.py

# 2. Baseline sin CNN (landmarks + scikit-learn)
$PY -m src.baseline_landmarks

# 3. Entrenamiento y evaluación en test
for exp in cnn_residual_scratch resnet18_scratch resnet18_linear_probe resnet18_finetune; do
  $PY -m src.train --config configs/$exp.yaml
  $PY -m src.evaluate --ckpt checkpoints/$exp/best.pt --split test
done

# 4. Tabla comparativa y curvas
$PY -m src.compare
