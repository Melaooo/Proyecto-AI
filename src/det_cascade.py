"""Hito 2 (Persona 2): cascada detector de manos + clasificador del Hito 1.

En simple: en vez de que un solo modelo encuentre la mano Y diga el gesto, se hace en dos pasos.
Primero el detector de manos marca dónde hay manos (sin saber qué gesto es). Después cada mano
se recorta y el clasificador del Hito 1 (que ya era muy bueno con recortes) dice qué gesto es.

Pregunta: ¿conviene detectar el gesto directamente (det_gestures_hito1_ft) o detectar la mano
y luego clasificar el recorte con el modelo del Hito 1?

Pasos por imagen:
  1. Detector de manos (1 clase) -> cajas + score_det (score >= 0.05, NMS, máx. 20).
  2. Cada caja se recorta IGUAL que en el Hito 1 (scripts/make_crops.py): cuadrado centrado,
     lado = 1.3 x max(w, h) en píxeles, 128x128, normalización ImageNet.
  3. Clasificador del Hito 1 (resnet18_finetune) -> p(clase).
  4. Detección final: caja del detector, clase = argmax, score = score_det x p(clase).
  5. Evaluación con src.det.evaluate_detections (la misma función que el detector directo).

El umbral de operación (precision/recall/F1) se elige en VALIDACIÓN, igual que en det_evaluate.
Las cajas se evalúan en coordenadas normalizadas de la imagen original (el IoU no cambia al
escalar los ejes, así que es comparable con la evaluación en el lienzo).

Uso: python -m src.det_cascade [--det_ckpt checkpoints/det/det_hands_hito1/best.pt]
                               [--cls_ckpt checkpoints/resnet18_finetune/best.pt] [--name cascade]

Salida en results/det/cascade/: metrics_test.json, predictions_test.json, examples_test.png
"""

import argparse
import json

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.data import CLASSES, ROOT, build_transforms
from src.det import (HagridDetDataset, canvas_xyxy_to_original, collate, evaluate_detections,
                     load_detector, predict)
from src.det_evaluate import plot_examples
from src.models import build_model
from src.train import get_device

MARGIN, CROP_SIZE = 1.3, 128          # mismos valores que scripts/make_crops.py


# Carga el clasificador de gestos del Hito 1 (ResNet18 con fine tuning) ya entrenado.
def load_classifier(path, device):
    ckpt = torch.load(path, map_location="cpu")
    model = build_model({**ckpt["config"], "pretrained": False}, len(CLASSES))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()


# Recorta la mano igual que en el Hito 1, para que el clasificador vea recortes parecidos
# a los que vio cuando se entrenó (si no, rendiría peor).
def crop_hand(img, box):
    """box: xyxy normalizada a la imagen original -> recorte cuadrado 128x128 (como en el Hito 1)."""
    W, H = img.size
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2 * W, (y1 + y2) / 2 * H
    side = max((x2 - x1) * W, (y2 - y1) * H) * MARGIN
    sq = (round(cx - side / 2), round(cy - side / 2), round(cx + side / 2), round(cy + side / 2))
    return img.crop(sq).resize((CROP_SIZE, CROP_SIZE), Image.BILINEAR)   # fuera de la imagen: negro


@torch.no_grad()
def run_cascade(split, detector, classifier, device, num_workers=4, batch=256):
    """Devuelve (items del split, predicciones en xyxy original con clases de gesto)."""
    # Paso 1: el detector de manos busca manos en todas las imágenes del split
    ds = HagridDetDataset(split, mode="hands")
    dl = DataLoader(ds, batch_size=32, num_workers=num_workers, collate_fn=collate)
    det_preds = predict(detector, dl, device)                     # cajas en el lienzo 288x384

    # Paso 2: imagen por imagen, recortar cada mano encontrada y clasificar su gesto
    tf = build_transforms(False, CROP_SIZE)
    out = []
    for it, p in zip(ds.items, det_preds):
        # las cajas salen en la imagen redimensionada (288x384); se pasan a la foto original
        boxes = canvas_xyxy_to_original(p["boxes"], it["width"], it["height"])
        if len(boxes) == 0:                                       # no encontró manos
            out.append({"boxes": np.zeros((0, 4)), "scores": np.zeros(0), "labels": np.zeros(0, int)})
            continue
        img = Image.open(ROOT / "data/images" / it["folder"] / f"{it['image_id']}.jpg").convert("RGB")
        x = torch.stack([tf(crop_hand(img, b)) for b in boxes]).to(device)
        probs = torch.cat([classifier(x[i:i + batch]).softmax(1) for i in range(0, len(x), batch)]).cpu()
        p_cls, cls = probs.max(1)                                 # gesto más probable y su probabilidad
        # score final = (qué tan seguro está de que hay mano) x (qué tan seguro está del gesto)
        out.append({"boxes": boxes, "scores": p["scores"] * p_cls.numpy(), "labels": cls.numpy()})
    return ds.items, out


def gts_original(items):
    """GT en xyxy normalizada a la imagen original, con la clase de GESTO (no 'hand')."""
    return [(np.array([[x, y, x + w, y + h] for x, y, w, h in it["boxes"]]).reshape(-1, 4),
             np.array([CLASSES.index(l) for l in it["labels"]])) for it in items]


def export_predictions(items, preds):
    """Mismo formato que src/det_evaluate.py (para que det_analysis lea ambos sistemas igual)."""
    rows = []
    for it, p in zip(items, preds):
        rows.append({
            "image_id": it["image_id"], "folder": it["folder"], "user_id": it["user_id"],
            "width": it["width"], "height": it["height"],
            "gt_boxes": [[x, y, x + w, y + h] for x, y, w, h in it["boxes"]], "gt_labels": it["labels"],
            "pred_boxes": np.round(p["boxes"], 5).tolist(),
            "pred_scores": np.round(p["scores"].astype(float), 4).tolist(),
            "pred_labels": [CLASSES[int(l)] for l in p["labels"]],
        })
    return rows


def main(args):
    device = get_device()
    detector, det_cfg = load_detector(args.det_ckpt, device)
    assert det_cfg["mode"] == "hands", "la cascada usa el detector de manos (mode: hands)"
    classifier = load_classifier(args.cls_ckpt, device)

    # Umbral de operación elegido en validación (máximo F1 a IoU 0.5).
    # Se prueban umbrales 0.05, 0.10, ..., 0.90 y se queda el que da mejor F1. Se hace en val
    # (no en test) para no "hacer trampa" eligiendo con los datos con que se evalúa.
    val_items, val_preds = run_cascade("val", detector, classifier, device, args.num_workers)
    val_gts = gts_original(val_items)
    sweep = []
    for t in np.round(np.arange(0.05, 0.91, 0.05), 2):
        r = evaluate_detections(val_preds, val_gts, CLASSES, op_score=t, iou_thrs=[0.5])
        sweep.append({"score_thr": float(t), "precision": r["precision"], "recall": r["recall"], "F1": r["F1"]})
    op_score = max(sweep, key=lambda r: r["F1"])["score_thr"]

    # Ahora sí, test: misma función de evaluación que el detector directo de Tomás -> comparable
    items, preds = run_cascade(args.split, detector, classifier, device, args.num_workers)
    metrics = evaluate_detections(preds, gts_original(items), CLASSES, op_score=op_score)
    metrics.update({"op_score_from_val": op_score, "val_threshold_sweep": sweep,
                    "det_ckpt": args.det_ckpt, "cls_ckpt": args.cls_ckpt})

    # Guardar métricas, predicciones (las usa det_analysis) y una figura de ejemplos
    out_dir = ROOT / "results/det" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(metrics, open(out_dir / f"metrics_{args.split}.json", "w"), indent=2)
    rows = export_predictions(items, preds)
    json.dump(rows, open(out_dir / f"predictions_{args.split}.json", "w"))
    plot_examples(rows, out_dir / f"examples_{args.split}.png", score_thr=op_score)
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()
                      if k != "val_threshold_sweep"}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--det_ckpt", default="checkpoints/det/det_hands_hito1/best.pt")
    ap.add_argument("--cls_ckpt", default="checkpoints/resnet18_finetune/best.pt")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--name", default="cascade", help="carpeta de salida en results/det/")
    main(ap.parse_args())
