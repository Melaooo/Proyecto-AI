"""Evalúa un detector del Hito 2 en test y exporta sus predicciones.

Uso: python -m src.det_evaluate --ckpt checkpoints/det/det_gestures_hito1/best.pt [--split test]

Salida en results/det/<nombre>/:
  metrics_<split>.json       mAP50, mAP50_95, AP50 por clase, IoU medio de TP y precision/recall/F1 con el
                             umbral de score que maximiza F1 en validación (op_score_from_val)
  predictions_<split>.json   por imagen: image_id, folder, user_id, width, height, GT y predicciones
                             (cajas xyxy normalizadas a la imagen ORIGINAL, score >= 0.05) -> para análisis
  examples_<split>.png       ejemplos de detecciones
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.data import ROOT
from src.det import (HagridDetDataset, canvas_xyxy_to_original, collate, evaluate_detections,
                     gts_from_dataset, load_detector, predict)
from src.train import get_device


def export_predictions(ds, preds):
    rows = []
    for it, p in zip(ds.items, preds):
        W, H = it["width"], it["height"]
        gt = [[x, y, x + w, y + h] for x, y, w, h in it["boxes"]]
        rows.append({
            "image_id": it["image_id"], "folder": it["folder"], "user_id": it["user_id"], "width": W, "height": H,
            "gt_boxes": gt, "gt_labels": [ds.names[l] for l in ds.labels_of(it)],
            "pred_boxes": np.round(canvas_xyxy_to_original(p["boxes"], W, H), 5).tolist(),
            "pred_scores": np.round(p["scores"].astype(float), 4).tolist(),
            "pred_labels": [ds.names[int(l)] for l in p["labels"]],
        })
    return rows


def plot_examples(rows, path, n=12, score_thr=0.5, seed=0):
    rng = np.random.RandomState(seed)
    sel = rng.choice(len(rows), n, replace=False)
    fig, axes = plt.subplots(2, n // 2, figsize=(3 * n // 2, 8.5))
    for ax, i in zip(axes.flat, sel):
        r = rows[i]
        img = Image.open(ROOT / "data/images" / r["folder"] / f"{r['image_id']}.jpg")
        W, H = img.size
        ax.imshow(img)
        ax.axis("off")
        for x1, y1, x2, y2 in r["gt_boxes"]:
            ax.add_patch(patches.Rectangle((x1 * W, y1 * H), (x2 - x1) * W, (y2 - y1) * H,
                                           fill=False, edgecolor="lime", lw=2))
        for (x1, y1, x2, y2), s, l in zip(r["pred_boxes"], r["pred_scores"], r["pred_labels"]):
            if s < score_thr:
                continue
            ax.add_patch(patches.Rectangle((x1 * W, y1 * H), (x2 - x1) * W, (y2 - y1) * H,
                                           fill=False, edgecolor="red", lw=2))
            ax.text(x1 * W, y1 * H - 4, f"{l} {s:.2f}", color="white", fontsize=9,
                    bbox=dict(facecolor="red", alpha=0.8, pad=1, edgecolor="none"))
        ax.set_title("GT: " + ", ".join(r["gt_labels"]), fontsize=10)
    fig.suptitle(f"Verde: cajas reales | Rojo: detecciones (score >= {score_thr})", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(args):
    device = get_device()
    model, cfg = load_detector(args.ckpt, device)

    # Punto de operación: umbral de score que maximiza F1 (IoU 0.5) en VALIDACIÓN
    val_ds = HagridDetDataset("val", cfg["mode"])
    val_dl = DataLoader(val_ds, batch_size=32, num_workers=args.num_workers, collate_fn=collate)
    val_preds, val_gts = predict(model, val_dl, device), gts_from_dataset(val_ds)
    sweep = []
    for t in np.round(np.arange(0.05, 0.91, 0.05), 2):
        r = evaluate_detections(val_preds, val_gts, val_ds.names, op_score=t, iou_thrs=[0.5])
        sweep.append({"score_thr": float(t), "precision": r["precision"], "recall": r["recall"], "F1": r["F1"]})
    op_score = max(sweep, key=lambda r: r["F1"])["score_thr"]

    ds = HagridDetDataset(args.split, cfg["mode"])
    dl = DataLoader(ds, batch_size=32, num_workers=args.num_workers, collate_fn=collate)
    preds = predict(model, dl, device)
    metrics = evaluate_detections(preds, gts_from_dataset(ds), ds.names, op_score=op_score)
    metrics["op_score_from_val"] = op_score
    metrics["val_threshold_sweep"] = sweep
    metrics["params_head"] = sum(p.numel() for p in model.head.parameters())
    metrics["params_total"] = sum(p.numel() for p in model.parameters())
    metrics["best_epoch"] = torch.load(args.ckpt, map_location="cpu", weights_only=False)["epoch"]

    out_dir = ROOT / "results/det" / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(metrics, open(out_dir / f"metrics_{args.split}.json", "w"), indent=2)
    rows = export_predictions(ds, preds)
    json.dump(rows, open(out_dir / f"predictions_{args.split}.json", "w"))
    plot_examples(rows, out_dir / f"examples_{args.split}.png", score_thr=op_score)
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()
                      if k != "val_threshold_sweep"}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--num_workers", type=int, default=4)
    main(ap.parse_args())
