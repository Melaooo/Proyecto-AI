"""Evalúa un checkpoint en el split de test (o val) y guarda métricas y figuras.

Uso: python -m src.evaluate --ckpt checkpoints/resnet18_finetune/best.pt [--split test]

Salida en results/<nombre>/: metrics.json, per_class.csv, confusion_matrix.png,
errors.png (ejemplos mal clasificados), predictions.csv
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score)
from torch.utils.data import DataLoader

from src.data import CLASSES, ROOT, HandCropDataset, build_transforms
from src.models import build_model
from src.train import get_device


@torch.no_grad()
def latency_ms(model, device, img_size, n=50):
    """Tiempo medio por imagen (batch 1), modo inferencia."""
    model.eval()
    x = torch.randn(1, 3, img_size, img_size, device=device)
    for _ in range(10):
        model(x)
    if device.type == "mps":
        torch.mps.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        model(x)
    if device.type == "mps":
        torch.mps.synchronize()
    return (time.perf_counter() - t) / n * 1000


def plot_confusion(cm, path, title):
    cm_norm = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                        color="white" if cm_norm[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_title(title)
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_errors(df, path, n=24, seed=0):
    err = df[df["label"] != df["pred"]]
    err = err.sample(n=min(n, len(err)), random_state=seed)
    cols = 8
    rows = max(1, int(np.ceil(len(err) / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.9))
    for ax in np.array(axes).flat:
        ax.axis("off")
    for ax, r in zip(np.array(axes).flat, err.itertuples()):
        ax.imshow(Image.open(ROOT / r.path))
        ax.set_title(f"R:{r.label}\nP:{r.pred}", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(args):
    device = get_device()
    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt["config"]
    model = build_model({**cfg, "pretrained": False}, len(CLASSES))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    ds = HandCropDataset(args.split, build_transforms(False, cfg["img_size"]))
    dl = DataLoader(ds, batch_size=256, num_workers=4)
    preds, probs = [], []
    with torch.no_grad():
        for x, _ in dl:
            logits = model(x.to(device))
            probs.append(logits.softmax(1).cpu())
            preds.append(logits.argmax(1).cpu())
    y = ds.labels.numpy()
    p = torch.cat(preds).numpy()
    conf = torch.cat(probs).max(1).values.numpy()

    out_dir = ROOT / "results" / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    df = ds.df[["path", "label", "user_id", "hand_px"]].copy()
    df["pred"] = [CLASSES[i] for i in p]
    df["confidence"] = conf
    df.to_csv(out_dir / f"predictions_{args.split}.csv", index=False)

    # Precisión por tamaño de mano en la imagen (proxy de distancia a la cámara)
    bins = pd.qcut(df["hand_px"], 3, labels=["pequeña (lejos)", "mediana", "grande (cerca)"])
    by_size = df.assign(ok=df["label"] == df["pred"]).groupby(bins, observed=True)["ok"].mean().round(4).to_dict()

    metrics = {
        "name": cfg["name"], "split": args.split, "n": int(len(y)), "best_epoch": ckpt["epoch"],
        "accuracy": accuracy_score(y, p),
        "balanced_accuracy": balanced_accuracy_score(y, p),
        "macro_precision": precision_score(y, p, average="macro"),
        "macro_recall": recall_score(y, p, average="macro"),
        "macro_f1": f1_score(y, p, average="macro"),
        "params": sum(t.numel() for t in model.parameters()),
        "encoder_dim": model.encoder.out_dim,
        "latency_ms_mps": latency_ms(model, device, cfg["img_size"]),
        "latency_ms_cpu": latency_ms(model.cpu(), torch.device("cpu"), cfg["img_size"]),
        "accuracy_by_hand_size": by_size,
    }
    json.dump(metrics, open(out_dir / f"metrics_{args.split}.json", "w"), indent=2, ensure_ascii=False)

    report = classification_report(y, p, target_names=CLASSES, output_dict=True)
    pd.DataFrame(report).T.round(4).to_csv(out_dir / f"per_class_{args.split}.csv")
    cm = confusion_matrix(y, p, labels=range(len(CLASSES)))
    plot_confusion(cm, out_dir / f"confusion_matrix_{args.split}.png", f"{cfg['name']} ({args.split})")
    plot_errors(df, out_dir / f"errors_{args.split}.png")

    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()}, indent=2,
                     ensure_ascii=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", default="test", choices=["val", "test"])
    main(p.parse_args())
