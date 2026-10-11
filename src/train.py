"""Entrena un clasificador de gestos sobre recortes de manos.

Uso: python -m src.train --config configs/resnet18_finetune.yaml

Selección de modelo: mejor macro-F1 en validación (early stopping). El test NO se usa aquí.
Guarda en checkpoints/<nombre>/: best.pt (modelo completo), encoder.pt (solo encoder), history.csv
"""

import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from src.data import CLASSES, ROOT, HandCropDataset, build_transforms
from src.models import build_model


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, preds, losses = [], [], []
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    for x, y in loader:
        logits = model(x.to(device))
        losses.append(loss_fn(logits, y.to(device)).item())
        preds.append(logits.argmax(1).cpu())
        ys.append(y)
    y, p = torch.cat(ys).numpy(), torch.cat(preds).numpy()
    return y, p, sum(losses) / len(y)


def main(cfg: dict):
    set_seed(cfg["seed"])
    device = get_device()
    out_dir = ROOT / "checkpoints" / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(cfg, open(out_dir / "config.yaml", "w"))

    train_ds = HandCropDataset("train", build_transforms(True, cfg["img_size"]))
    val_ds = HandCropDataset("val", build_transforms(False, cfg["img_size"]))
    g = torch.Generator().manual_seed(cfg["seed"])
    loader_kw = dict(batch_size=cfg["batch_size"], num_workers=cfg["num_workers"], persistent_workers=True)
    train_dl = DataLoader(train_ds, shuffle=True, generator=g, drop_last=True, **loader_kw)
    val_dl = DataLoader(val_ds, shuffle=False, **loader_kw)

    model = build_model(cfg, len(CLASSES)).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"{cfg['name']}: {sum(p.numel() for p in model.parameters()):,} parámetros "
          f"({sum(p.numel() for p in params):,} entrenables) en {device}")

    optimizer = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"] * len(train_dl))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smoothing", 0.0))

    best_f1, bad_epochs, history = -1.0, 0, []
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        if cfg.get("freeze_encoder", False):
            model.encoder.eval()  # BatchNorm congelado también
        t0, run_loss, n = time.time(), 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            run_loss += loss.item() * len(y)
            n += len(y)

        y_val, p_val, val_loss = predict(model, val_dl, device)
        val_f1 = f1_score(y_val, p_val, average="macro")
        val_acc = (y_val == p_val).mean()
        history.append({"epoch": epoch, "train_loss": run_loss / n, "val_loss": val_loss,
                        "val_acc": val_acc, "val_macro_f1": val_f1, "time_s": time.time() - t0})
        print(f"época {epoch:2d} | train_loss {run_loss / n:.3f} | val_loss {val_loss:.3f} | "
              f"val_acc {val_acc:.3f} | val_F1 {val_f1:.3f} | {time.time() - t0:.0f}s", flush=True)

        if val_f1 > best_f1:
            best_f1, bad_epochs = val_f1, 0
            torch.save({"model": model.state_dict(), "config": cfg, "classes": CLASSES, "epoch": epoch}, out_dir / "best.pt")
            torch.save({"encoder": model.encoder.state_dict(), "config": cfg, "out_dim": model.encoder.out_dim},
                       out_dir / "encoder.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= cfg["patience"]:
                print(f"Early stopping (sin mejora en {cfg['patience']} épocas)")
                break

    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    print(f"Mejor val macro-F1: {best_f1:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    main(yaml.safe_load(open(args.config)))
