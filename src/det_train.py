"""Entrena el detector denso del Hito 2.

Uso: python -m src.det_train --config configs/det_gestures_hito1.yaml [--limit 200 --epochs 1]

Selección: mejor mAP@0.5 en validación. Guarda checkpoints/det/<nombre>/best.pt y history.csv.
"""

import argparse
import time

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.data import ROOT
from src.det import (HagridDetDataset, build_detector, collate, detection_loss, encode_targets,
                     evaluate_detections, gts_from_dataset, predict)
from src.train import get_device, set_seed


def main(cfg, limit=None):
    set_seed(cfg["seed"])
    device = get_device()
    out_dir = ROOT / "checkpoints/det" / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(cfg, open(out_dir / "config.yaml", "w"))

    train_ds = HagridDetDataset("train", cfg["mode"], train=True, limit=limit)
    val_ds = HagridDetDataset("val", cfg["mode"], limit=limit and max(limit // 4, 50))
    g = torch.Generator().manual_seed(cfg["seed"])
    kw = dict(batch_size=cfg["batch_size"], num_workers=cfg["num_workers"], collate_fn=collate,
              persistent_workers=cfg["num_workers"] > 0)
    train_dl = DataLoader(train_ds, shuffle=True, generator=g, drop_last=True, **kw)
    val_dl = DataLoader(val_ds, shuffle=False, **kw)
    val_gts = gts_from_dataset(val_ds)

    model = build_detector(cfg).to(device)
    head_params = list(model.head.parameters())
    bb_params = [p for p in model.backbone.parameters() if p.requires_grad]
    groups = [{"params": head_params, "lr": cfg["lr"]}]
    if bb_params:
        groups.append({"params": bb_params, "lr": cfg["lr_backbone"]})
    n_train = sum(p.numel() for p in head_params + bb_params)
    print(f"{cfg['name']}: {n_train:,} parámetros entrenables en {device} | "
          f"{len(train_ds)} imágenes train, {len(val_ds)} val")

    optimizer = torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"] * len(train_dl))

    best, history = -1.0, []
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        t0, tot, parts, n = time.time(), 0.0, {"obj": 0.0, "box": 0.0, "cls": 0.0}, 0
        for x, boxes, _ in train_dl:
            targets = [t.to(device) for t in encode_targets(boxes)]
            optimizer.zero_grad()
            loss, p = detection_loss(model(x.to(device)), targets)
            loss.backward()
            optimizer.step()
            scheduler.step()
            tot += loss.item()
            n += 1
            for k in parts:
                parts[k] += p[k]
        val = evaluate_detections(predict(model, val_dl, device), val_gts, train_ds.names, iou_thrs=[0.5])
        row = {"epoch": epoch, "train_loss": tot / n, **{f"loss_{k}": v / n for k, v in parts.items()},
               "val_mAP50": val["mAP50"], "val_recall": val["recall"],
               "val_precision": val["precision"], "time_s": time.time() - t0}
        history.append(row)
        print(f"época {epoch:2d} | loss {row['train_loss']:.3f} (obj {row['loss_obj']:.3f} box {row['loss_box']:.3f} "
              f"cls {row['loss_cls']:.3f}) | val mAP50 {val['mAP50']:.3f} "
              f"P {val['precision']:.3f} R {val['recall']:.3f} | {row['time_s']:.0f}s", flush=True)
        if val["mAP50"] > best:
            best = val["mAP50"]
            torch.save({"model": model.state_dict(), "config": cfg, "epoch": epoch,
                        "val": {k: float(v) for k, v in val.items()}}, out_dir / "best.pt")
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    print(f"Mejor val mAP50: {best:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, default=None, help="usar solo N imágenes (prueba rápida)")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.epochs:
        cfg["epochs"] = args.epochs
    main(cfg, args.limit)
