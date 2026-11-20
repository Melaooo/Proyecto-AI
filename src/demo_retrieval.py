"""Demo final del proyecto: imagen -> mano localizada -> gesto -> ejemplos parecidos + comando.

Por cada mano que encuentra el detector de manos del Hito 2:
  1. se recorta igual que en el Hito 1 y se calcula su embedding (resnet18_ft, largo 1);
  2. se buscan sus 10 vecinos más parecidos en la galería de train (similitud coseno);
  3. el gesto es el más votado entre esos 10 vecinos (kNN) y de ahí sale el comando.
La figura muestra la imagen con las manos y, por cada mano, su recorte y los 5 vecinos más parecidos.

Uso:
  python -m src.demo_retrieval --image demo/Foto_Javier1.jpg    # una foto propia (imagen externa)
  python -m src.demo_retrieval --random 3                       # 3 imágenes al azar de test
Guarda en results/demo_retrieval/<nombre>_retrieval.jpg
"""

import argparse
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from src.data import CLASSES, ROOT
from src.demo import COMMANDS, letterbox
from src.det import canvas_xyxy_to_original, decode_predictions, load_detector
from src.det_cascade import crop_hand
from src.retrieval import K_VOTE, load_encoder, search, split_embeddings
from src.retrieval_sistema import embed_crops
from src.train import get_device

DET_CKPT = ROOT / "checkpoints/det/det_hands_hito1/best.pt"
OP_FROM = ROOT / "results/det/cascade/metrics_test.json"     # umbral del detector elegido en val (Hito 2)
ENCODER = "resnet18_ft"
K_SHOW = 5


@torch.no_grad()
def find_hands(img, detector, device, score_thr):
    raw = detector(letterbox(img)[None].to(device)).float().cpu()
    p = decode_predictions(raw, score_thr=0.05)[0]
    boxes = canvas_xyxy_to_original(p["boxes"], *img.size)
    keep = p["scores"] >= score_thr
    return boxes[keep], p["scores"][keep]


def describe(idx, sims, g_df):
    """Votación kNN entre los K_VOTE vecinos -> (gesto, votos)."""
    labels = [g_df.label.iloc[int(i)] for i in idx[:K_VOTE]]
    votes = {c: labels.count(c) for c in CLASSES}
    gesture = max(CLASSES, key=lambda c: (votes[c], -labels.index(c) if c in labels else -99))
    return gesture, votes[gesture]


def draw(img, hands, g_df, path, title):
    n = max(len(hands), 1)
    fig = plt.figure(figsize=(14, max(4.2, 2.1 * n)))
    gs = fig.add_gridspec(n, K_SHOW + 3, width_ratios=[1.6, 1.6] + [1] * (K_SHOW + 1))
    ax = fig.add_subplot(gs[:, :2])
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(title, fontsize=9)
    W, H = img.size
    for h, (box, score, crop, idx, sims, gesture, votes) in enumerate(hands):
        x1, y1, x2, y2 = box
        ax.add_patch(patches.Rectangle((x1 * W, y1 * H), (x2 - x1) * W, (y2 - y1) * H, fill=False,
                                       edgecolor="red", lw=2))
        ax.text(x1 * W, y1 * H - 4, f"mano {h + 1}: {gesture} -> {COMMANDS[gesture]}", color="white", fontsize=8,
                bbox=dict(facecolor="red", alpha=0.85, pad=1, edgecolor="none"))
        a = fig.add_subplot(gs[h, 2])
        a.imshow(crop)
        a.set_title(f"mano {h + 1} (det {score:.2f})\n{gesture}: {votes}/{K_VOTE} votos", fontsize=7,
                    fontweight="bold")
        a.set_xticks([])
        a.set_yticks([])
        for j in range(K_SHOW):
            a = fig.add_subplot(gs[h, 3 + j])
            gi = int(idx[j])
            a.imshow(Image.open(ROOT / g_df.path.iloc[gi]))
            same = g_df.label.iloc[gi] == gesture
            a.set_title(f"#{j + 1} {g_df.label.iloc[gi]} ({sims[j]:.2f})", fontsize=7,
                        color="green" if same else "red")
            a.set_xticks([])
            a.set_yticks([])
    if not hands:
        ax.set_title(title + "\n(no se detectaron manos)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(args):
    device = get_device()
    detector, _ = load_detector(DET_CKPT, device)
    encoder = load_encoder(ENCODER, device)
    g_df, g_emb, _ = split_embeddings(ENCODER, "train", device, num_workers=0)
    thr = args.score if args.score is not None else json.load(open(OP_FROM))["op_score_from_val"]

    if args.image:
        paths = [Path(args.image)]
    else:
        items = json.load(open(ROOT / "data/splits/det_test.json"))
        random.seed(args.seed)
        paths = [ROOT / "data/images" / it["folder"] / f"{it['image_id']}.jpg"
                 for it in random.sample(items, args.random)]

    out_dir = ROOT / "results/demo_retrieval"
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        img = Image.open(path).convert("RGB")
        boxes, scores = find_hands(img, detector, device, thr)
        crops = [crop_hand(img, b) for b in boxes]
        hands = []
        if crops:
            sims, idx = search(embed_crops(encoder, crops, device), g_emb, K_VOTE, device)
            for b, s, c, i, sm in zip(boxes, scores, crops, idx.numpy(), sims.numpy()):
                gesture, votes = describe(i, sm, g_df)
                hands.append((b, float(s), c, i, sm, gesture, votes))
        dest = out_dir / f"{path.stem}_retrieval.jpg"
        name = path.name if len(path.name) <= 24 else path.stem[:8] + "…" + path.suffix
        draw(img, hands, g_df, dest, f"{name} | detector umbral {thr} | galería: {len(g_df)} manos de train")
        print(f"\n{path.name}: {len(hands)} mano(s)")
        for h, (_, s, _, _, sm, gesture, votes) in enumerate(hands):
            print(f"  mano {h + 1}: {gesture:<11} {votes}/{K_VOTE} votos, similitud top-1 {sm[0]:.2f}"
                  f"  -> comando: {COMMANDS[gesture]}")
        print(f"  imagen: {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="ruta a una imagen (jpg/png)")
    ap.add_argument("--random", type=int, default=3, help="si no se da --image: N imágenes al azar de test")
    ap.add_argument("--score", type=float, default=None, help="umbral del detector (por defecto, el de val)")
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
