"""Demo del Hito 2: imagen -> manos localizadas -> gesto -> comando.

Uso:
  python -m src.demo --image ruta/a/foto.jpg                 # cascada (detector de manos + clasificador Hito 1)
  python -m src.demo --image ruta/a/foto.jpg --system directo
  python -m src.demo --random 4                              # 4 imágenes al azar del split de test

Guarda la imagen anotada en results/demo/ e imprime cada mano detectada con su gesto y comando.
Si la foto no es de HaGRID (por ejemplo, una foto propia), debe declararse como imagen externa en el informe.
"""

import argparse
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms as T

from src.data import CLASSES, MEAN, ROOT, STD, build_transforms
from src.det import (IMG_H, IMG_W, PAD_VALUE, canvas_xyxy_to_original, decode_predictions, letterbox_params,
                     load_detector)
from src.det_cascade import crop_hand, load_classifier
from src.train import get_device

# Comando de interfaz asociado a cada gesto (ejemplo de uso como control de un dispositivo)
COMMANDS = {
    "like": "subir volumen", "dislike": "bajar volumen", "palm": "pausar", "stop": "detener",
    "fist": "seleccionar", "one": "canal 1", "peace": "canal 2", "ok": "confirmar",
    "call": "contestar llamada", "mute": "silenciar", "no_gesture": "(sin comando)",
}
SYSTEMS = {"cascada": "checkpoints/det/det_hands_hito1/best.pt",
           "directo": "checkpoints/det/det_gestures_hito1_ft/best.pt"}
OP_SCORE = {"cascada": "results/det/cascade/metrics_test.json",
            "directo": "results/det/det_gestures_hito1_ft/metrics_test.json"}


# Achica la foto para que quepa en 288x384 sin deformarla (rellena con gris lo que sobra),
# igual que se hizo al entrenar el detector.
def letterbox(img):
    _, new_w, new_h, pad_x, pad_y = letterbox_params(*img.size)
    canvas = Image.new("RGB", (IMG_W, IMG_H), (PAD_VALUE,) * 3)
    canvas.paste(img.resize((new_w, new_h), Image.BILINEAR), (pad_x, pad_y))
    return T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])(canvas)


# Detecta las manos de una foto. "directo": el detector ya dice el gesto.
# "cascada": el detector solo encuentra la mano y el clasificador del Hito 1 dice el gesto.
@torch.no_grad()
def detect(img, system, detector, classifier, device, score_thr):
    raw = detector(letterbox(img)[None].to(device)).float().cpu()
    p = decode_predictions(raw, score_thr=0.05)[0]
    boxes = canvas_xyxy_to_original(p["boxes"], *img.size)
    out = []
    if system == "directo":
        for b, s, l in zip(boxes, p["scores"], p["labels"]):
            out.append((b, float(s), CLASSES[int(l)]))
    elif len(boxes):
        tf = build_transforms(False, 128)
        probs = classifier(torch.stack([tf(crop_hand(img, b)) for b in boxes]).to(device)).softmax(1).cpu()
        pc, cls = probs.max(1)
        for b, s, q, c in zip(boxes, p["scores"], pc, cls):
            out.append((b, float(s) * float(q), CLASSES[int(c)]))
    return [d for d in out if d[1] >= score_thr]


# Guarda la foto con cada mano marcada: gesto, score y el comando que activaría.
def draw(img, dets, path, title):
    W, H = img.size
    fig, ax = plt.subplots(figsize=(6, 6 * H / W))
    ax.imshow(img)
    ax.axis("off")
    for (x1, y1, x2, y2), s, l in dets:
        ax.add_patch(patches.Rectangle((x1 * W, y1 * H), (x2 - x1) * W, (y2 - y1) * H, fill=False,
                                       edgecolor="red", lw=2))
        ax.text(x1 * W, y1 * H - 4, f"{l} {s:.2f} -> {COMMANDS[l]}", color="white", fontsize=9,
                bbox=dict(facecolor="red", alpha=0.8, pad=1, edgecolor="none"))
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(args):
    device = get_device()
    detector, _ = load_detector(ROOT / SYSTEMS[args.system], device)
    classifier = load_classifier(ROOT / "checkpoints/resnet18_finetune/best.pt", device) \
        if args.system == "cascada" else None
    # umbral: el que se eligió en validación, salvo que se pase --score
    thr = args.score if args.score is not None else json.load(open(ROOT / OP_SCORE[args.system]))["op_score_from_val"]

    if args.image:
        paths = [Path(args.image)]
    else:
        items = json.load(open(ROOT / "data/splits/det_test.json"))
        random.seed(args.seed)
        paths = [ROOT / "data/images" / it["folder"] / f"{it['image_id']}.jpg" for it in random.sample(items, args.random)]

    out_dir = ROOT / "results/demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        img = Image.open(path).convert("RGB")
        dets = detect(img, args.system, detector, classifier, device, thr)
        dest = out_dir / f"{path.stem}_{args.system}.jpg"
        draw(img, dets, dest, f"{path.name} | sistema: {args.system} | umbral {thr}")
        print(f"\n{path.name}: {len(dets)} mano(s)")
        for (x1, y1, x2, y2), s, l in sorted(dets, key=lambda d: -d[1]):
            print(f"  {l:<11} score {s:.2f}  -> comando: {COMMANDS[l]}")
        print(f"  imagen anotada: {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="ruta a una imagen (jpg/png)")
    ap.add_argument("--random", type=int, default=4, help="si no se da --image: N imágenes al azar de test")
    ap.add_argument("--system", default="cascada", choices=list(SYSTEMS))
    ap.add_argument("--score", type=float, default=None, help="umbral de score (por defecto, el elegido en val)")
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
