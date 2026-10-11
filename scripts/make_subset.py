"""Construye el subconjunto reproducible de HaGRIDv2 para el Hito 1.

Lee las anotaciones oficiales (splits train/val/test ya separados por user_id),
muestrea N imágenes por gesto y split (máx. `--max_per_user` imágenes por usuario y gesto)
y genera dos manifests por split:

  data/splits/images_<split>.csv  -> imágenes a descargar
  data/splits/hands_<split>.csv   -> una fila por mano (bbox + label + landmarks)

La clase `no_gesture` se obtiene de las manos sin gesto que aparecen en esas mismas
imágenes (segunda mano en reposo) y se submuestrea al mismo N que el resto.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GESTURES = ["like", "dislike", "palm", "stop", "fist", "one", "peace", "ok", "call", "mute"]
N_PER_SPLIT = {"train": 1500, "val": 300, "test": 500}


def sample_images(ann: dict, n: int, max_per_user: int, rng: np.random.Generator) -> list:
    ids = np.array(sorted(ann.keys()))
    rng.shuffle(ids)
    per_user, chosen = {}, []
    for image_id in ids:
        user = ann[image_id]["user_id"]
        if per_user.get(user, 0) >= max_per_user:
            continue
        per_user[user] = per_user.get(user, 0) + 1
        chosen.append(image_id)
        if len(chosen) == n:
            break
    return chosen


def main(args):
    rng = np.random.default_rng(args.seed)
    ann_dir = ROOT / "data/raw/annotations"
    out_dir = ROOT / "data/splits"
    out_dir.mkdir(parents=True, exist_ok=True)

    for split, n in N_PER_SPLIT.items():
        images, hands = [], []
        for gesture in GESTURES:
            ann = json.load(open(ann_dir / split / f"{gesture}.json"))
            chosen = sample_images(ann, n, args.max_per_user, rng)
            if len(chosen) < n:
                print(f"[aviso] {split}/{gesture}: solo {len(chosen)} imágenes con max_per_user={args.max_per_user}")
            for image_id in chosen:
                a = ann[image_id]
                images.append({"image_id": image_id, "folder": gesture, "user_id": a["user_id"]})
                landmarks = a.get("hand_landmarks") or [[] for _ in a["labels"]]
                for k, (box, label) in enumerate(zip(a["bboxes"], a["labels"])):
                    lm = landmarks[k] if k < len(landmarks) else []
                    hands.append({
                        "image_id": image_id, "folder": gesture, "hand_idx": k, "label": label,
                        "x": box[0], "y": box[1], "w": box[2], "h": box[3],
                        "user_id": a["user_id"], "landmarks": json.dumps(lm),
                    })

        hands = pd.DataFrame(hands)
        # Balancear no_gesture al mismo tamaño que las demás clases
        is_ng = hands["label"] == "no_gesture"
        ng = hands[is_ng]
        if len(ng) > n:
            ng = ng.sample(n=n, random_state=args.seed)
        hands = pd.concat([hands[~is_ng], ng]).sort_values(["folder", "image_id", "hand_idx"])

        pd.DataFrame(images).to_csv(out_dir / f"images_{split}.csv", index=False)
        hands.to_csv(out_dir / f"hands_{split}.csv", index=False)
        print(f"{split}: {len(images)} imágenes, {len(hands)} manos, {hands['user_id'].nunique()} usuarios")
        print(hands["label"].value_counts().to_string(), "\n")

    # Comprobación: ningún usuario compartido entre splits
    users = {s: set(pd.read_csv(out_dir / f"images_{s}.csv")["user_id"]) for s in N_PER_SPLIT}
    assert not (users["train"] & users["val"]) and not (users["train"] & users["test"]) and not (users["val"] & users["test"])
    print("OK: splits disjuntos por usuario")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_per_user", type=int, default=1)
    main(p.parse_args())
