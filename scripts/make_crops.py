"""Genera recortes cuadrados de cada mano a partir de las bboxes oficiales.

Recorte: cuadrado centrado en la bbox, lado = max(w, h) * MARGIN (en píxeles), con relleno
negro si sale de la imagen, redimensionado a CROP_SIZE. Determinista (sin aleatoriedad).

Salida: data/crops/<split>/<label>/<image_id>_<hand_idx>.jpg
        data/splits/crops_<split>.csv (path relativo, label, user_id, tamaño de la mano, landmarks)
"""

from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MARGIN = 1.3
CROP_SIZE = 128


def crop_one(row: dict) -> dict | None:
    src = ROOT / "data/images" / row["folder"] / f"{row['image_id']}.jpg"
    rel = Path("data/crops") / row["split"] / row["label"] / f"{row['image_id']}_{row['hand_idx']}.jpg"
    dest = ROOT / rel
    try:
        img = Image.open(src).convert("RGB")
    except Exception:  # noqa: BLE001
        return None
    W, H = img.size
    cx, cy = (row["x"] + row["w"] / 2) * W, (row["y"] + row["h"] / 2) * H
    side = max(row["w"] * W, row["h"] * H) * MARGIN
    box = (round(cx - side / 2), round(cy - side / 2), round(cx + side / 2), round(cy + side / 2))
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        img.crop(box).resize((CROP_SIZE, CROP_SIZE), Image.BILINEAR).save(dest, quality=95)
    return {
        "path": str(rel), "label": row["label"], "user_id": row["user_id"],
        "image_id": row["image_id"], "hand_idx": row["hand_idx"],
        "hand_px": round(max(row["w"] * W, row["h"] * H)), "img_w": W, "img_h": H, "landmarks": row["landmarks"],
        "x": row["x"], "y": row["y"], "w": row["w"], "h": row["h"],
    }


def main():
    for split in ["train", "val", "test"]:
        hands = pd.read_csv(ROOT / f"data/splits/hands_{split}.csv")
        hands["split"] = split
        with Pool() as pool:
            rows = pool.map(crop_one, hands.to_dict("records"), chunksize=64)
        ok = [r for r in rows if r is not None]
        pd.DataFrame(ok).to_csv(ROOT / f"data/splits/crops_{split}.csv", index=False)
        print(f"{split}: {len(ok)} recortes ({len(rows) - len(ok)} imágenes faltantes)")


if __name__ == "__main__":
    main()
