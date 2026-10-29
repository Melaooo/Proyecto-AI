"""Anotaciones de detección para el Hito 2 (mismas imágenes y splits por usuario que el Hito 1).

A diferencia de hands_<split>.csv (donde no_gesture se submuestreó para balancear los recortes),
aquí se guardan TODAS las manos de cada imagen: un detector debe aprender que toda mano es un
objeto, si no, manos reales sin etiquetar se entrenan como fondo.

Salida: data/splits/det_<split>.json  -> lista de
  {image_id, folder, user_id, width, height, boxes: [[x, y, w, h] normalizados, ...], labels: [...]}
"""

import json
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def main():
    for split in ["train", "val", "test"]:
        images = pd.read_csv(ROOT / f"data/splits/images_{split}.csv")
        anns = {}
        out = []
        for row in images.itertuples():
            if row.folder not in anns:
                anns[row.folder] = json.load(open(ROOT / f"data/raw/annotations/{split}/{row.folder}.json"))
            a = anns[row.folder][row.image_id]
            with Image.open(ROOT / "data/images" / row.folder / f"{row.image_id}.jpg") as im:
                width, height = im.size
            out.append({
                "image_id": row.image_id, "folder": row.folder, "user_id": row.user_id,
                "width": width, "height": height,
                "boxes": [[round(v, 6) for v in b] for b in a["bboxes"]], "labels": a["labels"],
            })
        json.dump(out, open(ROOT / f"data/splits/det_{split}.json", "w"))
        n_boxes = sum(len(o["boxes"]) for o in out)
        n_two = sum(len(o["boxes"]) >= 2 for o in out)
        print(f"{split}: {len(out)} imágenes, {n_boxes} manos, {n_two} imágenes con 2+ manos")


if __name__ == "__main__":
    main()
