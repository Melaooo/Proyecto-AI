"""Descarga solo las imágenes del subconjunto desde el zip oficial HaGRIDv2-512 (128 GB).

El servidor admite peticiones HTTP Range: se lee el índice del zip una vez (remotezip),
se guarda en data/raw/zip_index.csv y luego cada imagen se baja con una petición por rango.
Es reanudable: las imágenes ya presentes en data/images/ se saltan.
"""

import argparse
import ssl
import struct
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import certifi
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
URL = "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/hagridv2_512.zip"
PREFIX = "HaGRIDv2_dataset_512"
SSL_CTX = ssl.create_default_context(cafile=certifi.where())


def load_index() -> pd.DataFrame:
    path = ROOT / "data/raw/zip_index.csv"
    if path.exists():
        return pd.read_csv(path)
    from remotezip import RemoteZip

    print("Leyendo índice remoto del zip (~1 min)...")
    with RemoteZip(URL) as z:
        rows = [
            (i.filename, i.header_offset, i.compress_size, i.compress_type)
            for i in z.infolist() if i.filename.endswith(".jpg")
        ]
    df = pd.DataFrame(rows, columns=["filename", "offset", "compress_size", "compress_type"])
    df.to_csv(path, index=False)
    return df


def fetch(entry, dest: Path, retries: int = 5):
    # Cabecera local del zip: 30 bytes + nombre + campo extra; se pide un margen para el extra.
    start = int(entry.offset)
    end = start + 30 + len(entry.filename) + 1024 + int(entry.compress_size)
    for attempt in range(retries):
        try:
            req = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=60, context=SSL_CTX) as r:
                buf = r.read()
            name_len, extra_len = struct.unpack("<HH", buf[26:30])
            data_start = 30 + name_len + extra_len
            data = buf[data_start:data_start + int(entry.compress_size)]
            if int(entry.compress_type) == 8:
                data = zlib.decompress(data, -15)
            tmp = dest.with_suffix(".part")
            tmp.write_bytes(data)
            tmp.rename(dest)
            return
        except Exception as e:  # noqa: BLE001
            if attempt == retries - 1:
                raise RuntimeError(f"{dest.name}: {e}")


def main(args):
    index = load_index().set_index("filename")
    images = pd.concat(pd.read_csv(ROOT / f"data/splits/images_{s}.csv") for s in ["train", "val", "test"])
    todo = []
    for row in images.itertuples():
        dest = ROOT / "data/images" / row.folder / f"{row.image_id}.jpg"
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        entry = index.loc[f"{PREFIX}/{row.folder}/{row.image_id}.jpg"]
        entry = entry.copy()
        entry["filename"] = f"{PREFIX}/{row.folder}/{row.image_id}.jpg"
        todo.append((entry, dest))
    print(f"{len(images)} imágenes en el subconjunto, faltan {len(todo)}")

    errors = []
    with ThreadPoolExecutor(args.workers) as ex:
        futures = [ex.submit(fetch, e, d) for e, d in todo]
        for f in tqdm(as_completed(futures), total=len(futures), mininterval=10):
            try:
                f.result()
            except RuntimeError as e:
                errors.append(str(e))
    print(f"Terminado. Errores: {len(errors)}")
    for e in errors[:20]:
        print(" ", e)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=24)
    main(p.parse_args())
