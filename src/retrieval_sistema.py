"""Hito 3 (Persona 2): sistema completo imagen -> detección -> recorte -> embedding -> Top-K.

En simple: la tabla de src/retrieval.py usa recortes hechos con las cajas REALES (el techo del
sistema). Aquí las cajas las entrega el detector de manos del Hito 2, como pasaría con una foto
nueva, y se mide cuánto se pierde.

Pasos para cada imagen de test (data/splits/det_test.json, 5 000 imágenes, 6 049 manos):
  1. Detector de manos del Hito 2 (results/det/det_hands_hito1/predictions_test.json), con el
     umbral de score elegido en VALIDACIÓN en el Hito 2 (op_score_from_val de la cascada).
  2. Recorte igual que en el Hito 1 (src.det_cascade.crop_hand: cuadrado 1.3x, 128x128).
  3. Embedding con el encoder elegido en el Hito 3 (resnet18_ft, normalizado a largo 1).
  4. Top-K en la galería de train (los mismos 16 500 recortes de src/retrieval.py).
  5. Cada detección se empareja con la mano real de mayor IoU (>= 0.5, greedy por score). Así se
     sabe su gesto real. Detecciones sin pareja = falsos positivos; manos sin detección = perdidas.

Comparación justa: el techo se recalcula con las cajas reales de las MISMAS 6 049 manos (la tabla
de src/retrieval.py usa 5 500 porque ahí no_gesture está submuestreada). Las manos perdidas cuentan
como fallo en todas las métricas del sistema.

Métricas por mano real (relevante = vecino de la galería con el mismo gesto):
  R@1, R@5, P@10, AP sobre el ranking completo (mAP) y kNN10 (gesto por votación de 10 vecinos).

Uso: python -m src.retrieval_sistema
Salida en results/retrieval/sistema/: metrics_test.json, resumen.md, por_tamano.csv, por_iou.csv,
detecciones.csv y casos_sistema.png
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.data import CLASSES, ROOT, build_transforms
from src.det import np_iou_matrix
from src.det_cascade import crop_hand
from src.retrieval import K_VOTE, labels_of, load_encoder, ranking_metrics, search, split_embeddings
from src.train import get_device

ENCODER = "resnet18_ft"
DET_PREDS = ROOT / "results/det/det_hands_hito1/predictions_test.json"
OP_FROM = ROOT / "results/det/cascade/metrics_test.json"        # umbral elegido en val (Hito 2)
OUT = ROOT / "results/retrieval/sistema"
SIZE_BINS, SIZE_NAMES = [0, 83, 117, np.inf], ["lejos (<83 px)", "medio (83-117 px)", "cerca (>117 px)"]
IOU_BINS, IOU_NAMES = [0.5, 0.7, 0.85, 1.01], ["0.50-0.70", "0.70-0.85", "0.85-1.00"]
SEED = 42


# Empareja las detecciones de una imagen con sus manos reales (sin mirar el gesto: el detector
# es de manos). Greedy por score, IoU >= 0.5, cada mano real a lo más una vez.
def match(det_boxes, det_scores, gt_boxes, thr=0.5):
    order = np.argsort(-det_scores)
    ious = np_iou_matrix(det_boxes, gt_boxes)
    gt_of_det, iou_of_det = np.full(len(det_boxes), -1), np.zeros(len(det_boxes))
    used = np.zeros(len(gt_boxes), bool)
    for i in order:
        if len(gt_boxes) == 0:
            break
        cand = np.where(used, -1, ious[i])
        j = int(cand.argmax())
        if cand[j] >= thr:
            used[j] = True
            gt_of_det[i], iou_of_det[i] = j, cand[j]
    return gt_of_det, iou_of_det


@torch.no_grad()
def embed_crops(encoder, crops, device, batch=256):
    tf = build_transforms(False, 128)
    out = []
    for i in range(0, len(crops), batch):
        x = torch.stack([tf(c) for c in crops[i:i + batch]]).to(device)
        out.append(torch.nn.functional.normalize(encoder(x).float(), dim=1).cpu())
    return torch.cat(out) if out else torch.zeros(0, 512)


def build_queries(preds, items, op):
    """Recorta en memoria las manos reales (techo) y las detectadas (sistema) de todas las imágenes.
    preds trae las cajas (el detector es de 1 clase: sus gt_labels dicen 'hand'); el gesto real sale de items."""
    gt_rows, det_rows, gt_crops, det_crops = [], [], [], []
    for r, it in zip(preds, items):
        assert r["image_id"] == it["image_id"]
        labels = it["labels"]
        img = Image.open(ROOT / "data/images" / r["folder"] / f"{r['image_id']}.jpg").convert("RGB")
        W, H = img.size
        gt = np.array(r["gt_boxes"]).reshape(-1, 4)
        db = np.array(r["pred_boxes"]).reshape(-1, 4)
        ds = np.array(r["pred_scores"])
        keep = ds >= op
        db, ds = db[keep], ds[keep]
        gt_of_det, iou_of_det = match(db, ds, gt)
        det_of_gt = {g: d for d, g in enumerate(gt_of_det) if g >= 0}
        g0 = len(gt_rows)
        for j, (box, lab) in enumerate(zip(gt, labels)):
            px = max((box[2] - box[0]) * W, (box[3] - box[1]) * H)
            gt_rows.append({"image_id": r["image_id"], "folder": r["folder"], "hand_idx": j, "label": lab,
                            "hand_px": px, "box": box.tolist(), "det_idx": -1})
            gt_crops.append(crop_hand(img, box))
        for d, (box, s) in enumerate(zip(db, ds)):
            g = gt_of_det[d]
            det_rows.append({"image_id": r["image_id"], "folder": r["folder"], "box": box.tolist(),
                             "score": float(s), "iou": float(iou_of_det[d]),
                             "gt_row": g0 + g if g >= 0 else -1,
                             "label": labels[g] if g >= 0 else "falso_positivo"})
            if g >= 0:
                gt_rows[g0 + g]["det_idx"] = len(det_rows) - 1
            det_crops.append(crop_hand(img, box))
    return pd.DataFrame(gt_rows), pd.DataFrame(det_rows), gt_crops, det_crops


def metrics_per_hand(q_emb, q_lab, g_emb, g_lab, device):
    per_q = ranking_metrics(q_emb, q_lab, g_emb, g_lab, device)
    return pd.DataFrame({"R@1": per_q["R@1"], "R@5": per_q["R@5"], "P@10": per_q["P@10"], "AP": per_q["AP"],
                         "knn_pred": per_q["knn_pred"], "top1": per_q["top"][:, 0]})


def summary(df, y):
    """Promedios por mano real. Las manos sin consulta (perdidas) tienen 0 y knn_pred = -1."""
    return {"manos": int(len(df)), "mAP": df["AP"].mean(), "R@1": df["R@1"].mean(), "R@5": df["R@5"].mean(),
            "P@10": df["P@10"].mean(), f"kNN{K_VOTE}_acc": (df["knn_pred"] == y).mean()}
    # (sin F1 macro: las manos perdidas no tienen predicción, no bajan la precisión y el F1 sale
    #  más alto que el accuracy, lo que daría una imagen optimista del sistema)


# ---------------------------------------------------------------------------------------------
# Figura de casos del sistema completo
# ---------------------------------------------------------------------------------------------
def plot_cases(cases, g_df, path, k=5):
    """cases: lista de (título, imagen PIL recorte, etiqueta real, índices top-k, sims top-k, color título)."""
    fig, axes = plt.subplots(len(cases), k + 1, figsize=((k + 1) * 1.75, len(cases) * 2.3))
    for r, (title, crop, lab, idx, sims, color) in enumerate(cases):
        ax = axes[r, 0]
        ax.imshow(crop)
        ax.set_title(f"{title}\nreal: {lab}", fontsize=8, fontweight="bold", color=color)
        for j in range(k):
            ax = axes[r, j + 1]
            if idx is None:
                ax.axis("off")
                continue
            gi = idx[j]
            ax.imshow(Image.open(ROOT / g_df.path.iloc[gi]))
            same = g_df.label.iloc[gi] == lab
            ax.set_title(f"#{j + 1} {g_df.label.iloc[gi]} ({sims[j]:.2f})", fontsize=7,
                         color="green" if same else "red")
            for s in ax.spines.values():
                s.set_edgecolor("green" if same else "red")
                s.set_linewidth(3)
        for ax in axes[r]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("Sistema completo en test: recorte del DETECTOR (izquierda) y sus 5 vecinos en la galería de train\n"
                 "verde = mismo gesto que la mano real, rojo = otro gesto, (similitud coseno)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main(args):
    device = get_device()
    torch.manual_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    op = json.load(open(OP_FROM))["op_score_from_val"]
    preds = json.load(open(DET_PREDS))

    g_df, g_emb, g_valid = split_embeddings(ENCODER, "train", device, num_workers=args.num_workers)
    g_lab = labels_of(g_df)
    encoder = load_encoder(ENCODER, device)

    items = json.load(open(ROOT / "data/splits/det_test.json"))
    gt_df, det_df, gt_crops, det_crops = build_queries(preds, items, op)
    test_users = {it["user_id"] for it in items}
    assert not test_users & set(g_df.user_id), "hay personas en la galería y en las consultas"
    y = np.array([CLASSES.index(l) for l in gt_df.label])

    # Techo: recortes con la caja real de las mismas 6 049 manos
    gt_emb = embed_crops(encoder, gt_crops, device)
    techo = metrics_per_hand(gt_emb, torch.from_numpy(y), g_emb, g_lab, device)

    # Sistema: recortes del detector. Las detecciones emparejadas heredan el gesto de su mano real
    det_emb = embed_crops(encoder, det_crops, device)
    tp = det_df.gt_row.to_numpy() >= 0
    det_lab = torch.from_numpy(y[det_df.gt_row.to_numpy()[tp]])
    m_tp = metrics_per_hand(det_emb[tp], det_lab, g_emb, g_lab, device)
    sistema = pd.DataFrame({"R@1": 0.0, "R@5": 0.0, "P@10": 0.0, "AP": 0.0, "knn_pred": -1, "top1": -1},
                           index=range(len(gt_df)))
    sistema.loc[det_df.gt_row.to_numpy()[tp], m_tp.columns] = m_tp.to_numpy()
    sistema["knn_pred"] = sistema["knn_pred"].astype(int)

    # Vecinos y similitud de TODAS las detecciones (también falsos positivos) para el análisis
    sims, idx = search(det_emb, g_emb, 10, device)
    det_df["top1_sim"] = sims[:, 0].numpy()
    det_df["knn_pred"] = [CLASSES[int(np.bincount(g_lab[i].numpy(), minlength=len(CLASSES)).argmax())]
                          for i in idx]
    det_df["top1_label"] = [g_df.label.iloc[int(i[0])] for i in idx]

    res = {"encoder": ENCODER, "op_score_det": op, "galeria": int(len(g_df)),
           "techo_cajas_reales": summary(techo, y), "sistema_detector": summary(sistema, y)}
    res["manos_perdidas"] = int((gt_df.det_idx < 0).sum())
    res["detecciones"] = int(len(det_df))
    res["falsos_positivos"] = int((~tp).sum())
    res["sim_top1_media"] = {"detecciones correctas": float(det_df.top1_sim[tp].mean()),
                            "falsos positivos": float(det_df.top1_sim[~tp].mean())}
    res["knn_de_falsos_positivos"] = det_df.knn_pred[~tp].value_counts().to_dict()

    # Por tamaño de mano (distancia a la cámara) y por IoU de la caja detectada
    size_rows, iou_rows = [], []
    bins = pd.cut(gt_df.hand_px, SIZE_BINS, labels=SIZE_NAMES, right=False)
    for name in SIZE_NAMES:
        m = (bins == name).to_numpy()
        for sys_name, d in [("techo", techo), ("sistema", sistema)]:
            size_rows.append({"tamano": name, "sistema": sys_name, **summary(d[m], y[m]),
                              "perdidas": int((gt_df.det_idx[m] < 0).sum())})
    det_tp = det_df[tp].reset_index(drop=True)
    ib = pd.cut(det_tp.iou, IOU_BINS, labels=IOU_NAMES, right=False)
    for name in IOU_NAMES:
        m = (ib == name).to_numpy()
        rows_gt = det_tp.gt_row.to_numpy()[m]
        iou_rows.append({"IoU caja detectada": name, "manos": int(m.sum()),
                         "R@1 sistema": m_tp["R@1"].to_numpy()[m].mean(), "R@1 techo (caja real)": techo["R@1"].to_numpy()[rows_gt].mean(),
                         "P@10 sistema": m_tp["P@10"].to_numpy()[m].mean(), "P@10 techo": techo["P@10"].to_numpy()[rows_gt].mean()})
    size_df, iou_df = pd.DataFrame(size_rows).round(4), pd.DataFrame(iou_rows).round(4)

    json.dump(res, open(OUT / "metrics_test.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False,
              default=float)
    size_df.to_csv(OUT / "por_tamano.csv", index=False)
    iou_df.to_csv(OUT / "por_iou.csv", index=False)
    det_df.drop(columns=["folder"]).to_csv(OUT / "detecciones.csv", index=False)
    comp = pd.DataFrame([{"sistema": "techo: caja real", **res["techo_cajas_reales"]},
                         {"sistema": "detector + retrieval", **res["sistema_detector"]}]).round(4)
    with open(OUT / "resumen.md", "w", encoding="utf-8") as f:
        f.write(f"# Hito 3 · Sistema completo (test, {len(gt_df)} manos, galería = train)\n\n")
        f.write(comp.to_markdown(index=False) + "\n\n")
        f.write(f"Detector de manos con umbral {op} (elegido en val, Hito 2): {res['detecciones']} detecciones, "
                f"{res['falsos_positivos']} falsos positivos, {res['manos_perdidas']} manos perdidas.\n\n")
        f.write(f"Similitud top-1 media: correctas {res['sim_top1_media']['detecciones correctas']:.3f}, "
                f"falsos positivos {res['sim_top1_media']['falsos positivos']:.3f}\n\n")
        f.write("## Por tamaño de mano\n\n" + size_df.to_markdown(index=False) + "\n\n")
        f.write("## Por IoU de la caja detectada (solo manos encontradas)\n\n" + iou_df.to_markdown(index=False) + "\n")

    # Casos para la figura (semilla fija): aciertos, caja mal centrada, falso positivo, mano perdida
    rng = np.random.default_rng(SEED)
    _, gt_idx = search(gt_emb, g_emb, 5, device)
    gt_sims, _ = search(gt_emb, g_emb, 5, device)
    cases = []
    ok = det_tp.index[(m_tp["R@1"].to_numpy() == 1) & (det_tp.iou >= 0.85)].to_numpy()
    for d in rng.choice(ok, 2, replace=False):
        i = det_df.index[tp][d]
        cases.append(("Acierto", det_crops[i], det_df.label[i], idx[i].numpy(), sims[i].numpy(), "green"))
    off = det_tp.index[(m_tp["R@1"].to_numpy() == 0) & (det_tp.iou < 0.7)].to_numpy()
    if len(off) == 0:
        off = det_tp.index[m_tp["R@1"].to_numpy() == 0].to_numpy()
    if len(off):
        d = rng.choice(off)
        i = det_df.index[tp][d]
        cases.append((f"Caja mal centrada (IoU {det_df.iou[i]:.2f})", det_crops[i], det_df.label[i],
                      idx[i].numpy(), sims[i].numpy(), "red"))
    fp = np.where(~tp)[0]
    if len(fp):
        i = rng.choice(fp)
        cases.append((f"Falso positivo (score {det_df.score[i]:.2f})", det_crops[i], "sin mano",
                      idx[i].numpy(), sims[i].numpy(), "red"))
    lost = np.where(gt_df.det_idx.to_numpy() < 0)[0]
    if len(lost):
        j = rng.choice(lost)
        cases.append(("Mano perdida: no detectada\n(vecinos con la caja real)", gt_crops[j], gt_df.label[j],
                      gt_idx[j].numpy(), gt_sims[j].numpy(), "red"))
    plot_cases(cases, g_df, OUT / "casos_sistema.png")

    print(comp.to_string(index=False))
    print({k: v for k, v in res.items() if k not in ("techo_cajas_reales", "sistema_detector")})
    print(size_df.to_string(index=False))
    print(iou_df.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_workers", type=int, default=0)
    main(ap.parse_args())
