"""Hito 2 (Persona 2): análisis por condición pedido por el enunciado.

  "Evaluar mAP y revisar distancia a cámara, una/dos manos y fondos diversos."

En simple: este archivo NO entrena nada. Lee las predicciones que ya guardaron el detector directo
y la cascada, separa las imágenes de test en grupos (manos lejos/cerca, 1 o 2 manos, fondos
oscuros/claros/simples/cargados) y calcula el mAP de cada sistema en cada grupo, para ver dónde falla.

Compara dos sistemas sobre el MISMO test, leyendo sus predictions_test.json (xyxy normalizadas a la
imagen original, score >= 0.05):
  - directo : results/det/det_gestures_hito1_ft  (mejor detector directo de gestos)
  - cascada : results/det/cascade                (detector de manos + clasificador del Hito 1)

Condiciones:
  1. Distancia a la cámara: tamaño de la mano max(w·W, h·H) en px de la imagen original,
     tercios del Hito 1 (< 83 / 83-117 / > 117 px). AP por rango con criterio tipo COCO: las manos
     fuera del rango se ignoran (se descartan las predicciones que calzan con ellas y las
     predicciones sin pareja cuyo tamaño cae fuera del rango).
  2. Una / dos manos: imágenes con 1 vs 2 cajas reales; qué pasa con la segunda mano.
  3. Fondos: HaGRID no etiqueta el fondo -> proxies por imagen: brillo medio (luminancia),
     contraste (desv. estándar) y densidad de bordes Sobel FUERA de las cajas. Tercios de cada uno.

Métricas: mAP50 y mAP50_95 (src.det.evaluate_detections, la misma función de todo el Hito 2) y, en
el umbral de operación elegido en validación, recall de gesto (caja IoU >= 0.5 y clase correcta) y
recall de localización (caja IoU >= 0.5 con cualquier clase).

Uso: python -m src.det_analysis [--directo det_gestures_hito1_ft --cascada cascade --out analysis]
Salida en results/det/analysis/: tablas .csv, resumen.md, figuras .png e image_stats.csv (proxies)
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.data import CLASSES, ROOT
from src.det import evaluate_detections, np_iou_matrix

SYSTEMS = {"directo": "det_gestures_hito1_ft", "cascada": "cascade"}
SIZE_BINS = [0, 83, 117, np.inf]
SIZE_NAMES = ["lejos (<83 px)", "medio (83-117 px)", "cerca (>117 px)"]
OUT = ROOT / "results/det/analysis"
EDGE_THR = 0.25          # umbral de magnitud Sobel (gris en [0, 1]) para contar un píxel como borde


# ----------------------------------------------------------------------------------------------
# Carga
# ----------------------------------------------------------------------------------------------
# Lee las predicciones de test de un sistema y el umbral que se eligió en validación.
def load_system(exp):
    d = ROOT / "results/det" / exp
    rows = json.load(open(d / "predictions_test.json"))
    op = json.load(open(d / "metrics_test.json"))["op_score_from_val"]
    return rows, op


def to_eval(rows):
    """Filas exportadas -> (preds, gts) en el formato de evaluate_detections."""
    preds, gts = [], []
    for r in rows:
        preds.append({"boxes": np.array(r["pred_boxes"]).reshape(-1, 4), "scores": np.array(r["pred_scores"]),
                      "labels": np.array([CLASSES.index(l) for l in r["pred_labels"]], dtype=int)})
        gts.append((np.array(r["gt_boxes"]).reshape(-1, 4), np.array([CLASSES.index(l) for l in r["gt_labels"]])))
    return preds, gts


# Tamaño de cada mano en píxeles (el lado más largo de la caja). Mano chica = lejos de la cámara.
def box_px(boxes, W, H):
    b = np.asarray(boxes).reshape(-1, 4)
    return np.maximum((b[:, 2] - b[:, 0]) * W, (b[:, 3] - b[:, 1]) * H)


# ----------------------------------------------------------------------------------------------
# Matching en el punto de operación (para recall y para clasificar errores)
# ----------------------------------------------------------------------------------------------
# Para cada mano real dice qué le pasó: 'ok' (la encontró con el gesto correcto), 'clase'
# (la encontró pero con otro gesto) o 'perdida' (no la encontró). Encontrar = caja con IoU >= 0.5.
def match_op(pred, gt, op, iou_thr=0.5):
    """Greedy por score, consciente de la clase. Devuelve estado por GT y si cada pred (>= op) es TP.
    Estado GT: 'ok' (caja + clase), 'clase' (caja bien pero otra clase), 'perdida' (sin caja)."""
    keep = pred["scores"] >= op
    pb, ps, pl = pred["boxes"][keep], pred["scores"][keep], pred["labels"][keep]
    order = np.argsort(-ps)
    pb, ps, pl = pb[order], ps[order], pl[order]
    gb, gl = gt
    ious = np_iou_matrix(pb, gb)
    used = np.zeros(len(gb), bool)
    tp = np.zeros(len(pb), bool)
    for i in range(len(pb)):
        if len(gb) == 0:
            break
        cand = np.where(~used & (gl == pl[i]), ious[i], -1)    # solo GT libres de la misma clase
        j = int(cand.argmax())
        if cand[j] >= iou_thr:
            used[j] = tp[i] = True
    state = []
    for j in range(len(gb)):
        if used[j]:
            state.append("ok")
        elif len(pb) and ious[:, j].max() >= iou_thr:
            state.append("clase")
        else:
            state.append("perdida")
    return state, (pb, ps, pl, tp)


def hand_table(rows, op):
    """Una fila por mano real con su estado en el punto de operación."""
    preds, gts = to_eval(rows)
    out = []
    for r, p, g in zip(rows, preds, gts):
        state, _ = match_op(p, g, op)
        px = box_px(r["gt_boxes"], r["width"], r["height"])
        for k, s in enumerate(state):
            out.append({"image_id": r["image_id"], "hand_idx": k, "label": r["gt_labels"][k],
                        "hand_px": px[k], "n_hands": len(state), "estado": s})
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------------------------
# AP por subconjunto
# ----------------------------------------------------------------------------------------------
# mAP usando solo algunas imágenes (por ejemplo, solo las que tienen 2 manos).
def ap_subset(rows, idx):
    preds, gts = to_eval([rows[i] for i in idx])
    m = evaluate_detections(preds, gts, CLASSES)
    return m["mAP50"], m["mAP50_95"]


# mAP solo para manos de cierto tamaño. Las manos de otro tamaño se ignoran: no cuentan como
# perdidas, y si el modelo las detecta tampoco cuenta como error (así se hace en COCO).
def ap_by_size(rows, lo, hi):
    """AP en manos de tamaño [lo, hi) ignorando las demás (criterio tipo COCO)."""
    preds, gts = [], []
    for r in rows:
        W, H = r["width"], r["height"]
        gb = np.array(r["gt_boxes"]).reshape(-1, 4)
        gl = np.array([CLASSES.index(l) for l in r["gt_labels"]])
        pb = np.array(r["pred_boxes"]).reshape(-1, 4)
        ps, pl = np.array(r["pred_scores"]), np.array([CLASSES.index(l) for l in r["pred_labels"]], dtype=int)
        gpx, ppx = box_px(gb, W, H), box_px(pb, W, H)
        inside = (gpx >= lo) & (gpx < hi)
        ious = np_iou_matrix(pb, gb)
        hits_out = (ious[:, ~inside] >= 0.5).any(1) if (~inside).any() and len(pb) else np.zeros(len(pb), bool)
        hits_in = (ious[:, inside] >= 0.5).any(1) if inside.any() and len(pb) else np.zeros(len(pb), bool)
        own_in = (ppx >= lo) & (ppx < hi)
        keep = ~(hits_out & ~hits_in) & (hits_in | own_in)
        preds.append({"boxes": pb[keep], "scores": ps[keep], "labels": pl[keep]})
        gts.append((gb[inside], gl[inside]))
    m = evaluate_detections(preds, gts, CLASSES)
    return m["mAP50"], m["mAP50_95"]


def recalls(h):
    n = len(h)
    return {"manos": n, "recall_gesto": (h["estado"] == "ok").mean() if n else np.nan,
            "recall_localizacion": (h["estado"] != "perdida").mean() if n else np.nan}


# ----------------------------------------------------------------------------------------------
# Proxies de fondo
# ----------------------------------------------------------------------------------------------
# HaGRID no dice qué fondo tiene cada foto, así que se miden 3 cosas por imagen:
#   brillo     = qué tan clara u oscura es (promedio de gris)
#   contraste  = cuánto varía el gris (desviación estándar)
#   bordes     = qué tan "cargado" es el fondo: % de píxeles con bordes (filtro Sobel), sin contar las manos
# Se guarda en image_stats.csv para no recalcularlo cada vez.
def image_stats(rows):
    path = OUT / "image_stats.csv"
    if path.exists():
        return pd.read_csv(path)
    stats = []
    for r in rows:
        img = Image.open(ROOT / "data/images" / r["folder"] / f"{r['image_id']}.jpg").convert("L")
        g = np.asarray(img, dtype=np.float32) / 255.0
        H, W = g.shape
        p = np.pad(g, 1, mode="edge")
        gx = (p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[1:-1, :-2] + p[2:, :-2])
        gy = (p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:])
        edges = np.hypot(gx, gy) > EDGE_THR
        outside = np.ones_like(edges)
        for x1, y1, x2, y2 in r["gt_boxes"]:          # se excluyen las manos: solo cuenta el fondo
            outside[int(y1 * H):int(np.ceil(y2 * H)), int(x1 * W):int(np.ceil(x2 * W))] = False
        stats.append({"image_id": r["image_id"], "brillo": g.mean(), "contraste": g.std(),
                      "bordes_fondo": edges[outside].mean()})
    df = pd.DataFrame(stats)
    df.to_csv(path, index=False)
    return df


# ----------------------------------------------------------------------------------------------
# Figuras
# ----------------------------------------------------------------------------------------------
# Dibuja una imagen con las cajas reales (verde) y las detectadas (rojo).
def draw(ax, r, op, title=None):
    img = Image.open(ROOT / "data/images" / r["folder"] / f"{r['image_id']}.jpg")
    W, H = img.size
    ax.imshow(img)
    ax.axis("off")
    for x1, y1, x2, y2 in r["gt_boxes"]:
        ax.add_patch(patches.Rectangle((x1 * W, y1 * H), (x2 - x1) * W, (y2 - y1) * H, fill=False,
                                       edgecolor="lime", lw=2))
    for (x1, y1, x2, y2), s, l in zip(r["pred_boxes"], r["pred_scores"], r["pred_labels"]):
        if s < op:
            continue
        ax.add_patch(patches.Rectangle((x1 * W, y1 * H), (x2 - x1) * W, (y2 - y1) * H, fill=False,
                                       edgecolor="red", lw=1.5))
        ax.text(x1 * W, y1 * H - 3, f"{l} {s:.2f}", color="white", fontsize=6,
                bbox=dict(facecolor="red", alpha=0.8, pad=0.5, edgecolor="none"))
    if title:
        ax.set_title(title, fontsize=7)


# Gráfico de barras: mAP del directo vs la cascada en cada grupo.
def plot_conditions(table, path):
    conds = list(dict.fromkeys(table["condicion"]))
    fig, axes = plt.subplots(1, len(conds), figsize=(3.6 * len(conds), 3.4), sharey=True)
    for ax, c in zip(np.atleast_1d(axes), conds):
        t = table[table["condicion"] == c]
        groups = list(dict.fromkeys(t["grupo"]))
        x = np.arange(len(groups))
        for k, (sys_name, color) in enumerate([("directo", "#7B3294"), ("cascada", "#E66101")]):
            v = [t[(t["grupo"] == g) & (t["sistema"] == sys_name)]["mAP50"].item() for g in groups]
            ax.bar(x + (k - 0.5) * 0.38, v, 0.38, label=sys_name, color=color)
        ax.set_xticks(x, groups, rotation=20, ha="right", fontsize=8)
        ax.set_title(c, fontsize=10)
        ax.set_ylim(0.5, 1.0)
        ax.grid(axis="y", alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("mAP@0.5 (test)")
    np.atleast_1d(axes)[0].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_background_examples(rows, op, stats, path, per=3, seed=0):
    """Por cada proxy de fondo: ejemplos del tercio bajo, medio y alto."""
    by_id = {r["image_id"]: r for r in rows}
    proxies = ["brillo", "bordes_fondo"]
    fig, axes = plt.subplots(len(proxies) * 3, per, figsize=(per * 2.3, len(proxies) * 3 * 2.9))
    rng = np.random.RandomState(seed)
    for pi, prox in enumerate(proxies):
        terc = pd.qcut(stats[prox], 3, labels=["bajo", "medio", "alto"])
        for ti, name in enumerate(["bajo", "medio", "alto"]):
            ids = stats["image_id"][terc == name].to_numpy()
            for k, iid in enumerate(rng.choice(ids, per, replace=False)):
                ax = axes[pi * 3 + ti, k]
                v = stats.loc[stats["image_id"] == iid, prox].item()
                draw(ax, by_id[iid], op, f"{prox} {name} ({v:.2f})")
    fig.suptitle("Verde: real | Rojo: detección (umbral de operación)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_errors(rows, op, path, per=4, seed=0):
    """Errores típicos: mano perdida, clase equivocada, falso positivo."""
    preds, gts = to_eval(rows)
    kinds = {"mano perdida": [], "clase equivocada": [], "falso positivo": []}
    for i, (p, g) in enumerate(zip(preds, gts)):
        state, (_, _, _, tp) = match_op(p, g, op)
        if "perdida" in state:
            kinds["mano perdida"].append(i)
        if "clase" in state:
            kinds["clase equivocada"].append(i)
        if (~tp).sum() > state.count("clase"):          # FP que no son solo la clase equivocada
            kinds["falso positivo"].append(i)
    rng = np.random.RandomState(seed)
    fig, axes = plt.subplots(3, per, figsize=(per * 2.4, 3 * 3.1))
    for ri, (k, idx) in enumerate(kinds.items()):
        sel = rng.choice(idx, min(per, len(idx)), replace=False) if idx else []
        for ci in range(per):
            ax = axes[ri, ci]
            ax.axis("off")
            if ci < len(sel):
                r = rows[sel[ci]]
                draw(ax, r, op, f"{k} | GT: {', '.join(r['gt_labels'])}")
    fig.suptitle("Errores típicos (verde: real | rojo: detección)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return {k: len(v) for k, v in kinds.items()}


# ----------------------------------------------------------------------------------------------
def main(args):
    global OUT
    OUT = ROOT / "results/det" / args.out
    OUT.mkdir(parents=True, exist_ok=True)
    systems = {"directo": load_system(args.directo), "cascada": load_system(args.cascada)}
    ref_rows = systems["directo"][0]
    # Los dos sistemas tienen que haberse evaluado sobre las mismas imágenes para poder compararlos
    assert [r["image_id"] for r in ref_rows] == [r["image_id"] for r in systems["cascada"][0]], \
        "los dos sistemas deben evaluarse sobre las mismas imágenes y en el mismo orden"
    stats = image_stats(ref_rows).set_index("image_id").loc[[r["image_id"] for r in ref_rows]].reset_index()
    n_hands = np.array([len(r["gt_boxes"]) for r in ref_rows])

    # Se arma una tabla con una fila por (sistema, condición, grupo): mAP y recall de cada grupo
    table, second = [], []
    for sys_name, (rows, op) in systems.items():
        hands = hand_table(rows, op)
        # 0. Global
        m50, m5095 = ap_subset(rows, range(len(rows)))
        table.append({"sistema": sys_name, "condicion": "global", "grupo": "todo", "imagenes": len(rows),
                      "mAP50": m50, "mAP50_95": m5095, **recalls(hands)})
        # 1. Distancia a la cámara
        for (lo, hi), name in zip(zip(SIZE_BINS[:-1], SIZE_BINS[1:]), SIZE_NAMES):
            m50, m5095 = ap_by_size(rows, lo, hi)
            h = hands[(hands["hand_px"] >= lo) & (hands["hand_px"] < hi)]
            table.append({"sistema": sys_name, "condicion": "distancia a la cámara", "grupo": name,
                          "imagenes": h["image_id"].nunique(), "mAP50": m50, "mAP50_95": m5095, **recalls(h)})
        # 2. Una / dos manos
        for k, name in [(1, "1 mano"), (2, "2 manos")]:
            idx = np.where(n_hands == k)[0]
            m50, m5095 = ap_subset(rows, idx)
            table.append({"sistema": sys_name, "condicion": "una / dos manos", "grupo": name,
                          "imagenes": len(idx), "mAP50": m50, "mAP50_95": m5095,
                          **recalls(hands[hands["n_hands"] == k])})
        # En imágenes con 2 manos casi siempre una hace el gesto y la otra está en reposo
        # (no_gesture): se cuenta cuántas de cada tipo se pierden
        two = hands[hands["n_hands"] == 2]
        for is_ng, name in [(False, "mano con gesto"), (True, "mano no_gesture")]:
            h = two[(two["label"] == "no_gesture") == is_ng]
            second.append({"sistema": sys_name, "mano (imágenes con 2 manos)": name, "manos": len(h),
                           "perdidas": int((h["estado"] == "perdida").sum()),
                           "clase equivocada": int((h["estado"] == "clase").sum()),
                           **{k: v for k, v in recalls(h).items() if k != "manos"}})
        # 3. Fondos: se ordenan las imágenes por brillo / contraste / bordes y se parten en tercios
        for prox, cond in [("brillo", "fondo: brillo"), ("contraste", "fondo: contraste"),
                           ("bordes_fondo", "fondo: bordes (complejidad)")]:
            terc = pd.qcut(stats[prox], 3, labels=["bajo", "medio", "alto"])
            for name in ["bajo", "medio", "alto"]:
                idx = np.where(terc == name)[0]
                ids = set(stats["image_id"].iloc[idx])
                m50, m5095 = ap_subset(rows, idx)
                rng_txt = f"{stats[prox].iloc[idx].min():.2f}-{stats[prox].iloc[idx].max():.2f}"
                table.append({"sistema": sys_name, "condicion": cond, "grupo": f"{name} ({rng_txt})",
                              "imagenes": len(idx), "mAP50": m50, "mAP50_95": m5095,
                              **recalls(hands[hands["image_id"].isin(ids)])})
        counts = plot_errors(rows, op, OUT / f"errores_{sys_name}.png")
        print(f"{sys_name}: imágenes con cada tipo de error (umbral {op}): {counts}")

    # Guardar tablas (.csv), figuras (.png) y un resumen.md listo para copiar al informe
    table = pd.DataFrame(table).round(4)
    second = pd.DataFrame(second).round(4)
    table.to_csv(OUT / "por_condicion.csv", index=False)
    second.to_csv(OUT / "segunda_mano.csv", index=False)
    plot_conditions(table[table["condicion"] != "global"].assign(
        grupo=lambda t: t["grupo"].str.replace(r" \(.*\)$", "", regex=True)), OUT / "map_por_condicion.png")
    plot_background_examples(ref_rows, systems["directo"][1], stats, OUT / "fondos_ejemplos.png")

    wide = table.pivot_table(index=["condicion", "grupo"], columns="sistema",
                             values=["mAP50", "recall_gesto", "recall_localizacion"], sort=False)
    wide.columns = [f"{v} {s}" for v, s in wide.columns]
    with open(OUT / "resumen.md", "w") as f:
        f.write("# Análisis por condición (test)\n\n")
        f.write(wide.round(3).to_markdown() + "\n\n## Imágenes con dos manos\n\n")
        f.write(second.to_markdown(index=False) + "\n")
    print(wide.round(3).to_string())
    print(second.to_string(index=False))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--directo", default=SYSTEMS["directo"], help="experimento en results/det/ del detector directo")
    ap.add_argument("--cascada", default=SYSTEMS["cascada"], help="carpeta en results/det/ de la cascada")
    ap.add_argument("--out", default="analysis", help="carpeta de salida en results/det/")
    main(ap.parse_args())
