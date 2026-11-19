"""Hito 3 (Persona 1): retrieval de gestos con el encoder del Hito 1.

En simple: cada recorte de mano se pasa por el encoder y sale un vector (embedding). Dos manos con
el mismo gesto deberían quedar con vectores parecidos. Para una mano nueva (consulta) se buscan
los K vectores más parecidos de una galería de manos ya conocidas: eso es el Top-K.

Es la misma idea del KNN del laboratorio (Lab_Introduccion_Clasificacion_KNN): buscar los vecinos
más cercanos. La diferencia es que aquí la "distancia" se mide sobre los embeddings del encoder,
no sobre los píxeles.

Protocolo (galería y consultas separadas por usuario, como pide la pauta):
  - Galería = recortes de TRAIN (16 500 manos, 7 720 personas).
  - Consultas = recortes de VAL (para elegir) y de TEST (se reporta una vez).
  - Ninguna persona de las consultas está en la galería (se comprueba con assert), así que nunca
    se busca la propia imagen ni a la misma persona.
  - Similitud coseno: embeddings normalizados a largo 1 y producto punto (como Q K^T en atención).

Encoders que se comparan (cada uno responde una pregunta):
  - imagenet     : ResNet18 de ImageNet sin ajustar. ¿Basta un encoder genérico?
  - cnn_residual : nuestra CNN residual del Hito 1 (256 dims). ¿Sirve la red pequeña?
  - resnet18_ft  : ResNet18 con fine tuning del Hito 1 (512 dims). Encoder elegido en el Hito 1.
  - landmarks    : 42 números de la geometría de la mano (baseline sin CNN del Hito 1).

Métricas por consulta (relevante = vecino de la galería con el MISMO gesto):
  - Recall@K    : 1 si hay al menos un relevante en el Top-K (= Top-K accuracy de retrieval).
  - Precision@K : fracción de relevantes dentro del Top-K.
  - AP          : average precision sobre el ranking COMPLETO de la galería; mAP = promedio.
  - kNN@K       : el gesto se decide por votación de los K vecinos (como KNeighborsClassifier).

Uso: python -m src.retrieval                       # todos los encoders, val y test
     python -m src.retrieval --encoders resnet18_ft --limit 500   # prueba rápida

Salida en results/retrieval/: comparison.csv/.md, <encoder>/metrics_{val,test}.json,
per_class_test.csv, topk_ejemplos_test.png, precision_vs_k.png, pca_embeddings.png
Los embeddings quedan en data/embeddings/ (se regeneran solos si no están).
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from src.baseline_landmarks import features as landmark_features
from src.data import CLASSES, ROOT, HandCropDataset, build_transforms
from src.models import CNNResidualEncoder, ResNet18Encoder
from src.train import get_device

SEED = 42
KS = [1, 5, 10, 20]               # valores de K que se reportan
K_VOTE = 10                       # vecinos para la votación kNN
ENCODERS = {                      # nombre -> (clase, checkpoint del encoder o None)
    "imagenet": (ResNet18Encoder, None),
    "cnn_residual": (CNNResidualEncoder, "checkpoints/cnn_residual_scratch/encoder.pt"),
    "resnet18_ft": (ResNet18Encoder, "checkpoints/resnet18_finetune/encoder.pt"),
    "landmarks": (None, None),
}
EMB_DIR = ROOT / "data/embeddings"
OUT_DIR = ROOT / "results/retrieval"


# ---------------------------------------------------------------------------------------------
# 1. Embeddings
# ---------------------------------------------------------------------------------------------

# Carga solo el encoder (sin la cabeza de clasificación). Es lo que se guardó en el Hito 1
# justamente para poder reutilizarlo aquí.
def load_encoder(name, device):
    cls, path = ENCODERS[name]
    if name == "imagenet":
        encoder = cls(pretrained=True)                      # pesos IMAGENET1K_V1 de torchvision
    elif cls is CNNResidualEncoder:
        encoder = cls()
    else:
        encoder = cls(pretrained=False)
    if path:
        encoder.load_state_dict(torch.load(ROOT / path, map_location="cpu")["encoder"])
    return encoder.to(device).eval()


# Pasa una lista/lote de recortes 128x128 (tensores ya normalizados) por el encoder y deja cada
# vector con largo 1. Así el producto punto entre dos vectores ES la similitud coseno.
@torch.no_grad()
def embed(encoder, x, device):
    return F.normalize(encoder(x.to(device)).float(), dim=1).cpu()


# Embeddings de los 21 landmarks: mismas features del baseline del Hito 1 (centradas en la muñeca
# y escaladas). Las manos sin landmarks (MediaPipe falló) quedan marcadas con valid = False.
def landmark_embeddings(df):
    X, keep = landmark_features(df)
    emb = torch.zeros(len(df), X.shape[1])
    emb[torch.from_numpy(keep)] = F.normalize(torch.from_numpy(X).float(), dim=1)
    return emb, torch.from_numpy(keep)


# Calcula (o lee del disco) los embeddings de todos los recortes de un split.
def split_embeddings(name, split, device, recompute=False, num_workers=4):
    """Devuelve (df del split, embeddings [N, D], valid [N]) en el mismo orden que crops_<split>.csv."""
    df = pd.read_csv(ROOT / f"data/splits/crops_{split}.csv")
    path = EMB_DIR / f"{name}_{split}.pt"
    if path.exists() and not recompute:
        saved = torch.load(path)
        return df, saved["emb"], saved["valid"]

    if name == "landmarks":
        emb, valid = landmark_embeddings(df)
    else:
        encoder = load_encoder(name, device)
        dl = DataLoader(HandCropDataset(split, build_transforms(False, 128)), batch_size=256,
                        num_workers=num_workers)
        emb = torch.cat([embed(encoder, x, device) for x, _ in dl])
        valid = torch.ones(len(df), dtype=torch.bool)
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"emb": emb, "valid": valid}, path)
    return df, emb, valid


def labels_of(df):
    return torch.tensor([CLASSES.index(l) for l in df["label"]])


# ---------------------------------------------------------------------------------------------
# 2. Búsqueda Top-K y métricas
# ---------------------------------------------------------------------------------------------

# Busca los K vecinos más parecidos de cada consulta. La usa también el sistema final
# (imagen -> detector -> recorte -> embedding -> Top-K).
@torch.no_grad()
def search(q_emb, g_emb, k, device="cpu"):
    """q_emb [Nq, D], g_emb [Ng, D] normalizados -> (similitudes [Nq, k], índices [Nq, k])."""
    sims = q_emb.to(device) @ g_emb.to(device).T             # coseno = producto punto
    s, idx = sims.topk(k, dim=1)
    return s.cpu(), idx.cpu()


@torch.no_grad()
def ranking_metrics(q_emb, q_lab, g_emb, g_lab, device, batch=512):
    """Métricas por consulta. Ordena TODA la galería por similitud (necesario para el AP)."""
    g, gl = g_emb.to(device), g_lab.to(device)
    n_rel = torch.bincount(g_lab, minlength=len(CLASSES)).to(device)    # relevantes por gesto
    ranks = torch.arange(1, len(g) + 1, device=device)
    out = {f"R@{k}": [] for k in KS} | {f"P@{k}": [] for k in KS} | {"AP": [], "knn_pred": [], "top": []}
    for i in range(0, len(q_emb), batch):
        q, ql = q_emb[i:i + batch].to(device), q_lab[i:i + batch].to(device)
        order = (q @ g.T).argsort(dim=1, descending=True)              # galería de más a menos parecida
        rel = (gl[order] == ql[:, None]).float()                       # 1 si el vecino tiene el mismo gesto
        for k in KS:
            out[f"R@{k}"].append((rel[:, :k].sum(1) > 0).float())
            out[f"P@{k}"].append(rel[:, :k].mean(1))
        # AP = promedio de la precisión acumulada en cada posición donde aparece un relevante
        out["AP"].append(((rel.cumsum(1) / ranks) * rel).sum(1) / n_rel[ql])
        # kNN: el gesto más repetido entre los K_VOTE vecinos (empate -> el de índice menor)
        out["knn_pred"].append(gl[order[:, :K_VOTE]].mode(dim=1).values)
        out["top"].append(order[:, :max(KS)])
    return {k: torch.cat(v).cpu().numpy() for k, v in out.items()}


# Resume las métricas por consulta. Las consultas sin embedding válido (solo pasa con landmarks)
# cuentan como fallo: así se compara con los encoders que sí procesan todas las manos.
def summarize(per_q, q_lab, valid_q):
    n = len(q_lab)
    full = {}
    for key, v in per_q.items():
        if key == "top":
            continue
        arr = np.full(n, -1 if key == "knn_pred" else 0.0)
        arr[valid_q] = v
        full[key] = arr
    y = q_lab.numpy()
    res = {"n_queries": int(n), "coverage": float(valid_q.mean())}
    res |= {k: float(full[k].mean()) for k in full if k not in ("knn_pred",)}
    res["mAP"] = res.pop("AP")
    res[f"kNN{K_VOTE}_accuracy"] = float((full["knn_pred"] == y).mean())
    res[f"kNN{K_VOTE}_macro_f1"] = float(f1_score(y, full["knn_pred"], labels=range(len(CLASSES)), average="macro",
                                                   zero_division=0))
    return res, full


def per_class_table(full, q_lab):
    df = pd.DataFrame({"label": [CLASSES[i] for i in q_lab.numpy()], "R@1": full["R@1"],
                       "P@10": full["P@10"], "AP": full["AP"]})
    return df.groupby("label").mean().reindex(CLASSES).round(4)


# Errores más frecuentes del Top-1: gesto de la consulta -> gesto del vecino más parecido.
def top1_confusions(top, q_lab, g_lab, n=8):
    nn1 = g_lab.numpy()[top[:, 0]]
    y = q_lab.numpy()
    err = pd.Series([f"{CLASSES[a]} -> {CLASSES[b]}" for a, b in zip(y, nn1) if a != b])
    return err.value_counts().head(n).to_dict()


# ---------------------------------------------------------------------------------------------
# 3. Figuras
# ---------------------------------------------------------------------------------------------

# Consulta + sus 5 vecinos de la galería. Borde verde = mismo gesto, rojo = otro gesto.
# Se muestran consultas acertadas (Top-1 correcto) y falladas, elegidas al azar con semilla fija.
def plot_topk(q_df, g_df, top, q_lab, g_lab, path, n_ok=4, n_bad=4, k=5):
    rng = np.random.default_rng(SEED)
    ok = np.where(g_lab.numpy()[top[:, 0]] == q_lab.numpy())[0]
    bad = np.where(g_lab.numpy()[top[:, 0]] != q_lab.numpy())[0]
    rows = list(rng.choice(ok, min(n_ok, len(ok)), replace=False)) + \
        list(rng.choice(bad, min(n_bad, len(bad)), replace=False))
    fig, axes = plt.subplots(len(rows), k + 1, figsize=((k + 1) * 1.6, len(rows) * 1.8))
    for r, qi in enumerate(rows):
        ax = axes[r, 0]
        ax.imshow(Image.open(ROOT / q_df.path.iloc[qi]))
        ax.set_title(f"CONSULTA\n{q_df.label.iloc[qi]}", fontsize=7, fontweight="bold")
        for j in range(k):
            gi = top[qi, j]
            ax = axes[r, j + 1]
            ax.imshow(Image.open(ROOT / g_df.path.iloc[gi]))
            same = g_lab[gi] == q_lab[qi]
            ax.set_title(f"#{j + 1} {g_df.label.iloc[gi]}", fontsize=7, color="green" if same else "red")
            for s in ax.spines.values():
                s.set_edgecolor("green" if same else "red")
                s.set_linewidth(3)
        for ax in axes[r]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("Top-5 en la galería (train) para consultas de test. Arriba aciertos, abajo fallos", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_precision_vs_k(curves, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, (ks, p) in curves.items():
        ax.plot(ks, p, marker="o", ms=3, label=name)
    ax.set_xlabel("K (vecinos recuperados)")
    ax.set_ylabel("Precision@K (test)")
    ax.set_title("Fracción de vecinos con el mismo gesto")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# PCA a 2 dimensiones (como en los laboratorios) de los embeddings de test, coloreado por gesto.
# Si el encoder separa bien los gestos, deberían verse nubes por color.
def plot_pca(embs, q_lab, path):
    fig, axes = plt.subplots(1, len(embs), figsize=(4.5 * len(embs), 4.3))
    cmap = plt.get_cmap("tab20")
    for ax, (name, e) in zip(np.atleast_1d(axes), embs.items()):
        z = PCA(n_components=2, random_state=SEED).fit_transform(e.numpy())
        for c in range(len(CLASSES)):
            m = q_lab.numpy() == c
            ax.scatter(z[m, 0], z[m, 1], s=3, alpha=0.5, color=cmap(c), label=CLASSES[c])
        ax.set_title(f"PCA de embeddings (test): {name}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    np.atleast_1d(axes)[-1].legend(markerscale=4, fontsize=7, loc="center left", bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------
# 4. Experimento
# ---------------------------------------------------------------------------------------------

def run_encoder(name, device, args):
    g_df, g_emb, g_valid = split_embeddings(name, "train", device, args.recompute, args.num_workers)
    g_df, g_emb = g_df[g_valid.numpy()].reset_index(drop=True), g_emb[g_valid]   # galería: solo válidos
    g_lab = labels_of(g_df)
    rows = {}
    for split in ["val", "test"]:
        q_df, q_emb, q_valid = split_embeddings(name, split, device, args.recompute, args.num_workers)
        if args.limit:
            q_df, q_emb, q_valid = q_df.iloc[:args.limit], q_emb[:args.limit], q_valid[:args.limit]
        # Galería y consultas separadas por usuario: nadie aparece en ambas
        assert not set(q_df.user_id) & set(g_df.user_id), "hay personas en la galería y en las consultas"
        q_lab = labels_of(q_df)
        vq = q_valid.numpy()
        per_q = ranking_metrics(q_emb[q_valid], q_lab[q_valid], g_emb, g_lab, device)
        res, full = summarize(per_q, q_lab, vq)
        res |= {"encoder": name, "split": split, "dim": int(g_emb.shape[1]), "gallery_size": len(g_df),
                "top1_confusions": top1_confusions(per_q["top"], q_lab[q_valid], g_lab)}
        rows[split] = res

        out = OUT_DIR / name
        out.mkdir(parents=True, exist_ok=True)
        if not args.limit:                                   # la prueba rápida no guarda nada
            json.dump(res, open(out / f"metrics_{split}.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        if split == "test" and not args.limit:
            per_class_table(full, q_lab).to_csv(out / "per_class_test.csv")
            plot_topk(q_df[vq].reset_index(drop=True), g_df, per_q["top"], q_lab[q_valid], g_lab,
                      out / "topk_ejemplos_test.png")
            # Precision@K para K = 1..50 (curva), solo con la búsqueda Top-K
            _, idx = search(q_emb[q_valid], g_emb, 50, device)
            rel = (g_lab[idx] == q_lab[q_valid][:, None]).float()
            p_curve = (rel.cumsum(1) / torch.arange(1, 51)).sum(0) / len(q_lab)   # inválidas = 0
            rows["curve"] = (list(range(1, 51)), p_curve.numpy())
            rows["test_emb"] = (q_emb, q_lab)
        print(f"[{name}] {split}: mAP {res['mAP']:.4f} | R@1 {res['R@1']:.4f} | P@10 {res['P@10']:.4f} | "
              f"kNN{K_VOTE} F1 {res[f'kNN{K_VOTE}_macro_f1']:.4f} | cobertura {res['coverage']:.3f}")
    return rows


def main(args):
    torch.manual_seed(SEED)
    device = get_device()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table, curves, test_embs = [], {}, {}
    for name in args.encoders:
        rows = run_encoder(name, device, args)
        for split in ["val", "test"]:
            r = rows[split]
            table.append({"encoder": name, "split": split, "dim": r["dim"], "coverage": r["coverage"],
                          "mAP": r["mAP"], **{f"R@{k}": r[f"R@{k}"] for k in KS},
                          **{f"P@{k}": r[f"P@{k}"] for k in KS},
                          f"kNN{K_VOTE}_acc": r[f"kNN{K_VOTE}_accuracy"], f"kNN{K_VOTE}_F1": r[f"kNN{K_VOTE}_macro_f1"]})
        if args.limit:
            continue
        curves[name] = rows["curve"]
        if name != "landmarks":
            test_embs[name] = rows["test_emb"]

    df = pd.DataFrame(table).round(4)
    if args.limit:                                           # prueba rápida: no pisa los resultados
        print(df.to_string(index=False))
        return
    df.to_csv(OUT_DIR / "comparison.csv", index=False)
    # El encoder se ELIGE con validación (mAP); test solo se reporta
    val = df[df.split == "val"].sort_values("mAP", ascending=False)
    best = val.encoder.iloc[0]
    with open(OUT_DIR / "comparison.md", "w", encoding="utf-8") as f:     # utf-8: tildes bien también en Windows
        f.write("# Hito 3 · Retrieval: galería = train, consultas = val/test (personas distintas)\n\n")
        f.write(f"Encoder elegido por mAP en validación: **{best}**\n\n")
        for split in ["val", "test"]:
            f.write(f"## {split}\n\n{df[df.split == split].drop(columns='split').to_markdown(index=False)}\n\n")
    plot_precision_vs_k(curves, OUT_DIR / "precision_vs_k.png")
    if test_embs:
        q_lab = next(iter(test_embs.values()))[1]
        plot_pca({n: e for n, (e, _) in test_embs.items()}, q_lab, OUT_DIR / "pca_embeddings.png")
    print(df.to_string(index=False))
    print(f"\nEncoder elegido (mAP val): {best}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="+", default=list(ENCODERS), choices=list(ENCODERS))
    ap.add_argument("--limit", type=int, default=0, help="prueba rápida: solo las primeras N consultas")
    ap.add_argument("--recompute", action="store_true", help="recalcular embeddings aunque existan")
    ap.add_argument("--num_workers", type=int, default=4)
    main(ap.parse_args())
