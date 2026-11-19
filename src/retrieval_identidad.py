"""Hito 3 (Persona 1): ¿el embedding representa el GESTO o la IDENTIDAD de la persona?

En simple: el enunciado pide verificar que el encoder no esté "reconociendo a la persona" (su piel,
su ropa, su pieza) en vez del gesto. En la galería de train eso no se puede ver, porque ahí no hay
nadie del test. Por eso este análisis se hace DENTRO del test, donde cada persona aparece en varias
fotos con gestos distintos (1 007 personas, ~5 manos cada una).

Para cada mano de test (consulta) se comparan las otras manos de test, sin contar las de la
misma foto (la otra mano de la misma foto tiene el mismo fondo y sería trampa). Cada par cae en
uno de estos grupos:
  - mismo gesto, otra persona   -> lo que queremos que quede cerca
  - misma persona, otro gesto   -> si queda cerca, el embedding guarda identidad
  - misma persona, mismo gesto  -> casi siempre manos no_gesture de otra foto de la misma persona
  - otra persona, otro gesto    -> lo que debería quedar lejos

Se reporta:
  1. Similitud coseno media de cada grupo.
  2. Composición del Top-10: qué fracción de los vecinos cae en cada grupo, comparada con lo que
     saldría por azar (la fracción de ese grupo entre todos los candidatos).
  3. "Ventaja de identidad": entre las manos de otro gesto, ¿qué tan arriba queda la persona
     correcta? Se mide como el percentil medio de la mano de la misma persona dentro del ranking de
     manos de otro gesto (50 % = azar: el embedding no sabe quién es; 100 % = siempre la primera).

Uso: python -m src.retrieval_identidad     (usa los embeddings que dejó src/retrieval.py)

Salida en results/retrieval/identidad/: identidad.csv, resumen.md, identidad_vs_gesto.png,
identidad_ejemplos.png
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.data import ROOT
from src.retrieval import SEED, labels_of, split_embeddings
from src.train import get_device

K = 10
OUT_DIR = ROOT / "results/retrieval/identidad"
GROUPS = ["mismo gesto, otra persona", "misma persona, otro gesto", "misma persona, mismo gesto",
          "otra persona, otro gesto"]


# Matrices [N, N] de verdadero/falso: ¿mismo gesto?, ¿misma persona?, ¿par válido?
def pair_masks(df):
    lab = labels_of(df)
    users = torch.tensor(pd.factorize(df.user_id)[0])
    imgs = torch.tensor(pd.factorize(df.image_id)[0])
    same_g = lab[:, None] == lab[None, :]
    same_u = users[:, None] == users[None, :]
    valid = imgs[:, None] != imgs[None, :]          # descarta la misma foto (y la diagonal)
    return same_g, same_u, valid


def group_masks(same_g, same_u, valid):
    return [same_g & ~same_u & valid, ~same_g & same_u & valid, same_g & same_u & valid, ~same_g & ~same_u & valid]


@torch.no_grad()
def analyze(name, df, emb, device):
    same_g, same_u, valid = [m.to(device) for m in pair_masks(df)]
    groups = group_masks(same_g, same_u, valid)
    sims = emb.to(device) @ emb.to(device).T
    row = {"encoder": name}

    # 1. Similitud media por grupo
    for g, m in zip(GROUPS, groups):
        row[f"sim media: {g}"] = float(sims[m].mean())

    # 2. Composición del Top-K vs azar
    s = sims.masked_fill(~valid, -2)                 # los pares inválidos nunca entran al Top-K
    top = s.topk(K, dim=1).indices
    n_valid = valid.sum(1, keepdim=True).float()
    for g, m in zip(GROUPS, groups):
        row[f"top{K}: {g}"] = float(m.gather(1, top).float().mean())
        row[f"azar: {g}"] = float((m.sum(1, keepdim=True) / n_valid).mean())

    # 3. Ventaja de identidad: solo entre manos de OTRO gesto, ¿la misma persona queda arriba?
    other_g = ~same_g & valid
    s_other = sims.masked_fill(~other_g, float("nan"))
    target = groups[1]                                # misma persona, otro gesto
    has = target.any(1)
    # para cada par (consulta, mano de la misma persona) se cuenta cuántas manos de otro gesto
    # tienen MENOR similitud -> percentil (0.5 = azar)
    pct = []
    for i in torch.where(has)[0].tolist():
        cand = s_other[i][other_g[i]]                 # similitudes con todas las manos de otro gesto
        vals = sims[i][target[i]]                     # similitudes con la misma persona (otro gesto)
        pct.append((cand[None, :] < vals[:, None]).float().mean(1).cpu())
    pct = torch.cat(pct).numpy()
    row["percentil medio de la misma persona (50% = azar)"] = float(np.mean(pct))
    row[f"consultas con la misma persona en top{K} (otro gesto)"] = float(target.gather(1, top).any(1).float().mean())
    return row, sims.cpu(), top.cpu()


# Para algunas consultas muestra el Top-5 (dentro del test) de dos encoders. Borde azul =
# misma persona con otro gesto (identidad), verde = mismo gesto, rojo = otro gesto y otra persona.
def plot_examples(df, results, path, n=4, k=5):
    names = list(results)
    same_g, same_u, valid = pair_masks(df)
    ident = same_u & ~same_g & valid
    # consultas donde el primer encoder trae a la misma persona en su Top-5 (casos interesantes)
    top0 = results[names[0]][1]
    cand = torch.where(ident.gather(1, top0[:, :k]).any(1))[0].numpy()
    rng = np.random.default_rng(SEED)
    rows = rng.choice(cand, min(n, len(cand)), replace=False) if len(cand) else rng.choice(len(df), n, replace=False)
    fig, axes = plt.subplots(len(rows) * len(names), k + 1, figsize=((k + 1) * 1.5, len(rows) * len(names) * 1.7))
    r = 0
    for qi in rows:
        for name in names:
            top = results[name][1]
            ax = axes[r, 0]
            ax.imshow(Image.open(ROOT / df.path.iloc[qi]))
            ax.set_title(f"{name}\nconsulta: {df.label.iloc[qi]}", fontsize=6.5, fontweight="bold")
            for j in range(k):
                gi = int(top[qi, j])
                if same_u[qi, gi] and not same_g[qi, gi]:
                    color, tag = "blue", "MISMA PERSONA"
                elif same_g[qi, gi]:
                    color, tag = "green", ""
                else:
                    color, tag = "red", ""
                ax = axes[r, j + 1]
                ax.imshow(Image.open(ROOT / df.path.iloc[gi]))
                ax.set_title(f"#{j + 1} {df.label.iloc[gi]}\n{tag}", fontsize=6.5, color=color)
                for s in ax.spines.values():
                    s.set_edgecolor(color)
                    s.set_linewidth(3)
            for ax in axes[r]:
                ax.set_xticks([])
                ax.set_yticks([])
            r += 1
    fig.suptitle("Top-5 dentro de test (sin la misma foto). Azul = misma persona con otro gesto", fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.985])          # deja espacio arriba para el título
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_bars(df_res, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(df_res))
    w = 0.2
    for i, g in enumerate(GROUPS):
        axes[0].bar(x + (i - 1.5) * w, df_res[f"sim media: {g}"], w, label=g)
        axes[1].bar(x + (i - 1.5) * w, df_res[f"top{K}: {g}"], w, label=g)
    axes[0].set_title("Similitud coseno media por tipo de par")
    axes[1].set_title(f"Composición del Top-{K} (fracción de vecinos)")
    for ax in axes:
        ax.set_xticks(x, df_res.encoder)
        ax.grid(axis="y", alpha=0.3)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, ncol=4, fontsize=8, loc="lower center")     # leyenda abajo, sin tapar barras
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(args):
    device = get_device()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, results = [], {}
    for name in args.encoders:
        df, emb, _ = split_embeddings(name, "test", device)
        row, sims, top = analyze(name, df, emb, device)
        rows.append(row)
        results[name] = (sims, top)
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})

    df_res = pd.DataFrame(rows)
    df_res.round(4).to_csv(OUT_DIR / "identidad.csv", index=False)
    plot_bars(df_res, OUT_DIR / "identidad_vs_gesto.png")
    plot_examples(df, results, OUT_DIR / "identidad_ejemplos.png")
    with open(OUT_DIR / "resumen.md", "w", encoding="utf-8") as f:       # utf-8: tildes bien también en Windows
        f.write("# Hito 3 · ¿Gesto o identidad? (análisis dentro de test, sin pares de la misma foto)\n\n")
        f.write(df_res.set_index("encoder").T.round(4).to_markdown() + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # landmarks no se incluye: el ~2 % de manos sin landmarks no tiene embedding y ensuciaría los pares
    ap.add_argument("--encoders", nargs="+", default=["imagenet", "cnn_residual", "resnet18_ft"],
                    choices=["imagenet", "cnn_residual", "resnet18_ft"])
    main(ap.parse_args())
