"""Junta las métricas de todos los experimentos en una tabla y grafica las curvas de entrenamiento.

Uso: python -m src.compare
Salida: results/comparison.csv, results/comparison.md, results/learning_curves.png
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.data import ROOT

EXPERIMENTS = ["cnn_residual_scratch", "resnet18_scratch", "resnet18_linear_probe", "resnet18_finetune"]


def main():
    rows = []
    base = ROOT / "results/baseline_landmarks/metrics.csv"
    if base.exists():
        for r in pd.read_csv(base).itertuples():
            rows.append({"modelo": r.name, "accuracy": r.test_accuracy_todas,
                         "balanced_accuracy": r.test_balanced_accuracy_todas, "macro_f1": r.test_macro_f1_todas,
                         "macro_f1_solo_con_landmarks": r.test_macro_f1_con_landmarks})
    for name in EXPERIMENTS:
        path = ROOT / f"results/{name}/metrics_test.json"
        if not path.exists():
            continue
        m = json.load(open(path))
        rows.append({"modelo": name, "accuracy": m["accuracy"], "balanced_accuracy": m["balanced_accuracy"],
                     "macro_precision": m["macro_precision"], "macro_recall": m["macro_recall"],
                     "macro_f1": m["macro_f1"], "params_M": m["params"] / 1e6, "dim_embedding": m["encoder_dim"],
                     "latencia_ms_cpu": m["latency_ms_cpu"], "latencia_ms_mps": m["latency_ms_mps"],
                     "mejor_epoca": m["best_epoch"], **{f"acc_{k}": v for k, v in m["accuracy_by_hand_size"].items()}})
    df = pd.DataFrame(rows).round(4)
    df.to_csv(ROOT / "results/comparison.csv", index=False)
    (ROOT / "results/comparison.md").write_text(df.to_markdown(index=False))
    print(df.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name in EXPERIMENTS:
        path = ROOT / f"checkpoints/{name}/history.csv"
        if not path.exists():
            continue
        h = pd.read_csv(path)
        axes[0].plot(h["epoch"], h["train_loss"], "--", label=f"{name} (train)")
        axes[0].plot(h["epoch"], h["val_loss"], label=f"{name} (val)")
        axes[1].plot(h["epoch"], h["val_macro_f1"], label=name)
    axes[0].set(xlabel="época", ylabel="loss", title="Loss")
    axes[1].set(xlabel="época", ylabel="macro-F1", title="Macro-F1 en validación")
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "results/learning_curves.png", dpi=150)


if __name__ == "__main__":
    main()
