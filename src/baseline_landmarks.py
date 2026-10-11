"""Baseline sin CNN: clasifica el gesto a partir de los 21 landmarks de MediaPipe que trae HaGRID.

Features: coordenadas (x, y) en píxeles, centradas en la muñeca (punto 0) y escaladas por la
distancia máxima a la muñeca -> invariante a posición y tamaño de la mano. 42 valores.

Las manos sin landmarks (MediaPipe falló) no se pueden clasificar con este método: se reportan
las métricas sobre las manos con landmarks y también contando las demás como error.

Uso: python -m src.baseline_landmarks
"""

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data import CLASSES, ROOT

SEED = 42


def features(df: pd.DataFrame):
    X, keep = [], []
    for r in df.itertuples():
        lm = json.loads(r.landmarks)
        if len(lm) != 21:
            keep.append(False)
            continue
        pts = np.array(lm) * [r.img_w, r.img_h]
        pts = pts - pts[0]
        pts = pts / (np.linalg.norm(pts, axis=1).max() + 1e-6)
        X.append(pts.flatten())
        keep.append(True)
    return np.array(X), np.array(keep)


def main():
    data = {}
    for split in ["train", "val", "test"]:
        df = pd.read_csv(ROOT / f"data/splits/crops_{split}.csv")
        X, keep = features(df)
        y = np.array([CLASSES.index(l) for l in df["label"]])
        data[split] = (X, y[keep], y, keep)
        print(f"{split}: {keep.mean():.1%} de las manos tienen landmarks")

    candidates = {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
    }
    out_dir = ROOT / "results/baseline_landmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for name, clf in candidates.items():
        clf.fit(data["train"][0], data["train"][1])
        row = {"name": f"landmarks_{name}"}
        for split in ["val", "test"]:
            X, y_kept, y_all, keep = data[split]
            p = clf.predict(X)
            # Versión estricta: manos sin landmarks = error (predicción -1)
            p_all = np.full(len(y_all), -1)
            p_all[keep] = p
            row[f"{split}_macro_f1_con_landmarks"] = f1_score(y_kept, p, average="macro")
            row[f"{split}_accuracy_con_landmarks"] = accuracy_score(y_kept, p)
            row[f"{split}_accuracy_todas"] = accuracy_score(y_all, p_all)
            row[f"{split}_balanced_accuracy_todas"] = balanced_accuracy_score(y_all, p_all)
            row[f"{split}_macro_f1_todas"] = f1_score(y_all, p_all, labels=range(len(CLASSES)), average="macro")
        results.append(row)
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})
    pd.DataFrame(results).round(4).to_csv(out_dir / "metrics.csv", index=False)


if __name__ == "__main__":
    main()
