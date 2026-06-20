"""Stage 4b: kNN-distance applicability domain (standardized Euclidean)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from features.lib.io import load_base  # noqa: E402


def run_ad(
    X,
    out_dir,
    k: int = 5,
    percentile: float = 95.0,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_std = StandardScaler().fit_transform(np.asarray(X, dtype=float))
    D = euclidean_distances(X_std, X_std)

    D = D.copy()
    np.fill_diagonal(D, np.inf)
    knn = np.sort(D, axis=1)[:, :k].mean(axis=1)
    threshold = float(np.percentile(knn, percentile))
    in_domain = knn <= threshold

    base = load_base()
    results_path = out_dir / "results_ad.csv"
    pd.DataFrame({
        "compound": base["compound"].tolist(),
        "compound_idx": np.arange(len(knn)),
        "knn_distance_mean": knn,
        "in_chemical_domain": in_domain.astype(bool),
    }).to_csv(results_path, index=False)

    return {
        "results_ad_path": results_path,
        "knn_dist_p95_train": threshold,
        "in_chemical_domain_rate": float(in_domain.mean()),
        "per_compound_distances": knn,
    }
