"""Mordred stage 3: |r|>=0.90 average-linkage hierarchical clustering + mechanism-priority representative pick.

Intra-cluster representative uses mechanism strength order (dominant > auxiliary > partial > cross)
with alphabetical tie-break. This is NOT a geometric medoid (no intra-cluster centrality is computed).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from features.lib.io import PROJECT_ROOT  # noqa: E402
from features.lib.seed import seed_all  # noqa: E402

REP_ID = "mordred"
PRE_DIR = PROJECT_ROOT / "data" / "features" / REP_ID / f"{REP_ID}_preprocessing"
IN_CSV = PRE_DIR / "02_select.csv"
OUT_CSV = PROJECT_ROOT / "data" / "features" / REP_ID / f"{REP_ID}.csv"
YAML_PATH = PROJECT_ROOT / "configs" / "mordred_mechanism_groups.yaml"
META_COLS = ("compound", "smiles", "retention_time_min")
R_THRESHOLD = 0.90
DIST_THRESHOLD = 1.0 - R_THRESHOLD
STRENGTH_RANK = {"dominant": 0, "auxiliary": 1, "partial": 2, "cross": 3}


def main():
    seed_all(42)
    assert IN_CSV.exists(), f"mordred stage 3: input missing - {IN_CSV.relative_to(PROJECT_ROOT)}"
    assert YAML_PATH.exists(), f"mordred stage 3: yaml missing - {YAML_PATH.relative_to(PROJECT_ROOT)}"

    df = pd.read_csv(IN_CSV)
    feat_cols_all = [c for c in df.columns if c not in META_COLS]

    nan_counts = df[feat_cols_all].isnull().sum()
    nan_cols = nan_counts[nan_counts > 0].index.tolist()
    feat_cols = [c for c in feat_cols_all if c not in nan_cols]
    X = df[feat_cols].to_numpy(dtype=np.float64)

    with open(YAML_PATH, "r", encoding="utf-8") as f:
        groups = yaml.safe_load(f)
    desc_strength = {}
    for ginfo in groups.values():
        s = ginfo["strength"]
        for d in ginfo["descriptors"]:
            desc_strength[d] = s

    r = np.corrcoef(X.T)
    r = np.nan_to_num(r, nan=0.0)
    dist = 1.0 - np.abs(r)
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    labels = fcluster(Z, t=DIST_THRESHOLD, criterion="distance")
    n_clusters = int(labels.max())

    representatives = []
    for cl in range(1, n_clusters + 1):
        members = [feat_cols[i] for i, lab in enumerate(labels) if lab == cl]
        members.sort(key=lambda d: (STRENGTH_RANK.get(desc_strength.get(d, "cross"), 99), d))
        representatives.append(members[0])
    assert len(representatives) == n_clusters

    df_out = pd.concat(
        [
            df[["compound", "smiles"]].reset_index(drop=True),
            df[representatives].reset_index(drop=True),
            df[["retention_time_min"]].reset_index(drop=True),
        ],
        axis=1,
    )
    p = len(representatives)
    assert df_out.shape == (94, p + 3)
    assert int(df_out.isnull().sum().sum()) == 0

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUT_CSV, index=False)


if __name__ == "__main__":
    main()
