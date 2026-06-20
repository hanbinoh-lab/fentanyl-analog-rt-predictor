"""Mordred stage 1: any-NaN column drop + constant column drop (NaN-only policy)."""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from features.lib.io import PROJECT_ROOT  # noqa: E402
from features.lib.seed import seed_all  # noqa: E402

REP_ID = "mordred"
PRE_DIR = PROJECT_ROOT / "data" / "features" / REP_ID / f"{REP_ID}_preprocessing"
IN_CSV = PRE_DIR / "00_compute.csv"
OUT_CSV = PRE_DIR / "01_filter.csv"
META_COLS = ("compound", "smiles", "retention_time_min")


def main():
    seed_all(42)
    assert IN_CSV.exists(), f"mordred stage 1: input missing - {IN_CSV.relative_to(PROJECT_ROOT)}"
    df = pd.read_csv(IN_CSV)
    feat_cols = [c for c in df.columns if c not in META_COLS]

    nan_counts = df[feat_cols].isnull().sum()
    nan_cols = nan_counts[nan_counts > 0].index.tolist()
    after_nan = [c for c in feat_cols if c not in nan_cols]
    const_cols = [c for c in after_nan if df[c].nunique(dropna=True) <= 1]
    keep = [c for c in after_nan if c not in const_cols]

    df_out = pd.concat(
        [
            df[["compound", "smiles"]].reset_index(drop=True),
            df[keep].reset_index(drop=True),
            df[["retention_time_min"]].reset_index(drop=True),
        ],
        axis=1,
    )
    df_out.to_csv(OUT_CSV, index=False)


if __name__ == "__main__":
    main()
