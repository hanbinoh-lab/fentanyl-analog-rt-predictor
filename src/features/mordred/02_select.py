"""Mordred stage 2: RT-mechanism chemistry-prior descriptor selection (groups defined in YAML)."""

import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from features.lib.io import PROJECT_ROOT  # noqa: E402
from features.lib.seed import seed_all  # noqa: E402

REP_ID = "mordred"
PRE_DIR = PROJECT_ROOT / "data" / "features" / REP_ID / f"{REP_ID}_preprocessing"
IN_CSV = PRE_DIR / "01_filter.csv"
OUT_CSV = PRE_DIR / "02_select.csv"
YAML_PATH = PROJECT_ROOT / "configs" / "mordred_mechanism_groups.yaml"
META_COLS = ("compound", "smiles", "retention_time_min")


def main():
    seed_all(42)
    assert IN_CSV.exists(), f"mordred stage 2: input missing - {IN_CSV.relative_to(PROJECT_ROOT)}"
    assert YAML_PATH.exists(), f"mordred stage 2: yaml missing - {YAML_PATH.relative_to(PROJECT_ROOT)}"

    df = pd.read_csv(IN_CSV)
    feat_cols = [c for c in df.columns if c not in META_COLS]

    with open(YAML_PATH, "r", encoding="utf-8") as f:
        groups = yaml.safe_load(f)
    listed = []
    for ginfo in groups.values():
        listed.extend(ginfo["descriptors"])
    listed_unique = list(dict.fromkeys(listed))
    available = [d for d in listed_unique if d in feat_cols]

    df_out = pd.concat(
        [
            df[["compound", "smiles"]].reset_index(drop=True),
            df[available].reset_index(drop=True),
            df[["retention_time_min"]].reset_index(drop=True),
        ],
        axis=1,
    )
    df_out.to_csv(OUT_CSV, index=False)


if __name__ == "__main__":
    main()
