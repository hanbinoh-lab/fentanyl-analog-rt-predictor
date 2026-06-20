"""Shared I/O: project root, relative path helper, base.csv loader."""

import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BASE_CSV = PROJECT_ROOT / "data" / "raw" / "base.csv"

META_COLS = ("compound", "smiles", "retention_time_min")


def rel(path) -> Path:
    p = Path(path)
    return p.relative_to(PROJECT_ROOT) if p.is_absolute() else p


def load_base() -> pd.DataFrame:
    df = pd.read_csv(BASE_CSV)
    assert df.shape == (94, 3), f"base.csv shape mismatch: {df.shape}, expected (94, 3)"
    assert list(df.columns) == list(META_COLS), \
        f"base.csv columns mismatch: {list(df.columns)}"
    return df
