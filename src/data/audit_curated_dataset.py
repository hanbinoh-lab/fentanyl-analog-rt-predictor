"""Post-curation checks for base.csv."""

from pathlib import Path

import pandas as pd
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[2]

SOURCE = ROOT / "data" / "raw" / "source_kim2025.csv"
BASE = ROOT / "data" / "raw" / "base.csv"


def audit(df_src, df_base):
    expected_columns = ["compound", "smiles", "retention_time_min"]
    if list(df_base.columns) != expected_columns:
        raise ValueError(f"base columns mismatch: {list(df_base.columns)}")

    nulls = int(df_base.isnull().sum().sum())
    if nulls:
        raise ValueError(f"base has {nulls} null cell(s)")

    for i, smi in enumerate(df_base["smiles"]):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"SMILES parse failed at row {i}: {smi!r}")
        canon = Chem.MolToSmiles(mol, canonical=True)
        if canon != smi:
            raise ValueError(
                f"non-canonical SMILES at row {i}: "
                f"stored={smi!r}, canonical={canon!r}"
            )

    dupe_names = df_base["compound"][df_base["compound"].duplicated()].tolist()
    if dupe_names:
        raise ValueError(f"duplicate compound name(s): {dupe_names}")

    src_set = set(df_src["compound"])
    base_set = set(df_base["compound"])
    if src_set != base_set:
        raise ValueError(
            f"source/base compound mismatch: "
            f"source-only={src_set - base_set}, base-only={base_set - src_set}"
        )

    src_pairs = set(zip(df_src["compound"], df_src["retention_time_min"]))
    base_pairs = set(zip(df_base["compound"], df_base["retention_time_min"]))
    if src_pairs != base_pairs:
        raise ValueError("source/base retention_time_min mismatch")


def main():
    df_src = pd.read_csv(SOURCE)
    df_base = pd.read_csv(BASE)

    audit(df_src, df_base)
    print("curated dataset check passed")


if __name__ == "__main__":
    main()
