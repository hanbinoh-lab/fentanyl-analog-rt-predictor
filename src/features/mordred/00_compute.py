"""Mordred 2D+3D descriptor calculation (ignore_3D=False)."""

import sys
from pathlib import Path

import pandas as pd
from mordred import Calculator, descriptors
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from features.lib.conformer import generate_lowest_e_conformer  # noqa: E402
from features.lib.io import PROJECT_ROOT, load_base  # noqa: E402
from features.lib.seed import seed_all  # noqa: E402

REP_ID = "mordred"
PRE_DIR = PROJECT_ROOT / "data" / "features" / REP_ID / f"{REP_ID}_preprocessing"
OUT_CSV = PRE_DIR / "00_compute.csv"


def main():
    seed_all(42)
    df_base = load_base()

    calc = Calculator(descriptors, ignore_3D=False)

    rows = []
    for name, smi in zip(df_base["compound"], df_base["smiles"]):
        mol = Chem.MolFromSmiles(smi)
        assert mol is not None, f"{REP_ID}: SMILES parse failed for {name!r}: {smi!r}"
        mol_3d = generate_lowest_e_conformer(mol)
        result = calc(mol_3d)
        rows.append({str(k): v for k, v in result.asdict().items()})

    df_feat = pd.DataFrame(rows)
    # mordred Missing/Error implement __float__ -> nan; to_numeric also captures inf/nan from
    # divide-by-zero descriptor formulas.
    df_feat = df_feat.apply(pd.to_numeric, errors="coerce")

    df_out = pd.concat(
        [
            df_base[["compound", "smiles"]].reset_index(drop=True),
            df_feat.reset_index(drop=True),
            df_base[["retention_time_min"]].reset_index(drop=True),
        ],
        axis=1,
    )

    PRE_DIR.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUT_CSV, index=False)


if __name__ == "__main__":
    main()
