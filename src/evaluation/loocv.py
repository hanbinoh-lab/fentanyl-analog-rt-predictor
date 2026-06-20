"""Stage 4a: LOOCV sensitivity per candidate (HP fixed = Stage 2 final HP)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.algorithm_registry import make_estimator  # noqa: E402
from evaluation.evaluate import mae, rmse  # noqa: E402
from features.lib.io import load_base  # noqa: E402


def run_loocv(
    X,
    y,
    rep_id: str,
    algo_id: str,
    final_hp: dict,
    out_dir,
    random_state: int = 42,
) -> dict:
    X = np.asarray(X)
    y = np.asarray(y, dtype=float)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(y)

    y_loo = np.empty(n)
    for i in range(n):
        X_tr = np.delete(X, i, axis=0)
        y_tr = np.delete(y, i, axis=0)
        est = make_estimator(algo_id, rep_id, random_state)
        est.set_params(**final_hp)
        est.fit(X_tr, y_tr)
        y_loo[i] = float(np.asarray(est.predict(X[i:i + 1])).ravel()[0])

    residuals = y - y_loo
    press = float(np.sum(residuals ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    q2_loo = 1.0 - press / sst

    base = load_base()
    results_path = out_dir / "results_loocv.csv"
    pd.DataFrame({
        "compound": base["compound"].tolist(),
        "compound_idx": np.arange(n),
        "y_true": y,
        "y_loo_pred": y_loo,
        "residual": residuals,
        "abs_residual": np.abs(residuals),
    }).to_csv(results_path, index=False)

    return {
        "results_loocv_path": results_path,
        "q2_loo": float(q2_loo),
        "loocv_rmse": rmse(y, y_loo),
        "loocv_mae": mae(y, y_loo),
        "per_sample_residuals": residuals,
    }
