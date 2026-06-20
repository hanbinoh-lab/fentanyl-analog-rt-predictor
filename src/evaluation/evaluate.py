"""Metric helpers: R2, RMSE, MAE, and repeat-level pooled aggregation."""

from typing import Sequence

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score


def r2(y_true, y_pred) -> float:
    return float(r2_score(y_true, y_pred))


def rmse(y_true, y_pred) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def repeat_level_pooled(per_fold_rows: Sequence[dict], n_repeats: int) -> dict:
    """Pool per-fold OOF predictions per repeat, compute mean/SD over repeats."""
    repeat_r2, repeat_rmse, repeat_mae = [], [], []
    for rep_idx in range(n_repeats):
        rep_rows = [r for r in per_fold_rows if r["repeat_idx"] == rep_idx]
        if not rep_rows:
            raise ValueError(f"repeat_level_pooled: no rows for repeat {rep_idx}")
        y_true = np.concatenate([np.asarray(r["y_true"]) for r in rep_rows])
        y_pred = np.concatenate([np.asarray(r["y_pred"]) for r in rep_rows])
        repeat_r2.append(r2(y_true, y_pred))
        repeat_rmse.append(rmse(y_true, y_pred))
        repeat_mae.append(mae(y_true, y_pred))
    return {
        "cv_r2_mean": float(np.mean(repeat_r2)),
        "cv_r2_sd": float(np.std(repeat_r2, ddof=1)),
        "cv_rmse_mean": float(np.mean(repeat_rmse)),
        "cv_rmse_sd": float(np.std(repeat_rmse, ddof=1)),
        "cv_mae_mean": float(np.mean(repeat_mae)),
        "cv_mae_sd": float(np.std(repeat_mae, ddof=1)),
        "_repeat_r2": repeat_r2,
        "_repeat_rmse": repeat_rmse,
        "_repeat_mae": repeat_mae,
    }
