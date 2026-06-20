"""Stage 1: repeated nested cross-validation per candidate."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.algorithm_registry import get_hp_grid, make_estimator  # noqa: E402
from evaluation.evaluate import mae, r2, repeat_level_pooled, rmse  # noqa: E402

N_OUTER_FOLDS = 5
N_REPEATS_DEFAULT = 10
N_INNER_DEFAULT = 5
N_STRATA_BINS = 5
RANDOM_STATE = 42


def make_strata(y, n_bins: int = N_STRATA_BINS):
    return pd.qcut(np.asarray(y), n_bins, labels=False, duplicates="drop")


def n_repeats_for(rep_id: str) -> int:
    return N_REPEATS_DEFAULT


def n_inner_for(rep_id: str) -> int:
    return N_INNER_DEFAULT


def _fit_outer_fold(X_tr, y_tr, rep_id, algo_id, seed, fold_idx):
    """Inner HP search on the outer-train fold, refit best. Returns (fitted_est, best_hp)."""
    strata_inner = make_strata(y_tr)
    skf = StratifiedKFold(
        n_inner_for(rep_id), shuffle=True, random_state=seed + fold_idx
    )
    inner_splits = list(skf.split(X_tr, strata_inner))
    grid = get_hp_grid(algo_id, rep_id)
    est = make_estimator(algo_id, rep_id, seed)
    gs = GridSearchCV(
        est, grid, cv=inner_splits,
        scoring="neg_root_mean_squared_error",
        n_jobs=1, refit=True,
    )
    gs.fit(X_tr, y_tr)
    return gs.best_estimator_, dict(gs.best_params_)


def run_nested_cv(
    X,
    y,
    rep_id: str,
    algo_id: str,
    out_dir,
    random_state: int = RANDOM_STATE,
) -> dict:
    """Execute repeated nested CV for one (rep_id, algo_id) pair.

    Writes results_cv.csv (long format). Returns per-fold best HP and main-table partial scalars.
    """
    X = np.asarray(X)
    y = np.asarray(y, dtype=float)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_repeats = n_repeats_for(rep_id)
    outer_cv = RepeatedStratifiedKFold(
        n_splits=N_OUTER_FOLDS, n_repeats=n_repeats, random_state=random_state
    )
    strata_full = make_strata(y)

    per_fold_rows: list[dict] = []
    per_fold_best_hp: list[dict] = []
    long_rows: list[dict] = []

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, strata_full)):
        repeat_idx = fold_idx // N_OUTER_FOLDS
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx], y[test_idx]

        est, best_hp = _fit_outer_fold(X_tr, y_tr, rep_id, algo_id, random_state, fold_idx)
        y_tr_pred = np.asarray(est.predict(X_tr)).ravel()
        y_te_pred = np.asarray(est.predict(X_te)).ravel()

        train_r2 = r2(y_tr, y_tr_pred)
        train_rmse_f = rmse(y_tr, y_tr_pred)
        train_mae_f = mae(y_tr, y_tr_pred)
        cv_r2_f = r2(y_te, y_te_pred)
        cv_rmse_f = rmse(y_te, y_te_pred)
        cv_mae_f = mae(y_te, y_te_pred)

        per_fold_rows.append({
            "fold_idx": fold_idx,
            "repeat_idx": repeat_idx,
            "y_true": y_te.tolist(),
            "y_pred": y_te_pred.tolist(),
            "train_r2": train_r2,
        })
        per_fold_best_hp.append(best_hp)

        best_hp_json = json.dumps(best_hp, default=float, sort_keys=True)
        for compound_idx, yt, yp in zip(test_idx, y_te, y_te_pred):
            long_rows.append({
                "fold_idx": int(fold_idx),
                "repeat_idx": int(repeat_idx),
                "compound_idx": int(compound_idx),
                "y_true": float(yt),
                "y_pred": float(yp),
                "train_r2_fold": train_r2,
                "train_rmse_fold": train_rmse_f,
                "train_mae_fold": train_mae_f,
                "cv_r2_fold": cv_r2_f,
                "cv_rmse_fold": cv_rmse_f,
                "cv_mae_fold": cv_mae_f,
                "best_hp_json": best_hp_json,
            })

    results_cv_path = out_dir / "results_cv.csv"
    pd.DataFrame(long_rows).to_csv(results_cv_path, index=False)

    pooled = repeat_level_pooled(per_fold_rows, n_repeats)
    mean_train_r2 = float(np.mean([r["train_r2"] for r in per_fold_rows]))

    main_partial = {
        "cv_rmse_mean": pooled["cv_rmse_mean"], "cv_rmse_sd": pooled["cv_rmse_sd"],
        "cv_r2_mean": pooled["cv_r2_mean"], "cv_r2_sd": pooled["cv_r2_sd"],
        "cv_mae_mean": pooled["cv_mae_mean"], "cv_mae_sd": pooled["cv_mae_sd"],
        "delta_train_cv_r2": mean_train_r2 - pooled["cv_r2_mean"],
    }

    return {
        "results_cv_path": results_cv_path,
        "per_fold_best_hp": per_fold_best_hp,
        "main_partial": main_partial,
    }
