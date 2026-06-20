"""Stage 3: fixed-HP y-scrambling per candidate."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.algorithm_registry import make_estimator  # noqa: E402
from evaluation.evaluate import repeat_level_pooled  # noqa: E402
from evaluation.nested_cv import (  # noqa: E402
    N_OUTER_FOLDS,
    RANDOM_STATE,
    make_strata,
    n_repeats_for,
)


def n_perm_for(algo_id: str) -> int:
    return 100


def run_y_scrambling(
    X,
    y,
    rep_id: str,
    algo_id: str,
    per_fold_best_hp: list[dict],
    n_perm: int,
    observed_r2_mean: float,
    out_dir,
    random_state: int = RANDOM_STATE,
) -> dict:
    X = np.asarray(X)
    y = np.asarray(y, dtype=float)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_repeats = n_repeats_for(rep_id)
    outer_cv = RepeatedStratifiedKFold(
        n_splits=N_OUTER_FOLDS, n_repeats=n_repeats, random_state=random_state
    )
    strata_orig = make_strata(y)
    outer_splits = list(outer_cv.split(X, strata_orig))

    if len(per_fold_best_hp) != len(outer_splits):
        raise ValueError(
            f"per_fold_best_hp length {len(per_fold_best_hp)} "
            f"!= outer splits {len(outer_splits)}"
        )

    null_r2 = np.empty(n_perm, dtype=float)
    for perm_idx in range(n_perm):
        rng = np.random.default_rng(random_state + perm_idx)
        y_perm = rng.permutation(y)
        per_fold_rows = []
        for fold_idx, (train_idx, test_idx) in enumerate(outer_splits):
            est = make_estimator(algo_id, rep_id, random_state)
            est.set_params(**per_fold_best_hp[fold_idx])
            est.fit(X[train_idx], y_perm[train_idx])
            y_pred = np.asarray(est.predict(X[test_idx])).ravel()
            per_fold_rows.append({
                "fold_idx": fold_idx,
                "repeat_idx": fold_idx // N_OUTER_FOLDS,
                "y_true": y_perm[test_idx].tolist(),
                "y_pred": y_pred.tolist(),
            })
        pooled = repeat_level_pooled(per_fold_rows, n_repeats)
        null_r2[perm_idx] = pooled["cv_r2_mean"]

    results_path = out_dir / "results_yscrambling.csv"
    pd.DataFrame({
        "perm_idx": np.arange(n_perm),
        "seed_perm": [random_state + i for i in range(n_perm)],
        "null_r2": null_r2,
    }).to_csv(results_path, index=False)

    resolution_floor = 1.0 / (1 + n_perm)
    k = int(np.sum(null_r2 >= observed_r2_mean))
    p_exact = (1 + k) / (1 + n_perm)
    p_reported = f"< {resolution_floor:.4f}" if k == 0 else f"{p_exact:.4f}"

    return {
        "results_yscrambling_path": results_path,
        "null_r2_array": null_r2,
        "null_r2_mean": float(null_r2.mean()),
        "null_r2_max": float(null_r2.max()),
        "observed_r2": float(observed_r2_mean),
        "p_y_scram_exact": float(p_exact),
        "p_y_scram_reported": p_reported,
        "resolution_floor": float(resolution_floor),
    }
