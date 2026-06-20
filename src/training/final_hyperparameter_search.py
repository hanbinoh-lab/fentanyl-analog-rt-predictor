"""Stage 2: independent inner-CV HP search on full n=94 (deployment context)."""

import sys
from pathlib import Path

from sklearn.model_selection import GridSearchCV, StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.algorithm_registry import get_hp_grid, make_estimator  # noqa: E402
from evaluation.nested_cv import make_strata, n_inner_for  # noqa: E402


def search_final_hp(X, y, rep_id: str, algo_id: str, seed: int):
    """Returns (final_hp, refit_estimator) from an inner-CV grid search on all data."""
    grid = get_hp_grid(algo_id, rep_id)
    strata = make_strata(y)
    skf = StratifiedKFold(n_inner_for(rep_id), shuffle=True, random_state=seed)
    inner_splits = list(skf.split(X, strata))
    est = make_estimator(algo_id, rep_id, seed)
    gs = GridSearchCV(
        est, grid, cv=inner_splits,
        scoring="neg_root_mean_squared_error",
        n_jobs=1, refit=True,
    )
    gs.fit(X, y)
    return dict(gs.best_params_), gs.best_estimator_
