"""Algorithm registry: estimator factory and per-algorithm hyperparameter grid.

Public API:
    make_estimator(algo_id, rep_id, random_state=42) -> Pipeline
    get_hp_grid(algo_id, rep_id) -> dict
    get_applicable_algos(rep_id) -> list[str]

Descriptor inputs are standardized inside a Pipeline (leakage-safe under CV), so
hyperparameter keys carry the 'est__' prefix. Inner-CV scoring
('neg_root_mean_squared_error') is applied by the evaluation orchestrator.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
except ImportError as e:
    raise ImportError(
        "algorithm_registry: xgboost import failed; check fa-rt env activation"
    ) from e

ALGO_IDS = ("ridge", "rf", "xgb", "gp_rbf", "mlp")
REP_IDS = ("mordred",)


def _check_algo(algo_id: str) -> None:
    if algo_id not in ALGO_IDS:
        raise ValueError(
            f"algorithm_registry: unknown algo_id {algo_id!r}, must be one of {ALGO_IDS}"
        )


def _check_rep(rep_id: str) -> None:
    if rep_id not in REP_IDS:
        raise ValueError(
            f"algorithm_registry: unknown rep_id {rep_id!r}, must be one of {REP_IDS}"
        )


def get_applicable_algos(rep_id: str) -> list:
    _check_rep(rep_id)
    return list(ALGO_IDS)


def _wrap(estimator):
    """Standardize features, then fit the estimator (fit inside each CV split)."""
    return Pipeline([("scaler", StandardScaler()), ("est", estimator)])


def make_estimator(algo_id: str, rep_id: str, random_state: int = 42):
    _check_algo(algo_id)
    _check_rep(rep_id)

    if algo_id == "ridge":
        return _wrap(Ridge())

    if algo_id == "rf":
        rf = RandomForestRegressor(random_state=random_state, n_jobs=1)
        return _wrap(rf)

    if algo_id == "xgb":
        xgb = XGBRegressor(
            random_state=random_state,
            n_jobs=1,
            objective="reg:squarederror",
            verbosity=0,
        )
        return _wrap(xgb)

    if algo_id == "gp_rbf":
        gp = GaussianProcessRegressor(
            kernel=ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)),
            n_restarts_optimizer=5,
            normalize_y=True,
            random_state=random_state,
        )
        return _wrap(gp)

    if algo_id == "mlp":
        mlp = MLPRegressor(
            solver="lbfgs",
            max_iter=500,
            random_state=random_state,
        )
        return _wrap(mlp)

    raise ValueError(f"algorithm_registry: unreachable algo_id {algo_id!r}")


def get_hp_grid(algo_id: str, rep_id: str) -> dict:
    _check_algo(algo_id)
    _check_rep(rep_id)

    if algo_id == "ridge":
        return {"est__alpha": np.logspace(-3, 3, 50).tolist()}

    if algo_id == "rf":
        return {
            "est__n_estimators": [200, 500],
            "est__max_features": ["sqrt", "log2", 0.3],
            "est__min_samples_leaf": [2, 3, 5],
            "est__max_depth": [3, 5, 8],
        }

    if algo_id == "xgb":
        return {
            "est__max_depth": [2, 3, 5],
            "est__n_estimators": [50, 100, 200],
            "est__colsample_bytree": [0.2, 0.3, 0.5],
            "est__learning_rate": [0.05, 0.1],
        }

    if algo_id == "gp_rbf":
        return {"est__alpha": np.logspace(-3, 1, 20).tolist()}

    if algo_id == "mlp":
        return {
            "est__alpha": np.logspace(-3, 1, 5).tolist(),
            "est__hidden_layer_sizes": [(8,), (16,)],
        }

    raise ValueError(f"algorithm_registry: unreachable algo_id {algo_id!r}")
