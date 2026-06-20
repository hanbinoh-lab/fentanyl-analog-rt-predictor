"""Stage 2: final all-data model fit + save deployment artifact."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from training.final_hyperparameter_search import search_final_hp  # noqa: E402


def _save(model, out_dir: Path) -> Path:
    import joblib
    path = out_dir / "trained.joblib"
    joblib.dump(model, path)
    return path


def run_stage2_final(
    X,
    y,
    rep_id: str,
    algo_id: str,
    out_dir,
    random_state: int = 42,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    final_hp, final_model = search_final_hp(X, y, rep_id, algo_id, random_state)
    trained_path = _save(final_model, out_dir)

    return {
        "final_hp": final_hp,
        "trained_model_path": trained_path,
        "stage2_summary": {
            "n": int(len(y)),
            "hp": final_hp,
            "python": sys.version.split()[0],
        },
    }
