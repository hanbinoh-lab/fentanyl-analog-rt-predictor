"""Stage 1-4b orchestrator: per-candidate runner + results aggregation."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.applicability_domain import run_ad  # noqa: E402
from evaluation.evaluate import repeat_level_pooled  # noqa: E402
from evaluation.loocv import run_loocv  # noqa: E402
from evaluation.nested_cv import n_repeats_for, run_nested_cv  # noqa: E402
from evaluation.y_scrambling import n_perm_for, run_y_scrambling  # noqa: E402
from training.train_final import run_stage2_final  # noqa: E402

DEFAULT_OUT_ROOT = ROOT / "models"
ALGORITHMS_YAML = ROOT / "configs" / "algorithms.yaml"
BASE_CSV = ROOT / "data" / "raw" / "base.csv"
FEATURES_DIR = ROOT / "data" / "features"
ALL_STAGES = ("1", "2", "3", "4a", "4b")
PER_CANDIDATE_STAGES = ("1", "2", "3", "4a", "4b")

STAGE_PRIMARY_OUTPUT = {
    "1": ("results_cv.csv",),
    "2": ("trained.joblib",),
    "3": ("results_yscrambling.csv",),
    "4a": ("results_loocv.csv",),
    "4b": ("results_ad.csv",),
}


def _load_algorithms():
    with ALGORITHMS_YAML.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def ssot_num_lookup(rep_id: str, algo_id: str) -> int:
    for num, entry in _load_algorithms().items():
        if entry["rep_id"] == rep_id and entry["algo_id"] == algo_id:
            return int(num)
    raise ValueError(f"ssot_num_lookup: ({rep_id}, {algo_id}) not in configs/algorithms.yaml")


def parse_candidates(spec: str) -> list[tuple]:
    algos = _load_algorithms()
    if spec == "all":
        return [(int(n), e["rep_id"], e["algo_id"]) for n, e in algos.items()]
    if spec.startswith("by-rep:"):
        rep_id = spec.split(":", 1)[1]
        return [
            (int(n), e["rep_id"], e["algo_id"])
            for n, e in algos.items() if e["rep_id"] == rep_id
        ]
    parsed = []
    for item in spec.split(","):
        rep_id, algo_id = item.strip().split(":")
        parsed.append((ssot_num_lookup(rep_id, algo_id), rep_id, algo_id))
    return parsed


def parse_stages(spec: str) -> set[str]:
    if spec == "all":
        return set(ALL_STAGES)
    return {s.strip() for s in spec.split(",")}


def load_features(rep_id: str):
    df = pd.read_csv(FEATURES_DIR / rep_id / f"{rep_id}.csv")
    feat_cols = [c for c in df.columns
                 if c not in ("compound", "smiles", "retention_time_min")]
    X = df[feat_cols].values
    y = df["retention_time_min"].values.astype(float)
    return X, y


def _stage_done(out_dir: Path, stage: str) -> bool:
    return any((out_dir / name).exists() for name in STAGE_PRIMARY_OUTPUT[stage])


def _load_stage1_artifacts(out_dir: Path, rep_id: str) -> dict:
    """Restore Stage 3 inputs from results_cv.csv when Stage 1 is not in this process."""
    cv_path = out_dir / "results_cv.csv"
    if not cv_path.exists():
        raise RuntimeError(
            f"Stage 3 requires results_cv.csv (missing at {cv_path}); run Stage 1 first."
        )
    df = pd.read_csv(cv_path)
    per_fold_best_hp = (
        df.groupby("fold_idx")["best_hp_json"]
        .first()
        .apply(json.loads)
        .tolist()
    )
    per_fold_rows = [
        {
            "fold_idx": int(fold_idx),
            "repeat_idx": int(g["repeat_idx"].iloc[0]),
            "y_true": g["y_true"].tolist(),
            "y_pred": g["y_pred"].tolist(),
        }
        for fold_idx, g in df.groupby("fold_idx")
    ]
    pooled = repeat_level_pooled(per_fold_rows, n_repeats_for(rep_id))
    return {
        "per_fold_best_hp": per_fold_best_hp,
        "main_partial": {"cv_r2_mean": pooled["cv_r2_mean"]},
    }


def _load_stage2_artifact(out_dir: Path) -> dict:
    """Restore Stage 4a inputs from config.json final_hp when Stage 2 is not in this process."""
    cfg_path = out_dir / "config.json"
    if not cfg_path.exists():
        raise RuntimeError(
            f"Stage 4a requires config.json final_hp (missing at {cfg_path}); run Stage 2 first."
        )
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    final_hp = cfg.get("final_hp")
    if not final_hp:
        raise RuntimeError(f"{cfg_path} missing final_hp; run Stage 2 first.")
    return {"final_hp": final_hp}


def write_config_json(
    out_dir: Path, rep_id: str, algo_id: str, ssot_num: int,
    s1=None, s2=None, s3=None, s4a=None,
) -> Path:
    """Merge new scalars into existing config.json (preserves prior stages)."""
    cfg_path = out_dir / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    cfg["rep_id"] = rep_id
    cfg["algo_id"] = algo_id
    cfg["ssot_num"] = ssot_num
    cfg["python"] = sys.version.split()[0]

    if s2 and "final_hp" in s2:
        cfg["final_hp"] = s2["final_hp"]

    main = dict(cfg.get("main_table", {}))
    if s1 and "main_partial" in s1:
        for k, v in s1["main_partial"].items():
            main[k] = v
    if s4a:
        main["q2_loo"] = s4a["q2_loo"]
    if s3:
        main["p_y_scram_reported"] = s3["p_y_scram_reported"]
    if main:
        cfg["main_table"] = main

    si = dict(cfg.get("si_table", {}))
    if s4a:
        si["loocv_rmse"] = s4a["loocv_rmse"]
        si["loocv_mae"] = s4a["loocv_mae"]
    if si:
        cfg["si_table"] = si

    cfg_path.write_text(json.dumps(cfg, indent=2, default=float), encoding="utf-8")
    return cfg_path


def run_per_candidate(
    rep_id: str, algo_id: str, stages: set[str],
    out_root: Path, seed: int, resume: bool = False,
) -> Path:
    ssot_num = ssot_num_lookup(rep_id, algo_id)
    out_dir = out_root / rep_id / f"{ssot_num:02d}_{algo_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    X, y = load_features(rep_id)

    s1 = s2 = s3 = s4a = None

    if "1" in stages and not (resume and _stage_done(out_dir, "1")):
        s1 = run_nested_cv(X, y, rep_id, algo_id, out_dir, seed)

    if "2" in stages and not (resume and _stage_done(out_dir, "2")):
        s2 = run_stage2_final(X, y, rep_id, algo_id, out_dir, seed)

    if "3" in stages and not (resume and _stage_done(out_dir, "3")):
        if s1 is None:
            s1 = _load_stage1_artifacts(out_dir, rep_id)
        s3 = run_y_scrambling(
            X, y, rep_id, algo_id,
            per_fold_best_hp=s1["per_fold_best_hp"],
            n_perm=n_perm_for(algo_id),
            observed_r2_mean=s1["main_partial"]["cv_r2_mean"],
            out_dir=out_dir, random_state=seed,
        )

    if "4a" in stages and not (resume and _stage_done(out_dir, "4a")):
        if s2 is None:
            s2 = _load_stage2_artifact(out_dir)
        s4a = run_loocv(X, y, rep_id, algo_id, s2["final_hp"], out_dir, seed)

    if "4b" in stages and not (resume and _stage_done(out_dir, "4b")):
        run_ad(X, out_dir, k=5, percentile=95.0)

    write_config_json(out_dir, rep_id, algo_id, ssot_num, s1, s2, s3, s4a)
    return out_dir


def _candidate_id(cfg: dict) -> dict:
    rep_id = cfg.get("rep_id")
    algo_id = cfg.get("algo_id")
    return {
        "ssot_num": cfg.get("ssot_num"),
        "rep_id": rep_id,
        "algo_id": algo_id,
        "candidate": f"{rep_id}__{algo_id}",
    }


def _iter_candidates(out_root: Path):
    for rep_dir in sorted(out_root.iterdir()):
        if not rep_dir.is_dir() or rep_dir.name == "aggregated":
            continue
        for cand_dir in sorted(rep_dir.iterdir()):
            cfg_path = cand_dir / "config.json"
            if cfg_path.exists():
                yield cand_dir, json.loads(cfg_path.read_text(encoding="utf-8"))


def aggregate_master_results(out_root: Path) -> Path:
    """Build master_results.csv (top) + archive/log.csv + archive/hp.csv from per-candidate raw + config."""
    agg = out_root / "aggregated"
    arch = agg / "archive"
    arch.mkdir(parents=True, exist_ok=True)

    master, log, hp = [], [], []
    for cand_dir, cfg in _iter_candidates(out_root):
        ids = _candidate_id(cfg)
        mt, st = cfg.get("main_table", {}), cfg.get("si_table", {})

        cv_df = pd.read_csv(cand_dir / "results_cv.csv")
        train_r2_mean = float(cv_df["train_r2_fold"].mean())
        rmse_repeat = cv_df.groupby("repeat_idx").apply(
            lambda g: float(np.sqrt(((g["y_true"] - g["y_pred"]) ** 2).mean())),
            include_groups=False,
        )

        ys_df = pd.read_csv(cand_dir / "results_yscrambling.csv")
        n_perm = len(ys_df)
        obs_r2 = mt.get("cv_r2_mean")
        k = int((ys_df["null_r2"] >= obs_r2).sum()) if obs_r2 is not None else None
        p_exact = (1 + k) / (1 + n_perm) if k is not None else None

        ad_df = pd.read_csv(cand_dir / "results_ad.csv")

        master.append({**ids,
            "cv_rmse_mean": mt.get("cv_rmse_mean"), "cv_rmse_sd": mt.get("cv_rmse_sd"),
            "cv_r2_mean": mt.get("cv_r2_mean"), "cv_r2_sd": mt.get("cv_r2_sd"),
            "cv_mae_mean": mt.get("cv_mae_mean"), "cv_mae_sd": mt.get("cv_mae_sd"),
            "delta_train_cv_r2": mt.get("delta_train_cv_r2"),
            "q2_loo": mt.get("q2_loo"),
            "p_y_scram_reported": mt.get("p_y_scram_reported"),
            "loocv_rmse": st.get("loocv_rmse"), "loocv_mae": st.get("loocv_mae")})

        log.append({**ids,
            "p_y_scram_exact": p_exact,
            "null_r2_mean": float(ys_df["null_r2"].mean()),
            "null_r2_sd": float(ys_df["null_r2"].std(ddof=1)) if n_perm > 1 else None,
            "null_r2_max": float(ys_df["null_r2"].max()),
            "ad_coverage_rate": float(ad_df["in_chemical_domain"].mean()),
            "ad_threshold": float(np.percentile(ad_df["knn_distance_mean"], 95)),
            "train_r2_mean": train_r2_mean,
            "cv_rmse_min_repeat": float(rmse_repeat.min()),
            "cv_rmse_max_repeat": float(rmse_repeat.max())})

        hp.append({**ids, "final_hp_json": json.dumps(cfg.get("final_hp", {}), sort_keys=True)})

    master_path = agg / "master_results.csv"
    log_path = arch / "log.csv"
    hp_path = arch / "hp.csv"
    pd.DataFrame(master).sort_values("ssot_num").to_csv(master_path, index=False)
    pd.DataFrame(log).sort_values("ssot_num").to_csv(log_path, index=False)
    pd.DataFrame(hp).sort_values("ssot_num").to_csv(hp_path, index=False)
    return master_path


def main():
    parser = argparse.ArgumentParser(description="Stage 1-4b evaluation orchestrator.")
    parser.add_argument("--candidates", default="all",
                        help='"all" | "by-rep:<rep_id>" | "rep:algo,rep:algo,..."')
    parser.add_argument("--stages", default="all",
                        help="comma-separated subset of {1,2,3,4a,4b}")
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true",
                        help="skip per-stage outputs already present")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    requested = parse_stages(args.stages)

    per_cand = requested & set(PER_CANDIDATE_STAGES)
    if per_cand:
        for ssot_num, rep_id, algo_id in parse_candidates(args.candidates):
            print(f"run: {rep_id}:{algo_id}")
            run_per_candidate(
                rep_id, algo_id, per_cand, out_root, args.seed, resume=args.resume
            )

    if per_cand == set(PER_CANDIDATE_STAGES):
        aggregate_master_results(out_root)


if __name__ == "__main__":
    main()
