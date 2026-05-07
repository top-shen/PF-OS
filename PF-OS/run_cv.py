# -*- coding: utf-8 -*-
"""
run_cv.py

Subject-wise K-fold cross-validation runner for PhysioFormer-v3.

Outputs
-------
outdir/
  folds/fold_00/ ...
  cv_summary.json
  per_subject_all.csv
  per_fold.csv

Also writes paper assets (LaTeX tables) into:
  report/tables/main_results.tex   (by default)

Author: assistant (v3)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import train
from data import validate_dataset
from stats_utils import bootstrap_ci, format_mean_ci
from paper_utils import write_latex_table, write_placeholder_tables


def make_folds(subjects: List[str], k: int, seed: int) -> List[List[str]]:
    rng = np.random.RandomState(seed)
    subs = list(subjects)
    rng.shuffle(subs)
    folds = [list(x) for x in np.array_split(subs, k)]
    # ensure no empty folds (can happen if k>n)
    folds = [f for f in folds if len(f) > 0]
    return folds


def apply_overrides(args: argparse.Namespace, overrides: Dict[str, object]) -> argparse.Namespace:
    out = argparse.Namespace(**vars(args))
    for k, v in overrides.items():
        setattr(out, k, v)
    return out


def average_prediction_frames(pred_frames: List[pd.DataFrame]) -> pd.DataFrame:
    if not pred_frames:
        raise ValueError("Need at least one prediction frame for consensus averaging.")

    key_cols = ["participant", "t_start_us", "t_end_us"]
    proba_cols = ["p_low", "p_med", "p_high"]

    base = pred_frames[0].sort_values(key_cols).reset_index(drop=True).copy()
    prob_stack = [base[proba_cols].to_numpy(dtype=np.float32)]

    for df in pred_frames[1:]:
        cur = df.sort_values(key_cols).reset_index(drop=True)
        if not base[key_cols].equals(cur[key_cols]):
            raise RuntimeError("Consensus components produced mismatched test windows.")
        if not np.array_equal(base["y_true"].to_numpy(dtype=np.int64), cur["y_true"].to_numpy(dtype=np.int64)):
            raise RuntimeError("Consensus components produced mismatched y_true labels.")
        prob_stack.append(cur[proba_cols].to_numpy(dtype=np.float32))

    avg_proba = np.mean(np.stack(prob_stack, axis=0), axis=0)
    out = base.copy()
    out[proba_cols] = avg_proba
    out["y_pred"] = np.argmax(avg_proba, axis=1).astype(np.int64)
    return out


def run_one_fold(
    root: Path,
    fold_outdir: Path,
    args: argparse.Namespace,
    seed: int,
    train_subjects: List[str],
    val_subjects: List[str],
    test_subjects: List[str],
) -> Dict[str, float]:
    train_dirs = train.resolve_dirs(root, train_subjects)
    val_dirs = train.resolve_dirs(root, val_subjects)
    test_dirs = train.resolve_dirs(root, test_subjects)

    # ensure fold has its own predictions
    args_fold = argparse.Namespace(**vars(args))
    # data check was already performed at CV start; avoid re-checking each fold
    args_fold.skip_data_check = True
    args_fold.data_check_log = ""
    args_fold.save_test_predictions = "predictions_test.csv"
    if str(getattr(args, "cv_profile", "classification")).lower() == "classification":
        args_fold = train.apply_model_profile(args_fold, "ocular_static")

    summary = train.train_one_split(
        root=root,
        outdir=fold_outdir,
        seed=seed,
        train_dirs=train_dirs,
        val_dirs=val_dirs,
        test_dirs=test_dirs,
        args=args_fold,
    )
    return summary["test_metrics"]


def run_one_fold_consensus(
    root: Path,
    fold_outdir: Path,
    args: argparse.Namespace,
    seed: int,
    train_subjects: List[str],
    val_subjects: List[str],
    test_subjects: List[str],
) -> Dict[str, float]:
    train_dirs = train.resolve_dirs(root, train_subjects)
    val_dirs = train.resolve_dirs(root, val_subjects)
    test_dirs = train.resolve_dirs(root, test_subjects)

    component_specs = [
        ("eye_plus_static", {}),
        ("eye_only", {"no_static": True, "use_static_gate": False, "split_static_modalities": False, "use_residual_static_fusion": False}),
    ]

    pred_frames: List[pd.DataFrame] = []
    component_meta: List[Dict[str, object]] = []

    for name, overrides in component_specs:
        comp_out = fold_outdir / name
        comp_out.mkdir(parents=True, exist_ok=True)
        args_comp = argparse.Namespace(**vars(args))
        args_comp.skip_data_check = True
        args_comp.data_check_log = ""
        args_comp.save_test_predictions = "predictions_test.csv"
        args_comp = train.apply_model_profile(args_comp, "ocular_static")
        args_comp = apply_overrides(args_comp, overrides)

        summary_path = comp_out / "summary.json"
        pred_path = comp_out / "predictions_test.csv"
        if summary_path.exists() and pred_path.exists():
            comp_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            comp_summary = train.train_one_split(
                root=root,
                outdir=comp_out,
                seed=seed,
                train_dirs=train_dirs,
                val_dirs=val_dirs,
                test_dirs=test_dirs,
                args=args_comp,
            )

        pred_frames.append(pd.read_csv(pred_path))
        component_meta.append(
            {
                "name": name,
                "outdir": str(comp_out),
                "test_metrics": comp_summary.get("test_metrics", {}),
            }
        )

    consensus_pred = average_prediction_frames(pred_frames)
    consensus_path = fold_outdir / "predictions_test.csv"
    consensus_pred.to_csv(consensus_path, index=False, encoding="utf-8")

    per_subject = train.compute_per_subject_metrics(consensus_pred)
    per_subject_path = fold_outdir / "per_subject_metrics.csv"
    per_subject.to_csv(per_subject_path, index=False, encoding="utf-8")

    y_true = consensus_pred["y_true"].to_numpy(dtype=np.int64)
    y_pred = consensus_pred["y_pred"].to_numpy(dtype=np.int64)
    y_proba = consensus_pred[["p_low", "p_med", "p_high"]].to_numpy(dtype=np.float32)
    test_metrics = train.compute_classification_metrics(y_true, y_pred, y_proba)
    test_metrics.update({"tlx_mae": float("nan"), "tlx_rmse": float("nan"), "tlx_pearson": float("nan")})

    (fold_outdir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (fold_outdir / "summary.json").write_text(
        json.dumps(
            {
                "profile": "consensus",
                "components": component_meta,
                "test_metrics": test_metrics,
                "predictions_csv": str(consensus_path),
                "per_subject_csv": str(per_subject_path),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return test_metrics


def aggregate_subject_metrics(per_subject_all: pd.DataFrame, metrics: List[str], n_boot: int, seed: int) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for m in metrics:
        vals = per_subject_all[m].to_numpy(dtype=np.float64)
        ci = bootstrap_ci(vals, n_boot=n_boot, seed=seed)
        out[m] = {"mean": ci.mean, "lo": ci.lo, "hi": ci.hi, "n": ci.n}
    return out


def main() -> None:
    ap = train.build_argparser()

    # CV-specific args
    ap.add_argument("--k_folds", "--k", type=int, default=5, help="Subject-wise K for K-fold CV (alias: --k)")
    ap.add_argument("--cv_outdir", type=str, default="runs/cv_physioformer_v3", help="Where to write CV folds and summary")
    ap.add_argument("--val_ratio_in_train", type=float, default=0.15, help="Fraction of non-test subjects used for validation in each fold")
    ap.add_argument("--n_boot", type=int, default=10000, help="Bootstrap replicates for subject-level CI")
    ap.add_argument("--report_dir", type=str, default="report", help="Project report directory (contains main.tex)")
    ap.add_argument("--table_digits", type=int, default=3)
    ap.add_argument(
        "--cv_profile",
        type=str,
        default="classification",
        choices=["classification", "consensus", "full"],
        help="classification=single PhysioFormer-OS expert; consensus=average OS + eye-only experts; full=use raw train.py settings.",
    )
    args = ap.parse_args()

    # Backward-compatible: many users pass --outdir expecting CV output (train.build_argparser defines --outdir).
    # If --cv_outdir is untouched (default) but --outdir is custom, redirect CV outputs to --outdir.
    if getattr(args, "cv_outdir", "runs/cv_physioformer_v3") == "runs/cv_physioformer_v3":
        if getattr(args, "outdir", "runs/physioformer_v3") != "runs/physioformer_v3":
            args.cv_outdir = args.outdir

    root = Path(args.root)
    outdir = Path(args.cv_outdir)
    folds_dir = outdir / "folds"
    folds_dir.mkdir(parents=True, exist_ok=True)

    # Ensure placeholder tables exist for compile-on-clone behavior
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    write_placeholder_tables(report_dir)

    subject_dirs = train.list_participants(root)
    subjects = [p.name for p in subject_dirs]

    # -----------------------------
    # Pre-flight dataset integrity check (once, before folding)
    # -----------------------------
    if not getattr(args, "skip_data_check", False):
        log_path = Path(getattr(args, "data_check_log", "")).expanduser() if getattr(args, "data_check_log", "") else (outdir / "data_check.log")
        summary = validate_dataset(
            root=root,
            participant_dirs=subject_dirs,
            log_path=log_path,
            window_sec=float(args.window_sec),
            stride_sec=float(args.stride_sec),
        )
        if not getattr(args, "no_drop_bad_subjects", False):
            ok = set(summary.get("ok_subjects", []))
            dropped = [p.name for p in subject_dirs if p.name not in ok]
            if dropped:
                print(f"[data-check] Dropped {len(dropped)} subjects with critical issues before CV. See: {log_path.resolve()}")
            subject_dirs = [p for p in subject_dirs if p.name in ok]
            subjects = [p.name for p in subject_dirs]
        else:
            print(f"[data-check] Completed. See: {log_path.resolve()} (keeping bad subjects per --no_drop_bad_subjects)")
    if len(subjects) < 2:
        raise RuntimeError("Need at least 2 subjects for subject-wise CV. (For n<3, consider train.py window-level fallback).")

    print(f"[cv] profile={args.cv_profile}")
    if str(args.cv_profile).lower() == "classification":
        print("[cv] profile=PhysioFormer-OS: no_ppg=True, no_bilinear=True, no_regression_head=True, reg_weight=0, contrastive_weight=0, select_metric=f1, use_static_gate=True")
    elif str(args.cv_profile).lower() == "consensus":
        print("[cv] profile=PhysioFormer-OS consensus: fixed average of the ocular-static expert and the eye-only expert under identical folds")

    folds = make_folds(subjects, k=int(args.k_folds), seed=int(args.seed))

    fold_rows = []
    per_subject_rows = []

    for i, test_subjects in enumerate(folds):
        remaining = [s for s in subjects if s not in set(test_subjects)]
        rng = np.random.RandomState(int(args.seed) + 1000 + i)
        rng.shuffle(remaining)

        n_val = max(1, int(round(len(remaining) * float(args.val_ratio_in_train))))
        n_val = min(n_val, max(1, len(remaining) - 1))
        val_subjects = remaining[:n_val]
        train_subjects = remaining[n_val:]

        fold_out = folds_dir / f"fold_{i:02d}"
        fold_out.mkdir(parents=True, exist_ok=True)

        print(f"[fold {i:02d}] train={len(train_subjects)} val={len(val_subjects)} test={len(test_subjects)} -> {fold_out}")

        if str(args.cv_profile).lower() == "consensus":
            test_metrics = run_one_fold_consensus(
                root=root,
                fold_outdir=fold_out,
                args=args,
                seed=int(args.seed) + i,
                train_subjects=train_subjects,
                val_subjects=val_subjects,
                test_subjects=test_subjects,
            )
        else:
            test_metrics = run_one_fold(
                root=root,
                fold_outdir=fold_out,
                args=args,
                seed=int(args.seed) + i,
                train_subjects=train_subjects,
                val_subjects=val_subjects,
                test_subjects=test_subjects,
            )
        fold_rows.append({"fold": i, **test_metrics})

        # gather per-subject metrics from fold output
        subj_csv = fold_out / "per_subject_metrics.csv"
        if subj_csv.exists():
            df_sub = pd.read_csv(subj_csv)
            df_sub["fold"] = i
            per_subject_rows.append(df_sub)

    per_fold = pd.DataFrame(fold_rows)
    per_fold.to_csv(outdir / "per_fold.csv", index=False, encoding="utf-8")

    if not per_subject_rows:
        raise RuntimeError("No per-subject metrics found. Check that train.py exported predictions correctly.")
    per_subject_all = pd.concat(per_subject_rows, axis=0, ignore_index=True)

    # sanity: each participant should appear once (if folds cover all subjects)
    per_subject_all.to_csv(outdir / "per_subject_all.csv", index=False, encoding="utf-8")

    metrics_to_report = ["f1_macro", "acc", "recall_macro", "auc_ovr_macro"]
    # optional if present
    if "tlx_pearson" in per_subject_all.columns:
        metrics_to_report.append("tlx_pearson")

    agg = aggregate_subject_metrics(per_subject_all, metrics_to_report, n_boot=int(args.n_boot), seed=int(args.seed))

    summary = {
        "n_subjects": int(per_subject_all["participant"].nunique()),
        "n_folds": int(len(folds)),
        "cv_profile": str(args.cv_profile),
        "metrics_subject_mean_ci": agg,
    }
    (outdir / "cv_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # Write LaTeX table for main results
    digits = int(args.table_digits)
    rows = []
    # single-row for PhysioFormer-v3
    row = [
        r"\textbf{PhysioFormer-OS}",
        format_mean_ci(bootstrap_ci(per_subject_all["acc"], n_boot=int(args.n_boot), seed=int(args.seed)), digits=digits),
        format_mean_ci(bootstrap_ci(per_subject_all["recall_macro"], n_boot=int(args.n_boot), seed=int(args.seed)+1), digits=digits),
        format_mean_ci(bootstrap_ci(per_subject_all["f1_macro"], n_boot=int(args.n_boot), seed=int(args.seed)+2), digits=digits),
        format_mean_ci(bootstrap_ci(per_subject_all["auc_ovr_macro"], n_boot=int(args.n_boot), seed=int(args.seed)+3), digits=digits),
    ]
    if "tlx_pearson" in per_subject_all.columns:
        row.append(format_mean_ci(bootstrap_ci(per_subject_all["tlx_pearson"], n_boot=int(args.n_boot), seed=int(args.seed)+4), digits=digits))
        col_names = ["Model", "Acc", "Recall", "Macro-F1", "AUC (OvR)", "TLX $r$"]
    else:
        col_names = ["Model", "Acc", "Recall", "Macro-F1", "AUC (OvR)"]
    rows.append(row)

    table_path = report_dir / "tables" / "main_results.tex"
    write_latex_table(
        table_path,
        caption="Subject-wise cross-validated performance (mean with 95\\% bootstrap CI; resampling over subjects).",
        label="tab:main_results",
        col_names=col_names,
        rows=rows,
        notes="CIs are percentile bootstrap over independent subjects (not over windows).",
    )

    print(f"[done] CV summary -> {outdir.resolve()}")
    print(f"[done] LaTeX table -> {table_path.resolve()}")


if __name__ == "__main__":
    main()
