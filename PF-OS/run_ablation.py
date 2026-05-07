# -*- coding: utf-8 -*-
"""
run_ablation.py

Ablation runner for the restored final paper profile centered on PhysioFormer-OS.

The paper-facing profile is `ocular_static`, which evaluates:

- PhysioFormer-OS
- PhysioFormer-S
- w/o cross-attn
- Early fusion
- Eye only

All variants reuse identical subject folds (paired design), so subject-level
significance testing remains valid.
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
from stats_utils import bootstrap_ci, format_mean_ci, paired_significance_tests, holm_bonferroni
from paper_utils import write_latex_table, write_placeholder_tables, latex_escape


def make_folds(subjects: List[str], k: int, seed: int) -> List[List[str]]:
    rng = np.random.RandomState(seed)
    subs = list(subjects)
    rng.shuffle(subs)
    folds = [list(x) for x in np.array_split(subs, k)]
    folds = [f for f in folds if len(f) > 0]
    return folds


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
            raise RuntimeError("Consensus ablation components produced mismatched test windows.")
        if not np.array_equal(base["y_true"].to_numpy(dtype=np.int64), cur["y_true"].to_numpy(dtype=np.int64)):
            raise RuntimeError("Consensus ablation components produced mismatched labels.")
        prob_stack.append(cur[proba_cols].to_numpy(dtype=np.float32))

    avg_proba = np.mean(np.stack(prob_stack, axis=0), axis=0)
    out = base.copy()
    out[proba_cols] = avg_proba
    out["y_pred"] = np.argmax(avg_proba, axis=1).astype(np.int64)
    return out


def run_consensus_fold(
    root: Path,
    fold_out: Path,
    args: argparse.Namespace,
    seed: int,
    train_dirs: List[Path],
    val_dirs: List[Path],
    test_dirs: List[Path],
) -> Dict[str, float]:
    component_specs = [
        ("eye_plus_static", {}),
        ("eye_only", {"no_static": True, "use_static_gate": False, "split_static_modalities": False, "use_residual_static_fusion": False}),
    ]
    pred_frames: List[pd.DataFrame] = []
    component_meta: List[Dict[str, object]] = []

    for name, overrides in component_specs:
        comp_out = fold_out / name
        comp_out.mkdir(parents=True, exist_ok=True)
        args_comp = argparse.Namespace(**vars(args))
        args_comp.save_test_predictions = "predictions_test.csv"
        args_comp.skip_data_check = True
        args_comp.data_check_log = ""
        args_comp = train.apply_model_profile(args_comp, "ocular_static")
        for k, v in overrides.items():
            setattr(args_comp, k, v)

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

    pred_df = average_prediction_frames(pred_frames)
    pred_path = fold_out / "predictions_test.csv"
    pred_df.to_csv(pred_path, index=False, encoding="utf-8")

    per_subject = train.compute_per_subject_metrics(pred_df)
    per_subject_path = fold_out / "per_subject_metrics.csv"
    per_subject.to_csv(per_subject_path, index=False, encoding="utf-8")

    test_metrics = train.compute_classification_metrics(
        pred_df["y_true"].to_numpy(dtype=np.int64),
        pred_df["y_pred"].to_numpy(dtype=np.int64),
        pred_df[["p_low", "p_med", "p_high"]].to_numpy(dtype=np.float32),
    )
    test_metrics.update({"tlx_mae": float("nan"), "tlx_rmse": float("nan"), "tlx_pearson": float("nan")})

    (fold_out / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (fold_out / "summary.json").write_text(
        json.dumps(
            {
                "profile": "consensus",
                "components": component_meta,
                "test_metrics": test_metrics,
                "predictions_csv": str(pred_path),
                "per_subject_csv": str(per_subject_path),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return test_metrics


def variant_defs(profile: str = "ocular_static") -> List[Dict[str, object]]:
    """
    Returns a list of variants with arg overrides for the CLI namespace.
    """
    profile = str(profile).lower().strip()

    if profile == "ocular_static":
        base_cls = {
            "no_regression_head": True,
            "reg_weight": 0.0,
            "contrastive_weight": 0.0,
            "select_metric": "f1",
            "use_static_gate": True,
        }
        return [
            {
                "id": "consensus",
                "name": "PhysioFormer-OS",
                "mode": "consensus",
                "overrides": {},
            },
            {
                "id": "eye_plus_static",
                "name": "Static expert only",
                "overrides": {
                    **base_cls,
                    "no_ppg": True,
                    "no_bilinear": True,
                },
            },
            {
                "id": "raw_seq_multimodal",
                "name": "PhysioFormer-S",
                "overrides": {
                    **base_cls,
                    "no_static": True,
                },
            },
            {
                "id": "no_cross",
                "name": "w/o cross-attn",
                "overrides": {
                    **base_cls,
                    "no_static": True,
                    "no_cross_attn": True,
                },
            },
            {
                "id": "early_fusion",
                "name": "Early fusion",
                "overrides": {
                    **base_cls,
                    "no_static": True,
                    "no_cross_attn": True,
                    "no_bilinear": True,
                },
            },
            {
                "id": "eye_only",
                "name": "Eye only",
                "overrides": {
                    **base_cls,
                    "no_ppg": True,
                    "no_static": True,
                    "no_bilinear": True,
                },
            },
        ]

    return [
        {"id": "full", "name": "Full (legacy)", "overrides": {}},
        {"id": "no_cross", "name": "w/o cross-attn", "overrides": {"no_cross_attn": True}},
        {"id": "no_bilin", "name": "w/o bilinear", "overrides": {"no_bilinear": True}},
        {"id": "no_static", "name": "w/o static feats", "overrides": {"no_static": True}},
        {"id": "eye_only", "name": "eye only", "overrides": {"no_ppg": True, "no_static": True}},
    ]


def apply_overrides(args: argparse.Namespace, overrides: Dict[str, object]) -> argparse.Namespace:
    a = argparse.Namespace(**vars(args))
    for k, v in overrides.items():
        setattr(a, k, v)
    return a


def run_variant(
    variant_id: str,
    variant_name: str,
    args: argparse.Namespace,
    folds: List[List[str]],
    subjects: List[str],
    outdir: Path,
    resume: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Runs CV for one variant. Returns (per_subject_all_df, summary_dict).
    """
    var_out = outdir / variant_id
    folds_dir = var_out / "folds"
    folds_dir.mkdir(parents=True, exist_ok=True)

    per_subject_rows = []
    fold_rows = []

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

        # resume logic
        summ_path = fold_out / "summary.json"
        if resume and summ_path.exists():
            summ = json.loads(summ_path.read_text(encoding="utf-8"))
            test_metrics = summ.get("test_metrics", {})
        else:
            train_dirs = train.resolve_dirs(Path(args.root), train_subjects)
            val_dirs = train.resolve_dirs(Path(args.root), val_subjects)
            test_dirs = train.resolve_dirs(Path(args.root), test_subjects)
            if variant_id == "consensus":
                test_metrics = run_consensus_fold(
                    root=Path(args.root),
                    fold_out=fold_out,
                    args=args,
                    seed=int(args.seed) + i,
                    train_dirs=train_dirs,
                    val_dirs=val_dirs,
                    test_dirs=test_dirs,
                )
            else:
                # ensure per-fold predictions
                args_fold = argparse.Namespace(**vars(args))
                args_fold.save_test_predictions = "predictions_test.csv"

                _ = train.train_one_split(
                    root=Path(args.root),
                    outdir=fold_out,
                    seed=int(args.seed) + i,
                    train_dirs=train_dirs,
                    val_dirs=val_dirs,
                    test_dirs=test_dirs,
                    args=args_fold,
                )
                test_metrics = json.loads((fold_out / "test_metrics.json").read_text(encoding="utf-8"))

        fold_rows.append({"variant": variant_id, "fold": i, **test_metrics})

        subj_csv = fold_out / "per_subject_metrics.csv"
        if subj_csv.exists():
            df_sub = pd.read_csv(subj_csv)
            df_sub["fold"] = i
            df_sub["variant"] = variant_id
            df_sub["variant_name"] = variant_name
            per_subject_rows.append(df_sub)

    per_fold = pd.DataFrame(fold_rows)
    per_fold.to_csv(var_out / "per_fold.csv", index=False, encoding="utf-8")

    per_subject_all = pd.concat(per_subject_rows, axis=0, ignore_index=True)
    per_subject_all.to_csv(var_out / "per_subject_all.csv", index=False, encoding="utf-8")

    summary = {
        "variant_id": variant_id,
        "variant_name": variant_name,
        "n_subjects": int(per_subject_all["participant"].nunique()),
        "n_folds": int(len(folds)),
    }
    (var_out / "cv_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return per_subject_all, summary


def main() -> None:
    ap = train.build_argparser()
    ap.add_argument("--k_folds", "--k", type=int, default=5)
    ap.add_argument("--val_ratio_in_train", type=float, default=0.15)
    ap.add_argument("--ablation_outdir", type=str, default="runs/ablation_physioformer_v3")
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--report_dir", type=str, default="report")
    ap.add_argument("--table_digits", type=int, default=3)
    ap.add_argument("--no_resume", action="store_true", help="Disable resume; always retrain")
    ap.add_argument(
        "--ablation_profile",
        type=str,
        default="ocular_static",
        choices=["ocular_static", "legacy"],
        help="Paper-facing ablation family. ocular_static matches the restored final manuscript.",
    )
    args = ap.parse_args()

    root = Path(args.root)
    outdir = Path(args.ablation_outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    write_placeholder_tables(report_dir)

    subject_dirs = train.list_participants(root)
    subjects = [p.name for p in subject_dirs]

    # -----------------------------
    # Pre-flight dataset integrity check (once, before ablation folds)
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
                print(f"[data-check] Dropped {len(dropped)} subjects with critical issues before ablations. See: {log_path.resolve()}")
            subject_dirs = [p for p in subject_dirs if p.name in ok]
            subjects = [p.name for p in subject_dirs]
        else:
            print(f"[data-check] Completed. See: {log_path.resolve()} (keeping bad subjects per --no_drop_bad_subjects)")

        # avoid repeating the scan inside each variant/fold
        args.skip_data_check = True
        args.data_check_log = ""
    if len(subjects) < 2:
        raise RuntimeError("Need at least 2 subjects for ablations.")

    folds = make_folds(subjects, k=int(args.k_folds), seed=int(args.seed))

    variants = variant_defs(args.ablation_profile)

    # Run all variants
    all_variant_subject = {}
    for v in variants:
        var_args = apply_overrides(args, v["overrides"])
        print(f"[variant] {v['id']} : {v['name']} overrides={v['overrides']}")
        df_sub, summ = run_variant(
            variant_id=str(v["id"]),
            variant_name=str(v["name"]),
            args=var_args,
            folds=folds,
            subjects=subjects,
            outdir=outdir,
            resume=(not args.no_resume),
        )
        all_variant_subject[str(v["id"])] = df_sub

    # Build ablation table (subject-level mean with CI)
    digits = int(args.table_digits)
    metrics = ["f1_macro", "acc", "auc_ovr_macro"]

    rows = []
    for v in variants:
        df = all_variant_subject[str(v["id"])]
        row = [latex_escape(str(v["name"]))]
        for m in ["f1_macro", "acc", "auc_ovr_macro"]:
            seed_off = {'f1_macro': 0, 'acc': 1, 'auc_ovr_macro': 2}.get(m, 9)
            ci = bootstrap_ci(df[m].to_numpy(dtype=np.float64), n_boot=int(args.n_boot), seed=int(args.seed) + seed_off)
            row.append(format_mean_ci(ci, digits=digits))
        rows.append(row)

    table_path = report_dir / "tables" / "ablation.tex"
    write_latex_table(
        table_path,
        caption="Ablation study (mean with 95\\% bootstrap CI; resampling over subjects).",
        label="tab:ablation",
        col_names=["Variant", "Macro-F1", "Acc", "AUC (OvR)"],
        rows=rows,
        notes="All variants evaluated with identical subject folds (paired design).",
    )

    # Statistical tests: compare each variant vs full on macro-F1
    ref_key = "consensus" if str(args.ablation_profile) == "ocular_static" else "full"
    full_df = all_variant_subject[ref_key].set_index("participant")
    comp_rows = []
    pvals = []
    raw = []

    for v in variants:
        vid = str(v["id"])
        if vid == ref_key:
            continue
        dfv = all_variant_subject[vid].set_index("participant")
        # intersection of subjects (should be all)
        common = sorted(list(set(full_df.index).intersection(set(dfv.index))))
        x = full_df.loc[common, "f1_macro"].to_numpy(dtype=np.float64)
        y = dfv.loc[common, "f1_macro"].to_numpy(dtype=np.float64)

        tests = paired_significance_tests(x, y)
        p = tests.get("p_wilcoxon", float("nan"))
        pvals.append(p)
        raw.append((v["name"], tests))

    # Holm correction
    p_adj = holm_bonferroni(pvals) if pvals else []

    for (name, tests), p_corr in zip(raw, p_adj):
        comp_rows.append([
            latex_escape(f"{variants[0]['name']} vs {name}"),
            "Wilcoxon",
            f"{tests.get('p_wilcoxon', float('nan')):.3g}",
            f"d={tests.get('d_cohen', float('nan')):.2f}, Holm $p$={p_corr:.3g}",
        ])

    stats_path = report_dir / "tables" / "stats_tests.tex"
    write_latex_table(
        stats_path,
        caption="Paired statistical comparison (subject-level macro-F1) between the full model and ablations.",
        label="tab:stats",
        col_names=["Comparison", "Test", "$p$", "Effect size / correction"],
        rows=comp_rows if comp_rows else [["NA", "NA", "NA", "NA"]],
        notes="Wilcoxon signed-rank test over per-subject macro-F1. Multiple comparisons corrected with Holm-Bonferroni.",
    )

    ablation_summary = {
        "profile": str(args.ablation_profile),
        "variants": [{k: v[k] for k in ("id", "name")} for v in variants],
        "table_ablation": str(table_path),
        "table_stats": str(stats_path),
    }
    (outdir / "ablation_summary.json").write_text(json.dumps(ablation_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[done] ablation tables -> {table_path.resolve()}, {stats_path.resolve()}")
    print(f"[done] outputs -> {outdir.resolve()}")


if __name__ == "__main__":
    main()
