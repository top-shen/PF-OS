# -*- coding: utf-8 -*-
"""
explain.py

Explainability + physiological consistency analysis for PhysioFormer-v3.

What it produces (journal-ready assets)
---------------------------------------
- Attention heatmaps (cross-modal attention) for representative windows
- Window-level CSV with predicted probabilities, attention summaries, and HRV features
- Subject-wise Spearman correlation:
    (A) p(High) vs HRV indices
    (B) attention summary vs HRV indices  (attention–physiology consistency)
- LaTeX table for the paper: report/tables/hrv_consistency.tex
- Figures: report/figs/attn_example.png, report/figs/hrv_consistency.png, report/figs/hrv_consistency_attn.png

Methodological note
-------------------
Windows within the same participant are NOT independent. Therefore:
1) correlations are computed *within each subject*
2) then summarized across subjects (Fisher-z + bootstrap CI)
3) and significance is tested at the subject level

Author: assistant (v3)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader, Subset

from data import MultiModalWindowDataset, get_static_feature_names, validate_dataset
from net import ModelConfig, PhysioFormerNet
from train import Standardizer, StaticNormWrapper, attention_entropy, attention_max, dataset_meta_list
from stats_utils import bootstrap_ci
from paper_utils import write_latex_table, write_placeholder_tables

try:
    from scipy import stats
except Exception:  # pragma: no cover
    stats = None


def pick_hrv_columns() -> List[str]:
    # A compact, interpretable HRV subset (available in static features)
    return [
        "stat_hr_mean_bpm",
        "stat_rmssd_s",
        "stat_sdnn_s",
        "stat_lf_hf_ratio",
        "stat_ppg_quality_0_1",
    ]


@torch.no_grad()
def predict_with_attention(
    model: PhysioFormerNet,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Returns:
      y_true, y_pred, y_proba
      attn_summaries: entropy/max per window (aligned with dataset order)
    """
    model.eval()
    ys, yp, prob = [], [], []
    ent_e2p, ent_p2e, max_e2p, max_p2e = [], [], [], []

    for eye_seq, ppg_seq, x_static, y, y_reg in loader:
        eye_seq = eye_seq.to(device)
        ppg_seq = ppg_seq.to(device)
        x_static = x_static.to(device)

        logits, _tlx_hat, _e, _p, attn = model(eye_seq, ppg_seq, x_static, return_embeddings=False, return_attn=True)
        p = torch.softmax(logits, dim=1)
        pred = torch.argmax(p, dim=1)

        ys.append(y.numpy())
        yp.append(pred.cpu().numpy())
        prob.append(p.cpu().numpy())

        if attn is None or "e2p" not in attn or attn["e2p"] is None:
            # cross-attn disabled
            bsz = int(p.shape[0])
            ent_e2p.extend([np.nan] * bsz)
            ent_p2e.extend([np.nan] * bsz)
            max_e2p.extend([np.nan] * bsz)
            max_p2e.extend([np.nan] * bsz)
            continue

        w_e2p = attn["e2p"].detach().cpu().numpy()  # (B,H,Tq,Tk)
        w_p2e = attn["p2e"].detach().cpu().numpy()

        for b in range(w_e2p.shape[0]):
            ent_e2p.append(attention_entropy(w_e2p[b]))
            ent_p2e.append(attention_entropy(w_p2e[b]))
            max_e2p.append(attention_max(w_e2p[b]))
            max_p2e.append(attention_max(w_p2e[b]))

    y_true = np.concatenate(ys, axis=0) if ys else np.array([], dtype=int)
    y_pred = np.concatenate(yp, axis=0) if yp else np.array([], dtype=int)
    y_proba = np.concatenate(prob, axis=0) if prob else np.zeros((0, 3), dtype=np.float32)

    attn_sum = {
        "attn_ent_e2p": np.asarray(ent_e2p, dtype=np.float32),
        "attn_ent_p2e": np.asarray(ent_p2e, dtype=np.float32),
        "attn_max_e2p": np.asarray(max_e2p, dtype=np.float32),
        "attn_max_p2e": np.asarray(max_p2e, dtype=np.float32),
    }
    return y_true, y_pred, y_proba, attn_sum


def save_attention_example_figure(out_png: Path, attn_w_e2p: np.ndarray, title: str) -> None:
    """
    Saves a single heatmap for e2p attention averaged over heads.
    """
    import matplotlib.pyplot as plt

    if attn_w_e2p.ndim == 3:
        w = np.mean(attn_w_e2p, axis=0)
    else:
        w = attn_w_e2p
    plt.figure(figsize=(5.2, 4.4))
    plt.imshow(w, aspect="auto", interpolation="nearest")
    plt.colorbar(label="attention weight")
    plt.xlabel("PPG tokens")
    plt.ylabel("Eye tokens")
    plt.title(title)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def fisher_z(r: np.ndarray) -> np.ndarray:
    r = np.clip(r, -0.999999, 0.999999)
    return np.arctanh(r)


def inv_fisher_z(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)


def summarize_correlations(corr_df: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    """
    corr_df columns: signal, target, rho
    Returns summary rows with mean rho and CI in rho-space.
    """
    out_rows = []
    for (signal, target), sub in corr_df.groupby(["signal", "target"]):
        rhos = sub["rho"].to_numpy(dtype=np.float64)
        zs = fisher_z(rhos)
        import zlib
        stable = zlib.adler32((signal + '|' + target).encode('utf-8'))
        ci = bootstrap_ci(zs, statistic=lambda a: float(np.mean(a)), n_boot=int(n_boot), seed=int(seed) + int(stable % 10000))
        mean_r = float(inv_fisher_z(ci.mean))
        lo_r = float(inv_fisher_z(ci.lo))
        hi_r = float(inv_fisher_z(ci.hi))

        # one-sample test on z-scores
        pval = float("nan")
        if stats is not None and zs.size >= 2:
            try:
                t = stats.ttest_1samp(zs, 0.0)
                pval = float(t.pvalue)
            except Exception:
                pval = float("nan")

        out_rows.append({
            "signal": signal,
            "target": target,
            "mean_rho": mean_r,
            "lo_rho": lo_r,
            "hi_rho": hi_r,
            "p": pval,
            "n_subjects": int(sub["participant"].nunique()),
        })
    return pd.DataFrame(out_rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="HPO-CLD")
    ap.add_argument("--ckpt", type=str, required=True, help="Checkpoint path from train.py (best.pt)")
    ap.add_argument("--participants", type=str, default="", help="Comma-separated participants; default=all")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--report_dir", type=str, default="report")
    ap.add_argument("--out_csv", type=str, default="report/analysis_windows.csv")
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--table_digits", type=int, default=3)
    ap.add_argument("--skip_data_check", action="store_true", help="Skip pre-flight dataset integrity scan")
    ap.add_argument("--data_check_log", type=str, default="", help="Data-check log file (default: <report_dir>/data_check_explain.log)")
    ap.add_argument("--no_drop_bad_subjects", action="store_true", help="Do not drop subjects with critical data issues (may crash later)")
    args = ap.parse_args()

    device = torch.device(args.device)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ckpt["model_config"])
    model = PhysioFormerNet(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    std = Standardizer()
    std.load_state_dict(ckpt["static_standardizer"])

    # Use training windowing config if present
    train_args = ckpt.get("args", {})
    window_sec = float(train_args.get("window_sec", 10.0))
    stride_sec = float(train_args.get("stride_sec", 5.0))
    seq_len = int(train_args.get("seq_len", cfg.seq_len))
    cache_dir = Path(train_args.get("cache_dir", "cache_mm"))

    root = Path(args.root)
    pdirs = sorted([p for p in root.iterdir() if p.is_dir() and p.name.lower().startswith("hpo-cld")])
    if not pdirs:
        raise RuntimeError(f"No participant folders found: {root.resolve()}")

    if args.participants.strip():
        wanted = set([s.strip() for s in args.participants.split(",") if s.strip()])
        pdirs = [p for p in pdirs if p.name in wanted]
        if not pdirs:
            raise RuntimeError("No matching participant folders found for --participants")

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    write_placeholder_tables(report_dir)
    (report_dir / "figs").mkdir(parents=True, exist_ok=True)


    # -----------------------------
    # Pre-flight dataset integrity check (optional)
    # -----------------------------
    if not args.skip_data_check:
        log_path = Path(args.data_check_log).expanduser() if args.data_check_log else (report_dir / "data_check_explain.log")
        summary = validate_dataset(
            root=root,
            participant_dirs=pdirs,
            log_path=log_path,
            window_sec=float(window_sec),
            stride_sec=float(stride_sec),
        )
        if not args.no_drop_bad_subjects:
            ok = set(summary.get("ok_subjects", []))
            dropped = [p.name for p in pdirs if p.name not in ok]
            if dropped:
                print(f"[data-check] Dropped {len(dropped)} subjects with critical issues before explain. See: {log_path.resolve()}")
            pdirs = [p for p in pdirs if p.name in ok]
        else:
            print(f"[data-check] Completed. See: {log_path.resolve()} (keeping bad subjects per --no_drop_bad_subjects)")
        if not pdirs:
            raise RuntimeError("No valid participants left after data check. Inspect data_check_explain.log or pass --no_drop_bad_subjects.")
    ds_raw = MultiModalWindowDataset(
        pdirs, window_sec=window_sec, stride_sec=stride_sec,
        seq_len=seq_len, cache_dir=cache_dir,
        augment=False, seed=42, verbose=False,
    )
    ds = StaticNormWrapper(ds_raw, std)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    y_true, y_pred, y_proba, attn_sum = predict_with_attention(model, dl, device)

    # Construct window-level table (raw static features for physiology analysis)
    metas = dataset_meta_list(ds_raw)
    static_raw = []
    for i in range(len(ds_raw)):
        _eye_seq, _ppg_seq, x_static_raw, _y, _y_reg = ds_raw[i]
        static_raw.append(np.asarray(x_static_raw, dtype=np.float32))
    static_raw = np.stack(static_raw, axis=0) if static_raw else np.zeros((0, len(get_static_feature_names())), dtype=np.float32)

    names = get_static_feature_names()
    rows = []
    for i, meta in enumerate(metas):
        r = {
            "participant": str(getattr(meta, "participant")),
            "t_start_us": int(getattr(meta, "t_start_us")),
            "t_end_us": int(getattr(meta, "t_end_us")),
            "y_true": int(y_true[i]),
            "y_pred": int(y_pred[i]),
            "p_low": float(y_proba[i, 0]),
            "p_med": float(y_proba[i, 1]),
            "p_high": float(y_proba[i, 2]),
        }
        # all static features
        for j, name in enumerate(names):
            r[f"stat_{name}"] = float(static_raw[i, j])

        # attention summaries
        for k, arr in attn_sum.items():
            r[k] = float(arr[i]) if i < len(arr) else float("nan")
        rows.append(r)

    df = pd.DataFrame(rows)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8")

    # --- correlations ---
    hrv_cols = pick_hrv_columns()
    targets = [
        ("p_high", "p(High)"),
        ("attn_ent_e2p", "AttnEntropy(Eye→PPG)"),
        ("attn_max_e2p", "AttnMax(Eye→PPG)"),
    ]

    corr_rows = []
    if stats is not None:
        for pid, g in df.groupby("participant"):
            if len(g) < 5:
                continue
            for target_col, target_name in targets:
                x = g[target_col].to_numpy(dtype=np.float64)
                for sig in hrv_cols:
                    y = g[sig].to_numpy(dtype=np.float64)
                    mask = np.isfinite(x) & np.isfinite(y)
                    if mask.sum() < 5:
                        continue
                    rho, p = stats.spearmanr(x[mask], y[mask])
                    corr_rows.append({
                        "participant": pid,
                        "signal": sig,
                        "target": target_name,
                        "rho": float(rho),
                        "p": float(p),
                        "n": int(mask.sum()),
                    })

    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(report_dir / "corr_subjectwise.csv", index=False, encoding="utf-8")

    summary_df = summarize_correlations(corr_df, n_boot=int(args.n_boot), seed=42) if not corr_df.empty else pd.DataFrame()
    summary_df.to_csv(report_dir / "corr_summary.csv", index=False, encoding="utf-8")

    # --- LaTeX table ---
    digits = int(args.table_digits)
    table_rows = []
    for _, r in summary_df.iterrows():
        sig = str(r["signal"]).replace("stat_", "").replace("_", r"\_")
        tgt = str(r["target"]).replace("_", r"\_")
        table_rows.append([
            sig,
            tgt,
            f"{float(r['mean_rho']):.{digits}f} [{float(r['lo_rho']):.{digits}f}, {float(r['hi_rho']):.{digits}f}]",
            f"{float(r['p']):.3g}" if np.isfinite(float(r["p"])) else "NA",
        ])

    hrv_table_path = report_dir / "tables" / "hrv_consistency.tex"
    write_latex_table(
        hrv_table_path,
        caption="Physiological consistency analysis. Subject-wise Spearman correlation between model outputs and HRV indices (mean with 95\\% bootstrap CI across subjects).",
        label="tab:hrv",
        col_names=["HRV index", "Model target", "Mean $\\rho$ (95\\% CI)", "$p$"],
        rows=table_rows if table_rows else [["NA", "NA", "NA", "NA"]],
        notes="Correlations are computed within each subject. Summary uses Fisher-$z$ transform and bootstrap CIs over subjects; $p$ from one-sample t-test on Fisher-$z$.",
    )

    # --- Figures ---
    figs_dir = report_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    # 1) Attention map example: pick the window with max p_high and draw Eye→PPG attention
    try:
        idx_best = int(df["p_high"].to_numpy().argmax()) if len(df) else 0
        ds_one_raw = Subset(ds_raw, [idx_best])
        ds_one = StaticNormWrapper(ds_one_raw, std)
        dl_one = DataLoader(ds_one, batch_size=1, shuffle=False)

        for eye_seq, ppg_seq, x_static, y, y_reg in dl_one:
            eye_seq = eye_seq.to(device)
            ppg_seq = ppg_seq.to(device)
            x_static = x_static.to(device)
            logits, _tlx, _e, _p, attn = model(eye_seq, ppg_seq, x_static, return_embeddings=False, return_attn=True)
            if attn is not None and attn.get("e2p") is not None:
                w = attn["e2p"].detach().cpu().numpy()[0]  # (H,Tq,Tk)
                save_attention_example_figure(figs_dir / "attn_example.png", w, title="Cross-attention (Eye→PPG) example")
            break
    except Exception:
        pass

    # 2) Correlation distribution plots for RMSSD
    try:
        import matplotlib.pyplot as plt

        sig = "stat_rmssd_s"
        # a) p(High) vs RMSSD
        sub1 = corr_df[(corr_df["signal"] == sig) & (corr_df["target"] == "p(High)")]
        if not sub1.empty:
            plt.figure(figsize=(5.2, 3.6))
            vals = sub1["rho"].to_numpy(dtype=np.float64)
            plt.boxplot(vals, vert=True)
            plt.axhline(0.0, linestyle="--")
            plt.ylabel("Spearman ρ (subject-wise)")
            plt.title("Consistency: p(High) vs RMSSD")
            plt.tight_layout()
            plt.savefig(figs_dir / "hrv_consistency.png", dpi=200)
            plt.close()

        # b) Attention entropy vs RMSSD
        sub2 = corr_df[(corr_df["signal"] == sig) & (corr_df["target"] == "AttnEntropy(Eye→PPG)")]
        if not sub2.empty:
            plt.figure(figsize=(5.2, 3.6))
            vals = sub2["rho"].to_numpy(dtype=np.float64)
            plt.boxplot(vals, vert=True)
            plt.axhline(0.0, linestyle="--")
            plt.ylabel("Spearman ρ (subject-wise)")
            plt.title("Consistency: AttnEntropy(Eye→PPG) vs RMSSD")
            plt.tight_layout()
            plt.savefig(figs_dir / "hrv_consistency_attn.png", dpi=200)
            plt.close()
    except Exception:
        pass

    print(f"[saved] window-level analysis CSV -> {out_csv.resolve()}")
    print(f"[saved] LaTeX table -> {hrv_table_path.resolve()}")
    print(f"[saved] figs -> {figs_dir.resolve()}")


if __name__ == "__main__":
    main()
