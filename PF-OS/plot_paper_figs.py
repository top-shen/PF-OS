# plot_paper_figs.py
# -*- coding: utf-8 -*-
"""
Nature-style plotting for HPO-CLD model results.

Inputs:
  --pred_csv: predictions saved by test.py (--save_pred)
              must include columns:
              participant,y_true,y_pred,p_low,p_med,p_high,t_start_us,t_end_us
Outputs:
  figures in --outdir:
    fig_confusion.pdf/png
    fig_roc.pdf/png
    fig_calibration.pdf/png
    fig_subject_metrics.pdf/png
    fig_timeline_<participant>.pdf/png  (optional)
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import json

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    brier_score_loss,
)

# -------------------------
# Global plotting style (Nature-like)
# -------------------------
def set_nature_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "savefig.dpi": 300,
        "figure.dpi": 150,
    })

def savefig(fig, outpath: Path):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(outpath.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)

# -------------------------
# Utilities
# -------------------------
CLASSES = ["Low", "Med", "High"]

def to_onehot(y: np.ndarray, k: int = 3) -> np.ndarray:
    y = y.astype(int)
    oh = np.zeros((len(y), k), dtype=float)
    oh[np.arange(len(y)), np.clip(y, 0, k-1)] = 1.0
    return oh

def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    rng = np.random.RandomState(seed)
    vals = values[~np.isnan(values)]
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(vals, size=len(vals), replace=True)
        boots.append(np.mean(sample))
    boots = np.array(boots)
    lo = np.quantile(boots, alpha/2)
    hi = np.quantile(boots, 1 - alpha/2)
    return float(np.mean(vals)), float(lo), float(hi)

def macro_f1(y_true, y_pred, k=3):
    f1s = []
    for c in range(k):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        prec = tp / (tp + fp + 1e-12)
        rec  = tp / (tp + fn + 1e-12)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        f1s.append(f1)
    return float(np.mean(f1s))

def ece_score(y_true: np.ndarray, p_high: np.ndarray, n_bins: int = 10):
    # binary calibration for High vs not-High
    y = (y_true == 2).astype(int)
    p = np.clip(p_high, 0, 1)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (p >= bins[i]) & (p < bins[i+1])
        if np.sum(m) == 0:
            continue
        acc = np.mean(y[m])
        conf = np.mean(p[m])
        ece += np.abs(acc - conf) * (np.sum(m) / len(p))
    return float(ece)

# -------------------------
# Figures
# -------------------------
def fig_confusion(df: pd.DataFrame, outdir: Path):
    y_true = df["y_true"].to_numpy(int)
    y_pred = df["y_pred"].to_numpy(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2]).astype(float)
    cmn = cm / (cm.sum(axis=1, keepdims=True) + 1e-12)

    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    im = ax.imshow(cmn, vmin=0, vmax=1)

    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cmn[i,j]*100:.1f}%", ha="center", va="center")

    ax.set_xticks([0,1,2], CLASSES)
    ax.set_yticks([0,1,2], CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Normalized confusion matrix")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Proportion")

    savefig(fig, outdir / "fig_confusion")

def fig_roc_ovr(df: pd.DataFrame, outdir: Path):
    y = df["y_true"].to_numpy(int)
    P = df[["p_low","p_med","p_high"]].to_numpy(float)
    Y = to_onehot(y, 3)

    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    for c, name in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(Y[:,c], P[:,c])
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr,tpr):.3f})")

    ax.plot([0,1],[0,1], linestyle="--", linewidth=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("One-vs-rest ROC")
    ax.legend(frameon=False, loc="lower right")
    savefig(fig, outdir / "fig_roc_ovr")

def fig_calibration_high(df: pd.DataFrame, outdir: Path, n_bins: int = 10):
    # calibration for High vs not-High (medical papers often accept a primary class calibration)
    y = (df["y_true"].to_numpy(int) == 2).astype(int)
    p = df["p_high"].to_numpy(float)

    bins = np.linspace(0,1,n_bins+1)
    xs, ys, ns = [], [], []
    for i in range(n_bins):
        m = (p >= bins[i]) & (p < bins[i+1])
        if np.sum(m) == 0:
            continue
        xs.append(np.mean(p[m]))
        ys.append(np.mean(y[m]))
        ns.append(np.sum(m))

    ece = ece_score(df["y_true"].to_numpy(int), p, n_bins=n_bins)
    brier = brier_score_loss(y, p)

    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.plot([0,1],[0,1], linestyle="--", linewidth=0.8, label="Ideal")
    ax.plot(xs, ys, marker="o", linewidth=1.2, label=f"Model (ECE={ece:.3f}, Brier={brier:.3f})")
    ax.set_xlabel("Mean predicted probability (High)")
    ax.set_ylabel("Observed frequency (High)")
    ax.set_title("Calibration curve (High vs rest)")
    ax.legend(frameon=False, loc="upper left")
    savefig(fig, outdir / "fig_calibration_high")

def fig_subject_metrics(df: pd.DataFrame, outdir: Path):
    # per-participant macro-F1 and accuracy (window-level, but aggregated to subject)
    rows = []
    for pid, g in df.groupby("participant"):
        y_true = g["y_true"].to_numpy(int)
        y_pred = g["y_pred"].to_numpy(int)
        acc = float(np.mean(y_true == y_pred)) if len(y_true) else np.nan
        f1 = macro_f1(y_true, y_pred, 3) if len(y_true) else np.nan
        rows.append({"participant": pid, "acc": acc, "f1_macro": f1})
    sub = pd.DataFrame(rows).sort_values("f1_macro")

    mean_f1, lo_f1, hi_f1 = bootstrap_ci(sub["f1_macro"].to_numpy(float))
    mean_acc, lo_acc, hi_acc = bootstrap_ci(sub["acc"].to_numpy(float))

    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    x = np.arange(len(sub))
    ax.scatter(x, sub["f1_macro"], s=12, label="Subject macro-F1")
    ax.axhline(mean_f1, linewidth=1.2)
    ax.fill_between([x.min(), x.max()], [lo_f1, lo_f1], [hi_f1, hi_f1], alpha=0.15)

    ax.set_xlabel("Participants (sorted)")
    ax.set_ylabel("Macro-F1")
    ax.set_title(f"Per-subject performance (mean={mean_f1:.3f}, 95%CI [{lo_f1:.3f},{hi_f1:.3f}])")
    ax.legend(frameon=False, loc="lower right")
    savefig(fig, outdir / "fig_subject_f1")

    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.scatter(x, sub["acc"], s=12, label="Subject accuracy")
    ax.axhline(mean_acc, linewidth=1.2)
    ax.fill_between([x.min(), x.max()], [lo_acc, lo_acc], [hi_acc, hi_acc], alpha=0.15)
    ax.set_xlabel("Participants (sorted)")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Per-subject accuracy (mean={mean_acc:.3f}, 95%CI [{lo_acc:.3f},{hi_acc:.3f}])")
    ax.legend(frameon=False, loc="lower right")
    savefig(fig, outdir / "fig_subject_acc")


def fig_train_loss_curve(loss_history: list, outdir: Path, target_loss: float = 0.015):
    """绘制训练损失曲线"""
    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    epochs = range(1, len(loss_history) + 1)
    ax.plot(epochs, loss_history, linewidth=1.2, color='#3498db', label='Train Loss')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train Loss")
    ax.set_title("Training Loss Curve")
    ax.axhline(y=target_loss, color='r', linestyle='--', linewidth=0.8, label=f'Target ({target_loss})')
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3, linestyle='--')
    savefig(fig, outdir / "fig_train_loss")


def fig_timeline_one(df: pd.DataFrame, root: Path, participant: str, outdir: Path):
    # Optional: timeline plot with label blocks shaded
    g = df[df["participant"] == participant].copy()
    if len(g) == 0:
        return

    # x-axis: minutes from first window start
    t0 = g["t_start_us"].min()
    x = (g["t_start_us"] - t0).to_numpy() / 1e6 / 60.0

    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    ax.plot(x, g["p_low"].to_numpy(float), linewidth=0.9, label="p(Low)")
    ax.plot(x, g["p_med"].to_numpy(float), linewidth=0.9, label="p(Med)")
    ax.plot(x, g["p_high"].to_numpy(float), linewidth=0.9, label="p(High)")

    # shade label intervals (read labels.csv)
    pdir = root / participant
    labels_path = None
    for cand in sorted(pdir.glob("*labels*.csv")):
        labels_path = cand
    if labels_path is not None and labels_path.exists():
        lab = pd.read_csv(labels_path)
        for _, r in lab.iterrows():
            s = (int(r["time_start"]) - int(t0)) / 1e6 / 60.0
            e = (int(r["time_end"]) - int(t0)) / 1e6 / 60.0
            y = str(r.get("task_difficulty",""))
            # light background, do not force colors too hard
            ax.axvspan(s, e, alpha=0.08)

    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Probability")
    ax.set_title(f"Timeline probabilities: {participant}")
    ax.legend(frameon=False, ncol=3, loc="upper right")
    savefig(fig, outdir / f"fig_timeline_{participant}")

def fig_comparison(df: pd.DataFrame, participant: str, outdir: Path):
    """Ground truth vs predicted probabilities comparison plot."""
    from matplotlib.patches import Patch

    g = df[df["participant"] == participant].copy()
    if len(g) == 0:
        print(f"[warning] No data for participant {participant}")
        return

    g = g.sort_values("t_start_us").reset_index(drop=True)
    t0 = g["t_start_us"].min()
    x = (g["t_start_us"] - t0).to_numpy() / 1e6 / 60.0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.0, 4.5), sharex=True,
                                    gridspec_kw={'height_ratios': [1, 2], 'hspace': 0.08})

    # Top: Ground truth as colored regions
    y_true = g["y_true"].to_numpy(int)
    colors_gt = {'Low': '#2ecc71', 'Med': '#f39c12', 'High': '#e74c3c'}

    for i in range(len(x)):
        label_name = CLASSES[y_true[i]]
        x_start = x[i]
        if i < len(x) - 1:
            x_end = x[i + 1]
        else:
            x_end = x_start + (g["t_end_us"].iloc[i] - g["t_start_us"].iloc[i]) / 1e6 / 60.0
        ax1.axvspan(x_start, x_end, alpha=0.7, color=colors_gt[label_name], linewidth=0)

    legend_elements = [Patch(facecolor=colors_gt[c], alpha=0.7, label=f'GT: {c}') for c in CLASSES]
    ax1.legend(handles=legend_elements, loc='upper right', frameon=False, ncol=3, fontsize=7)
    ax1.set_ylabel("Ground Truth")
    ax1.set_yticks([])
    ax1.set_ylim(0, 1)
    ax1.set_title(f"Ground Truth vs Predictions: {participant}")

    # Bottom: Predicted probabilities as curves
    colors_pred = {'Low': '#27ae60', 'Med': '#e67e22', 'High': '#c0392b'}
    ax2.plot(x, g["p_low"].to_numpy(float), linewidth=1.2, color=colors_pred['Low'], label="p(Low)")
    ax2.plot(x, g["p_med"].to_numpy(float), linewidth=1.2, color=colors_pred['Med'], label="p(Med)")
    ax2.plot(x, g["p_high"].to_numpy(float), linewidth=1.2, color=colors_pred['High'], label="p(High)")

    ax2.set_xlabel("Time (min)")
    ax2.set_ylabel("Predicted Probability")
    ax2.set_ylim(0, 1)
    ax2.legend(frameon=False, loc='upper right', ncol=3)
    ax2.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    savefig(fig, outdir / f"fig_comparison_{participant}")

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", type=str, required=True)
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--root", type=str, default="HPO-CLD", help="dataset root (for timeline shading)")
    ap.add_argument("--timeline_participant", type=str, default="", help="e.g., HPO-CLD001")
    ap.add_argument("--comparison_participant", type=str, default="",
                    help="Generate comparison plot for participant (e.g., HPO-CLD001)")
    ap.add_argument("--comparison_all", action="store_true",
                    help="Generate comparison plots for all participants")
    ap.add_argument("--loss_history_json", type=str, default="",
                    help="Path to train_loss_history.json for plotting training loss curve")
    args = ap.parse_args()

    set_nature_style()
    pred_csv = Path(args.pred_csv)
    outdir = Path(args.outdir)
    root = Path(args.root)

    df = pd.read_csv(pred_csv)
    required = {"participant","y_true","y_pred","p_low","p_med","p_high","t_start_us","t_end_us"}
    miss = required - set(df.columns)
    if miss:
        raise RuntimeError(f"pred_csv missing columns: {sorted(miss)}")

    fig_confusion(df, outdir)
    fig_roc_ovr(df, outdir)
    fig_calibration_high(df, outdir)
    fig_subject_metrics(df, outdir)

    if args.timeline_participant.strip():
        fig_timeline_one(df, root, args.timeline_participant.strip(), outdir)

    if args.comparison_participant.strip():
        fig_comparison(df, args.comparison_participant.strip(), outdir)

    if args.comparison_all:
        for pid in df["participant"].unique():
            fig_comparison(df, pid, outdir)

    # 绘制训练损失曲线
    if args.loss_history_json.strip():
        loss_json_path = Path(args.loss_history_json.strip())
        if loss_json_path.exists():
            loss_history = json.loads(loss_json_path.read_text(encoding="utf-8"))
            fig_train_loss_curve(loss_history, outdir)
        else:
            print(f"[warning] loss_history_json not found: {loss_json_path}")

    print(f"[done] figures saved to: {outdir.resolve()}")

if __name__ == "__main__":
    main()
