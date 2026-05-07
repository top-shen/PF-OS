# HPO-CLD PhysioFormer (v3, journal-grade)

This repository contains a **reproducible, subject-wise** multimodal pipeline for cognitive workload modeling
from **Tobii eye tracking** + **BITalino PPG** recordings in an HPO-CLD-like file format.

v3 focus: **publication-ready experiments** (CV + CI + ablations + explainability + physiological consistency).

---

## 0) Expected folder layout

```
project/
  data.py
  net.py
  train.py
  test.py
  run_cv.py
  run_ablation.py
  explain.py
  stats_utils.py
  paper_utils.py
  report/
    main.tex
    refs.bib
    tables/   (auto-generated; placeholders included)
    figs/     (auto-generated)
  HPO-CLD/
    HPO-CLD001/
      *tobii*.csv
      *bitalino*.csv
      *labels*.csv
    HPO-CLD002/
    ...
```

**All timestamps are expected to be in Unix microseconds** for Tobii/PPG and labels.

---

## 1) Install dependencies

```bash
pip install -r requirements.txt
```

## 1.5) Dataset integrity check (recommended)

Before long training runs, run a fast integrity scan (logs + JSON summary):

```bash
python check_data.py --root HPO-CLD --log runs/data_check.log --window_sec 10 --stride_sec 5
```

By default, **train.py / run_cv.py / run_ablation.py / test.py / explain.py** also run this scan automatically
and write a `data_check*.log` file into the corresponding output directory.

If you are confident your data are clean and want to skip it:
- add `--skip_data_check`

If you want to keep subjects with critical issues (not recommended):
- add `--no_drop_bad_subjects`


---

## 2) Single training run (quick sanity)

```bash
python train.py --root HPO-CLD --outdir runs/physioformer_v3 --amp --verbose
```

This performs a **subject-wise split** when the dataset has ≥3 subjects.
(If only 1–2 subjects exist, it falls back to a window-level split for debugging only.)

Outputs include:
- `runs/physioformer_v3/best.pt`
- `runs/physioformer_v3/predictions_test.csv`
- `runs/physioformer_v3/per_subject_metrics.csv`

Checkpoint selection defaults to `--select_metric f1` (classification-first).
If you want the older auxiliary-aware policy, use `--select_metric composite`.

---

## 3) Journal-grade evaluation: subject-wise K-fold CV + 95% CI

```bash
python run_cv.py --root HPO-CLD --k_folds 5 --cv_outdir runs/cv_physioformer_v3 --amp
```

`run_cv.py` supports two paper-facing profiles for the ocular-static family:

- `--cv_profile classification`
  Reproduces the single ocular-static expert:
- `no_ppg`
- `no_bilinear`
- `no_regression_head`
- `reg_weight=0`
- `contrastive_weight=0`
- `use_static_gate`
- `--cv_profile consensus`
  Runs the same ocular-static expert together with the matched `eye-only` expert under
  identical folds, then averages their test probabilities with a fixed 0.5/0.5 rule.

The updated main paper result uses `--cv_profile consensus`, which keeps the comparison
fair because both internal experts are trained/evaluated on the same subject-wise splits
and the fusion rule has no tuned test-time parameters.
To reproduce the older full-objective setting, pass `--cv_profile full`.

This will:
- train/evaluate across folds (subject-wise)
- export per-subject metrics (`per_subject_all.csv`)
- compute 95% bootstrap CIs (resampling **subjects**, not windows)
- write a LaTeX table to `report/tables/main_results.tex`

---

## 4) Ablations + significance tests (paired across subjects)

```bash
python run_ablation.py --root HPO-CLD --k_folds 5 --ablation_outdir runs/ablation_physioformer_v3 --amp
```

Outputs:
- `report/tables/ablation.tex` (mean + 95% CI)
- `report/tables/stats_tests.tex` (paired Wilcoxon + Holm correction)

> Tip: add `--no_resume` if you want to force retraining even when fold outputs already exist.

---

## 5) Explainability: attention maps + HRV consistency

```bash
python explain.py --root HPO-CLD --ckpt runs/physioformer_v3/best.pt
```

Writes:
- `report/figs/attn_example.png` (cross-attention heatmap example)
- `report/figs/hrv_consistency.png` (subject-wise correlation distribution example)
- `report/tables/hrv_consistency.tex` (journal-ready table)
- `report/analysis_windows.csv` (window-level analysis file; includes HRV + attention summaries)

---

## 6) Compile the paper (LaTeX)

From the project root:

```bash
cd report
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

All tables/figures are included via `\input{tables/...}` and `\IfFileExists{figs/...}`.
Placeholders are shipped so the paper compiles even before you run experiments.

---

## Methodology note (important)

Physiological windows within a subject are correlated.  
Therefore, **all confidence intervals and hypothesis tests are computed at the subject level**
(per-subject metrics or within-subject correlations → then aggregated across subjects).
