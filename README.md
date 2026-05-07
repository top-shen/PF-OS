# PF-OS: Ocular-Dominant Workload Sensing for Adaptive Virtual Reality Displays

This repository contains the code used for the paper **PF-OS: Ocular-Dominant Workload Sensing for Adaptive Virtual Reality Displays**. The project studies short-window cognitive workload decoding in VR from eye-tracking and PPG signals, and argues that 10-second VR workload sensing should be treated as an ocular-dominant expert-allocation problem rather than symmetric raw multimodal fusion.

PF-OS, short for PhysioFormer-Ocular-Static, uses a dual-expert design:

- an ocular-static expert, where eye-tracking sequences provide the main temporal evidence and PPG/HRV enters through robust static summaries;
- a matched eye-only expert, combined at decision level with a fixed consensus rule.

The manuscript evaluates PF-OS on the HP Omnicept Cognitive Load Dataset with subject-wise 5-fold cross-validation over 98 participants.

## Main Results

| Family | Method | Acc. (%) | Macro-F1 (%) | AUC (%) |
| --- | --- | ---: | ---: | ---: |
| Shallow | KNN | 53.80 +/- 2.04 | 52.21 +/- 2.25 | 73.66 +/- 1.60 |
| Shallow | Gaussian NB | 57.65 +/- 1.90 | 51.56 +/- 2.22 | 78.20 +/- 1.69 |
| Shallow | LDA | 63.09 +/- 1.99 | 59.06 +/- 2.40 | 83.07 +/- 1.39 |
| Shallow | SVM | 61.96 +/- 2.01 | 58.52 +/- 2.37 | 82.37 +/- 1.35 |
| Shallow | Logistic regression | 64.17 +/- 2.02 | 60.47 +/- 2.38 | 83.25 +/- 1.46 |
| Shallow | Random forest | 64.31 +/- 2.02 | 61.72 +/- 2.34 | 82.96 +/- 1.31 |
| Deep | TFN | 61.59 +/- 2.33 | 59.14 +/- 2.54 | 81.15 +/- 1.64 |
| Deep | MLP | 64.61 +/- 2.31 | 61.33 +/- 2.70 | 83.66 +/- 1.73 |
| Deep | CNN | 70.79 +/- 2.63 | 68.74 +/- 2.93 | 88.23 +/- 1.74 |
| Deep | LSTM | 71.78 +/- 2.72 | 69.50 +/- 3.08 | 88.66 +/- 1.64 |
| Ours | PF-OS | **72.66 +/- 2.40** | **70.90 +/- 2.67** | **88.41 +/- 1.60** |

## Ablation Results

| Variant | Acc. (%) | Macro-F1 (%) | AUC (%) | Delta Macro-F1 vs PF-OS (pp) | Adj. p | Cohen's d |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PF-OS | **72.66 +/- 2.40** | **70.90 +/- 2.67** | **88.41 +/- 1.60** | - | - | - |
| Static expert only | 70.54 +/- 2.29 | 68.74 +/- 2.62 | 86.67 +/- 1.56 | -2.15 | 9.52e-04 | 0.39 |
| PF-S | 70.73 +/- 2.17 | 68.74 +/- 2.47 | 86.27 +/- 1.65 | -2.16 | 3.17e-02 | 0.27 |
| w/o cross-attention | 69.77 +/- 2.35 | 67.40 +/- 2.70 | 86.20 +/- 1.71 | -3.50 | 1.51e-03 | 0.38 |
| Early fusion | 69.57 +/- 2.33 | 67.31 +/- 2.68 | 85.49 +/- 1.72 | -3.59 | 2.78e-03 | 0.37 |
| Eye only | 71.76 +/- 2.19 | 70.13 +/- 2.39 | 87.70 +/- 1.70 | -0.77 | 0.239 | 0.12 |

## Repository Layout

```text
PF-OS/
  PF-OS/                         Main model, training, CV, ablation, and analysis scripts
  comparison/                    Unified baseline and fair-comparison runners
  KNN/, LDA/, SVM/, ...          Recovered baseline entry points
  PhysioFormer-S/                Raw-sequence multimodal ablation wrapper
  Early Fusion Transformer/      Early-fusion ablation wrapper
  Eye Only Transformer/          Eye-only ablation wrapper
  No Cross Attention/            No-cross-attention ablation wrapper
  TFN/                           Tensor Fusion Network baseline
  signal_visualizations/         Signal visualization script and example figures
  report/tables/                 Small generated paper tables retained for reference
  baseline_common.py             Shared static-feature baseline utilities
```

Generated data, caches, checkpoints, fold-level predictions, and training runs are intentionally excluded from Git by `.gitignore`.

## Data

The raw HP Omnicept Cognitive Load Dataset is not included in this repository. Place the dataset under `HPO-CLD/` using this structure:

```text
HPO-CLD/
  HPO-CLD001/
    *tobii*.csv
    *bitalino*.csv
    *labels*.csv
  HPO-CLD002/
  ...
```

Timestamps are expected in Unix microseconds. The default windowing protocol is 10 seconds with a 5-second stride.

## Environment

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Install the PyTorch build that matches your machine first if you need CUDA support.

## Reproduction

Run a dataset integrity check:

```bash
python PF-OS/check_data.py --root HPO-CLD --log runs/data_check.log --window_sec 10 --stride_sec 5
```

Run PF-OS subject-wise 5-fold CV:

```bash
python PF-OS/run_cv.py --root HPO-CLD --k_folds 5 --cv_outdir runs/cv_pf_os --cv_profile consensus --amp
```

Run the fair baseline suite:

```bash
python comparison/prepare_data.py --root HPO-CLD --outdir prepared_data
python comparison/run_fair_suite.py --root HPO-CLD --data_dir prepared_data --outdir comparison/results/fair_suite
```

Run controlled ablations:

```bash
python PF-OS/run_ablation.py --root HPO-CLD --k_folds 5 --ablation_outdir runs/ablation_pf_os --amp
```

## Notes

All reported confidence intervals and paired tests are computed at subject level, not window level, to avoid over-counting correlated windows from the same participant.

No raw dataset, participant-level cache, trained checkpoint, or full prediction dump is committed to this repository.
