# PF-OS

Official code implementation and supplementary material for **PF-OS: Ocular-Dominant Workload Sensing for Adaptive Virtual Reality Displays**(Accepted by IEEE OJCS).

Short-window cognitive workload sensing in virtual reality is difficult because ocular and cardiovascular signals do not behave like equal partners. Eye-tracking features can respond quickly to task demand, while photoplethysmography (PPG) and heart-rate-variability cues evolve more slowly and are more vulnerable to motion or contact artifacts. PF-OS addresses this mismatch with an ocular-dominant, dual-expert Transformer design: eye sequences carry the main temporal evidence, physiological signals enter as robust static context, and a matched eye-only expert supplies complementary decision-level evidence.

<p align="center">
  <img src="figures/tu1.png" alt="PF-OS dual-expert architecture" width="720">
</p>

The architecture is organized around a role-specialized operating point rather than generic symmetric multimodal fusion. The ocular-static expert uses a shared Transformer backbone to encode the eye sequence, then combines that representation with engineered eye and PPG/HRV summaries. A parallel eye-only expert keeps the same temporal backbone while removing the static branch. Their predicted probabilities are averaged with a fixed consensus weight in the reported PF-OS setting, making the final decision simple, interpretable, and easy to deploy in adaptive VR display pipelines.

Before training, PF-OS builds inputs that reflect the different reliability and time scales of the modalities. The model-facing data pipeline keeps the eye sequence as the main temporal stream and forms a compact static vector from ocular and cardiovascular descriptors. Raw derivative-based PPG sequences are preserved only for controlled ablations, while frequency-domain HRV features are used only when the available inter-beat interval segment is long enough to support them.

<p align="center">
  <img src="figures/tu2.png" alt="PF-OS preprocessing and feature construction pipeline" width="580">
</p>

## Environment

The code was organized for Python 3.8. Core dependencies are:

```text
torch>=1.10
numpy>=1.23
pandas>=1.5
scipy>=1.9
scikit-learn>=1.1
tqdm>=4.64
matplotlib>=3.5
```

Install dependencies with:

```bash
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA/CPU environment before running full experiments.

## Dataset and Preprocessing

The raw HP Omnicept Cognitive Load Dataset is not included in this repository. Place the dataset under `HPO-CLD/` using the following structure:

```text
HPO-CLD/
  HPO-CLD001/
    *tobii*.csv
    *bitalino*.csv
    *labels*.csv
  HPO-CLD002/
  ...
```

Timestamps are expected in Unix microseconds. The default windowing protocol is 10 s windows with a 5 s stride.

To run a dataset integrity check:

```bash
python PF-OS/check_data.py --root HPO-CLD --log runs/data_check.log --window_sec 10 --stride_sec 5
```

For static-feature baselines, preprocess the dataset once:

```bash
python comparison/prepare_data.py --root HPO-CLD --outdir prepared_data
```

## Running the Code

Run PF-OS subject-wise 5-fold cross-validation:

```bash
python PF-OS/run_cv.py --root HPO-CLD --k_folds 5 --cv_outdir runs/cv_pf_os --cv_profile consensus --amp
```

Run the fair comparison suite:

```bash
python comparison/run_fair_suite.py --root HPO-CLD --data_dir prepared_data --outdir comparison/results/fair_suite
```

Run controlled ablations:

```bash
python PF-OS/run_ablation.py --root HPO-CLD --k_folds 5 --ablation_outdir runs/ablation_pf_os --amp
```

## Repository Layout

```text
PF-OS/
  PF-OS/                         Main PF-OS model, training, CV, ablation, and analysis scripts
  baselines/                     Baseline model implementations and shared baseline utilities
    KNN/, LDA/, SVM/, ...        Classical machine-learning baselines
    CNN/, MLP/, LSTM/, TFN/      Deep-learning baselines
    baseline_common.py           Shared static-feature extraction and evaluation helpers
  ablations/                     Paper ablation wrappers
    PhysioFormer-S/              Raw-sequence multimodal variant
    Early Fusion Transformer/    Early-fusion variant
    Eye Only Transformer/        Eye-only variant
    No Cross Attention/          No-cross-attention variant
  comparison/                    Unified fair-comparison runners and result collectors
  figures/                       Selected paper figures used by this README
  signal_visualizations/         Signal visualization script and example figures
```
