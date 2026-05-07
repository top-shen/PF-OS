# -*- coding: utf-8 -*-
"""
data.py

High-performance, research-grade dataloading for HPO-CLD-style recordings.

This module implements a *windowed multimodal* dataset for cognitive workload modeling:
- Tobii eye-tracking stream (120 Hz): pupil + gaze direction + pupil position + validity
- BITalino PPG stream (1000 Hz): A1..A6 (auto-select best channel)

Key design goals
----------------
1) Timestamp-consistent slicing (Unix microseconds) using np.searchsorted
2) Fixed-length resampling (seq_len points per window) for sequence models
3) Medical-signal–aware preprocessing (bandpass filtering for PPG)
4) Robustness to missing/invalid samples (validity-aware interpolation)
5) Optional caching to NPZ for fast iteration

Expected folder layout
---------------------
root/
  HPO-CLD001/
    *tobii*.csv
    *bitalino*.csv
    *labels*.csv
  HPO-CLD002/
    ...

labels.csv must contain:
  time_start, time_end, task_difficulty
Optionally:
  TLX_score, reweighted_TLX_score

Author: updated by assistant (v2)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Iterable

import numpy as np
import pandas as pd
import warnings
import csv
import logging
import os
from datetime import datetime

from scipy.signal import butter, filtfilt, welch, find_peaks


# -----------------------------
# Basic helpers
# -----------------------------
def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return default
        return v
    except Exception:
        return default


def robust_zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = x.astype(np.float64, copy=False)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if not np.isfinite(med):
        med = 0.0
    if not np.isfinite(mad) or mad < eps:
        mad = 1.0
    return (x - med) / (1.4826 * mad + eps)


def map_task_to_class(task: str) -> int:
    """
    labels.csv task_difficulty examples: Low1, Med2, High3 ...
    -> map to 3-class: Low=0, Med=1, High=2
    """
    if task is None:
        return 0
    s = str(task).strip().lower()
    if s.startswith("low") or "low" in s:
        return 0
    if s.startswith("med") or s.startswith("mid") or s.startswith("medium") or "med" in s or "mid" in s:
        return 1
    if s.startswith("high") or "high" in s:
        return 2
    return 0


def find_best_file(pdir: Path, pattern: str) -> Path:
    """
    If multiple matches exist, pick the "best" by:
      1) largest file size
      2) if tie, lexicographically last
    """
    matches = sorted(pdir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"Missing file in {pdir} pattern={pattern}")
    matches = sorted(matches, key=lambda p: (p.stat().st_size, p.name))
    return matches[-1]



# -----------------------------
# Dataset integrity check (pre-flight) + logging
# -----------------------------
def _setup_data_check_logger(log_path: Path) -> logging.Logger:
    """Create a dedicated logger that writes to both console and file (overwrites file)."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("HPOCLD_DataCheck")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Reset handlers to avoid duplicate logs when called multiple times.
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(str(log_path), mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


def _try_read_csv_columns(path: Path) -> List[str]:
    """Fast header-only CSV read via pandas (nrows=0)."""
    try:
        df0 = pd.read_csv(path, nrows=0)
        return [str(c) for c in df0.columns.tolist()]
    except Exception:
        return []


def _first_last_csv_value(path: Path, col: str = "t", max_tail_bytes: int = 1 << 16) -> Tuple[Optional[float], Optional[float]]:
    """
    Return (first_value, last_value) for a CSV column with minimal IO.

    Notes
    -----
    - Works best for the common case where timestamps are stored in column "t".
    - Does not load full file into memory (important for 1000 Hz streams).
    """
    try:
        with open(path, "rb") as f:
            header_raw = f.readline()
            if not header_raw:
                return None, None
            header = header_raw.decode("utf-8-sig", errors="ignore").strip()
            cols = [c.strip() for c in header.split(",")]
            if col not in cols:
                return None, None
            idx = cols.index(col)

            # first non-empty data line
            first = None
            while True:
                line = f.readline()
                if not line:
                    break
                s = line.decode("utf-8-sig", errors="ignore").strip()
                if not s:
                    continue
                parts = s.split(",")
                if len(parts) <= idx:
                    continue
                try:
                    first = float(parts[idx])
                except Exception:
                    first = None
                break

            # last non-empty data line (tail scan)
            f.seek(0, os.SEEK_END)
            end = f.tell()
            if end <= 0:
                return first, None
            size = int(min(max_tail_bytes, end))
            f.seek(end - size)
            tail = f.read(size)
            # If we started in the middle of a line, drop the first partial line.
            if b"\n" in tail:
                tail = tail.split(b"\n", 1)[-1] if end > size else tail
            cands = tail.splitlines()
            last = None
            for cand in reversed(cands):
                if not cand.strip():
                    continue
                s = cand.decode("utf-8-sig", errors="ignore").strip()
                if not s or s == header:
                    continue
                parts = s.split(",")
                if len(parts) <= idx:
                    continue
                try:
                    last = float(parts[idx])
                except Exception:
                    last = None
                break

            return first, last
    except Exception:
        return None, None


def validate_dataset(
    root: Path,
    participant_dirs: Optional[List[Path]] = None,
    log_path: Optional[Path] = None,
    window_sec: float = 10.0,
    stride_sec: float = 5.0,
) -> Dict[str, Any]:
    """
    Pre-flight dataset check.

    This runs a fast integrity scan and writes a log + JSON summary:
      - missing files / empty files
      - missing required columns
      - labels sanity: time_start < time_end, non-empty
      - coarse overlap check between labels and streams using first/last timestamps
      - estimated number of windows per subject (given window_sec/stride_sec)

    The goal is to fail *early* with actionable diagnostics instead of crashing mid-training.
    """
    root = Path(root)
    if participant_dirs is None:
        participant_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    if log_path is None:
        log_path = Path("data_check.log")

    logger = _setup_data_check_logger(Path(log_path))
    logger.info(f"[data-check] start | root={root.resolve()} | n_subjects={len(participant_dirs)} | window={window_sec}s stride={stride_sec}s")

    win_us = int(window_sec * 1e6)
    stride_us = int(stride_sec * 1e6)

    details: List[Dict[str, Any]] = []
    ok_subjects: List[str] = []
    n_critical = 0
    n_warning = 0

    for pdir in participant_dirs:
        pid = pdir.name
        critical: List[str] = []
        warn: List[str] = []

        # Resolve files
        tobii = None
        bitalino = None
        labels = None
        try:
            tobii = find_best_file(pdir, "*tobii*.csv")
        except Exception as e:
            critical.append(f"missing tobii file (*tobii*.csv): {e}")
        try:
            bitalino = find_best_file(pdir, "*bitalino*.csv")
        except Exception as e:
            critical.append(f"missing bitalino file (*bitalino*.csv): {e}")
        try:
            labels = find_best_file(pdir, "*labels*.csv")
        except Exception as e:
            critical.append(f"missing labels file (*labels*.csv): {e}")

        # File size check
        for fp, tag in [(tobii, "tobii"), (bitalino, "bitalino"), (labels, "labels")]:
            if fp is None:
                continue
            try:
                if fp.stat().st_size <= 0:
                    critical.append(f"{tag} file is empty (0 bytes): {fp}")
            except Exception as e:
                warn.append(f"cannot stat {tag} file: {fp} ({e})")

        # Header/column checks
        if tobii is not None and tobii.exists():
            cols = _try_read_csv_columns(tobii)
            if "t" not in cols:
                critical.append(f"tobii missing required column 't': {tobii}")
            else:
                t_first, t_last = _first_last_csv_value(tobii, col="t")
                if t_first is None:
                    critical.append(f"tobii has no data rows (cannot read first timestamp): {tobii}")

        if bitalino is not None and bitalino.exists():
            cols = _try_read_csv_columns(bitalino)
            if "t" not in cols:
                critical.append(f"bitalino missing required column 't': {bitalino}")
            if not any(str(c).startswith("A") for c in cols):
                critical.append(f"bitalino missing any analog channel A*: {bitalino} (cols={cols[:20]})")
            if "t" in cols:
                t_first, t_last = _first_last_csv_value(bitalino, col="t")
                if t_first is None:
                    critical.append(f"bitalino has no data rows (cannot read first timestamp): {bitalino}")

        # Labels checks
        label_min, label_max, est_wins = None, None, 0
        if labels is not None and labels.exists():
            try:
                lab = pd.read_csv(labels)
                required = {"time_start", "time_end", "task_difficulty"}
                if not required.issubset(set(lab.columns)):
                    critical.append(f"labels missing required columns {sorted(list(required))} (got={lab.columns.tolist()})")
                elif len(lab) == 0:
                    critical.append("labels.csv has 0 rows")
                else:
                    # basic sanity
                    bad_rows = 0
                    for _, r in lab.iterrows():
                        try:
                            t0 = int(r["time_start"])
                            t1 = int(r["time_end"])
                        except Exception:
                            bad_rows += 1
                            continue
                        if t1 <= t0:
                            bad_rows += 1
                            continue
                        # estimate windows
                        dur = t1 - t0
                        if dur >= win_us:
                            est_wins += int(1 + max(0, (dur - win_us) // stride_us))
                    if bad_rows > 0:
                        warn.append(f"labels has {bad_rows} invalid rows (time_end<=time_start or non-int)")
                    try:
                        label_min = int(np.nanmin(lab["time_start"].to_numpy(dtype=np.float64)))
                        label_max = int(np.nanmax(lab["time_end"].to_numpy(dtype=np.float64)))
                    except Exception:
                        warn.append("cannot compute label time range (non-numeric time_start/time_end)")
            except Exception as e:
                critical.append(f"failed to read labels.csv: {e}")

        # Coarse overlap checks using first/last timestamp
        if label_min is not None and label_max is not None:
            if tobii is not None and tobii.exists():
                t_first, t_last = _first_last_csv_value(tobii, col="t")
                if t_first is None or t_last is None:
                    warn.append("tobii: cannot read first/last timestamp (t)")
                else:
                    if label_max <= t_first or label_min >= t_last:
                        warn.append(f"labels time range [{label_min},{label_max}] does not overlap tobii t range [{t_first:.0f},{t_last:.0f}]")
            if bitalino is not None and bitalino.exists():
                t_first, t_last = _first_last_csv_value(bitalino, col="t")
                if t_first is None or t_last is None:
                    warn.append("bitalino: cannot read first/last timestamp (t)")
                else:
                    if label_max <= t_first or label_min >= t_last:
                        warn.append(f"labels time range [{label_min},{label_max}] does not overlap bitalino t range [{t_first:.0f},{t_last:.0f}]")

        # Flag extremely small usable windows
        if est_wins == 0 and (labels is not None and labels.exists()):
            warn.append(f"estimated windows=0 for window={window_sec}s (check label durations vs window length)")

        ok = len(critical) == 0
        if ok:
            ok_subjects.append(pid)
        n_critical += int(len(critical) > 0)
        n_warning += int(len(warn) > 0)

        # Write per-subject line
        if ok:
            logger.info(f"[ok] {pid} | est_windows={est_wins} | warnings={len(warn)}")
        else:
            logger.error(f"[bad] {pid} | CRITICAL={len(critical)} warnings={len(warn)}")
        for msg in critical:
            logger.error(f"  - {msg}")
        for msg in warn:
            logger.warning(f"  - {msg}")

        details.append({
            "participant": pid,
            "ok": ok,
            "est_windows": int(est_wins),
            "critical": critical,
            "warnings": warn,
            "tobii": str(tobii) if tobii is not None else "",
            "bitalino": str(bitalino) if bitalino is not None else "",
            "labels": str(labels) if labels is not None else "",
        })

    summary: Dict[str, Any] = {
        "root": str(root.resolve()),
        "time": datetime.now().isoformat(timespec="seconds"),
        "n_subjects": int(len(participant_dirs)),
        "n_ok": int(len(ok_subjects)),
        "n_bad": int(len(participant_dirs) - len(ok_subjects)),
        "n_subjects_with_critical": int(n_critical),
        "n_subjects_with_warning": int(n_warning),
        "ok_subjects": ok_subjects,
        "details": details,
    }

    # Also write JSON sidecar for programmatic filtering
    try:
        json_path = Path(log_path).with_suffix(".json")
        with open(json_path, "w", encoding="utf-8") as f:
            import json as _json
            _json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info(f"[data-check] wrote JSON summary: {json_path.resolve()}")
    except Exception as e:
        logger.warning(f"[data-check] failed to write JSON summary: {e}")

    logger.info(f"[data-check] done | ok={len(ok_subjects)}/{len(participant_dirs)} | log={Path(log_path).resolve()}")
    return summary

def _linear_fill_nan(x: np.ndarray) -> np.ndarray:
    """
    Linear interpolate NaNs. If too few finite samples, return zeros.
    """
    x = x.astype(np.float64, copy=True)
    n = x.size
    if n == 0:
        return x
    idx = np.arange(n)
    mask = np.isfinite(x)
    if mask.sum() < 2:
        # if 0 or 1 sample, fall back to zeros
        return np.zeros_like(x)
    x[~mask] = np.interp(idx[~mask], idx[mask], x[mask])
    return x


def resample_by_time(
    t_us: np.ndarray,
    x: np.ndarray,
    t0_us: int,
    t1_us: int,
    seq_len: int,
) -> np.ndarray:
    """
    Resample signal x(t) on [t0_us, t1_us] to fixed seq_len points.

    Parameters
    ----------
    t_us : (N,) int/float array, strictly increasing
    x    : (N,) or (N,C)
    """
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")

    if t_us.size < 2:
        # Not enough points: return zeros
        if x.ndim == 1:
            return np.zeros((seq_len,), dtype=np.float32)
        return np.zeros((seq_len, x.shape[1]), dtype=np.float32)

    dur = float(t1_us - t0_us)
    if dur <= 0:
        if x.ndim == 1:
            return np.zeros((seq_len,), dtype=np.float32)
        return np.zeros((seq_len, x.shape[1]), dtype=np.float32)

    # Normalize to [0,1]
    tt = (t_us.astype(np.float64) - float(t0_us)) / dur
    grid = np.linspace(0.0, 1.0, seq_len, endpoint=False, dtype=np.float64)

    if x.ndim == 1:
        xs = _linear_fill_nan(x)
        y = np.interp(grid, tt, xs).astype(np.float32)
        return y

    # multi-channel
    out = np.zeros((seq_len, x.shape[1]), dtype=np.float32)
    for c in range(x.shape[1]):
        xs = _linear_fill_nan(x[:, c])
        out[:, c] = np.interp(grid, tt, xs).astype(np.float32)
    return out


# -----------------------------
# PPG processing and features
# -----------------------------
def bandpass_ppg(x: np.ndarray, fs: float, lo_hz: float = 0.7, hi_hz: float = 4.0, order: int = 3) -> np.ndarray:
    """
    Butterworth bandpass for PPG (approx 42–240 bpm).
    """
    x = x.astype(np.float64, copy=False)
    if x.size < max(10, order * 6):
        return x.astype(np.float32)
    nyq = 0.5 * fs
    lo = max(1e-6, lo_hz / nyq)
    hi = min(0.999, hi_hz / nyq)
    if lo >= hi:
        return x.astype(np.float32)
    b, a = butter(order, [lo, hi], btype="band")
    y = filtfilt(b, a, x, method="pad")
    return y.astype(np.float32)


def ppg_artifact_mask(x: np.ndarray, threshold_mad: float = 5.0) -> np.ndarray:
    """
    Detect motion artifacts in PPG signal using MAD-based amplitude thresholding.

    Returns a boolean mask where True = clean sample, False = artifact.
    Artifacts are samples whose absolute deviation from median exceeds
    threshold_mad * MAD (median absolute deviation).
    """
    if x.size < 10:
        return np.ones(x.size, dtype=bool)
    xf = x.astype(np.float64)
    med = np.nanmedian(xf)
    mad = np.nanmedian(np.abs(xf - med))
    if not np.isfinite(mad) or mad < 1e-12:
        return np.ones(x.size, dtype=bool)
    deviation = np.abs(xf - med) / (1.4826 * mad)
    clean = deviation < threshold_mad
    return clean


def ppg_peaks_from_signal(x: np.ndarray, fs: float) -> np.ndarray:
    """
    Robust peak detection using scipy.signal.find_peaks on a filtered, z-scored waveform.
    """
    if x.size < 50:
        return np.array([], dtype=np.int64)

    z = robust_zscore(x)

    # If the window is entirely NaN (e.g., sensor dropout), no peaks can be detected.
    if z.size == 0 or np.all(np.isnan(z)):
        return np.array([], dtype=np.int64)

    # Replace NaN/Inf to stabilize downstream peak detection.
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

    # dynamic height threshold and minimal distance
    height = float(np.median(z) + 0.3 * np.std(z))
    if not np.isfinite(height):
        height = 0.0
    min_dist = int(max(1, fs * 0.3))  # at most ~200 bpm
    peaks, _ = find_peaks(z, distance=min_dist, height=height)
    return peaks.astype(np.int64)


def ppg_static_features(x: np.ndarray, fs: float) -> np.ndarray:
    """
    Compute compact HR/HRV + spectral + quality metrics from a PPG window.

    Output dimension: 14
      0 mean_hr
      1 std_hr
      2 rmssd
      3 sdnn
      4 pnn50
      5 mean_rr
      6 std_rr
      7 sd1 (Poincaré)
      8 sd2 (Poincaré)
      9 lf_power (0.04–0.15 Hz) from IBI series
      10 hf_power (0.15–0.4 Hz) from IBI series
      11 lf_hf_ratio
      12 skew_rr
      13 quality (0–1)
    """
    if x.size < 100:
        return np.zeros((14,), dtype=np.float32)

    # Peak detect (ppg_peaks_from_signal already applies robust_zscore internally)
    peaks = ppg_peaks_from_signal(x, fs=fs)

    dur = x.size / float(fs)
    if dur <= 0.1:
        return np.zeros((14,), dtype=np.float32)

    # crude expected peak count range for quality
    exp_min = dur * 40.0 / 60.0
    exp_max = dur * 200.0 / 60.0
    n = float(peaks.size)
    quality = float(np.clip((n - exp_min) / (exp_max - exp_min + 1e-6), 0.0, 1.0)) if exp_max > 1e-6 else 0.0

    if peaks.size < 3:
        out = np.zeros((14,), dtype=np.float32)
        out[-1] = quality
        return out

    rr = np.diff(peaks) / float(fs)  # seconds
    rr = rr[(rr > 0.25) & (rr < 2.0)]
    if rr.size < 3:
        out = np.zeros((14,), dtype=np.float32)
        out[-1] = quality
        return out

    hr = 60.0 / rr
    mean_hr = float(np.mean(hr))
    std_hr = float(np.std(hr))

    diff_rr = np.diff(rr)
    rmssd = float(np.sqrt(np.mean(diff_rr ** 2))) if diff_rr.size else 0.0
    sdnn = float(np.std(rr))
    pnn50 = float(np.mean(np.abs(diff_rr) > 0.05) * 100.0) if diff_rr.size else 0.0

    mean_rr = float(np.mean(rr))
    std_rr = float(np.std(rr))

    # Poincaré SD1/SD2
    sd1 = float(np.sqrt(0.5) * np.std(diff_rr)) if diff_rr.size else 0.0
    if diff_rr.size:
        # Numerical guard: expression inside sqrt can become slightly negative due to short/noisy RR series
        val = 2.0 * (sdnn ** 2) - 0.5 * (np.std(diff_rr) ** 2)
        val = max(val, 0.0)
        sd2 = float(np.sqrt(val))
    else:
        sd2 = 0.0

    # RR skewness
    rr_c = rr - mean_rr
    m3 = float(np.mean(rr_c ** 3))
    m2 = float(np.mean(rr_c ** 2))
    skew_rr = float(m3 / (m2 ** 1.5 + 1e-12)) if m2 > 1e-12 else 0.0

    # IBI spectral features (Welch on evenly sampled IBI)
    # NOTE: ESC/NASPE Task Force (1996) recommends >=2 min for LF/HF analysis.
    # LF band (0.04-0.15 Hz) requires >=25s for one full cycle.
    # We require >=30s of usable IBI data; shorter windows get zeros.
    lf = 0.0
    hf = 0.0
    lfhf = 0.0
    _MIN_DURATION_LF_HF = 30.0  # seconds
    try:
        t_peaks = peaks[1:1 + rr.size] / float(fs)
        ibi_duration = float(t_peaks[-1] - t_peaks[0]) if t_peaks.size >= 2 else 0.0
        if ibi_duration >= _MIN_DURATION_LF_HF:
            # Convert to "tachogram": interpolate RR at 4 Hz (common HRV practice)
            t_grid = np.arange(t_peaks[0], t_peaks[-1], 0.25)  # 4 Hz
            rr_i = np.interp(t_grid, t_peaks, rr)

            f, pxx = welch(rr_i - np.mean(rr_i), fs=4.0, nperseg=min(256, rr_i.size))
            lf_band = (f >= 0.04) & (f < 0.15)
            hf_band = (f >= 0.15) & (f < 0.40)
            lf = float(np.trapz(pxx[lf_band], f[lf_band])) if np.any(lf_band) else 0.0
            hf = float(np.trapz(pxx[hf_band], f[hf_band])) if np.any(hf_band) else 0.0
            lfhf = float(lf / (hf + 1e-12))
    except Exception:
        lf, hf, lfhf = 0.0, 0.0, 0.0

    out = np.array(
        [
            safe_float(mean_hr),
            safe_float(std_hr),
            safe_float(rmssd),
            safe_float(sdnn),
            safe_float(pnn50),
            safe_float(mean_rr),
            safe_float(std_rr),
            safe_float(sd1),
            safe_float(sd2),
            safe_float(lf),
            safe_float(hf),
            safe_float(lfhf),
            safe_float(skew_rr),
            safe_float(quality),
        ],
        dtype=np.float32,
    )
    return out


# -----------------------------
# Eye features
# -----------------------------
def eye_static_features(
    pupil: np.ndarray,
    gaze_dir: np.ndarray,
    pupil_pos: np.ndarray,
    valid: np.ndarray,
    fs: float,
) -> np.ndarray:
    """
    Static features from eye stream for a window.
    Output dimension: 16

      0 mean_pupil
      1 std_pupil
      2 pupil_trend (linear slope)
      3 blink_rate (Hz)
      4 gaze_speed_mean
      5 gaze_speed_std
      6 gaze_acc_mean
      7 gaze_acc_std
      8 gaze_dispersion
      9 pos_x_mean
      10 pos_x_std
      11 pos_y_mean
      12 pos_y_std
      13 valid_ratio
      14 pupil_cv
      15 gaze_dir_z_mean
    """
    n = pupil.size
    if n < 5:
        return np.zeros((16,), dtype=np.float32)

    t = np.arange(n, dtype=np.float64) / float(max(fs, 1e-6))

    # Apply validity mask: invalid -> NaN
    pupil_m = pupil.astype(np.float64).copy()
    gaze_m = gaze_dir.astype(np.float64).copy()
    pos_m = pupil_pos.astype(np.float64).copy()

    if valid is not None and valid.size == n:
        bad = ~valid.astype(bool)
        pupil_m[bad] = np.nan
        gaze_m[bad, :] = np.nan
        pos_m[bad, :] = np.nan

    valid_ratio = float(np.mean(np.isfinite(pupil_m))) if n else 0.0

    mean_pupil = float(np.nanmean(pupil_m)) if np.any(np.isfinite(pupil_m)) else 0.0
    std_pupil = float(np.nanstd(pupil_m)) if np.any(np.isfinite(pupil_m)) else 0.0
    pupil_cv = float(std_pupil / (mean_pupil + 1e-6)) if np.isfinite(mean_pupil) else 0.0

    # Trend (slope)
    slope = 0.0
    mask = np.isfinite(pupil_m)
    if mask.sum() >= 3:
        tt = t[mask]
        yy = pupil_m[mask]
        tt = tt - np.mean(tt)
        denom = np.sum(tt ** 2)
        slope = float(np.sum(tt * (yy - np.mean(yy))) / (denom + 1e-12))

    # Blink rate: count validity gaps with duration consistent with real blinks.
    # Typical blink duration: 100-400ms. Gaps outside this range are likely
    # tracking losses (head movement, looking away) rather than blinks.
    blink_rate = 0.0
    if valid is not None and valid.size == n and n > 2:
        v = valid.astype(bool)
        dur_total = float(t[-1] - t[0]) if n > 1 else 0.0
        # Find invalid (gap) segments
        falling = np.where((v[:-1] == True) & (v[1:] == False))[0]
        rising = np.where((v[:-1] == False) & (v[1:] == True))[0]
        blink_count = 0
        for f_idx in falling:
            # Find the next rising edge after this falling edge
            candidates = rising[rising > f_idx]
            if candidates.size > 0:
                gap_samples = int(candidates[0]) - f_idx
                gap_sec = gap_samples / float(max(fs, 1e-6))
                if 0.05 <= gap_sec <= 0.5:  # 50-500ms (generous blink range)
                    blink_count += 1
            # else: gap extends to end of window — not a blink
        blink_rate = float(blink_count / dur_total) if dur_total > 1e-6 else 0.0

    # Gaze kinematics — compute only on valid (non-interpolated) samples
    # to avoid step/impulse artifacts from linear interpolation.
    gaze_speed_mean = 0.0
    gaze_speed_std = 0.0
    gaze_acc_mean = 0.0
    gaze_acc_std = 0.0
    gaze_disp = 0.0

    gaze_valid_mask = np.all(np.isfinite(gaze_m), axis=1)  # per-sample validity
    if np.sum(gaze_valid_mask) >= 3:
        # Find contiguous valid runs and compute kinematics within each run
        all_speeds: list = []
        all_accs: list = []
        # Identify contiguous valid segments
        changes = np.diff(gaze_valid_mask.astype(np.int8))
        starts = np.where(changes == 1)[0] + 1
        ends = np.where(changes == -1)[0] + 1
        # Handle edge cases
        if gaze_valid_mask[0]:
            starts = np.concatenate([[0], starts])
        if gaze_valid_mask[-1]:
            ends = np.concatenate([ends, [len(gaze_valid_mask)]])
        for s_idx, e_idx in zip(starts, ends):
            seg = gaze_m[s_idx:e_idx]
            if seg.shape[0] >= 2:
                d_seg = np.diff(seg, axis=0)
                spd = np.linalg.norm(d_seg, axis=1) * float(fs)
                all_speeds.append(spd)
                if spd.size >= 2:
                    all_accs.append(np.diff(spd) * float(fs))
        if all_speeds:
            speeds_cat = np.concatenate(all_speeds)
            gaze_speed_mean = float(np.mean(speeds_cat))
            gaze_speed_std = float(np.std(speeds_cat))
        if all_accs:
            accs_cat = np.concatenate(all_accs)
            gaze_acc_mean = float(np.mean(accs_cat))
            gaze_acc_std = float(np.std(accs_cat))
        # Dispersion on valid samples only
        valid_gaze = gaze_m[gaze_valid_mask]
        gaze_disp = float(np.std(np.linalg.norm(valid_gaze, axis=1)))

    # Interpolated gaze for position stats and z-mean (these are less sensitive)
    gaze_f = gaze_m.copy()
    for k in range(3):
        gaze_f[:, k] = _linear_fill_nan(gaze_f[:, k])

    # Pupil position stats
    pos_f = pos_m.copy()
    for k in range(2):
        pos_f[:, k] = _linear_fill_nan(pos_f[:, k])
    pos_x_mean = float(np.mean(pos_f[:, 0])) if pos_f.size else 0.0
    pos_x_std = float(np.std(pos_f[:, 0])) if pos_f.size else 0.0
    pos_y_mean = float(np.mean(pos_f[:, 1])) if pos_f.size else 0.0
    pos_y_std = float(np.std(pos_f[:, 1])) if pos_f.size else 0.0

    gaze_dir_z_mean = float(np.mean(gaze_f[:, 2])) if gaze_f.size else 0.0

    out = np.array(
        [
            safe_float(mean_pupil),
            safe_float(std_pupil),
            safe_float(slope),
            safe_float(blink_rate),
            safe_float(gaze_speed_mean),
            safe_float(gaze_speed_std),
            safe_float(gaze_acc_mean),
            safe_float(gaze_acc_std),
            safe_float(gaze_disp),
            safe_float(pos_x_mean),
            safe_float(pos_x_std),
            safe_float(pos_y_mean),
            safe_float(pos_y_std),
            safe_float(valid_ratio),
            safe_float(pupil_cv),
            safe_float(gaze_dir_z_mean),
        ],
        dtype=np.float32,
    )
    return out


# -----------------------------
# Stream loader
# -----------------------------
@dataclass
class StreamConfig:
    seq_len: int = 256
    ppg_seq_len: int = 0  # 0 = same as seq_len; >0 = independent PPG sequence length
    # PPG bandpass
    ppg_lo_hz: float = 0.7
    ppg_hi_hz: float = 4.0
    ppg_order: int = 3


class ParticipantStreams:
    """
    Load tobii/bitalino once per participant and slice by timestamps using searchsorted.
    We keep arrays in memory (fast slicing); for very large datasets you may adapt to memmap/parquet.
    """

    def __init__(self, tobii_path: Path, bitalino_path: Path):
        # Read only needed Tobii columns
        tobii_cols = [
            "t",
            "right_pupil_diameter", "left_pupil_diameter",
            "right_pupil_validity", "left_pupil_validity",
            "right_gaze_direction_validity", "left_gaze_direction_validity",
            "right_gaze_direction_unit_vector_x", "right_gaze_direction_unit_vector_y", "right_gaze_direction_unit_vector_z",
            "left_gaze_direction_unit_vector_x", "left_gaze_direction_unit_vector_y", "left_gaze_direction_unit_vector_z",
            "right_pupil_position_in_tracking_area_x", "right_pupil_position_in_tracking_area_y",
            "left_pupil_position_in_tracking_area_x", "left_pupil_position_in_tracking_area_y",
        ]
        df_t = pd.read_csv(tobii_path, usecols=lambda c: c in set(tobii_cols))
        if "t" not in df_t.columns:
            raise RuntimeError(f"Tobii file missing column 't': {tobii_path}")

        df_t = df_t.sort_values("t").reset_index(drop=True)

        self.t_tobii = df_t["t"].to_numpy(dtype=np.float64)

        # validity
        rv = df_t.get("right_pupil_validity", pd.Series(np.ones(len(df_t)))).to_numpy(dtype=np.float64)
        lv = df_t.get("left_pupil_validity", pd.Series(np.ones(len(df_t)))).to_numpy(dtype=np.float64)
        rgv = df_t.get("right_gaze_direction_validity", pd.Series(np.ones(len(df_t)))).to_numpy(dtype=np.float64)
        lgv = df_t.get("left_gaze_direction_validity", pd.Series(np.ones(len(df_t)))).to_numpy(dtype=np.float64)
        self.valid_eye = (rv > 0.5) & (lv > 0.5) & (rgv > 0.5) & (lgv > 0.5)

        # pupil
        rp = df_t.get("right_pupil_diameter", pd.Series(np.nan, index=df_t.index)).to_numpy(dtype=np.float32)
        lp = df_t.get("left_pupil_diameter", pd.Series(np.nan, index=df_t.index)).to_numpy(dtype=np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            self.pupil = np.nanmean(np.vstack([rp, lp]), axis=0).astype(np.float32)

        # gaze direction (mean of left/right)
        rg = np.vstack([
            df_t.get("right_gaze_direction_unit_vector_x", pd.Series(np.nan, index=df_t.index)).to_numpy(dtype=np.float32),
            df_t.get("right_gaze_direction_unit_vector_y", pd.Series(np.nan, index=df_t.index)).to_numpy(dtype=np.float32),
            df_t.get("right_gaze_direction_unit_vector_z", pd.Series(np.nan, index=df_t.index)).to_numpy(dtype=np.float32),
        ]).T
        lg = np.vstack([
            df_t.get("left_gaze_direction_unit_vector_x", pd.Series(np.nan, index=df_t.index)).to_numpy(dtype=np.float32),
            df_t.get("left_gaze_direction_unit_vector_y", pd.Series(np.nan, index=df_t.index)).to_numpy(dtype=np.float32),
            df_t.get("left_gaze_direction_unit_vector_z", pd.Series(np.nan, index=df_t.index)).to_numpy(dtype=np.float32),
        ]).T
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            self.gaze_dir = np.nanmean(np.stack([rg, lg], axis=0), axis=0).astype(np.float32)  # (N,3)

        # pupil position (mean of left/right) in tracking area
        rpos = np.vstack([
            df_t.get("right_pupil_position_in_tracking_area_x", pd.Series(np.nan, index=df_t.index)).to_numpy(dtype=np.float32),
            df_t.get("right_pupil_position_in_tracking_area_y", pd.Series(np.nan, index=df_t.index)).to_numpy(dtype=np.float32),
        ]).T
        lpos = np.vstack([
            df_t.get("left_pupil_position_in_tracking_area_x", pd.Series(np.nan, index=df_t.index)).to_numpy(dtype=np.float32),
            df_t.get("left_pupil_position_in_tracking_area_y", pd.Series(np.nan, index=df_t.index)).to_numpy(dtype=np.float32),
        ]).T
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            self.pupil_pos = np.nanmean(np.stack([rpos, lpos], axis=0), axis=0).astype(np.float32)  # (N,2)

        # Estimate Tobii fs from median dt
        dt = np.diff(self.t_tobii)
        self.fs_eye = float(1e6 / np.median(dt)) if dt.size > 0 and np.median(dt) > 0 else 120.0

        # BITalino
        df_b = pd.read_csv(bitalino_path)
        if "t" not in df_b.columns:
            raise RuntimeError(f"BITalino file missing column 't': {bitalino_path}")
        df_b = df_b.sort_values("t").reset_index(drop=True)
        self.t_bitalino = df_b["t"].to_numpy(dtype=np.float64)

        # Pick best analog channel by signal quality index (SQI).
        # Good PPG signals have: (1) negative skewness (sharp systolic peaks),
        # (2) dominant periodicity in physiological HR range.
        # Variance alone can select noisy/artifact-laden channels.
        chans = [c for c in df_b.columns if c.startswith("A")]
        if not chans:
            raise RuntimeError(f"No analog channels A* found in {bitalino_path}")

        best = None
        best_sqi = -np.inf
        for c in chans:
            x = df_b[c].to_numpy(dtype=np.float64)
            if x.size == 0:
                continue
            x_abs = np.abs(x)
            if np.all(np.isnan(x_abs)):
                continue
            if np.nanmax(x_abs) < 1e-9:
                continue
            # SQI: combine skewness and periodicity
            xc = x[np.isfinite(x)]
            if xc.size < 100:
                continue
            xz = (xc - np.mean(xc)) / (np.std(xc) + 1e-12)
            # Skewness: good PPG tends to have |skewness| > 0 (peaked waveform)
            m3 = float(np.mean(xz ** 3))
            skew_score = abs(m3)
            # Periodicity: autocorrelation at expected HR lag (0.3-1.5s = 40-200 bpm)
            fs_est = float(1e6 / np.median(np.diff(self.t_bitalino))) if self.t_bitalino.size > 1 else 1000.0
            lag_lo = int(max(1, 0.3 * fs_est))
            lag_hi = int(min(xc.size // 2, 1.5 * fs_est))
            period_score = 0.0
            if lag_hi > lag_lo and xc.size > lag_hi + 1:
                acf = np.correlate(xz[:min(len(xz), lag_hi * 3)], xz[:min(len(xz), lag_hi * 3)], mode='full')
                acf = acf[len(acf) // 2:]  # positive lags only
                acf = acf / (acf[0] + 1e-12)  # normalize
                if lag_hi < len(acf):
                    period_score = float(np.max(acf[lag_lo:lag_hi]))
            sqi = skew_score + 2.0 * max(period_score, 0.0)  # weight periodicity higher
            if sqi > best_sqi:
                best_sqi = sqi
                best = c
        if best is None:
            best = chans[0]
        self.ppg_chan = best
        self.ppg_raw = df_b[best].to_numpy(dtype=np.float32)

        # BITalino fs from median dt
        dtb = np.diff(self.t_bitalino)
        self.fs_ppg = float(1e6 / np.median(dtb)) if dtb.size > 0 and np.median(dtb) > 0 else 1000.0

    def slice_eye(self, t0_us: int, t1_us: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        i0 = int(np.searchsorted(self.t_tobii, t0_us, side="left"))
        i1 = int(np.searchsorted(self.t_tobii, t1_us, side="right"))
        return (
            self.t_tobii[i0:i1],
            self.pupil[i0:i1],
            self.gaze_dir[i0:i1],
            self.pupil_pos[i0:i1],
            self.valid_eye[i0:i1],
        )

    def slice_ppg(self, t0_us: int, t1_us: int) -> Tuple[np.ndarray, np.ndarray]:
        i0 = int(np.searchsorted(self.t_bitalino, t0_us, side="left"))
        i1 = int(np.searchsorted(self.t_bitalino, t1_us, side="right"))
        return self.t_bitalino[i0:i1], self.ppg_raw[i0:i1]


# -----------------------------
# Dataset
# -----------------------------
@dataclass
class WindowSampleMeta:
    participant: str
    t_start_us: int
    t_end_us: int
    y: int
    tlx: float
    difficulty: str


class MultiModalWindowDataset:
    """
    Returns per-window tensors:
      eye_seq:  (seq_len, C_eye=7)  float32
      ppg_seq:  (seq_len, C_ppg=2)  float32
      x_static: (D_static=16+16=32) float32
      y:        int64
      y_reg:    float32 (reweighted TLX, [0,1] if available)

    eye_seq channels:
      [pupil, gaze_x, gaze_y, gaze_z, pos_x, pos_y, valid_mask]

    ppg_seq channels:
      [ppg_filtered, ppg_derivative]

    Notes
    -----
    - Windowing is done within each labeled interval [time_start, time_end)
    - If seq_len is small, this behaves like a learnable "compressed representation" of raw physiology
    """
    def __init__(
        self,
        participant_dirs: List[Path],
        window_sec: float = 10.0,
        stride_sec: float = 5.0,
        seq_len: int = 256,
        stream_cfg: Optional[StreamConfig] = None,
        cache_dir: Optional[Path] = None,
        augment: bool = False,
        seed: int = 42,
        verbose: bool = False,
    ):
        self.participant_dirs = participant_dirs
        self.window_us = int(window_sec * 1e6)
        self.stride_us = int(stride_sec * 1e6)
        self.seq_len = int(seq_len)
        self.stream_cfg = stream_cfg or StreamConfig(seq_len=seq_len)
        self.cache_dir = cache_dir
        self.augment = augment
        self.verbose = verbose
        self.rng = np.random.RandomState(seed)

        self.meta_files: List[Dict[str, Path]] = []
        self.samples: List[Tuple[int, int, int, int, float, str]] = []  # (pidx, s, e, y, tlx, difficulty)
        self.sample_meta: List[WindowSampleMeta] = []
        self._stream_cache: Dict[int, ParticipantStreams] = {}

        # Resolve files
        for pdir in participant_dirs:
            tobii = find_best_file(pdir, "*tobii*.csv")
            bitalino = find_best_file(pdir, "*bitalino*.csv")
            labels = find_best_file(pdir, "*labels*.csv")
            self.meta_files.append({"dir": pdir, "tobii": tobii, "bitalino": bitalino, "labels": labels})

        # Build windows
        for pidx, m in enumerate(self.meta_files):
            lab = pd.read_csv(m["labels"])
            required = {"time_start", "time_end", "task_difficulty"}
            if not required.issubset(set(lab.columns)):
                raise RuntimeError(f"labels.csv missing required columns {required}. got={lab.columns.tolist()} file={m['labels']}")
            lab = lab.sort_values("time_start").reset_index(drop=True)

            for _, r in lab.iterrows():
                t0 = int(r["time_start"])
                t1 = int(r["time_end"])
                diff = str(r.get("task_difficulty", "")).strip()
                y = map_task_to_class(diff)

                # Prefer reweighted TLX in [0,1] if available
                if "reweighted_TLX_score" in lab.columns and np.isfinite(r.get("reweighted_TLX_score", np.nan)):
                    tlx = float(r.get("reweighted_TLX_score"))
                elif "TLX_score" in lab.columns and np.isfinite(r.get("TLX_score", np.nan)):
                    # Auto-detect TLX range: NASA-TLX subscales are 0-20, overall score is 0-100.
                    raw_tlx = float(r.get("TLX_score"))
                    tlx_max = lab["TLX_score"].max() if "TLX_score" in lab.columns else 20.0
                    divisor = 100.0 if tlx_max > 21.0 else 20.0
                    tlx = float(np.clip(raw_tlx / divisor, 0.0, 1.0))
                else:
                    tlx = float("nan")

                start = t0
                while start + self.window_us <= t1:
                    self.samples.append((pidx, start, start + self.window_us, y, tlx, diff))
                    self.sample_meta.append(WindowSampleMeta(
                        participant=m["dir"].name, t_start_us=int(start), t_end_us=int(start + self.window_us),
                        y=int(y), tlx=float(tlx) if np.isfinite(tlx) else float("nan"), difficulty=diff
                    ))
                    start += self.stride_us

        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_streams(self, pidx: int) -> ParticipantStreams:
        if pidx not in self._stream_cache:
            m = self.meta_files[pidx]
            if self.verbose:
                print(f"[streams] loading {m['dir'].name}")
            self._stream_cache[pidx] = ParticipantStreams(m["tobii"], m["bitalino"])
        return self._stream_cache[pidx]

    def _cache_file(self, pidx: int) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        pid = self.meta_files[pidx]["dir"].name
        ppg_L = self.stream_cfg.ppg_seq_len if self.stream_cfg.ppg_seq_len > 0 else self.seq_len
        return self.cache_dir / f"{pid}_mmseq_w{self.window_us}_s{self.stride_us}_L{self.seq_len}_P{ppg_L}.npz"

    def _maybe_build_cache(self, pidx: int) -> Optional[Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
        """
        Cache per-window:
          eye_seq (L,7), ppg_seq (L,2), x_static (32,)
        """
        cf = self._cache_file(pidx)
        if cf is None:
            return None
        if cf.exists():
            data = np.load(cf, allow_pickle=True)
            keys = data["keys"]
            eye = data["eye"]
            ppg = data["ppg"]
            stat = data["stat"]
            out: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
            for k, a, b, c in zip(keys, eye, ppg, stat):
                s, e = int(k[0]), int(k[1])
                out[(s, e)] = (a.astype(np.float32), b.astype(np.float32), c.astype(np.float32))
            return out

        if self.verbose:
            print(f"[cache] building multimodal cache for {self.meta_files[pidx]['dir'].name} -> {cf}")

        stream = self._load_streams(pidx)
        wins = [(s, e) for (pi, s, e, _, _, _) in self.samples if pi == pidx]

        keys_list, eye_list, ppg_list, stat_list = [], [], [], []
        for (s, e) in wins:
            eye_seq, ppg_seq, x_stat = self._build_one(stream, s, e)
            keys_list.append((s, e))
            eye_list.append(eye_seq)
            ppg_list.append(ppg_seq)
            stat_list.append(x_stat)

        keys_arr = np.array(keys_list, dtype=np.int64)
        eye_arr = np.stack(eye_list, axis=0).astype(np.float32) if eye_list else np.zeros((0, self.seq_len, 7), np.float32)
        _ppg_L = self.stream_cfg.ppg_seq_len if self.stream_cfg.ppg_seq_len > 0 else self.seq_len
        ppg_arr = np.stack(ppg_list, axis=0).astype(np.float32) if ppg_list else np.zeros((0, _ppg_L, 2), np.float32)
        _n_static = len(get_static_feature_names())
        stat_arr = np.stack(stat_list, axis=0).astype(np.float32) if stat_list else np.zeros((0, _n_static), np.float32)

        np.savez_compressed(cf, keys=keys_arr, eye=eye_arr, ppg=ppg_arr, stat=stat_arr)

        out = {(int(k[0]), int(k[1])): (a, b, c) for k, a, b, c in zip(keys_arr, eye_arr, ppg_arr, stat_arr)}
        return out

    def _build_one(self, stream: ParticipantStreams, s_us: int, e_us: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Eye slices
        t_e, pupil, gaze, pos, valid = stream.slice_eye(s_us, e_us)
        # PPG slice
        t_p, ppg = stream.slice_ppg(s_us, e_us)

        # Build eye sequence (L,7)
        # Apply validity as a channel; we do not hard-drop samples to keep temporal continuity
        pupil_v = pupil.astype(np.float32).copy()
        gaze_v = gaze.astype(np.float32).copy()
        pos_v = pos.astype(np.float32).copy()
        valid_f = valid.astype(np.float32).copy()

        # Set invalid points to NaN for interpolation
        bad = ~valid.astype(bool)
        if bad.any():
            pupil_v[bad] = np.nan
            gaze_v[bad, :] = np.nan
            pos_v[bad, :] = np.nan

        pupil_r = resample_by_time(t_e, pupil_v, s_us, e_us, self.seq_len)  # (L,)
        gaze_r = resample_by_time(t_e, gaze_v, s_us, e_us, self.seq_len)    # (L,3)
        pos_r = resample_by_time(t_e, pos_v, s_us, e_us, self.seq_len)      # (L,2)
        val_r = resample_by_time(t_e, valid_f, s_us, e_us, self.seq_len)    # (L,)

        eye_seq = np.concatenate(
            [pupil_r[:, None], gaze_r, pos_r, val_r[:, None]],
            axis=1,
        ).astype(np.float32)  # (L,7)

        # Estimate fs for the slice (fallback to stream estimate)
        fs_eye = stream.fs_eye

        # Static eye features from *raw* slice (higher fidelity than resampled)
        eye_stat = eye_static_features(pupil=pupil, gaze_dir=gaze, pupil_pos=pos, valid=valid, fs=fs_eye)

        # Build PPG sequence
        fs_ppg = stream.fs_ppg
        # Detect and interpolate motion artifacts before bandpass filtering
        ppg_clean = ppg.astype(np.float32).copy()
        art_mask = ppg_artifact_mask(ppg_clean, threshold_mad=5.0)
        if not np.all(art_mask) and np.sum(art_mask) >= 2:
            # Replace artifact samples with linear interpolation from clean neighbors
            ppg_clean[~art_mask] = np.nan
            ppg_clean = _linear_fill_nan(ppg_clean)
        ppg_f = bandpass_ppg(ppg_clean, fs=fs_ppg,
                            lo_hz=self.stream_cfg.ppg_lo_hz, hi_hz=self.stream_cfg.ppg_hi_hz, order=self.stream_cfg.ppg_order)
        # Resample filtered PPG to grid (use ppg_seq_len if set, else seq_len)
        ppg_L = self.stream_cfg.ppg_seq_len if self.stream_cfg.ppg_seq_len > 0 else self.seq_len
        ppg_r = resample_by_time(t_p, ppg_f, s_us, e_us, ppg_L)  # (L_ppg,)
        ppg_r = ppg_r.astype(np.float32)

        # Derivative channel
        dp = np.diff(ppg_r, prepend=ppg_r[0]).astype(np.float32)

        # Per-window robust normalization
        # NOTE: per-window z-scoring removes absolute PPG amplitude information.
        # We preserve it by adding pre-normalization stats to static features below.
        ppg_win_mean = float(np.nanmedian(ppg_r))
        ppg_win_mad = float(np.nanmedian(np.abs(ppg_r - ppg_win_mean)))
        ppg_rz = robust_zscore(ppg_r).astype(np.float32)
        dp_rz = robust_zscore(dp).astype(np.float32)

        ppg_seq = np.stack([ppg_rz, dp_rz], axis=1).astype(np.float32)  # (L,2)

        # Static PPG features from filtered raw window
        ppg_stat = ppg_static_features(ppg_f.astype(np.float32), fs=fs_ppg)

        # Append pre-normalization amplitude stats so the model can recover absolute level
        ppg_amp_stats = np.array([safe_float(ppg_win_mean), safe_float(ppg_win_mad)], dtype=np.float32)
        ppg_stat_ext = np.concatenate([ppg_stat, ppg_amp_stats], axis=0)

        x_static = np.concatenate([eye_stat, ppg_stat_ext], axis=0).astype(np.float32)
        return eye_seq, ppg_seq, x_static

    def _augment(self, eye_seq: np.ndarray, ppg_seq: np.ndarray, x_static: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Lightweight physiological augmentations (train only).
        """
        # Random amplitude scaling
        a1 = self.rng.uniform(0.9, 1.1)
        a2 = self.rng.uniform(0.9, 1.1)
        eye_seq = eye_seq.copy()
        ppg_seq = ppg_seq.copy()
        x_static = x_static.copy()

        # Apply scaling to selected channels
        eye_seq[:, 0] *= a1  # pupil
        ppg_seq[:, 0] *= a2
        ppg_seq[:, 1] *= a2

        # Small Gaussian noise (exclude validity channel at index -1)
        eye_seq[:, :-1] += self.rng.normal(0.0, 0.01, size=eye_seq[:, :-1].shape).astype(np.float32)
        ppg_seq += self.rng.normal(0.0, 0.01, size=ppg_seq.shape).astype(np.float32)
        x_static += self.rng.normal(0.0, 0.01, size=x_static.shape).astype(np.float32)
        # Clip bounded features to valid ranges after noise addition
        # eye_valid_ratio (idx 13) and ppg_quality (idx 29) are in [0, 1]
        for idx in [13, 29]:
            if idx < x_static.size:
                x_static[idx] = np.clip(x_static[idx], 0.0, 1.0)

        # Random time shift (circular) for PPG channels only
        shift = int(self.rng.randint(-5, 6))
        if shift != 0:
            ppg_seq = np.roll(ppg_seq, shift=shift, axis=0)

        # Randomly drop a small fraction of eye validity channel (simulate blinks)
        if self.rng.rand() < 0.3:
            mask = self.rng.rand(eye_seq.shape[0]) < 0.05
            eye_seq[mask, -1] = 0.0

        return eye_seq.astype(np.float32), ppg_seq.astype(np.float32), x_static.astype(np.float32)

    def __getitem__(self, idx: int):
        pidx, s, e, y, tlx, _diff = self.samples[idx]

        cache = None
        if self.cache_dir is not None:
            key = f"_mmcache_{pidx}"
            if not hasattr(self, key):
                setattr(self, key, self._maybe_build_cache(pidx))
            cache = getattr(self, key)

        if cache is not None and (s, e) in cache:
            eye_seq, ppg_seq, x_static = cache[(s, e)]
        else:
            stream = self._load_streams(pidx)
            eye_seq, ppg_seq, x_static = self._build_one(stream, s, e)

        if self.augment:
            eye_seq, ppg_seq, x_static = self._augment(eye_seq, ppg_seq, x_static)

        # y_reg: if tlx is nan, set to -1 and let loss ignore it
        y_reg = float(tlx) if np.isfinite(tlx) else -1.0

        return (
            eye_seq.astype(np.float32),
            ppg_seq.astype(np.float32),
            x_static.astype(np.float32),
            np.int64(y),
            np.float32(y_reg),
        )


# -----------------------------
# Feature name registry (for journal-grade reporting & explainability)
# -----------------------------
EYE_STATIC_FEATURE_NAMES = [
    "pupil_mean",
    "pupil_std",
    "pupil_trend",
    "blink_rate_hz",
    "gaze_speed_mean",
    "gaze_speed_std",
    "gaze_acc_mean",
    "gaze_acc_std",
    "gaze_dispersion",
    "pupil_pos_x_mean",
    "pupil_pos_x_std",
    "pupil_pos_y_mean",
    "pupil_pos_y_std",
    "eye_valid_ratio",
    "pupil_cv",
    "gaze_dir_z_mean",
]

PPG_STATIC_FEATURE_NAMES = [
    "hr_mean_bpm",
    "hr_std_bpm",
    "rmssd_s",
    "sdnn_s",
    "pnn50_pct",
    "rr_mean_s",
    "rr_std_s",
    "poincare_sd1_s",
    "poincare_sd2_s",
    "lf_power",
    "hf_power",
    "lf_hf_ratio",
    "rr_skewness",
    "ppg_quality_0_1",
    "ppg_win_amplitude_median",
    "ppg_win_amplitude_mad",
]

STATIC_FEATURE_NAMES = EYE_STATIC_FEATURE_NAMES + PPG_STATIC_FEATURE_NAMES


def get_static_feature_names() -> List[str]:
    """
    Returns the names for x_static (length = 16 eye + 16 ppg = 32 by default).
    This is used by:
      - permutation importance
      - publication tables/plots
      - HRV consistency analysis
    """
    return list(STATIC_FEATURE_NAMES)
