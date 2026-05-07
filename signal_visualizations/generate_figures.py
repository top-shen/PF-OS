"""
Professional Signal Visualization for Physiological Time-Series Data
Author: Signal Processing Visualization Expert
Purpose: Generate publication-quality figures for Eye tracking and PPG signals
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from matplotlib import rcParams
import warnings
import sys
warnings.filterwarnings('ignore')

# Set UTF-8 encoding for console output
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Data paths
TOBII_FILE = '../HPO-CLD/HPO-CLD060/HPO-CLD060-tobii 2020-02-14-1606.csv'
BITALINO_FILE = '../HPO-CLD/HPO-CLD060/HPO-CLD060-bitalino-1 2020-02-14-1606.csv'
OUTPUT_DIR = 'figures'

# Sampling rates
EYE_SAMPLING_RATE = 120  # Hz
PPG_SAMPLING_RATE = 1000  # Hz

# Professional color palette
COLORS = {
    'eye': '#1f77b4',      # Professional blue
    'ppg': '#d62728',      # Clinical red
    'artifact': '#ff7f0e', # Warning orange
    'missing': '#7f7f7f',  # Neutral gray
    'grid': '#e0e0e0'      # Subtle grid
}

# ============================================================================
# MATPLOTLIB STYLE CONFIGURATION
# ============================================================================

def setup_plot_style():
    """Configure matplotlib for publication-quality figures."""
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    rcParams['font.size'] = 11
    rcParams['axes.labelsize'] = 12
    rcParams['axes.titlesize'] = 14
    rcParams['xtick.labelsize'] = 10
    rcParams['ytick.labelsize'] = 10
    rcParams['legend.fontsize'] = 10
    rcParams['figure.titlesize'] = 14

    # Line and marker settings
    rcParams['lines.linewidth'] = 1.5
    rcParams['lines.antialiased'] = True

    # Axes settings
    rcParams['axes.linewidth'] = 1.0
    rcParams['axes.spines.top'] = False
    rcParams['axes.spines.right'] = False
    rcParams['axes.grid'] = True
    rcParams['grid.alpha'] = 0.3
    rcParams['grid.linewidth'] = 0.5

    # Tick settings
    rcParams['xtick.direction'] = 'out'
    rcParams['ytick.direction'] = 'out'
    rcParams['xtick.major.size'] = 5
    rcParams['ytick.major.size'] = 5

    # Figure settings
    rcParams['figure.dpi'] = 100
    rcParams['savefig.dpi'] = 300
    rcParams['savefig.bbox'] = 'tight'
    rcParams['savefig.pad_inches'] = 0.1

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_eye_data(smooth=True):
    """Load and preprocess eye tracking data."""
    df = pd.read_csv(TOBII_FILE)
    df = df[df['right_pupil_diameter'].notna()]
    signal = df['right_pupil_diameter'].values[:5000]

    if smooth:
        # Apply Savitzky-Golay filter for smooth visualization
        signal = savgol_filter(signal, window_length=11, polyorder=2)

    return signal

def load_ppg_data(smooth=True):
    """Load and preprocess PPG data."""
    df = pd.read_csv(BITALINO_FILE)
    signal = df['A1'].values[:10000]

    if smooth:
        # Light smoothing for PPG
        signal = savgol_filter(signal, window_length=5, polyorder=2)

    return signal

def create_time_axis(signal_length, sampling_rate):
    """Create time axis in seconds."""
    return np.arange(signal_length) / sampling_rate

# ============================================================================
# PLOTTING UTILITY FUNCTIONS
# ============================================================================

def save_figure(filename, dpi=300):
    """Save figure with consistent settings."""
    plt.savefig(f'{OUTPUT_DIR}/{filename}', dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"✓ Saved: {filename}")

# ============================================================================
# FIGURE GENERATION FUNCTIONS
# ============================================================================

def generate_figure1_eye_signal(eye_signal):
    """Figure 1: Clean Eye Signal Time-Series."""
    n_samples = 1000
    time = create_time_axis(n_samples, EYE_SAMPLING_RATE)

    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax.plot(time, eye_signal[:n_samples], color=COLORS['eye'], linewidth=1.5, alpha=0.9)

    ax.set_xlabel('Time (s)', fontweight='bold')
    ax.set_ylabel('Pupil Diameter (mm)', fontweight='bold')
    ax.set_title('Eye Signal', fontsize=16, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    save_figure('fig1_eye_signal.png')

def generate_figure2_ppg_waveform(ppg_signal):
    """Figure 2: PPG Waveform Time-Series."""
    n_samples = 2000
    time = create_time_axis(n_samples, PPG_SAMPLING_RATE)

    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax.plot(time, ppg_signal[:n_samples], color=COLORS['ppg'], linewidth=1.2, alpha=0.9)

    ax.set_xlabel('Time (s)', fontweight='bold')
    ax.set_ylabel('Amplitude (a.u.)', fontweight='bold')
    ax.set_title('PPG Waveform', fontsize=16, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    save_figure('fig2_ppg_waveform.png')

def generate_figure3_raw_eye_missing(eye_signal):
    """Figure 3: Raw Eye Signal with Missing Data Annotation."""
    n_samples = 1200
    time = create_time_axis(n_samples, EYE_SAMPLING_RATE)

    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax.plot(time, eye_signal[:n_samples], color=COLORS['eye'], linewidth=1.5, alpha=0.9)

    # Missing data region
    missing_start, missing_end = 400, 600
    time_start = missing_start / EYE_SAMPLING_RATE
    time_end = missing_end / EYE_SAMPLING_RATE

    ax.axvspan(time_start, time_end, alpha=0.25, color=COLORS['missing'], zorder=0)
    ax.axvline(time_start, color=COLORS['missing'], linestyle='--', linewidth=2, alpha=0.7)
    ax.axvline(time_end, color=COLORS['missing'], linestyle='--', linewidth=2, alpha=0.7)

    # Annotation - place text above the signal in the shaded region
    mid_time = (time_start + time_end) / 2
    y_pos = eye_signal[missing_start:missing_end].max() + 0.15
    ax.text(mid_time, y_pos, 'Missing Data',
            fontsize=11, fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='red', linewidth=2))

    ax.set_xlabel('Time (s)', fontweight='bold')
    ax.set_ylabel('Pupil Diameter (mm)', fontweight='bold')
    ax.set_title('Raw Eye Signal (120 Hz)', fontsize=16, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    save_figure('fig3_raw_eye_signal.png')

def generate_figure4_raw_ppg_artifact(ppg_signal):
    """Figure 4: Raw PPG Signal with Motion Artifact Annotation."""
    n_samples = 2500
    time = create_time_axis(n_samples, PPG_SAMPLING_RATE)

    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax.plot(time, ppg_signal[:n_samples], color=COLORS['ppg'], linewidth=1.2, alpha=0.9)

    # Motion artifact region
    artifact_start, artifact_end = 1000, 1500
    time_start = artifact_start / PPG_SAMPLING_RATE
    time_end = artifact_end / PPG_SAMPLING_RATE

    ax.axvspan(time_start, time_end, alpha=0.3, color=COLORS['artifact'], zorder=0)

    # Annotation
    mid_time = (time_start + time_end) / 2
    y_pos = ppg_signal[artifact_start:artifact_end].max()
    ax.text(mid_time, y_pos + 10, 'Motion Artifact', fontsize=11, fontweight='bold',
            ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='white',
                     edgecolor=COLORS['artifact'], linewidth=2))

    ax.set_xlabel('Time (s)', fontweight='bold')
    ax.set_ylabel('Amplitude (a.u.)', fontweight='bold')
    ax.set_title('Raw PPG Signal (1000 Hz)', fontsize=16, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    save_figure('fig4_raw_ppg_signal.png')

def generate_figure5_windowed_eye(eye_signal):
    """Figure 5: Windowed Eye Signal."""
    window_start, window_end = 200, 400
    window_duration_ms = (window_end - window_start) / EYE_SAMPLING_RATE * 1000

    window_signal = eye_signal[window_start:window_end]
    time = create_time_axis(len(window_signal), EYE_SAMPLING_RATE)

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.plot(time, window_signal, color=COLORS['eye'], linewidth=2, alpha=0.9)

    # Window boundaries
    ax.axvline(time[0], color='black', linestyle='--', linewidth=1.5, alpha=0.6, label='Window Boundary')
    ax.axvline(time[-1], color='black', linestyle='--', linewidth=1.5, alpha=0.6)

    ax.set_xlabel('Time (s)', fontweight='bold')
    ax.set_ylabel('Pupil Diameter (mm)', fontweight='bold')
    ax.set_title(f'Windowed Signal ({window_duration_ms:.0f} ms)',
                fontsize=16, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.legend(loc='upper right', framealpha=0.9)

    save_figure('fig5_windowed_eye.png')

def generate_figure6_stacked_ppg(ppg_signal):
    """Figure 6: Stacked PPG Channels."""
    segment_length = 500
    segment1 = ppg_signal[0:segment_length]
    segment2 = ppg_signal[1000:1000+segment_length]
    segment3 = ppg_signal[2000:2000+segment_length]

    time = create_time_axis(segment_length, PPG_SAMPLING_RATE)

    fig, axes = plt.subplots(3, 1, figsize=(8, 7), constrained_layout=True, sharex=True)

    segments = [segment1, segment2, segment3]
    labels = ['Channel 1', 'Channel 2', 'Channel 3']

    for ax, segment, label in zip(axes, segments, labels):
        ax.plot(time, segment, color=COLORS['ppg'], linewidth=1.2, alpha=0.9)
        ax.set_ylabel(label, fontweight='bold', fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.tick_params(axis='both', which='major', labelsize=9)

    axes[-1].set_xlabel('Time (s)', fontweight='bold')
    fig.suptitle('Stacked PPG Channels', fontsize=16, fontweight='bold', y=0.995)

    save_figure('fig6_stacked_ppg.png')

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Generate all professional signal visualization figures."""
    print("\n" + "="*60)
    print("Professional Signal Visualization Generator")
    print("="*60 + "\n")

    # Setup
    setup_plot_style()

    # Load data
    print("Loading data...")
    eye_signal = load_eye_data(smooth=True)
    ppg_signal = load_ppg_data(smooth=True)
    print("✓ Data loaded successfully\n")

    # Generate figures
    print("Generating figures...")
    generate_figure1_eye_signal(eye_signal)
    generate_figure2_ppg_waveform(ppg_signal)
    generate_figure3_raw_eye_missing(eye_signal)
    generate_figure4_raw_ppg_artifact(ppg_signal)
    generate_figure5_windowed_eye(eye_signal)
    generate_figure6_stacked_ppg(ppg_signal)

    print("\n" + "="*60)
    print(f"✓ All 6 figures generated successfully!")
    print(f"✓ Saved to: {OUTPUT_DIR}/")
    print(f"✓ Resolution: 300 DPI (publication quality)")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
