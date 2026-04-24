# ==========================================================
# CONFIGURATION: Sampling, Processing, and Channel Mapping
# ==========================================================

# 1. Sampling & Data Dimensions
FS = 250                    # Sampling frequency (Hz) for Unicorn
DURATION = 500          # Duration of dummy signal for testing (seconds)
N_CHANNELS = 8              # Number of channels on Unicorn

# 2. Signal Processing Parameters
BANDPASS_LOW = 0.5          # High-pass filter cutoff (Hz)
BANDPASS_HIGH = 45.0        # Low-pass filter cutoff (Hz)
FILTER_ORDER = 4            # Butterworth filter order

# 3. ASR (Artifact Subspace Reconstruction) Parameters
ASR_CALIB_S = 3.0           # ✅ Duration of "clean" calibration window (seconds)
ASR_WIN_S = 0.5             # Sliding window step for processing (seconds)
ASR_CUTOFF = 2.5            # Standard deviation threshold for rejecting artifacts

# 4. Detection & Baseline Parameters
BASELINE_DURATION = 10.0    # Duration of baseline recording (seconds)
THRESHOLD_Z = 2.0           # Z-score threshold for "High Concentration" detection
WINDOW_S = 2.0              # Window size for Welch Power Spectral Density (seconds)
STEP_S = 0.5                # Time step for live loop updates (seconds)

# 5. Channel Mapping (Unicorn Layout)
# Unicorn Order: [Fz, C3, Cz, C4, Pz, PO7, Oz, PO8]
# Indices:        0    1    2    3    4    5     6    7

# ✅ Frontal/Central channels (Beta-dominant: Active Cognition)
FRONTAL_CHANNELS = [0, 1, 2, 3]   

# ✅ Posterior channels (Alpha-dominant: Relaxation/Idling)
POSTERIOR_CHANNELS = [4, 5, 6, 7] 

