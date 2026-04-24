import numpy as np
from scipy.signal import butter, filtfilt, welch

def bandpass_filter(data, fs, low=0.5, high=45.0, order=4):
    """Zero-phase Butterworth bandpass filter"""
    nyq = fs / 2.0
    # Handle edge case where low/high are out of bounds
    low = max(low, 0.1)
    high = min(high, nyq - 0.1)
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, data, axis=1)

def apply_asr(data, fs, calib_s=3.0, win_s=0.5, cutoff=2.5):
    """Artifact Subspace Reconstruction via PCA + Eigenvalue Thresholding"""
    n_ch, n_samp = data.shape
    calib_n = min(int(calib_s * fs), n_samp - 1)  # ✅ Bounds check
    win_n = int(win_s * fs)
    step = win_n

    # Ensure we have enough data for calibration
    if calib_n < n_ch:
        print(f"⚠️ Warning: Not enough data for ASR calibration ({calib_n} samples < {n_ch} channels)")
        return data.copy()

    # Learn "clean" spatial covariance from calibration window
    C_clean = np.cov(data[:, :calib_n]) + 1e-6 * np.eye(n_ch)
    w_clean, v_clean = np.linalg.eigh(C_clean)
    idx = np.argsort(w_clean)[::-1]
    w_clean, v_clean = w_clean[idx], v_clean[:, idx]

    cleaned = data.copy()
    for i in range(calib_n, n_samp - win_n + 1, step):
        win = data[:, i:i+win_n]
        proj = v_clean.T @ win
        var_win = np.var(proj, axis=1)
        mask = var_win > (cutoff**2) * w_clean
        proj[mask] = 0.0
        cleaned[:, i:i+win_n] = v_clean @ proj
    return cleaned

def extract_band_powers(eeg_segment, fs, window_s=None):
    """
    Extract alpha, beta, gamma power using Welch's method.
    
    Parameters:
    -----------
    eeg_segment : numpy array (n_channels, n_samples)
        The EEG data segment
    fs : float
        Sampling frequency in Hz
    window_s : float or None
        Window length in seconds. If None, auto-detects from segment length.
    """
    n_samples = eeg_segment.shape[1]
    
    # ✅ If window_s not provided, calculate from actual segment length
    if window_s is None:
        window_s = n_samples / fs
    
    win_n = int(window_s * fs)
    
    # ✅ Ensure nperseg doesn't exceed actual segment length
    nperseg = min(win_n, n_samples)
    
    # ✅ Use the frequency array returned by welch
    f, Pxx = welch(eeg_segment, fs=fs, nperseg=nperseg, noverlap=nperseg//2)
    
    # Create boolean masks using the CORRECT frequency array
    alpha_mask = (f >= 8) & (f <= 12)
    beta_mask = (f >= 13) & (f <= 30)
    gamma_mask = (f >= 30) & (f <= 45)
    
    # Handle edge case where no bins fall in a band
    alpha = np.mean(Pxx[:, alpha_mask], axis=1) if np.any(alpha_mask) else 0.0
    beta  = np.mean(Pxx[:, beta_mask], axis=1) if np.any(beta_mask) else 0.0
    gamma = np.mean(Pxx[:, gamma_mask], axis=1) if np.any(gamma_mask) else 0.0
    
    return alpha, beta, gamma

def compute_zscore(value, mu, sigma):
    """Compute z-score with division-by-zero protection"""
    return (value - mu) / (sigma + 1e-6)

def calibrate_baseline(eeg, fs, duration_s=10.0, window_s=2.0, step_s=0.5):
    """Compute baseline stats from a known reference state"""
    n_samp = int(duration_s * fs)
    win_n = int(window_s * fs)
    step_n = int(step_s * fs)
    
    # Ensure we have enough data
    if n_samp < win_n:
        print(f"⚠️ Warning: Baseline duration too short for window size")
        return {'mu': 0.5, 'sigma': 0.1}  # Fallback defaults
    
    ratios = []
    for start in range(0, n_samp - win_n + 1, step_n):
        seg = eeg[:, start:start+win_n]
        # ✅ Pass window_s explicitly (optional now, but kept for clarity)
        a, b, _ = extract_band_powers(seg, fs, window_s)
        # Protect against division by zero
        ratio = b / (a + 1e-6)
        ratios.append(ratio)
    
    # Handle edge case of empty ratios list
    if len(ratios) == 0:
        return {'mu': 0.5, 'sigma': 0.1}
        
    return {'mu': np.mean(ratios), 'sigma': np.std(ratios)}
