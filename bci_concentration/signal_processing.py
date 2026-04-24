import numpy as np
from scipy.signal import butter, filtfilt, welch

def bandpass_filter(data, fs, low=0.5, high=45.0, order=4):
    nyq = fs / 2.0
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, data, axis=1)

def apply_asr(data, fs, calib_s=3.0, win_s=0.5, cutoff=2.5):
    n_ch, n_samp = data.shape
    calib_n = int(calib_s * fs)
    win_n = int(win_s * fs)
    step = win_n

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

def extract_band_powers(eeg_segment, fs, window_s=2.0):
    win_n = int(window_s * fs)
    _, Pxx = welch(eeg_segment, fs=fs, nperseg=win_n, noverlap=win_n//2)
    f = np.fft.rfftfreq(win_n, 1/fs)
    alpha = np.mean(Pxx[:, (f>=8)&(f<=12)])
    beta  = np.mean(Pxx[:, (f>=13)&(f<=30)])
    gamma = np.mean(Pxx[:, (f>=30)&(f<=45)])
    return alpha, beta, gamma

def compute_zscore(value, mu, sigma):
    return (value - mu) / (sigma + 1e-6)

def calibrate_baseline(eeg, fs, duration_s=10.0, window_s=2.0, step_s=0.5):
    n_samp = int(duration_s * fs)
    win_n = int(window_s * fs)
    step_n = int(step_s * fs)
    ratios = []
    for start in range(0, n_samp - win_n + 1, step_n):
        seg = eeg[:, start:start+win_n]
        a, b, _ = extract_band_powers(seg, fs, window_s)
        ratios.append(b / (a + 1e-6))
    return {'mu': np.mean(ratios), 'sigma': np.std(ratios)}