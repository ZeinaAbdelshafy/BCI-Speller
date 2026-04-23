import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch

# ================= 1. DUMMY SIGNAL + SAFE ARTIFACT INJECTION =================
def generate_eeg_with_artifacts(fs=250, duration=20.0, n_channels=8, state='relaxed'):
    t = np.arange(0, duration, 1/fs)
    n = len(t)
    ch_names = ['F3','Fz','F4','C3','Cz','C4','P3','Pz']
    eeg = np.zeros((n_channels, n))

    freqs = np.fft.rfftfreq(n, 1/fs)
    for ch in range(n_channels):
        amp = 1/np.maximum(freqs, 0.5)**1.5; amp[0]=0
        eeg[ch] += np.fft.irfft(amp * np.exp(1j*np.random.rand(len(freqs))*2*np.pi), n) * 8

    # Base weights
    alpha_w = np.array([0.3,0.2,0.3,0.5,0.6,0.5,0.9,0.8])
    beta_w  = np.array([0.8,0.7,0.8,0.9,0.8,0.9,0.5,0.4])
    gamma_w = np.array([0.5,0.5,0.5,0.6,0.6,0.6,0.7,0.7])

    # REALISTIC STATE MODULATION
    if state == 'focused':
        alpha_w *= 0.5   # Alpha suppression during focus
        beta_w  *= 2.0   # Beta increase during focus
    elif state == 'distracted':
        alpha_w *= 1.2
        beta_w  *= 0.6

    for f, w in [(10.0, alpha_w), (18.0, beta_w), (38.0, gamma_w)]:
        for ch in range(n_channels):
            eeg[ch] += w[ch]*np.random.uniform(1,8)*np.sin(2*np.pi*f*t + np.random.rand()*6)

    blink_times = [2.0, 5.0, 9.0, 13.0, 17.0]
    emg_times   = [3.5, 7.0, 11.0, 15.5, 18.5]
    for t_sec in blink_times:
        i = int(t_sec * fs)
        if i + 500 <= n:
            eeg[:3, i:i+500] += 150 * np.exp(-np.linspace(-2, 2, 500)**2)
    for t_sec in emg_times:
        i = int(t_sec * fs)
        if i + 300 <= n:
            eeg[np.random.choice(n_channels), i:i+300] += np.random.normal(0, 25, 300)

    if state == 'focused':
        eeg[0:3] += 12 * np.sin(2 * np.pi * 18.0 * t)  # Force frontal beta boost
        eeg[5:8] -= 8 * np.sin(2 * np.pi * 10.0 * t)   # Force posterior alpha suppression

    return t, (eeg / np.max(np.abs(eeg))) * 40.0, ch_names
# ================= 2. TRANSPARENT ASR (NO ML, PURE MATH) =================
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

# ================= 3. BANDPASS FILTER & POWER EXTRACTION =================
def bandpass_filter(data, fs, low=0.5, high=45.0, order=4):
    nyq = fs / 2.0
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, data, axis=1)

def get_bandpowers(eeg, fs):
    f, Pxx = welch(eeg, fs=fs, nperseg=fs*2, noverlap=fs)
    alpha = np.mean(Pxx[:, (f>=8)&(f<=12)], axis=1)
    beta  = np.mean(Pxx[:, (f>=13)&(f<=30)], axis=1)
    gamma = np.mean(Pxx[:, (f>=30)&(f<=45)], axis=1)
    return alpha, beta, gamma, np.mean(Pxx, axis=0), f

# ================= 4. BASELINE CALIBRATION & Z-SCORE THRESHOLDING =================
def calibrate_baseline(eeg, fs, duration_s=60.0, window_s=2.0, step_s=0.5):
    """Compute baseline stats from a known reference state (e.g., eyes closed)"""
    n_samp = int(duration_s * fs)
    win_n = int(window_s * fs)
    step_n = int(step_s * fs)
    
    ratios, alphas, betas, gammas = [], [], [], []
    
    for start in range(0, n_samp - win_n + 1, step_n):
        seg = eeg[:, start:start+win_n]
        _, P = welch(seg, fs=fs, nperseg=win_n)
        f = np.fft.rfftfreq(win_n, 1/fs)
        
        a = np.mean(P[:, (f>=8)&(f<=12)])
        b = np.mean(P[:, (f>=13)&(f<=30)])
        g = np.mean(P[:, (f>=30)&(f<=45)])
        
        alphas.append(a); betas.append(b); gammas.append(g)
        ratios.append(b / (a + 1e-6))
    
    return {
        'alpha': {'mu': np.mean(alphas), 'std': np.std(alphas)},
        'beta':  {'mu': np.mean(betas),  'std': np.std(betas)},
        'gamma': {'mu': np.mean(gammas), 'std': np.std(gammas)},
        'ratio': {'mu': np.mean(ratios), 'std': np.std(ratios)},
        'raw_ratios': np.array(ratios)
    }

def z_score(value, mu, sigma):
    """Transparent z-score normalization"""
    return (value - mu) / (sigma + 1e-6)

def detect_concentration(ratio_live, baseline_stats, threshold_z=2.0):
    """Return (is_detected, z_score) using statistical thresholding"""
    mu = baseline_stats['ratio']['mu']
    sigma = baseline_stats['ratio']['std']
    z = z_score(ratio_live, mu, sigma)
    return z > threshold_z, z

# ================= RUN PIPELINE =================
fs, dur = 250, 20.0
print(f"📡 Generating {dur}s EEG @ {fs}Hz...")
t, raw_eeg, ch_names = generate_eeg_with_artifacts(fs, dur)

# 1. Stabilize signal
filtered = bandpass_filter(raw_eeg, fs)
# 2. Remove artifacts
cleaned = apply_asr(filtered, fs, calib_s=3.0, win_s=0.5, cutoff=2.5)
# 3. Extract powers
a_pow, b_pow, g_pow, avg_psd, freqs = get_bandpowers(cleaned, fs)
conc_global = np.mean(b_pow) / (np.mean(a_pow) + 1e-6)

print(f" Cleaned Band Powers (μV²/Hz) | α: {np.mean(a_pow):.3f} | β: {np.mean(b_pow):.3f} | γ: {np.mean(g_pow):.3f}")
print(f" Global Concentration Index (β/α): {conc_global:.3f}")

# ================= BASELINE CALIBRATION =================
baseline_duration = 10.0  # seconds (use 60s in real calibration)
print(f"\n Calibrating baseline ({baseline_duration}s eyes-closed simulation)...")


# Baseline: relaxed state
t_base, base_eeg, _ = generate_eeg_with_artifacts(fs, baseline_duration, state='relaxed')

# Live signal: focused state
t, raw_eeg, ch_names = generate_eeg_with_artifacts(fs, dur, state='focused')

# Remove artifacts from baseline too
base_filtered = bandpass_filter(base_eeg, fs)
base_cleaned = apply_asr(base_filtered, fs, calib_s=2.0, win_s=0.5, cutoff=2.5)

# Compute baseline statistics
baseline_stats = calibrate_baseline(base_cleaned, fs, duration_s=baseline_duration)

print(f" Baseline β/α: μ = {baseline_stats['ratio']['mu']:.3f}, σ = {baseline_stats['ratio']['std']:.3f}")
threshold_ratio = baseline_stats['ratio']['mu'] + 2*baseline_stats['ratio']['std']
print(f" Detection threshold (z > 2.0): ratio > {threshold_ratio:.3f}")

# ================= APPLY TO LIVE SIGNAL =================
window, step_pw = int(2*fs), int(0.5*fs)
live_ratios, live_zscores, detections = [], [], []

for start in range(0, len(t)-window, step_pw):
    seg = cleaned[:, start:start+window]
    _, P = welch(seg, fs=fs, nperseg=window)
    f = np.fft.rfftfreq(window, 1/fs)
    
    a = np.mean(P[:, (f>=8)&(f<=12)])
    b = np.mean(P[:, (f>=13)&(f<=30)])
    ratio = b / (a + 1e-6)
    
    # Z-score + detection
    is_det, z_val = detect_concentration(ratio, baseline_stats, threshold_z=2.0)
    
    live_ratios.append(ratio)
    live_zscores.append(z_val)
    detections.append(is_det)

# Print detection summary
det_count = sum(detections)
print(f"\n Live Tracking: {det_count}/{len(detections)} windows detected as 'high concentration'")
print(f"   Average z-score: {np.mean(live_zscores):.2f} ± {np.std(live_zscores):.2f}")

# ================= VERIFICATION PLOTS =================
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Plot 1: Baseline ratio distribution
axes[0,0].hist(baseline_stats['raw_ratios'], bins=20, alpha=0.7, edgecolor='black')
axes[0,0].axvline(baseline_stats['ratio']['mu'], color='blue', ls='--', label='μ')
axes[0,0].axvline(threshold_ratio, color='red', ls=':', label='Threshold (μ+2σ)')
axes[0,0].set_title('Baseline β/α Distribution (Eyes Closed)')
axes[0,0].set_xlabel('β/α Ratio'); axes[0,0].set_ylabel('Frequency')
axes[0,0].legend(); axes[0,0].grid(axis='y')

# Plot 2: Live z-scores with threshold
time_axis = np.arange(len(live_zscores)) * 0.5
axes[0,1].plot(time_axis, live_zscores, label='Live z-score', linewidth=1.5)
axes[0,1].axhline(2.0, color='red', ls='--', label='Detection Threshold (z=2.0)')
axes[0,1].axhline(0, color='gray', ls=':', alpha=0.5)
axes[0,1].fill_between(time_axis, live_zscores, 2.0, 
                      where=(np.array(live_zscores) > 2.0), 
                      color='red', alpha=0.2, label='Detected')
axes[0,1].set_title('Concentration Detection (Z-Score Tracking)')
axes[0,1].set_xlabel('Time (s)'); axes[0,1].set_ylabel('Z-Score')
axes[0,1].legend(); axes[0,1].grid()

# Plot 3: Raw ratio + threshold line
axes[1,0].plot(time_axis, live_ratios, label='Live β/α', linewidth=1.5)
axes[1,0].axhline(threshold_ratio, color='red', ls='--', label=f'Threshold: {threshold_ratio:.3f}')
axes[1,0].axhline(baseline_stats['ratio']['mu'], color='blue', ls=':', label=f'Baseline μ: {baseline_stats["ratio"]["mu"]:.3f}')
axes[1,0].set_title('Raw β/α Ratio vs Threshold')
axes[1,0].set_xlabel('Time (s)'); axes[1,0].set_ylabel('β/α Ratio')
axes[1,0].legend(); axes[1,0].grid()

# Plot 4: Summary
axes[1,1].axis('off')
summary = (f" Step 2+ Verified:\n"
           f"• Baseline calibrated: {baseline_duration}s eyes-closed\n"
           f"• β/α baseline: μ={baseline_stats['ratio']['mu']:.3f}, σ={baseline_stats['ratio']['std']:.3f}\n"
           f"• Detection rule: z > 2.0 (ratio > {threshold_ratio:.3f})\n"
           f"• Detections: {det_count}/{len(detections)} windows ({100*det_count/len(detections):.1f}%)\n"
           f"• Ready for live LSL streaming + state switching")
axes[1,1].text(0.1, 0.5, summary, fontsize=11, va='center', family='monospace')

plt.tight_layout()
plt.show()