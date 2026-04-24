import numpy as np

def generate_realistic_eeg(fs=250, duration=20.0, n_channels=8, state='relaxed'):
    t = np.arange(0, duration, 1/fs)
    n = len(t)
    eeg = np.zeros((n_channels, n))

    # 1/f background noise
    freqs = np.fft.rfftfreq(n, 1/fs)
    for ch in range(n_channels):
        amp = 1/np.maximum(freqs, 0.5)**1.5; amp[0]=0
        eeg[ch] += np.fft.irfft(amp * np.exp(1j*np.random.rand(len(freqs))*2*np.pi), n) * 8

    # Base channel weights
    alpha_w = np.array([0.3,0.2,0.3,0.5,0.6,0.5,0.9,0.8])
    beta_w  = np.array([0.8,0.7,0.8,0.9,0.8,0.9,0.5,0.4])
    gamma_w = np.array([0.5,0.5,0.5,0.6,0.6,0.6,0.7,0.7])

    # Apply time-varying focus modulation with correct broadcasting
    if state == 'focused':
        # Envelope oscillates between 0.3 (low focus) and 1.0 (high focus)
        focus_env = 0.3 + 0.7 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.15 * t + np.pi/2))
        
        # Broadcast: (8, 1) * (1, N) -> (8, N)
        alpha_mod = alpha_w[:, np.newaxis] * (1.5 - 0.9 * focus_env)
        beta_mod  = beta_w[:, np.newaxis] * (0.6 + 1.4 * focus_env)
        gamma_mod = gamma_w[:, np.newaxis] * 1.0
        weights = [alpha_mod, beta_mod, gamma_mod]
    else:
        weights = [alpha_w[:, np.newaxis], beta_w[:, np.newaxis], gamma_w[:, np.newaxis]]

    # Generate oscillations
    for idx, f in enumerate([10.0, 18.0, 38.0]):
        for ch in range(n_channels):
            eeg[ch] += weights[idx][ch] * np.random.uniform(1, 8) * np.sin(2*np.pi*f*t + np.random.rand()*6)

    # Inject artifacts (scaled to duration)
    blink_times = np.linspace(0, duration, 8)
    emg_times   = np.linspace(0, duration, 6) + 1.0
    for t_sec in blink_times:
        i = int(t_sec * fs)
        if i + 500 <= n:
            eeg[:3, i:i+500] += 150 * np.exp(-np.linspace(-2, 2, 500)**2)
    for t_sec in emg_times:
        i = int(t_sec * fs)
        if i + 300 <= n:
            eeg[np.random.choice(n_channels), i:i+300] += np.random.normal(0, 25, 300)

    return t, (eeg / np.max(np.abs(eeg))) * 40.0