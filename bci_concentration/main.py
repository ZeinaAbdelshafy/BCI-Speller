import numpy as np
from psychopy import core, event
from config import *
from eeg_generator import generate_realistic_eeg
from signal_processing import bandpass_filter, apply_asr, extract_band_powers, compute_zscore, calibrate_baseline
from gui_concentration import ConcentrationBarGUI

def main():
    print(f"📡 Generating {DURATION}s EEG @ {FS}Hz...")
    _, raw_eeg = generate_realistic_eeg(FS, DURATION, N_CHANNELS, state='focused')
    
    # 1. Filter & Clean
    filtered = bandpass_filter(raw_eeg, FS, BANDPASS_LOW, BANDPASS_HIGH, FILTER_ORDER)
    cleaned = apply_asr(filtered, FS, ASR_CALIB_S, ASR_WIN_S, ASR_CUTOFF)
    
    # 2. Baseline Calibration
    print("🔬 Calibrating baseline...")
    _, base_eeg = generate_realistic_eeg(FS, BASELINE_DURATION, N_CHANNELS, state='relaxed')
    base_filtered = bandpass_filter(base_eeg, FS, BANDPASS_LOW, BANDPASS_HIGH, FILTER_ORDER)
    base_cleaned = apply_asr(base_filtered, FS, 2.0, ASR_WIN_S, ASR_CUTOFF)
    baseline_stats = calibrate_baseline(base_cleaned, FS, BASELINE_DURATION, WINDOW_S, STEP_S)
    print(f"📊 Baseline β/α: μ = {baseline_stats['mu']:.3f}, σ = {baseline_stats['sigma']:.3f}")
    
    # 3. Initialize GUI
    gui = ConcentrationBarGUI(baseline_stats['mu'], baseline_stats['sigma'], THRESHOLD_Z)
    print("\n🟢 Running concentration tracking (Press ESC to exit)...")
    
    # 4. Real-time Loop
    win_n = int(WINDOW_S * FS)
    step_n = int(STEP_S * FS)
    for start in range(0, len(cleaned[0]) - win_n, step_n):
        seg = cleaned[:, start:start+win_n]
        alpha, beta, _ = extract_band_powers(seg, FS, WINDOW_S)
        ratio = beta / (alpha + 1e-6)
        z_score = compute_zscore(ratio, baseline_stats['mu'], baseline_stats['sigma'])
        
        gui.update(ratio, z_score)
        
        if event.getKeys(keyList=['escape']):
            break
        core.wait(0.01)
        
    print("\n✅ Session complete.")
    
    gui.close()
    core.quit()

if __name__ == "__main__":
    main()