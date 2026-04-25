import numpy as np
from psychopy import visual, core, event
from config import *
from signal_processing import bandpass_filter, apply_asr, extract_band_powers, compute_zscore, calibrate_baseline
from gui_concentration import ConcentrationBarGUI
from lsl_streamer import UnicornLSLStreamer

def main():
    print("🚀 Starting Live Concentration Detector")
    
    # 1. Connect Unicorn
    streamer = UnicornLSLStreamer(FS)
    if not streamer.connect():
        return
    streamer.start()
    print("⏳ Waiting for data buffer to fill (3s worth)...")
    wait_start = core.getTime()
    while len(streamer.buffer) < 3 * FS:
        if core.getTime() - wait_start > 10.0:
            print("❌ Timeout: No data received after 10s. Check Bluetooth & LSL stream.")
            streamer.stop()
            return
        core.wait(0.1)  # Yield CPU, avoid busy-spinning
    print("✅ Buffer ready. Starting calibration...")
    
    print(" Buffering (3s)...")
    core.wait(3.0)

    # 2. Live Baseline Calibration
    print(f" CALIBRATION ({BASELINE_DURATION}s) - Relax & close eyes...")
    core.wait(1.0)
    baseline_buffer = []
    calib_clock = core.Clock()
    while calib_clock.getTime() < BASELINE_DURATION:
        win = streamer.get_window(0.5)
        if win is not None:
            baseline_buffer.append(win)
        core.wait(0.1)

    if not baseline_buffer:
        print(" Calibration failed. No data received.")
        streamer.stop()
        return

    base_eeg = np.hstack(baseline_buffer)
    base_filt = bandpass_filter(base_eeg, FS, BANDPASS_LOW, BANDPASS_HIGH, FILTER_ORDER)
    base_clean = apply_asr(base_filt, FS, ASR_CALIB_S, ASR_WIN_S, ASR_CUTOFF)
    baseline_stats = calibrate_baseline(base_clean, FS, BASELINE_DURATION, WINDOW_S, STEP_S)
    print(f"📊 Baseline μ={baseline_stats['mu']:.3f}, σ={baseline_stats['sigma']:.3f}")

    # 3. GUI & State Management
    gui = ConcentrationBarGUI(THRESHOLD_Z)
    gui.win.setMouseVisible(False)
    
    # State flags for future P300/SSVEP integration
    STATE = {"concentration": True, "p300": False, "ssvep": False}
    print("🟢 LIVE TRACKING ACTIVE | ESC to exit\n")

    main_clock = core.Clock()
    last_update = 0.0

    try:
        while True:
            if event.getKeys(keyList=['escape']):
                break

            # --- EEG PROCESSING (Runs every STEP_S) ---
            if STATE["concentration"] and main_clock.getTime() - last_update >= STEP_S:
                window = streamer.get_window(WINDOW_S)
                if window is not None:
                    filt = bandpass_filter(window, FS, BANDPASS_LOW, BANDPASS_HIGH, FILTER_ORDER)
                    clean = apply_asr(filt, FS, ASR_CALIB_S, ASR_WIN_S, ASR_CUTOFF)
                    alpha, beta, _ = extract_band_powers(clean, FS, WINDOW_S)
                    
                    a_post = np.mean(alpha[POSTERIOR_CHANNELS])
                    b_front = np.mean(beta[FRONTAL_CHANNELS])
                    ratio = b_front / (a_post + 1e-6)
                    z = compute_zscore(ratio, baseline_stats['mu'], baseline_stats['sigma'])
                    
                    gui.update(ratio, z)
                last_update = main_clock.getTime()

            # --- UNIFIED RENDER (Exactly 1 flip/frame) ---
            if STATE["p300"]:
                pass  # p300_module.draw() goes here later
            if STATE["ssvep"]:
                pass  # ssvep_module.draw() goes here later
            if STATE["concentration"]:
                pass  # gui already draws internally, but structure is ready
            
            gui.win.flip()  # ✅ Single point of screen update

    except KeyboardInterrupt:
        print("\n⏹️ Interrupted")
    finally:
        print("\n👋 Shutting down...")
        streamer.stop()
        gui.close()
        core.quit()

if __name__ == "__main__":
    main()
