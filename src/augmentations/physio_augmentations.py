"""
PA-SSL: Physiology-Aware ECG Augmentations
===========================================

The CORE NOVELTY of this paper. Each augmentation is formally constrained
to produce physiologically plausible ECG signals.

Key constraints:
  - R-peak morphology is preserved (QRS complex is sacred)
  - Heart rate stays within physiological bounds (30-200 BPM)
  - Introduced artifacts mimic real-world noise sources
  - Temporal relationships (P-QRS-T ordering) are maintained

All augmentations operate on numpy arrays of shape (250,) representing
a single beat at 100 Hz sampling rate, centered on the R-peak.
"""

import numpy as np
from scipy.signal import resample
from scipy.interpolate import interp1d
import pywt


# ─── CONSTANTS ────────────────────────────────────────────────────────────────
FS = 100                # Sampling rate (Hz)
BEAT_LEN = 250          # Beat length in samples (2.5s at 100Hz)
QRS_HALF_WIDTH = 3      # ±30ms around R-peak = ±3 samples at 100Hz (protected zone)
MIN_HR_BPM = 30         # Minimum physiological heart rate
MAX_HR_BPM = 200        # Maximum physiological heart rate
MIN_RR_SAMPLES = int(60 * FS / MAX_HR_BPM)   # ~30 samples at 200 BPM
MAX_RR_SAMPLES = int(60 * FS / MIN_HR_BPM)   # ~200 samples at 30 BPM


# ─── 1. CONSTRAINED TIME WARPING ─────────────────────────────────────────────

def constrained_time_warp(signal, r_peak_pos=125, max_warp=0.15, n_knots=5):
    """
    Piecewise time-warping that preserves QRS morphology.
    
    The QRS complex (R-peak ± QRS_HALF_WIDTH) is locked — its timing cannot
    shift. Only the P-wave and T-wave segments are warped.
    
    Args:
        signal: (250,) numpy array
        r_peak_pos: R-peak sample index within the beat
        max_warp: Maximum warping factor (0.15 = ±15% stretch/compression)
        n_knots: Number of control points for the warping function
    
    Returns:
        Warped signal of same length (250,)
    """
    n = len(signal)
    
    # Define anchor points that CANNOT move:
    # 0 (start), R-peak zone, n-1 (end)
    qrs_start = max(0, r_peak_pos - QRS_HALF_WIDTH)
    qrs_end = min(n - 1, r_peak_pos + QRS_HALF_WIDTH)
    
    # Create source timeline (original sample indices)
    source = np.linspace(0, n - 1, n)
    
    # Create warped target timeline
    # Fixed points: start, QRS zone boundaries, end
    fixed_source = [0, qrs_start, r_peak_pos, qrs_end, n - 1]
    fixed_target = [0, qrs_start, r_peak_pos, qrs_end, n - 1]
    
    # Add random knots in the non-QRS regions
    rng = np.random.RandomState()
    
    # Pre-QRS knots (P-wave region)
    if qrs_start > 20:
        pre_knots = np.linspace(10, qrs_start - 5, min(n_knots // 2, 2)).astype(int)
        for k in pre_knots:
            warp = rng.uniform(-max_warp, max_warp)
            fixed_source.append(k)
            fixed_target.append(k * (1 + warp))
    
    # Post-QRS knots (T-wave region)
    if n - 1 - qrs_end > 20:
        post_knots = np.linspace(qrs_end + 5, n - 10, min(n_knots // 2, 2)).astype(int)
        for k in post_knots:
            warp = rng.uniform(-max_warp, max_warp)
            fixed_source.append(k)
            fixed_target.append(k * (1 + warp))
    
    # Sort by source position and remove duplicates
    pairs = sorted(zip(fixed_source, fixed_target), key=lambda x: x[0])
    # Remove duplicate source positions
    seen = set()
    unique_pairs = []
    for s, t in pairs:
        s_int = int(round(s))
        if s_int not in seen:
            seen.add(s_int)
            unique_pairs.append((s, t))
    
    if len(unique_pairs) < 2:
        return signal.copy()
    
    src_pts = np.array([p[0] for p in unique_pairs])
    tgt_pts = np.array([p[1] for p in unique_pairs])
    
    # Clip targets to valid range
    tgt_pts = np.clip(tgt_pts, 0, n - 1)
    
    # Interpolate warping function
    try:
        warp_fn = interp1d(src_pts, tgt_pts, kind='linear', fill_value='extrapolate')
        warped_timeline = warp_fn(source)
        warped_timeline = np.clip(warped_timeline, 0, n - 1)
        
        # Resample signal along warped timeline
        signal_interp = interp1d(source, signal, kind='linear', fill_value='extrapolate')
        warped_signal = signal_interp(warped_timeline)
    except Exception:
        return signal.copy()
    
    return warped_signal.astype(np.float32)


# ─── 2. AMPLITUDE PERTURBATION WITH QRS PROTECTION ───────────────────────────

def amplitude_perturbation(signal, r_peak_pos=125, scale_range=(0.8, 1.2), 
                           qrs_protect=True):
    """
    Scale signal amplitude while protecting QRS morphology.
    
    If qrs_protect=True, the QRS complex amplitude is preserved while
    the P-wave and T-wave regions are scaled independently. This maintains
    the clinically critical QRS:T-wave amplitude ratio.
    
    Args:
        signal: (250,) numpy array
        r_peak_pos: R-peak position
        scale_range: (min_scale, max_scale) tuple
        qrs_protect: If True, QRS region retains original amplitude
    
    Returns:
        Amplitude-perturbed signal (250,)
    """
    result = signal.copy()
    
    if qrs_protect:
        qrs_start = max(0, r_peak_pos - QRS_HALF_WIDTH)
        qrs_end = min(len(signal), r_peak_pos + QRS_HALF_WIDTH + 1)
        
        # Scale non-QRS regions
        scale = np.random.uniform(*scale_range)
        
        # Create smooth blending mask (avoids sharp transitions at QRS boundaries)
        mask = np.ones(len(signal)) * scale
        # Transition zone: 5 samples on each side of QRS boundary
        transition = 5
        for i in range(max(0, qrs_start - transition), qrs_start):
            alpha = (qrs_start - i) / transition
            mask[i] = 1.0 * (1 - alpha) + scale * alpha
        for i in range(qrs_end, min(len(signal), qrs_end + transition)):
            alpha = (i - qrs_end) / transition
            mask[i] = 1.0 * (1 - alpha) + scale * alpha
        mask[qrs_start:qrs_end] = 1.0  # QRS untouched
        
        result = result * mask
    else:
        scale = np.random.uniform(*scale_range)
        result = result * scale
    
    return result.astype(np.float32)


# ─── 3. BASELINE WANDER INJECTION ────────────────────────────────────────────

def baseline_wander(signal, fs=FS, max_amplitude=0.15, max_freq=0.5):
    """
    Add realistic baseline wander (low-frequency drift).
    
    Simulates respiration-induced and electrode-movement artifacts.
    Constrained to < 0.5 Hz to ensure it doesn't corrupt morphological features.
    
    Args:
        signal: (250,) numpy array
        fs: Sampling rate
        max_amplitude: Maximum wander amplitude (relative to signal std)
        max_freq: Maximum frequency of wander (Hz)
    
    Returns:
        Signal with added baseline wander (250,)
    """
    n = len(signal)
    t = np.arange(n) / fs
    
    # Random number of sinusoidal components (1-3)
    n_components = np.random.randint(1, 4)
    wander = np.zeros(n)
    
    sig_std = np.std(signal)
    if sig_std < 1e-6:
        sig_std = 1.0
    
    for _ in range(n_components):
        freq = np.random.uniform(0.05, max_freq)
        amplitude = np.random.uniform(0.02, max_amplitude) * sig_std
        phase = np.random.uniform(0, 2 * np.pi)
        wander += amplitude * np.sin(2 * np.pi * freq * t + phase)
    
    return (signal + wander).astype(np.float32)


# ─── 4. EMG NOISE INJECTION ──────────────────────────────────────────────────

def emg_noise_injection(signal, snr_range=(15, 30)):
    """
    Add realistic electromyographic (muscle) artifact noise.
    
    EMG noise is high-frequency (20-100 Hz range at our 100 Hz sampling),
    appearing as random tremor-like interference. Controlled via SNR to
    avoid overwhelming the cardiac signal.
    
    Args:
        signal: (250,) numpy array
        snr_range: (min_snr_db, max_snr_db) — signal-to-noise ratio bounds
    
    Returns:
        Signal with EMG noise added (250,)
    """
    snr_db = np.random.uniform(*snr_range)
    
    # Calculate signal power
    p_signal = np.mean(signal ** 2)
    if p_signal < 1e-10:
        return signal.copy()
    
    # Calculate noise power from SNR
    p_noise = p_signal / (10 ** (snr_db / 10))
    
    # Generate EMG-like noise
    # EMG has more high-frequency content than Gaussian white noise
    noise = np.random.normal(0, np.sqrt(p_noise), len(signal))
    
    # Shape the noise spectrum to be more EMG-like
    # Simple approach: apply a mild high-pass effect by differencing
    if np.random.rand() > 0.5:
        noise_shaped = np.diff(noise, prepend=noise[0]) * 0.7 + noise * 0.3
        noise = noise_shaped * np.sqrt(p_noise / (np.mean(noise_shaped**2) + 1e-10))
    
    return (signal + noise).astype(np.float32)


# ─── 5. HEART-RATE PLAUSIBLE RESAMPLING ──────────────────────────────────────

def heart_rate_resample(signal, r_peak_pos=125, rate_factor_range=(0.85, 1.15)):
    """
    Resample the beat to simulate a slightly different heart rate.
    
    The signal is stretched/compressed to simulate what the beat would
    look like at a different heart rate. Bounded to ±15% to stay within
    physiologically plausible rate changes.
    
    The R-peak remains centered in the output window.
    
    Args:
        signal: (250,) numpy array
        r_peak_pos: R-peak position
        rate_factor_range: (min_factor, max_factor) — 1.0 = same rate
    
    Returns:
        Resampled signal of original length (250,)
    """
    n = len(signal)
    rate_factor = np.random.uniform(*rate_factor_range)
    
    # New length before re-fitting to original window
    new_len = int(n * rate_factor)
    new_len = max(50, new_len)  # Safety minimum
    
    # Resample
    resampled = resample(signal, new_len)
    
    # Fit back to original length, centered on R-peak
    new_rpeak = int(r_peak_pos * rate_factor)
    new_rpeak = np.clip(new_rpeak, 0, new_len - 1)
    
    output = np.zeros(n, dtype=np.float32)
    
    # Place resampled signal so R-peak aligns with original position
    offset = r_peak_pos - new_rpeak
    
    for i in range(n):
        src_idx = i - offset
        if 0 <= src_idx < new_len:
            output[i] = resampled[src_idx]
        # else: zero-padded (rare edge case)
    
    return output


# ─── 6. POWERLINE INTERFERENCE ───────────────────────────────────────────────

def powerline_interference(signal, fs=FS, amplitude_range=(0.01, 0.05)):
    """
    Add realistic 50/60 Hz powerline interference.
    
    Common artifact in clinical ECG recordings, especially from portable
    devices. At our 100 Hz sampling rate, 50 Hz appears close to Nyquist,
    creating aliased patterns.
    
    Args:
        signal: (250,) numpy array
        fs: Sampling rate
        amplitude_range: (min, max) amplitude relative to signal std
    
    Returns:
        Signal with powerline interference (250,)
    """
    n = len(signal)
    t = np.arange(n) / fs
    
    sig_std = np.std(signal)
    if sig_std < 1e-6:
        sig_std = 1.0
    
    # Choose 50 Hz or 60 Hz
    freq = np.random.choice([50.0, 60.0])
    
    # At 100 Hz sampling, 50 Hz aliases to 50 Hz (Nyquist=50) → appears as DC or near-DC
    # More realistic: add a frequency slightly below Nyquist  
    effective_freq = min(freq, fs / 2 - 1)
    
    amplitude = np.random.uniform(*amplitude_range) * sig_std
    phase = np.random.uniform(0, 2 * np.pi)
    
    interference = amplitude * np.sin(2 * np.pi * effective_freq * t + phase)
    
    return (signal + interference).astype(np.float32)


# ─── 7. RANDOM SEGMENT DROPOUT ───────────────────────────────────────────────

def segment_dropout(signal, r_peak_pos=125, max_dropout_frac=0.08):
    """
    Zero out a small random segment of the signal, avoiding the QRS complex.
    
    Simulates momentary signal loss or electrode detachment. The QRS
    region is protected to ensure diagnostic features are preserved.
    
    Args:
        signal: (250,) numpy array
        r_peak_pos: R-peak position
        max_dropout_frac: Maximum fraction of signal to drop
    
    Returns:
        Signal with segment dropout (250,)
    """
    result = signal.copy()
    n = len(signal)
    
    # Protected QRS zone
    qrs_start = max(0, r_peak_pos - QRS_HALF_WIDTH - 2)
    qrs_end = min(n, r_peak_pos + QRS_HALF_WIDTH + 3)
    
    # Dropout length
    dropout_len = int(n * np.random.uniform(0.02, max_dropout_frac))
    dropout_len = max(2, dropout_len)
    
    # Choose start position outside QRS zone
    # Available regions: [0, qrs_start - dropout_len] or [qrs_end, n - dropout_len]
    candidates = []
    if qrs_start - dropout_len > 0:
        candidates.append(('pre', 0, qrs_start - dropout_len))
    if n - dropout_len > qrs_end:
        candidates.append(('post', qrs_end, n - dropout_len))
    
    if len(candidates) == 0:
        return result
    
    region = candidates[np.random.randint(len(candidates))]
    start = np.random.randint(region[1], region[2] + 1)
    
    # Smooth dropout with fade-in/fade-out (3 samples each side)
    fade = 3
    for i in range(start, min(start + dropout_len, n)):
        if i < start + fade:
            alpha = (i - start) / fade
            result[i] *= (1 - alpha)
        elif i >= start + dropout_len - fade:
            alpha = (start + dropout_len - i) / fade
            result[i] *= (1 - alpha)
        else:
            result[i] = 0.0
    return result.astype(np.float32)


# ─── 8. WAVELET MASKING (FREQUENCY-DOMAIN) ───────────────────────────────────

def wavelet_masking(signal, wavelet='db4', level=None, max_mask_ratio=0.3):
    """
    Randomly mask (zero out) high-frequency wavelet coefficients.
    
    This operates in the frequency-domain (Next-Gen SSL feature) to force
    the model to learn frequency-invariant representations. It preserves the
    approximation coefficients (low frequency) to maintain the overall morphological
    integrity of the beat, but randomlyDrops detail coefficients.
    
    Args:
        signal: (250,) numpy array
        wavelet: PyWavelets wavelet name (e.g., 'db4', 'sym8')
        level: Decomposition level (default max possible)
        max_mask_ratio: Max fraction of detail coefficients to drop
    
    Returns:
        Signal with masked frequencies (250,)
    """
    result = signal.copy()
    
    try:
        # Decompose the signal
        coeffs = pywt.wavedec(result, wavelet, level=level)
        
        # coeffs[0] is the approximation (low freq, preserves shape). DO NOT MASK.
        # coeffs[1:] are the details (high freq features). We randomly mask these.
        
        for i in range(1, len(coeffs)):
            if np.random.rand() < 0.5:  # 50% chance to mask something in this level
                n_coeffs = len(coeffs[i])
                mask_len = int(n_coeffs * np.random.uniform(0.1, max_mask_ratio))
                if mask_len > 0:
                    start_idx = np.random.randint(0, max(1, n_coeffs - mask_len))
                    coeffs[i][start_idx:start_idx + mask_len] = 0.0
                    
        # Reconstruct signal
        reconstructed = pywt.waverec(coeffs, wavelet)
        
        # Make sure length matches exactly (waverec can sometimes be off by 1)
        if len(reconstructed) > len(result):
            reconstructed = reconstructed[:len(result)]
        elif len(reconstructed) < len(result):
            # Pad with last value
            reconstructed = np.pad(reconstructed, (0, len(result) - len(reconstructed)), 'edge')
            
        return reconstructed.astype(np.float32)
    except Exception:
        return result.astype(np.float32)
