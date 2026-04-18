"""
PA-SSL: Signal Processing Utilities
Common DSP functions (filtering, normalization, peak detection) shared across
all dataset adapters to ensure identical preprocessing.
"""

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

try:
    import wfdb.processing
    HAS_WFDB_PROCESSING = True
except ImportError:
    HAS_WFDB_PROCESSING = False


def bandpass_filter(data: np.ndarray, lowcut: float, highcut: float, fs: int, order: int = 3) -> np.ndarray:
    """
    Apply Butterworth bandpass filter.
    
    Args:
        data: 1D signal array
        lowcut: Low cutoff frequency in Hz
        highcut: High cutoff frequency in Hz
        fs: Sampling rate in Hz
        order: Filter order
        
    Returns:
        Filtered signal
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)


def z_score_normalize(signal: np.ndarray) -> np.ndarray:
    """
    Per-beat z-score normalization.
    Returns zero array if signal is flat (std < 1e-6).
    """
    mean = np.mean(signal)
    std = np.std(signal)
    if std < 1e-6:
        return np.zeros_like(signal)
    return (signal - mean) / std


def detect_r_peaks(signal: np.ndarray, fs: int) -> np.ndarray:
    """
    Detect R-peaks using wfdb.processing or scipy fallback.
    
    Args:
        signal: 1D ECG signal
        fs: Sampling rate in Hz
        
    Returns:
        Array of peak indices
    """
    if HAS_WFDB_PROCESSING:
        return wfdb.processing.gqrs_detect(signal, fs=fs)
    else:
        # Scipy fallback: find_peaks with height and distance constraints
        height = np.mean(signal) + 0.5 * np.std(signal)
        distance = int(0.4 * fs)  # minimum 400ms between peaks
        peaks, _ = find_peaks(signal, height=height, distance=distance)
        return peaks
