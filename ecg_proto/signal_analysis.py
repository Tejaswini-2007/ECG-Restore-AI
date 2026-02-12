from typing import Dict, Optional, Tuple

import numpy as np
from scipy.signal import find_peaks


def detect_heartbeats(
    time: np.ndarray,
    amplitude: np.ndarray,
    min_bpm: float = 30.0,
    max_bpm: float = 220.0,
    min_peak_height: float = 0.2,
) -> Tuple[np.ndarray, Optional[float], Dict[str, np.ndarray]]:
    """
    Detect approximate R-peaks and estimate heart rate from an ECG signal.

    Parameters
    ----------
    time : np.ndarray
        Time axis in seconds.
    amplitude : np.ndarray
        Normalized amplitude signal in [-1, 1].
    min_bpm : float
        Minimum physiologically plausible heart rate.
    max_bpm : float
        Maximum physiologically plausible heart rate.
    min_peak_height : float
        Minimum relative peak height (on normalized amplitude) to consider a beat.

    Returns
    -------
    peaks : np.ndarray
        Indices of detected peaks in the signal.
    heart_rate_bpm : Optional[float]
        Estimated heart rate (BPM) using median RR interval, or None if not enough peaks.
    debug : dict
        Additional arrays such as RR intervals.
    """
    if time.size != amplitude.size:
        raise ValueError("time and amplitude must have the same length.")
    if time.size < 5:
        return np.array([], dtype=int), None, {"rr_intervals": np.array([])}

    duration = float(time[-1] - time[0])
    if duration <= 0:
        return np.array([], dtype=int), None, {"rr_intervals": np.array([])}

    fs = len(time) / duration  # approximate sampling frequency

    # Convert BPM bounds to sample distances
    max_hr = max_bpm
    min_hr = min_bpm

    min_rr_sec = 60.0 / max_hr  # smallest RR interval (fastest heart rate)
    max_rr_sec = 60.0 / min_hr  # largest RR interval (slowest heart rate)

    min_distance_samples = max(int(min_rr_sec * fs * 0.8), 1)

    # Use SciPy peak finding on the positive signal
    peaks, properties = find_peaks(amplitude, height=min_peak_height, distance=min_distance_samples)

    if peaks.size < 2:
        return peaks, None, {"rr_intervals": np.array([])}

    rr_intervals = np.diff(time[peaks])

    # Filter RR intervals by physiological plausibility
    mask = (rr_intervals >= min_rr_sec) & (rr_intervals <= max_rr_sec)
    rr_valid = rr_intervals[mask]

    if rr_valid.size == 0:
        return peaks, None, {"rr_intervals": rr_intervals}

    median_rr = float(np.median(rr_valid))
    heart_rate_bpm = 60.0 / median_rr if median_rr > 0 else None

    debug = {
        "rr_intervals": rr_intervals,
        "rr_valid": rr_valid,
        "peak_heights": properties.get("peak_heights", np.array([])),
    }
    return peaks, heart_rate_bpm, debug

