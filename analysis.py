from typing import Any
import numpy as np
from scipy import signal
from scipy.stats import linregress

SAMPLE_RATE_HZ = 30

def compute_psd(signal_1d, fs):
    nperseg = min(256, len(signal_1d) // 4)
    if nperseg < 4: nperseg = max(4, len(signal_1d) // 2)
    freqs, power = signal.welch(signal_1d, fs=fs, nperseg=nperseg, scaling="density")
    mask = freqs > 0
    return freqs[mask], power[mask]

def fit_power_law(freqs, power):
    slope, _, r, _, _ = linregress(np.log10(freqs), np.log10(power))
    return -slope, r ** 2

def analyze_samples(samples: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    if not samples or len(samples) < 16:
        raise ValueError(f"Need at least 16 samples, got {len(samples) if samples else 0}")
    samples_sorted = sorted(samples, key=lambda s: s["t"])
    times = np.array([s["t"] for s in samples_sorted], dtype=float)
    arrays = {k: np.array([s[k] for s in samples_sorted], dtype=float)
              for k in ["x", "y", "z", "magnitude"]}
    duration_s = times[-1] - times[0]
    fs = len(times) / duration_s if duration_s > 0 else SAMPLE_RATE_HZ
    results = {}
    for col, arr in arrays.items():
        freqs, power = compute_psd(arr, fs)
        beta, r2 = fit_power_law(freqs, power)
        results[col] = {"beta": float(beta), "r2": float(r2)}
    return results
