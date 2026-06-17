"""PI's research-grade power law pipeline (Jones et al. 2023)."""
from typing import Any

import numpy as np
import pandas as pd

from signal_processing2 import butter_gravity_filter, detect_behavioral_events
from scale_free_math2 import analyze_scale_free_events

SAMPLE_RATE_HZ = 30


def analyze_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Run the full PI pipeline on a list of sample dicts with keys: t, x, y, z, magnitude.

    Returns dict with: tau, power_law_range, goodness_of_fit, is_scale_free, n_events.
    """
    if not samples or len(samples) < 64:
        raise ValueError(f"Need at least 64 samples, got {len(samples) if samples else 0}")

    samples_sorted = sorted(samples, key=lambda s: s["t"])
    times = np.array([s["t"] for s in samples_sorted], dtype=float)
    x_arr = np.array([s["x"] for s in samples_sorted], dtype=float)
    y_arr = np.array([s["y"] for s in samples_sorted], dtype=float)
    z_arr = np.array([s["z"] for s in samples_sorted], dtype=float)

    duration_s = times[-1] - times[0]
    fs = len(times) / duration_s if duration_s > 0 else SAMPLE_RATE_HZ

    # butter_gravity_filter expects integer column names 0, 1, 2 for x, y, z
    df = pd.DataFrame({0: x_arr, 1: y_arr, 2: z_arr, "time": times})

    # Step 1: remove gravity
    df = butter_gravity_filter(df, cutoff=0.3, order=4, sampling_rate=fs)

    # Step 2: smooth dynamic magnitude (~0.2s window, matching PI's pipeline)
    smooth_window = max(3, int(fs / 5))
    df["dynamic_mag_butter_smooth"] = (
        pd.Series(df["dynamic_mag_butter"].values)
        .rolling(smooth_window, center=True, min_periods=1)
        .mean()
        .values
    )

    # Step 3: detect behavioral events
    events_df = detect_behavioral_events(
        df,
        column="dynamic_mag_butter_smooth",
        sampling_rate=fs,
        min_duration=0,
        min_auc=0,
    )

    n_detected = len(events_df)
    if n_detected < 20:
        return {
            "tau": None,
            "power_law_range": None,
            "goodness_of_fit": None,
            "is_scale_free": False,
            "n_events": n_detected,
            "error": f"Too few events detected ({n_detected}); need at least 20",
        }

    # Step 4: MLE truncated power law fit
    results = analyze_scale_free_events(events_df, event_size_column="auc", verbose=False)

    tau = results.get("tau")
    plr = results.get("power_law_range")
    gof = results.get("goodness_of_fit")

    return {
        "tau": float(tau) if tau is not None and not (isinstance(tau, float) and np.isnan(tau)) else None,
        "power_law_range": float(plr) if plr is not None and not (isinstance(plr, float) and np.isnan(plr)) else None,
        "goodness_of_fit": float(gof) if gof is not None and not (isinstance(gof, float) and np.isnan(gof)) else None,
        "is_scale_free": bool(results.get("is_scale_free", False)),
        "n_events": int(results.get("n_samples") or n_detected),
    }
