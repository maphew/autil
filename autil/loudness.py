"""Loudness analysis using ITU-R BS.1770-4 standard."""

from typing import Any, Dict, List, Optional

import numpy as np
import pyloudnorm as pln

from .audio_loader import to_mono


def analyze_loudness(
    audio: np.ndarray, sample_rate: int, moment_width: float = 1.0
) -> Dict[str, Any]:
    """Analyze loudness of audio using ITU-R BS.1770-4.

    Args:
        audio: Audio data (numpy array).
        sample_rate: Sample rate in Hz.
        moment_width: Width of momentary loudness windows in seconds.

    Returns:
        Dict with integrated_lufs, loudness_range_lra, moments, loudest, softest.
    """
    audio_mono = to_mono(audio)

    # Create meter
    meter = pln.Meter(sample_rate)

    # Compute integrated loudness
    integrated_loudness = meter.integrated_loudness(audio_mono)

    # Compute loudness range (LRA) - wrapped in try/except
    try:
        loudness_range = meter.loudness_range(audio_mono)
    except Exception:
        loudness_range = 0.0

    # Compute momentary loudness
    moments = compute_momentary_loudness(audio_mono, sample_rate, moment_width)

    # Find loudest and softest moments
    loudest = None
    softest = None

    if moments:
        lufs_values = [m["lufs"] for m in moments]
        loudest_idx = np.argmax(lufs_values)
        softest_idx = np.argmin(lufs_values)
        loudest = moments[loudest_idx]
        softest = moments[softest_idx]

    return {
        "integrated_lufs": round(integrated_loudness, 2),
        "loudness_range_lra": round(loudness_range, 2),
        "moments": moments,
        "loudest": loudest,
        "softest": softest,
    }


def compute_momentary_loudness(
    audio: np.ndarray, sample_rate: int, moment_width: float = 1.0
) -> List[Dict[str, float]]:
    """Compute momentary loudness over time.

    Args:
        audio: Mono audio data.
        sample_rate: Sample rate in Hz.
        moment_width: Width of windows in seconds.

    Returns:
        List of {"time": float, "lufs": float}.
    """
    meter = pln.Meter(sample_rate)
    hop_size = int(sample_rate * moment_width)
    min_samples = int(sample_rate * 0.1)  # Skip segments shorter than 0.1s

    moments = []
    position = 0

    while position + hop_size <= len(audio):
        segment = audio[position : position + hop_size]

        if len(segment) >= min_samples:
            try:
                lufs = meter.integrated_loudness(segment)
            except Exception:
                lufs = -70.0  # Floor value

            moments.append(
                {
                    "time": round(position / sample_rate, 2),
                    "lufs": round(lufs, 2),
                }
            )

        position += hop_size

    return moments
