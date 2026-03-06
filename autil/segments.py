"""Speaker change and solo/activity region detection."""

from typing import Dict, List

import numpy as np

from .audio_loader import to_mono


def detect_speaker_changes(
    audio: np.ndarray,
    sample_rate: int,
    sensitivity: float = 0.5,
    segment_duration: float = 0.5,
) -> List[Dict[str, any]]:
    """Detect speaker changes based on energy differences.

    Args:
        audio: Audio data (numpy array).
        sample_rate: Sample rate in Hz.
        sensitivity: Detection sensitivity (0.0-1.0).
        segment_duration: Duration of each segment in seconds.

    Returns:
        List of {"time": float, "confidence": float, "type": str}.
    """
    audio_mono = to_mono(audio)

    # Compute mean-squared energy in non-overlapping windows
    samples_per_segment = int(segment_duration * sample_rate)
    num_segments = len(audio_mono) // samples_per_segment

    energies = []
    for i in range(num_segments):
        segment = audio_mono[i * samples_per_segment : (i + 1) * samples_per_segment]
        mse = np.mean(segment**2)
        energies.append(mse)

    energies = np.array(energies)

    # Normalize to [0, 1]
    max_energy = np.max(energies)
    if max_energy > 0:
        normalized = energies / max_energy
    else:
        normalized = energies

    # Compute absolute differences between consecutive windows
    diffs = np.abs(np.diff(normalized))

    # Threshold based on sensitivity
    threshold = (1.0 - sensitivity) * 0.3 + 0.05

    # Find changes exceeding threshold with verification
    changes = []
    for i in range(len(diffs)):
        if diffs[i] >= threshold:
            # Verify by comparing average energy 2 windows before vs 2 windows after
            before_start = max(0, i - 2)
            after_end = min(len(normalized), i + 3)

            before_avg = np.mean(normalized[before_start : i + 1]) if i >= 1 else 0
            after_avg = (
                np.mean(normalized[i + 1 : after_end]) if i + 1 < len(normalized) else 0
            )

            verify_diff = abs(after_avg - before_avg)

            if verify_diff >= threshold * 0.5:
                confidence = min(1.0, diffs[i] * 2)
                time_seconds = (i + 1) * samples_per_segment / sample_rate
                changes.append(
                    {
                        "time": round(time_seconds, 2),
                        "confidence": round(confidence, 2),
                        "type": "amplitude_change",
                    }
                )

    return changes


def detect_solo_regions(
    audio: np.ndarray,
    sample_rate: int,
    sensitivity: float = 0.5,
    segment_duration: float = 0.25,
) -> List[Dict[str, float]]:
    """Detect solo/activity regions in audio.

    Args:
        audio: Audio data (numpy array).
        sample_rate: Sample rate in Hz.
        sensitivity: Detection sensitivity (0.0-1.0).
        segment_duration: Duration of each segment in seconds.

    Returns:
        List of {"start": float, "end": float, "duration": float, "type": str}.
    """
    audio_mono = to_mono(audio)

    # Compute RMS in non-overlapping windows
    samples_per_segment = int(segment_duration * sample_rate)
    num_segments = len(audio_mono) // samples_per_segment

    rms_values = []
    for i in range(num_segments):
        segment = audio_mono[i * samples_per_segment : (i + 1) * samples_per_segment]
        rms = np.sqrt(np.mean(segment**2))
        rms_values.append(rms)

    rms_values = np.array(rms_values)

    # Normalize to [0, 1]
    max_rms = np.max(rms_values)
    if max_rms > 0:
        normalized = rms_values / max_rms
    else:
        normalized = rms_values

    # Threshold based on sensitivity
    threshold = (1.0 - sensitivity) * 0.15 + 0.02

    # Mark active windows
    active = normalized > threshold

    # Find contiguous active regions
    regions = []
    in_active = False
    start_segment = 0

    for i, is_active in enumerate(active):
        if is_active and not in_active:
            start_segment = i
            in_active = True
        elif not is_active and in_active:
            start_time = start_segment * samples_per_segment / sample_rate
            end_time = i * samples_per_segment / sample_rate
            duration = end_time - start_time

            if duration >= 0.5:  # Minimum 0.5 seconds
                regions.append(
                    {
                        "start": round(start_time, 2),
                        "end": round(end_time, 2),
                        "duration": round(duration, 2),
                        "type": "possible_solo",
                    }
                )
            in_active = False

    # Handle trailing active region
    if in_active:
        start_time = start_segment * samples_per_segment / sample_rate
        end_time = len(audio_mono) / sample_rate
        duration = end_time - start_time

        if duration >= 0.5:
            regions.append(
                {
                    "start": round(start_time, 2),
                    "end": round(end_time, 2),
                    "duration": round(duration, 2),
                    "type": "possible_solo",
                }
            )

    return regions
