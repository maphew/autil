"""Silence detection utilities."""

from typing import Dict, List

import numpy as np

from .audio_loader import to_mono


def detect_silence(
    audio: np.ndarray,
    sample_rate: int,
    threshold_db: float = -40.0,
    min_duration: float = 0.3,
) -> List[Dict[str, float]]:
    """Detect silence regions in audio.

    Args:
        audio: Audio data (numpy array).
        sample_rate: Sample rate in Hz.
        threshold_db: Silence threshold in dB relative to max RMS.
        min_duration: Minimum duration of silence in seconds.

    Returns:
        List of {"start": float, "end": float, "duration": float}.
    """
    audio_mono = to_mono(audio)

    # Frame parameters
    frame_length = int(0.02 * sample_rate)  # 20 ms frames
    hop_length = int(0.01 * sample_rate)  # 10 ms hops

    # Compute RMS energy per frame
    frames = []
    for i in range(0, len(audio_mono) - frame_length, hop_length):
        frame = audio_mono[i : i + frame_length]
        rms = np.sqrt(np.mean(frame**2))
        frames.append(rms)

    frames = np.array(frames)

    # Handle edge case: no frames generated (audio too short)
    if len(frames) == 0:
        return []

    # Convert to dB relative to max RMS
    max_rms = np.max(frames)
    if max_rms > 0:
        db_values = 20 * np.log10(frames / max_rms)
    else:
        db_values = np.full_like(frames, -np.inf)

    # Mark silent frames
    silent_frames = db_values < threshold_db

    # Find contiguous silent regions
    silence_regions = []
    in_silence = False
    start_frame = 0

    for i, is_silent in enumerate(silent_frames):
        if is_silent and not in_silence:
            start_frame = i
            in_silence = True
        elif not is_silent and in_silence:
            start_time = start_frame * hop_length / sample_rate
            end_time = i * hop_length / sample_rate
            duration = end_time - start_time

            if duration >= min_duration:
                silence_regions.append(
                    {
                        "start": round(start_time, 3),
                        "end": round(end_time, 3),
                        "duration": round(duration, 3),
                    }
                )
            in_silence = False

    # Handle trailing silence
    if in_silence:
        start_time = start_frame * hop_length / sample_rate
        end_time = len(audio_mono) / sample_rate
        duration = end_time - start_time

        if duration >= min_duration:
            silence_regions.append(
                {
                    "start": round(start_time, 3),
                    "end": round(end_time, 3),
                    "duration": round(duration, 3),
                }
            )

    return silence_regions
