"""Visualization utilities for audio analysis results."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

import librosa
import librosa.display

from .audio_loader import to_mono


def create_visualization(
    audio: np.ndarray,
    sample_rate: int,
    loudness_data: Dict[str, Any],
    silence_regions: List[Dict[str, float]],
    speaker_changes: List[Dict[str, Any]],
    output_path: str,
) -> str:
    """Create visualization with waveform, spectrogram, and loudness.

    Args:
        audio: Audio data.
        sample_rate: Sample rate in Hz.
        loudness_data: Loudness analysis results.
        silence_regions: List of silence regions.
        speaker_changes: List of speaker changes.
        output_path: Path to save PNG.

    Returns:
        Path to saved visualization.
    """
    audio_mono = to_mono(audio)
    duration = len(audio_mono) / sample_rate

    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(16, 10))

    # === Subplot 1: Waveform ===
    ax1 = axes[0]
    time_axis = np.linspace(0, duration, len(audio_mono))
    ax1.plot(time_axis, audio_mono, linewidth=0.3, alpha=0.7, color="steelblue")
    ax1.set_xlim(0, duration)
    ax1.set_ylim(-1.05, 1.05)
    ax1.set_title("Waveform", fontsize=12)
    ax1.set_xlabel("Time (s)")
    ax1.grid(alpha=0.3)

    # Overlay silence regions
    for region in silence_regions:
        ax1.axvspan(region["start"], region["end"], alpha=0.3, color="gray")

    # Overlay speaker changes
    for change in speaker_changes:
        ax1.axvline(
            x=change["time"], alpha=0.5, linewidth=0.8, color="red", linestyle="--"
        )

    # === Subplot 2: Mel Spectrogram ===
    ax2 = axes[1]
    S = librosa.feature.melspectrogram(y=audio_mono, sr=sample_rate, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(
        S_dB, x_axis="time", y_axis="mel", sr=sample_rate, ax=ax2
    )
    fig.colorbar(img, ax=ax2, format="%+2.0f dB")
    ax2.set_title("Mel Spectrogram")

    # === Subplot 3: Loudness ===
    ax3 = axes[2]
    moments = loudness_data.get("moments", [])

    if moments:
        # Filter out -inf values for plotting
        valid_moments = [m for m in moments if m["lufs"] != float("-inf")]

        if valid_moments:
            times = [m["time"] for m in valid_moments]
            lufs_values = [m["lufs"] for m in valid_moments]

            ax3.plot(
                times, lufs_values, linewidth=1.5, color="darkgreen", label="Loudness"
            )

            # Mark loudest and softest (filter out -inf)
            loudest = loudness_data.get("loudest")
            softest = loudness_data.get("softest")

            if loudest and loudest.get("lufs") != float("-inf"):
                ax3.scatter(
                    loudest["time"],
                    loudest["lufs"],
                    color="red",
                    s=100,
                    zorder=5,
                    label="Loudest",
                )
            if softest and softest.get("lufs") != float("-inf"):
                ax3.scatter(
                    softest["time"],
                    softest["lufs"],
                    color="blue",
                    s=100,
                    zorder=5,
                    label="Softest",
                )

            # Set limits
            min_lufs = min(lufs_values)
            max_lufs = max(lufs_values)
            ax3.set_ylim([min(-70, min_lufs - 5), max(0, max_lufs + 5)])

            integrated = loudness_data.get("integrated_lufs", 0)
            lra = loudness_data.get("loudness_range_lra", 0)
            ax3.set_title(f"Loudness (Integrated: {integrated} LUFS, LRA: {lra} LU)")
            ax3.legend(loc="lower right")
        else:
            ax3.text(
                0.5,
                0.5,
                "No loudness data available",
                ha="center",
                va="center",
                fontsize=14,
            )
            ax3.set_title("Loudness")

        integrated = loudness_data.get("integrated_lufs", 0)
        lra = loudness_data.get("loudness_range_lra", 0)
        ax3.set_title(f"Loudness (Integrated: {integrated} LUFS, LRA: {lra} LU)")
        ax3.legend(loc="lower right")
    else:
        ax3.text(
            0.5,
            0.5,
            "No loudness data available",
            ha="center",
            va="center",
            fontsize=14,
        )
        ax3.set_title("Loudness")

    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("LUFS")
    ax3.grid(alpha=0.3)
    ax3.set_xlim(0, duration)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path
