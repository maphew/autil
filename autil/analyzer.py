"""Audio analysis orchestrator."""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .audio_loader import get_audio_info, load_audio, to_mono
from .loudness import analyze_loudness
from .segments import detect_solo_regions, detect_speaker_changes
from .silence import detect_silence
from .viz import create_visualization


def analyze_audio(
    file_path: str,
    silence_threshold_db: float = -40.0,
    sensitivity: float = 0.5,
    create_viz: bool = True,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Analyze audio file and produce fingerprint.

    Args:
        file_path: Path to audio file.
        silence_threshold_db: Silence threshold in dB.
        sensitivity: Detection sensitivity (0.0-1.0).
        create_viz: Whether to create visualization.
        output_dir: Output directory for visualization (default: input directory).

    Returns:
        Dict with analysis results.
    """
    # Load audio
    audio, sample_rate = load_audio(file_path)

    # Get file info
    file_info = get_audio_info(file_path)
    file_info["file"] = str(Path(file_path).resolve())

    # Convert to mono
    audio_mono = to_mono(audio)

    # Build results
    results = {
        "file": file_info["file"],
        "duration_seconds": round(file_info["duration"], 3),
        "sample_rate": file_info["sample_rate"],
        "channels": file_info["channels"],
        "codec": file_info["codec"],
    }

    # Loudness analysis
    results["loudness"] = analyze_loudness(audio_mono, sample_rate)

    # Silence detection
    results["silence"] = detect_silence(
        audio_mono, sample_rate, threshold_db=silence_threshold_db
    )

    # Speaker change detection
    results["speaker_changes"] = detect_speaker_changes(
        audio_mono, sample_rate, sensitivity=sensitivity
    )

    # Solo region detection
    results["solo_regions"] = detect_solo_regions(
        audio_mono, sample_rate, sensitivity=sensitivity
    )

    # Visualization
    if create_viz:
        input_path = Path(file_path)
        if output_dir:
            output_path = Path(output_dir)
        else:
            output_path = input_path.parent
        viz_path = output_path / f"{input_path.stem}_fingerprint.png"

        create_visualization(
            audio,
            sample_rate,
            results["loudness"],
            results["silence"],
            results["speaker_changes"],
            str(viz_path),
        )

        results["visualization"] = str(viz_path)

    return results


def save_results(results: Dict[str, Any], output_path: str) -> None:
    """Save results to JSON file.

    Args:
        results: Analysis results.
        output_path: Path to save JSON.
    """

    # Convert numpy types to Python native types for JSON serialization
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(i) for i in obj]
        return obj

    results = convert(results)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


def format_summary(results: Dict[str, Any]) -> str:
    """Format results as human-readable summary.

    Args:
        results: Analysis results.

    Returns:
        Formatted summary string.
    """
    lines = []
    lines.append("=" * 60)
    lines.append(f"File: {results['file']}")
    lines.append(f"Duration: {results['duration_seconds']}s")
    lines.append(f"Sample Rate: {results['sample_rate']} Hz")
    lines.append(f"Channels: {results['channels']}")
    lines.append("=" * 60)

    # Loudness
    loudness = results["loudness"]
    lines.append(f"\nLoudness:")
    lines.append(f"  Integrated: {loudness['integrated_lufs']} LUFS")
    lines.append(f"  LRA: {loudness['loudness_range_lra']} LU")
    if loudness.get("loudest") and loudness["loudest"].get(
        "lufs", -float("inf")
    ) != -float("inf"):
        lines.append(
            f"  Loudest: {loudness['loudest']['lufs']} LUFS at {loudness['loudest']['time']}s"
        )
    if loudness.get("softest") and loudness["softest"].get(
        "lufs", -float("inf")
    ) != -float("inf"):
        lines.append(
            f"  Softest: {loudness['softest']['lufs']} LUFS at {loudness['softest']['time']}s"
        )

    # Silence
    silence = results["silence"]
    lines.append(f"\nSilence: {len(silence)} regions")
    if silence:
        for i, region in enumerate(silence[:5]):
            lines.append(
                f"  {region['start']}s - {region['end']}s ({region['duration']}s)"
            )
        if len(silence) > 5:
            lines.append(f"  +{len(silence) - 5} more")

    # Speaker changes
    speakers = results["speaker_changes"]
    lines.append(f"\nSpeaker Changes: {len(speakers)} detected")
    if speakers:
        for change in speakers[:5]:
            lines.append(
                f"  {change['time']}s (confidence: {round(change['confidence'], 2)})"
            )
        if len(speakers) > 5:
            lines.append(f"  +{len(speakers) - 5} more")

    # Solo regions
    solos = results["solo_regions"]
    lines.append(f"\nActivity/Solo Regions: {len(solos)} detected")
    if solos:
        for region in solos[:5]:
            lines.append(
                f"  {region['start']}s - {region['end']}s ({region['duration']}s)"
            )
        if len(solos) > 5:
            lines.append(f"  +{len(solos) - 5} more")

    # Visualization
    if results.get("visualization"):
        lines.append(f"\nVisualization: {results['visualization']}")

    lines.append("=" * 60)

    return "\n".join(lines)
