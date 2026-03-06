"""Audio loading utilities with dual-backend support (soundfile + audioread)."""

from typing import Tuple

import numpy as np
import soundfile as sf


def load_audio(file_path: str) -> Tuple[np.ndarray, int]:
    """Load audio file and return audio data and sample rate.

    Tries soundfile first, falls back to audioread+ffmpeg on failure.

    Args:
        file_path: Path to audio file.

    Returns:
        Tuple of (audio_data, sample_rate). Audio shape is (samples,) for mono
        or (samples, channels) for stereo.
    """
    try:
        audio, sr = sf.read(file_path, dtype="float32")
        return audio, sr
    except Exception:
        # Fallback to audioread
        import audioread

        with audioread.audio_open(file_path) as f:
            sample_rate = f.samplerate
            channels = f.channels
            chunks = []
            for chunk in f:
                # Convert bytes to int16 numpy array
                int_data = np.frombuffer(chunk, dtype=np.int16)
                # Convert to float32
                float_data = int_data.astype(np.float32) / 32768.0
                chunks.append(float_data)

        audio = np.concatenate(chunks)

        if channels > 1:
            audio = audio.reshape((-1, channels))

        return audio, sample_rate


def get_audio_info(file_path: str) -> dict:
    """Get audio file metadata.

    Args:
        file_path: Path to audio file.

    Returns:
        Dict with keys: duration, sample_rate, channels, codec, frames.
    """
    try:
        info = sf.info(file_path)
        return {
            "duration": info.duration,
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "codec": info.format,
            "frames": info.frames,
        }
    except Exception:
        # Fallback to audioread
        import audioread

        with audioread.audio_open(file_path) as f:
            duration = f.duration
            sample_rate = f.samplerate
            channels = f.channels
            frames = int(duration * sample_rate)

        return {
            "duration": duration,
            "sample_rate": sample_rate,
            "channels": channels,
            "codec": "unknown",
            "frames": frames,
        }


def to_mono(audio: np.ndarray) -> np.ndarray:
    """Convert audio to mono.

    Args:
        audio: Audio array, either 1D (mono) or 2D (stereo).

    Returns:
        Mono audio array.

    Raises:
        ValueError: If audio is not 1D or 2D.
    """
    if audio.ndim == 1:
        return audio
    elif audio.ndim == 2:
        return np.mean(audio, axis=1)
    else:
        raise ValueError(f"Expected 1D or 2D audio, got {audio.ndim}D")
