"""Pytest fixtures for autil tests."""

import numpy as np
import pytest


@pytest.fixture
def sample_rate():
    """Standard sample rate for tests."""
    return 44100


@pytest.fixture
def mono_audio(sample_rate):
    """Generate a 1-second mono audio sine wave at 440Hz."""
    t = np.linspace(0, 1, sample_rate, dtype=np.float32)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return audio


@pytest.fixture
def stereo_audio(sample_rate):
    """Generate a 1-second stereo audio sine wave at 440Hz."""
    t = np.linspace(0, 1, sample_rate, dtype=np.float32)
    left = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    right = np.sin(2 * np.pi * 880 * t).astype(np.float32)
    return np.column_stack([left, right]).astype(np.float32)


@pytest.fixture
def silence_audio(sample_rate):
    """Generate 1 second of silence."""
    return np.zeros(sample_rate, dtype=np.float32)


@pytest.fixture
def mixed_audio(sample_rate):
    """Generate audio with silence at start, middle, and end.

    Structure: [silent, loud, silent, loud, silent]
    Each segment: 0.2 seconds
    """
    segment_samples = int(0.2 * sample_rate)
    silence = np.zeros(segment_samples, dtype=np.float32)

    t = np.linspace(0, 0.2, segment_samples, dtype=np.float32)
    loud = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.8

    audio = np.concatenate([silence, loud, silence, loud, silence])
    return audio


@pytest.fixture
def short_audio():
    """Very short audio (10 samples) for edge case testing."""
    return np.array(
        [0.1, 0.2, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2, -0.1, 0.0], dtype=np.float32
    )
