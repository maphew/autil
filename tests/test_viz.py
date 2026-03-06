"""Tests for viz module."""

import os
import tempfile

import numpy as np
import pytest

from autil.viz import create_visualization


class TestCreateVisualization:
    """Tests for create_visualization function."""

    def test_returns_string(self, mono_audio, sample_rate):
        """Should return the output path."""
        loudness = {
            "integrated_lufs": -20.0,
            "loudness_range_lra": 5.0,
            "moments": [
                {"time": 0.0, "lufs": -25.0},
                {"time": 1.0, "lufs": -15.0},
            ],
            "loudest": {"time": 1.0, "lufs": -15.0},
            "softest": {"time": 0.0, "lufs": -25.0},
        }

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = f.name

        try:
            result = create_visualization(
                mono_audio,
                sample_rate,
                loudness,
                [],  # silence regions
                [],  # speaker changes
                temp_path,
            )
            assert isinstance(result, str)
            assert os.path.exists(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_with_silence_regions(self, mono_audio, sample_rate):
        """Should handle silence regions."""
        loudness = {
            "integrated_lufs": -20.0,
            "loudness_range_lra": 5.0,
            "moments": [{"time": 0.0, "lufs": -20.0}],
            "loudest": None,
            "softest": None,
        }

        silence = [
            {"start": 0.0, "end": 0.5, "duration": 0.5},
        ]

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = f.name

        try:
            result = create_visualization(
                mono_audio, sample_rate, loudness, silence, [], temp_path
            )
            assert os.path.exists(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_with_speaker_changes(self, mono_audio, sample_rate):
        """Should handle speaker changes."""
        loudness = {
            "integrated_lufs": -20.0,
            "loudness_range_lra": 5.0,
            "moments": [{"time": 0.0, "lufs": -20.0}],
            "loudest": None,
            "softest": None,
        }

        speaker_changes = [
            {"time": 0.5, "confidence": 0.8, "type": "amplitude_change"},
        ]

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = f.name

        try:
            result = create_visualization(
                mono_audio, sample_rate, loudness, [], speaker_changes, temp_path
            )
            assert os.path.exists(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_with_inf_loudness(self, mono_audio, sample_rate):
        """Should handle -inf loudness values."""
        loudness = {
            "integrated_lufs": float("-inf"),
            "loudness_range_lra": 0.0,
            "moments": [
                {"time": 0.0, "lufs": float("-inf")},
                {"time": 1.0, "lufs": -20.0},
            ],
            "loudest": {"time": 1.0, "lufs": -20.0},
            "softest": {"time": 0.0, "lufs": float("-inf")},
        }

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = f.name

        try:
            result = create_visualization(
                mono_audio, sample_rate, loudness, [], [], temp_path
            )
            assert os.path.exists(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_empty_moments(self, mono_audio, sample_rate):
        """Should handle empty moments."""
        loudness = {
            "integrated_lufs": -20.0,
            "loudness_range_lra": 5.0,
            "moments": [],
            "loudest": None,
            "softest": None,
        }

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = f.name

        try:
            result = create_visualization(
                mono_audio, sample_rate, loudness, [], [], temp_path
            )
            assert os.path.exists(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_stereo_audio(self, stereo_audio, sample_rate):
        """Should handle stereo audio."""
        loudness = {
            "integrated_lufs": -20.0,
            "loudness_range_lra": 5.0,
            "moments": [{"time": 0.0, "lufs": -20.0}],
            "loudest": None,
            "softest": None,
        }

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = f.name

        try:
            result = create_visualization(
                stereo_audio, sample_rate, loudness, [], [], temp_path
            )
            assert os.path.exists(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
