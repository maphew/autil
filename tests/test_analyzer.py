"""Tests for analyzer module."""

import json
import os
import tempfile

import numpy as np
import pytest

from autil.analyzer import analyze_audio, format_summary, save_results


class TestAnalyzeAudio:
    """Tests for analyze_audio function."""

    def test_returns_dict(self):
        """Should return a dictionary."""
        result = analyze_audio(
            "samples/Smoke Alarm - Carsie Blanton.mp3", create_viz=False
        )
        assert isinstance(result, dict)

    def test_has_required_keys(self):
        """Should have all required keys."""
        result = analyze_audio(
            "samples/Smoke Alarm - Carsie Blanton.mp3", create_viz=False
        )
        assert "file" in result
        assert "duration_seconds" in result
        assert "sample_rate" in result
        assert "channels" in result
        assert "codec" in result
        assert "loudness" in result
        assert "silence" in result
        assert "speaker_changes" in result
        assert "solo_regions" in result

    def test_loudness_has_required_keys(self):
        """Loudness should have required keys."""
        result = analyze_audio(
            "samples/Smoke Alarm - Carsie Blanton.mp3", create_viz=False
        )
        assert "integrated_lufs" in result["loudness"]
        assert "loudness_range_lra" in result["loudness"]
        assert "moments" in result["loudness"]
        assert "loudest" in result["loudness"]
        assert "softest" in result["loudness"]

    def test_silence_is_list(self):
        """Silence should be a list."""
        result = analyze_audio(
            "samples/Smoke Alarm - Carsie Blanton.mp3", create_viz=False
        )
        assert isinstance(result["silence"], list)

    def test_speaker_changes_is_list(self):
        """Speaker changes should be a list."""
        result = analyze_audio(
            "samples/Smoke Alarm - Carsie Blanton.mp3", create_viz=False
        )
        assert isinstance(result["speaker_changes"], list)

    def test_solo_regions_is_list(self):
        """Solo regions should be a list."""
        result = analyze_audio(
            "samples/Smoke Alarm - Carsie Blanton.mp3", create_viz=False
        )
        assert isinstance(result["solo_regions"], list)

    def test_create_viz_false(self):
        """Should not create visualization when create_viz=False."""
        result = analyze_audio(
            "samples/Smoke Alarm - Carsie Blanton.mp3", create_viz=False
        )
        assert "visualization" not in result

    def test_custom_silence_threshold(self):
        """Should accept custom silence threshold."""
        result = analyze_audio(
            "samples/Smoke Alarm - Carsie Blanton.mp3",
            silence_threshold_db=-50.0,
            create_viz=False,
        )
        assert isinstance(result, dict)

    def test_custom_sensitivity(self):
        """Should accept custom sensitivity."""
        result = analyze_audio(
            "samples/Smoke Alarm - Carsie Blanton.mp3",
            sensitivity=0.8,
            create_viz=False,
        )
        assert isinstance(result, dict)


class TestSaveResults:
    """Tests for save_results function."""

    def test_saves_json_file(self):
        """Should save results to JSON file."""
        results = {
            "file": "/test/file.mp3",
            "duration_seconds": 10.0,
            "loudness": {"integrated_lufs": -20.0},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            save_results(results, temp_path)
            assert os.path.exists(temp_path)

            with open(temp_path) as f:
                loaded = json.load(f)
            assert loaded["file"] == "/test/file.mp3"
            assert loaded["duration_seconds"] == 10.0
        finally:
            os.unlink(temp_path)

    def test_handles_numpy_types(self):
        """Should handle numpy types in results."""
        results = {
            "duration_seconds": np.float64(10.0),
            "sample_rate": np.int32(44100),
            "loudness": {
                "integrated_lufs": np.float64(-20.0),
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            save_results(results, temp_path)

            with open(temp_path) as f:
                loaded = json.load(f)
            assert isinstance(loaded["duration_seconds"], float)
            assert isinstance(loaded["sample_rate"], int)
            assert isinstance(loaded["loudness"]["integrated_lufs"], float)
        finally:
            os.unlink(temp_path)


class TestFormatSummary:
    """Tests for format_summary function."""

    def test_returns_string(self):
        """Should return a string."""
        results = {
            "file": "/test/file.mp3",
            "duration_seconds": 10.0,
            "sample_rate": 44100,
            "channels": 2,
            "loudness": {
                "integrated_lufs": -20.0,
                "loudness_range_lra": 5.0,
                "loudest": {"time": 5.0, "lufs": -15.0},
                "softest": {"time": 1.0, "lufs": -30.0},
            },
            "silence": [{"start": 0.0, "end": 1.0, "duration": 1.0}],
            "speaker_changes": [],
            "solo_regions": [],
        }

        summary = format_summary(results)
        assert isinstance(summary, str)
        assert "/test/file.mp3" in summary
        assert "10.0s" in summary
        assert "44100 Hz" in summary

    def test_includes_loudness(self):
        """Should include loudness info."""
        results = {
            "file": "/test/file.mp3",
            "duration_seconds": 10.0,
            "sample_rate": 44100,
            "channels": 2,
            "loudness": {
                "integrated_lufs": -20.0,
                "loudness_range_lra": 5.0,
                "loudest": None,
                "softest": None,
            },
            "silence": [],
            "speaker_changes": [],
            "solo_regions": [],
        }

        summary = format_summary(results)
        assert "Loudness:" in summary
        assert "-20.0 LUFS" in summary

    def test_includes_silence_count(self):
        """Should include silence region count."""
        results = {
            "file": "/test/file.mp3",
            "duration_seconds": 10.0,
            "sample_rate": 44100,
            "channels": 2,
            "loudness": {
                "integrated_lufs": -20.0,
                "loudness_range_lra": 5.0,
                "loudest": None,
                "softest": None,
            },
            "silence": [
                {"start": 0.0, "end": 1.0, "duration": 1.0},
                {"start": 5.0, "end": 6.0, "duration": 1.0},
            ],
            "speaker_changes": [],
            "solo_regions": [],
        }

        summary = format_summary(results)
        assert "Silence: 2 regions" in summary

    def test_includes_visualization_path(self):
        """Should include visualization path if present."""
        results = {
            "file": "/test/file.mp3",
            "duration_seconds": 10.0,
            "sample_rate": 44100,
            "channels": 2,
            "loudness": {
                "integrated_lufs": -20.0,
                "loudness_range_lra": 5.0,
                "loudest": None,
                "softest": None,
            },
            "silence": [],
            "speaker_changes": [],
            "solo_regions": [],
            "visualization": "/test/file_fingerprint.png",
        }

        summary = format_summary(results)
        assert "/test/file_fingerprint.png" in summary
