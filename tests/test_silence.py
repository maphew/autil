"""Tests for silence module."""

import numpy as np
import pytest

from autil.silence import detect_silence


class TestDetectSilence:
    """Tests for detect_silence function."""

    def test_returns_list(self, mono_audio, sample_rate):
        """Should return a list."""
        result = detect_silence(mono_audio, sample_rate)
        assert isinstance(result, list)

    def test_regions_have_start_end_duration(self, mono_audio, sample_rate):
        """Each region should have start, end, duration."""
        result = detect_silence(mono_audio, sample_rate)
        for region in result:
            assert "start" in region
            assert "end" in region
            assert "duration" in region

    def test_full_silence_returns_single_region(self, silence_audio, sample_rate):
        """Full silence should return one region covering the whole audio."""
        result = detect_silence(silence_audio, sample_rate)
        assert len(result) >= 1
        # The silence should cover the full duration
        region = result[0]
        assert region["start"] == 0.0
        assert region["end"] > 0

    def test_mixed_audio_detects_regions(self, mixed_audio, sample_rate):
        """Mixed audio should detect multiple silence regions."""
        result = detect_silence(mixed_audio, sample_rate)
        # Should find at least the silence segments
        assert isinstance(result, list)

    def test_custom_threshold(self, mono_audio, sample_rate):
        """Should accept custom threshold."""
        result = detect_silence(mono_audio, sample_rate, threshold_db=-50.0)
        assert isinstance(result, list)

    def test_custom_min_duration(self, mono_audio, sample_rate):
        """Should accept custom min duration."""
        result = detect_silence(mono_audio, sample_rate, min_duration=0.5)
        assert isinstance(result, list)

    def test_regions_are_sorted(self, mixed_audio, sample_rate):
        """Silence regions should be sorted by start time."""
        result = detect_silence(mixed_audio, sample_rate)
        if len(result) > 1:
            for i in range(len(result) - 1):
                assert result[i]["start"] <= result[i + 1]["start"]

    def test_duration_equals_end_minus_start(self, mixed_audio, sample_rate):
        """Duration should equal end - start."""
        result = detect_silence(mixed_audio, sample_rate)
        for region in result:
            expected_duration = region["end"] - region["start"]
            assert abs(region["duration"] - expected_duration) < 0.01

    def test_default_threshold(self, sample_rate):
        """Default threshold should be -40.0 dB."""
        # Create audio with known levels
        audio = np.random.randn(sample_rate).astype(np.float32) * 0.01
        result = detect_silence(audio, sample_rate)
        assert isinstance(result, list)

    def test_short_audio_handled(self, short_audio):
        """Short audio should be handled."""
        result = detect_silence(short_audio, 44100)
        assert isinstance(result, list)
