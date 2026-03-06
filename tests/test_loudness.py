"""Tests for loudness module."""

import numpy as np
import pytest

from autil.loudness import analyze_loudness, compute_momentary_loudness


class TestAnalyzeLoudness:
    """Tests for analyze_loudness function."""

    def test_returns_dict(self, mono_audio, sample_rate):
        """Should return a dictionary."""
        result = analyze_loudness(mono_audio, sample_rate)
        assert isinstance(result, dict)

    def test_has_required_keys(self, mono_audio, sample_rate):
        """Should have all required keys."""
        result = analyze_loudness(mono_audio, sample_rate)
        assert "integrated_lufs" in result
        assert "loudness_range_lra" in result
        assert "moments" in result
        assert "loudest" in result
        assert "softest" in result

    def test_integrated_lufs_is_float(self, mono_audio, sample_rate):
        """Integrated LUFS should be a float."""
        result = analyze_loudness(mono_audio, sample_rate)
        assert isinstance(result["integrated_lufs"], float)

    def test_loudness_range_is_float(self, mono_audio, sample_rate):
        """LRA should be a float."""
        result = analyze_loudness(mono_audio, sample_rate)
        assert isinstance(result["loudness_range_lra"], float)

    def test_loudest_has_time_and_lufs(self, mono_audio, sample_rate):
        """Loudest should have time and lufs keys."""
        result = analyze_loudness(mono_audio, sample_rate)
        if result["loudest"]:
            assert "time" in result["loudest"]
            assert "lufs" in result["loudest"]

    def test_softest_has_time_and_lufs(self, mono_audio, sample_rate):
        """Softest should have time and lufs keys."""
        result = analyze_loudness(mono_audio, sample_rate)
        if result["softest"]:
            assert "time" in result["softest"]
            assert "lufs" in result["softest"]

    def test_moments_is_list(self, mono_audio, sample_rate):
        """Moments should be a list."""
        result = analyze_loudness(mono_audio, sample_rate)
        assert isinstance(result["moments"], list)

    def test_moments_have_time_and_lufs(self, mono_audio, sample_rate):
        """Each moment should have time and lufs."""
        result = analyze_loudness(mono_audio, sample_rate)
        for moment in result["moments"]:
            assert "time" in moment
            assert "lufs" in moment

    def test_silence_audio_handled(self, silence_audio, sample_rate):
        """Silence should be handled without error."""
        result = analyze_loudness(silence_audio, sample_rate)
        assert isinstance(result, dict)
        # Silence typically results in -inf LUFS
        assert result["integrated_lufs"] <= 0


class TestComputeMomentaryLoudness:
    """Tests for compute_momentary_loudness function."""

    def test_returns_list(self, mono_audio, sample_rate):
        """Should return a list."""
        result = compute_momentary_loudness(mono_audio, sample_rate)
        assert isinstance(result, list)

    def test_moments_have_time_and_lufs(self, mono_audio, sample_rate):
        """Each moment should have time and lufs keys."""
        result = compute_momentary_loudness(mono_audio, sample_rate)
        for moment in result:
            assert "time" in moment
            assert "lufs" in moment

    def test_custom_moment_width(self, mono_audio, sample_rate):
        """Should accept custom moment width."""
        result = compute_momentary_loudness(mono_audio, sample_rate, moment_width=0.5)
        assert isinstance(result, list)

    def test_short_audio_handled(self, short_audio):
        """Short audio should be handled."""
        result = compute_momentary_loudness(short_audio, 44100)
        assert isinstance(result, list)
