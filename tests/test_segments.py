"""Tests for segments module."""

import numpy as np
import pytest

from autil.segments import detect_solo_regions, detect_speaker_changes


class TestDetectSpeakerChanges:
    """Tests for detect_speaker_changes function."""

    def test_returns_list(self, mono_audio, sample_rate):
        """Should return a list."""
        result = detect_speaker_changes(mono_audio, sample_rate)
        assert isinstance(result, list)

    def test_changes_have_time_confidence_type(self, mono_audio, sample_rate):
        """Each change should have time, confidence, type."""
        result = detect_speaker_changes(mono_audio, sample_rate)
        for change in result:
            assert "time" in change
            assert "confidence" in change
            assert "type" in change

    def test_confidence_in_range(self, mono_audio, sample_rate):
        """Confidence should be between 0 and 1."""
        result = detect_speaker_changes(mono_audio, sample_rate)
        for change in result:
            assert 0.0 <= change["confidence"] <= 1.0

    def test_custom_sensitivity(self, mono_audio, sample_rate):
        """Should accept custom sensitivity."""
        result = detect_speaker_changes(mono_audio, sample_rate, sensitivity=0.8)
        assert isinstance(result, list)

    def test_custom_segment_duration(self, mono_audio, sample_rate):
        """Should accept custom segment duration."""
        result = detect_speaker_changes(mono_audio, sample_rate, segment_duration=0.3)
        assert isinstance(result, list)

    def test_silence_audio_returns_empty(self, silence_audio, sample_rate):
        """Silence should return no speaker changes."""
        result = detect_speaker_changes(silence_audio, sample_rate)
        assert result == []

    def test_short_audio_handled(self, short_audio):
        """Short audio should be handled."""
        result = detect_speaker_changes(short_audio, 44100)
        assert isinstance(result, list)


class TestDetectSoloRegions:
    """Tests for detect_solo_regions function."""

    def test_returns_list(self, mono_audio, sample_rate):
        """Should return a list."""
        result = detect_solo_regions(mono_audio, sample_rate)
        assert isinstance(result, list)

    def test_regions_have_start_end_duration_type(self, mono_audio, sample_rate):
        """Each region should have start, end, duration, type."""
        result = detect_solo_regions(mono_audio, sample_rate)
        for region in result:
            assert "start" in region
            assert "end" in region
            assert "duration" in region
            assert "type" in region

    def test_type_is_possible_solo(self, mono_audio, sample_rate):
        """Type should be 'possible_solo'."""
        result = detect_solo_regions(mono_audio, sample_rate)
        for region in result:
            assert region["type"] == "possible_solo"

    def test_custom_sensitivity(self, mono_audio, sample_rate):
        """Should accept custom sensitivity."""
        result = detect_solo_regions(mono_audio, sample_rate, sensitivity=0.8)
        assert isinstance(result, list)

    def test_custom_segment_duration(self, mono_audio, sample_rate):
        """Should accept custom segment duration."""
        result = detect_solo_regions(mono_audio, sample_rate, segment_duration=0.1)
        assert isinstance(result, list)

    def test_silence_returns_empty(self, silence_audio, sample_rate):
        """Silence should return no solo regions."""
        result = detect_solo_regions(silence_audio, sample_rate)
        assert result == []

    def test_regions_are_sorted(self, mixed_audio, sample_rate):
        """Solo regions should be sorted by start time."""
        result = detect_solo_regions(mixed_audio, sample_rate)
        if len(result) > 1:
            for i in range(len(result) - 1):
                assert result[i]["start"] <= result[i + 1]["start"]

    def test_duration_equals_end_minus_start(self, mono_audio, sample_rate):
        """Duration should equal end - start."""
        result = detect_solo_regions(mono_audio, sample_rate)
        for region in result:
            expected_duration = region["end"] - region["start"]
            assert abs(region["duration"] - expected_duration) < 0.01

    def test_short_audio_handled(self, short_audio):
        """Short audio should be handled."""
        result = detect_solo_regions(short_audio, 44100)
        assert isinstance(result, list)
