"""Tests for audio_loader module."""

import numpy as np
import pytest

from autil.audio_loader import get_audio_info, load_audio, to_mono


class TestToMono:
    """Tests for to_mono function."""

    def test_mono_audio_returns_as_is(self, mono_audio):
        """Mono audio should be returned unchanged."""
        result = to_mono(mono_audio)
        assert result.shape == mono_audio.shape
        np.testing.assert_array_equal(result, mono_audio)

    def test_stereo_to_mono(self, stereo_audio):
        """Stereo audio should be converted to mono by averaging channels."""
        result = to_mono(stereo_audio)
        expected = np.mean(stereo_audio, axis=1)
        np.testing.assert_array_almost_equal(result, expected)

    def test_mono_1d_array(self):
        """1D array should work."""
        audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        result = to_mono(audio)
        np.testing.assert_array_equal(result, audio)

    def test_stereo_2d_array(self):
        """2D array (channels, samples) should be averaged."""
        audio = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32).T
        result = to_mono(audio)
        expected = np.array([2.0, 3.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_invalid_dimensions_raises(self):
        """3D array should raise ValueError."""
        audio = np.ones((2, 2, 2), dtype=np.float32)
        with pytest.raises(ValueError, match="Expected 1D or 2D"):
            to_mono(audio)


class TestLoadAudio:
    """Tests for load_audio function."""

    def test_load_mp3_returns_array(self):
        """Loading an MP3 should return numpy array and sample rate."""
        audio, sr = load_audio("samples/Smoke Alarm - Carsie Blanton.mp3")
        assert isinstance(audio, np.ndarray)
        assert sr == 44100

    def test_load_audio_shape(self):
        """Loaded audio should have correct shape."""
        audio, sr = load_audio("samples/Smoke Alarm - Carsie Blanton.mp3")
        # Should be mono after our implementation processes it
        assert audio.ndim in (1, 2)


class TestGetAudioInfo:
    """Tests for get_audio_info function."""

    def test_get_info_returns_dict(self):
        """Should return dict with required keys."""
        info = get_audio_info("samples/Smoke Alarm - Carsie Blanton.mp3")
        assert isinstance(info, dict)
        assert "duration" in info
        assert "sample_rate" in info
        assert "channels" in info
        assert "codec" in info
        assert "frames" in info

    def test_get_info_duration_format(self):
        """Duration should be float."""
        info = get_audio_info("samples/Smoke Alarm - Carsie Blanton.mp3")
        assert isinstance(info["duration"], float)
        assert info["duration"] > 0

    def test_get_info_sample_rate(self):
        """Sample rate should be positive int."""
        info = get_audio_info("samples/Smoke Alarm - Carsie Blanton.mp3")
        assert isinstance(info["sample_rate"], int)
        assert info["sample_rate"] > 0

    def test_get_info_channels(self):
        """Channels should be positive int."""
        info = get_audio_info("samples/Smoke Alarm - Carsie Blanton.mp3")
        assert isinstance(info["channels"], int)
        assert info["channels"] > 0

    def test_get_info_frames(self):
        """Frames should be positive int."""
        info = get_audio_info("samples/Smoke Alarm - Carsie Blanton.mp3")
        assert isinstance(info["frames"], int)
        assert info["frames"] > 0
