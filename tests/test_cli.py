"""Tests for CLI module."""

import os
import tempfile

import pytest
from typer.testing import CliRunner

from autil.cli import app

runner = CliRunner()


class TestVersion:
    """Tests for version option."""

    def test_version_flag(self):
        """--version should print version."""
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "autil version" in result.stdout

    def test_fingerprint_version_flag(self):
        """fingerprint --version should print version."""
        result = runner.invoke(
            app,
            ["fingerprint", "--version", "samples/Smoke Alarm - Carsie Blanton.mp3"],
        )
        assert result.exit_code == 0
        assert "autil version" in result.stdout

    def test_info_version_flag(self):
        """info --version should print version."""
        result = runner.invoke(
            app, ["info", "--version", "samples/Smoke Alarm - Carsie Blanton.mp3"]
        )
        assert result.exit_code == 0
        assert "autil version" in result.stdout


class TestInfoCommand:
    """Tests for info command."""

    def test_info_valid_file(self):
        """info should display file info."""
        result = runner.invoke(
            app, ["info", "samples/Smoke Alarm - Carsie Blanton.mp3"]
        )
        assert result.exit_code == 0
        assert "Duration:" in result.stdout
        assert "Sample Rate:" in result.stdout
        assert "Channels:" in result.stdout
        assert "Codec:" in result.stdout
        assert "Frames:" in result.stdout

    def test_info_invalid_file(self):
        """info should error on invalid file."""
        result = runner.invoke(app, ["info", "nonexistent.mp3"])
        assert result.exit_code == 1
        assert "Error" in result.stdout


class TestFingerprintCommand:
    """Tests for fingerprint command."""

    def test_fingerprint_valid_file(self):
        """fingerprint should analyze file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(
                app,
                [
                    "fingerprint",
                    "--json-only",
                    "-o",
                    tmpdir,
                    "samples/Smoke Alarm - Carsie Blanton.mp3",
                ],
            )
            assert result.exit_code == 0
            assert "Loudness:" in result.stdout
            assert "Silence:" in result.stdout

    def test_fingerprint_invalid_file(self):
        """fingerprint should error on invalid file."""
        result = runner.invoke(app, ["fingerprint", "nonexistent.mp3"])
        assert result.exit_code == 1
        assert "Error" in result.stdout

    def test_fingerprint_json_only(self):
        """--json-only should skip PNG creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(
                app,
                [
                    "fingerprint",
                    "--json-only",
                    "-o",
                    tmpdir,
                    "samples/Smoke Alarm - Carsie Blanton.mp3",
                ],
            )
            assert result.exit_code == 0

    def test_fingerprint_custom_json_path(self):
        """--json should allow custom output path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(
                app,
                [
                    "fingerprint",
                    "--json",
                    os.path.join(tmpdir, "custom.json"),
                    "samples/Smoke Alarm - Carsie Blanton.mp3",
                ],
            )
            assert result.exit_code == 0
            assert os.path.exists(os.path.join(tmpdir, "custom.json"))

    def test_fingerprint_sensitivity(self):
        """--sensitivity should accept value."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(
                app,
                [
                    "fingerprint",
                    "--json-only",
                    "--sensitivity",
                    "0.8",
                    "-o",
                    tmpdir,
                    "samples/Smoke Alarm - Carsie Blanton.mp3",
                ],
            )
            assert result.exit_code == 0

    def test_fingerprint_silence_threshold(self):
        """--silence-threshold should accept value."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(
                app,
                [
                    "fingerprint",
                    "--json-only",
                    "--silence-threshold",
                    "-50.0",
                    "-o",
                    tmpdir,
                    "samples/Smoke Alarm - Carsie Blanton.mp3",
                ],
            )
            assert result.exit_code == 0

    def test_fingerprint_verbose(self):
        """--verbose should show details."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(
                app,
                [
                    "fingerprint",
                    "--json-only",
                    "--verbose",
                    "-o",
                    tmpdir,
                    "samples/Smoke Alarm - Carsie Blanton.mp3",
                ],
            )
            assert result.exit_code == 0


class TestHelp:
    """Tests for help output."""

    def test_help(self):
        """--help should show help."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "fingerprint" in result.stdout
        assert "fingerprint" in result.stdout
        assert "info" in result.stdout

    def test_fingerprint_help(self):
        """fingerprint --help should show fingerprint help."""
        result = runner.invoke(app, ["fingerprint", "--help"])
        assert result.exit_code == 0
        assert "--output-dir" in result.stdout
        assert "--silence-threshold" in result.stdout
        assert "--sensitivity" in result.stdout
