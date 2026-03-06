# Audio Utilities

**autil** (Audio Utilities) is a CLI-first audio analysis toolkit written in Python. It produces an audio "fingerprint" — a JSON report and PNG visualization — containing loudness analysis, silence detection, speaker-change detection, and solo/activity region detection.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/maphew/autil.git
cd autil

# Create virtual environment
uv venv .venv
source .venv/bin/activate

# Install in editable mode
uv pip install -e .

# Run
autil fingerprint "samples/Carl Sagan - Commencement Address 1990.mp3"
autil info "samples/Mussorgsky - Night on Bald Mountain (Skidmore College Orchestra).mp3"
```

## Requirements

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager
- **ffmpeg** — required for audio decoding
- **libsndfile** — required by the `soundfile` Python package

Install system dependencies on Ubuntu/Debian:
```bash
sudo apt-get install -y ffmpeg libsndfile1
```

## Commands

### `autil fingerprint`

Analyze an audio file and produce a JSON report and PNG visualization.

```
autil fingerprint <input> [OPTIONS]
```

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--output-dir, -o` | `Optional[str]` | None (=input dir) | Output directory |
| `--silence-threshold` | `float` | -40.0 | Silence threshold in dB |
| `--sensitivity` | `float` | 0.5 | Detection sensitivity 0.0-1.0 |
| `--json-only` | `bool` | False | Skip visualization image |
| `--json, -j` | `Optional[str]` | None | Custom JSON output path |
| `--verbose, -v` | `bool` | False | Show detailed output |

**Output:**

- `<sample>_fingerprint.json` — JSON report with all analysis data
- `<sample>_fingerprint.png` — Visualization with waveform, spectrogram, and loudness

### `autil info`

Show basic audio file information.

```
autil info <input>
```

## Features

- **Loudness Analysis (ITU-R BS.1770-4)**
  - Integrated loudness (LUFS)
  - Loudness range (LRA)
  - Momentary loudness over time
  - Loudest and softest points

- **Silence Detection**
  - Configurable threshold and minimum duration
  - Returns start, end, and duration of each region

- **Speaker Change Detection**
  - Energy-based change detection
  - Confidence scores for each change

- **Solo/Activity Detection**
  - Identifies active regions above threshold
  - Minimum duration filtering

- **Visualization**
  - Waveform with silence and speaker change overlays
  - Mel spectrogram
  - Loudness graph with loudest/softest markers

## Example Output

### JSON Structure

```json
{
  "file": "/path/to/audio.mp3",
  "duration_seconds": 180.5,
  "sample_rate": 44100,
  "channels": 2,
  "codec": "MP3",
  "loudness": {
    "integrated_lufs": -20.5,
    "loudness_range_lra": 12.3,
    "moments": [
      {"time": 0.0, "lufs": -25.1},
      {"time": 1.0, "lufs": -22.3}
    ],
    "loudest": {"time": 45.2, "lufs": -10.5},
    "softest": {"time": 120.0, "lufs": -45.2}
  },
  "silence": [
    {"start": 0.0, "end": 2.5, "duration": 2.5}
  ],
  "speaker_changes": [
    {"time": 30.5, "confidence": 0.85, "type": "amplitude_change"}
  ],
  "solo_regions": [
    {"start": 5.0, "end": 15.2, "duration": 10.2, "type": "possible_solo"}
  ],
  "visualization": "/path/to/audio_fingerprint.png"
}
```

## Development

This project uses pytest for testing with TDD.

```bash
# Install dev dependencies
uv pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=autil
```

## History

2026-03-06: Initial idea prototyped with assistance from Ampcode. That experience used to write a plan. The prototype was thrown out and the plan given to Kilocode CLI to generate the current codebase. Kilo was used in Orchestrator mode with 'kilo-auto: Free' model provider.

## License

Copyright 2026 Matt Wilkie <maphew@gmail.com> MIT License
