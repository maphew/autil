# Audio Utilities

## 1. Project Overview

**autil** (Audio Utilities) is a CLI-first audio analysis toolkit written in Python. It produces an audio "fingerprint" — a JSON report and PNG visualization — containing loudness analysis, silence detection, speaker-change detection, and solo/activity region detection.

**Repository:** `https://github.com/maphew/autil.git`

### Key characteristics
- Python 3.10+, packaged with setuptools
- CLI built with Typer + Rich
- Audio decoding via soundfile (primary) with audioread/ffmpeg fallback
- Loudness per ITU-R BS.1770-4 via `pyloudnorm`
- Spectral analysis via `librosa`
- Visualization via `matplotlib`
- No tests exist yet (TDD is aspirational in the README)

---

## 2. Directory Structure

```
autil/                      ← project root
├── .gitignore
├── pyproject.toml
├── README.md
├── autil/                  ← Python package
│   ├── __init__.py
│   ├── cli.py
│   ├── analyzer.py
│   ├── audio_loader.py
│   ├── loudness.py
│   ├── silence.py
│   ├── segments.py
│   └── viz.py
└── samples/                ← sample audio files (binaries gitignored)
    ├── .gitignore
    ├── Carl Sagan - Commencement Address 1990.txt
    ├── Mussorgsky - Night on Bald Mountain (Skidmore College Orchestra).txt
    └── Smoke Alarm - Carsie Blanton.txt
```

---

## 3. Step-by-step Recreation

### 3.1 Create project root and `.gitignore`

Create `.gitignore` with these sections:

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
*.egg-info/
*.egg
dist/
build/
.eggs/
.venv/
venv/
env/

# Dolt database files (added by bd init)
.dolt/
*.db

# Audio sample binaries (reproducible via recipes in .txt sidecar files)
samples/*.mp3
samples/*.webm
samples/*.ogg
```

### 3.2 Create `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "autil"
version = "0.1.0"
description = "Audio Utilities - CLI-first audio analysis tools"
readme = "README.md"
requires-python = ">=3.10"
license = {text = "MIT"}
authors = [
    {name = "autil", email = "dev@autil.local"}
]
keywords = ["audio", "fingerprint", "loudness", "analysis"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]

dependencies = [
    "typer>=0.9.0",
    "rich>=13.0.0",
    "soundfile>=0.12.0",
    "audioread>=3.0.0",
    "librosa>=0.10.0",
    "numpy>=1.24.0",
    "pyloudnorm>=0.1.0",
    "matplotlib>=3.7.0",
    "scipy>=1.10.0",
    "pillow>=9.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "ruff>=0.1.0",
]

[project.scripts]
autil = "autil.cli:main"

[tool.setuptools.packages.find]
where = ["."]
include = ["autil*"]

[tool.ruff]
line-length = 88
target-version = "py310"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

### 3.3 Create `autil/__init__.py`

```python
"""autil - Audio Utilities

CLI-first audio analysis tools with future TUI/GUI support.
"""

__version__ = "0.1.0"
__author__ = "autil"
__license__ = "MIT"
```

### 3.4 Create `autil/audio_loader.py` — Audio Loading

This module provides three functions:

#### `load_audio(file_path: str) -> Tuple[np.ndarray, int]`
- Try `soundfile.read(file_path, dtype='float32')` first.
- On failure, fall back to `audioread.audio_open()`:
  - Read all chunks, converting bytes to int16 numpy then to float32 by dividing by 32768.0.
  - Concatenate chunks.
  - If channels > 1, reshape to `(-1, channels)`.
- Returns `(audio_data, sample_rate)`. Audio shape is `(samples,)` for mono or `(samples, channels)` for stereo.

#### `get_audio_info(file_path: str) -> dict`
- Try `soundfile.info()` first; return dict with keys: `duration`, `sample_rate`, `channels`, `codec` (= `info.format`), `frames`.
- On failure, fall back to audioread; set `codec` to `"unknown"` and compute `frames` as `int(duration * samplerate)`.

#### `to_mono(audio: np.ndarray) -> np.ndarray`
- If 1D, return as-is.
- If 2D, return `np.mean(audio, axis=1)`.
- Otherwise raise `ValueError`.

### 3.5 Create `autil/loudness.py` — Loudness Analysis (ITU-R BS.1770-4)

Uses `pyloudnorm` for measurement.

#### `analyze_loudness(audio, sample_rate, moment_width=1.0) -> dict`
1. Force mono (mean over axis=1 if needed).
2. Create `pyln.Meter(sample_rate)`.
3. Compute `integrated_loudness` from meter.
4. Compute `loudness_range` (catch exceptions, default to 0.0).
5. Call `compute_momentary_loudness()`.
6. Find loudest/softest moments by argmax/argmin of LUFS values.
7. Return dict:
   ```python
   {
       "integrated_lufs": round(integrated_loudness, 2),
       "loudness_range_lra": round(loudness_range, 2),
       "moments": [...],      # list of {"time": float, "lufs": float}
       "loudest": {...},       # or None
       "softest": {...},       # or None
   }
   ```

#### `compute_momentary_loudness(audio, sample_rate, moment_width=1.0) -> list`
- Hop size = `int(sample_rate * moment_width)`.
- Iterate through audio in non-overlapping windows of `hop_size` samples.
- Skip segments shorter than `sample_rate * 0.1`.
- Measure each segment's integrated loudness; on error use -70.0 LUFS floor.
- Return list of `{"time": round(pos/sr, 2), "lufs": round(lufs, 2)}`.

### 3.6 Create `autil/silence.py` — Silence Detection

#### `detect_silence(audio, sample_rate, threshold_db=-40.0, min_duration=0.3) -> list`
1. Force mono.
2. Use 20 ms frames with 10 ms hops.
3. Compute RMS energy per frame.
4. Convert to dB relative to max RMS: `20 * log10(rms / max(rms))`.
5. Mark frames below `threshold_db` as silent.
6. Find contiguous silent regions lasting ≥ `min_duration`.
7. Return list of `{"start": float, "end": float, "duration": float}` (rounded to 3 decimals).
8. Handle trailing silence at end of file.

### 3.7 Create `autil/segments.py` — Speaker Change & Solo Detection

#### `detect_speaker_changes(audio, sample_rate, sensitivity=0.5, segment_duration=0.5) -> list`
1. Force mono.
2. Compute mean-squared energy in non-overlapping windows of `segment_duration` seconds.
3. Normalize energies to [0, 1] by dividing by max.
4. Compute absolute differences between consecutive windows.
5. Threshold = `(1.0 - sensitivity) * 0.3 + 0.05`.
6. For each diff exceeding threshold, verify it's a real change by comparing the average energy of the 2 windows before vs. 2 windows after; accept if that difference exceeds `threshold * 0.5`.
7. Confidence = `min(1.0, diff * 2)`.
8. Return list of `{"time": float, "confidence": float, "type": "amplitude_change"}`.

#### `detect_solo_regions(audio, sample_rate, sensitivity=0.5, segment_duration=0.25) -> list`
1. Force mono.
2. Compute RMS in non-overlapping windows of `segment_duration` seconds.
3. Normalize to [0, 1].
4. Threshold = `(1.0 - sensitivity) * 0.15 + 0.02`.
5. Mark windows above threshold as "active".
6. Find contiguous active regions lasting ≥ 0.5 seconds.
7. Return list of `{"start": float, "end": float, "duration": float, "type": "possible_solo"}`.
8. Handle trailing active region.

### 3.8 Create `autil/viz.py` — Visualization

#### `create_visualization(audio, sample_rate, loudness_data, silence_regions, speaker_changes, output_path) -> str`
1. Set matplotlib backend to `'Agg'` (non-interactive) — must be done before importing `pyplot`.
2. Import `librosa.display`.
3. Force mono for display.
4. Create a `figsize=(16, 10)` figure with 3 vertically stacked subplots (3,1,N):

**Subplot 1 — Waveform:**
- Plot amplitude vs. time, `linewidth=0.3, alpha=0.7, color='steelblue'`.
- X-axis 0 to duration, Y-axis -1.05 to 1.05.
- Overlay silence regions as gray `axvspan(alpha=0.3)`.
- Overlay speaker changes as red dashed `axvline(alpha=0.5, linewidth=0.8)`.
- Title: "Waveform", grid alpha 0.3.

**Subplot 2 — Mel Spectrogram:**
- Compute `librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)`.
- Convert to dB with `librosa.power_to_db(S, ref=np.max)`.
- Display with `librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr)`.
- Add colorbar formatted as `'%+2.0f dB'`.
- Title: "Mel Spectrogram".

**Subplot 3 — Loudness:**
- If moments exist: plot LUFS over time (`linewidth=1.5, color='darkgreen'`).
  - Scatter plot loudest point (red, s=100) and softest point (blue, s=100).
  - Title includes integrated LUFS and LRA values.
  - Legend lower-right. Y limits = `[min(-70, min_lufs - 5), max(0, max_lufs + 5)]`.
- If no moments: show centered text "No loudness data available".

5. `tight_layout()`, save at `dpi=150, bbox_inches='tight'`, close figure.

### 3.9 Create `autil/analyzer.py` — Orchestrator

#### `analyze_audio(file_path, silence_threshold_db=-40.0, sensitivity=0.5, create_viz=True) -> dict`
1. `load_audio(file_path)` → `(audio, sample_rate)`.
2. `get_audio_info(file_path)` → metadata; set `file_info["file"]` to resolved absolute path.
3. `to_mono(audio)` → `audio_mono`.
4. Build results dict with: `file`, `duration_seconds` (rounded to 3), `sample_rate`, `channels`, `codec`.
5. Call `analyze_loudness(audio_mono, sample_rate)` → add as `results["loudness"]`.
6. Call `detect_silence(audio_mono, sample_rate, threshold_db=...)` → add as `results["silence"]`.
7. Call `detect_speaker_changes(audio_mono, sample_rate, sensitivity=...)` → add as `results["speaker_changes"]`.
8. Call `detect_solo_regions(audio_mono, sample_rate, sensitivity=...)` → add as `results["solo_regions"]`.
9. If `create_viz`: call `create_visualization(audio, sample_rate, loudness, silence, speakers, viz_path)` where `viz_path = input_dir / f"{stem}_fingerprint.png"`. Add as `results["visualization"]`.
10. Return results.

#### `save_results(results, output_path)`
- Write JSON with `indent=2`.

#### `format_summary(results) -> str`
- Print a human-readable report with `=` dividers, showing:
  - File, duration, sample rate, channels.
  - Loudness: integrated LUFS, LRA, loudest/softest with timestamps.
  - Silence regions count, first 5 with timestamps ("+N more" if truncated).
  - Speaker changes count, first 5 with timestamps & confidence.
  - Activity regions count, first 5 with timestamps.
  - Visualization path if present.

### 3.10 Create `autil/cli.py` — CLI Interface

Uses Typer with Rich console.

#### Global setup
- `app = typer.Typer(name="autil", help="Audio Utilities - CLI-first audio analysis tools", add_completion=False)`
- `console = Console()`

#### `version_callback(value: bool)`
- If truthy, print `autil version {__version__}` and `raise typer.Exit()`.

#### Command: `fingerprint`

```
autil fingerprint <input> [OPTIONS]
```

Options (all via `typer.Option`):
| Option | Type | Default | Description |
|---|---|---|---|
| `--output-dir, -o` | `Optional[str]` | None (=input dir) | Output directory |
| `--silence-threshold` | `float` | -40.0 | Silence threshold in dB |
| `--sensitivity` | `float` | 0.5 | Detection sensitivity 0.0-1.0 |
| `--json-only` | `bool` | False | Skip visualization image |
| `--json, -j` | `Optional[str]` | None | Custom JSON output path |
| `--verbose, -v` | `bool` | False | Show detailed output |
| `--version` | `Optional[bool]` | None | Show version (eager callback) |

Logic:
1. Validate input file exists and is a file.
2. Resolve output directory (create with `mkdir(parents=True, exist_ok=True)` if specified).
3. Default JSON path: `{output_dir}/{stem}_fingerprint.json`.
4. Validate sensitivity range [0.0, 1.0].
5. Run `analyze_audio()` inside a Rich `Progress` spinner.
6. On exception: print error (+ traceback if verbose), exit 1.
7. `save_results()` to JSON path.
8. Print `format_summary()`.
9. Always show saved file paths.

#### Command: `info`

```
autil info <input>
```

- Lazy-imports `get_audio_info` from `.audio_loader`.
- Prints: Duration, Sample Rate, Channels, Codec, Frames (comma-formatted).

#### Entry point: `main()` → `app()`

Registered as `autil = "autil.cli:main"` in `[project.scripts]`.

### 3.11 Create `samples/` Directory

#### `samples/.gitignore`
```gitignore
# Keep samples dir clean - input files only
*_fingerprint.*
```

#### Sample sidecar files

Each sample has a `.txt` file that serves as provenance documentation and download recipe. The binary audio files are gitignored — they can be reproduced from the recipes below.

---

**File: `samples/Carl Sagan - Commencement Address 1990.txt`**

Content:
```
Carl Sagan - Commencement Address
Date: May 1990
Duration: ~32 minutes
Topics: Carl Sagan, Cosmos, Space, Astronomy

Source: https://archive.org/details/CarlSagan
Uploaded by: Wilhem Nyland (February 4, 2014)

License: Public Domain Mark 1.0
License URL: http://creativecommons.org/publicdomain/mark/1.0/

This work has been identified as being free of known restrictions under
copyright law, including all related and neighboring rights. You can copy,
modify, distribute, and perform the work, even for commercial purposes,
all without asking permission.

Files:
  Carl Sagan - Commencement Address 1990.mp3  (renamed from CarlSagan.mp3)

Download:
  curl -L -o "Carl Sagan - Commencement Address 1990.mp3" "https://archive.org/download/CarlSagan/CarlSagan.mp3"
```

---

**File: `samples/Mussorgsky - Night on Bald Mountain (Skidmore College Orchestra).txt`**

Content:
```
Night on Bald Mountain (Rimsky-Korsakov's Edition, 1886)
Composer: Modest Petrovich Mussorgsky (1839-1881)
Performer: Skidmore College Orchestra
Duration: 12 min 13 s
Form: Symphonic Poem
Key: F Major
Period: Romantic

Source: https://commons.wikimedia.org/wiki/File:Modest_Mussorgsky_-_night_on_bald_mountain.ogg
Also available at: https://musopen.org/music/4874-night-on-bald-mountain/

License (composition): Public Domain
  The composition is in the public domain worldwide (author died 1881).

License (recording): Public Domain Mark 1.0
  License URL: https://creativecommons.org/publicdomain/mark/1.0/
  Released into the public domain by Musopen (http://musopen.com).

Musopen requests, out of courtesy:
  1. Do not directly sell these recordings for profit.
  2. Attribute Musopen in any commercial or derived works.

Files:
  Mussorgsky - Night on Bald Mountain (Skidmore College Orchestra).ogg  (original, renamed from Wikimedia default)
  Mussorgsky - Night on Bald Mountain (Skidmore College Orchestra).mp3  (converted from ogg)

Download and convert:
  curl -L -o "Mussorgsky - Night on Bald Mountain (Skidmore College Orchestra).ogg" "https://upload.wikimedia.org/wikipedia/commons/c/ca/Modest_Mussorgsky_-_night_on_bald_mountain.ogg"
  ffmpeg -i "Mussorgsky - Night on Bald Mountain (Skidmore College Orchestra).ogg" -vn -ab 320k -ar 44100 "Mussorgsky - Night on Bald Mountain (Skidmore College Orchestra).mp3"
```

---

**File: `samples/Smoke Alarm - Carsie Blanton.txt`**

Content:
```
Smoke Alarm
by Carsie Blanton
Album: Idiot Heart
Released: Mar 28, 2012
Genres: Pop, Folk, Americana

Source: https://www.youtube.com/watch?v=dQQ09Lxy7dI
Free Music Archive: https://freemusicarchive.org/music/Carsie_Blanton/Idiot_Heart/Smoke_Alarm/

License: Creative Commons Attribution-NonCommercial-NoDerivatives 3.0 Unported (CC BY-NC-ND 3.0)
License URL: https://creativecommons.org/licenses/by-nc-nd/3.0/

You are free to:
  Share — copy and redistribute the material in any medium or format.

Under the following terms:
  Attribution — You must give appropriate credit, provide a link to the
      license, and indicate if changes were made.
  NonCommercial — You may not use the material for commercial purposes.
  NoDerivatives — If you remix, transform, or build upon the material,
      you may not distribute the modified material.
  No additional restrictions — You may not apply legal terms or
      technological measures that legally restrict others from doing
      anything the license permits.

Files:
  Smoke Alarm - Carsie Blanton.webm  (original audio, renamed from yt-dlp default)
  Smoke Alarm - Carsie Blanton.mp3   (converted from webm)

Download and convert:
  yt-dlp -f bestaudio -o "Smoke Alarm - Carsie Blanton.%(ext)s" "https://www.youtube.com/watch?v=dQQ09Lxy7dI"
  ffmpeg -i "Smoke Alarm - Carsie Blanton.webm" -vn -ab 320k -ar 44100 "Smoke Alarm - Carsie Blanton.mp3"
```

### 3.12 Create `README.md`

Write the README documenting:
- Quick start (clone, `uv venv`, `uv pip install -e .`, run)
- Requirements: Python 3.10+, uv, ffmpeg
- Two commands: `fingerprint` and `info`
- `fingerprint` options table (output-dir, silence-threshold, sensitivity, json-only, json, verbose)
- Output: `*_fingerprint.json` and `*_fingerprint.png`
- Features list: Loudness (LUFS, LRA, moments, loudest/softest), Silence detection, Speaker change detection, Solo/activity detection, Visualization (waveform, spectrogram, loudness)
- Example JSON output format showing all top-level keys and nested structures
- Development section mentioning TDD and pytest commands
- MIT License

---

## 4. System Dependencies

The host system needs:
- **Python 3.10+**
- **ffmpeg** — required by audioread for decoding MP3, OGG, WebM, etc.
- **libsndfile** — required by the `soundfile` Python package
- **yt-dlp** — only needed to download the Smoke Alarm sample

Install on Ubuntu/Debian:
```bash
sudo apt-get install -y ffmpeg libsndfile1
```

---

## 5. Build & Run

```bash
# Create virtual environment
uv venv .venv
source .venv/bin/activate

# Install in editable mode
uv pip install -e .

# Download sample audio (from samples/ directory)
cd samples
curl -L -o "Carl Sagan - Commencement Address 1990.mp3" "https://archive.org/download/CarlSagan/CarlSagan.mp3"
curl -L -o "Mussorgsky - Night on Bald Mountain (Skidmore College Orchestra).ogg" "https://upload.wikimedia.org/wikipedia/commons/c/ca/Modest_Mussorgsky_-_night_on_bald_mountain.ogg"
ffmpeg -i "Mussorgsky - Night on Bald Mountain (Skidmore College Orchestra).ogg" -vn -ab 320k -ar 44100 "Mussorgsky - Night on Bald Mountain (Skidmore College Orchestra).mp3"
cd ..

# Run
autil fingerprint "samples/Carl Sagan - Commencement Address 1990.mp3"
autil info "samples/Mussorgsky - Night on Bald Mountain (Skidmore College Orchestra).mp3"
```

---

## 6. Architecture Diagram

```
CLI (Typer)
  └── analyzer.py  (orchestrator)
        ├── audio_loader.py   load_audio(), get_audio_info(), to_mono()
        ├── loudness.py       analyze_loudness()  [pyloudnorm]
        ├── silence.py        detect_silence()     [numpy]
        ├── segments.py       detect_speaker_changes(), detect_solo_regions()  [numpy]
        └── viz.py            create_visualization()  [matplotlib, librosa]
```

Data flows top-down: CLI validates input → analyzer loads audio once → passes mono audio to each analysis module → collects results into a dict → saves JSON + PNG → prints summary.

---

## 7. Design Notes

- **Dual-backend audio loading**: soundfile is tried first (fast, native); audioread+ffmpeg is the fallback for formats soundfile can't handle (MP3, etc.).
- **All analysis modules accept mono numpy arrays** and sample rate. The `to_mono()` conversion happens once in `analyzer.py`.
- **Sensitivity parameter** (0.0–1.0) controls thresholds inversely: higher sensitivity → lower threshold → more detections.
- **Loudness range (`loudness_range`)** call is wrapped in try/except because pyloudnorm can fail on very short or silent audio.
- **Momentary loudness** uses non-overlapping windows (not sliding); segments shorter than 0.1s are skipped.
- **Silence detection** uses 20ms frames with 10ms hops; dB is computed relative to the audio's own max RMS (not absolute).
- **Speaker change detection** uses a two-stage filter: first checks if the energy difference between adjacent windows exceeds a threshold, then verifies the 2-window averages before/after are also sufficiently different.
- **Solo region detection** requires minimum 0.5s duration to be reported.
- **Visualization** uses `matplotlib.use('Agg')` before importing pyplot to avoid display server requirements.
- **`verbose` flag** currently always shows file paths (`if verbose or True`); the guard is present for future refinement.

---

## 8. Verification Checklist

After recreating, verify:

1. `autil --version` prints `autil version 0.1.0`
2. `autil info <any-sample.mp3>` prints duration, sample rate, channels, codec, frames
3. `autil fingerprint <sample.mp3>` produces:
   - `<sample>_fingerprint.json` with keys: `file`, `duration_seconds`, `sample_rate`, `channels`, `codec`, `loudness`, `silence`, `speaker_changes`, `solo_regions`, `visualization`
   - `<sample>_fingerprint.png` with 3 subplot panels (waveform, spectrogram, loudness)
4. `autil fingerprint --json-only <sample.mp3>` skips PNG creation
5. `autil fingerprint -o /tmp <sample.mp3>` writes outputs to `/tmp/`
6. Non-existent input file prints red error and exits 1
7. All three sample `.txt` sidecar files are present in `samples/`
8. Sample download recipes in `.txt` files work and produce playable audio
