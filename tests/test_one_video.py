"""
tests/test_one_video.py — Smoke test for step 01 (audio extraction)

Requires: tests/sample_clip.mp4 — a short video clip provided by the researcher.
This test is intentionally excluded from the default `pytest tests/` run:

    pytest tests/ -v --ignore=tests/test_one_video.py   # unit tests only (fast)
    pytest tests/test_one_video.py -v                    # smoke test (requires sample_clip.mp4)

Steps 02–09 are NOT exercised here (too slow; require model downloads and HF_TOKEN).
"""

import subprocess
import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SAMPLE_CLIP = Path(__file__).parent / "sample_clip.mp4"


# ── Skip the entire module if sample_clip.mp4 is not present ─────────────────

pytestmark = pytest.mark.skipif(
    not SAMPLE_CLIP.exists(),
    reason="tests/sample_clip.mp4 not found — provide a short clip to run the smoke test",
)


# ── ffmpeg availability ───────────────────────────────────────────────────────

def test_ffmpeg_is_callable():
    """ffmpeg is installed and reachable on PATH."""
    result = subprocess.run(
        ["ffmpeg", "-version"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"ffmpeg -version failed:\n{result.stderr}"
    assert "ffmpeg version" in result.stdout.lower() or "ffmpeg version" in result.stderr.lower()


# ── Audio extraction smoke test ───────────────────────────────────────────────

def test_extract_audio_creates_wav(tmp_path):
    """
    Step 01 smoke test: extract_audio.run() produces a non-empty WAV file
    from sample_clip.mp4 at 16 kHz mono.
    """
    import config
    from pipeline.extract_audio import run as extract_run

    # Temporarily redirect outputs to tmp_path
    # extract_audio.run() places the WAV at AUDIO_DIR/<vid>.wav and creates
    # a symlink at PER_VIDEO_DIR/<vid>/audio.wav
    # We call the function and verify it exits without raising.
    # The actual WAV location depends on config paths — we patch them.

    import pipeline.extract_audio as ea

    original_audio_dir    = config.AUDIO_DIR
    original_per_video    = config.PER_VIDEO_DIR

    # Redirect to tmp_path so we don't pollute real outputs
    config.AUDIO_DIR     = tmp_path / "audio"
    config.PER_VIDEO_DIR = tmp_path / "per_video"
    config.AUDIO_DIR.mkdir(parents=True)
    config.PER_VIDEO_DIR.mkdir(parents=True)

    try:
        result_path = extract_run(SAMPLE_CLIP, force=False)
        assert result_path is not None, "extract_audio.run() returned None"
        assert result_path.exists(), f"Expected WAV at {result_path} but file not found"
        assert result_path.suffix == ".wav", f"Expected .wav suffix, got {result_path.suffix}"
        assert result_path.stat().st_size > 0, "WAV file is empty"
    finally:
        # Restore config
        config.AUDIO_DIR     = original_audio_dir
        config.PER_VIDEO_DIR = original_per_video


def test_extract_audio_sample_rate(tmp_path):
    """
    Extracted WAV has the correct sample rate (16 000 Hz) and is mono.
    """
    import soundfile as sf
    import config
    from pipeline.extract_audio import run as extract_run

    original_audio_dir    = config.AUDIO_DIR
    original_per_video    = config.PER_VIDEO_DIR

    config.AUDIO_DIR     = tmp_path / "audio"
    config.PER_VIDEO_DIR = tmp_path / "per_video"
    config.AUDIO_DIR.mkdir(parents=True)
    config.PER_VIDEO_DIR.mkdir(parents=True)

    try:
        result_path = extract_run(SAMPLE_CLIP, force=False)
        info = sf.info(str(result_path))
        assert info.samplerate == config.AUDIO_SAMPLE_RATE, (
            f"Expected {config.AUDIO_SAMPLE_RATE} Hz, got {info.samplerate} Hz"
        )
        assert info.channels == config.AUDIO_CHANNELS, (
            f"Expected {config.AUDIO_CHANNELS} channel(s), got {info.channels}"
        )
    finally:
        config.AUDIO_DIR     = original_audio_dir
        config.PER_VIDEO_DIR = original_per_video


def test_extract_audio_cached_on_second_call(tmp_path):
    """
    Calling extract_audio.run() twice without force=True skips re-extraction
    (idempotent / cached).
    """
    import config
    from pipeline.extract_audio import run as extract_run
    from pipeline.cache import is_cached

    original_audio_dir    = config.AUDIO_DIR
    original_per_video    = config.PER_VIDEO_DIR

    config.AUDIO_DIR     = tmp_path / "audio"
    config.PER_VIDEO_DIR = tmp_path / "per_video"
    config.AUDIO_DIR.mkdir(parents=True)
    config.PER_VIDEO_DIR.mkdir(parents=True)

    try:
        path1 = extract_run(SAMPLE_CLIP, force=False)
        mtime1 = path1.stat().st_mtime

        path2 = extract_run(SAMPLE_CLIP, force=False)
        mtime2 = path2.stat().st_mtime

        assert path1 == path2, "Second call returned a different path"
        assert mtime1 == mtime2, "File was re-written on cached call (should have been skipped)"
    finally:
        config.AUDIO_DIR     = original_audio_dir
        config.PER_VIDEO_DIR = original_per_video
