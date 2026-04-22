"""
pipeline/extract_audio.py — Step 01: Audio extraction

Extracts a 16 kHz mono WAV from each source MP4 using ffmpeg.
This format is required by all downstream audio models (Whisper, pyannote,
Parselmouth). The source video in DATA/ is never modified.

Output: data/audio/<video_id>.wav
        outputs/per_video/<video_id>/audio.wav  (symlink for step scripts)

Caching: skipped if audio.wav already exists and is non-empty.
Force:   use --force flag in run_all.py to re-extract.
"""

from __future__ import annotations

from pathlib import Path

import ffmpeg
from tqdm import tqdm

import config
from pipeline.cache import (
    get_video_output_dir,
    should_skip,
    video_id_from_path,
)
from pipeline.logger import get_logger

log = get_logger(__name__)


def _extract_single(video_path: Path, wav_path: Path, force: bool) -> Path:
    """
    Extract audio from one video file to a WAV file.

    Args:
        video_path: Source MP4 file.
        wav_path:   Destination WAV file path.
        force:      If True, overwrite existing file.

    Returns:
        Path to the created WAV file.

    Raises:
        ffmpeg.Error: If ffmpeg fails.
        FileNotFoundError: If the source video does not exist.
    """
    if not video_path.exists():
        log.error("Source video not found: %s", video_path)
        raise FileNotFoundError(f"Video not found: {video_path}")

    if should_skip(wav_path, force):
        log.info("[01] Cached — skipping audio extraction for %s", video_path.name)
        return wav_path

    wav_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("[01] Extracting audio: %s → %s", video_path.name, wav_path.name)
    log.debug(
        "[01] ffmpeg settings: sample_rate=%d Hz, channels=%d (mono), codec=pcm_s16le",
        config.AUDIO_SAMPLE_RATE,
        config.AUDIO_CHANNELS,
    )

    try:
        (
            ffmpeg
            .input(str(video_path))
            .output(
                str(wav_path),
                ar=config.AUDIO_SAMPLE_RATE,
                ac=config.AUDIO_CHANNELS,
                acodec="pcm_s16le",
            )
            .overwrite_output()
            .run(quiet=True)
        )
    except ffmpeg.Error as e:
        log.error("[01] ffmpeg failed for %s: %s", video_path.name, e.stderr.decode() if e.stderr else str(e))
        raise

    size_mb = wav_path.stat().st_size / 1_048_576
    log.info("[01] Done — WAV written: %s (%.1f MB)", wav_path, size_mb)
    return wav_path


def run(video_path: str | Path, force: bool = False) -> Path:
    """
    Step 01 entry point — extract audio for a single video.

    The WAV is written to two locations:
      - data/audio/<video_id>.wav      (primary audio store)
      - outputs/per_video/<vid>/audio.wav  (symlink, used by downstream steps)

    Args:
        video_path: Path to the source MP4 in DATA/.
        force:      Bypass cache and re-extract even if WAV exists.

    Returns:
        Path to the WAV file in outputs/per_video/<vid>/.
    """
    video_path = Path(video_path)
    vid = video_id_from_path(video_path)

    log.debug("[01] Processing video_id=%s  source=%s", vid, video_path)

    # Primary WAV location
    config.AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    wav_primary = config.AUDIO_DIR / f"{vid}.wav"

    # Per-video output dir (creates the folder)
    out_dir = get_video_output_dir(video_path, config.PER_VIDEO_DIR)
    wav_link = out_dir / "audio.wav"

    # Extract to primary location
    _extract_single(video_path, wav_primary, force)

    # Create a symlink from the per-video dir to the primary WAV
    # so all downstream steps can use a consistent path: <out_dir>/audio.wav
    if not wav_link.exists() or force:
        if wav_link.is_symlink() or wav_link.exists():
            wav_link.unlink()
        wav_link.symlink_to(wav_primary)
        log.debug("[01] Symlink created: %s → %s", wav_link, wav_primary)
    else:
        log.debug("[01] Symlink already exists: %s", wav_link)

    return wav_link


def run_batch(video_paths: list[Path], force: bool = False) -> list[Path]:
    """
    Run step 01 on a list of videos with a tqdm progress bar.

    Args:
        video_paths: List of MP4 files to process.
        force:       Bypass cache for all videos.

    Returns:
        List of WAV file paths in the same order as input.
    """
    results = []
    log.info("[01] Starting batch audio extraction for %d video(s)", len(video_paths))

    for video_path in tqdm(video_paths, desc="[01] Extracting audio", unit="video"):
        try:
            wav = run(video_path, force=force)
            results.append(wav)
        except Exception as exc:
            log.error("[01] FAILED for %s: %s", video_path.name, exc, exc_info=True)
            results.append(None)

    succeeded = sum(1 for r in results if r is not None)
    log.info("[01] Batch complete — %d/%d succeeded", succeeded, len(video_paths))
    return results


if __name__ == "__main__":
    # Allow direct invocation for testing a single file:
    #   python -m pipeline.extract_audio DATA/001*.mp4
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m pipeline.extract_audio <video_path> [--force]")
        sys.exit(1)
    _path = Path(sys.argv[1])
    _force = "--force" in sys.argv
    run(_path, force=_force)
