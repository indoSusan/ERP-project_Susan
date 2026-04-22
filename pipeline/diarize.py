"""
pipeline/diarize.py — Step 02: Speaker diarization

Uses pyannote/speaker-diarization-3.1 to segment the audio into
speaker turns and write an RTTM file. The pipeline is constrained to
num_speakers=2 since all sessions have exactly two participants
(the human researcher and the AI companion via TTS).

Prerequisites:
  - HF_TOKEN must be set (in .env or environment variable).
  - The user must have accepted the gated model licenses at:
      https://huggingface.co/pyannote/speaker-diarization-3.1
      https://huggingface.co/pyannote/segmentation-3.0

RTTM format (one line per turn):
    SPEAKER <session_id> 1 <start_sec> <duration_sec> <NA> <NA> <SPEAKER_XX> <NA> <NA>

Output: outputs/per_video/<vid>/diarization.rttm
Caching: skipped if RTTM already exists and is non-empty.
"""

from __future__ import annotations

from pathlib import Path

import torch
from tqdm import tqdm

import config
from pipeline.cache import (
    get_video_output_dir,
    should_skip,
    video_id_from_path,
)
from pipeline.logger import get_logger

log = get_logger(__name__)


def _load_pipeline() -> object:
    """
    Load the pyannote diarization pipeline onto the best available device.

    Tries MPS first (Apple Silicon acceleration), falls back to CPU if
    MPS causes an error. The model weights are cached by HuggingFace in
    ~/.cache/huggingface/ and only downloaded once.

    Returns:
        Loaded pyannote Pipeline object.

    Raises:
        ValueError: If HF_TOKEN is missing.
        RuntimeError: If the model cannot be loaded.
    """
    if not config.HF_TOKEN:
        msg = (
            "HF_TOKEN is not set. "
            "Add it to .env (HF_TOKEN=hf_...) or export it in your terminal. "
            "You also need to accept the model license at "
            "https://huggingface.co/pyannote/speaker-diarization-3.1"
        )
        log.error("[02] %s", msg)
        raise ValueError(msg)

    log.info("[02] Loading pyannote model: %s", config.DIARIZATION_MODEL)

    from pyannote.audio import Pipeline  # deferred import — heavy load

    pipeline = Pipeline.from_pretrained(
        config.DIARIZATION_MODEL,
        token=config.HF_TOKEN,
    )

    device = torch.device(config.TORCH_DEVICE)
    log.debug("[02] Attempting to move pipeline to device: %s", device)

    try:
        pipeline = pipeline.to(device)
        log.info("[02] Pipeline running on: %s", device)
    except Exception as exc:
        log.warning(
            "[02] Failed to use %s (%s) — falling back to CPU",
            device,
            exc,
        )
        pipeline = pipeline.to(torch.device("cpu"))
        log.info("[02] Pipeline running on: cpu (fallback)")

    return pipeline


def _run_diarization(audio_path: Path, rttm_path: Path, pipeline: object) -> Path:
    """
    Run the diarization pipeline on a single audio file and write the RTTM.

    Args:
        audio_path:  16 kHz mono WAV file.
        rttm_path:   Destination RTTM file.
        pipeline:    Loaded pyannote Pipeline.

    Returns:
        Path to the written RTTM file.
    """
    log.info("[02] Running diarization on: %s", audio_path.name)
    log.debug(
        "[02] Diarization params: num_speakers=%d, model=%s",
        config.NUM_SPEAKERS,
        config.DIARIZATION_MODEL,
    )

    diarization = pipeline(
        str(audio_path),
        num_speakers=config.NUM_SPEAKERS,
    )

    turn_count = sum(1 for _ in diarization.speaker_diarization.itertracks(yield_label=True))
    log.info("[02] Diarization complete — %d speaker turns detected", turn_count)

    rttm_path.parent.mkdir(parents=True, exist_ok=True)
    with open(rttm_path, "w", encoding="utf-8") as f:
        diarization.speaker_diarization.write_rttm(f)

    size_b = rttm_path.stat().st_size
    log.info("[02] RTTM written: %s (%d bytes)", rttm_path, size_b)
    return rttm_path


def run(video_path: str | Path, force: bool = False) -> Path:
    """
    Step 02 entry point — diarize a single video's audio.

    Args:
        video_path: Path to the source MP4 (used to resolve the video ID
                    and locate the extracted WAV from step 01).
        force:      Bypass cache and re-diarize even if RTTM exists.

    Returns:
        Path to the RTTM file.

    Raises:
        FileNotFoundError: If step 01 has not been run (WAV missing).
        ValueError: If HF_TOKEN is missing.
    """
    video_path = Path(video_path)
    vid = video_id_from_path(video_path)
    out_dir = get_video_output_dir(video_path, config.PER_VIDEO_DIR)

    audio_path = out_dir / "audio.wav"
    rttm_path  = out_dir / "diarization.rttm"

    log.debug("[02] video_id=%s  audio=%s  rttm=%s", vid, audio_path, rttm_path)

    if not audio_path.exists():
        log.error("[02] WAV not found at %s — run step 01 first", audio_path)
        raise FileNotFoundError(f"Audio file not found: {audio_path}. Run step 01 first.")

    if should_skip(rttm_path, force):
        log.info("[02] Cached — skipping diarization for %s", video_path.name)
        return rttm_path

    pipeline = _load_pipeline()
    return _run_diarization(audio_path, rttm_path, pipeline)


def run_batch(video_paths: list[Path], force: bool = False) -> list[Path]:
    """
    Run step 02 on a list of videos.
    The pipeline is loaded once and reused across all videos.

    Args:
        video_paths: List of MP4 source files.
        force:       Bypass cache for all videos.

    Returns:
        List of RTTM paths (None for any that failed).
    """
    log.info("[02] Starting batch diarization for %d video(s)", len(video_paths))

    # Determine which videos actually need processing (not cached)
    to_process = [
        v for v in video_paths
        if not should_skip(
            get_video_output_dir(v, config.PER_VIDEO_DIR) / "diarization.rttm",
            force,
        )
    ]
    skipped = len(video_paths) - len(to_process)
    if skipped:
        log.info("[02] %d video(s) already cached — loading pipeline for %d", skipped, len(to_process))

    # Load pipeline once
    pipeline = _load_pipeline() if to_process else None

    results = []
    for video_path in tqdm(video_paths, desc="[02] Diarizing", unit="video"):
        vid = video_id_from_path(video_path)
        out_dir = get_video_output_dir(video_path, config.PER_VIDEO_DIR)
        rttm_path = out_dir / "diarization.rttm"
        audio_path = out_dir / "audio.wav"

        if should_skip(rttm_path, force):
            log.info("[02] Cached — skipping %s", video_path.name)
            results.append(rttm_path)
            continue

        if not audio_path.exists():
            log.error("[02] WAV missing for %s — skipping", video_path.name)
            results.append(None)
            continue

        try:
            rttm = _run_diarization(audio_path, rttm_path, pipeline)
            results.append(rttm)
        except Exception as exc:
            log.error("[02] FAILED for %s: %s", video_path.name, exc, exc_info=True)
            results.append(None)

    succeeded = sum(1 for r in results if r is not None)
    log.info("[02] Batch complete — %d/%d succeeded", succeeded, len(video_paths))
    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m pipeline.diarize <video_path> [--force]")
        sys.exit(1)
    _path = Path(sys.argv[1])
    _force = "--force" in sys.argv
    run(_path, force=_force)
