"""
run_all.py — SRHI Pipeline Master Runner

Orchestrates all 9 pipeline steps for one or more videos.
Supports per-step granularity, single-video targeting, and forced recomputation.

Usage examples:
  python run_all.py                              # All videos, all steps
  python run_all.py --video "DATA/001*.mp4"      # Single video (glob pattern)
  python run_all.py --step 01 02 03              # Specific steps only
  python run_all.py --force                      # Bypass all caches
  python run_all.py --video "DATA/001*.mp4" --step 05 08 09 --force

Step map:
  01 = extract_audio     03 = transcribe    05 = sentiment    07 = turn_dynamics
  02 = diarize           04 = prosody       06 = embeddings   08 = merge
                                                              09 = visualize

All output is logged to both the terminal and logs/<run_id>/pipeline_run.log.
"""

from __future__ import annotations

import argparse
import glob
import sys
from datetime import datetime
from pathlib import Path

# ── Bootstrap path so 'config' and 'pipeline' are importable ─────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from pipeline.logger import get_logger, set_run_id

# Step module imports (deferred to avoid slow model loads on --help)
_STEP_MODULES: dict[str, object] = {}


def _import_steps() -> None:
    """Lazily import all step modules (avoids slow loads on --help)."""
    global _STEP_MODULES
    if _STEP_MODULES:
        return
    from pipeline import extract_audio, diarize, transcribe, prosody
    from pipeline import sentiment, embeddings, turn_dynamics, merge, visualize
    _STEP_MODULES = {
        "01": extract_audio,
        "02": diarize,
        "03": transcribe,
        "04": prosody,
        "05": sentiment,
        "06": embeddings,
        "07": turn_dynamics,
        "08": merge,
        "09": visualize,
    }


ALL_STEPS = ["01", "02", "03", "04", "05", "06", "07", "08", "09"]

STEP_DESCRIPTIONS = {
    "01": "Audio extraction (ffmpeg)",
    "02": "Speaker diarization (pyannote)",
    "03": "Transcription + word timestamps (faster-whisper)",
    "04": "Prosody features (Praat/parselmouth)",
    "05": "Sentiment, sycophancy, hedging (RoBERTa + lexicon)",
    "06": "Sentence embeddings + lexical entrainment",
    "07": "Turn dynamics (pure pandas from RTTM)",
    "08": "Merge + SRHI computation + windowed metrics",
    "09": "Visualization (matplotlib, per-session + cross-session)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_all.py",
        description="SRHI Pipeline — Multimodal conversation analysis for human-AI companion research.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(
            f"  {k}: {v}" for k, v in STEP_DESCRIPTIONS.items()
        ),
    )
    parser.add_argument(
        "--video",
        nargs="+",
        metavar="VIDEO",
        help=(
            "Path(s) or glob pattern(s) to video files. "
            "Default: all MP4s in DATA/. "
            "Example: --video \"DATA/001*.mp4\" \"DATA/002*.mp4\""
        ),
    )
    parser.add_argument(
        "--step",
        nargs="+",
        metavar="STEP",
        help=(
            "Step number(s) to run (01–09). Default: all steps. "
            "Example: --step 01 02 03"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Bypass cache and recompute all outputs even if they already exist.",
    )
    parser.add_argument(
        "--list-steps",
        action="store_true",
        help="Print the step map and exit.",
    )
    return parser.parse_args()


def resolve_videos(video_args: list[str] | None) -> list[Path]:
    """Resolve --video argument(s) to a sorted list of existing MP4 paths."""
    if video_args is None:
        paths = sorted(config.DATA_DIR.glob("*.mp4"))
        if not paths:
            return []
        return paths

    paths = []
    for pattern in video_args:
        matched = [Path(p) for p in glob.glob(pattern)]
        if not matched:
            print(f"[WARNING] No files matched: {pattern}", file=sys.stderr)
        paths.extend(matched)
    return sorted(set(paths))


def resolve_steps(step_args: list[str] | None) -> list[str]:
    """Resolve --step argument(s) to a validated, ordered list of step keys."""
    if step_args is None:
        return ALL_STEPS[:]

    steps = []
    for s in step_args:
        key = s.zfill(2)
        if key not in STEP_DESCRIPTIONS:
            print(f"[ERROR] Unknown step: {s!r}. Valid steps: {', '.join(ALL_STEPS)}", file=sys.stderr)
            sys.exit(1)
        steps.append(key)
    return steps


def _run_step_for_video(step_key: str, video_path: Path, force: bool, log) -> bool:
    """
    Run a single step for a single video.

    Args:
        step_key:   Two-digit step key ('01'–'09').
        video_path: Source MP4 path.
        force:      Bypass cache.
        log:        Logger instance.

    Returns:
        True on success, False on failure.
    """
    module = _STEP_MODULES[step_key]
    step_fn = getattr(module, "run", None)

    if step_fn is None:
        log.error("Step %s has no run() function — skipping", step_key)
        return False

    log.info("  ▶  Step %s: %s", step_key, STEP_DESCRIPTIONS[step_key])
    try:
        result = step_fn(video_path, force=force)
        log.info("  ✓  Step %s complete → %s", step_key,
                 result.name if hasattr(result, "name") else result)
        return True
    except Exception as exc:
        log.error("  ✗  Step %s FAILED: %s", step_key, exc, exc_info=True)
        return False


def main() -> None:
    args = parse_args()

    if args.list_steps:
        print("\nStep map:")
        for k, v in STEP_DESCRIPTIONS.items():
            print(f"  {k}: {v}")
        print()
        return

    # ── Set up run ID and logging ─────────────────────────────────────────────
    run_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = config.LOGS_DIR / run_id
    set_run_id(run_id, log_dir)

    log = get_logger("run_all")
    log.info("=" * 60)
    log.info("SRHI Pipeline — run_id: %s", run_id)
    log.info("=" * 60)

    # ── Resolve videos and steps ──────────────────────────────────────────────
    videos = resolve_videos(args.video)
    steps  = resolve_steps(args.step)

    if not videos:
        log.error("No video files found. Put MP4s in the DATA/ folder or use --video.")
        sys.exit(1)

    log.info("Videos to process : %d", len(videos))
    for v in videos:
        log.info("  • %s  (%.1f MB)", v.name, v.stat().st_size / 1_048_576)
    log.info("Steps to run      : %s", " → ".join(steps))
    log.info("Force recompute   : %s", args.force)
    log.info("Device            : %s", config.TORCH_DEVICE)
    log.info("Log directory     : %s", log_dir)
    log.info("-" * 60)

    # ── Import step modules ───────────────────────────────────────────────────
    _import_steps()

    # ── Ensure output directories exist ──────────────────────────────────────
    config.PER_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    config.MERGED_DIR.mkdir(parents=True, exist_ok=True)
    config.AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    (config.FIGURES_DIR / "cross_session").mkdir(parents=True, exist_ok=True)

    # ── Main processing loop ──────────────────────────────────────────────────
    overall_success = True
    total_steps_run = 0
    total_steps_failed = 0

    for video_path in videos:
        log.info("")
        log.info("━" * 60)
        log.info("VIDEO: %s", video_path.name)
        log.info("━" * 60)

        # Step 09 (visualize) is handled separately after all videos
        video_steps = [s for s in steps if s != "09"]

        for step_key in video_steps:
            success = _run_step_for_video(step_key, video_path, args.force, log)
            total_steps_run += 1
            if not success:
                total_steps_failed += 1
                overall_success = False
                # Continue processing other steps (don't abort the whole run)

    # ── Step 09 — visualization (runs once after all videos processed) ────────
    if "09" in steps:
        log.info("")
        log.info("━" * 60)
        log.info("STEP 09: %s", STEP_DESCRIPTIONS["09"])
        log.info("━" * 60)
        try:
            vis_module = _STEP_MODULES["09"]
            vis_module.run_batch(videos, force=args.force)
            log.info("✓ Step 09 complete")
        except Exception as exc:
            log.error("✗ Step 09 FAILED: %s", exc, exc_info=True)
            overall_success = False

    # ── Final summary ─────────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 60)
    log.info("Run complete — %s", "ALL STEPS SUCCEEDED" if overall_success else "SOME STEPS FAILED")
    log.info("Steps run: %d  |  Failed: %d", total_steps_run, total_steps_failed)
    log.info("Outputs in: %s", config.OUTPUTS_DIR if hasattr(config, 'OUTPUTS_DIR') else config.PER_VIDEO_DIR.parent)
    log.info("Log file: %s", log_dir / "run_all.log")
    log.info("=" * 60)

    if not overall_success:
        sys.exit(1)


if __name__ == "__main__":
    main()
