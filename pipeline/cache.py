"""
pipeline/cache.py — Caching utilities for the SRHI pipeline.

Every step script uses these helpers to:
  1. Derive a stable, filesystem-safe video ID from a video filename.
  2. Resolve the per-video output directory.
  3. Check whether a step's output already exists (skipping re-computation).

Caching is intentionally simple: a file is "cached" if it exists and
is non-empty. The --force flag in run_all.py bypasses all cache checks.
"""

import re
from pathlib import Path


def video_id_from_path(video_path: str | Path) -> str:
    """
    Derive a stable, filesystem-safe ID from a video filename.

    Replaces any non-alphanumeric characters with underscores,
    strips leading/trailing underscores, and lowercases the result.

    Examples:
        '001 Rec Noah 06-03-2026mp4.mp4' → '001_rec_noah_06_03_2026mp4'
        '005 Rec Charlie 27-03-2026.mp4' → '005_rec_charlie_27_03_2026'
    """
    stem = Path(video_path).stem
    vid = re.sub(r"[^a-zA-Z0-9]+", "_", stem).strip("_").lower()
    return vid


def get_video_output_dir(video_path: str | Path, per_video_root: Path) -> Path:
    """
    Return the per-video output directory, creating it if it does not exist.

    Args:
        video_path:      Path to the source MP4 file.
        per_video_root:  Root directory for all per-video outputs
                         (config.PER_VIDEO_DIR).

    Returns:
        Path to the specific video's output subdirectory.
    """
    vid = video_id_from_path(video_path)
    d = per_video_root / vid
    d.mkdir(parents=True, exist_ok=True)
    return d


def is_cached(output_path: str | Path) -> bool:
    """
    Return True if output_path exists on disk and is non-empty.

    A zero-byte file is treated as not cached (it may be a failed
    partial write from a previous interrupted run).
    """
    p = Path(output_path)
    return p.exists() and p.stat().st_size > 0


def all_cached(output_paths: list[str | Path]) -> bool:
    """Return True only if every path in the list is cached."""
    return all(is_cached(p) for p in output_paths)


def should_skip(output_path: str | Path | list, force: bool) -> bool:
    """
    Return True when the step should be skipped (output exists and force is off).

    Accepts a single path or a list of paths — all must be cached.
    """
    if force:
        return False
    if isinstance(output_path, list):
        return all_cached(output_path)
    return is_cached(output_path)
