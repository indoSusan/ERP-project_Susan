"""
tests/test_cache.py — Unit tests for pipeline/cache.py

Tests the caching utility functions using temporary files and
the actual video filenames from the DATA/ folder.
"""

import tempfile
from pathlib import Path

import pytest

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.cache import (
    all_cached,
    get_video_output_dir,
    is_cached,
    should_skip,
    video_id_from_path,
)


# ── video_id_from_path ────────────────────────────────────────────────────────

KNOWN_FILENAMES = [
    ("001 Rec Noah 06-03-2026mp4.mp4", "001_rec_noah_06_03_2026mp4"),
    ("002 Rec Noah 18-03-2026 .mp4",   "002_rec_noah_18_03_2026"),
    ("003 Rec noah 20-03.mp4",          "003_rec_noah_20_03"),
    ("004 Rec noah 26-03-2026.mp4",     "004_rec_noah_26_03_2026"),
    ("005 Rec Charlie 27-03-2026.mp4",  "005_rec_charlie_27_03_2026"),
    ("006 Rec Charlie 31-03-2026.mp4",  "006_rec_charlie_31_03_2026"),
    ("007 Rec Noah 31-03-2026.mp4",     "007_rec_noah_31_03_2026"),
]


@pytest.mark.parametrize("filename, expected_id", KNOWN_FILENAMES)
def test_video_id_from_path_known_files(filename, expected_id):
    """video_id_from_path produces stable, expected IDs for all 7 known filenames."""
    vid = video_id_from_path(Path(f"DATA/{filename}"))
    assert vid == expected_id, f"Expected '{expected_id}', got '{vid}'"


def test_video_id_filesystem_safe():
    """Generated IDs contain only alphanumeric characters and underscores."""
    import re
    for filename, _ in KNOWN_FILENAMES:
        vid = video_id_from_path(Path(filename))
        assert re.match(r"^[a-z0-9_]+$", vid), f"ID '{vid}' contains invalid characters"


def test_video_id_no_leading_trailing_underscores():
    """Generated IDs don't start or end with underscores."""
    for filename, _ in KNOWN_FILENAMES:
        vid = video_id_from_path(Path(filename))
        assert not vid.startswith("_"), f"ID '{vid}' starts with underscore"
        assert not vid.endswith("_"), f"ID '{vid}' ends with underscore"


def test_video_id_stable():
    """Same input always produces the same output (deterministic)."""
    path = "DATA/001 Rec Noah 06-03-2026mp4.mp4"
    assert video_id_from_path(path) == video_id_from_path(path)


# ── is_cached ─────────────────────────────────────────────────────────────────

def test_is_cached_returns_false_for_nonexistent():
    """is_cached returns False for a path that doesn't exist."""
    assert not is_cached("/tmp/definitely_does_not_exist_abc123.csv")


def test_is_cached_returns_false_for_empty_file():
    """is_cached returns False for an existing but empty file."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = Path(f.name)
    # File exists but is empty (0 bytes)
    assert path.stat().st_size == 0
    assert not is_cached(path)
    path.unlink()


def test_is_cached_returns_true_for_nonempty_file():
    """is_cached returns True for an existing, non-empty file."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        f.write("session_id,speaker\nvid001,SPEAKER_00\n")
        path = Path(f.name)
    assert is_cached(path)
    path.unlink()


# ── all_cached ────────────────────────────────────────────────────────────────

def test_all_cached_true_when_all_exist():
    """all_cached returns True when every path in the list is cached."""
    with tempfile.TemporaryDirectory() as d:
        p1 = Path(d) / "a.csv"
        p2 = Path(d) / "b.npy"
        p1.write_text("data\n")
        p2.write_bytes(b"\x00\x01\x02")
        assert all_cached([p1, p2])


def test_all_cached_false_when_one_missing():
    """all_cached returns False if any path is missing."""
    with tempfile.TemporaryDirectory() as d:
        p1 = Path(d) / "a.csv"
        p2 = Path(d) / "b_missing.npy"
        p1.write_text("data\n")
        assert not all_cached([p1, p2])


# ── should_skip ───────────────────────────────────────────────────────────────

def test_should_skip_false_when_force():
    """should_skip returns False when force=True, regardless of cache state."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        f.write("data\n")
        path = Path(f.name)
    assert not should_skip(path, force=True)
    path.unlink()


def test_should_skip_true_when_cached_and_no_force():
    """should_skip returns True when output is cached and force=False."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        f.write("data\n")
        path = Path(f.name)
    assert should_skip(path, force=False)
    path.unlink()


def test_should_skip_false_when_not_cached_and_no_force():
    """should_skip returns False when output doesn't exist and force=False."""
    assert not should_skip("/tmp/srhi_test_not_exist_zz.csv", force=False)


def test_should_skip_list_of_paths():
    """should_skip accepts a list of paths (all must be cached to skip)."""
    with tempfile.TemporaryDirectory() as d:
        p1 = Path(d) / "a.csv"
        p2 = Path(d) / "b.npy"
        p1.write_text("data\n")
        p2.write_bytes(b"\x00\x01")
        assert should_skip([p1, p2], force=False)
        assert not should_skip([p1, p2], force=True)


# ── get_video_output_dir ──────────────────────────────────────────────────────

def test_get_video_output_dir_creates_directory():
    """get_video_output_dir creates the output directory if it doesn't exist."""
    with tempfile.TemporaryDirectory() as d:
        root  = Path(d)
        video = Path("DATA/001 Rec Noah 06-03-2026mp4.mp4")
        out   = get_video_output_dir(video, root)
        assert out.exists()
        assert out.is_dir()


def test_get_video_output_dir_correct_name():
    """get_video_output_dir uses the derived video ID as the folder name."""
    with tempfile.TemporaryDirectory() as d:
        root  = Path(d)
        video = Path("DATA/005 Rec Charlie 27-03-2026.mp4")
        out   = get_video_output_dir(video, root)
        assert out.name == "005_rec_charlie_27_03_2026"


def test_get_video_output_dir_idempotent():
    """get_video_output_dir does not raise if the directory already exists."""
    with tempfile.TemporaryDirectory() as d:
        root  = Path(d)
        video = Path("DATA/001 Rec Noah 06-03-2026mp4.mp4")
        out1  = get_video_output_dir(video, root)
        out2  = get_video_output_dir(video, root)
        assert out1 == out2
