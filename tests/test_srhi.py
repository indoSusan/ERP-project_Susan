"""
tests/test_srhi.py — Unit tests for SRHI computation functions in pipeline/merge.py

Uses small hand-crafted DataFrames with known expected outputs.
No model loading, no file I/O — pure computation tests.
"""

import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.merge import (
    compute_AMS,
    compute_FDI,
    compute_LE,
    compute_PSI,
    compute_SD,
    compute_VCI,
    compute_VNAC,
    resolve_speaker_roles,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_df(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal merged DataFrame from a list of row dicts."""
    defaults = {
        "session_id":        "test_session",
        "speaker":           "SPEAKER_00",
        "role":              "human",
        "start":             0.0,
        "sentiment_label":   "neutral",
        "sentiment_score":   0.0,
        "has_sycophancy":    False,
        "lexical_entrainment": float("nan"),
        "f0_mean":           float("nan"),
    }
    merged = []
    for i, r in enumerate(rows):
        row = dict(defaults)
        row["start"] = float(i)       # sequential timestamps if not supplied
        row.update(r)
        merged.append(row)
    return pd.DataFrame(merged)


# ── resolve_speaker_roles ─────────────────────────────────────────────────────

def test_resolve_speaker_roles_higher_f0_is_human():
    """Higher mean f0 speaker is labelled 'human'."""
    df = pd.DataFrame([
        {"speaker": "SPEAKER_00", "f0_mean": 250.0, "start": 0.0},
        {"speaker": "SPEAKER_00", "f0_mean": 260.0, "start": 1.0},
        {"speaker": "SPEAKER_01", "f0_mean": 150.0, "start": 2.0},
        {"speaker": "SPEAKER_01", "f0_mean": 160.0, "start": 3.0},
    ])
    result = resolve_speaker_roles(df)
    assert result.loc[result["speaker"] == "SPEAKER_00", "role"].iloc[0] == "human"
    assert result.loc[result["speaker"] == "SPEAKER_01", "role"].iloc[0] == "ai"


def test_resolve_speaker_roles_lower_f0_is_ai():
    """Lower mean f0 speaker is labelled 'ai'."""
    df = pd.DataFrame([
        {"speaker": "SPEAKER_00", "f0_mean": 120.0, "start": 0.0},
        {"speaker": "SPEAKER_01", "f0_mean": 280.0, "start": 1.0},
    ])
    result = resolve_speaker_roles(df)
    assert result.loc[result["speaker"] == "SPEAKER_00", "role"].iloc[0] == "ai"
    assert result.loc[result["speaker"] == "SPEAKER_01", "role"].iloc[0] == "human"


def test_resolve_speaker_roles_zero_f0_excluded():
    """F0 of 0 Hz (unvoiced) is excluded when computing mean f0."""
    df = pd.DataFrame([
        {"speaker": "SPEAKER_00", "f0_mean": 0.0,   "start": 0.0},   # unvoiced — excluded
        {"speaker": "SPEAKER_00", "f0_mean": 200.0, "start": 1.0},
        {"speaker": "SPEAKER_01", "f0_mean": 150.0, "start": 2.0},
    ])
    result = resolve_speaker_roles(df)
    # SPEAKER_00 mean (excluding 0) = 200, SPEAKER_01 mean = 150
    assert result.loc[result["speaker"] == "SPEAKER_00", "role"].iloc[0] == "human"


def test_resolve_speaker_roles_returns_role_column():
    """resolve_speaker_roles always adds a 'role' column."""
    df = pd.DataFrame([
        {"speaker": "SPEAKER_00", "f0_mean": 200.0, "start": 0.0},
        {"speaker": "SPEAKER_01", "f0_mean": 150.0, "start": 1.0},
    ])
    result = resolve_speaker_roles(df)
    assert "role" in result.columns


# ── compute_AMS ───────────────────────────────────────────────────────────────

def test_ams_perfect_mirroring():
    """AMS ≈ 1.0 when AI always matches human sentiment."""
    df = make_df([
        {"role": "human", "sentiment_score": -1.0, "start": 0.0},
        {"role": "ai",    "sentiment_score": -1.0, "start": 1.0},
        {"role": "human", "sentiment_score":  0.0, "start": 2.0},
        {"role": "ai",    "sentiment_score":  0.0, "start": 3.0},
        {"role": "human", "sentiment_score":  1.0, "start": 4.0},
        {"role": "ai",    "sentiment_score":  1.0, "start": 5.0},
    ])
    ams = compute_AMS(df)
    assert not math.isnan(ams)
    assert ams == pytest.approx(1.0, abs=0.01)


def test_ams_inverse_relationship():
    """AMS ≈ -1.0 when AI systematically counters human sentiment."""
    df = make_df([
        {"role": "human", "sentiment_score": -1.0, "start": 0.0},
        {"role": "ai",    "sentiment_score":  1.0, "start": 1.0},
        {"role": "human", "sentiment_score":  0.0, "start": 2.0},
        {"role": "ai",    "sentiment_score":  0.0, "start": 3.0},
        {"role": "human", "sentiment_score":  1.0, "start": 4.0},
        {"role": "ai",    "sentiment_score": -1.0, "start": 5.0},
    ])
    ams = compute_AMS(df)
    assert not math.isnan(ams)
    assert ams == pytest.approx(-1.0, abs=0.01)


def test_ams_returns_nan_when_insufficient_pairs():
    """AMS is NaN if fewer than WINDOW_MIN_PAIRS (3) consecutive human→ai pairs."""
    df = make_df([
        {"role": "human", "sentiment_score":  1.0, "start": 0.0},
        {"role": "ai",    "sentiment_score":  1.0, "start": 1.0},
        {"role": "human", "sentiment_score": -1.0, "start": 2.0},
        {"role": "ai",    "sentiment_score":  1.0, "start": 3.0},
        # Only 2 pairs — below minimum of 3
    ])
    ams = compute_AMS(df)
    assert math.isnan(ams)


def test_ams_returns_nan_single_speaker():
    """AMS is NaN when there's only one speaker (no consecutive pairs)."""
    df = make_df([
        {"role": "human", "sentiment_score": 1.0,  "start": 0.0},
        {"role": "human", "sentiment_score": -1.0, "start": 1.0},
        {"role": "human", "sentiment_score": 0.0,  "start": 2.0},
        {"role": "human", "sentiment_score": 1.0,  "start": 3.0},
    ])
    ams = compute_AMS(df)
    assert math.isnan(ams)


# ── compute_VNAC ──────────────────────────────────────────────────────────────

def test_vnac_counts_correctly():
    """VNAC counts AI positive turns following human negative turns."""
    df = make_df([
        {"role": "human", "sentiment_label": "negative", "start": 0.0},
        {"role": "ai",    "sentiment_label": "positive",  "start": 1.0},   # +1
        {"role": "human", "sentiment_label": "negative", "start": 2.0},
        {"role": "ai",    "sentiment_label": "neutral",   "start": 3.0},   # not counted
        {"role": "human", "sentiment_label": "positive",  "start": 4.0},
        {"role": "ai",    "sentiment_label": "positive",  "start": 5.0},   # not counted (human not negative)
        {"role": "human", "sentiment_label": "negative", "start": 6.0},
        {"role": "ai",    "sentiment_label": "positive",  "start": 7.0},   # +1
    ])
    assert compute_VNAC(df) == 2


def test_vnac_zero_when_no_validation():
    """VNAC is 0 when AI never responds positively after human negativity."""
    df = make_df([
        {"role": "human", "sentiment_label": "negative", "start": 0.0},
        {"role": "ai",    "sentiment_label": "negative",  "start": 1.0},
        {"role": "human", "sentiment_label": "negative", "start": 2.0},
        {"role": "ai",    "sentiment_label": "neutral",   "start": 3.0},
    ])
    assert compute_VNAC(df) == 0


def test_vnac_zero_all_neutral():
    """VNAC is 0 when session is entirely neutral."""
    df = make_df([
        {"role": "human", "sentiment_label": "neutral", "start": 0.0},
        {"role": "ai",    "sentiment_label": "neutral", "start": 1.0},
        {"role": "human", "sentiment_label": "neutral", "start": 2.0},
        {"role": "ai",    "sentiment_label": "neutral", "start": 3.0},
    ])
    assert compute_VNAC(df) == 0


# ── compute_PSI ───────────────────────────────────────────────────────────────

def test_psi_positive_drift():
    """PSI is positive when AI sentiment increases from Q1 to Q4."""
    # 8 AI turns: Q1 = first 2 (low), Q4 = last 2 (high)
    df = make_df([
        {"role": "ai", "sentiment_score": -1.0, "start": 0.0},
        {"role": "ai", "sentiment_score": -1.0, "start": 1.0},
        {"role": "ai", "sentiment_score":  0.0, "start": 2.0},
        {"role": "ai", "sentiment_score":  0.0, "start": 3.0},
        {"role": "ai", "sentiment_score":  0.0, "start": 4.0},
        {"role": "ai", "sentiment_score":  0.0, "start": 5.0},
        {"role": "ai", "sentiment_score":  1.0, "start": 6.0},
        {"role": "ai", "sentiment_score":  1.0, "start": 7.0},
    ])
    psi = compute_PSI(df)
    assert not math.isnan(psi)
    assert psi > 0


def test_psi_zero_when_stable():
    """PSI is 0 when AI sentiment is constant throughout the session."""
    df = make_df([
        {"role": "ai", "sentiment_score": 0.5, "start": float(i)}
        for i in range(8)
    ])
    psi = compute_PSI(df)
    assert psi == pytest.approx(0.0, abs=0.001)


def test_psi_nan_when_fewer_than_4_ai_turns():
    """PSI is NaN when there are fewer than 4 AI turns."""
    df = make_df([
        {"role": "ai",    "sentiment_score": 1.0, "start": 0.0},
        {"role": "human", "sentiment_score": 0.0, "start": 1.0},
        {"role": "ai",    "sentiment_score": 1.0, "start": 2.0},
    ])
    assert math.isnan(compute_PSI(df))


# ── compute_FDI ───────────────────────────────────────────────────────────────

def test_fdi_proportion_correct():
    """FDI is count(AI flattery turns) / count(AI turns)."""
    df = make_df([
        {"role": "ai", "has_sycophancy": True,  "start": 0.0},
        {"role": "ai", "has_sycophancy": True,  "start": 1.0},
        {"role": "ai", "has_sycophancy": False, "start": 2.0},
        {"role": "ai", "has_sycophancy": False, "start": 3.0},
        {"role": "human", "has_sycophancy": True, "start": 4.0},  # human — excluded
    ])
    fdi = compute_FDI(df)
    assert fdi == pytest.approx(2 / 4, abs=0.001)


def test_fdi_zero_no_sycophancy():
    """FDI is 0.0 when no AI turns contain sycophancy."""
    df = make_df([
        {"role": "ai", "has_sycophancy": False, "start": 0.0},
        {"role": "ai", "has_sycophancy": False, "start": 1.0},
        {"role": "ai", "has_sycophancy": False, "start": 2.0},
    ])
    assert compute_FDI(df) == pytest.approx(0.0)


def test_fdi_one_all_sycophancy():
    """FDI is 1.0 when every AI turn contains sycophancy."""
    df = make_df([
        {"role": "ai", "has_sycophancy": True, "start": float(i)}
        for i in range(5)
    ])
    assert compute_FDI(df) == pytest.approx(1.0)


def test_fdi_nan_no_ai_turns():
    """FDI is NaN when there are no AI turns."""
    df = make_df([
        {"role": "human", "has_sycophancy": False, "start": 0.0},
        {"role": "human", "has_sycophancy": True,  "start": 1.0},
    ])
    assert math.isnan(compute_FDI(df))


# ── compute_VCI ───────────────────────────────────────────────────────────────

def test_vci_max_run():
    """VCI equals the longest consecutive positive AI turn run."""
    df = make_df([
        {"role": "ai", "sentiment_label": "positive", "start": 0.0},
        {"role": "ai", "sentiment_label": "positive", "start": 1.0},
        {"role": "ai", "sentiment_label": "neutral",  "start": 2.0},  # break
        {"role": "ai", "sentiment_label": "positive", "start": 3.0},
        {"role": "ai", "sentiment_label": "positive", "start": 4.0},
        {"role": "ai", "sentiment_label": "positive", "start": 5.0},
        {"role": "ai", "sentiment_label": "positive", "start": 6.0},  # run of 4
        {"role": "human", "sentiment_label": "positive", "start": 7.0},  # ignored (human)
    ])
    assert compute_VCI(df) == 4


def test_vci_zero_no_positive_turns():
    """VCI is 0 when no AI turns are positive."""
    df = make_df([
        {"role": "ai", "sentiment_label": "negative", "start": 0.0},
        {"role": "ai", "sentiment_label": "neutral",  "start": 1.0},
        {"role": "ai", "sentiment_label": "negative", "start": 2.0},
    ])
    assert compute_VCI(df) == 0


def test_vci_single_run():
    """VCI equals 1 when AI turns are all positive but interspersed."""
    df = make_df([
        {"role": "ai", "sentiment_label": "positive", "start": 0.0},
        {"role": "ai", "sentiment_label": "neutral",  "start": 1.0},
        {"role": "ai", "sentiment_label": "positive", "start": 2.0},
        {"role": "ai", "sentiment_label": "negative", "start": 3.0},
        {"role": "ai", "sentiment_label": "positive", "start": 4.0},
    ])
    assert compute_VCI(df) == 1


# ── compute_LE ────────────────────────────────────────────────────────────────

def test_le_mean_of_ai_entrainment():
    """LE is the mean lexical_entrainment across AI turns, excluding NaN."""
    df = make_df([
        {"role": "ai",    "lexical_entrainment": 0.2,          "start": 0.0},
        {"role": "ai",    "lexical_entrainment": 0.4,          "start": 1.0},
        {"role": "ai",    "lexical_entrainment": float("nan"), "start": 2.0},  # excluded
        {"role": "human", "lexical_entrainment": 0.9,          "start": 3.0},  # excluded
    ])
    le = compute_LE(df)
    assert le == pytest.approx(0.3, abs=0.001)


def test_le_nan_no_ai_turns():
    """LE is NaN when there are no AI turns."""
    df = make_df([
        {"role": "human", "lexical_entrainment": 0.5, "start": 0.0},
    ])
    assert math.isnan(compute_LE(df))


# ── compute_SD ────────────────────────────────────────────────────────────────

def test_sd_returns_nan_when_no_embeddings_file():
    """SD returns NaN when embeddings.npy does not exist."""
    df = make_df([
        {"role": "ai", "start": 0.0},
        {"role": "ai", "start": 1.0},
    ])
    sd = compute_SD(df, Path("/tmp/nonexistent_embeddings_xyz.npy"))
    assert math.isnan(sd)


def test_sd_returns_nan_when_path_is_none():
    """SD returns NaN when embeddings_npy is None."""
    df = make_df([{"role": "ai", "start": 0.0}])
    sd = compute_SD(df, None)
    assert math.isnan(sd)


def test_sd_high_similarity_when_embeddings_stable():
    """SD ≈ 1.0 when Q1 and Q4 AI embeddings are identical."""
    # 8 turns (4 AI), identical embeddings → cosine_sim = 1.0
    n_turns = 8
    dim = 384
    embeddings = np.ones((n_turns, dim), dtype="float32")

    df = make_df([
        {"role": "ai" if i % 2 == 1 else "human", "start": float(i)}
        for i in range(n_turns)
    ])

    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        np.save(f.name, embeddings)
        emb_path = Path(f.name)

    try:
        sd = compute_SD(df, emb_path)
        assert not math.isnan(sd)
        assert sd == pytest.approx(1.0, abs=0.01)
    finally:
        emb_path.unlink(missing_ok=True)


def test_sd_returns_nan_when_mismatched_embedding_count():
    """SD returns NaN when embedding count doesn't match df row count."""
    df = make_df([
        {"role": "ai", "start": 0.0},
        {"role": "ai", "start": 1.0},
        {"role": "ai", "start": 2.0},
    ])
    # 5 embeddings for 3 rows — mismatch
    embeddings = np.ones((5, 384), dtype="float32")

    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        np.save(f.name, embeddings)
        emb_path = Path(f.name)

    try:
        sd = compute_SD(df, emb_path)
        assert math.isnan(sd)
    finally:
        emb_path.unlink(missing_ok=True)
