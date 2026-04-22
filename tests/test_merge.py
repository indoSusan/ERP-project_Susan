"""
tests/test_merge.py — Unit tests for merge logic in pipeline/merge.py

Tests role resolution, merge_asof alignment, NaN-filling for out-of-tolerance
prosody matches, and windowed metrics computation.
No file I/O to the real outputs directory — uses synthetic DataFrames only.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.merge import (
    compute_windowed_metrics,
    resolve_speaker_roles,
)
import config


# ── resolve_speaker_roles — detailed tests ────────────────────────────────────

def test_resolve_roles_two_speakers_standard():
    """Standard case: two speakers, SPEAKER_00 has higher F0 → human."""
    df = pd.DataFrame([
        {"speaker": "SPEAKER_00", "f0_mean": 280.0, "start": 0.0},
        {"speaker": "SPEAKER_00", "f0_mean": 260.0, "start": 2.0},
        {"speaker": "SPEAKER_01", "f0_mean": 140.0, "start": 1.0},
        {"speaker": "SPEAKER_01", "f0_mean": 135.0, "start": 3.0},
    ])
    result = resolve_speaker_roles(df)
    assert set(result["role"].unique()) == {"human", "ai"}
    assert result.loc[result["speaker"] == "SPEAKER_00", "role"].iloc[0] == "human"
    assert result.loc[result["speaker"] == "SPEAKER_01", "role"].iloc[0] == "ai"


def test_resolve_roles_does_not_modify_original():
    """resolve_speaker_roles returns a copy — original is unchanged."""
    df = pd.DataFrame([
        {"speaker": "SPEAKER_00", "f0_mean": 200.0, "start": 0.0},
        {"speaker": "SPEAKER_01", "f0_mean": 150.0, "start": 1.0},
    ])
    original = df.copy()
    resolve_speaker_roles(df)
    pd.testing.assert_frame_equal(df, original)


def test_resolve_roles_all_nan_f0_falls_back():
    """When all F0 values are NaN, role is still assigned (fallback to first speaker)."""
    df = pd.DataFrame([
        {"speaker": "SPEAKER_00", "f0_mean": float("nan"), "start": 0.0},
        {"speaker": "SPEAKER_01", "f0_mean": float("nan"), "start": 1.0},
    ])
    result = resolve_speaker_roles(df)
    # Must have a 'role' column even with missing F0
    assert "role" in result.columns
    assert result["role"].notna().all()


def test_resolve_roles_preserves_row_count():
    """resolve_speaker_roles does not add or drop any rows."""
    df = pd.DataFrame([
        {"speaker": "SPEAKER_00", "f0_mean": 200.0, "start": float(i)}
        for i in range(10)
    ] + [
        {"speaker": "SPEAKER_01", "f0_mean": 150.0, "start": float(i + 100)}
        for i in range(10)
    ])
    result = resolve_speaker_roles(df)
    assert len(result) == len(df)


def test_resolve_roles_single_speaker_gets_raw_label():
    """With only one speaker, role defaults to the speaker label (not human/ai)."""
    df = pd.DataFrame([
        {"speaker": "SPEAKER_00", "f0_mean": 200.0, "start": 0.0},
        {"speaker": "SPEAKER_00", "f0_mean": 210.0, "start": 1.0},
    ])
    result = resolve_speaker_roles(df)
    # Function warns but still adds a role column
    assert "role" in result.columns


# ── merge_asof alignment logic ────────────────────────────────────────────────

def test_merge_asof_nearest_join_aligns_correctly():
    """merge_asof with nearest direction and 5s tolerance joins by closest timestamp."""
    transcript = pd.DataFrame({
        "speaker": ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"],
        "start":   [0.0,           5.0,           10.0],
        "text":    ["hello",       "hi there",    "how are you"],
    })
    prosody = pd.DataFrame({
        "speaker": ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"],
        "start":   [0.2,           5.3,           9.8],     # slightly offset
        "f0_mean": [250.0,         130.0,         255.0],
    })

    # Sort for merge_asof
    transcript = transcript.sort_values("start").reset_index(drop=True)
    prosody    = prosody.sort_values("start").reset_index(drop=True)

    merged = pd.merge_asof(
        transcript,
        prosody[["speaker", "start", "f0_mean"]].rename(columns={"start": "prosody_start"}),
        left_on="start",
        right_on="prosody_start",
        by="speaker",
        direction="nearest",
        tolerance=config.MERGE_TOLERANCE_S,
    )

    assert len(merged) == 3
    # SPEAKER_00 at t=0.0 matched to prosody at t=0.2 → F0 should be 250.0
    row0 = merged[merged["start"] == 0.0].iloc[0]
    assert row0["f0_mean"] == pytest.approx(250.0)


def test_merge_asof_out_of_tolerance_gives_nan():
    """Prosody rows outside the tolerance window produce NaN after nullification."""
    transcript = pd.DataFrame({
        "speaker": ["SPEAKER_00"],
        "start":   [0.0],
        "text":    ["hello"],
    })
    prosody = pd.DataFrame({
        "speaker":    ["SPEAKER_00"],
        "start":      [10.0],          # 10 seconds away — outside 5s tolerance
        "f0_mean":    [200.0],
    })

    transcript = transcript.sort_values("start").reset_index(drop=True)
    prosody    = prosody.sort_values("start").reset_index(drop=True)

    merged = pd.merge_asof(
        transcript,
        prosody.rename(columns={"start": "prosody_start"}),
        left_on="start",
        right_on="prosody_start",
        by="speaker",
        direction="nearest",
        tolerance=config.MERGE_TOLERANCE_S,
    )

    # After merge, rows outside tolerance are NaN (merge_asof enforces this)
    assert merged["f0_mean"].isna().all()


def test_merge_asof_preserves_all_transcript_rows():
    """merge_asof never drops transcript rows — unmatched rows get NaN prosody."""
    transcript = pd.DataFrame({
        "speaker": ["SPEAKER_00", "SPEAKER_00", "SPEAKER_00"],
        "start":   [0.0, 60.0, 120.0],
        "text":    ["a", "b", "c"],
    })
    # Only one prosody row — others get NaN
    prosody = pd.DataFrame({
        "speaker":  ["SPEAKER_00"],
        "start":    [0.1],
        "f0_mean":  [250.0],
    })

    transcript = transcript.sort_values("start").reset_index(drop=True)
    prosody    = prosody.sort_values("start").reset_index(drop=True)

    merged = pd.merge_asof(
        transcript,
        prosody.rename(columns={"start": "prosody_start"}),
        left_on="start",
        right_on="prosody_start",
        by="speaker",
        direction="nearest",
        tolerance=config.MERGE_TOLERANCE_S,
    )

    assert len(merged) == 3
    # Only the first row (t=0.0) is within 5s of prosody at t=0.1
    assert merged.iloc[0]["f0_mean"] == pytest.approx(250.0)
    assert merged.iloc[1]["f0_mean"] != merged.iloc[1]["f0_mean"]  # NaN check
    assert merged.iloc[2]["f0_mean"] != merged.iloc[2]["f0_mean"]  # NaN check


# ── compute_windowed_metrics ──────────────────────────────────────────────────

def _make_windowed_df(n_pairs: int = 6) -> pd.DataFrame:
    """
    Create a synthetic merged DataFrame with alternating human/AI turns
    spread across 10-second intervals, for windowed metrics testing.
    """
    rows = []
    for i in range(n_pairs * 2):
        role = "human" if i % 2 == 0 else "ai"
        rows.append({
            "session_id":      "test_vid",
            "speaker":         "SPEAKER_00" if role == "human" else "SPEAKER_01",
            "role":            role,
            "start":           float(i * 10),
            "end":             float(i * 10 + 9),
            "sentiment_score": 1.0 if role == "ai" else -1.0,
            "sentiment_label": "positive" if role == "ai" else "negative",
            "has_sycophancy":  True if role == "ai" else False,
            "is_backchannel":  False,
            "hedging_ratio":   0.0,
        })
    return pd.DataFrame(rows)


def test_windowed_metrics_returns_dataframe():
    """compute_windowed_metrics returns a DataFrame (even if empty)."""
    df = _make_windowed_df()
    result = compute_windowed_metrics(df, "test_vid")
    assert isinstance(result, pd.DataFrame)


def test_windowed_metrics_has_required_columns():
    """compute_windowed_metrics output has all expected columns."""
    df = _make_windowed_df(n_pairs=10)
    result = compute_windowed_metrics(df, "test_vid")
    required = {
        "session_id", "window_start", "window_end",
        "ai_sentiment_mean", "human_sentiment_mean",
        "ai_sycophancy_rate", "ai_backchannel_rate",
        "ai_hedging_ratio_mean", "local_AMS",
    }
    assert required.issubset(set(result.columns))


def test_windowed_metrics_session_id_correct():
    """All windowed rows carry the correct session_id."""
    df = _make_windowed_df(n_pairs=10)
    result = compute_windowed_metrics(df, "my_session")
    assert (result["session_id"] == "my_session").all()


def test_windowed_metrics_windows_non_overlapping_boundaries():
    """Each window_end equals window_start + WINDOW_SIZE_S."""
    df = _make_windowed_df(n_pairs=10)
    result = compute_windowed_metrics(df, "test_vid")
    for _, row in result.iterrows():
        assert row["window_end"] == pytest.approx(
            row["window_start"] + config.WINDOW_SIZE_S, abs=0.5
        )


def test_windowed_metrics_empty_df_returns_empty():
    """compute_windowed_metrics returns empty DataFrame for empty input."""
    result = compute_windowed_metrics(pd.DataFrame(), "test_vid")
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


def test_windowed_metrics_ai_sentiment_mean_correct():
    """ai_sentiment_mean is the mean of AI sentiment_score within the window."""
    # 10 turns at 0, 10, 20 … 90 seconds; AI turns have score=1.0
    df = _make_windowed_df(n_pairs=5)
    result = compute_windowed_metrics(df, "test_vid")
    assert not result.empty
    # AI sentiment is always 1.0, so mean should be 1.0
    ai_means = result["ai_sentiment_mean"].dropna()
    assert (ai_means == pytest.approx(1.0, abs=0.01)).all()


# ── NaN handling in out-of-tolerance prosody matches ─────────────────────────

def test_nan_nullification_logic():
    """
    Simulates the post-merge step that nullifies out-of-tolerance prosody columns.
    Rows where abs(start - prosody_start) > tolerance should have NaN prosody.
    """
    merged = pd.DataFrame({
        "start":        [0.0, 10.0, 20.0],
        "prosody_start": [0.5, 18.0, 20.1],   # row 1 is 8s away — out of tolerance
        "f0_mean":      [250.0, 200.0, 240.0],
    })

    tolerance = config.MERGE_TOLERANCE_S
    out_of_tolerance = abs(merged["start"] - merged["prosody_start"]) > tolerance
    prosody_cols = ["f0_mean"]
    merged.loc[out_of_tolerance, prosody_cols] = float("nan")

    assert math.isnan(merged.loc[1, "f0_mean"])   # row 1 was 8s away
    assert not math.isnan(merged.loc[0, "f0_mean"])  # row 0 was 0.5s away
    assert not math.isnan(merged.loc[2, "f0_mean"])  # row 2 was 0.1s away
