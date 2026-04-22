"""
pipeline/turn_dynamics.py — Step 07: Conversation turn dynamics

Computes turn-level and session-level conversational dynamics from the
RTTM file and transcript. This is the most reliable output in the pipeline
because it derives entirely from timestamps — no model inference involved.

Session-level metrics (one row per speaker per video):
  total_speaking_time_sec    : Sum of all turn durations
  turn_count                 : Number of speaker turns
  mean_turn_duration_sec     : Average turn length
  floor_holding_ratio        : total_speaking_time / total_session_duration
  mean_response_latency_sec  : Avg gap from other speaker end → this speaker start
  overlap_count              : Number of overlapping segments with other speaker
  overlap_total_duration_sec : Total overlap time
  back_channel_rate          : Proportion of turns that are back-channels
  mean_turn_length_words     : Average words per turn (from transcript)

Turn-level pair metrics (one row per human→AI adjacent pair):
  turn_length_ratio          : ai_word_count / human_word_count
  Saved as per_video/<vid>/turn_length_pairs.csv for time-series visualisation

Output: outputs/per_video/<vid>/turn_dynamics.csv
        outputs/per_video/<vid>/turn_length_pairs.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import config
from pipeline.cache import (
    get_video_output_dir,
    should_skip,
    video_id_from_path,
)
from pipeline.logger import get_logger
from pipeline.transcribe import parse_rttm

log = get_logger(__name__)


def _compute_session_dynamics(
    rttm_turns: list[dict],
    transcript_df: pd.DataFrame,
    vid: str,
) -> pd.DataFrame:
    """
    Compute session-level turn dynamics for each speaker.

    Args:
        rttm_turns:    Parsed RTTM turns (start, end, speaker dicts).
        transcript_df: Transcript with word_count column.
        vid:           Session ID.

    Returns:
        DataFrame with one row per speaker.
    """
    rttm_df = pd.DataFrame(rttm_turns)
    rttm_df["duration"] = rttm_df["end"] - rttm_df["start"]

    session_duration = rttm_df["end"].max() - rttm_df["start"].min()
    log.debug("[07] Session duration: %.1f s, total turns: %d", session_duration, len(rttm_df))

    # Merge word counts from transcript
    transcript_agg = (
        transcript_df
        .groupby(["speaker", "start"])["word_count"]
        .first()
        .reset_index()
    )

    # Back-channel rates from sentiment.csv (if available — loaded externally)
    # Will be joined in the calling function

    rows = []
    speakers = rttm_df["speaker"].unique()

    for speaker in speakers:
        sp_df    = rttm_df[rttm_df["speaker"] == speaker].copy().reset_index(drop=True)
        other_df = rttm_df[rttm_df["speaker"] != speaker].copy().reset_index(drop=True)

        total_speaking_time = sp_df["duration"].sum()
        turn_count          = len(sp_df)
        mean_turn_duration  = sp_df["duration"].mean()
        floor_holding_ratio = total_speaking_time / session_duration if session_duration > 0 else 0.0

        # ── Response latency ──────────────────────────────────────────────────
        # For each turn by this speaker, find the most recent end time of
        # any turn by the OTHER speaker that ended before this turn starts.
        sp_starts  = sp_df["start"].values
        other_ends = other_df["end"].values

        latencies = []
        for s in sp_starts:
            preceding = other_ends[other_ends < s]
            if len(preceding) > 0:
                latencies.append(float(s - preceding.max()))
        mean_response_latency = float(np.mean(latencies)) if latencies else float("nan")

        # ── Overlaps ──────────────────────────────────────────────────────────
        overlap_count = 0
        overlap_total = 0.0
        for _, row in sp_df.iterrows():
            for _, orow in other_df.iterrows():
                overlap = max(
                    0.0,
                    min(row["end"], orow["end"]) - max(row["start"], orow["start"])
                )
                if overlap > 0:
                    overlap_count += 1
                    overlap_total += overlap

        # ── Word counts from transcript ───────────────────────────────────────
        sp_transcript = transcript_df[transcript_df["speaker"] == speaker]
        mean_turn_length_words = sp_transcript["word_count"].mean() if len(sp_transcript) > 0 else float("nan")

        rows.append({
            "session_id":               vid,
            "speaker":                  speaker,
            "total_speaking_time_sec":  round(total_speaking_time, 3),
            "turn_count":               turn_count,
            "mean_turn_duration_sec":   round(mean_turn_duration, 3),
            "floor_holding_ratio":      round(floor_holding_ratio, 4),
            "mean_response_latency_sec": round(mean_response_latency, 3) if not np.isnan(mean_response_latency) else float("nan"),
            "overlap_count":            overlap_count,
            "overlap_total_duration_sec": round(overlap_total, 3),
            "mean_turn_length_words":   round(mean_turn_length_words, 2) if not np.isnan(mean_turn_length_words) else float("nan"),
        })

    return pd.DataFrame(rows)


def _compute_turn_length_pairs(
    transcript_df: pd.DataFrame,
    vid: str,
) -> pd.DataFrame:
    """
    Compute turn-length ratio for each adjacent (any-speaker, next-speaker) pair.

    The ratio = this_turn_word_count / prev_turn_word_count.
    High values (>1) for AI turns = AI is talking more than the human.

    Args:
        transcript_df: Transcript sorted by start time.
        vid:           Session ID.

    Returns:
        DataFrame with turn_number, speaker, prev_speaker, turn_start,
        word_count, prev_word_count, turn_length_ratio.
    """
    df = transcript_df.sort_values("start").reset_index(drop=True)
    pairs = []

    for i in range(1, len(df)):
        curr = df.iloc[i]
        prev = df.iloc[i - 1]
        ratio = (
            float(curr["word_count"]) / float(prev["word_count"])
            if prev["word_count"] > 0
            else float("nan")
        )
        pairs.append({
            "session_id":        vid,
            "turn_number":       i,
            "speaker":           curr["speaker"],
            "prev_speaker":      prev["speaker"],
            "turn_start":        curr["start"],
            "word_count":        int(curr["word_count"]),
            "prev_word_count":   int(prev["word_count"]),
            "turn_length_ratio": round(ratio, 4) if not np.isnan(ratio) else float("nan"),
        })

    return pd.DataFrame(pairs)


def run(video_path: str | Path, force: bool = False) -> Path:
    """
    Step 07 entry point — compute turn dynamics for a single video.

    Args:
        video_path: Path to the source MP4.
        force:      Bypass cache and recompute.

    Returns:
        Path to turn_dynamics.csv.
    """
    video_path = Path(video_path)
    vid        = video_id_from_path(video_path)
    out_dir    = get_video_output_dir(video_path, config.PER_VIDEO_DIR)

    rttm_path      = out_dir / "diarization.rttm"
    transcript_p   = out_dir / "transcript.csv"
    dynamics_p     = out_dir / "turn_dynamics.csv"
    pairs_p        = out_dir / "turn_length_pairs.csv"

    log.debug("[07] video_id=%s", vid)

    if not rttm_path.exists():
        raise FileNotFoundError(f"[07] RTTM not found: {rttm_path} — run step 02 first")
    if not transcript_p.exists():
        raise FileNotFoundError(f"[07] transcript.csv not found: {transcript_p} — run step 03 first")

    if should_skip([dynamics_p, pairs_p], force):
        log.info("[07] Cached — skipping turn dynamics for %s", video_path.name)
        return dynamics_p

    rttm_turns    = parse_rttm(rttm_path)
    transcript_df = pd.read_csv(transcript_p)

    log.info("[07] Computing session dynamics for %s (%d RTTM turns, %d transcript turns)",
             video_path.name, len(rttm_turns), len(transcript_df))

    dynamics_df = _compute_session_dynamics(rttm_turns, transcript_df, vid)
    pairs_df    = _compute_turn_length_pairs(transcript_df, vid)

    # Attempt to enrich with back_channel_rate from sentiment.csv (optional)
    sentiment_p = out_dir / "sentiment.csv"
    if sentiment_p.exists():
        try:
            sent_df = pd.read_csv(sentiment_p)
            bc_rates = (
                sent_df.groupby("speaker")["is_backchannel"]
                .mean()
                .reset_index()
                .rename(columns={"is_backchannel": "back_channel_rate"})
            )
            dynamics_df = dynamics_df.merge(bc_rates, on="speaker", how="left")
            log.debug("[07] Merged back_channel_rate from sentiment.csv")
        except Exception as exc:
            log.warning("[07] Could not merge back_channel_rate: %s", exc)
            dynamics_df["back_channel_rate"] = float("nan")
    else:
        log.debug("[07] sentiment.csv not found — back_channel_rate will be NaN")
        dynamics_df["back_channel_rate"] = float("nan")

    dynamics_df.to_csv(dynamics_p, index=False)
    pairs_df.to_csv(pairs_p, index=False)

    log.info("[07] Saved: %s (%d rows)", dynamics_p.name, len(dynamics_df))
    log.info("[07] Saved: %s (%d pairs)", pairs_p.name, len(pairs_df))

    for _, row in dynamics_df.iterrows():
        log.debug(
            "[07]   %s: %.1f s speaking, %d turns, floor_hold=%.2f, "
            "latency=%.2f s, overlaps=%d",
            row["speaker"],
            row["total_speaking_time_sec"],
            row["turn_count"],
            row["floor_holding_ratio"],
            row["mean_response_latency_sec"] if pd.notna(row["mean_response_latency_sec"]) else -1,
            row["overlap_count"],
        )

    return dynamics_p


def run_batch(video_paths: list[Path], force: bool = False) -> list[Path]:
    """Run step 07 on a list of videos with tqdm progress."""
    log.info("[07] Starting batch turn dynamics for %d video(s)", len(video_paths))
    results = []
    for video_path in tqdm(video_paths, desc="[07] Turn dynamics", unit="video"):
        try:
            results.append(run(video_path, force=force))
        except Exception as exc:
            log.error("[07] FAILED for %s: %s", video_path.name, exc, exc_info=True)
            results.append(None)
    succeeded = sum(1 for r in results if r is not None)
    log.info("[07] Batch complete — %d/%d succeeded", succeeded, len(video_paths))
    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m pipeline.turn_dynamics <video_path> [--force]")
        sys.exit(1)
    run(Path(sys.argv[1]), force="--force" in sys.argv)
