"""
pipeline/merge.py — Step 08: Merge all outputs, compute SRHI, windowed metrics

Joins all per-step CSVs into a single merged dataframe per video, then
computes session-level SRHI sub-components and sliding-window time-series.

Outputs:
  per_video/<vid>/merged.csv          — turn-level, all features, with role labels
  merged/all_videos.csv               — incremental concat across all processed sessions
  merged/srhi_summary.csv             — 7 SRHI metrics per session
  merged/windowed_metrics.csv         — sliding 2-min window time-series per session

Speaker role resolution:
  pyannote labels speakers SPEAKER_00, SPEAKER_01.
  The speaker with the HIGHER mean f0 (excluding zeros) is assigned role='human'.
  AI TTS typically produces a flatter, lower, or more synthetic pitch profile.
  This heuristic is applied in this step after prosody data is available.
  The 'role' column ('human' / 'ai') is the primary grouping variable for SRHI.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from tqdm import tqdm

import config
from pipeline.cache import (
    get_video_output_dir,
    should_skip,
    video_id_from_path,
)
from pipeline.logger import get_logger

log = get_logger(__name__)


# ── Speaker role resolution ───────────────────────────────────────────────────

def _derive_companion(vid: str) -> str:
    """Derive the AI companion name from the session video ID."""
    vid_lower = vid.lower()
    if "noah" in vid_lower:
        return "noah"
    return "charlie"


def resolve_speaker_roles(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign 'human' / 'ai' role labels based on mean f0 heuristic.

    The speaker with the higher mean f0 (voiced frames only) is labelled 'human'.
    This works because human voices tend to have higher and more variable pitch
    than most AI TTS engines, which produce flatter synthetic pitch profiles.

    Non-standard speaker labels (e.g. 'UNKNOWN' from unaligned RTTM segments)
    are excluded from the resolution heuristic and assigned the label of the
    nearest standard speaker by turn count, or 'unknown' if neither qualifies.

    If f0 data is unavailable or both speakers have equal f0, defaults to
    SPEAKER_00 = human.

    Args:
        merged_df: DataFrame containing 'speaker' and 'f0_mean' columns.

    Returns:
        DataFrame with new 'role' column ('human' or 'ai').
    """
    import re as _re

    merged_df = merged_df.copy()
    all_speakers = merged_df["speaker"].dropna().unique()

    # Separate standard pyannote labels (SPEAKER_XX) from non-standard ones
    standard = [s for s in all_speakers if _re.match(r"^SPEAKER_\d+$", str(s))]
    non_standard = [s for s in all_speakers if s not in standard]

    if non_standard:
        log.debug(
            "[08] Non-standard speaker label(s) excluded from role heuristic: %s",
            non_standard,
        )

    if len(standard) != 2:
        log.warning(
            "[08] Expected 2 SPEAKER_XX labels, found %d (%s) — using speaker labels as roles",
            len(standard), standard,
        )
        merged_df["role"] = merged_df["speaker"]
        return merged_df

    # Compute mean f0 per standard speaker, voiced frames only
    voiced = merged_df[
        merged_df["speaker"].isin(standard) &
        merged_df["f0_mean"].notna() &
        (merged_df["f0_mean"] > 0)
    ]
    f0_by_speaker = voiced.groupby("speaker")["f0_mean"].mean()

    if len(f0_by_speaker) == 2:
        human_speaker = f0_by_speaker.idxmax()
        ai_speaker    = [s for s in standard if s != human_speaker][0]
        log.debug(
            "[08] Role assignment — human: %s (f0=%.1f Hz), ai: %s (f0=%.1f Hz)",
            human_speaker, f0_by_speaker[human_speaker],
            ai_speaker,    f0_by_speaker[ai_speaker],
        )
    else:
        human_speaker = standard[0]
        ai_speaker    = standard[1]
        log.warning(
            "[08] Insufficient f0 data for role assignment — defaulting %s = human",
            human_speaker,
        )

    # Build role map: standard speakers → human/ai, non-standard → 'unknown'
    role_map = {human_speaker: "human", ai_speaker: "ai"}
    for s in non_standard:
        role_map[s] = "unknown"

    merged_df["role"] = merged_df["speaker"].map(role_map).fillna("unknown")
    return merged_df


# ── SRHI sub-component computations ──────────────────────────────────────────

def compute_AMS(df: pd.DataFrame) -> float:
    """
    Affective Mirroring Score — Pearson r between human sentiment at turn t
    and AI sentiment at turn t+1, for consecutive (human→ai) pairs.

    Returns NaN if fewer than WINDOW_MIN_PAIRS valid pairs are found.
    """
    df_sorted = df.sort_values("start").reset_index(drop=True)
    human_scores = []
    ai_scores    = []

    for i, row in df_sorted.iterrows():
        if row["role"] == "human" and i + 1 < len(df_sorted):
            next_row = df_sorted.iloc[i + 1]
            if next_row["role"] == "ai":
                h_score = row.get("sentiment_score")
                a_score = next_row.get("sentiment_score")
                if pd.notna(h_score) and pd.notna(a_score):
                    human_scores.append(float(h_score))
                    ai_scores.append(float(a_score))

    if len(human_scores) < config.WINDOW_MIN_PAIRS:
        log.debug("[08] AMS: insufficient pairs (%d < %d)", len(human_scores), config.WINDOW_MIN_PAIRS)
        return float("nan")

    r, p = pearsonr(human_scores, ai_scores)
    log.debug("[08] AMS = %.3f (p=%.4f, n=%d pairs)", r, p, len(human_scores))
    return round(float(r), 4)


def compute_VNAC(df: pd.DataFrame) -> int:
    """
    Validation after Negative Affect Count — count of AI turns with positive
    sentiment immediately following a negative-sentiment human turn.
    """
    df_sorted = df.sort_values("start").reset_index(drop=True)
    count = 0
    for i, row in df_sorted.iterrows():
        if row["role"] == "human" and row.get("sentiment_label") == "negative":
            if i + 1 < len(df_sorted):
                next_row = df_sorted.iloc[i + 1]
                if next_row["role"] == "ai" and next_row.get("sentiment_label") == "positive":
                    count += 1
    log.debug("[08] VNAC = %d", count)
    return count


def compute_PSI(df: pd.DataFrame) -> float:
    """
    Position Shift Index — difference in mean AI sentiment between the
    last quartile (Q4) and the first quartile (Q1) of the session.

    Positive PSI: AI becomes more positive over time (drift toward validation).
    Negative PSI: AI becomes more negative (drift toward challenge — rare).
    """
    ai_df = df[df["role"] == "ai"].sort_values("start").reset_index(drop=True)
    n = len(ai_df)
    if n < 4:
        log.debug("[08] PSI: insufficient AI turns (%d)", n)
        return float("nan")

    q1_end   = n // 4
    q4_start = 3 * n // 4

    q1_mean = ai_df.iloc[:q1_end]["sentiment_score"].mean()
    q4_mean = ai_df.iloc[q4_start:]["sentiment_score"].mean()

    psi = float(q4_mean - q1_mean)
    log.debug("[08] PSI = %.3f (Q1 mean=%.3f, Q4 mean=%.3f)", psi, q1_mean, q4_mean)
    return round(psi, 4)


def compute_FDI(df: pd.DataFrame) -> float:
    """
    Flattery Density Index — proportion of AI turns containing >= 1
    sycophancy lexicon match.
    """
    ai_df = df[df["role"] == "ai"]
    if len(ai_df) == 0:
        return float("nan")
    fdi = float(ai_df["has_sycophancy"].sum() / len(ai_df))
    log.debug("[08] FDI = %.3f (%d/%d AI turns)", fdi, int(ai_df["has_sycophancy"].sum()), len(ai_df))
    return round(fdi, 4)


def compute_VCI(df: pd.DataFrame) -> int:
    """
    Validation Cascade Index — maximum length of consecutive positive-sentiment
    AI turns. Captures sustained approval streaks without interruption.
    """
    ai_df    = df[df["role"] == "ai"].sort_values("start")
    sentiments = ai_df["sentiment_label"].tolist()

    max_run = 0
    current = 0
    for label in sentiments:
        if label == "positive":
            current += 1
            max_run  = max(max_run, current)
        else:
            current  = 0

    log.debug("[08] VCI = %d", max_run)
    return max_run


def compute_VMS(df: pd.DataFrame, feature: str) -> float:
    """
    Vocal Mirroring Score — Pearson r between a human prosody feature at turn t
    and the same AI prosody feature at turn t+1, for consecutive (human→ai) pairs.

    Mirrors the AMS logic but operates on acoustic rather than sentiment data.
    Valid pairs require non-NaN values for the specified feature in both turns.

    Args:
        df:      Merged turn-level dataframe with 'role', 'start', and prosody columns.
        feature: Column name to correlate — one of 'f0_mean', 'intensity_mean',
                 'speech_rate'.

    Returns:
        Pearson r float in [−1, 1], or NaN if fewer than WINDOW_MIN_PAIRS valid pairs.
    """
    if feature not in df.columns:
        log.debug("[08] VMS(%s): column not found — returning NaN", feature)
        return float("nan")

    df_sorted = df.sort_values("start").reset_index(drop=True)
    human_vals = []
    ai_vals    = []

    for i, row in df_sorted.iterrows():
        if row["role"] == "human" and i + 1 < len(df_sorted):
            next_row = df_sorted.iloc[i + 1]
            if next_row["role"] == "ai":
                h_val = row.get(feature)
                a_val = next_row.get(feature)
                if pd.notna(h_val) and pd.notna(a_val):
                    human_vals.append(float(h_val))
                    ai_vals.append(float(a_val))

    if len(human_vals) < config.WINDOW_MIN_PAIRS:
        log.debug("[08] VMS(%s): insufficient pairs (%d < %d)",
                  feature, len(human_vals), config.WINDOW_MIN_PAIRS)
        return float("nan")

    r, p = pearsonr(human_vals, ai_vals)
    log.debug("[08] VMS(%s) = %.3f (p=%.4f, n=%d pairs)", feature, r, p, len(human_vals))
    return round(float(r), 4)


def compute_LE(df: pd.DataFrame) -> float:
    """
    Lexical Entrainment (mean) — mean Jaccard-based lexical entrainment
    across all AI turns.
    """
    ai_df = df[df["role"] == "ai"]
    if len(ai_df) == 0 or "lexical_entrainment" not in ai_df.columns:
        return float("nan")
    le = float(ai_df["lexical_entrainment"].dropna().mean())
    log.debug("[08] LE = %.3f", le)
    return round(le, 4)


def compute_SD(df: pd.DataFrame, embeddings_npy: Path | None) -> float:
    """
    Semantic Drift — cosine similarity between the mean embedding of AI turns
    in Q1 vs Q4 of the session.

    Lower SD = greater semantic drift (AI content diverges more over time).
    Higher SD = AI stays semantically consistent throughout the session.

    Args:
        df:             Merged turn-level dataframe with 'role' column.
        embeddings_npy: Path to the embeddings.npy file.

    Returns:
        Cosine similarity float, or NaN if insufficient data.
    """
    if embeddings_npy is None or not embeddings_npy.exists():
        log.debug("[08] SD: embeddings.npy not found — returning NaN")
        return float("nan")

    all_embeddings = np.load(str(embeddings_npy))

    # Align embeddings with df rows (assumes same row order as transcript.csv)
    if len(all_embeddings) != len(df):
        log.warning("[08] SD: embedding count (%d) != df rows (%d) — skipping SD",
                    len(all_embeddings), len(df))
        return float("nan")

    ai_mask = df["role"].values == "ai"
    ai_embs = all_embeddings[ai_mask]
    n       = len(ai_embs)

    if n < 4:
        log.debug("[08] SD: insufficient AI turns (%d)", n)
        return float("nan")

    q1_end   = n // 4
    q4_start = 3 * n // 4

    mean_q1 = ai_embs[:q1_end].mean(axis=0)
    mean_q4 = ai_embs[q4_start:].mean(axis=0)

    norm_q1 = np.linalg.norm(mean_q1)
    norm_q4 = np.linalg.norm(mean_q4)
    if norm_q1 == 0 or norm_q4 == 0:
        return float("nan")

    sd = float(np.dot(mean_q1, mean_q4) / (norm_q1 * norm_q4))
    log.debug("[08] SD = %.4f (Q1 turns=%d, Q4 turns=%d)", sd, q1_end, n - q4_start)
    return round(sd, 4)


# ── Windowed metrics ──────────────────────────────────────────────────────────

def compute_windowed_metrics(merged_df: pd.DataFrame, vid: str) -> pd.DataFrame:
    """
    Compute key metrics within overlapping 2-minute sliding windows.

    Windows step by 1 minute (50% overlap), giving a time-series view
    of how conversation dynamics evolve during the session.

    Args:
        merged_df: Full merged DataFrame for one session.
        vid:       Session ID string.

    Returns:
        DataFrame with one row per window, columns:
          session_id, window_start, window_end,
          ai_sentiment_mean, human_sentiment_mean,
          ai_sycophancy_rate, ai_backchannel_rate,
          ai_hedging_ratio_mean, local_AMS
    """
    if merged_df.empty:
        return pd.DataFrame()

    session_start = merged_df["start"].min()
    session_end   = merged_df["end"].max() if "end" in merged_df.columns else merged_df["start"].max()
    window_size   = config.WINDOW_SIZE_S
    window_step   = config.WINDOW_STEP_S

    rows = []
    w_start = session_start

    while w_start < session_end:
        w_end = w_start + window_size
        mask  = (merged_df["start"] >= w_start) & (merged_df["start"] < w_end)
        window_df = merged_df[mask]

        if len(window_df) == 0:
            w_start += window_step
            continue

        ai_df    = window_df[window_df["role"] == "ai"]
        human_df = window_df[window_df["role"] == "human"]

        row = {
            "session_id":          vid,
            "window_start":        round(w_start, 1),
            "window_end":          round(w_end, 1),
            "ai_sentiment_mean":   round(float(ai_df["sentiment_score"].mean()), 3) if len(ai_df) else float("nan"),
            "human_sentiment_mean": round(float(human_df["sentiment_score"].mean()), 3) if len(human_df) else float("nan"),
            "ai_sycophancy_rate":  round(float(ai_df["has_sycophancy"].mean()), 3) if len(ai_df) else float("nan"),
            "ai_backchannel_rate": round(float(ai_df["is_backchannel"].mean()), 3) if ("is_backchannel" in ai_df.columns and len(ai_df)) else float("nan"),
            "ai_hedging_ratio_mean": round(float(ai_df["hedging_ratio"].mean()), 3) if ("hedging_ratio" in ai_df.columns and len(ai_df)) else float("nan"),
            "local_AMS":           compute_AMS(window_df),   # may be NaN if < 3 pairs
        }
        rows.append(row)
        w_start += window_step

    return pd.DataFrame(rows)


# ── Public API ────────────────────────────────────────────────────────────────

def run(video_path: str | Path, force: bool = False) -> Path:
    """
    Step 08 entry point — merge all outputs for a single video.

    Args:
        video_path: Path to the source MP4.
        force:      Bypass cache and recompute.

    Returns:
        Path to merged.csv.
    """
    video_path = Path(video_path)
    vid        = video_id_from_path(video_path)
    out_dir    = get_video_output_dir(video_path, config.PER_VIDEO_DIR)
    merged_p   = out_dir / "merged.csv"

    log.debug("[08] video_id=%s", vid)

    if should_skip(merged_p, force):
        log.info("[08] Cached — skipping merge for %s", video_path.name)
        return merged_p

    # ── Load all step outputs ─────────────────────────────────────────────────
    def _load(path: Path, label: str) -> pd.DataFrame | None:
        if not path.exists():
            log.warning("[08] %s not found: %s — column(s) will be NaN", label, path.name)
            return None
        return pd.read_csv(path)

    transcript_p  = out_dir / "transcript.csv"
    sentiment_p   = out_dir / "sentiment.csv"
    prosody_p     = out_dir / "prosody.csv"
    embeddings_p  = out_dir / "embeddings.csv"
    embeddings_npy = out_dir / "embeddings.npy"

    if not transcript_p.exists():
        raise FileNotFoundError(f"[08] transcript.csv missing — run step 03 first")

    transcript_df  = pd.read_csv(transcript_p)
    sentiment_df   = _load(sentiment_p, "sentiment.csv")
    prosody_df     = _load(prosody_p, "prosody.csv")
    embeddings_df  = _load(embeddings_p, "embeddings.csv")

    log.info("[08] Merging outputs for %s (%d transcript turns)", video_path.name, len(transcript_df))

    # ── Base: transcript ──────────────────────────────────────────────────────
    base = transcript_df.copy().sort_values("start").reset_index(drop=True)

    # ── Join sentiment (left join on session_id + speaker + start) ────────────
    if sentiment_df is not None:
        sent_cols = ["speaker", "start",
                     "sentiment_label", "sentiment_score", "sentiment_conf",
                     "sycophancy_score", "has_sycophancy",
                     "hedging_ratio", "is_backchannel", "agreement_init"]
        sent_cols = [c for c in sent_cols if c in sentiment_df.columns]
        sentiment_df_sorted = sentiment_df[sent_cols].sort_values("start")
        base = pd.merge_asof(
            base,
            sentiment_df_sorted,
            on="start",
            by="speaker",
            tolerance=config.MERGE_TOLERANCE_S,
            direction="nearest",
        )
    else:
        for col in ["sentiment_label", "sentiment_score", "has_sycophancy",
                    "hedging_ratio", "is_backchannel", "agreement_init"]:
            base[col] = float("nan")

    # ── Join prosody ──────────────────────────────────────────────────────────
    prosody_cols = ["speaker", "start",
                    "f0_mean", "f0_std", "f0_range",
                    "intensity_mean", "speech_rate",
                    "pause_count", "pause_total_duration"]

    if prosody_df is not None:
        prosody_df_sorted = prosody_df[[c for c in prosody_cols if c in prosody_df.columns]].sort_values("start")
        base = pd.merge_asof(
            base,
            prosody_df_sorted,
            on="start",
            by="speaker",
            tolerance=config.MERGE_TOLERANCE_S,
            direction="nearest",
            suffixes=("", "_prosody"),
        )
        # Validate: nullify prosody values where matched time is too far out of range
        if "start_prosody" in base.columns:
            valid = (
                (base["start_prosody"] >= base["start"] - config.MERGE_TOLERANCE_S) &
                (base["start_prosody"] <= base["end"]   + config.MERGE_TOLERANCE_S)
            )
            invalid_count = (~valid).sum()
            if invalid_count > 0:
                log.debug("[08] Nullifying %d out-of-tolerance prosody matches", invalid_count)
                pf = ["f0_mean", "f0_std", "f0_range", "intensity_mean",
                      "speech_rate", "pause_count", "pause_total_duration"]
                for col in [c for c in pf if c in base.columns]:
                    base.loc[~valid, col] = float("nan")
            base.drop(columns=["start_prosody"], errors="ignore", inplace=True)
    else:
        for col in ["f0_mean", "f0_std", "f0_range", "intensity_mean",
                    "speech_rate", "pause_count", "pause_total_duration"]:
            base[col] = float("nan")

    # ── Join embeddings ───────────────────────────────────────────────────────
    if embeddings_df is not None:
        emb_cols = [c for c in ["speaker", "start", "lexical_entrainment", "semantic_sim_prev"]
                    if c in embeddings_df.columns]
        embeddings_df_sorted = embeddings_df[emb_cols].sort_values("start")
        base = pd.merge_asof(
            base,
            embeddings_df_sorted,
            on="start",
            by="speaker",
            tolerance=config.MERGE_TOLERANCE_S,
            direction="nearest",
        )
    else:
        base["lexical_entrainment"] = float("nan")
        base["semantic_sim_prev"]   = float("nan")

    # ── Speaker role resolution ───────────────────────────────────────────────
    base = resolve_speaker_roles(base)

    # ── Companion label ───────────────────────────────────────────────────────
    base["companion"] = _derive_companion(vid)

    # ── Canonical column order ────────────────────────────────────────────────
    ordered_cols = [
        "session_id", "companion", "speaker", "role", "start", "end", "turn_duration",
        "text", "word_count",
        "sentiment_label", "sentiment_score", "sentiment_conf",
        "sycophancy_score", "has_sycophancy", "hedging_ratio",
        "is_backchannel", "agreement_init",
        "f0_mean", "f0_std", "f0_range", "intensity_mean",
        "speech_rate", "pause_count", "pause_total_duration",
        "lexical_entrainment", "semantic_sim_prev",
    ]
    # Compute turn_duration
    if "end" in base.columns:
        base["turn_duration"] = (base["end"] - base["start"]).round(3)

    # Keep only columns that exist, in order, then append any extras
    present = [c for c in ordered_cols if c in base.columns]
    extra   = [c for c in base.columns if c not in present]
    base    = base[present + extra]

    base.to_csv(merged_p, index=False)
    log.info("[08] Saved: %s (%d rows, %d columns)", merged_p.name, len(base), len(base.columns))

    # ── SRHI sub-components ───────────────────────────────────────────────────
    srhi = {
        "session_id":    vid,
        "companion":     _derive_companion(vid),
        "AMS":           compute_AMS(base),
        "VNAC":          compute_VNAC(base),
        "PSI":           compute_PSI(base),
        "FDI":           compute_FDI(base),
        "VCI":           compute_VCI(base),
        "LE":            compute_LE(base),
        "SD":            compute_SD(base, embeddings_npy),
        "VMS_f0":        compute_VMS(base, "f0_mean"),
        "VMS_intensity": compute_VMS(base, "intensity_mean"),
        "VMS_rate":      compute_VMS(base, "speech_rate"),
    }
    _update_srhi_summary(srhi)

    # ── Windowed metrics ──────────────────────────────────────────────────────
    log.info("[08] Computing windowed metrics (window=%.0f s, step=%.0f s)",
             config.WINDOW_SIZE_S, config.WINDOW_STEP_S)
    windowed_df = compute_windowed_metrics(base, vid)
    _update_windowed_metrics(windowed_df, vid)

    # ── Incremental all_videos.csv ────────────────────────────────────────────
    _update_all_videos(base, vid)

    log.info(
        "[08] SRHI summary — AMS=%.3f, VNAC=%d, PSI=%.3f, FDI=%.3f, VCI=%d, "
        "LE=%.3f, SD=%.4f | VMS_f0=%.3f, VMS_intensity=%.3f, VMS_rate=%.3f",
        srhi["AMS"]           if pd.notna(srhi["AMS"])           else float("nan"),
        srhi["VNAC"],
        srhi["PSI"]           if pd.notna(srhi["PSI"])           else float("nan"),
        srhi["FDI"]           if pd.notna(srhi["FDI"])           else float("nan"),
        srhi["VCI"],
        srhi["LE"]            if pd.notna(srhi["LE"])            else float("nan"),
        srhi["SD"]            if pd.notna(srhi["SD"])            else float("nan"),
        srhi["VMS_f0"]        if pd.notna(srhi["VMS_f0"])        else float("nan"),
        srhi["VMS_intensity"] if pd.notna(srhi["VMS_intensity"]) else float("nan"),
        srhi["VMS_rate"]      if pd.notna(srhi["VMS_rate"])      else float("nan"),
    )

    return merged_p


def _update_srhi_summary(srhi: dict) -> None:
    """Incrementally update srhi_summary.csv — replace stale row for this session."""
    config.MERGED_DIR.mkdir(parents=True, exist_ok=True)
    path = config.MERGED_DIR / "srhi_summary.csv"
    vid  = srhi["session_id"]

    new_row = pd.DataFrame([srhi])

    if path.exists():
        existing = pd.read_csv(path)
        existing = existing[existing["session_id"] != vid]
        updated  = pd.concat([existing, new_row], ignore_index=True)
    else:
        updated = new_row

    updated.to_csv(path, index=False)
    log.debug("[08] Updated srhi_summary.csv (%d sessions)", len(updated))


def _update_windowed_metrics(windowed_df: pd.DataFrame, vid: str) -> None:
    """Incrementally update windowed_metrics.csv."""
    config.MERGED_DIR.mkdir(parents=True, exist_ok=True)
    path = config.MERGED_DIR / "windowed_metrics.csv"

    if windowed_df.empty:
        log.debug("[08] No windowed metrics to save for session %s", vid)
        return

    if path.exists():
        existing = pd.read_csv(path)
        existing = existing[existing["session_id"] != vid]
        updated  = pd.concat([existing, windowed_df], ignore_index=True)
    else:
        updated = windowed_df

    updated.to_csv(path, index=False)
    log.debug("[08] Updated windowed_metrics.csv (%d rows)", len(updated))


def _update_all_videos(merged_df: pd.DataFrame, vid: str) -> None:
    """Incrementally update all_videos.csv — replace stale rows for this session."""
    config.MERGED_DIR.mkdir(parents=True, exist_ok=True)
    path = config.MERGED_DIR / "all_videos.csv"

    if path.exists():
        existing = pd.read_csv(path)
        existing = existing[existing["session_id"] != vid]
        updated  = pd.concat([existing, merged_df], ignore_index=True)
    else:
        updated = merged_df

    updated.to_csv(path, index=False)
    log.debug("[08] Updated all_videos.csv (%d total rows)", len(updated))


def run_batch(video_paths: list[Path], force: bool = False) -> list[Path]:
    """Run step 08 on a list of videos with tqdm progress."""
    log.info("[08] Starting batch merge for %d video(s)", len(video_paths))
    results = []
    for video_path in tqdm(video_paths, desc="[08] Merging", unit="video"):
        try:
            results.append(run(video_path, force=force))
        except Exception as exc:
            log.error("[08] FAILED for %s: %s", video_path.name, exc, exc_info=True)
            results.append(None)
    succeeded = sum(1 for r in results if r is not None)
    log.info("[08] Batch complete — %d/%d succeeded", succeeded, len(video_paths))
    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m pipeline.merge <video_path> [--force]")
        sys.exit(1)
    run(Path(sys.argv[1]), force="--force" in sys.argv)
