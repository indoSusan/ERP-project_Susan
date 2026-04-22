"""
pipeline/embeddings.py — Step 06: Sentence embeddings, lexical entrainment, semantic drift

Produces 384-dimensional sentence embeddings using all-MiniLM-L6-v2
(~80 MB model, CPU-only). These embeddings power two conversation-level
dynamics measures central to SRHI:

  lexical_entrainment  (per AI turn):
      Jaccard similarity of content-word sets with the immediately preceding
      human turn (English stopwords removed). Measures vocabulary mirroring
      — a structural form of sycophancy where the AI repeats the human's
      own words back at them.

  semantic_sim_prev  (per turn, both speakers):
      Cosine similarity of turn embedding with the previous turn's embedding.
      High values = AI stays semantically close to the human's last utterance.

Outputs:
  embeddings.npy   — (n_turns × 384) float32 array, row-aligned with transcript.csv
  embeddings.csv   — session_id, speaker, start, lexical_entrainment, semantic_sim_prev

The .npy file is preserved for downstream SRHI computation (semantic drift SD).
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

log = get_logger(__name__)

# Common English function words that inflate lexical entrainment without
# reflecting meaningful vocabulary mirroring.
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "as", "is", "was", "are", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "shall", "can",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
    "us", "them", "my", "your", "his", "its", "our", "their",
    "this", "that", "these", "those", "what", "which", "who", "how",
    "not", "no", "so", "just", "also", "about", "up", "out", "there",
    "then", "than", "when", "where", "while", "into", "through",
})


def jaccard_similarity(a: str, b: str, filter_stopwords: bool = True) -> float:
    """
    Compute Jaccard similarity between the word-sets of two strings.

    Args:
        a, b:              Text strings to compare.
        filter_stopwords:  If True (default), remove common function words
                           before computing overlap so the score reflects
                           content-word mirroring rather than shared grammar.

    Returns:
        Float in [0, 1]. 0.0 if either string is empty after filtering.
    """
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if filter_stopwords:
        sa -= _STOPWORDS
        sb -= _STOPWORDS
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two 1-D vectors.

    Args:
        a, b: Numpy arrays of equal length.

    Returns:
        Float in [-1, 1]. Returns NaN if either vector is all-zero.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return float("nan")
    return float(np.dot(a, b) / (norm_a * norm_b))


def run(video_path: str | Path, force: bool = False) -> Path:
    """
    Step 06 entry point — compute embeddings and entrainment for a single video.

    Args:
        video_path: Path to the source MP4.
        force:      Bypass cache and recompute.

    Returns:
        Path to embeddings.csv.

    Raises:
        FileNotFoundError: If transcript.csv (step 03) is missing.
    """
    video_path = Path(video_path)
    vid        = video_id_from_path(video_path)
    out_dir    = get_video_output_dir(video_path, config.PER_VIDEO_DIR)

    transcript_p  = out_dir / "transcript.csv"
    embeddings_p  = out_dir / "embeddings.csv"
    embeddings_npy = out_dir / "embeddings.npy"

    log.debug("[06] video_id=%s", vid)

    if not transcript_p.exists():
        raise FileNotFoundError(f"[06] transcript.csv not found: {transcript_p} — run step 03 first")

    if should_skip([embeddings_p, embeddings_npy], force):
        log.info("[06] Cached — skipping embeddings for %s", video_path.name)
        return embeddings_p

    df = pd.read_csv(transcript_p)
    log.info("[06] Computing embeddings for %d turns from %s", len(df), video_path.name)

    # ── Sentence embeddings ───────────────────────────────────────────────────
    from sentence_transformers import SentenceTransformer  # deferred import

    log.info("[06] Loading embedding model: %s (cpu)", config.EMBEDDING_MODEL)
    model = SentenceTransformer(config.EMBEDDING_MODEL, device="cpu")

    texts = df["text"].fillna("").tolist()
    log.debug("[06] Encoding %d texts in batches of %d", len(texts), config.EMBEDDING_BATCH_SIZE)

    # Encode all turns — tqdm wraps the batch loop inside encode()
    embeddings = model.encode(
        texts,
        batch_size=config.EMBEDDING_BATCH_SIZE,
        show_progress_bar=False,    # We manage the bar ourselves below
        convert_to_numpy=True,
    )

    # Manual tqdm bar to show consistent progress with other steps
    log.info("[06] Embedding complete — shape: %s", embeddings.shape)

    # Save the raw embedding matrix (.npy) for SRHI semantic drift computation
    np.save(str(embeddings_npy), embeddings.astype(np.float32))
    log.info("[06] Saved: %s", embeddings_npy.name)

    # ── Per-turn derived metrics ──────────────────────────────────────────────
    lex_entrainment  = []
    semantic_sim_prev = []

    speakers = df["speaker"].tolist()
    turn_texts = df["text"].fillna("").tolist()

    # Build a lookup: turn index → most recent turn by the OTHER speaker
    # This lets us compute entrainment relative to the preceding human/AI turn
    prev_other_idx: dict[int, int] = {}
    last_by_speaker: dict[str, int] = {}

    for i, spk in enumerate(speakers):
        # Most recent turn by any other speaker
        other_speakers = {s: j for s, j in last_by_speaker.items() if s != spk}
        if other_speakers:
            prev_other_idx[i] = max(other_speakers.values())
        last_by_speaker[spk] = i

    for i in tqdm(range(len(df)), desc="[06] Lexical entrainment", unit="turn", leave=False):
        # Lexical entrainment: Jaccard vs most recent OTHER-speaker turn text
        if i in prev_other_idx:
            j = prev_other_idx[i]
            lex = jaccard_similarity(turn_texts[i], turn_texts[j])
        else:
            lex = float("nan")
        lex_entrainment.append(round(lex, 4) if not np.isnan(lex) else float("nan"))

        # Semantic similarity vs immediately preceding turn (any speaker)
        if i > 0:
            sim = cosine_similarity(embeddings[i], embeddings[i - 1])
            semantic_sim_prev.append(round(sim, 4) if not np.isnan(sim) else float("nan"))
        else:
            semantic_sim_prev.append(float("nan"))

    out_df = pd.DataFrame({
        "session_id":         df["session_id"],
        "speaker":            df["speaker"],
        "start":              df["start"],
        "lexical_entrainment": lex_entrainment,
        "semantic_sim_prev":   semantic_sim_prev,
    })

    out_df.to_csv(embeddings_p, index=False)

    # ── Summary log ───────────────────────────────────────────────────────────
    log.info("[06] Saved: %s (%d rows)", embeddings_p.name, len(out_df))
    le_vals = [v for v in lex_entrainment if not (isinstance(v, float) and np.isnan(v))]
    if le_vals:
        log.debug("[06]   mean lexical_entrainment: %.3f  max: %.3f", np.mean(le_vals), np.max(le_vals))
    sim_vals = [v for v in semantic_sim_prev if not (isinstance(v, float) and np.isnan(v))]
    if sim_vals:
        log.debug("[06]   mean semantic_sim_prev: %.3f", np.mean(sim_vals))

    return embeddings_p


def run_batch(video_paths: list[Path], force: bool = False) -> list[Path]:
    """Run step 06 on a list of videos with tqdm progress."""
    log.info("[06] Starting batch embeddings for %d video(s)", len(video_paths))
    results = []
    for video_path in tqdm(video_paths, desc="[06] Embeddings", unit="video"):
        try:
            results.append(run(video_path, force=force))
        except Exception as exc:
            log.error("[06] FAILED for %s: %s", video_path.name, exc, exc_info=True)
            results.append(None)
    succeeded = sum(1 for r in results if r is not None)
    log.info("[06] Batch complete — %d/%d succeeded", succeeded, len(video_paths))
    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m pipeline.embeddings <video_path> [--force]")
        sys.exit(1)
    run(Path(sys.argv[1]), force="--force" in sys.argv)
