"""
pipeline/sentiment.py — Step 05: Sentiment, sycophancy, hedging, and back-channel detection

Enriches each speaker turn from transcript.csv with:

  sentiment_label   : 'negative' / 'neutral' / 'positive' (cardiffnlp RoBERTa)
  sentiment_score   : Numeric mapping: negative=-1, neutral=0, positive=+1
  sycophancy_score  : (regex matches) / word_count — flattery/validation density
  has_sycophancy    : Boolean flag — True if sycophancy_score > 0
  hedging_ratio     : (hedging marker matches) / word_count — epistemic uncertainty
  is_backchannel    : True if turn is a short affirmative minimal response (≤4 words)
  agreement_init    : True if turn text starts with a sycophancy/agreement phrase

Output: outputs/per_video/<vid>/sentiment.csv

CAUTION: The cardiffnlp model was trained on English tweets. Its sentiment
labels are useful as directional signals for identifying turns to examine
qualitatively — they are not ground truth emotion annotations.
"""

from __future__ import annotations

from pathlib import Path

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


# ── Lexicon-based feature functions ──────────────────────────────────────────

def compute_sycophancy_score(text: str) -> float:
    """
    Compute the proportion of words that match the sycophancy lexicon.

    Args:
        text: Raw turn text.

    Returns:
        Float in [0, ∞) — number of matches / word count.
        Returns 0.0 for empty or non-string inputs.
    """
    if not text or not isinstance(text, str):
        return 0.0
    words = text.split()
    if not words:
        return 0.0
    matches = config.SYCOPHANCY_REGEX.findall(text)
    return len(matches) / len(words)


def compute_hedging_ratio(text: str) -> float:
    """
    Compute the proportion of words that match epistemic hedging markers.

    Args:
        text: Raw turn text.

    Returns:
        Float in [0, ∞) — number of hedging matches / word count.
    """
    if not text or not isinstance(text, str):
        return 0.0
    words = text.split()
    if not words:
        return 0.0
    matches = config.HEDGING_REGEX.findall(text)
    return len(matches) / len(words)


def detect_backchannel(text: str, word_count: int) -> bool:
    """
    Detect back-channel turns: short affirmative minimal responses.

    A back-channel is defined as a turn of <= BACKCHANNEL_MAX_WORDS words
    whose first word is in the BACKCHANNEL_STARTERS set.

    Args:
        text:       Raw turn text.
        word_count: Pre-computed word count.

    Returns:
        True if the turn is a back-channel response.
    """
    if not text or word_count > config.BACKCHANNEL_MAX_WORDS:
        return False
    first_word = text.strip().split()[0].lower().rstrip(".,!?") if text.strip() else ""
    return first_word in config.BACKCHANNEL_STARTERS


def detect_agreement_init(text: str) -> bool:
    """
    Detect whether a turn begins with an agreement / sycophancy phrase.

    Checks if ANY sycophancy lexicon term appears in the first 8 words.

    Args:
        text: Raw turn text.

    Returns:
        True if the opening of the turn contains agreement language.
    """
    if not text:
        return False
    opening = " ".join(text.split()[:8])
    return bool(config.SYCOPHANCY_REGEX.search(opening))


# ── Sentiment inference ───────────────────────────────────────────────────────

def _load_sentiment_pipeline():
    """
    Load the HuggingFace sentiment pipeline on CPU.

    The cardiffnlp model is small enough to run efficiently on M4 CPU.
    MPS support in transformers is partial for this model class.

    Returns:
        HuggingFace pipeline object.
    """
    from transformers import pipeline as hf_pipeline  # deferred import

    log.info("[05] Loading sentiment model: %s (cpu)", config.SENTIMENT_MODEL)
    pipe = hf_pipeline(
        "sentiment-analysis",
        model=config.SENTIMENT_MODEL,
        device=-1,          # -1 = CPU
        truncation=True,
        max_length=512,
        top_k=1,
    )

    # Sanity check — verify label format before running
    test = pipe(["This is a test."])[0]
    label = test[0]["label"].lower() if isinstance(test, list) else test["label"].lower()
    valid_labels = {"negative", "neutral", "positive"}
    if label not in valid_labels:
        log.warning(
            "[05] Unexpected sentiment label format: '%s'. Expected one of %s. "
            "Check model version or update SENTIMENT_SCORE_MAP in config.py.",
            label,
            valid_labels,
        )
    else:
        log.debug("[05] Label format verified: '%s'", label)

    return pipe


def _infer_sentiment(
    texts: list[str],
    pipe,
) -> list[dict]:
    """
    Run batched sentiment inference with tqdm progress bar.

    Args:
        texts: List of turn texts.
        pipe:  HuggingFace pipeline.

    Returns:
        List of dicts with keys: label (str), score (float).
    """
    batch_size = config.SENTIMENT_BATCH_SIZE
    results    = []
    batches    = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

    log.debug("[05] Running sentiment inference: %d texts in %d batch(es)",
              len(texts), len(batches))

    for batch in tqdm(batches, desc="[05] Sentiment inference", unit="batch", leave=False):
        raw = pipe(batch)
        for item in raw:
            # Handle both list[dict] and dict output formats
            if isinstance(item, list):
                item = item[0]
            label = item["label"].lower()
            results.append({"label": label, "score": item["score"]})

    return results


# ── Public API ────────────────────────────────────────────────────────────────

def run(video_path: str | Path, force: bool = False) -> Path:
    """
    Step 05 entry point — enrich transcript with sentiment and linguistic features.

    Args:
        video_path: Path to the source MP4.
        force:      Bypass cache and recompute.

    Returns:
        Path to sentiment.csv.

    Raises:
        FileNotFoundError: If transcript.csv (step 03) is missing.
    """
    video_path  = Path(video_path)
    vid         = video_id_from_path(video_path)
    out_dir     = get_video_output_dir(video_path, config.PER_VIDEO_DIR)

    transcript_p = out_dir / "transcript.csv"
    sentiment_p  = out_dir / "sentiment.csv"

    log.debug("[05] video_id=%s", vid)

    if not transcript_p.exists():
        raise FileNotFoundError(f"[05] transcript.csv not found: {transcript_p} — run step 03 first")

    if should_skip(sentiment_p, force):
        log.info("[05] Cached — skipping sentiment for %s", video_path.name)
        return sentiment_p

    df = pd.read_csv(transcript_p)
    log.info("[05] Loaded transcript: %d turns from %s", len(df), video_path.name)

    # ── Lexicon-based features (fast, no model) ───────────────────────────────
    log.debug("[05] Computing lexicon-based features (sycophancy, hedging, back-channel)")
    df["sycophancy_score"] = df["text"].fillna("").apply(compute_sycophancy_score).round(4)
    df["has_sycophancy"]   = df["sycophancy_score"] > 0
    df["hedging_ratio"]    = df["text"].fillna("").apply(compute_hedging_ratio).round(4)
    df["is_backchannel"]   = df.apply(
        lambda row: detect_backchannel(row["text"] if pd.notna(row["text"]) else "",
                                       int(row["word_count"])),
        axis=1,
    )
    df["agreement_init"]   = df["text"].fillna("").apply(detect_agreement_init)

    # ── Sentiment inference ───────────────────────────────────────────────────
    pipe   = _load_sentiment_pipeline()
    texts  = df["text"].fillna("").tolist()
    preds  = _infer_sentiment(texts, pipe)

    df["sentiment_label"] = [p["label"] for p in preds]
    df["sentiment_conf"]  = [round(p["score"], 4) for p in preds]
    df["sentiment_score"] = df.apply(
        lambda r: r["sentiment_conf"] if r["sentiment_label"] == "positive"
                  else -r["sentiment_conf"] if r["sentiment_label"] == "negative"
                  else 0.0,
        axis=1,
    ).round(4)

    df.to_csv(sentiment_p, index=False)

    # ── Summary log ───────────────────────────────────────────────────────────
    log.info("[05] Saved: %s (%d rows)", sentiment_p.name, len(df))

    for lbl in ["positive", "neutral", "negative"]:
        n = (df["sentiment_label"] == lbl).sum()
        log.debug("[05]   %s: %d turns (%.0f%%)", lbl, n, 100 * n / len(df) if len(df) else 0)

    syco_turns = df["has_sycophancy"].sum()
    bc_turns   = df["is_backchannel"].sum()
    log.debug("[05]   sycophancy detected: %d turns", syco_turns)
    log.debug("[05]   back-channels detected: %d turns", bc_turns)

    return sentiment_p


def run_batch(video_paths: list[Path], force: bool = False) -> list[Path]:
    """Run step 05 on a list of videos with tqdm progress."""
    log.info("[05] Starting batch sentiment analysis for %d video(s)", len(video_paths))
    results = []
    for video_path in tqdm(video_paths, desc="[05] Sentiment", unit="video"):
        try:
            results.append(run(video_path, force=force))
        except Exception as exc:
            log.error("[05] FAILED for %s: %s", video_path.name, exc, exc_info=True)
            results.append(None)
    succeeded = sum(1 for r in results if r is not None)
    log.info("[05] Batch complete — %d/%d succeeded", succeeded, len(video_paths))
    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m pipeline.sentiment <video_path> [--force]")
        sys.exit(1)
    run(Path(sys.argv[1]), force="--force" in sys.argv)
