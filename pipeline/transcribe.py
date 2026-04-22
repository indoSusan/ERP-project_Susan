"""
pipeline/transcribe.py — Step 03: Transcription with word-level timestamps

Uses faster-whisper (medium model, int8 on CPU) to transcribe the audio,
then aligns each Whisper segment to a speaker label from the RTTM file
using maximum timestamp overlap.

Outputs two CSV files:

  transcript.csv — one row per speaker turn (aligned segment):
    session_id, speaker, start, end, text, word_count

  words.csv — one row per word (finest timestamp granularity):
    session_id, speaker, word_start, word_end, word, probability

Speaker labels are kept as SPEAKER_00 / SPEAKER_01 in this step.
Role assignment (human / ai) happens in step 08 (merge.py) after
prosody data is available for the pitch-based heuristic.

NOTE: ctranslate2 (the backend for faster-whisper) does NOT support MPS.
      Always use device="cpu" here regardless of TORCH_DEVICE.
"""

from __future__ import annotations

from collections import Counter
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


# ── RTTM parsing ──────────────────────────────────────────────────────────────

def parse_rttm(rttm_path: Path) -> list[dict]:
    """
    Parse an RTTM file into a list of speaker turn dicts.

    RTTM line format:
        SPEAKER <file> <channel> <start> <duration> <NA> <NA> <speaker> <NA> <NA>

    Returns:
        List of dicts with keys: start, end, speaker. Sorted by start time.
    """
    turns = []
    for line in rttm_path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.split()
        if len(parts) < 8 or parts[0] != "SPEAKER":
            log.debug("[03] Skipping non-SPEAKER RTTM line: %s", line[:60])
            continue
        start    = float(parts[3])
        duration = float(parts[4])
        speaker  = parts[7]
        turns.append({"start": start, "end": start + duration, "speaker": speaker})

    turns.sort(key=lambda t: t["start"])
    log.debug("[03] Parsed %d RTTM turns from %s", len(turns), rttm_path.name)
    return turns


def _assign_speaker(seg_start: float, seg_end: float, rttm_turns: list[dict]) -> str:
    """
    Assign a speaker label to a Whisper segment by maximum overlap with RTTM turns.

    Args:
        seg_start:   Whisper segment start time (seconds).
        seg_end:     Whisper segment end time (seconds).
        rttm_turns:  Parsed RTTM turns list.

    Returns:
        Speaker label string (e.g. 'SPEAKER_00') or 'UNKNOWN' if no overlap.
    """
    best_speaker = "UNKNOWN"
    best_overlap = 0.0
    for turn in rttm_turns:
        overlap = max(0.0, min(seg_end, turn["end"]) - max(seg_start, turn["start"]))
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = turn["speaker"]
    return best_speaker


# ── Core transcription ────────────────────────────────────────────────────────

def _transcribe(audio_path: Path, vid: str, rttm_turns: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run faster-whisper on audio_path and produce turn-level and word-level DataFrames.

    Args:
        audio_path:  Path to the 16 kHz mono WAV.
        vid:         Session ID string.
        rttm_turns:  Parsed speaker turns from RTTM.

    Returns:
        (transcript_df, words_df)
    """
    log.info("[03] Loading faster-whisper model: %s (cpu, int8)", config.WHISPER_MODEL_SIZE)

    from faster_whisper import WhisperModel  # deferred import

    model = WhisperModel(
        config.WHISPER_MODEL_SIZE,
        device="cpu",                       # ctranslate2 does NOT support MPS
        compute_type=config.WHISPER_COMPUTE_TYPE,
        cpu_threads=config.WHISPER_CPU_THREADS,
        num_workers=1,
    )

    log.info("[03] Transcribing %s (language=%s, beam_size=%d, vad=%s, word_ts=%s)",
             audio_path.name,
             config.WHISPER_LANGUAGE,
             config.WHISPER_BEAM_SIZE,
             config.WHISPER_VAD_FILTER,
             config.WHISPER_WORD_TS)

    segments_gen, info = model.transcribe(
        str(audio_path),
        language=config.WHISPER_LANGUAGE,
        beam_size=config.WHISPER_BEAM_SIZE,
        vad_filter=config.WHISPER_VAD_FILTER,
        word_timestamps=config.WHISPER_WORD_TS,
    )

    log.info("[03] Detected language: %s (probability=%.2f), duration=%.1f s",
             info.language, info.language_probability, info.duration)

    turn_rows: list[dict] = []
    word_rows: list[dict] = []

    # Consume the generator with a tqdm progress bar based on audio duration
    segments_list = list(
        tqdm(segments_gen, desc="[03] Transcribing segments", unit="seg", leave=False)
    )

    log.debug("[03] Processing %d segments", len(segments_list))

    for seg in segments_list:
        text  = (seg.text or "").strip()
        words = seg.words or []

        # Assign speaker per word using each word's own timestamps.
        # This prevents tail-of-segment misattribution when a Whisper segment
        # spans a speaker boundary.
        word_speakers = [
            _assign_speaker(w.start, w.end, rttm_turns) for w in words
        ]

        # Segment-level speaker = majority vote across words; fall back to
        # segment-level overlap when no word timestamps are available.
        if word_speakers:
            speaker = Counter(word_speakers).most_common(1)[0][0]
        else:
            speaker = _assign_speaker(seg.start, seg.end, rttm_turns)

        turn_rows.append({
            "session_id": vid,
            "speaker":    speaker,
            "start":      round(seg.start, 3),
            "end":        round(seg.end, 3),
            "text":       text,
            "word_count": len(text.split()) if text else 0,
        })

        for w, spk in zip(words, word_speakers):
            word_rows.append({
                "session_id":  vid,
                "speaker":     spk,
                "word_start":  round(w.start, 3),
                "word_end":    round(w.end, 3),
                "word":        w.word.strip(),
                "probability": round(w.probability, 4),
            })

    transcript_df = pd.DataFrame(turn_rows)
    words_df      = pd.DataFrame(word_rows)

    log.info("[03] Transcription complete — %d turns, %d words",
             len(transcript_df), len(words_df))

    # Log per-speaker word counts
    if not transcript_df.empty:
        for spk, grp in transcript_df.groupby("speaker"):
            log.debug("[03]   %s: %d turns, %d words",
                      spk, len(grp), grp["word_count"].sum())

    return transcript_df, words_df


# ── Public API ────────────────────────────────────────────────────────────────

def run(video_path: str | Path, force: bool = False) -> Path:
    """
    Step 03 entry point — transcribe a single video.

    Args:
        video_path: Path to the source MP4.
        force:      Bypass cache and re-transcribe even if outputs exist.

    Returns:
        Path to transcript.csv.

    Raises:
        FileNotFoundError: If step 01 or 02 outputs are missing.
    """
    video_path = Path(video_path)
    vid        = video_id_from_path(video_path)
    out_dir    = get_video_output_dir(video_path, config.PER_VIDEO_DIR)

    audio_path    = out_dir / "audio.wav"
    rttm_path     = out_dir / "diarization.rttm"
    transcript_p  = out_dir / "transcript.csv"
    words_p       = out_dir / "words.csv"

    log.debug("[03] video_id=%s", vid)

    if not audio_path.exists():
        raise FileNotFoundError(f"[03] WAV not found: {audio_path} — run step 01 first")
    if not rttm_path.exists():
        raise FileNotFoundError(f"[03] RTTM not found: {rttm_path} — run step 02 first")

    if should_skip([transcript_p, words_p], force):
        log.info("[03] Cached — skipping transcription for %s", video_path.name)
        return transcript_p

    rttm_turns = parse_rttm(rttm_path)
    transcript_df, words_df = _transcribe(audio_path, vid, rttm_turns)

    transcript_df.to_csv(transcript_p, index=False)
    words_df.to_csv(words_p, index=False)

    log.info("[03] Saved: %s (%d rows)", transcript_p.name, len(transcript_df))
    log.info("[03] Saved: %s (%d rows)", words_p.name, len(words_df))

    return transcript_p


def run_batch(video_paths: list[Path], force: bool = False) -> list[Path]:
    """Run step 03 on a list of videos with tqdm progress."""
    log.info("[03] Starting batch transcription for %d video(s)", len(video_paths))
    results = []
    for video_path in tqdm(video_paths, desc="[03] Transcribing", unit="video"):
        try:
            results.append(run(video_path, force=force))
        except Exception as exc:
            log.error("[03] FAILED for %s: %s", video_path.name, exc, exc_info=True)
            results.append(None)
    succeeded = sum(1 for r in results if r is not None)
    log.info("[03] Batch complete — %d/%d succeeded", succeeded, len(video_paths))
    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m pipeline.transcribe <video_path> [--force]")
        sys.exit(1)
    run(Path(sys.argv[1]), force="--force" in sys.argv)
