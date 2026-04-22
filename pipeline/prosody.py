"""
pipeline/prosody.py — Step 04: Prosody feature extraction

Extracts prosodic features for each speaker turn using Parselmouth
(Python wrapper for Praat). Operates on RTTM turn boundaries — not
Whisper segments — for consistent alignment with diarization output.

Features extracted per turn:
  - f0_mean, f0_std, f0_range     : Pitch statistics (Hz), voiced frames only
  - intensity_mean                 : RMS intensity (dB)
  - speech_rate                    : Fraction of voiced frames / segment duration
  - pause_count                    : Number of silences >= PROSODY_MIN_PAUSE_S
  - pause_total_duration           : Total silence duration (seconds)

Output: outputs/per_video/<vid>/prosody.csv
Columns: session_id, speaker, start, end, f0_mean, f0_std, f0_range,
         intensity_mean, speech_rate, pause_count, pause_total_duration

Parselmouth is pure CPU — fast on M4 without MPS.
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


def _extract_turn_prosody(
    sound,              # parselmouth.Sound
    start: float,
    end: float,
) -> dict:
    """
    Extract prosodic features for a single speaker turn segment.

    Args:
        sound:  Full parselmouth.Sound object.
        start:  Turn start time in seconds.
        end:    Turn end time in seconds.

    Returns:
        Dict of prosodic feature values. All NaN on failure.
    """
    import parselmouth  # deferred import

    empty = {
        "f0_mean": float("nan"),
        "f0_std":  float("nan"),
        "f0_range": float("nan"),
        "intensity_mean": float("nan"),
        "speech_rate": float("nan"),
        "pause_count": 0,
        "pause_total_duration": 0.0,
    }

    duration = end - start
    if duration < 0.05:
        log.debug("[04] Turn too short (%.3f s) — returning NaN", duration)
        return empty

    try:
        seg = sound.extract_part(
            from_time=start,
            to_time=end,
            preserve_times=False,
        )
    except Exception as exc:
        log.warning("[04] extract_part failed at [%.2f, %.2f]: %s", start, end, exc)
        return empty

    # ── Pitch (F0) ─────────────────────────────────────────────────────────────
    try:
        pitch = seg.to_pitch(
            time_step=config.PROSODY_FRAME_STEP,
            pitch_floor=config.PROSODY_PITCH_FLOOR,
            pitch_ceiling=config.PROSODY_PITCH_CEIL,
        )
        f0_values = pitch.selected_array["frequency"]  # 0 = unvoiced
        voiced    = f0_values[f0_values > 0]

        if len(voiced) > 0:
            f0_mean  = float(np.mean(voiced))
            f0_std   = float(np.std(voiced))
            f0_range = float(np.ptp(voiced))
        else:
            f0_mean = f0_std = f0_range = float("nan")
    except Exception as exc:
        log.warning("[04] Pitch extraction failed: %s", exc)
        f0_values = np.array([])
        voiced    = np.array([])
        f0_mean = f0_std = f0_range = float("nan")

    # ── Intensity ──────────────────────────────────────────────────────────────
    try:
        intensity    = seg.to_intensity(minimum_pitch=config.PROSODY_PITCH_FLOOR)
        intensity_mean = float(np.mean(intensity.values)) if intensity.n_frames > 0 else float("nan")
    except Exception as exc:
        log.warning("[04] Intensity extraction failed: %s", exc)
        intensity_mean = float("nan")

    # ── Speech rate (voiced fraction / duration) ───────────────────────────────
    total_frames  = len(f0_values)
    voiced_frames = len(voiced)
    if total_frames > 0 and duration > 0:
        speech_rate = voiced_frames / total_frames / duration
    else:
        speech_rate = float("nan")

    # ── Pauses (unvoiced gaps >= PROSODY_MIN_PAUSE_S) ─────────────────────────
    pause_count = 0
    pause_total = 0.0

    if len(f0_values) > 0:
        frame_step  = config.PROSODY_FRAME_STEP
        is_voiced   = (f0_values > 0).astype(int)
        in_pause    = False
        pause_start = 0

        for i, v in enumerate(is_voiced):
            if v == 0 and not in_pause:
                in_pause    = True
                pause_start = i
            elif v == 1 and in_pause:
                pause_dur = (i - pause_start) * frame_step
                if pause_dur >= config.PROSODY_MIN_PAUSE_S:
                    pause_count += 1
                    pause_total += pause_dur
                in_pause = False

        # Handle trailing pause
        if in_pause:
            pause_dur = (len(is_voiced) - pause_start) * frame_step
            if pause_dur >= config.PROSODY_MIN_PAUSE_S:
                pause_count += 1
                pause_total += pause_dur

    return {
        "f0_mean":              round(f0_mean, 3) if not np.isnan(f0_mean) else float("nan"),
        "f0_std":               round(f0_std, 3) if not np.isnan(f0_std) else float("nan"),
        "f0_range":             round(f0_range, 3) if not np.isnan(f0_range) else float("nan"),
        "intensity_mean":       round(intensity_mean, 3) if not np.isnan(intensity_mean) else float("nan"),
        "speech_rate":          round(speech_rate, 4) if not np.isnan(speech_rate) else float("nan"),
        "pause_count":          pause_count,
        "pause_total_duration": round(pause_total, 3),
    }


def run(video_path: str | Path, force: bool = False) -> Path:
    """
    Step 04 entry point — extract prosody features for a single video.

    Args:
        video_path: Path to the source MP4.
        force:      Bypass cache and recompute even if prosody.csv exists.

    Returns:
        Path to prosody.csv.

    Raises:
        FileNotFoundError: If step 01 or 02 outputs are missing.
    """
    import parselmouth  # deferred import — verify at start of step

    video_path = Path(video_path)
    vid        = video_id_from_path(video_path)
    out_dir    = get_video_output_dir(video_path, config.PER_VIDEO_DIR)

    audio_path  = out_dir / "audio.wav"
    rttm_path   = out_dir / "diarization.rttm"
    prosody_p   = out_dir / "prosody.csv"

    log.debug("[04] video_id=%s", vid)

    if not audio_path.exists():
        raise FileNotFoundError(f"[04] WAV not found: {audio_path} — run step 01 first")
    if not rttm_path.exists():
        raise FileNotFoundError(f"[04] RTTM not found: {rttm_path} — run step 02 first")

    if should_skip(prosody_p, force):
        log.info("[04] Cached — skipping prosody for %s", video_path.name)
        return prosody_p

    log.info("[04] Loading audio into Praat: %s", audio_path.name)
    sound = parselmouth.Sound(str(audio_path))
    log.debug("[04] Audio loaded: duration=%.1f s, sample_rate=%.0f Hz",
              sound.duration, sound.sampling_frequency)

    rttm_turns = parse_rttm(rttm_path)
    log.info("[04] Extracting prosody for %d speaker turns", len(rttm_turns))

    rows = []
    for turn in tqdm(rttm_turns, desc="[04] Extracting prosody", unit="turn", leave=False):
        feats = _extract_turn_prosody(sound, turn["start"], turn["end"])
        rows.append({
            "session_id": vid,
            "speaker":    turn["speaker"],
            "start":      round(turn["start"], 3),
            "end":        round(turn["end"], 3),
            **feats,
        })

    df = pd.DataFrame(rows)
    df.to_csv(prosody_p, index=False)

    log.info("[04] Saved: %s (%d rows)", prosody_p.name, len(df))

    # Log summary statistics per speaker
    for spk, grp in df.groupby("speaker"):
        log.debug(
            "[04]   %s: f0_mean=%.1f±%.1f Hz, intensity=%.1f dB, speech_rate=%.3f",
            spk,
            grp["f0_mean"].mean(),
            grp["f0_mean"].std(),
            grp["intensity_mean"].mean(),
            grp["speech_rate"].mean(),
        )

    return prosody_p


def run_batch(video_paths: list[Path], force: bool = False) -> list[Path]:
    """Run step 04 on a list of videos with tqdm progress."""
    log.info("[04] Starting batch prosody extraction for %d video(s)", len(video_paths))
    results = []
    for video_path in tqdm(video_paths, desc="[04] Prosody", unit="video"):
        try:
            results.append(run(video_path, force=force))
        except Exception as exc:
            log.error("[04] FAILED for %s: %s", video_path.name, exc, exc_info=True)
            results.append(None)
    succeeded = sum(1 for r in results if r is not None)
    log.info("[04] Batch complete — %d/%d succeeded", succeeded, len(video_paths))
    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m pipeline.prosody <video_path> [--force]")
        sys.exit(1)
    run(Path(sys.argv[1]), force="--force" in sys.argv)
