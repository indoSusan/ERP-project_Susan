# Changelog

All notable changes to the SRHI Pipeline are documented here.

Format:
```
## [YYYY-MM-DD] — short description
### Added / Changed / Fixed
- ...
```

---

## [2026-03-30] — initial project scaffold

### Added
- Project directory structure: DATA/, data/, outputs/, pipeline/, docs/, tests/, logs/
- .gitignore covering Python, data files, model caches, and macOS artifacts
- .gitkeep placeholders in all empty directories

---

## [2026-03-30] — virtual environment and requirements

### Added
- requirements.txt with all dependencies pinned for Python 3.13 + Apple Silicon
- setup_env.sh: one-shot bootstrap script that creates venv, installs torch first,
  then all remaining dependencies, and prints a verification summary

---

## [2026-03-30] — git configuration and .gitignore

### Added
- Git repository initialised
- .gitignore: excludes DATA/, outputs/, logs/*.log, venv/, .env, model caches

---

## [2026-03-30] — step 1: audio extraction

### Added
- pipeline/extract_audio.py: ffmpeg-python extraction to 16 kHz mono WAV
- Caching: skips if audio.wav already exists; --force bypasses
- Symlink from outputs/per_video/<vid>/audio.wav to data/audio/<vid>.wav
- tqdm progress bar for batch processing
- Granular logging via pipeline/logger.py

---

## [2026-03-30] — step 2: speaker diarization

### Added
- pipeline/diarize.py: pyannote/speaker-diarization-3.1
- num_speakers=2 constraint (human + AI companion)
- MPS acceleration with CPU fallback on failure
- RTTM output format with per-turn start/duration/speaker

---

## [2026-03-30] — step 3: transcription with word-level timestamps

### Added
- pipeline/transcribe.py: faster-whisper medium model (CPU, int8)
- word_timestamps=True — word-level timing for maximum timestamp granularity
- RTTM alignment: Whisper segments assigned to speakers by max overlap
- Two outputs: transcript.csv (turn-level) and words.csv (word-level)
- Speaker labels kept as SPEAKER_00/01; role resolution deferred to step 08

---

## [2026-03-30] — step 4: prosody analysis

### Added
- pipeline/prosody.py: Parselmouth (Praat) per RTTM turn segment
- Features: f0_mean, f0_std, f0_range, intensity_mean, speech_rate,
  pause_count, pause_total_duration
- Robust handling of very short segments (< 50 ms) and unvoiced turns

---

## [2026-03-30] — step 5: sentiment and sycophancy classification

### Added
- pipeline/sentiment.py: cardiffnlp/twitter-roberta-base-sentiment-latest
- Lexicon-based features: sycophancy_score, has_sycophancy, hedging_ratio
- Rule-based features: is_backchannel, agreement_init
- 30-term sycophancy lexicon and 14 hedging markers (in config.py)
- Batch inference with tqdm progress bar

---

## [2026-03-30] — step 6: sentence embeddings and lexical entrainment

### Added
- pipeline/embeddings.py: sentence-transformers/all-MiniLM-L6-v2 (384-dim)
- Lexical entrainment: Jaccard similarity of AI turn vs preceding human turn
- Semantic similarity to previous turn (cosine similarity)
- embeddings.npy saved for downstream SRHI semantic drift (SD) computation

---

## [2026-03-30] — step 7: turn dynamics

### Added
- pipeline/turn_dynamics.py: pure pandas/numpy from RTTM + transcript
- Session-level: total_speaking_time, floor_holding_ratio, response_latency,
  overlap statistics, mean turn length
- Turn-level: turn_length_pairs.csv with AI/human word-count ratios over time

---

## [2026-03-30] — step 8: merge, SRHI computation, windowed metrics

### Added
- pipeline/merge.py: merge_asof (5s tolerance) joining all step outputs
- Speaker role resolution: higher mean f0 = human (f0 heuristic)
- SRHI sub-components: AMS, VNAC, PSI, FDI, VCI, LE, SD
- Sliding window metrics (2-min windows, 1-min step): temporal dynamics view
- Incremental updates to all_videos.csv, srhi_summary.csv, windowed_metrics.csv

---

## [2026-03-30] — step 9: matplotlib visualization pipeline

### Added
- pipeline/visualize.py: 9 per-session + 8 cross-session figures
- Continuous flow plots using EWMA smoothing (span=5)
- Time-normalized cross-session overlays (0–100% elapsed x-axis)
- Cascade map: horizontal coloured ribbon showing sentiment block sequence
- Cross-session: SRHI radar, bars, correlation heatmap, Noah vs Charlie comparison

---

## [2026-03-30] — run_all.py master runner with caching

### Added
- run_all.py: argparse CLI with --video, --step, --force, --list-steps
- Step 09 (visualize) runs once after all videos are processed
- Shared run_id for log correlation across all step log files
- Per-run summary: total steps run, failures, output locations

---

## [2026-03-30] — documentation

### Added
- README.md: full beginner-friendly setup and usage guide
- docs/pipeline_decisions.md: model selection rationale and limitations
- docs/output_schema.md: full column-by-column schema for merged.csv
- docs/srhi_metric.md: SRHI sub-component formulas and interpretation guide

---

## [2026-03-30] — tests

### Added
- tests/test_cache.py: unit tests for cache utilities
- tests/test_srhi.py: unit tests for all 7 SRHI functions with known inputs
- tests/test_sentiment.py: regex, scoring, and rule-based feature tests
- tests/test_merge.py: role resolution and merge alignment tests
- tests/test_one_video.py: ffmpeg smoke test using sample_clip.mp4
