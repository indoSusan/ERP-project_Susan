# Output Schema

Column-by-column documentation for all pipeline output files.

---

## `outputs/per_video/<vid>/merged.csv`

The primary analysis table. One row per speaker turn, all features merged.

| Column | Type | Unit | Range / Values | Description |
|--------|------|------|---------------|-------------|
| `session_id` | string | — | e.g. `001_rec_noah_06_03_2026mp4` | Unique identifier derived from the video filename |
| `speaker` | string | — | `SPEAKER_00`, `SPEAKER_01` | Raw pyannote diarization label |
| `role` | string | — | `human`, `ai` | Inferred speaker role (higher F0 = human) |
| `start` | float | seconds | ≥ 0 | Turn start time in seconds from audio beginning |
| `end` | float | seconds | > `start` | Turn end time in seconds |
| `turn_duration` | float | seconds | ≥ 0 | `end − start` |
| `text` | string | — | — | Transcribed text of the turn (from Whisper) |
| `word_count` | int | words | ≥ 0 | Number of words in `text` |
| `sentiment_label` | string | — | `positive`, `neutral`, `negative` | Predicted sentiment class (RoBERTa) |
| `sentiment_score` | float | — | −1, 0, +1 | Numeric mapping of sentiment label |
| `sentiment_conf` | float | — | [0, 1] | Model confidence in the predicted label |
| `sycophancy_score` | float | matches/word | ≥ 0 | Proportion of words matching the sycophancy lexicon |
| `has_sycophancy` | bool | — | True/False | True if `sycophancy_score > 0` |
| `hedging_ratio` | float | matches/word | ≥ 0 | Proportion of words matching hedging markers |
| `is_backchannel` | bool | — | True/False | True if turn is a short affirmative minimal response (≤4 words) |
| `agreement_init` | bool | — | True/False | True if turn opens with an agreement/sycophancy phrase |
| `f0_mean` | float | Hz | typically 50–600 | Mean pitch (voiced frames only) |
| `f0_std` | float | Hz | ≥ 0 | Standard deviation of pitch |
| `f0_range` | float | Hz | ≥ 0 | F0 max − F0 min (pitch variability) |
| `intensity_mean` | float | dB | typically 40–90 | Mean intensity (loudness) |
| `speech_rate` | float | voiced fraction/s | ≥ 0 | Proportion of voiced frames divided by segment duration |
| `pause_count` | int | pauses | ≥ 0 | Number of silences ≥ 200 ms within the turn |
| `pause_total_duration` | float | seconds | ≥ 0 | Total silence duration within the turn |
| `lexical_entrainment` | float | — | [0, 1] | Jaccard similarity of this turn's word-set with preceding other-speaker turn |
| `semantic_sim_prev` | float | — | [−1, 1] | Cosine similarity of this turn's embedding with the previous turn's embedding |

**NaN values:** Prosody columns are NaN for very short turns (< 50 ms), unvoiced segments, or where the merge tolerance was exceeded. Embedding columns are NaN for the first turn of each session.

---

## `outputs/per_video/<vid>/words.csv`

Word-level timestamps — the finest temporal resolution in the pipeline.

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `session_id` | string | — | Session identifier |
| `speaker` | string | — | `SPEAKER_00` or `SPEAKER_01` (inherited from turn) |
| `word_start` | float | seconds | Word onset time |
| `word_end` | float | seconds | Word offset time |
| `word` | string | — | Transcribed word (may include punctuation) |
| `probability` | float | [0, 1] | Whisper's confidence in this word |

**Use case:** Cross-reference a specific word from your qualitative reading with the exact timestamp. Load in a dataframe and filter by `session_id` and `word_start` to locate the moment in the recording.

---

## `outputs/per_video/<vid>/turn_length_pairs.csv`

Word-count ratio for each adjacent turn pair — for turn-length trajectory analysis.

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `session_id` | string | — | Session identifier |
| `turn_number` | int | — | Sequential turn index |
| `speaker` | string | — | This turn's speaker |
| `prev_speaker` | string | — | Previous turn's speaker |
| `turn_start` | float | seconds | This turn's start time |
| `word_count` | int | words | This turn's word count |
| `prev_word_count` | int | words | Previous turn's word count |
| `turn_length_ratio` | float | — | `word_count / prev_word_count`. Values > 1 mean this speaker talked more than the previous. |

---

## `outputs/per_video/<vid>/turn_dynamics.csv`

Session-level turn-taking summary — one row per speaker.

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `session_id` | string | — | Session identifier |
| `speaker` | string | — | `SPEAKER_00` or `SPEAKER_01` |
| `total_speaking_time_sec` | float | seconds | Sum of all turn durations |
| `turn_count` | int | turns | Number of speaker turns |
| `mean_turn_duration_sec` | float | seconds | Average turn length |
| `floor_holding_ratio` | float | [0, 1] | Fraction of session duration occupied by this speaker |
| `mean_response_latency_sec` | float | seconds | Average gap from other-speaker end → this-speaker start |
| `overlap_count` | int | occurrences | Number of overlapping intervals with the other speaker |
| `overlap_total_duration_sec` | float | seconds | Total overlap time |
| `mean_turn_length_words` | float | words | Average words per turn |
| `back_channel_rate` | float | [0, 1] | Proportion of turns that are back-channels |

---

## `outputs/merged/srhi_summary.csv`

Session-level SRHI metrics — one row per session.

| Column | Type | Range | Description |
|--------|------|-------|-------------|
| `session_id` | string | — | Session identifier |
| `AMS` | float | [−1, 1] | **Affective Mirroring Score**: Pearson r between human_sentiment(t) and ai_sentiment(t+1). Higher = stronger mirroring. |
| `VNAC` | int | ≥ 0 | **Validation after Negative Affect Count**: How often does the AI respond positively after the human is negative. |
| `PSI` | float | [−2, 2] | **Position Shift Index**: AI sentiment in Q4 minus Q1. Positive = AI drifts toward positive affect. |
| `FDI` | float | [0, 1] | **Flattery Density Index**: Proportion of AI turns containing sycophantic language. |
| `VCI` | int | ≥ 0 | **Validation Cascade Index**: Max consecutive positive AI turns. |
| `LE` | float | [0, 1] | **Lexical Entrainment**: Mean Jaccard similarity of AI turn vs preceding human turn. |
| `SD` | float | [0, 1] | **Semantic Drift**: Cosine similarity between Q1 and Q4 AI embeddings. Lower = more semantic drift. |
| `VMS_f0` | float | [−1, 1] | **Vocal Mirroring Score — pitch**: Pearson r between human f0_mean(t) and AI f0_mean(t+1). Higher = AI pitch tracks human pitch. |
| `VMS_intensity` | float | [−1, 1] | **Vocal Mirroring Score — loudness**: Pearson r between human intensity_mean(t) and AI intensity_mean(t+1). |
| `VMS_rate` | float | [−1, 1] | **Vocal Mirroring Score — speech rate**: Pearson r between human speech_rate(t) and AI speech_rate(t+1). |

---

## `outputs/merged/windowed_metrics.csv`

Time-series view of dynamics in 2-minute sliding windows (1-minute step).

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `session_id` | string | — | Session identifier |
| `window_start` | float | seconds | Window start time |
| `window_end` | float | seconds | Window end time (= `window_start + 120`) |
| `ai_sentiment_mean` | float | [−1, 1] | Mean AI sentiment in window |
| `human_sentiment_mean` | float | [−1, 1] | Mean human sentiment in window |
| `ai_sycophancy_rate` | float | [0, 1] | Proportion of AI turns with sycophancy in window |
| `ai_backchannel_rate` | float | [0, 1] | Proportion of AI turns that are back-channels in window |
| `ai_hedging_ratio_mean` | float | ≥ 0 | Mean AI hedging ratio in window |
| `local_AMS` | float | [−1, 1] | Within-window Pearson r (requires ≥ 3 human→ai pairs; else NaN) |

---

## `outputs/merged/all_videos.csv`

All `merged.csv` rows concatenated — same columns as `merged.csv` above.
Use this for cross-session analysis in Python or any tabular tool.

---

## Notes on NaN values

NaN values are expected and informative:
- `f0_mean` etc. are NaN for unvoiced turns or very short segments
- `lexical_entrainment` is NaN for the first turn (no preceding turn)
- `local_AMS` is NaN for windows with fewer than 3 consecutive human→ai pairs
- `AMS` and `SD` are NaN if the session has fewer than 4 turns per speaker

When reading the data in Python:
```python
import pandas as pd
df = pd.read_csv("outputs/merged/all_videos.csv")
df["f0_mean"].describe()  # NaN rows are automatically excluded from statistics
df[df["role"] == "ai"]["has_sycophancy"].value_counts()
```
