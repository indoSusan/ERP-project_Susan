# Pipeline Decisions

Why each model was chosen, what alternatives were considered, and what
limitations to be aware of when interpreting the outputs.

---

## Step 02 — Speaker Diarization: pyannote/speaker-diarization-3.1

**What it does in plain language:**
Listens to the audio and figures out who is speaking at each moment,
producing a timeline of "who said what when". Outputs an RTTM file — a
text file listing each speaker's start time and duration.

**Why this model:**
pyannote 3.1 is the current state of the art for open-source speaker
diarization. It uses a cascade of three sub-models (voice activity detection,
speaker segmentation, and speaker embedding clustering) and has the best
published accuracy on benchmark datasets for 2-speaker scenarios.
We constrain it to `num_speakers=2` which significantly improves accuracy
when the number of speakers is known in advance.

**Why not alternatives:**
- *resemblyzer*: Older approach, lower accuracy on overlapping speech.
- *speechbrain*: More complex setup, no material accuracy gain for 2-speaker.
- *AWS Transcribe / Google Speech*: Cloud services, privacy concerns for
  research recordings, no local operation.

**Limitations for this use case:**
- The model was trained primarily on English speech data. Performance on
  AI TTS voices (which are synthetic) has not been independently benchmarked.
- Overlapping speech (both speakers talking at once) is often assigned to
  one speaker or discarded entirely.
- The model requires HuggingFace authentication and an accepted licence.
- **Critical caveat:** The heuristic for assigning 'human' vs 'ai' roles
  (higher mean F0 = human) is a proxy, not a ground truth label. Verify
  the role assignment by spot-checking `merged.csv` — confirm that the
  'human' speaker's text sounds like the researcher's voice.

---

## Step 03 — Transcription: faster-whisper (medium)

**What it does in plain language:**
Converts the audio speech to text, with precise timestamps for every word.
The 'medium' model balances accuracy and memory use for M4 MacBook Air.

**Why this model:**
faster-whisper is a reimplementation of OpenAI's Whisper that uses
CTranslate2 for CPU-optimised inference with int8 quantisation.
On an M4 CPU, it runs approximately 8–15× faster than real-time with
the `medium` model, making 15-minute sessions transcribable in 5–10 minutes.
Word-level timestamps (`word_timestamps=True`) are the key feature for this
pipeline — they enable the researcher to locate any specific word in the
recording by its precise timestamp.

**Why not alternatives:**
- *Whisper large-v3*: Higher accuracy but requires ~3 GB RAM and is 3×
  slower; acceptable if accuracy is a priority (change WHISPER_MODEL_SIZE
  in config.py).
- *Apple Speech Recognition*: No word-level timestamps, not accessible via Python.
- *Google Cloud Speech*: Privacy concerns, cloud dependency.

**Limitations for this use case:**
- Whisper sometimes hallucinates text during silence or background noise.
  The VAD filter (`vad_filter=True`) suppresses most hallucinations.
- Speaker assignment is inherited from the RTTM file: if pyannote
  incorrectly labels a turn, the transcript will also be incorrect.
- English language is specified explicitly (`language="en"`). If sessions
  contain code-switching, set `WHISPER_LANGUAGE = None` in config.py to
  enable per-segment language detection.

---

## Step 04 — Prosody: Parselmouth / Praat

**What it does in plain language:**
Analyses the acoustic properties of each speaker's voice during each turn:
pitch (how high or low the voice is), loudness, speech rate, and pauses.

**Why this model:**
Praat is the gold standard for speech analysis in linguistics research,
developed and maintained by the University of Amsterdam since 1992.
Parselmouth is its Python wrapper. It is pure CPU, deterministic (same
input always produces the same output), and very fast on M4.
The pitch floor (75 Hz) and ceiling (600 Hz) cover the full expected range
for both the human researcher's voice and the AI TTS output.

**Why not alternatives:**
- *librosa*: Good for music/general audio, but lacks linguistically-validated
  pitch extraction algorithms.
- *openSMILE*: Feature-rich but complex configuration; overkill for this use.
- *pyAudioAnalysis*: Abandoned, lacks word-level alignment.

**Limitations for this use case:**
- AI TTS voices have unnatural pitch profiles — very consistent, low variance.
  This means F0 statistics for the AI speaker are informative mainly as
  contrasts with the human, not as independent measures of affect.
- Speech rate is computed as voiced_frames / total_frames / duration —
  this is a proxy, not a count of syllables per second.
- Very short turns (< 50 ms) return NaN for all features — these are
  treated as non-voiced intervals.

---

## Step 05 — Sentiment: cardiffnlp/twitter-roberta-base-sentiment-latest

**What it does in plain language:**
Classifies each speaker turn as positive, neutral, or negative sentiment.
Also applies a custom lexicon to detect flattery and hedging language.

**Why this model:**
This RoBERTa model was fine-tuned on ~124M English tweets and achieves
strong results on short, informal, conversational text — which closely
matches the register of human–AI companion interactions. It outputs
calibrated probability scores alongside the label.
The 'latest' variant supports the three-class output (negative/neutral/positive)
directly without remapping.

**Why not alternatives:**
- *VADER*: Rule-based, works well on short text but lacks contextual
  understanding. Included as a fallback option if RoBERTa is too slow.
- *Llama/GPT-as-judge*: Excluded by design (no LLM-as-judge).
- *SentimentIntensityAnalyzer (NLTK)*: Same limitations as VADER.

**Limitations for this use case — important caveats:**
- This model was trained on tweets, not on therapeutic/companion conversations.
  The sentiment labels are useful as **directional signals** for identifying
  turns worthy of qualitative scrutiny, not as ground-truth emotion labels.
- AI TTS responses frequently receive 'positive' labels simply because they use
  affirmative, validating language — this is consistent with sycophancy but
  should not be read as evidence of genuine AI affect.
- **Do not cite raw sentiment_score values as empirical findings without
  qualification.** Always triangulate with the qualitative transcript evidence.

---

## Step 06 — Sentence Embeddings: sentence-transformers/all-MiniLM-L6-v2

**What it does in plain language:**
Converts each turn's text into a 384-number numerical representation (embedding)
that captures the meaning of the turn. Similar-meaning turns will have similar
embeddings. Used to measure lexical mirroring and semantic drift.

**Why this model:**
all-MiniLM-L6-v2 is the standard lightweight embedding model from the
sentence-transformers library. It is approximately 80 MB, runs in under 1
second per turn on CPU, and produces embeddings suitable for semantic
similarity computation. It was trained on 1B+ sentence pairs.

**Why not alternatives:**
- *text-embedding-3-large (OpenAI)*: Cloud API, privacy concerns.
- *all-mpnet-base-v2*: Higher quality but 3× slower; use if time permits.
- *TF-IDF vectors*: No semantic understanding; pure word overlap.

**Limitations for this use case:**
- Lexical entrainment uses simple Jaccard similarity (word-set overlap) rather
  than embeddings. This is intentional — it directly measures vocabulary
  copying without semantic conflation.
- Semantic drift (SD) measures the cosine similarity between the mean
  Q1 and Q4 AI turn embeddings. High similarity means consistent meaning;
  low similarity means the AI's content drifted. This does not distinguish
  *intentional* position change from *topic change*.

---

## Step 07 — Turn Dynamics: pure pandas / numpy

**What it does in plain language:**
Computes conversational metrics from the RTTM timestamps alone — no model
inference. This is the most reliable output in the pipeline because it
derives entirely from precise timestamps.

**Why no model:**
Diarization timestamps are deterministic and do not require inference.
All metrics (floor holding, response latency, overlaps, turn length)
are exact calculations from start/end times, not approximations.

**Limitations:**
- Response latency includes silence between turns, which may include
  processing time for the AI companion to generate a response. This
  cannot be separated from natural conversational pause without
  additional information about the application's response generation timing.
- Overlap detection counts any temporal co-occurrence of two speaker
  intervals. In AI companion apps, true simultaneous speech (interruption)
  is rare — most "overlaps" are RTTM annotation artefacts from diarization.

---

## Step 08 — Merge: pandas merge_asof

**Why merge_asof (not a standard join):**
The different step outputs are aligned by timestamp, not by a shared row ID.
`merge_asof` performs a nearest-neighbour join on a sorted time column,
which correctly handles the case where Whisper segment boundaries don't
perfectly coincide with RTTM turn boundaries.

**Tolerance setting (5 seconds):**
The 5-second tolerance means a prosody measurement from up to 5 seconds
away from a transcript turn start can be matched. This is conservative
for 15-minute sessions and ensures most turns receive prosody data.
Out-of-tolerance matches are nullified (set to NaN) rather than kept.

---

## What is NOT in this pipeline (and why)

| Excluded feature | Reason |
|-----------------|--------|
| LLM-as-judge scoring | Circular if studying AI behaviour; expensive; not reproducible |
| Facial expression analysis | Videos are screen recordings — no face visible |
| Topic modelling (LDA/NMF) | Too noisy for 15-minute sessions; manual topic labelling preferred |
| Emotion recognition (Ekman) | Acoustic emotion classifiers are notoriously unreliable; treat sentiment as a proxy only |
| Acoustic event detection | Not relevant for structured 2-speaker conversation |
