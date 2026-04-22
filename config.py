"""
config.py — Single source of truth for all pipeline settings.

Every script imports from here. Change values here to adjust the
entire pipeline. Nothing is hardcoded in individual step scripts.
"""

import os
import re
from pathlib import Path

from dotenv import load_dotenv

# Load HF_TOKEN and any other secrets from .env at import time
load_dotenv()

# ── Root paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent
DATA_DIR      = PROJECT_ROOT / "DATA"          # Source videos — never modified
AUDIO_DIR     = PROJECT_ROOT / "data" / "audio"
PER_VIDEO_DIR = PROJECT_ROOT / "outputs" / "per_video"
MERGED_DIR    = PROJECT_ROOT / "outputs" / "merged"
FIGURES_DIR   = PROJECT_ROOT / "outputs" / "figures"
LOGS_DIR      = PROJECT_ROOT / "logs"

# ── Audio extraction ──────────────────────────────────────────────────────────
AUDIO_SAMPLE_RATE = 16_000   # Hz — required by Whisper and pyannote
AUDIO_CHANNELS    = 1        # Mono

# ── Speaker diarization ───────────────────────────────────────────────────────
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
NUM_SPEAKERS      = 2        # Human + AI companion (hard constraint)
HF_TOKEN          = os.environ.get("HF_TOKEN", "")

# ── Transcription (faster-whisper) ────────────────────────────────────────────
WHISPER_MODEL_SIZE   = "medium"
WHISPER_LANGUAGE     = "en"     # English sessions
WHISPER_COMPUTE_TYPE = "int8"   # Safe on CPU; float16 is risky on MPS
WHISPER_BEAM_SIZE    = 5
WHISPER_VAD_FILTER   = True
WHISPER_WORD_TS      = True     # Word-level timestamps for maximum granularity
WHISPER_CPU_THREADS  = 8       # Use M4 performance cores

# ── Prosody (Praat via parselmouth) ───────────────────────────────────────────
PROSODY_PITCH_FLOOR  = 75.0    # Hz — suitable for mixed-gender/TTS range
PROSODY_PITCH_CEIL   = 600.0   # Hz
PROSODY_MIN_PAUSE_S  = 0.2     # Seconds — minimum silence counted as a pause
PROSODY_FRAME_STEP   = 0.01    # Seconds — Praat frame step for pitch analysis

# ── Sentiment (HuggingFace RoBERTa) ──────────────────────────────────────────
SENTIMENT_MODEL      = "cardiffnlp/twitter-roberta-base-sentiment-latest"
SENTIMENT_BATCH_SIZE = 32
SENTIMENT_SCORE_MAP  = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}

# ── Semantic embeddings (sentence-transformers) ───────────────────────────────
EMBEDDING_MODEL      = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 32
EMBEDDING_DIM        = 384

# ── Merge settings ────────────────────────────────────────────────────────────
MERGE_TOLERANCE_S = 5.0    # Seconds — tolerance for merge_asof alignment

# ── Sliding window for temporal metrics ──────────────────────────────────────
WINDOW_SIZE_S = 120.0      # 2-minute window
WINDOW_STEP_S = 60.0       # 1-minute step (50% overlap)
WINDOW_MIN_PAIRS = 3       # Minimum pairs needed to compute local AMS

# ── Time-normalised cross-session plots ──────────────────────────────────────
TIMELINE_BINS = 100        # Number of bins for 0–100% elapsed normalisation

# ── Back-channel detection thresholds ────────────────────────────────────────
BACKCHANNEL_MAX_WORDS = 4
BACKCHANNEL_STARTERS  = {
    "yes", "yeah", "yep", "right", "exactly", "mm", "mmm",
    "okay", "ok", "sure", "absolutely", "of course", "indeed",
}

# ── Sycophancy / Flattery Lexicon — English (30 terms) ───────────────────────
SYCOPHANCY_LEXICON = [
    "that's amazing",
    "you're so right",
    "you are so right",
    "absolutely",
    "exactly",
    "totally agree",
    "brilliant",
    "great point",
    "wonderful",
    "fantastic",
    "you're incredible",
    "you're incredibly",
    "so insightful",
    "i love that",
    "100%",
    "of course",
    "you're absolutely right",
    "that makes perfect sense",
    "i completely agree",
    "you're so smart",
    "you always know",
    "you're so wise",
    "i understand",
    "i hear you",
    "i get it",
    "that's so valid",
    "you deserve",
    "you're not alone",
    "couldn't agree more",
    "you hit the nail on the head",
    "you're amazing",
    "so true",
    "that's exactly right",
    "you look good",
    "that's really smart",
    "I'm here for you",
    "i'm here",
    "i'm with you",
    "that's amazing",
    "that's great",

]

# ── Epistemic hedging markers ─────────────────────────────────────────────────
HEDGING_MARKERS = [
    "i think",
    "i believe",
    "i'm not sure",
    "maybe",
    "perhaps",
    "possibly",
    "it might",
    "it could",
    "i guess",
    "sort of",
    "kind of",
    "in my opinion",
    "i feel like",
    "it seems",
    "might",
]

# Pre-compile regexes at import time for fast per-turn matching
SYCOPHANCY_REGEX = re.compile(
    "|".join(re.escape(t) for t in SYCOPHANCY_LEXICON),
    re.IGNORECASE,
)
HEDGING_REGEX = re.compile(
    "|".join(re.escape(t) for t in HEDGING_MARKERS),
    re.IGNORECASE,
)

# ── Device detection (MPS → CPU) ─────────────────────────────────────────────
def get_torch_device() -> str:
    """Return 'mps' on Apple Silicon if available, else 'cpu'."""
    try:
        import torch
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"
    except Exception:
        pass
    return "cpu"


TORCH_DEVICE = get_torch_device()

# ── Expected output filenames per video (used for cache checks) ───────────────
EXPECTED_OUTPUTS = {
    "01": "audio.wav",
    "02": "diarization.rttm",
    "03": ["transcript.csv", "words.csv"],
    "04": "prosody.csv",
    "05": "sentiment.csv",
    "06": ["embeddings.csv", "embeddings.npy"],
    "07": "turn_dynamics.csv",
    "08": "merged.csv",
}

# Expected per-session figure names (used for cache checks in step 09)
EXPECTED_FIGURES_PER_SESSION = [
    "sentiment_flow.png",
    "sycophancy_flow.png",
    "prosody_flow.png",
    "turn_length_flow.png",
    "windowed_dynamics.png",
    "cascade_map.png",
    "entrainment_flow.png",
    "hedging_vs_sycophancy.png",
    "metric_summary_panel.png",
]

EXPECTED_FIGURES_CROSS_SESSION = [
    "srhi_radar.png",
    "srhi_bars.png",
    "correlation_matrix.png",
    "ams_vs_fdi.png",
    "sentiment_trajectories_aligned.png",
    "turn_length_evolution.png",
    "sycophancy_evolution.png",
    "subject_comparison.png",
]
