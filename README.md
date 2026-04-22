# SRHI Pipeline — Multimodal Conversation Analysis

The Social Reward Hacking Index (SRHI) is a framework for identifying
patterns in AI companion behaviour where the system systematically rewards
social approval rather than pursuing honest, balanced interaction (Kirk et al., 2025).

This pipeline operationalises five core SRHI dimensions without an LLM-as-judge,
using computationally tractable signals from transcription, sentiment, prosody,
and turn dynamics.

Produces timestamped, per-turn and per-word data that provides empirical grounding
for close-reading analysis of **affective mirroring, validation cascades, position drift,
and sycophancy** 

This pipeline was created for research purposes 
Disclaimer: this project is with assistance from Claude Code 

---

## Prerequisites

Before you begin, you need:

1. **Python 3.13** — verify with: `python3.13 --version`
2. **ffmpeg** — verify with: `ffmpeg -version`
3. **A HuggingFace account** — free at [huggingface.co](https://huggingface.co)
4. **Accepted model licences** on HuggingFace:
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
   *(Click "Agree and access repository" on each page while logged in)*
5. **A HuggingFace access token** — generate one at:
   [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (read permissions are sufficient)

---

## Setup (do this once)

### Step 1 — Open Terminal
Open Terminal (Applications → Utilities → Terminal).

### Step 2 — Navigate to the project folder
```bash
cd "/Users/susangoldman/Documents/02 CULTURAL DATA & AI/04 ERP/03 ERP CODE PIPELINE"
```

### Step 3 — Run the setup script
```bash
bash setup_env.sh
```
This creates a virtual environment and installs all dependencies.
It takes several minutes — please wait for it to finish.

### Step 4 — Activate the environment
```bash
source venv/bin/activate
```
You will see `(venv)` at the start of your terminal prompt.

> **Important:** You must run `source venv/bin/activate` every time you open
> a new terminal window before using the pipeline.

### Step 5 — Set your HuggingFace token
```bash
echo 'HF_TOKEN=hf_your_token_here' > .env
```
Replace `hf_your_token_here` with your actual token from HuggingFace.

---

## Running the pipeline

### Run all steps on all videos
```bash
python run_all.py
```

### Run on a single video
```bash
python run_all.py --video "DATA/001 Rec Noah 06-03-2026mp4.mp4"
```

### Run specific steps only
```bash
# Re-run sentiment and merge steps only
python run_all.py --step 05 08 09
```

### Force recompute (ignore cached outputs)
```bash
python run_all.py --force
```

### Combined example: re-run sentiment for one video
```bash
python run_all.py --video "DATA/003 Rec noah 20-03.mp4" --step 05 08 09 --force
```

### See all options
```bash
python run_all.py --help
```

---

## What happens when you run it

The pipeline runs 9 steps for each video:

| Step | What it does | Output |
|------|-------------|--------|
| 01 | Extracts audio to 16 kHz WAV | `data/audio/<vid>.wav` |
| 02 | Identifies who spoke when (diarization) | `diarization.rttm` |
| 03 | Transcribes speech with word-level timestamps | `transcript.csv`, `words.csv` |
| 04 | Extracts pitch, intensity, speech rate per turn | `prosody.csv` |
| 05 | Analyses sentiment, sycophancy, hedging per turn | `sentiment.csv` |
| 06 | Computes sentence embeddings and lexical entrainment | `embeddings.csv` |
| 07 | Computes turn-taking dynamics from timestamps | `turn_dynamics.csv` |
| 08 | Merges everything; computes SRHI metrics | `merged.csv`, `srhi_summary.csv` |
| 09 | Generates all figures (per-session + cross-session) | `outputs/figures/` |

**Processing time per video:** approximately 15–30 minutes on an M4 MacBook Air.
The pipeline shows progress bars for each step.

---

## Outputs

All outputs go to `outputs/`. Nothing in `DATA/` is ever modified.

```
outputs/
├── per_video/
│   └── <video_id>/
│       ├── audio.wav           # Extracted audio (symlink)
│       ├── diarization.rttm    # Speaker turn timestamps
│       ├── transcript.csv      # Turn-level transcript with speaker labels
│       ├── words.csv           # Word-level with precise timestamps
│       ├── prosody.csv         # Pitch, intensity, speech rate per turn
│       ├── sentiment.csv       # Sentiment + sycophancy + hedging per turn
│       ├── embeddings.csv      # Lexical entrainment, semantic similarity
│       ├── turn_dynamics.csv   # Floor holding, response latency, overlaps
│       ├── turn_length_pairs.csv # Word-count ratio per adjacent turn pair
│       └── merged.csv          # All of the above merged into one table
│
└── merged/
    ├── all_videos.csv          # All sessions combined
    ├── srhi_summary.csv        # One row per session: AMS, VNAC, PSI, FDI, VCI, LE, SD
    └── windowed_metrics.csv    # 2-minute window time-series per session

outputs/figures/
├── per_session/<vid>/
│   ├── sentiment_flow.png
│   ├── sycophancy_flow.png
│   ├── prosody_flow.png
│   ├── turn_length_flow.png
│   ├── windowed_dynamics.png
│   ├── cascade_map.png
│   ├── entrainment_flow.png
│   ├── hedging_vs_sycophancy.png
│   └── metric_summary_panel.png
└── cross_session/
    ├── srhi_radar.png
    ├── srhi_bars.png
    ├── correlation_matrix.png
    ├── ams_vs_fdi.png
    ├── sentiment_trajectories_aligned.png
    ├── sycophancy_evolution.png
    ├── turn_length_evolution.png
    └── subject_comparison.png
```

---

## SRHI metrics

The `srhi_summary.csv` file contains seven metrics per session:

| Metric | Name | What it measures |
|--------|------|-----------------|
| AMS | Affective Mirroring Score | How much the AI matches the human's emotional tone (Pearson r) |
| VNAC | Validation after Negative Affect | How often AI responds positively after the human is negative |
| PSI | Position Shift Index | Whether the AI becomes more positive over the course of the session |
| FDI | Flattery Density Index | What fraction of AI turns contain flattery or sycophantic language |
| VCI | Validation Cascade Index | Longest consecutive run of positive AI turns |
| LE | Lexical Entrainment | How much the AI mirrors the human's vocabulary (Jaccard similarity) |
| SD | Semantic Drift | How much the AI's meaning shifts from early to late in the session |

---

## Log files

Every run creates a timestamped log directory:
```
logs/<YYYYMMDD_HHMMSS>/
├── pipeline_run.log    # Main run log
├── run_all.log         # Master runner events
├── pipeline_extract_audio.log
├── pipeline_diarize.log
└── ...                 # One log file per step module
```

Logs capture all `DEBUG`, `INFO`, `WARNING`, and `ERROR` messages.
The terminal only shows `INFO` and above.

---

## Adding new videos

1. Copy the new MP4 into the `DATA/` folder.
2. Run: `python run_all.py`

The pipeline automatically detects new files in `DATA/` and skips videos
that have already been fully processed. Only new videos are computed.

To force reprocessing of one specific video:
```bash
python run_all.py --video "DATA/new_video.mp4" --force
```

---

## Troubleshooting

**"HF_TOKEN is not set"**
Your `.env` file is missing or the token is wrong. Check:
```bash
cat .env
```
It should contain `HF_TOKEN=hf_...`. If not, run:
```bash
echo 'HF_TOKEN=hf_your_token_here' > .env
```

**"401 Unauthorized" when loading pyannote**
You have not accepted the model licence on HuggingFace. Visit:
- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/segmentation-3.0
Click "Agree and access repository" while logged in to your HF account.

**"(venv) not showing in terminal"**
You need to activate the environment first:
```bash
source venv/bin/activate
```

**"ModuleNotFoundError"**
The environment is not activated, or a package failed to install. Try:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

**Out of memory during step 02 (diarization)**
Close other applications before running step 02. The diarization model
uses approximately 2–3 GB of RAM.

**Step 03 is very slow**
This is expected. faster-whisper on M4 CPU processes roughly 15 minutes
of audio in about 5–10 minutes. Do not interrupt the process.

---

## Running the unit tests

The unit tests do NOT run the full pipeline (too slow). They test the
correctness of individual functions using synthetic data.

```bash
# Fast unit tests (no models loaded, no videos needed)
pytest tests/ -v --ignore=tests/test_one_video.py

# Smoke test (requires tests/sample_clip.mp4 — provide a short clip)
pytest tests/test_one_video.py -v
```
