#!/usr/bin/env bash
# setup_env.sh — One-shot environment setup for the SRHI pipeline
#
# Run this once after cloning/downloading the project:
#   bash setup_env.sh
#
# After it completes, activate the environment in every new terminal:
#   source venv/bin/activate
#
# Then set your HuggingFace token:
#   export HF_TOKEN="hf_your_token_here"
#   (or add it to .env: echo 'HF_TOKEN=hf_your_token_here' > .env)

set -euo pipefail

PYTHON=python3.13
VENV_DIR="venv"

echo "========================================================"
echo "  SRHI Pipeline — Environment Setup"
echo "========================================================"
echo ""

# ── 1. Check Python version ───────────────────────────────────────────────────
echo "[1/6] Checking Python version..."
if ! command -v "$PYTHON" &>/dev/null; then
    echo "ERROR: $PYTHON not found. Install Python 3.13 first."
    exit 1
fi
PY_VERSION=$($PYTHON --version 2>&1)
echo "      Found: $PY_VERSION"

# ── 2. Create virtual environment ─────────────────────────────────────────────
echo ""
echo "[2/6] Creating virtual environment at ./$VENV_DIR ..."
$PYTHON -m venv "$VENV_DIR"
echo "      Created."

# ── 3. Activate and upgrade pip ───────────────────────────────────────────────
echo ""
echo "[3/6] Activating environment and upgrading pip..."
source "$VENV_DIR/bin/activate"
pip install --upgrade pip --quiet
echo "      pip upgraded to $(pip --version | awk '{print $2}')"

# ── 4. Install PyTorch first (pyannote.audio depends on it) ───────────────────
echo ""
echo "[4/6] Installing PyTorch (this may take a few minutes)..."
pip install "torch>=2.7.0" --quiet
echo "      PyTorch installed."

# ── 5. Install remaining dependencies ─────────────────────────────────────────
echo ""
echo "[5/6] Installing all remaining dependencies from requirements.txt..."
echo "      (This will take several minutes — models are large)"
pip install -r requirements.txt --quiet
echo "      Dependencies installed."

# ── 6. Verify key packages and device ─────────────────────────────────────────
echo ""
echo "[6/6] Verifying installation..."
$PYTHON - <<'PYEOF'
import sys

results = []

try:
    import torch
    mps = torch.backends.mps.is_available() and torch.backends.mps.is_built()
    results.append(f"  torch         {torch.__version__}  (MPS: {'yes' if mps else 'no'})")
except ImportError as e:
    results.append(f"  torch         MISSING — {e}")

try:
    import faster_whisper
    results.append(f"  faster-whisper  ok")
except ImportError as e:
    results.append(f"  faster-whisper  MISSING — {e}")

try:
    import pyannote.audio
    results.append(f"  pyannote.audio  {pyannote.audio.__version__}")
except ImportError as e:
    results.append(f"  pyannote.audio  MISSING — {e}")

try:
    import parselmouth
    results.append(f"  parselmouth     {parselmouth.__version__}")
except ImportError as e:
    results.append(f"  parselmouth     MISSING — {e}")

try:
    import transformers
    results.append(f"  transformers    {transformers.__version__}")
except ImportError as e:
    results.append(f"  transformers    MISSING — {e}")

try:
    import sentence_transformers
    results.append(f"  sentence-trans  {sentence_transformers.__version__}")
except ImportError as e:
    results.append(f"  sentence-trans  MISSING — {e}")

try:
    import pandas
    results.append(f"  pandas          {pandas.__version__}")
except ImportError as e:
    results.append(f"  pandas          MISSING — {e}")

try:
    import matplotlib
    results.append(f"  matplotlib      {matplotlib.__version__}")
except ImportError as e:
    results.append(f"  matplotlib      MISSING — {e}")

for r in results:
    print(r)
PYEOF

echo ""
echo "========================================================"
echo "  Setup complete!"
echo ""
echo "  Next steps:"
echo "  1. Activate the environment (do this in every new terminal):"
echo "     source venv/bin/activate"
echo ""
echo "  2. Set your HuggingFace token (needed for speaker diarization):"
echo "     echo 'HF_TOKEN=hf_your_token_here' > .env"
echo "     You must also accept the model license at:"
echo "     https://huggingface.co/pyannote/speaker-diarization-3.1"
echo "     https://huggingface.co/pyannote/segmentation-3.0"
echo ""
echo "  3. Put your video files in the DATA/ folder."
echo ""
echo "  4. Run the pipeline:"
echo "     python run_all.py"
echo "========================================================"
