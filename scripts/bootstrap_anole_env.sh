#!/usr/bin/env bash
set -euo pipefail

ROOT="${MORPH_ROOT:-/srv/morph}"
ENV_NAME="${ANOLE_ENV_NAME:-morph-anole}"
MAMBA_BIN="${MAMBA_BIN:-micromamba}"
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$ROOT/micromamba}"
CODE_ROOT="${ANOLE_CODE_ROOT:-$ROOT/external/anole}"

if [[ ! -d "$CODE_ROOT/.git" ]]; then
  echo "Anole code repository not found: $CODE_ROOT" >&2
  exit 1
fi

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda create -n "$ENV_NAME" python=3.10 -y
  conda activate "$ENV_NAME"
elif command -v "$MAMBA_BIN" >/dev/null 2>&1; then
  export MAMBA_ROOT_PREFIX
  "$MAMBA_BIN" create -y -n "$ENV_NAME" -c conda-forge python=3.10 pip
  eval "$("$MAMBA_BIN" shell hook --shell bash)"
  micromamba activate "$ENV_NAME"
else
  echo "Neither conda nor $MAMBA_BIN is available." >&2
  exit 1
fi

# Keep Anole's custom Transformers branch in its own environment. Anole's
# xformers path requires the Torch 2.4 ABI, so it is intentionally independent
# from Lumina's Torch 2.3 environment.
python -m pip install --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}" \
  torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0
python -m pip install xformers==0.0.27.post2
python -m pip install \
  huggingface_hub==0.23.4 tokenizers==0.19.1 regex \
  python-dotenv==1.0.1 numpy==1.26.4 Pillow==10.0.1 \
  packaging pyyaml tqdm requests safetensors
python -m pip install -e "$CODE_ROOT/transformers" --no-deps
python -m pip freeze > "${PROJECT_ROOT:-$PWD}/environment/anole.freeze.txt"

cat <<MSG
Anole environment installed. Set CKPT_PATH to:
  ${ANOLE_MODEL_ROOT:-$ROOT/models/anole-7b-v0.1}
Then run the official text2image.py smoke test.
MSG
