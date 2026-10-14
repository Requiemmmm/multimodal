#!/usr/bin/env bash
set -euo pipefail

ROOT="${MORPH_ROOT:-/srv/morph}"
ENV_NAME="${ENV_NAME:-morph-lumina}"
MAMBA_BIN="${MAMBA_BIN:-micromamba}"
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$ROOT/micromamba}"

mkdir -p "$ROOT"/{external,models,hf,runs,scratch}

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
  echo "Neither conda nor $MAMBA_BIN is available; install one or set MAMBA_BIN." >&2
  exit 1
fi

python -m pip install -U pip setuptools wheel huggingface_hub

if [[ "${INSTALL_TORCH:-1}" == "1" ]]; then
  # Install the CUDA wheel first.  Several optional requirements (notably
  # fairscale) declare an unconstrained torch dependency and would otherwise
  # pull a newer, incompatible CPU/CUDA stack before this pin is applied.
  python -m pip install --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}" \
    torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0
fi

if [[ ! -d "$ROOT/external/Lumina-mGPT/.git" ]]; then
  git clone https://github.com/Alpha-VLLM/Lumina-mGPT.git "$ROOT/external/Lumina-mGPT"
fi

cd "$ROOT/external/Lumina-mGPT"
# Install the official requirements except torch: the CUDA wheel is selected
# explicitly below so pip cannot silently replace it with a CPU build.
sed -e '/^torch\(vision\|audio\)\?==/d' \
    -e '/^transformers[><=]/d' \
    -e '/^bitsandbytes[><=]/d' \
    -e '/^fairscale[><=]/d' \
    requirements.txt > "$ROOT/scratch/lumina-requirements-no-torch.txt"
python -m pip install -r "$ROOT/scratch/lumina-requirements-no-torch.txt"
python -m pip install --no-deps fairscale==0.4.13
python -m pip install "transformers==4.43.3"
python -m pip install -e .

cat <<MSG
Base Lumina environment installed.

Next:
  1. Verify CUDA with Python and nvidia-smi.
  2. Install FlashAttention when compatible:
       MAX_JOBS=4 python -m pip install flash-attn --no-build-isolation
  3. Run download_models.sh.
  4. Place Chameleon VQ files under:
       $ROOT/external/Lumina-mGPT/ckpts/chameleon/tokenizer/
MSG
