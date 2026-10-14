#!/usr/bin/env bash
set -euo pipefail

ROOT="${MORPH_ROOT:-/srv/morph}"
MODEL_ROOT="${MODEL_ROOT:-$ROOT/models}"
TARGET="${1:-lumina}"
HF_ENDPOINT_PRIMARY="${HF_ENDPOINT_PRIMARY:-https://huggingface.co}"
HF_ENDPOINT_MIRROR="${HF_ENDPOINT_MIRROR:-https://hf-mirror.com}"

mkdir -p "$MODEL_ROOT" "$ROOT/hf"
export HF_HOME="${HF_HOME:-$ROOT/hf}"

download_repo() {
  local repo="$1" local_dir="$2"
  local endpoint="${HF_ENDPOINT:-$HF_ENDPOINT_PRIMARY}"
  echo "Downloading $repo from $endpoint"
  if HF_ENDPOINT="$endpoint" hf download "$repo" --local-dir "$local_dir"; then
    return 0
  fi
  if [[ "${HF_MIRROR_FALLBACK:-1}" == "1" && "$endpoint" != "$HF_ENDPOINT_MIRROR" ]]; then
    echo "Primary Hugging Face endpoint failed; retrying via $HF_ENDPOINT_MIRROR" >&2
    HF_ENDPOINT="$HF_ENDPOINT_MIRROR" hf download "$repo" --local-dir "$local_dir"
    return $?
  fi
  return 1
}

case "$TARGET" in
  lumina)
    download_repo Alpha-VLLM/Lumina-mGPT-7B-768-Omni \
      "$MODEL_ROOT/lumina-mgpt-7b-768-omni"
    ;;
  anole)
    download_repo GAIR/Anole-7b-v0.1 "$MODEL_ROOT/anole-7b-v0.1"
    ;;
  emu3)
    download_repo BAAI/Emu3-Stage1 "$MODEL_ROOT/emu3-stage1"
    download_repo BAAI/Emu3-VisionTokenizer "$MODEL_ROOT/emu3-vision-tokenizer"
    ;;
  emu3-gen)
    download_repo BAAI/Emu3-Gen "$MODEL_ROOT/emu3-gen"
    download_repo BAAI/Emu3-VisionTokenizer "$MODEL_ROOT/emu3-vision-tokenizer"
    ;;
  *)
    echo "Usage: $0 {lumina|anole|emu3|emu3-gen}" >&2
    exit 2
    ;;
esac
