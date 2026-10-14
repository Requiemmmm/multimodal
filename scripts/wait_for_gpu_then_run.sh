#!/usr/bin/env bash
# Queue a dependent single-GPU experiment without competing with a live run.
# It exits rather than taking a GPU that became occupied by somebody else.
set -euo pipefail

if [[ $# -lt 4 || "$1" != "--wait-pid" || "$3" != "--" ]]; then
  echo "usage: $0 --wait-pid <pid> -- <command> [args...]" >&2
  exit 2
fi

WAIT_PID="$2"
shift 3

while kill -0 "$WAIT_PID" 2>/dev/null; do
  sleep 30
done

# The caller's CUDA_VISIBLE_DEVICES selects one physical GPU.  Any remaining
# allocation here means a new job acquired it after the dependency ended.
USED_MIB="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tail -n 1 | tr -d ' ')"
if [[ "$USED_MIB" != "0" ]]; then
  echo "dependent job finished but target GPU is occupied (${USED_MIB} MiB); not starting" >&2
  exit 3
fi

exec "$@"
