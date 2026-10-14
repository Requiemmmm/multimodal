#!/usr/bin/env bash
set -euo pipefail

# Launch a long MORPH command in a detached tmux session with durable logs.
# Usage: scripts/run_tmux.sh <session> <run-dir> -- <command> [args...]
if [[ $# -lt 4 || "$3" != "--" ]]; then
  echo "usage: $0 <session> <run-dir> -- <command> [args...]" >&2
  exit 2
fi
SESSION="$1"
RUN_DIR="$2"
shift 3
mkdir -p "$RUN_DIR"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi
printf '%q ' "$@" > "$RUN_DIR/command.sh"
printf '\n' >> "$RUN_DIR/command.sh"
cat > "$RUN_DIR/tmux_runner.sh" <<'EOF'
#!/usr/bin/env bash
set -o pipefail
date -Is > "$RUN_DIR/start_time.txt"
echo "pid=$$" > "$RUN_DIR/process.txt"
"$@" 2>&1 | tee "$RUN_DIR/stdout.log"
status=${PIPESTATUS[0]}
echo "$status" > "$RUN_DIR/exit_code.txt"
date -Is > "$RUN_DIR/end_time.txt"
exit "$status"
EOF
sed -i "s|\$RUN_DIR|$RUN_DIR|g" "$RUN_DIR/tmux_runner.sh"
chmod +x "$RUN_DIR/tmux_runner.sh" "$RUN_DIR/command.sh"
tmux new-session -d -s "$SESSION" -- "$RUN_DIR/tmux_runner.sh" "$@"
echo "$SESSION" > "$RUN_DIR/tmux_session.txt"
echo "started tmux session $SESSION; attach with: tmux attach -t $SESSION"
