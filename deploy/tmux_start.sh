#!/usr/bin/env bash
# Start (or restart) the Discord broadcaster inside a tmux session.
#
# Usage (on the Linux server, after .env and dependencies exist):
#   cd ~/SafetyAwarenessPulse
#   chmod +x deploy/tmux_start.sh
#   ./deploy/tmux_start.sh
#
# If python3 -m venv fails (no sudo / no python3-venv), create .venv first with:
#   python3 -m pip install --user virtualenv
#   python3 -m virtualenv .venv
#   source .venv/bin/activate && pip install -r requirements.txt

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SESSION="safety-bot"
cd "$ROOT"

export PATH="$HOME/.local/bin:$PATH"

if [[ ! -f .env ]]; then
  echo "ERROR: .env missing. Copy .env.example to .env and fill AZURE_OPENAI_*, SERPER_*, DISCORD_*."
  exit 1
fi

create_venv() {
  echo "[setup] Creating .venv ..."
  if python3 -m venv .venv 2>/dev/null; then
    return 0
  fi
  echo "[setup] python3 -m venv failed; trying virtualenv (no sudo) ..."
  python3 -m pip install --user virtualenv
  python3 -m virtualenv .venv
}

if [[ ! -x .venv/bin/python ]]; then
  rm -rf .venv
  create_venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  pip install -U pip
  pip install -r requirements.txt
else
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

if [[ ! -f data/events.db ]]; then
  echo "[setup] data/events.db missing — running document_ingest.py --reset ..."
  if [[ ! -d assets/ppts ]] || [[ -z "$(ls -A assets/ppts/*.pptx 2>/dev/null || true)" ]]; then
    echo "ERROR: Put PPT files in assets/ppts/ then re-run, or copy data/events.db from your PC."
    exit 1
  fi
  python document_ingest.py --reset
fi

mkdir -p logs

if ! command -v tmux >/dev/null 2>&1; then
  echo "ERROR: tmux not found. Ask admin to install tmux, or run: python discord_broadcast.py in a nohup session."
  exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "[tmux] Session '$SESSION' already exists. Stopping old bot ..."
  tmux send-keys -t "$SESSION" C-c || true
  sleep 1
  tmux kill-session -t "$SESSION" 2>/dev/null || true
fi

tmux new-session -d -s "$SESSION" -c "$ROOT" \
  "source .venv/bin/activate && python discord_broadcast.py 2>&1 | tee -a logs/discord_bot.log"

echo "[tmux] Started session '$SESSION'."
echo "  Attach:  tmux attach -t $SESSION"
echo "  Detach:  Ctrl+B then D"
echo "  Stop:    tmux kill-session -t $SESSION"
echo "  Logs:    tail -f $ROOT/logs/discord_bot.log"
