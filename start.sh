#!/usr/bin/env sh
# Starts the Research Agent and opens it in your browser (macOS / Linux).
set -e
cd "$(dirname "$0")"
PORT="${PORT:-8000}"

if [ ! -x .venv/bin/python ]; then
  echo "First run: setting up Python packages. This takes a few minutes..."
  python3 -m venv .venv
  .venv/bin/python -m pip install --quiet --upgrade pip
  .venv/bin/python -m pip install --quiet -r requirements.txt
fi

if [ ! -f .env ]; then
  cp .env.example .env
  echo "Add your OpenAI API key to .env, then run ./start.sh again."
  exit 0
fi

URL="http://localhost:$PORT"
echo "Starting the Research Agent at $URL (Ctrl+C to stop)"
echo "Your browser opens once it's ready, usually in 10-30 seconds."
.venv/bin/python scripts/open_when_ready.py "$URL" &
exec .venv/bin/python -m uvicorn research_agent.api.main:app --port "$PORT"
