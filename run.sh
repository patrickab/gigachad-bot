#!/bin/bash

export MORPHIC_URL="${MORPHIC_URL:-http://localhost:3001}"

export PATH="$(pwd)/.venv/bin:$PATH"

echo "Starting morphic container stack..."
docker compose -f docker-compose.morphic.yml up -d

echo "Waiting for morphic to be ready..."
until curl -sf http://localhost:3001 > /dev/null 2>&1; do
  sleep 2
done
echo "Morphic is ready."

echo "Starting backend on http://127.0.0.1:8001"
source .venv/bin/activate
uvicorn src.backend.server:app --host 127.0.0.1 --port 8001 &
BACKEND_PID=$!

echo "Starting frontend on http://127.0.0.1:2999"
npm --prefix src/frontend run dev &
FRONTEND_PID=$!

cleanup() {
  echo "Shutting down..."
  for PID in $BACKEND_PID $FRONTEND_PID; do
    kill -TERM -- "-$PID" 2>/dev/null || kill -TERM "$PID" 2>/dev/null || true
  done
  docker compose -f docker-compose.morphic.yml down
  wait 2>/dev/null || true
  echo "Done."
}

trap cleanup INT TERM

wait
