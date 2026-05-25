#!/bin/sh
set -e

# Use Node.js from the virtual environment
export PATH="$(pwd)/.venv/bin:$PATH"

# Start FastAPI backend
echo "Starting backend on http://127.0.0.1:8001"
source .venv/bin/activate
uvicorn src.backend.server:app --host 127.0.0.1 --port 8001 &
BACKEND_PID=$!

# Start Next.js frontend dev server
echo "Starting frontend on http://127.0.0.1:2999"
cd src/frontend && npm run dev &
FRONTEND_PID=$!

cleanup() {
  echo "Shutting down..."
  kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
  wait
}

trap cleanup INT TERM

wait $BACKEND_PID $FRONTEND_PID
