#!/bin/bash
set -euo pipefail

MODE="dev"
WORKERS=4
OLLAMA_MODEL="qwen3:8b"
OLLAMA_HOST="127.0.0.1:11434"
OLLAMA_NUM_PARALLEL=4  # Max concurrent requests per model

# Source nvm to ensure npm is available
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"

while [[ $# -gt 0 ]]; do
    case $1 in
        --prod|-p) MODE="prod"; shift ;;
        --workers|-w) WORKERS="$2"; shift 2 ;;
        *) shift ;;
    esac
done

get_script_dir() {
    local source="${BASH_SOURCE[0]}"
    while [ -h "$source" ]; do
        local dir="$(cd -P "$(dirname "$source")" && pwd)"
        source="$(readlink "$source")"
        [[ $source != /* ]] && source="$dir/$source"
    done
    echo "$(cd -P "$(dirname "$source")" && pwd)"
}

SCRIPT_DIR="$(get_script_dir)"

# Auto-launch in tmux if not already in one
if [ -z "${QSRR_IN_TMUX:-}" ]; then
    echo "Starting servers in tmux session 'qsrr'..."
    tmux kill-session -t qsrr 2>/dev/null || true
    exec tmux new-session -s qsrr "QSRR_IN_TMUX=1 '$SCRIPT_DIR/startup.sh' $*; exec bash"
fi

# Run setup first
echo "Running setup..."
"$SCRIPT_DIR/setup.sh"

echo
echo "============================================================"
echo "  Starting servers (Mode: $MODE)"
echo "============================================================"
echo

# Start Ollama (restart if already running to apply OLLAMA_NUM_PARALLEL)
if pgrep -x "ollama" > /dev/null; then
    echo "Ollama already running - restarting to apply parallel=$OLLAMA_NUM_PARALLEL..."
    pkill -x "ollama"
    sleep 2
fi

echo "Starting Ollama server on $OLLAMA_HOST (parallel=$OLLAMA_NUM_PARALLEL)..."
OLLAMA_HOST="$OLLAMA_HOST" OLLAMA_NUM_PARALLEL="$OLLAMA_NUM_PARALLEL" ollama serve &
OLLAMA_PID=$!
sleep 3

# Pull model
echo "Ensuring Ollama model $OLLAMA_MODEL is available..."
ollama pull "$OLLAMA_MODEL"

# Cleanup handler
cleanup() {
    echo
    echo "Shutting down servers..."
    [ -n "${OLLAMA_PID:-}" ] && kill "$OLLAMA_PID" 2>/dev/null
    [ -n "${BACKEND_PID:-}" ] && kill "$BACKEND_PID" 2>/dev/null
    [ -n "${FRONTEND_PID:-}" ] && kill "$FRONTEND_PID" 2>/dev/null
    exit 0
}
trap cleanup SIGINT SIGTERM

if [ "$MODE" = "prod" ]; then
    echo "Starting PRODUCTION servers..."
    echo "  - Ollama: http://$OLLAMA_HOST"
    echo "  - Backend: 0.0.0.0:8000 with $WORKERS workers"
    echo "  - Frontend: 0.0.0.0:3000 (static build)"
    echo

    cd "$SCRIPT_DIR/frontend"
    if [ ! -d "build" ]; then
        echo "Building frontend..."
        npm run build
    else
        echo "Frontend build exists. Delete /build folder to rebuild."
    fi

    cd "$SCRIPT_DIR/backend"
    uv run uvicorn app:app --host 0.0.0.0 --port 8000 --workers "$WORKERS" &
    BACKEND_PID=$!

    cd "$SCRIPT_DIR/frontend"
    npx serve -s build -l 3000 &
    FRONTEND_PID=$!
else
    echo "Starting DEVELOPMENT servers..."
    echo "  - Ollama: http://$OLLAMA_HOST"
    echo "  - Backend: http://localhost:8000 (hot reload)"
    echo "  - Frontend: http://localhost:3000 (hot reload)"
    echo

    cd "$SCRIPT_DIR/backend"
    uv run uvicorn app:app --reload --host 0.0.0.0 --port 8000 &
    BACKEND_PID=$!

    cd "$SCRIPT_DIR/frontend"
    CI=true npm start &
    FRONTEND_PID=$!
fi

echo
echo "Servers running!"
[ -n "${OLLAMA_PID:-}" ] && echo "  - Ollama: http://$OLLAMA_HOST (PID: $OLLAMA_PID)"
echo "  - Backend: http://localhost:8000 (PID: $BACKEND_PID)"
echo "  - Frontend: http://localhost:3000 (PID: $FRONTEND_PID)"
echo
echo "Press Ctrl+C to stop all servers."
wait