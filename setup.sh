#!/bin/bash
set -euo pipefail

MODE="dev"
WORKERS=4
OLLAMA_MODEL="qwen3:8b"
OLLAMA_HOST="127.0.0.1:11434"

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

if ! command -v uv &>/dev/null; then
    read -p "uv not found. Install it? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    else
        exit 1
    fi
fi

if ! command -v npm &>/dev/null; then
    read -p "npm not found. Install Node.js via nvm? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
        export NVM_DIR="$HOME/.nvm"
        [ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
        nvm install --lts
    else
        exit 1
    fi
fi

if ! command -v ollama &>/dev/null; then
    read -p "Ollama not found. Install it? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        curl -fsSL https://ollama.ai/install.sh | sh
    else
        exit 1
    fi
fi

cd "$SCRIPT_DIR/app/backend"
if [ ! -d ".venv" ]; then
    echo "Setting up backend dependencies..."
    uv sync
else
    echo "Backend dependencies already installed."
fi

cd "$SCRIPT_DIR/app/frontend"
if [ ! -d "node_modules" ]; then
    echo "Setting up frontend dependencies..."
    npm install
else
    echo "Frontend dependencies already installed."
fi

echo
echo "============================================================"
echo "  Mode: $MODE"
echo "============================================================"
echo

if ! pgrep -x "ollama" > /dev/null; then
    echo "Starting Ollama server on $OLLAMA_HOST..."
    OLLAMA_HOST="$OLLAMA_HOST" ollama serve &
    OLLAMA_PID=$!
    sleep 3
else
    echo "Ollama server already running."
    OLLAMA_PID=""
fi

echo "Ensuring Ollama model $OLLAMA_MODEL is available..."
ollama pull "$OLLAMA_MODEL"

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

    cd "$SCRIPT_DIR/app/frontend"
    if [ ! -d "build" ]; then
        echo "Building frontend..."
        npm run build
    else
        echo "Frontend build exists. Delete /build folder to rebuild."
    fi

    cd "$SCRIPT_DIR/app/backend"
    uv run uvicorn app:app --host 0.0.0.0 --port 8000 --workers "$WORKERS" &
    BACKEND_PID=$!

    cd "$SCRIPT_DIR/app/frontend"
    npx serve -s build -l 3000 &
    FRONTEND_PID=$!
else
    echo "Starting DEVELOPMENT servers..."
    echo "  - Ollama: http://$OLLAMA_HOST"
    echo "  - Backend: http://localhost:8000 (hot reload)"
    echo "  - Frontend: http://localhost:3000 (hot reload)"
    echo

    cd "$SCRIPT_DIR/app/backend"
    uv run uvicorn app:app --reload --host 0.0.0.0 --port 8000 &
    BACKEND_PID=$!

    cd "$SCRIPT_DIR/app/frontend"
    npm start &
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
