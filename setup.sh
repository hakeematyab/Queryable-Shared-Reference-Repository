#!/bin/bash
set -euo pipefail

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

echo "============================================================"
echo "  QSRR Setup"
echo "============================================================"
echo

# Check and install uv
if ! command -v uv &>/dev/null; then
    read -p "uv not found. Install it? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    else
        echo "uv is required. Exiting."
        exit 1
    fi
fi
echo "✓ uv available"

# Check and install npm (source nvm first if it exists)
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"

if ! command -v npm &>/dev/null; then
    read -p "npm not found. Install Node.js via nvm? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
        [ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
        nvm install --lts
    else
        echo "npm is required. Exiting."
        exit 1
    fi
fi
echo "✓ npm available"

# Check and install ollama
if ! command -v ollama &>/dev/null; then
    read -p "Ollama not found. Install it? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        curl -fsSL https://ollama.ai/install.sh | sh
    else
        echo "Ollama is required. Exiting."
        exit 1
    fi
fi
echo "✓ ollama available"

echo

# Backend dependencies
cd "$SCRIPT_DIR/app/backend"
if [ ! -d ".venv" ]; then
    echo "Setting up backend dependencies..."
    uv sync
else
    echo "✓ Backend dependencies already installed"
fi

# HuggingFace login (needed for gated models)
if [ ! -f "$HOME/.cache/huggingface/token" ]; then
    echo
    echo "HuggingFace login required for accessing gated models."
    uv run huggingface-cli login
else
    echo "✓ HuggingFace already logged in"
fi

# Frontend dependencies
cd "$SCRIPT_DIR/app/frontend"
if [ ! -d "node_modules" ]; then
    echo "Setting up frontend dependencies..."
    npm install
else
    echo "✓ Frontend dependencies already installed"
fi

echo
echo "============================================================"
echo "  Setup complete!"
echo "============================================================"
echo "  Run ./startup.sh to start all servers"
echo "============================================================"
