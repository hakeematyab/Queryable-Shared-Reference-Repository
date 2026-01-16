#!/bin/bash
# Stop all QSRR servers

echo "Stopping QSRR servers..."

# Kill the tmux session (this kills all processes inside it)
if tmux has-session -t qsrr 2>/dev/null; then
    tmux kill-session -t qsrr
    echo "✓ Stopped tmux session 'qsrr'"
else
    echo "No 'qsrr' tmux session found"
    
    # Fallback: kill processes directly if not using tmux
    pkill -f "uvicorn app:app" 2>/dev/null && echo "✓ Stopped backend"
    pkill -f "react-scripts" 2>/dev/null && echo "✓ Stopped frontend"
fi

echo "Done."
