#!/bin/bash
# Stop all QSRR servers

echo "Stopping QSRR servers..."

# Kill the screen session (this kills all processes inside it)
if screen -list 2>/dev/null | grep -q "qsrr"; then
    screen -S qsrr -X quit
    echo "✓ Stopped screen session 'qsrr'"
else
    echo "No 'qsrr' screen session found"
    
    # Fallback: kill processes directly if not using screen
    pkill -f "uvicorn app:app" 2>/dev/null && echo "✓ Stopped backend"
    pkill -f "react-scripts" 2>/dev/null && echo "✓ Stopped frontend"
fi

echo "Done."
