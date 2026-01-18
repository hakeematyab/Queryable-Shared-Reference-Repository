#!/bin/bash
# Usage:
#   ./profile.sh                      # Start profiler (launches in tmux)
#   tmux attach -t qsrr-profiler      # Attach to running profiler
#   tmux kill-session -t qsrr-profiler # Kill profiler

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/data/profiles"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/profile_$TIMESTAMP.log"

# Auto-launch in tmux if not already in one
if [ -z "${QSRR_PROFILER:-}" ]; then
    echo "Starting profiler in tmux session 'qsrr-profiler'..."
    tmux kill-session -t qsrr-profiler 2>/dev/null || true
    exec tmux new-session -s qsrr-profiler "QSRR_PROFILER=1 '$0'; exec bash"
fi

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║               QSRR Resource Profiler                         ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Logging to: $LOG_FILE"
echo "║  Press Ctrl+C to stop                                        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Write header to log file
{
    echo "# QSRR Resource Profile"
    echo "# Started: $(date)"
    echo "# Format: timestamp | qsrr_status | ollama_mem_mb | python_mem_mb | node_mem_mb | total_app_mb | system_pressure"
    echo "#"
} > "$LOG_FILE"

get_process_mem() {
    # Returns memory in MB for processes matching pattern
    ps aux 2>/dev/null | grep -E "$1" | grep -v grep | awk '{sum += $6} END {printf "%.1f", sum/1024}'
}

get_ollama_model_mem() {
    # Get Ollama's reported model memory
    ollama ps 2>/dev/null | tail -n +2 | awk '{print $4}' | head -1 || echo "0"
}

get_memory_free() {
    # macOS memory free percentage
    memory_pressure 2>/dev/null | grep "System-wide" | awk '{print $NF}' || echo "unknown"
}

get_total_memory_gb() {
    # Total physical memory in GB
    sysctl -n hw.memsize 2>/dev/null | awk '{printf "%.0f", $1 / 1024 / 1024 / 1024}' || echo "?"
}

check_qsrr_status() {
    if tmux has-session -t qsrr 2>/dev/null; then
        echo "RUNNING"
    else
        echo "STOPPED"
    fi
}

while true; do
    TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")
    
    QSRR_STATUS=$(check_qsrr_status)
    OLLAMA_MEM=$(get_process_mem "ollama")
    PYTHON_MEM=$(get_process_mem "uvicorn|python.*app")
    NODE_MEM=$(get_process_mem "node|npm")
    TOTAL_APP=$(echo "$OLLAMA_MEM + $PYTHON_MEM + $NODE_MEM" | bc 2>/dev/null || echo "0")
    MEM_FREE=$(get_memory_free)
    
    OLLAMA_MODEL_MEM=$(get_ollama_model_mem)
    TOTAL_MEM_GB=$(get_total_memory_gb)
    USAGE_PCT=$(echo "scale=1; $TOTAL_APP / ($TOTAL_MEM_GB * 1024) * 100" | bc 2>/dev/null || echo "?")
    
    LOG_LINE="$TIMESTAMP | $QSRR_STATUS | $OLLAMA_MEM | $PYTHON_MEM | $NODE_MEM | $TOTAL_APP | $MEM_FREE"
    
    echo "$LOG_LINE" >> "$LOG_FILE"
    
    clear
    echo "═══════════════════════════════════════════════════════════════"
    echo " QSRR Resource Profiler - Live View"
    echo " Log: $LOG_FILE"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
    echo " Timestamp:        $TIMESTAMP"
    echo " QSRR Service:     $QSRR_STATUS"
    echo " System Memory:    ${TOTAL_MEM_GB} GB total"
    echo ""
    echo " ┌─────────────────────────────────────────────────────────────┐"
    echo " │ Process Memory Usage (MB)                                  │"
    echo " ├─────────────────────────────────────────────────────────────┤"
    printf " │ Ollama:          %8s MB                               │\n" "$OLLAMA_MEM"
    printf " │ Python/Backend:  %8s MB                               │\n" "$PYTHON_MEM"
    printf " │ Node/Frontend:   %8s MB                               │\n" "$NODE_MEM"
    echo " ├─────────────────────────────────────────────────────────────┤"
    printf " │ TOTAL APP:       %8s MB  (%s%% of RAM)                │\n" "$TOTAL_APP" "$USAGE_PCT"
    echo " └─────────────────────────────────────────────────────────────┘"
    echo ""
    echo " Ollama Model Mem: $OLLAMA_MODEL_MEM"
    echo " Memory Free:      $MEM_FREE"
    echo ""
    echo " [Logging every 2 seconds - Press Ctrl+C to stop]"
    
    sleep 2
done
