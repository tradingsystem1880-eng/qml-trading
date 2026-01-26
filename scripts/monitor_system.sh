#!/bin/bash
# System Monitor for Phase 7.7 Optimization
# Checks CPU, memory, and optimization progress

LOG_FILE="results/system_monitor.log"
OPT_LOG="results/phase77_optimization.log"

echo "=== System Monitor Started $(date) ===" >> $LOG_FILE

while true; do
    echo "" >> $LOG_FILE
    echo "--- $(date) ---" >> $LOG_FILE

    # CPU and Memory
    top -l 1 | head -10 | tail -6 >> $LOG_FILE

    # Check if optimization is running
    OPT_PID=$(pgrep -f "run_phase77_optimization" | head -1)
    if [ -n "$OPT_PID" ]; then
        echo "Optimization running (PID: $OPT_PID)" >> $LOG_FILE

        # Get current iteration from log
        LAST_ITER=$(grep -o "\[Iter [0-9]*/[0-9]*\]" $OPT_LOG 2>/dev/null | tail -1)
        BEST_SCORE=$(grep "Best:" $OPT_LOG 2>/dev/null | tail -1 | grep -o "Best: [0-9.-]*" | tail -1)
        echo "Progress: $LAST_ITER | $BEST_SCORE" >> $LOG_FILE
    else
        echo "WARNING: Optimization not running!" >> $LOG_FILE
    fi

    # Check disk space
    DISK=$(df -h /Users/hunternovotny/Desktop | tail -1 | awk '{print $5}')
    echo "Disk usage: $DISK" >> $LOG_FILE

    # Sleep for 5 minutes
    sleep 300
done
