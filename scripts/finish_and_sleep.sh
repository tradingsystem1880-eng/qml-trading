#!/bin/bash
LOG="/Users/hunternovotny/Desktop/qml-trading-main/results/monitor.log"
OPT_LOG="/Users/hunternovotny/Desktop/qml-trading-main/results/phase77_optimization.log"

while true; do
    COMPLETE_COUNT=$(grep -c "COMPLETE" "$OPT_LOG" 2>/dev/null || echo "0")
    
    if [ "$COMPLETE_COUNT" -ge 6 ]; then
        echo "=== $(date) - ALL 6 OBJECTIVES COMPLETE ===" >> "$LOG"
        
        # Log final results
        echo "Final Results:" >> "$LOG"
        grep -E "COMPLETE|Objective:|Best parameters|Best score" "$OPT_LOG" | tail -20 >> "$LOG"
        
        # Verify process finished
        if ! pgrep -f "run_phase77" > /dev/null; then
            echo "Optimization process finished cleanly." >> "$LOG"
        else
            echo "Warning: Process still running, waiting 5 min..." >> "$LOG"
            sleep 300
        fi
        
        # Kill caffeinate
        pkill caffeinate
        echo "Caffeinate stopped - PC can sleep now." >> "$LOG"
        echo "=== OPTIMIZATION COMPLETE - GOOD NIGHT ===" >> "$LOG"
        exit 0
    fi
    
    # Log progress every check
    echo "=== $(date) - Waiting for completion ($COMPLETE_COUNT/6) ===" >> "$LOG"
    tail -3 "$OPT_LOG" 2>/dev/null | grep -E "Iter|Best" | head -1 >> "$LOG"
    
    # Check every 10 minutes
    sleep 600
done
