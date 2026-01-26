#!/bin/bash
LOG="/Users/hunternovotny/Desktop/qml-trading-main/results/monitor.log"
echo "=== $(date) - HEALTH CHECK ===" >> "$LOG"

# Check if process is running
if ps aux | grep -v grep | grep "run_phase77" > /dev/null; then
    PID=$(pgrep -f "run_phase77")
    CPU=$(ps -p $PID -o %cpu= 2>/dev/null || echo "N/A")
    MEM=$(ps -p $PID -o %mem= 2>/dev/null || echo "N/A")
    echo "Status: RUNNING | PID $PID | CPU: $CPU% | MEM: $MEM%" >> "$LOG"
    
    # Get latest progress
    tail -5 /Users/hunternovotny/Desktop/qml-trading-main/results/phase77_optimization.log 2>/dev/null | grep -E "Iter|Best" | tail -2 >> "$LOG"
else
    echo "Status: STOPPED or COMPLETED" >> "$LOG"
fi

# Check thermal (macOS)
if command -v powermetrics &> /dev/null; then
    sudo powermetrics --samplers smc -i1 -n1 2>/dev/null | grep -i "CPU die" | head -1 >> "$LOG" || echo "Thermal: Unable to read (normal)" >> "$LOG"
else
    echo "Thermal: powermetrics not available" >> "$LOG"
fi

echo "" >> "$LOG"
