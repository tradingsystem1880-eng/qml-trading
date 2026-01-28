#!/usr/bin/env python3
"""
Paper Trading Status Display
=============================
Shows current paper trading status, metrics, and recent trades.

Usage:
    python scripts/paper_trade_status.py           # One-time display
    python scripts/paper_trade_status.py --watch   # Auto-refresh every 30s
"""

import json
import argparse
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SUMMARY_PATH = PROJECT_ROOT / 'logs' / 'paper_trading' / 'summary.json'
TRADES_DIR = PROJECT_ROOT / 'logs' / 'paper_trading' / 'trades'


def load_summary() -> dict:
    """Load summary.json."""
    if not SUMMARY_PATH.exists():
        return None
    with open(SUMMARY_PATH) as f:
        return json.load(f)


def load_recent_trades(n: int = 5) -> list:
    """Load n most recent trades."""
    if not TRADES_DIR.exists():
        return []

    trade_files = sorted(TRADES_DIR.glob('*.json'), reverse=True)[:n]
    trades = []
    for f in trade_files:
        with open(f) as fp:
            trades.append(json.load(fp))
    return trades


def format_metric(value, fmt: str = '.2f', suffix: str = '') -> str:
    """Format a metric value, handling None."""
    if value is None:
        return '---'
    return f"{value:{fmt}}{suffix}"


def display_status():
    """Display the status dashboard."""
    summary = load_summary()

    if summary is None:
        print("❌ No summary.json found. Run launch_paper_trading.py first.")
        return

    # Extract values
    phase = summary.get('phase', 1)
    target = summary.get('target_trades', 50)
    status = summary.get('status', 'UNKNOWN')
    totals = summary.get('totals', {})
    metrics = summary.get('metrics', {})
    thresholds = summary.get('thresholds', {})
    open_positions = summary.get('open_positions', [])
    next_scan = summary.get('next_scan')

    trades = totals.get('trades', 0)
    wins = totals.get('wins', 0)
    losses = totals.get('losses', 0)
    total_r = totals.get('total_r', 0)

    win_rate = metrics.get('win_rate')
    pf = metrics.get('profit_factor')

    # Status color
    if status == 'ACTIVE':
        status_display = '🟢 ACTIVE'
    elif status == 'PAUSED':
        status_display = '🟡 PAUSED'
    elif status == 'SHUTDOWN':
        status_display = '🔴 SHUTDOWN'
    else:
        status_display = '⚪ NOT STARTED'

    # Calculate time to next scan
    if next_scan:
        try:
            next_dt = datetime.fromisoformat(next_scan)
            delta = next_dt - datetime.now()
            if delta.total_seconds() > 0:
                hours, remainder = divmod(int(delta.total_seconds()), 3600)
                minutes, seconds = divmod(remainder, 60)
                next_scan_str = f"{hours}h {minutes}m"
            else:
                next_scan_str = "scanning..."
        except:
            next_scan_str = "---"
    else:
        next_scan_str = "---"

    # Progress bar
    progress_pct = min(trades / target * 100, 100) if target > 0 else 0
    progress_bar = '█' * int(progress_pct / 5) + '░' * (20 - int(progress_pct / 5))

    # Threshold status
    progress_pf = thresholds.get('progress_pf', 1.5)
    pause_pf = thresholds.get('pause_pf', 1.0)
    shutdown_pf = thresholds.get('shutdown_pf', 0.7)

    # Dashboard
    print("\033[2J\033[H")  # Clear screen
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║  QML PAPER TRADING - Phase {phase} ({target} trades @ 0.5% risk)         ║
║  Status: {status_display:<20}                                   ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  Progress: [{progress_bar}] {progress_pct:5.1f}%                  ║
║  Trades:   {trades:3d}/{target}         Wins: {wins:3d}    Losses: {losses:3d}             ║
║                                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║  METRICS                                                         ║
║  ├─ Win Rate:      {format_metric(win_rate, '.1%'):>8}     (target: >45%)             ║
║  ├─ Profit Factor: {format_metric(pf):>8}     (target: >1.5)              ║
║  └─ Total R:       {format_metric(total_r, '+.2f', 'R'):>8}                               ║
╠══════════════════════════════════════════════════════════════════╣
║  THRESHOLDS                                                      ║
║  ├─ Progress: PF > {progress_pf:.1f}, WR > 45%  →  Advance to Phase {phase+1}       ║
║  ├─ Pause:    PF < {pause_pf:.1f}              →  Investigate              ║
║  └─ Shutdown: PF < {shutdown_pf:.1f}              →  Stop trading             ║
╠══════════════════════════════════════════════════════════════════╣
║  Open Positions: {len(open_positions)}/3          Next scan: {next_scan_str:<10}        ║
╚══════════════════════════════════════════════════════════════════╝
""")

    # Open positions
    if open_positions:
        print("OPEN POSITIONS:")
        print("-" * 60)
        for pos in open_positions:
            symbol = pos.get('symbol', '???')
            direction = pos.get('direction', '?')
            entry = pos.get('entry_price', 0)
            current_r = pos.get('current_r', 0)
            print(f"  {symbol} {direction}  Entry: {entry:,.2f}  Current: {current_r:+.2f}R")
        print()

    # Recent trades
    recent = load_recent_trades(5)
    print("RECENT TRADES:")
    print("-" * 60)
    if not recent:
        print("  (no trades yet)")
    else:
        for trade in recent:
            symbol = trade.get('symbol', '???')
            direction = trade.get('direction', '?')
            outcome = trade.get('outcome', '?')
            r_multiple = trade.get('r_multiple', 0)
            ts = trade.get('exit_time', trade.get('entry_time', '???'))

            if outcome == 'WIN':
                emoji = '✅'
            elif outcome == 'LOSS':
                emoji = '❌'
            else:
                emoji = '⏳'

            print(f"  {emoji} {symbol} {direction:5} {r_multiple:+.2f}R  {ts[:16]}")

    print()
    print(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def watch_mode(interval: int = 30):
    """Auto-refresh status display."""
    print(f"Watching status (refresh every {interval}s). Press Ctrl+C to stop.")
    try:
        while True:
            display_status()
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopped watching.")


def main():
    parser = argparse.ArgumentParser(description='Paper Trading Status')
    parser.add_argument('--watch', action='store_true', help='Auto-refresh mode')
    parser.add_argument('--interval', type=int, default=30, help='Refresh interval (seconds)')
    args = parser.parse_args()

    if args.watch:
        watch_mode(args.interval)
    else:
        display_status()


if __name__ == '__main__':
    main()
