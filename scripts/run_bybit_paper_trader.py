#!/usr/bin/env python3
"""
Phase 9.5: Bybit Paper Trader CLI
=================================
Command-line interface for running forward tests on Bybit testnet.

Features:
- Single scan mode: Detect and optionally execute signals
- Watch mode: Continuous monitoring with configurable interval
- Position check: Monitor open positions for SL/TP
- Status display: Show current forward test metrics

Setup:
1. Create Bybit testnet account at https://testnet.bybit.com
2. Generate API key/secret
3. Set environment variables or pass as arguments

Usage:
    # Single scan (detect signals only)
    python scripts/run_bybit_paper_trader.py scan

    # Scan and execute signals
    python scripts/run_bybit_paper_trader.py scan --execute

    # Watch mode (continuous)
    python scripts/run_bybit_paper_trader.py watch --interval 300

    # Check positions
    python scripts/run_bybit_paper_trader.py check

    # Show status
    python scripts/run_bybit_paper_trader.py status

Environment Variables:
    BYBIT_TESTNET_API_KEY: API key
    BYBIT_TESTNET_API_SECRET: API secret
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.execution.paper_trader_bybit import BybitPaperTrader
from src.execution.models import ForwardTestPhase

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


DEFAULT_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
STATE_FILE = PROJECT_ROOT / "results" / "forward_test" / "state.json"


def get_api_credentials(args) -> tuple:
    """Get API credentials from args or environment."""
    api_key = args.api_key or os.environ.get("BYBIT_TESTNET_API_KEY", "")
    api_secret = args.api_secret or os.environ.get("BYBIT_TESTNET_API_SECRET", "")
    return api_key, api_secret


def create_trader(args) -> BybitPaperTrader:
    """Create paper trader instance."""
    api_key, api_secret = get_api_credentials(args)

    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(',')]
        symbols = [s if '/' in s else f"{s[:-4]}/{s[-4:]}" for s in symbols]
    else:
        symbols = DEFAULT_SYMBOLS

    phase_map = {
        1: ForwardTestPhase.PHASE1_PAPER,
        2: ForwardTestPhase.PHASE2_MICRO,
        3: ForwardTestPhase.PHASE3_FULL,
    }
    phase = phase_map.get(args.phase, ForwardTestPhase.PHASE1_PAPER)

    return BybitPaperTrader(
        api_key=api_key,
        api_secret=api_secret,
        symbols=symbols,
        timeframe=args.timeframe,
        phase=phase,
        state_file=str(STATE_FILE),
    )


def cmd_scan(args):
    """Run single scan for signals."""
    print("=" * 60)
    print("BYBIT PAPER TRADER - SIGNAL SCAN")
    print("=" * 60)
    print(f"\nTimeframe: {args.timeframe}")
    print(f"Execute: {'Yes' if args.execute else 'No (dry run)'}")

    trader = create_trader(args)

    print(f"\nScanning {len(trader.symbols)} symbols...")
    signals = trader.run_scan()

    if not signals:
        print("\nNo signals found.")
        return

    print(f"\n{'=' * 60}")
    print(f"FOUND {len(signals)} SIGNALS")
    print("=" * 60)

    for sig in signals:
        print(f"\n{sig.symbol} - {sig.direction}")
        print(f"  Entry:  {sig.entry_price:.4f}")
        print(f"  SL:     {sig.stop_loss:.4f}")
        print(f"  TP:     {sig.take_profit:.4f}")
        print(f"  R:R:    {sig.risk_reward:.2f}")
        print(f"  Score:  {sig.score:.2f}")
        print(f"  Tier:   {sig.tier}")

        if args.execute:
            if trader.can_open_position(sig):
                position = trader.open_position(sig)
                if position:
                    print(f"  ✅ POSITION OPENED: {position.quantity} @ {position.entry_price}")
                else:
                    print(f"  ⚠️  Failed to open position")
            else:
                print(f"  ⏸️  Cannot open (check limits)")

    trader.print_status()


def cmd_watch(args):
    """Run continuous watch mode."""
    print("=" * 60)
    print("BYBIT PAPER TRADER - WATCH MODE")
    print("=" * 60)
    print(f"\nInterval: {args.interval} seconds")
    print(f"Press Ctrl+C to stop\n")

    trader = create_trader(args)

    try:
        while True:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Scanning...")

            # Check existing positions
            closed = trader.check_positions()
            for trade in closed:
                print(f"  CLOSED: {trade.symbol} {trade.direction} "
                      f"PnL={trade.pnl_r:+.2f}R ({trade.exit_reason})")

            # Scan for new signals
            signals = trader.run_scan()
            if signals:
                print(f"  Found {len(signals)} new signals")
                for sig in signals:
                    print(f"    {sig.symbol} {sig.direction} @ {sig.entry_price:.4f}")

                    if args.execute and trader.can_open_position(sig):
                        position = trader.open_position(sig)
                        if position:
                            print(f"      ✅ OPENED")

            # Print brief status
            status = trader.get_status()
            print(f"  Status: {status['trade_count']} trades, "
                  f"PF={status['profit_factor']:.2f}, "
                  f"WR={status['win_rate']:.1%}, "
                  f"{status['open_positions']} open")

            if status['is_shutdown']:
                print(f"\n⛔ SHUTDOWN: {status['shutdown_reason']}")
                break

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n\nStopping watch mode...")
        trader.print_status()


def cmd_check(args):
    """Check open positions."""
    print("=" * 60)
    print("BYBIT PAPER TRADER - POSITION CHECK")
    print("=" * 60)

    trader = create_trader(args)

    print(f"\nChecking {len(trader.state.open_positions)} open positions...")

    closed = trader.check_positions()

    if closed:
        print(f"\n{len(closed)} positions closed:")
        for trade in closed:
            print(f"  {trade.symbol} {trade.direction}: "
                  f"PnL={trade.pnl_r:+.2f}R ({trade.exit_reason})")
    else:
        print("\nNo positions hit SL/TP.")

    if trader.state.open_positions:
        print(f"\nRemaining open positions:")
        for pos in trader.state.open_positions:
            print(f"  {pos.symbol} {pos.side.value.upper()}: "
                  f"Entry={pos.entry_price:.4f}, "
                  f"SL={pos.stop_loss_price:.4f}, "
                  f"TP={pos.take_profit_price:.4f}")


def cmd_status(args):
    """Show current status."""
    trader = create_trader(args)
    trader.print_status()

    if trader.state.completed_trades:
        print(f"\nRecent Trades:")
        for trade in trader.state.completed_trades[-10:]:
            result = "✅" if trade.is_winner else "❌"
            print(f"  {result} {trade.symbol} {trade.direction}: "
                  f"{trade.pnl_r:+.2f}R ({trade.exit_reason})")


def cmd_advance(args):
    """Advance to next phase."""
    trader = create_trader(args)

    print(f"\nCurrent phase: {trader.state.phase.value}")
    print(f"Trades: {trader.state.trade_count}/{trader.phase_config.min_trades}")
    print(f"PF: {trader.state.profit_factor:.2f} (need {trader.phase_config.min_pf_progress})")
    print(f"WR: {trader.state.win_rate:.1%} (need {trader.phase_config.min_wr_progress:.0%})")

    if trader.state.should_progress():
        if trader.advance_phase():
            print(f"\n✅ Advanced to {trader.state.phase.value}")
        else:
            print("\n⚠️  Failed to advance")
    else:
        print("\n❌ Cannot advance - criteria not met")


def cmd_reset(args):
    """Reset forward test state."""
    if STATE_FILE.exists():
        confirm = input(f"Reset state file {STATE_FILE}? [y/N] ")
        if confirm.lower() == 'y':
            STATE_FILE.unlink()
            print("State reset.")
        else:
            print("Cancelled.")
    else:
        print("No state file to reset.")


def main():
    parser = argparse.ArgumentParser(
        description="Bybit Paper Trader for Forward Testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--api-key', type=str, help='Bybit testnet API key')
    parser.add_argument('--api-secret', type=str, help='Bybit testnet API secret')
    parser.add_argument('--symbols', type=str, help='Comma-separated symbols')
    parser.add_argument('--timeframe', type=str, default='4h', help='Detection timeframe')
    parser.add_argument('--phase', type=int, default=1, choices=[1, 2, 3], help='Forward test phase')

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Scan command
    scan_parser = subparsers.add_parser('scan', help='Scan for signals')
    scan_parser.add_argument('--execute', action='store_true', help='Execute signals')

    # Watch command
    watch_parser = subparsers.add_parser('watch', help='Continuous monitoring')
    watch_parser.add_argument('--interval', type=int, default=300, help='Scan interval in seconds')
    watch_parser.add_argument('--execute', action='store_true', help='Execute signals')

    # Check command
    subparsers.add_parser('check', help='Check open positions')

    # Status command
    subparsers.add_parser('status', help='Show current status')

    # Advance command
    subparsers.add_parser('advance', help='Advance to next phase')

    # Reset command
    subparsers.add_parser('reset', help='Reset forward test state')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Route to command handler
    commands = {
        'scan': cmd_scan,
        'watch': cmd_watch,
        'check': cmd_check,
        'status': cmd_status,
        'advance': cmd_advance,
        'reset': cmd_reset,
    }

    handler = commands.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
