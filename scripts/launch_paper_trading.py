#!/usr/bin/env python3
"""
Phase 9.8: Paper Trading Launch Script
======================================
Verifies infrastructure and launches paper trading with BASE system.

Usage:
    python scripts/launch_paper_trading.py           # Interactive launch
    python scripts/launch_paper_trading.py --check   # Check only, don't launch
    python scripts/launch_paper_trading.py --force   # Skip confirmation
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_api_keys() -> tuple[bool, str]:
    """Check if Bybit API keys are configured."""
    api_key = os.environ.get('BYBIT_TESTNET_API_KEY')
    api_secret = os.environ.get('BYBIT_TESTNET_API_SECRET')

    if not api_key or not api_secret:
        return False, """
API keys not configured. Set environment variables:

    export BYBIT_TESTNET_API_KEY="your_api_key"
    export BYBIT_TESTNET_API_SECRET="your_api_secret"

Get testnet keys at: https://testnet.bybit.com
1. Create account
2. Go to API Management
3. Create new API key with trading permissions
"""
    return True, f"API Key: {api_key[:8]}...{api_key[-4:]}"


def check_bybit_connection() -> tuple[bool, str]:
    """Test Bybit testnet connection."""
    try:
        from src.execution import BybitTestnetClient
        client = BybitTestnetClient(testnet=True)
        balance = client.get_balance()

        if balance and balance.total_equity > 0:
            return True, f"Balance: ${balance.total_equity:,.2f} USDT"
        else:
            return True, "Connected (balance may be 0 on testnet)"
    except Exception as e:
        return False, f"Connection failed: {e}"


def check_detection_module() -> tuple[bool, str]:
    """Verify detection module is available."""
    try:
        from src.detection.hierarchical_swing import HierarchicalSwingDetector
        return True, "HierarchicalSwingDetector available"
    except ImportError as e:
        return False, f"Detection module error: {e}"


def check_log_directory() -> tuple[bool, str]:
    """Verify logging directory exists."""
    log_dir = PROJECT_ROOT / 'logs' / 'paper_trading'
    summary_file = log_dir / 'summary.json'

    if log_dir.exists() and summary_file.exists():
        return True, f"Log directory: {log_dir}"
    else:
        return False, f"Missing log directory or summary.json"


def display_trading_parameters():
    """Display trading parameters."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║           QML PAPER TRADING - Phase 9.8 Configuration            ║
╠══════════════════════════════════════════════════════════════════╣
║  System: BASE (no funding filter)                                ║
║  Phase: 1 of 3                                                   ║
╠══════════════════════════════════════════════════════════════════╣
║  RISK PARAMETERS                                                 ║
║  ├─ Risk per trade:      0.5% of account                         ║
║  ├─ Max concurrent:      3 positions                             ║
║  ├─ Daily loss limit:    2%                                      ║
║  └─ Max consecutive:     5 losses → pause                        ║
╠══════════════════════════════════════════════════════════════════╣
║  TRADING SCOPE                                                   ║
║  ├─ Symbols:             BTCUSDT, ETHUSDT (most liquid)          ║
║  ├─ Timeframe:           4H only (simple start)                  ║
║  └─ Scan interval:       15 minutes                              ║
╠══════════════════════════════════════════════════════════════════╣
║  PHASE 1 TARGETS (50 trades)                                     ║
║  ├─ Progress criteria:   PF > 1.5, WR > 45%                      ║
║  ├─ Pause trigger:       PF < 1.0                                ║
║  └─ Shutdown trigger:    PF < 0.7                                ║
╚══════════════════════════════════════════════════════════════════╝
""")


def run_preflight_checks() -> bool:
    """Run all preflight checks."""
    print("\n" + "="*60)
    print("PREFLIGHT CHECKS")
    print("="*60)

    checks = [
        ("API Keys", check_api_keys),
        ("Bybit Connection", check_bybit_connection),
        ("Detection Module", check_detection_module),
        ("Log Directory", check_log_directory),
    ]

    all_passed = True
    for name, check_func in checks:
        passed, message = check_func()
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"\n{status}: {name}")
        print(f"       {message}")
        if not passed:
            all_passed = False

    print("\n" + "="*60)
    return all_passed


def update_summary_started():
    """Update summary.json with start time."""
    summary_path = PROJECT_ROOT / 'logs' / 'paper_trading' / 'summary.json'

    with open(summary_path) as f:
        summary = json.load(f)

    summary['started_at'] = datetime.now().isoformat()
    summary['status'] = 'ACTIVE'

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Updated {summary_path}")


def launch_paper_trader():
    """Launch the paper trader in watch mode."""
    import subprocess

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / 'scripts' / 'run_bybit_paper_trader.py'),
        'watch',
        '--execute',
        '--interval', '900',  # 15 minutes
    ]

    print("\nLaunching paper trader...")
    print(f"Command: {' '.join(cmd)}")
    print("\nPress Ctrl+C to stop.\n")

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nPaper trader stopped by user.")


def main():
    parser = argparse.ArgumentParser(description='Launch QML Paper Trading')
    parser.add_argument('--check', action='store_true', help='Check only, do not launch')
    parser.add_argument('--force', action='store_true', help='Skip confirmation prompt')
    args = parser.parse_args()

    print("""
╔══════════════════════════════════════════════════════════════════╗
║     ██████╗ ███╗   ███╗██╗         PAPER TRADING                 ║
║    ██╔═══██╗████╗ ████║██║         Phase 9.8 Launch              ║
║    ██║   ██║██╔████╔██║██║         BASE System                   ║
║    ██║▄▄ ██║██║╚██╔╝██║██║                                       ║
║    ╚██████╔╝██║ ╚═╝ ██║███████╗    PF: 4.49 | WR: 54.7%          ║
║     ╚══▀▀═╝ ╚═╝     ╚═╝╚══════╝                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")

    display_trading_parameters()

    all_passed = run_preflight_checks()

    if not all_passed:
        print("\n❌ PREFLIGHT CHECKS FAILED")
        print("Fix the issues above before launching paper trading.")
        sys.exit(1)

    if args.check:
        print("\n✅ ALL CHECKS PASSED - Ready to launch")
        print("Run without --check to start paper trading.")
        sys.exit(0)

    # Confirmation
    if not args.force:
        print("\n" + "="*60)
        print("READY TO LAUNCH")
        print("="*60)
        print("\nThis will start paper trading on Bybit testnet.")
        print("Real detection, simulated orders, real market data.")

        response = input("\nType 'START' to begin paper trading: ")
        if response.strip().upper() != 'START':
            print("Launch cancelled.")
            sys.exit(0)

    # Update summary and launch
    update_summary_started()
    launch_paper_trader()


if __name__ == '__main__':
    main()
