"""
Phase 9.0 Forward Testing CLI
==============================
Run forward paper trading with Phase 7.9 baseline parameters.

This script:
1. Runs pattern detection on recent data
2. Simulates trades with Phase 9.0 adaptive exits
3. Tracks performance with ForwardTestMonitor
4. Detects edge degradation vs baseline

Usage:
    # Run forward test on recent data
    python scripts/run_phase90_forward.py --symbols BTC/USDT,ETH/USDT --days 30

    # Show current status from previous runs
    python scripts/run_phase90_forward.py --status

    # Generate detailed report
    python scripts/run_phase90_forward.py --report

    # Test with adaptive exits
    python scripts/run_phase90_forward.py --exit-mode adaptive --halflife 20
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_engine import get_symbol_data_dir, normalize_symbol
from src.detection.historical_detector import HistoricalSwingDetector
from src.detection.pattern_validator import PatternValidator
from src.detection.pattern_scorer import PatternScorer, PatternTier
from src.detection.backtest_adapter import BacktestAdapter
from src.detection.regime import MarketRegimeDetector
from src.detection.config import (
    SwingDetectionConfig,
    PatternValidationConfig,
    PatternScoringConfig,
)
from src.optimization.trade_simulator import (
    TradeSimulator,
    TradeManagementConfig,
)
from src.risk.forward_monitor import ForwardTestMonitor, ForwardTestConfig
from src.risk.position_rules import PositionRulesManager, RiskConfig

# Results directory
RESULTS_DIR = PROJECT_ROOT / "results" / "phase90_forward"
RESULTS_FILE = RESULTS_DIR / "forward_test_state.json"

# Phase 7.9 optimized detection config
SWING_CONFIG = SwingDetectionConfig(
    atr_period=14,
    lookback=5,
    lookforward=3,
    min_zscore=0.5,
    min_threshold_pct=0.0005,
    atr_multiplier=0.5,
)

PATTERN_CONFIG = PatternValidationConfig(
    p3_min_extension_atr=0.3,
    p3_max_extension_atr=10.0,
    p5_max_symmetry_atr=5.0,
    min_pattern_bars=8,
    max_pattern_bars=200,
)

SCORING_CONFIG = PatternScoringConfig()

# Default symbols
DEFAULT_SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "BNB/USDT",
    "SOL/USDT",
    "XRP/USDT",
]


def load_data(symbol: str, timeframe: str = "4h") -> pd.DataFrame:
    """Load price data for a symbol."""
    data_dir = get_symbol_data_dir(symbol)
    data_path = data_dir / f"{timeframe}_master.parquet"

    if not data_path.exists():
        raise FileNotFoundError(f"No data for {symbol} at {data_path}")

    df = pd.read_parquet(data_path)
    df.columns = [c.lower() for c in df.columns]
    if 'timestamp' in df.columns:
        df = df.rename(columns={'timestamp': 'time'})

    return df


def run_detection_and_simulation(
    symbols: List[str],
    timeframe: str = "4h",
    days: int = 30,
    exit_mode: str = "adaptive",
    halflife: int = 20,
    min_r: float = 0.5,
) -> List[Dict]:
    """
    Run detection and trade simulation on recent data.

    Args:
        symbols: List of symbols to test
        timeframe: Timeframe for detection
        days: Number of recent days to test
        exit_mode: 'fixed' or 'adaptive'
        halflife: TP decay halflife for adaptive mode
        min_r: Minimum R-multiple for adaptive mode

    Returns:
        List of trade result dicts
    """
    # Configure simulator
    if exit_mode == "adaptive":
        config = TradeManagementConfig(
            tp_decay_enabled=True,
            tp_decay_halflife_bars=halflife,
            tp_minimum_r=min_r,
            tp_atr_mult=3.0,
            sl_atr_mult=1.5,
            trailing_activation_atr=1.0,
            trailing_step_atr=0.5,
            max_bars_held=50,
        )
    else:
        config = TradeManagementConfig(
            tp_decay_enabled=False,
            tp_atr_mult=3.0,
            sl_atr_mult=1.5,
            trailing_activation_atr=1.0,
            trailing_step_atr=0.5,
            max_bars_held=50,
        )

    simulator = TradeSimulator(config)
    all_trades = []

    for symbol in symbols:
        print(f"\n  Processing {symbol}...")

        try:
            # Load data
            df = load_data(symbol, timeframe)

            # Filter to recent days
            if 'time' in df.columns:
                cutoff = datetime.now() - timedelta(days=days)
                if df['time'].dt.tz is not None:
                    df['time'] = df['time'].dt.tz_localize(None)
                df = df[df['time'] >= cutoff].reset_index(drop=True)

            if len(df) < 100:
                print(f"    Insufficient data ({len(df)} bars)")
                continue

            print(f"    Loaded {len(df)} bars")

            # Run detection
            normalized = normalize_symbol(symbol)
            detector = HistoricalSwingDetector(SWING_CONFIG, normalized, timeframe)
            swings = detector.detect(df)

            validator = PatternValidator(PATTERN_CONFIG)
            patterns = validator.find_patterns(swings, df['close'].values)
            valid = [p for p in patterns if p.is_valid]

            scorer = PatternScorer(SCORING_CONFIG)
            regime_detector = MarketRegimeDetector()

            # IMPORTANT: Calculate regime AT EACH PATTERN'S TIME
            scored = []
            for p in valid:
                # Get regime at pattern's P5 time (need 110+ bars)
                p5_idx = p.p5.bar_index
                window_start = max(0, p5_idx - 150)
                window_df = df.iloc[window_start:p5_idx + 1].copy()
                regime_result = regime_detector.get_regime(window_df)

                sr = scorer.score(p, df=df, regime_result=regime_result)
                if sr.tier != PatternTier.REJECT:
                    scored.append((p, sr))

            print(f"    Patterns: {len(scored)}")

            if not scored:
                continue

            # Convert to signals
            adapter = BacktestAdapter()
            signals_raw = adapter.batch_convert_to_signals(
                validation_results=[vr for vr, sr in scored],
                scoring_results=[sr for vr, sr in scored],
                symbol=normalized,
                min_tier=PatternTier.C,
            )

            # Convert to dicts
            signals = []
            df_time = df['time']
            if df_time.dt.tz is not None:
                df_time = df_time.dt.tz_localize(None)

            for sig in signals_raw:
                sig_time = sig.timestamp
                if hasattr(sig_time, 'tzinfo') and sig_time.tzinfo is not None:
                    sig_time = sig_time.replace(tzinfo=None)

                bar_indices = df[df_time >= sig_time].index
                if len(bar_indices) == 0:
                    continue

                bar_idx = bar_indices[0]
                # Get ATR from signal or from dataframe
                signal_atr = sig.atr_at_signal if hasattr(sig, 'atr_at_signal') and sig.atr_at_signal else None
                if signal_atr is None and 'atr' in df.columns:
                    signal_atr = df.iloc[bar_idx]['atr']
                if signal_atr is None:
                    signal_atr = 0

                signals.append({
                    'bar_idx': bar_idx,
                    'direction': sig.signal_type.value.upper().replace('BUY', 'LONG').replace('SELL', 'SHORT'),
                    'entry_price': sig.price,
                    'atr': signal_atr,
                    'score': sig.validity_score if hasattr(sig, 'validity_score') else 0.5,
                })

            # Simulate trades
            result = simulator.simulate_trades(df, signals, symbol=normalized, timeframe=timeframe)

            # Convert to trade dicts for monitor
            for trade in result.trades:
                all_trades.append({
                    'timestamp': trade.exit_time or datetime.now(),
                    'symbol': normalized,
                    'r_multiple': trade.pnl_r,
                    'exit_type': trade.exit_reason.value if trade.exit_reason else 'unknown',
                    'pattern_quality': trade.pattern_score,
                    'bars_held': trade.bars_held,
                    'mfe_r': trade.mfe_r,
                    'mae_r': trade.mae_r,
                })

            print(f"    Trades: {len(result.trades)}, WR: {result.win_rate:.1%}, PF: {result.profit_factor:.2f}")

        except Exception as e:
            print(f"    Error: {e}")
            continue

    return all_trades


def show_status():
    """Show status from previous forward test runs."""
    if not RESULTS_FILE.exists():
        print("\nNo forward test data found.")
        print(f"Run: python scripts/run_phase90_forward.py --symbols BTC/USDT,ETH/USDT")
        return

    # Load monitor state
    monitor = ForwardTestMonitor(ForwardTestConfig(
        baseline_pf=1.23,
        baseline_wr=0.52,
    ))

    try:
        monitor.load_results(str(RESULTS_FILE))
        status = monitor._calculate_status()

        print("\n" + "=" * 60)
        print("FORWARD TEST STATUS")
        print("=" * 60)
        print(f"Total Trades: {status.total_trades}")
        print(f"Win Rate: {status.running_wr:.1%} (baseline: 52%)")
        print(f"Profit Factor: {status.running_pf:.2f} (baseline: 1.23)")
        print(f"Consecutive Losses: {status.consecutive_losses}")
        print(f"Ready for Live: {'Yes' if status.ready_for_deployment else 'No'}")

        if status.alerts:
            print("\nALERTS:")
            for alert in status.alerts:
                print(f"  - {alert}")

        print("=" * 60)

    except Exception as e:
        print(f"Error loading results: {e}")


def show_report():
    """Show detailed report from previous runs."""
    if not RESULTS_FILE.exists():
        print("\nNo forward test data found.")
        return

    monitor = ForwardTestMonitor(ForwardTestConfig(
        baseline_pf=1.23,
        baseline_wr=0.52,
    ))

    try:
        monitor.load_results(str(RESULTS_FILE))
        print(monitor.generate_report())
    except Exception as e:
        print(f"Error loading results: {e}")


def main():
    parser = argparse.ArgumentParser(description="Phase 9.0 Forward Testing")
    parser.add_argument("--symbols", type=str, help="Comma-separated symbols")
    parser.add_argument("--days", type=int, default=30, help="Days of recent data to test")
    parser.add_argument("--exit-mode", choices=["fixed", "adaptive"], default="adaptive")
    parser.add_argument("--halflife", type=int, default=20, help="TP decay halflife (bars)")
    parser.add_argument("--min-r", type=float, default=0.5, help="Minimum TP R-multiple")
    parser.add_argument("--status", action="store_true", help="Show current status")
    parser.add_argument("--report", action="store_true", help="Show detailed report")
    parser.add_argument("--reset", action="store_true", help="Reset forward test data")
    args = parser.parse_args()

    # Create results directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Handle status/report commands
    if args.status:
        show_status()
        return

    if args.report:
        show_report()
        return

    if args.reset:
        if RESULTS_FILE.exists():
            RESULTS_FILE.unlink()
            print("Forward test data reset.")
        return

    # Parse symbols
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(',')]
        symbols = [s if '/' in s else f"{s[:-4]}/{s[-4:]}" for s in symbols]
    else:
        symbols = DEFAULT_SYMBOLS

    print("=" * 60)
    print("PHASE 9.0 FORWARD TEST")
    print("=" * 60)
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Data Period: Last {args.days} days")
    print(f"Exit Mode: {args.exit_mode}")
    if args.exit_mode == "adaptive":
        print(f"TP Decay Halflife: {args.halflife} bars")
        print(f"Minimum TP: {args.min_r}R")
    print("=" * 60)

    # Initialize monitor
    monitor = ForwardTestMonitor(ForwardTestConfig(
        baseline_pf=1.23,
        baseline_wr=0.52,
        baseline_expectancy=0.18,
    ))

    # Load existing data if present
    if RESULTS_FILE.exists() and not args.reset:
        try:
            monitor.load_results(str(RESULTS_FILE))
            print(f"\nLoaded {len(monitor.trades)} existing trades")
        except Exception:
            pass

    # Run forward test
    print("\nRunning detection and simulation...")
    trades = run_detection_and_simulation(
        symbols=symbols,
        days=args.days,
        exit_mode=args.exit_mode,
        halflife=args.halflife,
        min_r=args.min_r,
    )

    # Record trades to monitor
    for trade in trades:
        monitor.record_trade(trade)

    # Save results
    monitor.save_results(str(RESULTS_FILE))
    print(f"\nResults saved to: {RESULTS_FILE}")

    # Print summary
    status = monitor._calculate_status()

    print("\n" + "=" * 60)
    print("FORWARD TEST RESULTS")
    print("=" * 60)
    print(f"New Trades: {len(trades)}")
    print(f"Total Trades: {status.total_trades}")
    print(f"Win Rate: {status.running_wr:.1%} (baseline: 52%)")
    print(f"Profit Factor: {status.running_pf:.2f} (baseline: 1.23)")
    print(f"Expectancy: {status.running_expectancy:.2f}R (baseline: 0.18R)")

    # Degradation check
    if status.is_degraded:
        print(f"\n  DEGRADATION DETECTED ({status.degradation_severity})")
        for alert in status.alerts:
            print(f"  - {alert}")
    else:
        print(f"\n  Performance within expected range")

    # Deployment readiness
    print("\n" + "-" * 60)
    if status.ready_for_deployment:
        print("VERDICT: READY FOR LIVE DEPLOYMENT")
    else:
        print(f"VERDICT: Continue testing ({status.trades_until_deployment} more trades needed)")

    print("=" * 60)


if __name__ == "__main__":
    main()
