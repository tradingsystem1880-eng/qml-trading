"""
Phase 7.9 Baseline Verification Script
=======================================
Verifies the TRUE baseline performance before making any changes.

This script runs with TRAILING STOP DISABLED to establish what the
actual fixed TP/SL performance is.

Expected baseline from CLAUDE.md:
- Win Rate: ~22-25% (NOT 52%!)
- Profit Factor: 1.18-1.28
- Trades: 2000-2300
- DSR: 0.986

If results differ significantly, there's a bug to investigate.

Usage:
    python scripts/verify_phase79_baseline.py
    python scripts/verify_phase79_baseline.py --tolerance 0.15
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_engine import get_symbol_data_dir, normalize_symbol
from src.detection.hierarchical_swing import HierarchicalSwingDetector, HierarchicalSwingConfig
from src.detection.pattern_validator import PatternValidator
from src.detection.pattern_scorer import PatternScorer, PatternTier
from src.detection.backtest_adapter import BacktestAdapter
from src.detection.regime import MarketRegimeDetector
from src.detection.config import (
    PatternValidationConfig,
    PatternScoringConfig,
)
from src.optimization.trade_simulator import (
    TradeSimulator,
    TradeManagementConfig,
    SimulationResult,
)

# Detection configs from Phase 7.9 profit_factor_penalized (2,155 trades baseline)
# Source: results/phase77_optimization/profit_factor_penalized/final_results.json
# CRITICAL: Must use HierarchicalSwingDetector (not HistoricalSwingDetector)
SWING_CONFIG = HierarchicalSwingConfig(
    min_bar_separation=3,
    min_move_atr=0.85,  # Phase 7.9: 0.85
    forward_confirm_pct=0.2,
    lookback=6,  # Phase 7.9: 6
    lookforward=8,  # Phase 7.9: 8
)

PATTERN_CONFIG = PatternValidationConfig(
    p3_min_extension_atr=0.3,
    p3_max_extension_atr=5.0,  # Phase 7.9: 5.0
    p5_max_symmetry_atr=4.6,  # Phase 7.9: 4.6
    min_pattern_bars=16,  # Phase 7.9: 16
    max_pattern_bars=200,
)

SCORING_CONFIG = PatternScoringConfig()

# All symbols to test for comprehensive baseline
ALL_SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
    "ADA/USDT", "DOGE/USDT", "LINK/USDT", "AVAX/USDT", "DOT/USDT",
    "MATIC/USDT", "ATOM/USDT", "UNI/USDT", "LTC/USDT", "FIL/USDT",
]

# Expected baseline metrics from Phase 7.9 profit_factor_penalized
# Source: results/phase77_optimization/profit_factor_penalized/final_results.json
EXPECTED_BASELINE = {
    "win_rate": 0.224,  # Phase 7.9: 22.4%
    "profit_factor": 1.23,  # Phase 7.9: 1.23
    "min_trades": 100,  # Phase 7.9 had 2,155 across 22 symbols (~98/symbol)
}


def load_data(symbol: str, timeframe: str = "4h") -> pd.DataFrame:
    """Load price data for a symbol."""
    data_dir = get_symbol_data_dir(symbol)
    data_path = data_dir / f"{timeframe}_master.parquet"

    if not data_path.exists():
        return None

    df = pd.read_parquet(data_path)
    df.columns = [c.lower() for c in df.columns]
    if 'timestamp' in df.columns:
        df = df.rename(columns={'timestamp': 'time'})

    return df


def run_detection(df: pd.DataFrame, symbol: str, timeframe: str = "4h") -> List[Dict]:
    """Run detection pipeline and return signals for simulation."""
    normalized = normalize_symbol(symbol)

    # 1. Detect swings (CRITICAL: use HierarchicalSwingDetector, not HistoricalSwingDetector)
    detector = HierarchicalSwingDetector(
        config=SWING_CONFIG,
        symbol=normalized,
        timeframe=timeframe,
    )
    swings = detector.detect(df)

    # 2. Find patterns
    validator = PatternValidator(PATTERN_CONFIG)
    patterns = validator.find_patterns(swings, df['close'].values)
    valid_patterns = [p for p in patterns if p.is_valid]

    # 3. Score patterns with regime (calculated per-pattern as per Phase 9.1 fix)
    scorer = PatternScorer(SCORING_CONFIG)
    regime_detector = MarketRegimeDetector()

    scored = []
    for p in valid_patterns:
        # Get regime at pattern's P5 time
        p5_idx = p.p5.bar_index
        window_start = max(0, p5_idx - 150)
        window_df = df.iloc[window_start:p5_idx + 1].copy()
        regime_result = regime_detector.get_regime(window_df)

        score_result = scorer.score(p, df=df, regime_result=regime_result)
        if score_result.tier != PatternTier.REJECT:
            scored.append((p, score_result))

    # 4. Convert to signals for TradeSimulator
    adapter = BacktestAdapter()
    validation_results = [vr for vr, sr in scored]
    scoring_results = [sr for vr, sr in scored]

    signals_raw = adapter.batch_convert_to_signals(
        validation_results=validation_results,
        scoring_results=scoring_results,
        symbol=normalized,
        min_tier=PatternTier.C,
    )

    # Convert Signal objects to dicts for TradeSimulator
    signals = []
    for sig in signals_raw:
        sig_time = sig.timestamp
        if hasattr(sig_time, 'tzinfo') and sig_time.tzinfo is not None:
            sig_time = sig_time.replace(tzinfo=None)

        df_time = df['time']
        if df_time.dt.tz is not None:
            df_time = df_time.dt.tz_localize(None)

        bar_indices = df[df_time >= sig_time].index
        if len(bar_indices) == 0:
            continue
        bar_idx = bar_indices[0]

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

    return signals


def run_baseline_verification(
    symbols: List[str] = None,
    timeframe: str = "4h",
    verbose: bool = True,
) -> Dict:
    """
    Run baseline verification with TRAILING STOP DISABLED.

    This establishes the true fixed TP/SL performance.
    """
    if symbols is None:
        symbols = ALL_SYMBOLS

    # CRITICAL: Use EXACT Phase 7.9 parameters
    # Source: results/phase77_optimization/profit_factor_penalized/final_results.json
    baseline_config = TradeManagementConfig(
        tp_decay_enabled=False,
        tp_atr_mult=4.6,  # Phase 7.9: 4.6
        sl_atr_mult=1.0,  # Phase 7.9: 1.0
        trailing_mode="none",  # Phase 7.9: trailing_activation_atr=0.0
        trailing_activation_atr=0.0,
        trailing_step_atr=0.0,
        max_bars_held=100,  # Phase 7.9: 100
        min_risk_reward=3.0,  # Phase 7.9: 3.0
    )

    simulator = TradeSimulator(baseline_config)
    all_trades = []

    for symbol in symbols:
        if verbose:
            print(f"Processing {symbol}...", end=" ")

        try:
            df = load_data(symbol, timeframe)
            if df is None:
                if verbose:
                    print("NO DATA")
                continue

            signals = run_detection(df, symbol, timeframe)
            if not signals:
                if verbose:
                    print("NO SIGNALS")
                continue

            result = simulator.simulate_trades(
                df=df,
                signals=signals,
                symbol=normalize_symbol(symbol),
                timeframe=timeframe,
            )
            all_trades.extend(result.trades)

            if verbose:
                print(f"{result.total_trades} trades, WR={result.win_rate:.1%}, PF={result.profit_factor:.2f}")

        except Exception as e:
            if verbose:
                print(f"ERROR: {e}")
            continue

    # Calculate aggregate metrics
    if not all_trades:
        return {"error": "No trades generated"}

    winners = [t for t in all_trades if t.pnl_r > 0]
    losers = [t for t in all_trades if t.pnl_r <= 0]

    total = len(all_trades)
    win_rate = len(winners) / total if total > 0 else 0

    avg_win_r = np.mean([t.pnl_r for t in winners]) if winners else 0
    avg_loss_r = abs(np.mean([t.pnl_r for t in losers])) if losers else 0

    gross_profit = sum(t.pnl_r for t in winners) if winners else 0
    gross_loss = abs(sum(t.pnl_r for t in losers)) if losers else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

    expectancy = (win_rate * avg_win_r) - ((1 - win_rate) * avg_loss_r)

    # Exit reason distribution
    exit_dist = {}
    for t in all_trades:
        reason = t.exit_reason.value if t.exit_reason else 'unknown'
        exit_dist[reason] = exit_dist.get(reason, 0) + 1

    # Bars held distribution
    bars_held = [t.bars_held for t in all_trades]
    avg_bars = np.mean(bars_held)
    median_bars = np.median(bars_held)

    # Dust profit analysis (trades with <0.1R profit counted as wins)
    dust_wins = [t for t in winners if 0 < t.pnl_r < 0.1]
    dust_pct = len(dust_wins) / len(winners) if winners else 0

    return {
        "total_trades": total,
        "winners": len(winners),
        "losers": len(losers),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "expectancy_r": expectancy,
        "avg_win_r": avg_win_r,
        "avg_loss_r": avg_loss_r,
        "avg_bars_held": avg_bars,
        "median_bars_held": median_bars,
        "exit_distribution": exit_dist,
        "dust_wins": len(dust_wins),
        "dust_pct": dust_pct,
        "symbols_tested": len(symbols),
    }


def verify_against_expected(results: Dict, tolerance: float = 0.15) -> bool:
    """
    Verify results match expected baseline within tolerance.

    Returns True if results are within acceptable range.
    """
    passed = True
    issues = []

    # Check win rate
    expected_wr = EXPECTED_BASELINE["win_rate"]
    actual_wr = results["win_rate"]
    wr_diff = abs(actual_wr - expected_wr) / expected_wr
    if wr_diff > tolerance:
        passed = False
        issues.append(f"Win rate: expected {expected_wr:.1%}, got {actual_wr:.1%} (diff={wr_diff:.1%})")

    # Check profit factor
    expected_pf = EXPECTED_BASELINE["profit_factor"]
    actual_pf = results["profit_factor"]
    pf_diff = abs(actual_pf - expected_pf) / expected_pf
    if pf_diff > tolerance:
        passed = False
        issues.append(f"Profit factor: expected {expected_pf:.2f}, got {actual_pf:.2f} (diff={pf_diff:.1%})")

    # Check minimum trades
    if results["total_trades"] < EXPECTED_BASELINE["min_trades"]:
        passed = False
        issues.append(f"Too few trades: {results['total_trades']} < {EXPECTED_BASELINE['min_trades']}")

    # Red flags - realistic thresholds for trading systems
    if results["win_rate"] > 0.60:
        passed = False
        issues.append(f"PATHOLOGICAL: Win rate {results['win_rate']:.1%} > 60% is unrealistic for R:R 4.6:1")

    if results["profit_factor"] > 2.5:
        passed = False
        issues.append(f"PATHOLOGICAL: PF {results['profit_factor']:.2f} > 2.5 is unrealistic")

    if results["avg_bars_held"] < 3:
        passed = False
        issues.append(f"PATHOLOGICAL: Avg bars held {results['avg_bars_held']:.1f} < 3 suggests broken exits")

    if results["dust_pct"] > 0.50:
        passed = False
        issues.append(f"PATHOLOGICAL: {results['dust_pct']:.1%} of wins are dust (<0.1R)")

    return passed, issues


def main():
    parser = argparse.ArgumentParser(description='Verify Phase 7.9 Baseline')
    parser.add_argument('--symbols', type=str, help='Comma-separated symbols (default: all)')
    parser.add_argument('--tolerance', type=float, default=0.15, help='Tolerance for metric comparison')
    parser.add_argument('--quiet', action='store_true', help='Suppress per-symbol output')
    args = parser.parse_args()

    # Parse symbols
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(',')]
        symbols = [s if '/' in s else f"{s[:-4]}/{s[-4:]}" for s in symbols]
    else:
        symbols = ALL_SYMBOLS

    print("=" * 70)
    print("PHASE 9.2: BASELINE VERIFICATION")
    print("=" * 70)
    print(f"Testing {len(symbols)} symbols with TRAILING STOP DISABLED")
    print(f"Tolerance: {args.tolerance:.0%}")
    print("=" * 70)

    # Run verification
    results = run_baseline_verification(
        symbols=symbols,
        verbose=not args.quiet,
    )

    if "error" in results:
        print(f"\nERROR: {results['error']}")
        return

    # Print results
    print("\n" + "=" * 70)
    print("BASELINE RESULTS (NO TRAILING STOP)")
    print("=" * 70)
    print(f"Total Trades:     {results['total_trades']}")
    print(f"Win Rate:         {results['win_rate']:.1%}")
    print(f"Profit Factor:    {results['profit_factor']:.2f}")
    print(f"Expectancy (R):   {results['expectancy_r']:.3f}")
    print(f"Avg Win (R):      {results['avg_win_r']:.2f}")
    print(f"Avg Loss (R):     {results['avg_loss_r']:.2f}")
    print(f"Avg Bars Held:    {results['avg_bars_held']:.1f}")
    print(f"Median Bars Held: {results['median_bars_held']:.0f}")
    print(f"\nExit Distribution:")
    for reason, count in results['exit_distribution'].items():
        pct = count / results['total_trades'] * 100
        print(f"  {reason}: {count} ({pct:.1f}%)")
    print(f"\nDust Wins (<0.1R): {results['dust_wins']} ({results['dust_pct']:.1%} of winners)")

    # Verify against expected
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)

    passed, issues = verify_against_expected(results, args.tolerance)

    if passed:
        print("PASSED - Baseline metrics are within expected range")
    else:
        print("FAILED - Issues found:")
        for issue in issues:
            print(f"  - {issue}")

    # Save results
    results_dir = PROJECT_ROOT / "results" / "phase92_verification"
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = results_dir / f"baseline_{timestamp}.json"

    serializable = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                   for k, v in results.items()}
    serializable['timestamp'] = timestamp
    serializable['tolerance'] = args.tolerance
    serializable['verification_passed'] = passed
    serializable['issues'] = issues

    with open(results_path, 'w') as f:
        json.dump(serializable, f, indent=2)

    print(f"\nResults saved: {results_path}")

    return results


if __name__ == "__main__":
    main()
