"""
Phase 9.3: Compare WITH vs WITHOUT TrendRegimeValidator
========================================================
Tests the hypothesis that TrendRegimeValidator explains the win rate discrepancy.

Phase 7.9 used:
- min_r_squared: 0.8
- min_adx: 30.0
- min_trend_move_atr: 2.0
- min_trend_swings: 4

If TrendValidator explains the discrepancy, we should see:
- WITH TrendValidator: ~22% WR, ~1.23 PF (matching Phase 7.9)
- WITHOUT TrendValidator: ~57% WR, ~5.0 PF (current results)

Usage:
    python scripts/compare_with_trend_validator.py
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_engine import get_symbol_data_dir, normalize_symbol
from src.detection.hierarchical_swing import HierarchicalSwingDetector, HierarchicalSwingConfig
from src.detection.pattern_validator import PatternValidator, PatternDirection
from src.detection.pattern_scorer import PatternScorer, PatternTier
from src.detection.backtest_adapter import BacktestAdapter
from src.detection.regime import MarketRegimeDetector
from src.detection.config import PatternValidationConfig, PatternScoringConfig
from src.detection.trend_validator import (
    TrendValidator,
    TrendValidationConfig,
)
from src.optimization.trade_simulator import (
    TradeSimulator,
    TradeManagementConfig,
)

# Phase 7.9 configs
SWING_CONFIG = HierarchicalSwingConfig(
    min_bar_separation=3,
    min_move_atr=0.85,
    forward_confirm_pct=0.2,
    lookback=6,
    lookforward=8,
)

PATTERN_CONFIG = PatternValidationConfig(
    p3_min_extension_atr=0.3,
    p3_max_extension_atr=5.0,
    p5_max_symmetry_atr=4.6,
    min_pattern_bars=16,
    max_pattern_bars=200,
)

SCORING_CONFIG = PatternScoringConfig()

# Phase 7.9 TrendValidator configs
TREND_VALIDATION_CONFIG = TrendValidationConfig(
    min_adx=30.0,
    min_trend_move_atr=2.0,
    min_trend_swings=4,
    min_trend_bars=15,
    max_trend_bars=200,
    direction_consistency=0.6,
)

# Note: TrendRegimeValidator (R²-based) was NOT used in Phase 7.9 filtering
# The min_r_squared parameter was in optimization space but never connected

# Trade management config (Phase 7.9)
TRADE_CONFIG = TradeManagementConfig(
    tp_decay_enabled=False,
    tp_atr_mult=4.6,
    sl_atr_mult=1.0,
    trailing_mode="none",
    trailing_activation_atr=0.0,
    trailing_step_atr=0.0,
    max_bars_held=100,
    min_risk_reward=3.0,
)

# Test symbols
TEST_SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "ADA/USDT", "DOGE/USDT", "LINK/USDT", "AVAX/USDT", "DOT/USDT",
]


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


def run_detection_with_trend_filter(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str = "4h",
    use_trend_validator: bool = False,
) -> Tuple[List[Dict], Dict]:
    """
    Run detection pipeline with optional TrendValidator filtering.

    Returns:
        (signals, stats) where stats contains filtering statistics
    """
    normalized = normalize_symbol(symbol)

    # 1. Detect swings
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

    # 3. Score patterns with regime
    scorer = PatternScorer(SCORING_CONFIG)
    regime_detector = MarketRegimeDetector()

    scored = []
    for p in valid_patterns:
        p5_idx = p.p5.bar_index
        window_start = max(0, p5_idx - 150)
        window_df = df.iloc[window_start:p5_idx + 1].copy()
        regime_result = regime_detector.get_regime(window_df)

        score_result = scorer.score(p, df=df, regime_result=regime_result)
        if score_result.tier != PatternTier.REJECT:
            scored.append((p, score_result))

    # Statistics
    stats = {
        "total_patterns": len(valid_patterns),
        "after_scoring": len(scored),
        "trend_filtered": 0,
        "trend_passed": 0,
        "trend_rejection_reasons": {},
    }

    # 4. Optional: Filter by TrendValidator
    if use_trend_validator:
        trend_validator = TrendValidator(TREND_VALIDATION_CONFIG)

        filtered_scored = []
        for p, score_result in scored:
            # Determine expected trend direction
            if p.direction == PatternDirection.BULLISH:
                expected_trend = 'UP'
            else:
                expected_trend = 'DOWN'

            # Validate using TrendValidator ONLY (Phase 7.9 did NOT use TrendRegimeValidator)
            # Note: min_r_squared was in optimization space but never connected to filtering
            trend_result = trend_validator.validate(
                swings=swings,
                p1_bar_index=p.p1.bar_index,
                df=df,
                pattern_direction=p.direction,
            )

            # Pattern must pass TrendValidator (Phase 7.9 behavior)
            if trend_result.is_valid:
                stats["trend_passed"] += 1
                filtered_scored.append((p, score_result))
            else:
                stats["trend_filtered"] += 1
                reason = trend_result.rejection_reason
                stats["trend_rejection_reasons"][reason] = stats["trend_rejection_reasons"].get(reason, 0) + 1

        scored = filtered_scored

    # 5. Convert to signals
    adapter = BacktestAdapter()
    validation_results = [vr for vr, sr in scored]
    scoring_results = [sr for vr, sr in scored]

    signals_raw = adapter.batch_convert_to_signals(
        validation_results=validation_results,
        scoring_results=scoring_results,
        symbol=normalized,
        min_tier=PatternTier.C,
    )

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

    return signals, stats


def calculate_metrics(trades: List) -> Dict:
    """Calculate aggregate metrics from trades."""
    if not trades:
        return {
            "total_trades": 0,
            "win_rate": 0,
            "profit_factor": 0,
            "avg_win_r": 0,
            "avg_loss_r": 0,
            "expectancy_r": 0,
        }

    winners = [t for t in trades if t.pnl_r > 0]
    losers = [t for t in trades if t.pnl_r <= 0]

    total = len(trades)
    win_rate = len(winners) / total

    avg_win_r = np.mean([t.pnl_r for t in winners]) if winners else 0
    avg_loss_r = abs(np.mean([t.pnl_r for t in losers])) if losers else 0

    gross_profit = sum(t.pnl_r for t in winners) if winners else 0
    gross_loss = abs(sum(t.pnl_r for t in losers)) if losers else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

    expectancy = (win_rate * avg_win_r) - ((1 - win_rate) * avg_loss_r)

    return {
        "total_trades": total,
        "winners": len(winners),
        "losers": len(losers),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_win_r": avg_win_r,
        "avg_loss_r": avg_loss_r,
        "expectancy_r": expectancy,
        "avg_bars_held": np.mean([t.bars_held for t in trades]) if trades else 0,
    }


def run_comparison(symbols: List[str], verbose: bool = True) -> Dict:
    """Run comparison with and without TrendValidator."""
    simulator = TradeSimulator(TRADE_CONFIG)

    # Results storage
    results = {
        "without_trend_validator": {
            "trades": [],
            "patterns_total": 0,
            "patterns_after_scoring": 0,
        },
        "with_trend_validator": {
            "trades": [],
            "patterns_total": 0,
            "patterns_after_scoring": 0,
            "patterns_trend_passed": 0,
            "patterns_trend_filtered": 0,
            "rejection_reasons": {},
        },
    }

    for symbol in symbols:
        if verbose:
            print(f"\nProcessing {symbol}...")

        df = load_data(symbol)
        if df is None:
            if verbose:
                print("  NO DATA")
            continue

        # Run WITHOUT TrendValidator
        signals_without, stats_without = run_detection_with_trend_filter(
            df, symbol, use_trend_validator=False
        )

        results["without_trend_validator"]["patterns_total"] += stats_without["total_patterns"]
        results["without_trend_validator"]["patterns_after_scoring"] += stats_without["after_scoring"]

        if signals_without:
            result = simulator.simulate_trades(
                df=df,
                signals=signals_without,
                symbol=normalize_symbol(symbol),
                timeframe="4h",
            )
            results["without_trend_validator"]["trades"].extend(result.trades)

            if verbose:
                print(f"  WITHOUT TrendValidator: {len(signals_without)} signals -> {result.total_trades} trades, WR={result.win_rate:.1%}")

        # Run WITH TrendValidator
        signals_with, stats_with = run_detection_with_trend_filter(
            df, symbol, use_trend_validator=True
        )

        results["with_trend_validator"]["patterns_total"] += stats_with["total_patterns"]
        results["with_trend_validator"]["patterns_after_scoring"] += stats_with["after_scoring"]
        results["with_trend_validator"]["patterns_trend_passed"] += stats_with["trend_passed"]
        results["with_trend_validator"]["patterns_trend_filtered"] += stats_with["trend_filtered"]

        for reason, count in stats_with.get("trend_rejection_reasons", {}).items():
            results["with_trend_validator"]["rejection_reasons"][reason] = \
                results["with_trend_validator"]["rejection_reasons"].get(reason, 0) + count

        if signals_with:
            result = simulator.simulate_trades(
                df=df,
                signals=signals_with,
                symbol=normalize_symbol(symbol),
                timeframe="4h",
            )
            results["with_trend_validator"]["trades"].extend(result.trades)

            if verbose:
                print(f"  WITH TrendValidator:    {len(signals_with)} signals -> {result.total_trades} trades, WR={result.win_rate:.1%}")

    # Calculate metrics
    results["without_trend_validator"]["metrics"] = calculate_metrics(
        results["without_trend_validator"]["trades"]
    )
    results["with_trend_validator"]["metrics"] = calculate_metrics(
        results["with_trend_validator"]["trades"]
    )

    # Clean up trades from results (too large to serialize)
    del results["without_trend_validator"]["trades"]
    del results["with_trend_validator"]["trades"]

    return results


def print_report(results: Dict):
    """Print formatted comparison report."""
    print("\n" + "=" * 70)
    print("PHASE 9.3: TRENDVALIDATOR COMPARISON")
    print("=" * 70)

    without = results["without_trend_validator"]
    with_tv = results["with_trend_validator"]

    print("\n--- PATTERN STATISTICS ---")
    print(f"Total patterns found: {without['patterns_total']}")
    print(f"After scoring:        {without['patterns_after_scoring']}")

    if with_tv['patterns_after_scoring'] > 0:
        filter_rate = with_tv['patterns_trend_filtered'] / with_tv['patterns_after_scoring'] * 100
        print(f"\nTrendValidator filtering:")
        print(f"  Passed:   {with_tv['patterns_trend_passed']}")
        print(f"  Filtered: {with_tv['patterns_trend_filtered']} ({filter_rate:.1f}%)")

        print(f"\nRejection reasons:")
        for reason, count in sorted(with_tv['rejection_reasons'].items(), key=lambda x: -x[1]):
            pct = count / with_tv['patterns_trend_filtered'] * 100 if with_tv['patterns_trend_filtered'] > 0 else 0
            print(f"  {reason}: {count} ({pct:.1f}%)")

    print("\n--- TRADE METRICS COMPARISON ---")
    print(f"{'Metric':<20} {'WITHOUT TV':<15} {'WITH TV':<15} {'Phase 7.9':<15}")
    print("-" * 65)

    m_without = without['metrics']
    m_with = with_tv['metrics']

    print(f"{'Total Trades':<20} {m_without['total_trades']:<15} {m_with['total_trades']:<15} {2155:<15}")
    print(f"{'Win Rate':<20} {m_without['win_rate']*100:>5.1f}%{'':8} {m_with['win_rate']*100:>5.1f}%{'':8} {22.4:>5.1f}%")
    print(f"{'Profit Factor':<20} {m_without['profit_factor']:>6.2f}{'':8} {m_with['profit_factor']:>6.2f}{'':8} {1.23:>6.2f}")
    print(f"{'Avg Win (R)':<20} {m_without['avg_win_r']:>6.2f}{'':8} {m_with['avg_win_r']:>6.2f}{'':8}")
    print(f"{'Avg Loss (R)':<20} {m_without['avg_loss_r']:>6.2f}{'':8} {m_with['avg_loss_r']:>6.2f}{'':8}")
    print(f"{'Expectancy (R)':<20} {m_without['expectancy_r']:>6.3f}{'':8} {m_with['expectancy_r']:>6.3f}{'':8} {0.183:>6.3f}")
    print(f"{'Avg Bars Held':<20} {m_without['avg_bars_held']:>6.1f}{'':8} {m_with['avg_bars_held']:>6.1f}{'':8}")

    print("\n--- ANALYSIS ---")

    # Check if TrendValidator explains the discrepancy
    wr_diff_without = abs(m_without['win_rate'] - 0.224) / 0.224 * 100
    wr_diff_with = abs(m_with['win_rate'] - 0.224) / 0.224 * 100 if m_with['total_trades'] > 0 else float('inf')

    pf_diff_without = abs(m_without['profit_factor'] - 1.23) / 1.23 * 100 if m_without['profit_factor'] > 0 else float('inf')
    pf_diff_with = abs(m_with['profit_factor'] - 1.23) / 1.23 * 100 if m_with['profit_factor'] > 0 else float('inf')

    print(f"Win Rate diff from Phase 7.9:")
    print(f"  WITHOUT TrendValidator: {wr_diff_without:.0f}%")
    print(f"  WITH TrendValidator:    {wr_diff_with:.0f}%")

    print(f"\nProfit Factor diff from Phase 7.9:")
    print(f"  WITHOUT TrendValidator: {pf_diff_without:.0f}%")
    print(f"  WITH TrendValidator:    {pf_diff_with:.0f}%")

    if wr_diff_with < wr_diff_without and pf_diff_with < pf_diff_without:
        print("\n✅ TrendValidator DOES explain part of the discrepancy")
        print("   Results with TrendValidator are closer to Phase 7.9")
    else:
        print("\n⚠️  TrendValidator alone does NOT explain the discrepancy")
        print("   Something else is different between Phase 7.9 and current implementation")


def main():
    parser = argparse.ArgumentParser(description='Compare with/without TrendValidator')
    parser.add_argument('--symbols', type=str, help='Comma-separated symbols')
    parser.add_argument('--quiet', action='store_true', help='Suppress per-symbol output')
    args = parser.parse_args()

    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(',')]
        symbols = [s if '/' in s else f"{s[:-4]}/{s[-4:]}" for s in symbols]
    else:
        symbols = TEST_SYMBOLS

    print(f"Comparing TrendValidator impact on {len(symbols)} symbols...")
    print("Phase 7.9 used: min_r_squared=0.8, min_adx=30.0, min_trend_move_atr=2.0, min_trend_swings=4")

    results = run_comparison(symbols, verbose=not args.quiet)
    print_report(results)

    # Save results
    results_dir = PROJECT_ROOT / "results" / "phase93_diagnostic"
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = results_dir / f"trend_validator_comparison_{timestamp}.json"

    # Convert numpy types
    def convert_numpy(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj

    serializable = convert_numpy(results)
    serializable['timestamp'] = timestamp
    serializable['symbols'] = symbols

    with open(results_path, 'w') as f:
        json.dump(serializable, f, indent=2)

    print(f"\nResults saved: {results_path}")


if __name__ == "__main__":
    main()
