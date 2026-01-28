"""
Compare Exit Strategies: Fixed TP vs Time-Decaying TP
======================================================
A/B test comparing the Phase 7.9 baseline (fixed TP) against
the Phase 9.0 adaptive exit strategy (time-decaying TP).

This script:
1. Runs detection on multiple symbols
2. Simulates trades with fixed TP
3. Simulates trades with time-decaying TP
4. Compares MFE capture, profit factor, win rate

Usage:
    python scripts/compare_exit_strategies.py
    python scripts/compare_exit_strategies.py --symbols BTCUSDT,ETHUSDT
    python scripts/compare_exit_strategies.py --halflife 15 --min-r 0.8
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

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

# Default symbols for testing
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
        raise FileNotFoundError(f"No data found for {symbol} at {data_path}")

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

    # 3. Score patterns with regime
    # IMPORTANT: Calculate regime AT EACH PATTERN'S TIME, not once for entire df
    scorer = PatternScorer(SCORING_CONFIG)
    regime_detector = MarketRegimeDetector()

    scored = []
    for p in valid_patterns:
        # Get regime at pattern's P5 time (need 110+ bars for proper calculation)
        p5_idx = p.p5.bar_index
        window_start = max(0, p5_idx - 150)  # 150 bars to ensure enough data
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
        # Find bar index for this signal
        sig_time = sig.timestamp
        if hasattr(sig_time, 'tzinfo') and sig_time.tzinfo is not None:
            sig_time = sig_time.replace(tzinfo=None)

        # Make df time tz-naive for comparison
        df_time = df['time']
        if df_time.dt.tz is not None:
            df_time = df_time.dt.tz_localize(None)

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

    return signals


def run_comparison(
    symbols: List[str],
    timeframe: str = "4h",
    halflife_bars: int = 20,
    min_r: float = 0.5,
    trailing_mode: str = "multi_stage",
    verbose: bool = True,
) -> Dict:
    """
    Run A/B comparison between fixed and adaptive exits.

    Returns dict with comparison results.
    """
    # Configure simulators
    # Phase 7.9 baseline: sl_atr_mult=1.0, tp_atr_mult=4.6, min_risk_reward=3.0
    # Phase 9.2: Use multi_stage trailing by default (fixes breakeven bug)
    fixed_config = TradeManagementConfig(
        tp_decay_enabled=False,
        tp_atr_mult=4.6,  # Phase 7.9: 4.6 (not 3.0)
        sl_atr_mult=1.0,  # Phase 7.9: 1.0 (not 1.5)
        trailing_mode=trailing_mode,
        max_bars_held=100,  # Phase 7.9: 100 (not 50)
        min_risk_reward=3.0,  # Phase 7.9: 3.0 (not 1.5)
    )

    adaptive_config = TradeManagementConfig(
        tp_decay_enabled=True,
        tp_decay_halflife_bars=halflife_bars,
        tp_minimum_r=min_r,
        tp_atr_mult=4.6,  # Phase 7.9: 4.6
        sl_atr_mult=1.0,  # Phase 7.9: 1.0
        trailing_mode=trailing_mode,
        max_bars_held=100,  # Phase 7.9: 100
        min_risk_reward=3.0,  # Phase 7.9: 3.0
    )

    fixed_sim = TradeSimulator(fixed_config)
    adaptive_sim = TradeSimulator(adaptive_config)

    # Aggregate results
    fixed_all_trades = []
    adaptive_all_trades = []

    for symbol in symbols:
        if verbose:
            print(f"\n{'='*50}")
            print(f"Processing {symbol}")
            print("=" * 50)

        try:
            # Load data
            df = load_data(symbol, timeframe)
            if verbose:
                print(f"  Loaded {len(df)} bars")

            # Run detection
            signals = run_detection(df, symbol, timeframe)
            if verbose:
                print(f"  Detected {len(signals)} signals")

            if not signals:
                continue

            # Simulate with fixed TP
            fixed_result = fixed_sim.simulate_trades(
                df=df,
                signals=signals,
                symbol=normalize_symbol(symbol),
                timeframe=timeframe,
            )
            fixed_all_trades.extend(fixed_result.trades)

            # Simulate with adaptive TP
            adaptive_result = adaptive_sim.simulate_trades(
                df=df,
                signals=signals,
                symbol=normalize_symbol(symbol),
                timeframe=timeframe,
            )
            adaptive_all_trades.extend(adaptive_result.trades)

            if verbose:
                print(f"  Fixed:    {fixed_result.total_trades} trades, "
                      f"WR={fixed_result.win_rate:.1%}, PF={fixed_result.profit_factor:.2f}")
                print(f"  Adaptive: {adaptive_result.total_trades} trades, "
                      f"WR={adaptive_result.win_rate:.1%}, PF={adaptive_result.profit_factor:.2f}")

        except Exception as e:
            if verbose:
                print(f"  Error: {e}")
            continue

    # Calculate aggregate metrics
    fixed_agg = _calculate_aggregate_metrics(fixed_all_trades)
    adaptive_agg = _calculate_aggregate_metrics(adaptive_all_trades)

    return {
        'fixed': fixed_agg,
        'adaptive': adaptive_agg,
        'improvement': {
            'mfe_capture_delta': adaptive_agg['mfe_capture'] - fixed_agg['mfe_capture'],
            'pf_delta': adaptive_agg['profit_factor'] - fixed_agg['profit_factor'],
            'wr_delta': adaptive_agg['win_rate'] - fixed_agg['win_rate'],
            'expectancy_delta': adaptive_agg['expectancy_r'] - fixed_agg['expectancy_r'],
        },
        'config': {
            'halflife_bars': halflife_bars,
            'min_r': min_r,
            'symbols': symbols,
        }
    }


def _calculate_aggregate_metrics(trades: List) -> Dict:
    """Calculate aggregate metrics from trade list."""
    if not trades:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'expectancy_r': 0,
            'mfe_capture': 0,
            'avg_mfe_r': 0,
            'avg_mae_r': 0,
            'avg_win_r': 0,
            'avg_loss_r': 0,
        }

    winners = [t for t in trades if t.pnl_r > 0]
    losers = [t for t in trades if t.pnl_r <= 0]

    total = len(trades)
    win_rate = len(winners) / total if total > 0 else 0

    avg_win_r = np.mean([t.pnl_r for t in winners]) if winners else 0
    avg_loss_r = abs(np.mean([t.pnl_r for t in losers])) if losers else 0

    gross_profit = sum(t.pnl_r for t in winners) if winners else 0
    gross_loss = abs(sum(t.pnl_r for t in losers)) if losers else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

    expectancy = (win_rate * avg_win_r) - ((1 - win_rate) * avg_loss_r)

    avg_mfe_r = np.mean([t.mfe_r for t in trades])
    avg_mae_r = np.mean([t.mae_r for t in trades])

    # MFE capture: what % of favorable move we actually captured
    winners_mfe = [t.mfe_r for t in winners] if winners else [0]
    avg_winners_mfe = np.mean(winners_mfe) if winners else 0
    mfe_capture = avg_win_r / avg_winners_mfe if avg_winners_mfe > 0 else 0

    # Avg bars held
    avg_bars = np.mean([t.bars_held for t in trades]) if trades else 0

    return {
        'total_trades': total,
        'winners': len(winners),
        'losers': len(losers),
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'expectancy_r': expectancy,
        'avg_win_r': avg_win_r,
        'avg_loss_r': avg_loss_r,
        'avg_mfe_r': avg_mfe_r,
        'avg_mae_r': avg_mae_r,
        'mfe_capture': mfe_capture,
        'avg_bars_held': avg_bars,
    }


def print_comparison_report(results: Dict):
    """Print formatted comparison report."""
    fixed = results['fixed']
    adaptive = results['adaptive']
    improvement = results['improvement']
    config = results['config']

    print("\n" + "=" * 70)
    print("EXIT STRATEGY COMPARISON REPORT")
    print("=" * 70)
    print(f"Symbols: {', '.join(config['symbols'])}")
    print(f"Adaptive Config: halflife={config['halflife_bars']} bars, min_r={config['min_r']}")
    print("=" * 70)

    print("\n" + "-" * 70)
    print(f"{'Metric':<25} {'Fixed TP':<15} {'Adaptive TP':<15} {'Delta':<15}")
    print("-" * 70)

    metrics = [
        ('Total Trades', 'total_trades', None, ''),
        ('Win Rate', 'win_rate', 'wr_delta', '%'),
        ('Profit Factor', 'profit_factor', 'pf_delta', ''),
        ('Expectancy (R)', 'expectancy_r', 'expectancy_delta', ''),
        ('Avg Win (R)', 'avg_win_r', None, ''),
        ('Avg Loss (R)', 'avg_loss_r', None, ''),
        ('Avg Bars Held', 'avg_bars_held', None, ''),
        ('MFE Capture', 'mfe_capture', 'mfe_capture_delta', '%'),
        ('Avg MFE (R)', 'avg_mfe_r', None, ''),
        ('Avg MAE (R)', 'avg_mae_r', None, ''),
    ]

    for label, key, delta_key, fmt in metrics:
        f_val = fixed.get(key, 0)
        a_val = adaptive.get(key, 0)

        if fmt == '%':
            f_str = f"{f_val:.1%}"
            a_str = f"{a_val:.1%}"
        elif isinstance(f_val, int):
            f_str = f"{f_val}"
            a_str = f"{a_val}"
        else:
            f_str = f"{f_val:.3f}"
            a_str = f"{a_val:.3f}"

        if delta_key:
            delta = improvement.get(delta_key, 0)
            if fmt == '%':
                d_str = f"{delta:+.1%}"
            else:
                d_str = f"{delta:+.3f}"
        else:
            d_str = "-"

        print(f"{label:<25} {f_str:<15} {a_str:<15} {d_str:<15}")

    print("-" * 70)

    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    mfe_improved = improvement['mfe_capture_delta'] > 0.05  # >5% improvement
    pf_maintained = improvement['pf_delta'] >= -0.05  # No more than 5% regression

    if mfe_improved and pf_maintained:
        print("✅ ADAPTIVE EXITS RECOMMENDED")
        print(f"   MFE capture improved by {improvement['mfe_capture_delta']:.1%}")
        print(f"   Profit factor delta: {improvement['pf_delta']:+.2f}")
    elif mfe_improved and not pf_maintained:
        print("⚠️  MIXED RESULTS - MORE TESTING NEEDED")
        print(f"   MFE capture improved by {improvement['mfe_capture_delta']:.1%}")
        print(f"   BUT profit factor regressed by {improvement['pf_delta']:.2f}")
    else:
        print("❌ KEEP FIXED EXITS (No significant improvement)")
        print(f"   MFE capture delta: {improvement['mfe_capture_delta']:.1%}")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Compare Fixed vs Adaptive Exit Strategies')
    parser.add_argument('--symbols', type=str, help='Comma-separated symbols')
    parser.add_argument('--halflife', type=int, default=20, help='TP decay halflife in bars')
    parser.add_argument('--min-r', type=float, default=0.5, help='Minimum R-multiple for TP')
    parser.add_argument('--trailing-mode', type=str, default='multi_stage',
                       choices=['none', 'simple', 'multi_stage'],
                       help='Trailing stop mode (Phase 9.2: multi_stage fixes breakeven bug)')
    parser.add_argument('--quiet', action='store_true', help='Suppress per-symbol output')
    args = parser.parse_args()

    # Parse symbols
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(',')]
        # Normalize format
        symbols = [s if '/' in s else f"{s[:-4]}/{s[-4:]}" for s in symbols]
    else:
        symbols = DEFAULT_SYMBOLS

    print("=" * 70)
    print("PHASE 9.2: EXIT STRATEGY A/B COMPARISON")
    print("=" * 70)
    print(f"Testing {len(symbols)} symbols")
    print(f"Trailing mode: {args.trailing_mode}")
    print(f"Adaptive config: halflife={args.halflife} bars, min_r={args.min_r}")

    # Run comparison
    results = run_comparison(
        symbols=symbols,
        halflife_bars=args.halflife,
        min_r=args.min_r,
        trailing_mode=args.trailing_mode,
        verbose=not args.quiet,
    )

    # Print report
    print_comparison_report(results)

    # Save results
    results_dir = PROJECT_ROOT / "results" / "phase90_exits"
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = results_dir / f"comparison_{timestamp}.json"

    import json
    with open(results_path, 'w') as f:
        # Convert numpy types for JSON serialization
        serializable = {
            'fixed': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                     for k, v in results['fixed'].items()},
            'adaptive': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                        for k, v in results['adaptive'].items()},
            'improvement': {k: float(v) for k, v in results['improvement'].items()},
            'config': results['config'],
            'timestamp': timestamp,
        }
        json.dump(serializable, f, indent=2)

    print(f"\n📊 Results saved: {results_path}")

    return results


if __name__ == "__main__":
    main()
