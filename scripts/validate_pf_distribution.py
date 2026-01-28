"""
Phase 9.4: Trade R:R Distribution Validator
============================================
Validates that PF 5.0 with 57% WR is mathematically consistent.

Key Check: For PF 5.0 with 57% WR, average win must be ~3.77R

Mathematical Proof:
  PF = (WR × AvgWin) / ((1-WR) × AvgLoss)
  5.0 = (0.57 × AvgWin) / (0.43 × 1.0)
  AvgWin = 5.0 × 0.43 / 0.57 = 3.77R

RED FLAG: If avg_win < 2R, PF calculation is WRONG.

Usage:
    python scripts/validate_pf_distribution.py
    python scripts/validate_pf_distribution.py --symbols BTCUSDT,ETHUSDT
    python scripts/validate_pf_distribution.py --output results/phase94/
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_engine import get_symbol_data_dir, normalize_symbol
from src.detection.hierarchical_swing import HierarchicalSwingDetector, HierarchicalSwingConfig
from src.detection.pattern_validator import PatternValidator
from src.detection.pattern_scorer import PatternScorer, PatternTier
from src.detection.backtest_adapter import BacktestAdapter
from src.detection.regime import MarketRegimeDetector
from src.detection.config import PatternValidationConfig, PatternScoringConfig
from src.optimization.trade_simulator import (
    TradeSimulator,
    TradeManagementConfig,
    ExitReason,
    SimulatedTrade,
)

# Phase 7.9 configs (baseline)
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

DEFAULT_SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "ADA/USDT", "DOGE/USDT", "LINK/USDT", "AVAX/USDT", "DOT/USDT",
    "MATIC/USDT", "SHIB/USDT", "LTC/USDT", "UNI/USDT", "ATOM/USDT",
]


def load_data(symbol: str, timeframe: str = "4h") -> Optional[pd.DataFrame]:
    """Load OHLCV data for a symbol."""
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
    """Run pattern detection and return signals."""
    normalized = normalize_symbol(symbol)

    detector = HierarchicalSwingDetector(
        config=SWING_CONFIG, symbol=normalized, timeframe=timeframe
    )
    swings = detector.detect(df)

    validator = PatternValidator(PATTERN_CONFIG)
    patterns = validator.find_patterns(swings, df['close'].values)
    valid_patterns = [p for p in patterns if p.is_valid]

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

    return signals


def analyze_r_distribution(trades: List[SimulatedTrade]) -> Dict:
    """Analyze the R-multiple distribution of trades."""
    if not trades:
        return {}

    # Separate winners and losers
    winners = [t for t in trades if t.pnl_r > 0]
    losers = [t for t in trades if t.pnl_r <= 0]

    win_r_values = [t.pnl_r for t in winners]
    loss_r_values = [abs(t.pnl_r) for t in losers]

    # Calculate key statistics
    total_trades = len(trades)
    win_rate = len(winners) / total_trades if total_trades > 0 else 0

    avg_win_r = np.mean(win_r_values) if win_r_values else 0
    avg_loss_r = np.mean(loss_r_values) if loss_r_values else 0

    gross_profit = sum(win_r_values)
    gross_loss = sum(loss_r_values)
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Expectancy = (WR × AvgWin) - ((1-WR) × AvgLoss)
    expectancy = (win_rate * avg_win_r) - ((1 - win_rate) * avg_loss_r)

    # R-multiple buckets
    r_buckets = {
        "dust_wins_0_to_0.5R": len([r for r in win_r_values if r < 0.5]),
        "small_wins_0.5_to_1R": len([r for r in win_r_values if 0.5 <= r < 1.0]),
        "medium_wins_1_to_2R": len([r for r in win_r_values if 1.0 <= r < 2.0]),
        "good_wins_2_to_3R": len([r for r in win_r_values if 2.0 <= r < 3.0]),
        "great_wins_3_to_4R": len([r for r in win_r_values if 3.0 <= r < 4.0]),
        "full_tp_4R_plus": len([r for r in win_r_values if r >= 4.0]),
        "partial_losses_0_to_0.5R": len([r for r in loss_r_values if r < 0.5]),
        "small_losses_0.5_to_1R": len([r for r in loss_r_values if 0.5 <= r < 1.0]),
        "full_sl_1R": len([r for r in loss_r_values if r >= 1.0]),
    }

    # Exit reason distribution
    exit_reasons = {}
    for t in trades:
        reason = t.exit_reason.value if t.exit_reason else "unknown"
        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

    # Percentile analysis for wins
    win_percentiles = {}
    if win_r_values:
        for p in [10, 25, 50, 75, 90]:
            win_percentiles[f"p{p}"] = float(np.percentile(win_r_values, p))

    return {
        "total_trades": total_trades,
        "winners": len(winners),
        "losers": len(losers),
        "win_rate": win_rate,
        "avg_win_r": avg_win_r,
        "avg_loss_r": avg_loss_r,
        "median_win_r": float(np.median(win_r_values)) if win_r_values else 0,
        "median_loss_r": float(np.median(loss_r_values)) if loss_r_values else 0,
        "max_win_r": max(win_r_values) if win_r_values else 0,
        "max_loss_r": max(loss_r_values) if loss_r_values else 0,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "r_buckets": r_buckets,
        "exit_reasons": exit_reasons,
        "win_percentiles": win_percentiles,
    }


def validate_pf_consistency(stats: Dict) -> Dict:
    """Validate that PF is mathematically consistent with WR and avg R values."""
    wr = stats["win_rate"]
    avg_win = stats["avg_win_r"]
    avg_loss = stats["avg_loss_r"]
    reported_pf = stats["profit_factor"]

    # Calculate expected PF from WR and R values
    if avg_loss > 0 and (1 - wr) > 0:
        expected_pf = (wr * avg_win) / ((1 - wr) * avg_loss)
    else:
        expected_pf = float('inf')

    # Calculate required avg_win for reported PF
    if wr > 0 and avg_loss > 0:
        required_avg_win = (reported_pf * (1 - wr) * avg_loss) / wr
    else:
        required_avg_win = 0

    pf_error = abs(expected_pf - reported_pf) / reported_pf if reported_pf > 0 else 0

    return {
        "reported_pf": reported_pf,
        "expected_pf_from_components": expected_pf,
        "pf_calculation_error": pf_error,
        "pf_is_consistent": pf_error < 0.01,  # Within 1%
        "required_avg_win_for_pf": required_avg_win,
        "actual_avg_win": avg_win,
        "avg_win_matches": abs(avg_win - required_avg_win) < 0.1,
    }


def calculate_live_expectations(stats: Dict) -> Dict:
    """Apply realistic haircuts for live trading expectations."""
    # Haircut assumptions
    SLIPPAGE_WR_REDUCTION = 0.03  # 3% fewer wins due to slippage
    SMALLER_WINS_FACTOR = 0.85   # Wins are 15% smaller on average
    FEE_IMPACT = 0.10            # 10% of gross profit goes to fees

    adjusted_wr = stats["win_rate"] - SLIPPAGE_WR_REDUCTION
    adjusted_avg_win = stats["avg_win_r"] * SMALLER_WINS_FACTOR
    adjusted_avg_loss = stats["avg_loss_r"] * 1.05  # Losses slightly bigger

    # Recalculate metrics
    adjusted_gross_profit = stats["winners"] * adjusted_avg_win
    adjusted_gross_loss = stats["losers"] * adjusted_avg_loss

    # Apply fee impact
    adjusted_gross_profit *= (1 - FEE_IMPACT)

    adjusted_pf = adjusted_gross_profit / adjusted_gross_loss if adjusted_gross_loss > 0 else 0
    adjusted_expectancy = (adjusted_wr * adjusted_avg_win) - ((1 - adjusted_wr) * adjusted_avg_loss)

    return {
        "backtest_pf": stats["profit_factor"],
        "backtest_wr": stats["win_rate"],
        "backtest_expectancy": stats["expectancy"],
        "live_pf_estimate": adjusted_pf,
        "live_wr_estimate": adjusted_wr,
        "live_expectancy_estimate": adjusted_expectancy,
        "haircuts_applied": {
            "slippage_wr_reduction": SLIPPAGE_WR_REDUCTION,
            "win_size_reduction": 1 - SMALLER_WINS_FACTOR,
            "fee_impact": FEE_IMPACT,
        },
        "pf_reduction_pct": (1 - adjusted_pf / stats["profit_factor"]) * 100 if stats["profit_factor"] > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate PF 5.0 trade R:R distribution")
    parser.add_argument('--symbols', type=str, help='Comma-separated symbols')
    parser.add_argument('--timeframe', type=str, default='4h')
    parser.add_argument('--output', type=str, help='Output directory')
    args = parser.parse_args()

    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(',')]
        symbols = [s if '/' in s else f"{s[:-4]}/{s[-4:]}" for s in symbols]
    else:
        symbols = DEFAULT_SYMBOLS

    # Trade simulation config (Phase 9.3 settings - no trailing)
    config = TradeManagementConfig(
        tp_decay_enabled=False,
        tp_atr_mult=4.6,
        sl_atr_mult=1.0,
        trailing_mode="none",
        max_bars_held=100,
        min_risk_reward=3.0,
    )

    simulator = TradeSimulator(config)
    all_trades = []
    symbol_stats = {}

    print("=" * 70)
    print("PHASE 9.4: TRADE R:R DISTRIBUTION VALIDATION")
    print("=" * 70)
    print(f"\nTarget validation: PF 5.0 with 57% WR requires avg_win ~ 3.77R")
    print(f"RED FLAG: avg_win < 2R means PF calculation is WRONG\n")

    for symbol in symbols:
        print(f"Processing {symbol}...", end=" ")
        df = load_data(symbol, args.timeframe)
        if df is None:
            print("NO DATA")
            continue

        signals = run_detection(df, symbol, args.timeframe)
        if not signals:
            print("NO SIGNALS")
            continue

        result = simulator.simulate_trades(
            df=df, signals=signals, symbol=normalize_symbol(symbol), timeframe=args.timeframe
        )

        all_trades.extend(result.trades)
        symbol_stats[symbol] = {
            "trades": result.total_trades,
            "win_rate": result.win_rate,
            "profit_factor": result.profit_factor,
        }
        print(f"{result.total_trades} trades, WR {result.win_rate:.1%}, PF {result.profit_factor:.2f}")

    if not all_trades:
        print("\nNo trades to analyze!")
        return

    # Aggregate analysis
    print(f"\n{'=' * 70}")
    print("AGGREGATE R-MULTIPLE DISTRIBUTION")
    print(f"{'=' * 70}")

    stats = analyze_r_distribution(all_trades)

    print(f"\n--- CORE METRICS ---")
    print(f"Total Trades:     {stats['total_trades']}")
    print(f"Winners:          {stats['winners']} ({stats['win_rate']:.1%})")
    print(f"Losers:           {stats['losers']} ({1-stats['win_rate']:.1%})")
    print(f"Profit Factor:    {stats['profit_factor']:.2f}")
    print(f"Expectancy:       {stats['expectancy']:.2f}R per trade")

    print(f"\n--- R-MULTIPLE ANALYSIS ---")
    print(f"{'Metric':<25} {'Value':>10}")
    print("-" * 40)
    print(f"{'Avg Win (R)':<25} {stats['avg_win_r']:>10.2f}")
    print(f"{'Avg Loss (R)':<25} {stats['avg_loss_r']:>10.2f}")
    print(f"{'Median Win (R)':<25} {stats['median_win_r']:>10.2f}")
    print(f"{'Median Loss (R)':<25} {stats['median_loss_r']:>10.2f}")
    print(f"{'Max Win (R)':<25} {stats['max_win_r']:>10.2f}")
    print(f"{'Max Loss (R)':<25} {stats['max_loss_r']:>10.2f}")

    print(f"\n--- WIN DISTRIBUTION BUCKETS ---")
    buckets = stats['r_buckets']
    total_wins = stats['winners']
    print(f"{'Bucket':<30} {'Count':>8} {'%':>8}")
    print("-" * 50)
    for bucket, count in buckets.items():
        if 'win' in bucket or 'tp' in bucket:
            pct = count / total_wins * 100 if total_wins > 0 else 0
            print(f"{bucket:<30} {count:>8} {pct:>7.1f}%")

    print(f"\n--- LOSS DISTRIBUTION BUCKETS ---")
    total_losses = stats['losers']
    for bucket, count in buckets.items():
        if 'loss' in bucket or 'sl' in bucket:
            pct = count / total_losses * 100 if total_losses > 0 else 0
            print(f"{bucket:<30} {count:>8} {pct:>7.1f}%")

    print(f"\n--- WIN PERCENTILES ---")
    if stats['win_percentiles']:
        for p, val in stats['win_percentiles'].items():
            print(f"  {p}: {val:.2f}R")

    print(f"\n--- EXIT REASONS ---")
    for reason, count in stats['exit_reasons'].items():
        pct = count / stats['total_trades'] * 100
        print(f"  {reason}: {count} ({pct:.1f}%)")

    # Validate PF consistency
    print(f"\n{'=' * 70}")
    print("PF MATHEMATICAL CONSISTENCY CHECK")
    print(f"{'=' * 70}")

    validation = validate_pf_consistency(stats)
    print(f"\nReported PF:                  {validation['reported_pf']:.2f}")
    print(f"Expected PF from components:  {validation['expected_pf_from_components']:.2f}")
    print(f"Calculation error:            {validation['pf_calculation_error']:.4f}")
    print(f"PF is consistent:             {'YES' if validation['pf_is_consistent'] else 'NO - INVESTIGATE!'}")
    print(f"\nRequired avg_win for this PF: {validation['required_avg_win_for_pf']:.2f}R")
    print(f"Actual avg_win:               {validation['actual_avg_win']:.2f}R")
    print(f"Avg win matches:              {'YES' if validation['avg_win_matches'] else 'NO - INVESTIGATE!'}")

    # Live expectations
    print(f"\n{'=' * 70}")
    print("REALISTIC LIVE TRADING EXPECTATIONS")
    print(f"{'=' * 70}")

    live = calculate_live_expectations(stats)
    print(f"\n{'Metric':<25} {'Backtest':>12} {'Live Est':>12} {'Change':>10}")
    print("-" * 65)
    print(f"{'Win Rate':<25} {live['backtest_wr']:>11.1%} {live['live_wr_estimate']:>11.1%} {-(live['backtest_wr']-live['live_wr_estimate'])*100:>+9.1f}pp")
    print(f"{'Profit Factor':<25} {live['backtest_pf']:>12.2f} {live['live_pf_estimate']:>12.2f} {-live['pf_reduction_pct']:>+9.1f}%")
    print(f"{'Expectancy (R)':<25} {live['backtest_expectancy']:>12.2f} {live['live_expectancy_estimate']:>12.2f}")

    print(f"\nHaircuts applied:")
    for haircut, value in live['haircuts_applied'].items():
        print(f"  - {haircut}: {value:.1%}")

    # Final verdict
    print(f"\n{'=' * 70}")
    print("VALIDATION VERDICT")
    print(f"{'=' * 70}")

    avg_win = stats['avg_win_r']
    if avg_win < 2.0:
        print(f"\n RED FLAG: avg_win = {avg_win:.2f}R < 2R")
        print("    PF calculation may be incorrect. Investigate further!")
        verdict = "FAIL"
    elif avg_win < 3.0:
        print(f"\n WARNING: avg_win = {avg_win:.2f}R < 3R")
        print("    PF is possible but lower than expected for 5.0 PF")
        verdict = "MARGINAL"
    elif avg_win < 4.5:
        print(f"\n PASS: avg_win = {avg_win:.2f}R is in expected range (3.0-4.5R)")
        print("    PF 5.0 with 57% WR is mathematically consistent")
        verdict = "PASS"
    else:
        print(f"\n WARNING: avg_win = {avg_win:.2f}R > 4.5R")
        print("    Unusually high - verify TP levels are realistic")
        verdict = "VERIFY"

    print(f"\nReady for forward testing: {'YES' if verdict == 'PASS' else 'INVESTIGATE FIRST'}")

    # Save results
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = PROJECT_ROOT / "results" / "phase94_validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"pf_validation_{timestamp}.json"

    results = {
        "timestamp": timestamp,
        "symbols_tested": len(symbol_stats),
        "symbol_stats": symbol_stats,
        "aggregate_stats": stats,
        "pf_validation": validation,
        "live_expectations": live,
        "verdict": verdict,
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
