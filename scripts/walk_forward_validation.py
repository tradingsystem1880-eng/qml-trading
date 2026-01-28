"""
Phase 9.4: Walk-Forward Validation
==================================
Validates PF consistency across 5 time-based folds.

Success Criteria: All 5 folds must have PF > 1.5

If any fold has PF < 1.0, the edge is not robust.

Usage:
    python scripts/walk_forward_validation.py
    python scripts/walk_forward_validation.py --folds 5
    python scripts/walk_forward_validation.py --symbols BTCUSDT,ETHUSDT
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
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

DEFAULT_SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "ADA/USDT", "DOGE/USDT", "LINK/USDT", "AVAX/USDT", "DOT/USDT",
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
    """Run pattern detection and return signals with bar indices."""
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
            'timestamp': sig_time,  # Keep for fold assignment
        })

    return signals


def split_data_into_folds(df: pd.DataFrame, n_folds: int) -> List[Tuple[int, int]]:
    """Split data into n time-based folds.

    Returns list of (start_idx, end_idx) tuples.
    """
    total_bars = len(df)
    fold_size = total_bars // n_folds

    folds = []
    for i in range(n_folds):
        start_idx = i * fold_size
        if i == n_folds - 1:
            end_idx = total_bars  # Last fold gets remaining
        else:
            end_idx = (i + 1) * fold_size
        folds.append((start_idx, end_idx))

    return folds


def calculate_fold_metrics(trades: List[SimulatedTrade]) -> Dict:
    """Calculate metrics for a single fold."""
    if not trades:
        return {
            "total_trades": 0,
            "win_rate": 0,
            "profit_factor": 0,
            "expectancy": 0,
            "avg_win_r": 0,
            "avg_loss_r": 0,
        }

    winners = [t for t in trades if t.pnl_r > 0]
    losers = [t for t in trades if t.pnl_r <= 0]

    total = len(trades)
    wr = len(winners) / total

    gross_profit = sum(t.pnl_r for t in winners) if winners else 0
    gross_loss = abs(sum(t.pnl_r for t in losers)) if losers else 0
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    avg_win = np.mean([t.pnl_r for t in winners]) if winners else 0
    avg_loss = abs(np.mean([t.pnl_r for t in losers])) if losers else 0

    expectancy = (wr * avg_win) - ((1 - wr) * avg_loss)

    return {
        "total_trades": total,
        "winners": len(winners),
        "losers": len(losers),
        "win_rate": wr,
        "profit_factor": pf,
        "expectancy": expectancy,
        "avg_win_r": avg_win,
        "avg_loss_r": avg_loss,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
    }


def main():
    parser = argparse.ArgumentParser(description="Walk-forward validation")
    parser.add_argument('--symbols', type=str, help='Comma-separated symbols')
    parser.add_argument('--timeframe', type=str, default='4h')
    parser.add_argument('--folds', type=int, default=5, help='Number of folds')
    parser.add_argument('--output', type=str, help='Output directory')
    args = parser.parse_args()

    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(',')]
        symbols = [s if '/' in s else f"{s[:-4]}/{s[-4:]}" for s in symbols]
    else:
        symbols = DEFAULT_SYMBOLS

    n_folds = args.folds

    # Trade simulation config
    config = TradeManagementConfig(
        tp_decay_enabled=False,
        tp_atr_mult=4.6,
        sl_atr_mult=1.0,
        trailing_mode="none",
        max_bars_held=100,
        min_risk_reward=3.0,
    )

    simulator = TradeSimulator(config)

    print("=" * 70)
    print(f"PHASE 9.4: WALK-FORWARD VALIDATION ({n_folds} FOLDS)")
    print("=" * 70)
    print(f"\nSuccess Criteria: All folds must have PF > 1.5")
    print(f"Warning: Any fold with PF < 1.0 indicates unreliable edge\n")

    # Aggregate trades by fold across all symbols
    fold_trades = {i: [] for i in range(n_folds)}
    fold_date_ranges = {}

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

        # Get fold boundaries
        folds = split_data_into_folds(df, n_folds)

        # Store date ranges for reporting
        for i, (start_idx, end_idx) in enumerate(folds):
            if i not in fold_date_ranges:
                start_date = df.iloc[start_idx]['time']
                end_date = df.iloc[min(end_idx - 1, len(df) - 1)]['time']
                fold_date_ranges[i] = (start_date, end_date)

        # Assign signals to folds based on bar_idx
        signals_by_fold = {i: [] for i in range(n_folds)}
        for sig in signals:
            bar_idx = sig['bar_idx']
            for fold_idx, (start_idx, end_idx) in enumerate(folds):
                if start_idx <= bar_idx < end_idx:
                    signals_by_fold[fold_idx].append(sig)
                    break

        # Simulate trades for each fold
        for fold_idx, fold_signals in signals_by_fold.items():
            if not fold_signals:
                continue

            result = simulator.simulate_trades(
                df=df,
                signals=fold_signals,
                symbol=normalize_symbol(symbol),
                timeframe=args.timeframe
            )
            fold_trades[fold_idx].extend(result.trades)

        signal_counts = [len(signals_by_fold[i]) for i in range(n_folds)]
        print(f"{len(signals)} signals distributed: {signal_counts}")

    # Calculate metrics for each fold
    print(f"\n{'=' * 70}")
    print("FOLD RESULTS")
    print(f"{'=' * 70}")

    fold_results = []
    header = f"{'Fold':<6} {'Date Range':<30} {'Trades':>8} {'WR':>8} {'PF':>8} {'Exp(R)':>8} {'Status':<10}"
    print(f"\n{header}")
    print("-" * 90)

    all_pass = True
    any_fail = False

    for fold_idx in range(n_folds):
        trades = fold_trades[fold_idx]
        metrics = calculate_fold_metrics(trades)

        date_range = fold_date_ranges.get(fold_idx, ("N/A", "N/A"))
        date_str = f"{date_range[0]:%Y-%m-%d} to {date_range[1]:%Y-%m-%d}" if date_range[0] != "N/A" else "N/A"

        pf = metrics['profit_factor']
        if pf >= 1.5:
            status = "PASS"
        elif pf >= 1.0:
            status = "MARGINAL"
            all_pass = False
        else:
            status = "FAIL"
            all_pass = False
            any_fail = True

        print(f"{fold_idx + 1:<6} {date_str:<30} {metrics['total_trades']:>8} "
              f"{metrics['win_rate']:>7.1%} {pf:>7.2f} {metrics['expectancy']:>+7.2f} {status:<10}")

        fold_results.append({
            "fold": fold_idx + 1,
            "date_range": date_str,
            **metrics,
            "status": status,
        })

    # Aggregate metrics
    print(f"\n{'=' * 70}")
    print("AGGREGATE METRICS")
    print(f"{'=' * 70}")

    all_trades = []
    for trades in fold_trades.values():
        all_trades.extend(trades)

    aggregate = calculate_fold_metrics(all_trades)

    print(f"\nTotal Trades:     {aggregate['total_trades']}")
    print(f"Win Rate:         {aggregate['win_rate']:.1%}")
    print(f"Profit Factor:    {aggregate['profit_factor']:.2f}")
    print(f"Expectancy:       {aggregate['expectancy']:.2f}R per trade")
    print(f"Avg Win:          {aggregate['avg_win_r']:.2f}R")
    print(f"Avg Loss:         {aggregate['avg_loss_r']:.2f}R")

    # Stability analysis
    print(f"\n{'=' * 70}")
    print("STABILITY ANALYSIS")
    print(f"{'=' * 70}")

    pf_values = [r['profit_factor'] for r in fold_results if r['total_trades'] >= 10]
    wr_values = [r['win_rate'] for r in fold_results if r['total_trades'] >= 10]

    if pf_values:
        pf_mean = np.mean(pf_values)
        pf_std = np.std(pf_values)
        pf_cv = pf_std / pf_mean if pf_mean > 0 else float('inf')

        wr_mean = np.mean(wr_values)
        wr_std = np.std(wr_values)

        print(f"\nProfit Factor across folds:")
        print(f"  Mean: {pf_mean:.2f}, Std: {pf_std:.2f}, CV: {pf_cv:.2%}")
        print(f"  Min:  {min(pf_values):.2f}, Max: {max(pf_values):.2f}")

        print(f"\nWin Rate across folds:")
        print(f"  Mean: {wr_mean:.1%}, Std: {wr_std:.1%}")
        print(f"  Min:  {min(wr_values):.1%}, Max: {max(wr_values):.1%}")

        # CV < 0.5 is considered stable
        stability = "STABLE" if pf_cv < 0.5 else "UNSTABLE"
        print(f"\nStability Assessment: {stability}")
        if pf_cv >= 0.5:
            print("  Warning: High variance in PF across folds")

    # Final verdict
    print(f"\n{'=' * 70}")
    print("VALIDATION VERDICT")
    print(f"{'=' * 70}")

    passing_folds = sum(1 for r in fold_results if r.get('status') == 'PASS')
    marginal_folds = sum(1 for r in fold_results if r.get('status') == 'MARGINAL')
    failing_folds = sum(1 for r in fold_results if r.get('status') == 'FAIL')

    print(f"\nFolds: {passing_folds} PASS, {marginal_folds} MARGINAL, {failing_folds} FAIL")

    if all_pass:
        verdict = "PASS"
        print(f"\n PASS: All {n_folds} folds have PF > 1.5")
        print("    Edge is robust across different time periods")
        print("    Ready for forward testing")
    elif any_fail:
        verdict = "FAIL"
        print(f"\n FAIL: {failing_folds} fold(s) have PF < 1.0")
        print("    Edge is NOT reliable - investigate before forward testing")
    else:
        verdict = "MARGINAL"
        print(f"\n MARGINAL: Some folds have PF between 1.0 and 1.5")
        print("    Edge exists but may be weak in certain market conditions")
        print("    Consider proceeding with reduced position sizes")

    # Save results
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = PROJECT_ROOT / "results" / "phase94_validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"walk_forward_{timestamp}.json"

    results = {
        "timestamp": timestamp,
        "n_folds": n_folds,
        "symbols_tested": symbols,
        "fold_results": fold_results,
        "aggregate_metrics": aggregate,
        "stability": {
            "pf_mean": pf_mean if pf_values else None,
            "pf_std": pf_std if pf_values else None,
            "pf_cv": pf_cv if pf_values else None,
        },
        "verdict": verdict,
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
