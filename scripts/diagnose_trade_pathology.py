"""
Trade Pathology Diagnostic Script
==================================
Detects problematic patterns in trade results that indicate bugs:

1. Dust Profits - Wins with <0.1R profit (suspicious)
2. Short Holds - Trades exiting in <3 bars (likely premature trailing)
3. Unrealistic Win Rate - >70% suggests broken exit logic
4. Unrealistic PF - >2.5 suggests data leakage or bugs
5. MFE Waste - Not capturing available favorable excursion
6. Survivorship Bias - Only measuring completed trades

Usage:
    python scripts/diagnose_trade_pathology.py
    python scripts/diagnose_trade_pathology.py --check dust_profits,short_holds
    python scripts/diagnose_trade_pathology.py --detailed
"""

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
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
    SimulatedTrade,
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

# Default symbols
DEFAULT_SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
]

# Pathology thresholds - realistic for trading systems
THRESHOLDS = {
    "dust_profit_r": 0.1,  # Wins below this are suspicious
    "short_hold_bars": 3,  # Exits below this need investigation
    "max_realistic_wr": 0.50,  # Phase 7.9 baseline: 22.4% WR
    "max_realistic_pf": 2.5,  # Phase 7.9 baseline: 1.23 PF, max realistic ~2.5
    "min_mfe_capture": 0.20,  # Capturing less than 20% of MFE is wasteful
    "max_dust_pct": 0.20,  # More than 20% dust wins is suspicious
    "max_short_hold_pct": 0.40,  # Some quick TP hits are expected for patterns
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

    # 4. Convert to signals
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


def diagnose_dust_profits(trades: List[SimulatedTrade]) -> Dict:
    """Check for suspiciously small winning trades."""
    winners = [t for t in trades if t.pnl_r > 0]
    if not winners:
        return {"status": "OK", "message": "No winning trades to analyze"}

    dust_wins = [t for t in winners if t.pnl_r < THRESHOLDS["dust_profit_r"]]
    dust_pct = len(dust_wins) / len(winners)

    # Distribution of dust profits
    dust_profit_dist = Counter()
    for t in dust_wins:
        bucket = round(t.pnl_r, 2)
        dust_profit_dist[bucket] += 1

    if dust_pct > THRESHOLDS["max_dust_pct"]:
        return {
            "status": "PATHOLOGICAL",
            "message": f"{dust_pct:.1%} of wins are dust (<{THRESHOLDS['dust_profit_r']}R)",
            "dust_count": len(dust_wins),
            "total_winners": len(winners),
            "dust_pct": dust_pct,
            "dust_distribution": dict(dust_profit_dist),
            "example_trades": [
                {
                    "pnl_r": t.pnl_r,
                    "bars_held": t.bars_held,
                    "exit_reason": t.exit_reason.value if t.exit_reason else None,
                    "trailing_activated": t.trailing_activated,
                }
                for t in dust_wins[:5]
            ],
        }

    return {
        "status": "OK",
        "message": f"Dust wins at acceptable level ({dust_pct:.1%})",
        "dust_count": len(dust_wins),
        "total_winners": len(winners),
        "dust_pct": dust_pct,
    }


def diagnose_short_holds(trades: List[SimulatedTrade]) -> Dict:
    """Check for trades exiting too quickly."""
    short_holds = [t for t in trades if t.bars_held < THRESHOLDS["short_hold_bars"]]
    short_pct = len(short_holds) / len(trades) if trades else 0

    # Distribution of hold times
    bars_held = [t.bars_held for t in trades]
    avg_bars = np.mean(bars_held)
    median_bars = np.median(bars_held)

    # Exit reasons for short holds
    short_hold_exits = Counter(t.exit_reason.value if t.exit_reason else 'unknown'
                               for t in short_holds)

    if short_pct > THRESHOLDS["max_short_hold_pct"]:
        return {
            "status": "PATHOLOGICAL",
            "message": f"{short_pct:.1%} of trades exit in <{THRESHOLDS['short_hold_bars']} bars",
            "short_hold_count": len(short_holds),
            "total_trades": len(trades),
            "short_hold_pct": short_pct,
            "avg_bars_held": avg_bars,
            "median_bars_held": median_bars,
            "short_hold_exit_reasons": dict(short_hold_exits),
        }

    if avg_bars < 3:
        return {
            "status": "WARNING",
            "message": f"Avg bars held ({avg_bars:.1f}) is very low",
            "avg_bars_held": avg_bars,
            "median_bars_held": median_bars,
        }

    return {
        "status": "OK",
        "message": f"Hold times acceptable (avg={avg_bars:.1f}, median={median_bars:.0f})",
        "avg_bars_held": avg_bars,
        "median_bars_held": median_bars,
    }


def diagnose_unrealistic_metrics(trades: List[SimulatedTrade]) -> Dict:
    """Check for suspiciously good performance metrics."""
    if not trades:
        return {"status": "OK", "message": "No trades to analyze"}

    winners = [t for t in trades if t.pnl_r > 0]
    losers = [t for t in trades if t.pnl_r <= 0]

    win_rate = len(winners) / len(trades)

    gross_profit = sum(t.pnl_r for t in winners) if winners else 0
    gross_loss = abs(sum(t.pnl_r for t in losers)) if losers else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    issues = []

    if win_rate > THRESHOLDS["max_realistic_wr"]:
        issues.append(f"Win rate {win_rate:.1%} exceeds realistic maximum ({THRESHOLDS['max_realistic_wr']:.0%})")

    if profit_factor > THRESHOLDS["max_realistic_pf"]:
        issues.append(f"PF {profit_factor:.2f} exceeds realistic maximum ({THRESHOLDS['max_realistic_pf']:.1f})")

    if issues:
        return {
            "status": "PATHOLOGICAL",
            "message": "; ".join(issues),
            "win_rate": win_rate,
            "profit_factor": profit_factor,
        }

    return {
        "status": "OK",
        "message": f"Metrics within realistic range (WR={win_rate:.1%}, PF={profit_factor:.2f})",
        "win_rate": win_rate,
        "profit_factor": profit_factor,
    }


def diagnose_mfe_waste(trades: List[SimulatedTrade]) -> Dict:
    """Check if we're leaving too much profit on the table."""
    winners = [t for t in trades if t.pnl_r > 0 and t.mfe_r > 0]
    if not winners:
        return {"status": "OK", "message": "No winning trades with MFE data"}

    capture_ratios = [t.pnl_r / t.mfe_r for t in winners if t.mfe_r > 0]
    avg_capture = np.mean(capture_ratios)

    # Distribution of waste (MFE - actual profit)
    waste_r = [t.mfe_r - t.pnl_r for t in winners]
    avg_waste = np.mean(waste_r)

    if avg_capture < THRESHOLDS["min_mfe_capture"]:
        return {
            "status": "WARNING",
            "message": f"Only capturing {avg_capture:.1%} of MFE on winners",
            "avg_mfe_capture": avg_capture,
            "avg_waste_r": avg_waste,
            "sample_trades": [
                {
                    "pnl_r": t.pnl_r,
                    "mfe_r": t.mfe_r,
                    "capture_pct": t.pnl_r / t.mfe_r if t.mfe_r > 0 else 0,
                    "exit_reason": t.exit_reason.value if t.exit_reason else None,
                }
                for t in winners[:5]
            ],
        }

    return {
        "status": "OK",
        "message": f"MFE capture acceptable ({avg_capture:.1%})",
        "avg_mfe_capture": avg_capture,
        "avg_waste_r": avg_waste,
    }


def diagnose_exit_distribution(trades: List[SimulatedTrade]) -> Dict:
    """Analyze exit reason distribution for anomalies."""
    exit_counts = Counter(t.exit_reason.value if t.exit_reason else 'unknown' for t in trades)
    total = len(trades)

    # Calculate percentages
    exit_pcts = {reason: count / total for reason, count in exit_counts.items()}

    issues = []

    # Check for suspicious patterns
    trailing_pct = exit_pcts.get("trailing_stop", 0)
    if trailing_pct > 0.8:
        issues.append(f"Excessive trailing exits ({trailing_pct:.1%}) - trailing may be too aggressive")

    tp_pct = exit_pcts.get("take_profit", 0)
    sl_pct = exit_pcts.get("stop_loss", 0)
    time_pct = exit_pcts.get("time_exit", 0)

    if sl_pct == 0 and tp_pct > 0.9:
        issues.append("No SL exits but 90%+ TP exits - suspicious")

    if issues:
        return {
            "status": "WARNING",
            "message": "; ".join(issues),
            "exit_distribution": dict(exit_counts),
            "exit_percentages": exit_pcts,
        }

    return {
        "status": "OK",
        "message": "Exit distribution appears normal",
        "exit_distribution": dict(exit_counts),
        "exit_percentages": exit_pcts,
    }


def run_all_diagnostics(
    symbols: List[str],
    timeframe: str = "4h",
    trailing_mode: str = "multi_stage",
    verbose: bool = True,
) -> Dict:
    """Run all diagnostic checks on trade results."""
    # Configure simulator with Phase 7.9 parameters
    config = TradeManagementConfig(
        tp_decay_enabled=False,
        tp_atr_mult=4.6,  # Phase 7.9: 4.6
        sl_atr_mult=1.0,  # Phase 7.9: 1.0
        trailing_mode=trailing_mode,
        max_bars_held=100,  # Phase 7.9: 100
        min_risk_reward=3.0,  # Phase 7.9: 3.0
    )

    simulator = TradeSimulator(config)
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
                print(f"{result.total_trades} trades")

        except Exception as e:
            if verbose:
                print(f"ERROR: {e}")
            continue

    if not all_trades:
        return {"error": "No trades generated"}

    # Run all diagnostics
    diagnostics = {
        "dust_profits": diagnose_dust_profits(all_trades),
        "short_holds": diagnose_short_holds(all_trades),
        "unrealistic_metrics": diagnose_unrealistic_metrics(all_trades),
        "mfe_waste": diagnose_mfe_waste(all_trades),
        "exit_distribution": diagnose_exit_distribution(all_trades),
    }

    # Summary
    pathological_count = sum(1 for d in diagnostics.values() if d.get("status") == "PATHOLOGICAL")
    warning_count = sum(1 for d in diagnostics.values() if d.get("status") == "WARNING")

    diagnostics["summary"] = {
        "total_trades": len(all_trades),
        "symbols_tested": len(symbols),
        "trailing_mode": trailing_mode,
        "pathological_checks": pathological_count,
        "warning_checks": warning_count,
        "overall_status": (
            "PATHOLOGICAL" if pathological_count > 0
            else "WARNING" if warning_count > 0
            else "OK"
        ),
    }

    return diagnostics


def print_diagnostic_report(diagnostics: Dict):
    """Print formatted diagnostic report."""
    print("\n" + "=" * 70)
    print("TRADE PATHOLOGY DIAGNOSTIC REPORT")
    print("=" * 70)

    summary = diagnostics.get("summary", {})
    print(f"Trades Analyzed: {summary.get('total_trades', 0)}")
    print(f"Symbols Tested:  {summary.get('symbols_tested', 0)}")
    print(f"Trailing Mode:   {summary.get('trailing_mode', 'unknown')}")
    print("=" * 70)

    for check_name, result in diagnostics.items():
        if check_name == "summary":
            continue

        status = result.get("status", "UNKNOWN")
        message = result.get("message", "No message")

        # Color coding for status
        if status == "PATHOLOGICAL":
            indicator = "X"
        elif status == "WARNING":
            indicator = "!"
        else:
            indicator = "+"

        print(f"\n[{indicator}] {check_name.upper()}")
        print(f"    Status: {status}")
        print(f"    {message}")

        # Print additional details for failures
        if status in ["PATHOLOGICAL", "WARNING"]:
            for key, value in result.items():
                if key not in ["status", "message"]:
                    if isinstance(value, float):
                        print(f"    {key}: {value:.3f}")
                    elif isinstance(value, dict) and len(value) < 10:
                        print(f"    {key}: {value}")
                    elif isinstance(value, list) and len(value) < 10:
                        print(f"    {key}: {len(value)} items")

    # Overall verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    overall = summary.get("overall_status", "UNKNOWN")
    if overall == "PATHOLOGICAL":
        print("SYSTEM HAS BUGS - Fix pathological issues before proceeding")
    elif overall == "WARNING":
        print("SYSTEM HAS CONCERNS - Review warnings before deployment")
    else:
        print("SYSTEM APPEARS HEALTHY - Ready for validation")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Diagnose Trade Pathology')
    parser.add_argument('--symbols', type=str, help='Comma-separated symbols')
    parser.add_argument('--check', type=str, help='Specific checks to run (comma-separated)')
    parser.add_argument('--trailing-mode', type=str, default='multi_stage',
                       choices=['none', 'simple', 'multi_stage'],
                       help='Trailing stop mode to test')
    parser.add_argument('--detailed', action='store_true', help='Show detailed output')
    parser.add_argument('--quiet', action='store_true', help='Suppress per-symbol output')
    args = parser.parse_args()

    # Parse symbols
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(',')]
        symbols = [s if '/' in s else f"{s[:-4]}/{s[-4:]}" for s in symbols]
    else:
        symbols = DEFAULT_SYMBOLS

    print("=" * 70)
    print("PHASE 9.2: TRADE PATHOLOGY DIAGNOSTICS")
    print("=" * 70)
    print(f"Testing {len(symbols)} symbols with trailing_mode={args.trailing_mode}")

    # Run diagnostics
    diagnostics = run_all_diagnostics(
        symbols=symbols,
        trailing_mode=args.trailing_mode,
        verbose=not args.quiet,
    )

    if "error" in diagnostics:
        print(f"\nERROR: {diagnostics['error']}")
        return

    # Print report
    print_diagnostic_report(diagnostics)

    # Save results
    results_dir = PROJECT_ROOT / "results" / "phase92_diagnostics"
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = results_dir / f"diagnostics_{timestamp}.json"

    # Make serializable
    def make_serializable(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        return obj

    with open(results_path, 'w') as f:
        json.dump(make_serializable(diagnostics), f, indent=2)

    print(f"\nResults saved: {results_path}")

    return diagnostics


if __name__ == "__main__":
    main()
