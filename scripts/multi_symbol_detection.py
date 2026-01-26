"""
Multi-Symbol Detection Pipeline
================================
Fetches data, runs Phase 7.5 detection, and generates aggregate results
for ML training across multiple symbols.

Usage:
    python scripts/multi_symbol_detection.py --symbols ETHUSDT,BNBUSDT,SOLUSDT
    python scripts/multi_symbol_detection.py --all  # Run all top liquidity pairs
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_engine import build_master_store, get_symbol_data_dir, normalize_symbol
from src.detection.historical_detector import HistoricalSwingDetector
from src.detection.pattern_validator import PatternValidator, PatternDirection
from src.detection.pattern_scorer import PatternScorer, PatternTier
from src.detection.backtest_adapter import BacktestAdapter
from src.detection.regime import MarketRegimeDetector
from src.detection.config import (
    SwingDetectionConfig,
    PatternValidationConfig,
    PatternScoringConfig,
)

# Top liquidity pairs to analyze (expanded list for more patterns)
DEFAULT_SYMBOLS = [
    "ETH/USDT",
    "BNB/USDT",
    "SOL/USDT",
    "XRP/USDT",
    "ADA/USDT",
    "DOGE/USDT",
    "AVAX/USDT",
    "DOT/USDT",
    "LINK/USDT",
    "MATIC/USDT",
    # Additional high-volume pairs
    "ATOM/USDT",
    "UNI/USDT",
    "LTC/USDT",
    "ETC/USDT",
    "FIL/USDT",
    "APT/USDT",
    "ARB/USDT",
    "OP/USDT",
    "NEAR/USDT",
    "INJ/USDT",
]

# Optimized detection config (from Phase 7.5 verification)
SWING_CONFIG = SwingDetectionConfig(
    atr_period=14,
    lookback=5,
    lookforward=3,
    min_zscore=0.5,  # Relaxed for more swings
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


def load_or_fetch_data(symbol: str, timeframe: str = "4h", years: int = 2) -> Optional[pd.DataFrame]:
    """Load existing data or fetch if not available."""
    normalized = normalize_symbol(symbol)
    data_dir = get_symbol_data_dir(symbol)
    data_path = data_dir / f"{timeframe}_master.parquet"

    # Check if data exists and is sufficient
    if data_path.exists():
        df = pd.read_parquet(data_path)
        # Need at least 1.5 years of data
        min_bars = int(365 * 1.5 * 6)  # 4h bars per year * 1.5
        if len(df) >= min_bars:
            print(f"  ✓ Using existing data: {len(df)} bars")
            return df
        else:
            print(f"  ⚠ Existing data insufficient ({len(df)} bars), fetching more...")

    # Fetch fresh data
    print(f"  📥 Fetching {years} years of {timeframe} data...")
    try:
        result = build_master_store(
            symbol=symbol,
            timeframes=[timeframe],
            years=years,
        )
        if result['success']:
            return pd.read_parquet(data_path)
        else:
            print(f"  ✗ Failed to fetch data")
            return None
    except Exception as e:
        print(f"  ✗ Error fetching data: {e}")
        return None


def run_detection(df: pd.DataFrame, symbol: str, timeframe: str) -> Tuple[List, List, List]:
    """Run full detection pipeline on a symbol."""
    normalized = normalize_symbol(symbol)

    # Normalize column names
    df.columns = [c.lower() for c in df.columns]
    if 'timestamp' in df.columns:
        df = df.rename(columns={'timestamp': 'time'})

    # 1. Detect swings
    detector = HistoricalSwingDetector(SWING_CONFIG, normalized, timeframe)
    swings = detector.detect(df)
    print(f"    Swings detected: {len(swings)}")

    # 2. Find patterns
    validator = PatternValidator(PATTERN_CONFIG)
    patterns = validator.find_patterns(swings, df['close'].values)
    valid_patterns = [p for p in patterns if p.is_valid]
    print(f"    Valid patterns: {len(valid_patterns)}")

    # 3. Score patterns (pass df and regime_result for Phase 7.6/7.8 metrics)
    scorer = PatternScorer(SCORING_CONFIG)
    regime_detector = MarketRegimeDetector()
    regime_result = regime_detector.get_regime(df)

    scored = []
    for p in valid_patterns:
        score_result = scorer.score(p, df=df, regime_result=regime_result)
        if score_result.tier != PatternTier.REJECT:
            scored.append((p, score_result))

    print(f"    Scored patterns (non-reject): {len(scored)}")

    # Tier breakdown
    tier_counts = {t.value: 0 for t in PatternTier}
    for _, sr in scored:
        tier_counts[sr.tier.value] += 1
    print(f"    Tiers: A={tier_counts.get('A', 0)}, B={tier_counts.get('B', 0)}, C={tier_counts.get('C', 0)}")

    return swings, valid_patterns, scored


def run_backtest(scored_patterns: List[Tuple], df: pd.DataFrame, symbol: str) -> Dict:
    """Run backtest on scored patterns."""
    if not scored_patterns:
        return {'total_trades': 0, 'win_rate': 0, 'profit_factor': 0}

    adapter = BacktestAdapter()
    validation_results = [vr for vr, sr in scored_patterns]
    scoring_results = [sr for vr, sr in scored_patterns]

    # Convert to signals
    signals = adapter.batch_convert_to_signals(
        validation_results=validation_results,
        scoring_results=scoring_results,
        symbol=symbol,
        min_tier=PatternTier.C,
    )

    if not signals:
        return {'total_trades': 0, 'win_rate': 0, 'profit_factor': 0}

    # Ensure df['time'] is tz-naive for comparison
    df = df.copy()
    if df['time'].dt.tz is not None:
        df['time'] = df['time'].dt.tz_localize(None)

    # Simple backtest simulation
    wins = 0
    losses = 0
    gross_profit = 0.0
    gross_loss = 0.0

    for signal in signals:
        # Convert signal timestamp to tz-naive for comparison
        sig_time = signal.timestamp
        if hasattr(sig_time, 'tzinfo') and sig_time.tzinfo is not None:
            sig_time = sig_time.replace(tzinfo=None)
        entry_idx = df[df['time'] >= sig_time].index
        if len(entry_idx) == 0:
            continue
        entry_idx = entry_idx[0]

        # Get entry bar and look forward
        entry_price = signal.price
        sl = signal.stop_loss
        tp = signal.take_profit

        # Simulate trade
        for i in range(entry_idx, min(entry_idx + 100, len(df))):
            high = df.iloc[i]['high']
            low = df.iloc[i]['low']

            if signal.signal_type.value == 'BUY':
                if low <= sl:
                    # Stop loss hit
                    losses += 1
                    gross_loss += abs(entry_price - sl)
                    break
                elif high >= tp:
                    # Take profit hit
                    wins += 1
                    gross_profit += abs(tp - entry_price)
                    break
            else:  # SELL
                if high >= sl:
                    # Stop loss hit
                    losses += 1
                    gross_loss += abs(sl - entry_price)
                    break
                elif low <= tp:
                    # Take profit hit
                    wins += 1
                    gross_profit += abs(entry_price - tp)
                    break

    total = wins + losses
    win_rate = wins / total if total > 0 else 0
    pf = gross_profit / gross_loss if gross_loss > 0 else 0

    return {
        'total_trades': total,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'profit_factor': pf,
        'gross_profit': gross_profit,
        'gross_loss': gross_loss,
    }


def create_ml_training_dataset(all_results: Dict) -> pd.DataFrame:
    """Create ML training dataset from all detection results."""
    records = []

    for symbol, data in all_results.items():
        if 'scored_patterns' not in data:
            continue

        for vr, sr in data['scored_patterns']:
            record = {
                'symbol': symbol,
                'timeframe': '4h',
                'detection_time': vr.p5.timestamp,
                'direction': vr.direction.value,
                'p1_bar': vr.p1.bar_index,
                'p2_bar': vr.p2.bar_index,
                'p3_bar': vr.p3.bar_index,
                'p4_bar': vr.p4.bar_index,
                'p5_bar': vr.p5.bar_index,
                'p1_price': vr.p1.price,
                'p2_price': vr.p2.price,
                'p3_price': vr.p3.price,
                'p4_price': vr.p4.price,
                'p5_price': vr.p5.price,
                'head_extension_atr': vr.head_extension_atr,
                'shoulder_diff_atr': vr.shoulder_diff_atr,
                'bos_efficiency': vr.bos_efficiency,
                'pattern_bars': vr.pattern_bars,
                'atr_p5': vr.atr_p5,
                'total_score': sr.total_score,
                'head_extension_score': sr.head_extension_score,
                'bos_efficiency_score': sr.bos_efficiency_score,
                'shoulder_symmetry_score': sr.shoulder_symmetry_score,
                'swing_significance_score': sr.swing_significance_score,
                'tier': sr.tier.value,
            }
            records.append(record)

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description='Multi-Symbol Detection Pipeline')
    parser.add_argument('--symbols', type=str, help='Comma-separated symbols (e.g., ETHUSDT,BNBUSDT)')
    parser.add_argument('--all', action='store_true', help='Run on all default symbols')
    parser.add_argument('--include-btc', action='store_true', help='Include BTCUSDT in analysis')
    parser.add_argument('--years', type=int, default=2, help='Years of data to fetch')
    parser.add_argument('--skip-fetch', action='store_true', help='Skip fetching, use existing data only')
    args = parser.parse_args()

    # Determine symbols to process
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(',')]
        # Normalize to exchange format
        symbols = [s if '/' in s else f"{s[:3]}/{s[3:]}" if len(s) > 4 else s for s in symbols]
    elif args.all:
        symbols = DEFAULT_SYMBOLS.copy()
    else:
        # Default: just a few symbols for testing
        symbols = ["ETH/USDT", "BNB/USDT", "SOL/USDT"]

    if args.include_btc:
        symbols.insert(0, "BTC/USDT")

    print("=" * 70)
    print("MULTI-SYMBOL QML DETECTION PIPELINE")
    print("=" * 70)
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Timeframe: 4h")
    print(f"Data period: {args.years} years")
    print("=" * 70)

    all_results = {}
    total_patterns = 0
    total_trades = 0

    for symbol in symbols:
        normalized = normalize_symbol(symbol)
        print(f"\n{'='*50}")
        print(f"Processing {symbol} ({normalized})")
        print("=" * 50)

        # 1. Load or fetch data
        if args.skip_fetch:
            data_path = get_symbol_data_dir(symbol) / "4h_master.parquet"
            if not data_path.exists():
                print(f"  ✗ No data found, skipping (use without --skip-fetch to download)")
                continue
            df = pd.read_parquet(data_path)
            print(f"  ✓ Loaded existing data: {len(df)} bars")
        else:
            df = load_or_fetch_data(symbol, "4h", args.years)

        if df is None or len(df) < 1000:
            print(f"  ✗ Insufficient data, skipping")
            continue

        # 2. Run detection
        print(f"  📊 Running detection...")
        swings, patterns, scored = run_detection(df, symbol, "4h")

        # 3. Run backtest
        print(f"  📈 Running backtest...")
        backtest_results = run_backtest(scored, df, normalized)

        # Store results
        all_results[normalized] = {
            'bars': len(df),
            'swings': len(swings),
            'patterns': len(patterns),
            'scored_patterns': scored,
            'backtest': backtest_results,
        }

        total_patterns += len(scored)
        total_trades += backtest_results['total_trades']

        print(f"  ✓ Patterns: {len(scored)}, Trades: {backtest_results['total_trades']}, WR: {backtest_results['win_rate']:.1%}")

    # Summary
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)
    print(f"Total Symbols: {len(all_results)}")
    print(f"Total Patterns: {total_patterns}")
    print(f"Total Trades: {total_trades}")

    # Save summary CSV
    summary_records = []
    for symbol, data in all_results.items():
        bt = data['backtest']
        summary_records.append({
            'symbol': symbol,
            'bars': data['bars'],
            'swings': data['swings'],
            'patterns': data['patterns'],
            'scored': len(data['scored_patterns']),
            'trades': bt['total_trades'],
            'wins': bt.get('wins', 0),
            'losses': bt.get('losses', 0),
            'win_rate': bt['win_rate'],
            'profit_factor': bt['profit_factor'],
        })

    summary_df = pd.DataFrame(summary_records)
    summary_path = PROJECT_ROOT / "results" / "altcoin_detection_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n📊 Summary saved: {summary_path}")

    # Print summary table
    print("\n" + summary_df.to_string(index=False))

    # Create ML training dataset
    if total_patterns > 0:
        ml_df = create_ml_training_dataset(all_results)
        ml_path = PROJECT_ROOT / "results" / "ml_training_patterns.parquet"
        ml_df.to_parquet(ml_path, index=False)
        print(f"\n🤖 ML training dataset saved: {ml_path}")
        print(f"   Patterns: {len(ml_df)}")
        print(f"   Features: {len(ml_df.columns)}")

        # Tier distribution
        tier_dist = ml_df['tier'].value_counts()
        print(f"\n   Tier Distribution:")
        for tier, count in tier_dist.items():
            print(f"     {tier}: {count}")

    # Combined backtest metrics
    if total_trades > 0:
        total_wins = sum(d['backtest'].get('wins', 0) for d in all_results.values())
        total_losses = sum(d['backtest'].get('losses', 0) for d in all_results.values())
        total_profit = sum(d['backtest'].get('gross_profit', 0) for d in all_results.values())
        total_loss = sum(d['backtest'].get('gross_loss', 0) for d in all_results.values())

        combined_wr = total_wins / total_trades if total_trades > 0 else 0
        combined_pf = total_profit / total_loss if total_loss > 0 else 0

        print(f"\n{'='*70}")
        print("COMBINED BACKTEST METRICS")
        print("=" * 70)
        print(f"Total Trades: {total_trades}")
        print(f"Wins: {total_wins}, Losses: {total_losses}")
        print(f"Combined Win Rate: {combined_wr:.1%}")
        print(f"Combined Profit Factor: {combined_pf:.2f}")

    print("\n✅ Multi-symbol detection complete!")
    return all_results


if __name__ == "__main__":
    main()
