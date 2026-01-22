#!/usr/bin/env python3
"""
Real Data Validation Script for QML Pattern Detection
======================================================
Tests detection quality on real market data with multiple configurations.

Usage:
    python scripts/validate_detection.py --symbol BTC/USDT --timeframe 4h --days 180

Requirements:
    pip install ccxt
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from loguru import logger


def setup_logging(verbose: bool = False):
    """Configure logging."""
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(sys.stderr, level=level, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")


def load_data(symbol: str, timeframe: str, days: int) -> Optional[pd.DataFrame]:
    """Load OHLCV data from best available source."""
    from src.data.loader import load_ohlcv

    df, source = load_ohlcv(symbol, timeframe, days, source="auto")

    if df is None or len(df) == 0:
        logger.error(f"Failed to load data for {symbol} {timeframe}")
        return None

    logger.info(f"Loaded {len(df)} candles from {source}")
    return df


def check_market_regime(df: pd.DataFrame) -> Dict:
    """Analyze current market regime."""
    from src.detection import MarketRegimeDetector

    detector = MarketRegimeDetector()
    result = detector.get_regime(df)

    regime_info = {
        'regime': result.regime.value,
        'adx': result.adx,
        'rsi': result.rsi,
        'volatility': result.volatility_percentile,
        'qml_favorable': result.regime.value == 'RANGING'
    }

    return regime_info


def get_detection_configs() -> Dict[str, Dict]:
    """Return preset detection configurations."""
    return {
        'Conservative': {
            'swing_lookback': 7,
            'min_head_extension_atr': 0.8,
            'bos_requirement': 2,
            'require_trend_alignment': True,
            'description': 'High quality, fewer signals'
        },
        'Balanced': {
            'swing_lookback': 5,
            'min_head_extension_atr': 0.5,
            'bos_requirement': 1,
            'require_trend_alignment': True,
            'description': 'Default settings'
        },
        'Aggressive': {
            'swing_lookback': 3,
            'min_head_extension_atr': 0.3,
            'bos_requirement': 1,
            'require_trend_alignment': False,
            'description': 'More signals, lower quality filter'
        }
    }


def run_detection(df: pd.DataFrame, config_name: str, config: Dict) -> List:
    """Run detection with specified configuration."""
    from src.detection import QMLPatternDetector, QMLConfig, SwingAlgorithm

    qml_config = QMLConfig(
        swing_algorithm=SwingAlgorithm.ROLLING,
        swing_lookback=config['swing_lookback'],
        min_head_extension_atr=config['min_head_extension_atr'],
        bos_requirement=config['bos_requirement'],
        require_trend_alignment=config['require_trend_alignment']
    )

    detector = QMLPatternDetector(qml_config)
    patterns = detector.detect(df)

    logger.info(f"  {config_name}: {len(patterns)} patterns found")
    return patterns


def format_pattern_summary(pattern, idx: int) -> str:
    """Format a single pattern for display."""
    direction = "LONG" if "bearish" in pattern.id.lower() else "SHORT"
    direction_label = f"{'🔴' if direction == 'SHORT' else '🟢'} {direction}"

    # Calculate R:R
    risk = abs(pattern.stop_loss - pattern.entry_price)
    reward = abs(pattern.take_profit_1 - pattern.entry_price)
    rr = reward / risk if risk > 0 else 0

    lines = [
        f"\n  Pattern #{idx + 1}: {pattern.id[:20]}...",
        f"    Direction: {direction_label}",
        f"    Entry: ${pattern.entry_price:,.2f}",
        f"    Stop Loss: ${pattern.stop_loss:,.2f}",
        f"    Take Profit: ${pattern.take_profit_1:,.2f}",
        f"    R:R: 1:{rr:.1f}",
        f"    Strength: {pattern.pattern_strength:.2f}",
        f"    Head Extension: {pattern.head_extension_atr:.2f} ATR",
        f"    Detection: {pattern.detection_time}"
    ]
    return "\n".join(lines)


def generate_assessment(regime_info: Dict, results: Dict[str, List]) -> str:
    """Generate overall assessment."""
    lines = ["\n" + "=" * 60, "ASSESSMENT", "=" * 60]

    # Regime assessment
    if regime_info['qml_favorable']:
        lines.append("✅ Market regime (RANGING) is favorable for QML patterns")
    else:
        lines.append(f"⚠️  Market regime ({regime_info['regime']}) may not be ideal for QML")

    # Detection quality
    balanced_count = len(results.get('Balanced', []))
    conservative_count = len(results.get('Conservative', []))

    if balanced_count == 0:
        lines.append("❌ No patterns detected - detection parameters may need tuning")
        lines.append("   Consider: Different timeframe, more historical data, or relaxed settings")
    elif conservative_count > 0:
        lines.append(f"✅ Found {conservative_count} high-quality patterns (Conservative)")
        lines.append("   These patterns have stronger confluence signals")
    else:
        lines.append(f"ℹ️  Found {balanced_count} patterns with Balanced settings")
        lines.append("   No patterns passed Conservative filter - may need more data")

    # Recommendations
    lines.append("\n📋 RECOMMENDATIONS:")
    if balanced_count > 10:
        lines.append("   - Plenty of patterns found, can use stricter filtering")
    elif balanced_count > 0:
        lines.append("   - Good signal density, current settings appropriate")
    else:
        lines.append("   - Try longer time period (--days 365)")
        lines.append("   - Or try different timeframe (1h vs 4h vs 1d)")

    return "\n".join(lines)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate QML pattern detection on real market data"
    )
    parser.add_argument(
        "--symbol", "-s",
        default="BTC/USDT",
        help="Trading pair (default: BTC/USDT)"
    )
    parser.add_argument(
        "--timeframe", "-t",
        default="4h",
        choices=["1h", "4h", "1d"],
        help="Candle timeframe (default: 4h)"
    )
    parser.add_argument(
        "--days", "-d",
        type=int,
        default=180,
        help="Days of historical data (default: 180)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--top", "-n",
        type=int,
        default=5,
        help="Number of top patterns to show (default: 5)"
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    print("\n" + "=" * 60)
    print("QML PATTERN DETECTION - REAL DATA VALIDATION")
    print("=" * 60)
    print(f"Symbol: {args.symbol}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Period: {args.days} days")
    print("=" * 60)

    # Step 1: Load data
    print("\n📥 LOADING DATA...")
    df = load_data(args.symbol, args.timeframe, args.days)
    if df is None:
        print("\n❌ Failed to load data. Check your internet connection or try later.")
        return 1

    # Data summary
    print(f"\n📊 DATA SUMMARY:")
    print(f"   Candles: {len(df)}")
    if 'time' in df.columns:
        print(f"   Range: {df['time'].iloc[0]} to {df['time'].iloc[-1]}")
    if 'close' in df.columns:
        latest = df['close'].iloc[-1]
        high = df['high'].max()
        low = df['low'].min()
        print(f"   Price range: ${low:,.2f} - ${high:,.2f}")
        print(f"   Latest: ${latest:,.2f}")

    # Step 2: Check market regime
    print("\n🌡️  MARKET REGIME:")
    regime_info = check_market_regime(df)
    print(f"   Regime: {regime_info['regime']}")
    print(f"   ADX: {regime_info['adx']:.1f}")
    print(f"   RSI: {regime_info['rsi']:.1f}")
    print(f"   Volatility Percentile: {regime_info['volatility']:.0%}")
    if regime_info['qml_favorable']:
        print("   ✅ QML-favorable conditions")
    else:
        print("   ⚠️  Not ideal for QML")

    # Step 3: Run detection with all configs
    print("\n🔍 RUNNING DETECTION...")
    configs = get_detection_configs()
    results = {}

    for config_name, config in configs.items():
        patterns = run_detection(df, config_name, config)
        results[config_name] = patterns

    # Step 4: Show top patterns
    print(f"\n📋 TOP {args.top} PATTERNS (Balanced Config):")
    balanced_patterns = results.get('Balanced', [])

    if balanced_patterns:
        # Sort by strength
        sorted_patterns = sorted(balanced_patterns, key=lambda p: p.pattern_strength, reverse=True)
        for i, pattern in enumerate(sorted_patterns[:args.top]):
            print(format_pattern_summary(pattern, i))
    else:
        print("   No patterns found with Balanced configuration")

    # Step 5: Generate assessment
    print(generate_assessment(regime_info, results))

    print("\n" + "=" * 60)
    print("Validation complete.")
    print("=" * 60 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
