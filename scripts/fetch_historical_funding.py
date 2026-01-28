#!/usr/bin/env python3
"""
Phase 9.7: Fetch Historical Funding Rates
==========================================
Fetches 365 days of historical funding rate data from Bybit.

This is a prerequisite for running the funding filter validation.

Usage:
    python scripts/fetch_historical_funding.py
    python scripts/fetch_historical_funding.py --symbols BTC/USDT,ETH/USDT
    python scripts/fetch_historical_funding.py --days 365
    python scripts/fetch_historical_funding.py --all-symbols

Outputs:
    data/funding_rates/<symbol>_funding.parquet
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.funding_rates import FundingRateFetcher, FundingRateFetcherConfig


# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5


def fetch_with_retry(
    fetcher: FundingRateFetcher,
    symbol: str,
    days: int,
    max_retries: int = MAX_RETRIES,
) -> Tuple[bool, any, str]:
    """
    Fetch funding data with exponential backoff retry.

    Returns:
        Tuple of (success: bool, data: DataFrame or None, error_msg: str)
    """
    last_error = ""

    for attempt in range(max_retries):
        try:
            df = fetcher.get_historical_funding(symbol, days=days)
            if len(df) > 0:
                return True, df, ""
            else:
                return False, None, "No data returned"
        except Exception as e:
            last_error = str(e)

            # Check for rate limiting
            if "rate" in last_error.lower() or "429" in last_error:
                wait_time = RETRY_DELAY_SECONDS * (2 ** attempt)
                print(f"\n  Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
            elif attempt < max_retries - 1:
                wait_time = RETRY_DELAY_SECONDS * (attempt + 1)
                print(f"\n  Retry {attempt + 1}/{max_retries} in {wait_time}s...")
                time.sleep(wait_time)

    return False, None, last_error


# Default symbols to fetch (top traded on Bybit)
DEFAULT_SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "BNB/USDT",
    "XRP/USDT",
    "DOGE/USDT",
    "ADA/USDT",
    "AVAX/USDT",
    "DOT/USDT",
    "LINK/USDT",
]

# All symbols from QML trading system
ALL_SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "ADA/USDT", "DOGE/USDT", "LINK/USDT", "AVAX/USDT", "DOT/USDT",
    "MATIC/USDT", "UNI/USDT", "ATOM/USDT", "LTC/USDT", "ETC/USDT",
    "FIL/USDT", "NEAR/USDT", "APT/USDT", "ARB/USDT", "OP/USDT",
    "INJ/USDT", "SUI/USDT", "TIA/USDT", "SEI/USDT", "RUNE/USDT",
    "FTM/USDT", "SAND/USDT", "MANA/USDT", "AAVE/USDT", "CRV/USDT",
    "ALGO/USDT", "XLM/USDT",
]


def fetch_funding_rates(
    symbols: List[str],
    days: int = 365,
    verbose: bool = True,
) -> dict:
    """
    Fetch historical funding rates for multiple symbols.

    Args:
        symbols: List of trading pairs
        days: Number of days of history
        verbose: Print progress

    Returns:
        Dict of symbol -> DataFrame
    """
    config = FundingRateFetcherConfig(
        rate_limit_delay=0.3,  # Be nice to API
    )
    fetcher = FundingRateFetcher(config)

    results = {}
    errors = []

    print("=" * 70)
    print("FETCHING HISTORICAL FUNDING RATES")
    print("=" * 70)
    print(f"Symbols: {len(symbols)}")
    print(f"Days: {days}")
    print(f"Expected records per symbol: ~{days * 3} (3 funding periods/day)")
    print("=" * 70)
    print()

    for i, symbol in enumerate(symbols, 1):
        if verbose:
            print(f"[{i}/{len(symbols)}] Fetching {symbol}...", end=" ", flush=True)

        success, df, error_msg = fetch_with_retry(fetcher, symbol, days)

        if success and df is not None:
            results[symbol] = df
            if verbose:
                start = df['timestamp'].min().strftime("%Y-%m-%d")
                end = df['timestamp'].max().strftime("%Y-%m-%d")
                print(f"{len(df)} records ({start} to {end})")
        else:
            if verbose:
                print(f"FAILED: {error_msg}")
            errors.append((symbol, error_msg))

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    success_count = len(results)
    fail_count = len(errors)
    success_pct = (success_count / len(symbols) * 100) if symbols else 0

    print(f"Successfully fetched: {success_count}/{len(symbols)} ({success_pct:.0f}%)")

    total_records = sum(len(df) for df in results.values())
    print(f"Total records: {total_records:,}")

    if results:
        # Check data coverage
        dfs_with_data = [df for df in results.values() if len(df) > 0]
        if dfs_with_data:
            earliest = min(df['timestamp'].min() for df in dfs_with_data)
            latest = max(df['timestamp'].max() for df in dfs_with_data)
            print(f"Date range: {earliest.strftime('%Y-%m-%d')} to {latest.strftime('%Y-%m-%d')}")

            # Records per symbol stats
            records_per_symbol = [len(df) for df in dfs_with_data]
            avg_records = sum(records_per_symbol) / len(records_per_symbol)
            min_records = min(records_per_symbol)
            max_records = max(records_per_symbol)
            print(f"Records per symbol: avg={avg_records:.0f}, min={min_records}, max={max_records}")

    if errors:
        print()
        print(f"Failed ({fail_count} symbols):")
        # Categorize errors
        rate_limit_errors = [(s, e) for s, e in errors if "rate" in e.lower() or "429" in e]
        no_data_errors = [(s, e) for s, e in errors if "no data" in e.lower()]
        other_errors = [(s, e) for s, e in errors if (s, e) not in rate_limit_errors and (s, e) not in no_data_errors]

        if rate_limit_errors:
            print(f"  Rate limited: {[s for s, _ in rate_limit_errors]}")
        if no_data_errors:
            print(f"  No data: {[s for s, _ in no_data_errors]}")
        for symbol, error in other_errors:
            print(f"  - {symbol}: {error}")

    print()
    print(f"Data saved to: {config.data_dir}")
    print("=" * 70)

    return results


def verify_data():
    """Verify downloaded funding data."""
    config = FundingRateFetcherConfig()
    fetcher = FundingRateFetcher(config)

    available = fetcher.get_available_symbols()

    print()
    print("=" * 70)
    print("AVAILABLE FUNDING DATA")
    print("=" * 70)

    if not available:
        print("No funding data found. Run fetch first.")
        return

    print(f"{'Symbol':<15} {'Records':<10} {'Start':<15} {'End':<15}")
    print("-" * 55)

    for symbol in available:
        df = fetcher.load_funding_data(symbol)
        if df is not None and len(df) > 0:
            start = df['timestamp'].min().strftime("%Y-%m-%d")
            end = df['timestamp'].max().strftime("%Y-%m-%d")
            print(f"{symbol:<15} {len(df):<10} {start:<15} {end:<15}")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Fetch historical funding rates from Bybit")
    parser.add_argument('--symbols', type=str, help='Comma-separated symbols (e.g., BTC/USDT,ETH/USDT)')
    parser.add_argument('--days', type=int, default=365, help='Days of history to fetch (default: 365)')
    parser.add_argument('--all-symbols', action='store_true', help='Fetch all 32 symbols')
    parser.add_argument('--verify', action='store_true', help='Verify existing data without fetching')
    args = parser.parse_args()

    if args.verify:
        verify_data()
        return

    # Determine symbols
    if args.all_symbols:
        symbols = ALL_SYMBOLS
    elif args.symbols:
        symbols = [s.strip() for s in args.symbols.split(',')]
        # Normalize format
        symbols = [s if '/' in s else f"{s[:-4]}/{s[-4:]}" for s in symbols]
    else:
        symbols = DEFAULT_SYMBOLS

    # Fetch data
    results = fetch_funding_rates(symbols, days=args.days)

    # Verify
    print()
    verify_data()


if __name__ == "__main__":
    main()
