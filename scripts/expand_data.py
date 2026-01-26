#!/usr/bin/env python3
"""
Data Expansion Script for Phase 7.6
====================================
Fetches historical data for 30 symbols across 3 timeframes.

Usage:
    python scripts/expand_data.py                    # Fetch all symbols
    python scripts/expand_data.py --tier 1           # Only Tier 1 (majors)
    python scripts/expand_data.py --dry-run          # Test with 30 days
    python scripts/expand_data.py --symbols BTC ETH  # Specific symbols
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional
import time

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_engine import build_master_store, normalize_symbol, get_symbol_data_dir

# Try loguru, fallback to standard logging
try:
    from loguru import logger
except ImportError:
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s'
    )
    logger = logging.getLogger(__name__)


# =============================================================================
# SYMBOL LISTS BY TIER
# =============================================================================

SYMBOLS_TIER1 = [
    # Majors - highest liquidity
    'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT',
]

SYMBOLS_TIER2 = [
    # Large caps - good liquidity
    'ADA/USDT', 'AVAX/USDT', 'DOGE/USDT', 'DOT/USDT', 'MATIC/USDT',
    'LINK/USDT', 'LTC/USDT', 'ATOM/USDT', 'UNI/USDT', 'XLM/USDT',
]

SYMBOLS_TIER3 = [
    # Mid caps - reasonable liquidity
    'APT/USDT', 'ARB/USDT', 'OP/USDT', 'INJ/USDT', 'SUI/USDT',
    'NEAR/USDT', 'FTM/USDT', 'AAVE/USDT', 'MKR/USDT', 'RUNE/USDT',
    'TIA/USDT', 'SEI/USDT', 'JUP/USDT', 'WIF/USDT', 'PEPE/USDT',
]

ALL_SYMBOLS = SYMBOLS_TIER1 + SYMBOLS_TIER2 + SYMBOLS_TIER3

TIMEFRAMES = ['1h', '4h', '1d']


# =============================================================================
# DATA EXPANSION
# =============================================================================

def expand_data(
    symbols: Optional[List[str]] = None,
    timeframes: Optional[List[str]] = None,
    years: float = 3.0,
    dry_run: bool = False,
    skip_existing: bool = True,
) -> dict:
    """
    Expand the data store with multiple symbols.

    Args:
        symbols: List of symbols to fetch (default: ALL_SYMBOLS)
        timeframes: List of timeframes (default: TIMEFRAMES)
        years: Years of historical data
        dry_run: If True, only fetch 30 days
        skip_existing: If True, skip symbols that already have data

    Returns:
        Summary dictionary with results
    """
    symbols = symbols or ALL_SYMBOLS
    timeframes = timeframes or TIMEFRAMES

    logger.info("="*70)
    logger.info(f"DATA EXPANSION - Phase 7.6")
    logger.info(f"Symbols: {len(symbols)}")
    logger.info(f"Timeframes: {timeframes}")
    logger.info(f"Years: {years if not dry_run else '~30 days (dry run)'}")
    logger.info("="*70)

    results = {
        'total': len(symbols),
        'success': [],
        'skipped': [],
        'failed': [],
    }

    for i, symbol in enumerate(symbols, 1):
        normalized = normalize_symbol(symbol)
        data_dir = get_symbol_data_dir(symbol)

        # Check if already exists
        if skip_existing:
            existing_files = list(data_dir.glob("*_master.parquet"))
            existing_tfs = [f.stem.replace("_master", "") for f in existing_files]
            missing_tfs = [tf for tf in timeframes if tf not in existing_tfs]

            if not missing_tfs:
                logger.info(f"[{i}/{len(symbols)}] {normalized}: Already exists, skipping")
                results['skipped'].append(normalized)
                continue
            else:
                timeframes_to_fetch = missing_tfs
                logger.info(f"[{i}/{len(symbols)}] {normalized}: Fetching missing timeframes {missing_tfs}")
        else:
            timeframes_to_fetch = timeframes

        logger.info(f"\n[{i}/{len(symbols)}] Fetching {normalized}...")

        try:
            result = build_master_store(
                symbol=symbol,
                timeframes=timeframes_to_fetch,
                years=years,
                dry_run=dry_run,
            )

            if result['success']:
                results['success'].append(normalized)
                for tf, stats in result['stats'].items():
                    if 'rows' in stats:
                        logger.info(f"  {tf}: {stats['rows']} rows")
            else:
                results['failed'].append(normalized)
                logger.error(f"  Failed with partial errors")

        except Exception as e:
            logger.error(f"  FAILED: {e}")
            results['failed'].append(normalized)

        # Rate limiting - be nice to the exchange
        if i < len(symbols):
            time.sleep(2)

    # Summary
    logger.info("\n" + "="*70)
    logger.info("EXPANSION COMPLETE")
    logger.info("="*70)
    logger.info(f"Success: {len(results['success'])}/{results['total']}")
    logger.info(f"Skipped: {len(results['skipped'])}")
    logger.info(f"Failed:  {len(results['failed'])}")

    if results['failed']:
        logger.warning(f"Failed symbols: {results['failed']}")

    return results


def check_data_status() -> dict:
    """
    Check the status of all symbols in the data store.

    Returns:
        Dictionary with status for each symbol
    """
    status = {}

    for symbol in ALL_SYMBOLS:
        normalized = normalize_symbol(symbol)
        data_dir = get_symbol_data_dir(symbol)

        if not data_dir.exists():
            status[normalized] = {'exists': False, 'timeframes': []}
        else:
            existing_files = list(data_dir.glob("*_master.parquet"))
            existing_tfs = [f.stem.replace("_master", "") for f in existing_files]

            status[normalized] = {
                'exists': True,
                'timeframes': existing_tfs,
                'path': str(data_dir),
            }

    return status


def print_data_status():
    """Print a formatted status report of all data."""
    status = check_data_status()

    print("\n" + "="*70)
    print("DATA STORE STATUS")
    print("="*70)
    print(f"{'Symbol':<12} {'1h':<8} {'4h':<8} {'1d':<8}")
    print("-"*70)

    complete = 0
    partial = 0
    missing = 0

    for symbol, info in status.items():
        if not info['exists']:
            print(f"{symbol:<12} {'--':<8} {'--':<8} {'--':<8}")
            missing += 1
        else:
            tfs = info['timeframes']
            h1 = '' if '1h' in tfs else '--'
            h4 = '' if '4h' in tfs else '--'
            d1 = '' if '1d' in tfs else '--'
            print(f"{symbol:<12} {h1:<8} {h4:<8} {d1:<8}")

            if len(tfs) == 3:
                complete += 1
            else:
                partial += 1

    print("-"*70)
    print(f"Complete: {complete} | Partial: {partial} | Missing: {missing}")
    print("="*70)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Expand data store with multiple symbols for Phase 7.6"
    )

    parser.add_argument(
        '--tier',
        type=int,
        choices=[1, 2, 3],
        help='Fetch only specific tier (1=majors, 2=large caps, 3=mid caps)'
    )
    parser.add_argument(
        '--symbols',
        nargs='+',
        help='Fetch specific symbols (e.g., BTC ETH SOL)'
    )
    parser.add_argument(
        '--timeframes',
        nargs='+',
        default=TIMEFRAMES,
        help='Timeframes to fetch (default: 1h 4h 1d)'
    )
    parser.add_argument(
        '--years',
        type=float,
        default=3.0,
        help='Years of historical data (default: 3)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Fetch only 30 days of data for testing'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Re-fetch even if data already exists'
    )
    parser.add_argument(
        '--status',
        action='store_true',
        help='Show data store status and exit'
    )

    args = parser.parse_args()

    # Status check
    if args.status:
        print_data_status()
        return

    # Determine symbols to fetch
    if args.symbols:
        symbols = [f"{s}/USDT" if '/' not in s else s for s in args.symbols]
    elif args.tier == 1:
        symbols = SYMBOLS_TIER1
    elif args.tier == 2:
        symbols = SYMBOLS_TIER2
    elif args.tier == 3:
        symbols = SYMBOLS_TIER3
    else:
        symbols = ALL_SYMBOLS

    # Run expansion
    results = expand_data(
        symbols=symbols,
        timeframes=args.timeframes,
        years=args.years,
        dry_run=args.dry_run,
        skip_existing=not args.force,
    )

    # Exit code based on results
    if results['failed']:
        sys.exit(1)


if __name__ == "__main__":
    main()
