"""
Master Data Store Engine
========================
Automated data pipeline for fetching, cleaning, and storing BTC OHLCV data
with pre-calculated technical indicators.

Single Source of Truth for all visualizations and backtesting.

Version: 1.0.0
"""

import os
import sys
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd

# Try loguru, fallback to standard logging
try:
    from loguru import logger
except ImportError:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(message)s')
    logger = logging.getLogger(__name__)

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# CONFIGURATION
# =============================================================================

# Default settings
DEFAULT_SYMBOL = "BTC/USDT"
DEFAULT_TIMEFRAMES = ["1h", "4h"]
DEFAULT_YEARS = 5
DEFAULT_ATR_PERIOD = 14

# Output directory
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "BTC"


# =============================================================================
# ATR CALCULATION
# =============================================================================

def calculate_atr(df: pd.DataFrame, period: int = DEFAULT_ATR_PERIOD) -> pd.Series:
    """
    Calculate Average True Range (ATR) using Wilder's smoothing.
    
    Args:
        df: OHLCV DataFrame with High, Low, Close columns
        period: ATR lookback period
        
    Returns:
        Series with ATR values
    """
    # Normalize column names (handle both lowercase and capitalized)
    high = df.get('High', df.get('high'))
    low = df.get('Low', df.get('low'))
    close = df.get('Close', df.get('close'))
    
    if high is None or low is None or close is None:
        raise ValueError("DataFrame must contain High, Low, Close columns")
    
    # Calculate True Range components
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    # True Range is the maximum of the three
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # ATR using Wilder's exponential moving average
    atr = true_range.ewm(span=period, adjust=False).mean()
    
    return atr


# =============================================================================
# DATA CLEANING
# =============================================================================

def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean OHLCV data by handling missing values and gaps.
    
    Cleaning steps:
    1. Remove duplicate timestamps
    2. Sort by time
    3. Forward-fill missing OHLCV values (for small gaps)
    4. Drop rows with invalid data
    
    Args:
        df: Raw OHLCV DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    df = df.copy()
    
    # Ensure time column exists and is datetime
    time_col = 'time' if 'time' in df.columns else df.index.name
    if time_col == 'time':
        df['time'] = pd.to_datetime(df['time'])
    
    # Remove duplicates
    before_count = len(df)
    df = df.drop_duplicates(subset=['time'] if 'time' in df.columns else None)
    if len(df) < before_count:
        logger.info(f"Removed {before_count - len(df)} duplicate rows")
    
    # Sort by time
    if 'time' in df.columns:
        df = df.sort_values('time').reset_index(drop=True)
    else:
        df = df.sort_index()
    
    # Identify OHLCV columns (handle both cases)
    ohlcv_cols = []
    for base in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if base in df.columns:
            ohlcv_cols.append(base)
        elif base.lower() in df.columns:
            ohlcv_cols.append(base.lower())
    
    # Forward-fill missing values (for small gaps)
    if ohlcv_cols:
        missing_before = df[ohlcv_cols].isna().sum().sum()
        df[ohlcv_cols] = df[ohlcv_cols].ffill()
        missing_after = df[ohlcv_cols].isna().sum().sum()
        if missing_before > missing_after:
            logger.info(f"Forward-filled {missing_before - missing_after} missing values")
    
    # Drop any remaining rows with NaN in critical columns
    critical_cols = [c for c in ohlcv_cols if 'volume' not in c.lower()]
    if critical_cols:
        before_count = len(df)
        df = df.dropna(subset=critical_cols)
        if len(df) < before_count:
            logger.warning(f"Dropped {before_count - len(df)} rows with missing data")
    
    # Validate price data (High >= Low, Open and Close within range)
    high_col = 'High' if 'High' in df.columns else 'high'
    low_col = 'Low' if 'Low' in df.columns else 'low'
    
    if high_col in df.columns and low_col in df.columns:
        invalid_mask = df[high_col] < df[low_col]
        if invalid_mask.any():
            logger.warning(f"Found {invalid_mask.sum()} rows where High < Low")
            # Swap High and Low for invalid rows
            df.loc[invalid_mask, [high_col, low_col]] = df.loc[invalid_mask, [low_col, high_col]].values
    
    return df


# =============================================================================
# DATA FETCHING
# =============================================================================

def fetch_btc_ohlcv(
    timeframe: str,
    years: int = DEFAULT_YEARS,
    symbol: str = DEFAULT_SYMBOL,
    exchange_id: str = "binance"
) -> pd.DataFrame:
    """
    Fetch BTC OHLCV data from exchange.
    
    Args:
        timeframe: Candle timeframe ('1h' or '4h')
        years: Number of years of historical data
        symbol: Trading pair symbol
        exchange_id: CCXT exchange ID
        
    Returns:
        DataFrame with OHLCV data
    """
    try:
        import ccxt
    except ImportError:
        logger.error("ccxt not installed. Run: pip install ccxt")
        raise
    
    logger.info(f"Fetching {years} years of {symbol} {timeframe} data from {exchange_id}")
    
    # Calculate date range
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=years * 365)
    
    # Initialize exchange
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'}
    })
    
    # Load markets
    exchange.load_markets()
    
    if symbol not in exchange.markets:
        raise ValueError(f"Symbol {symbol} not found on {exchange_id}")
    
    # Timeframe to milliseconds
    tf_ms = {
        '1h': 60 * 60 * 1000,
        '4h': 4 * 60 * 60 * 1000,
    }
    
    all_candles = []
    since_ms = int(start_date.timestamp() * 1000)
    end_ms = int(end_date.timestamp() * 1000)
    
    logger.info(f"Fetching from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Fetch in batches
    batch_count = 0
    while since_ms < end_ms:
        try:
            candles = exchange.fetch_ohlcv(
                symbol,
                timeframe,
                since=since_ms,
                limit=1000
            )
            
            if not candles:
                break
            
            all_candles.extend(candles)
            batch_count += 1
            
            # Progress logging every 10 batches
            if batch_count % 10 == 0:
                current_date = datetime.fromtimestamp(candles[-1][0] / 1000, tz=timezone.utc)
                logger.info(f"  Progress: {len(all_candles)} candles fetched (up to {current_date.strftime('%Y-%m-%d')})")
            
            # Move to next batch
            last_ts = candles[-1][0]
            since_ms = last_ts + tf_ms.get(timeframe, 3600000)
            
            if len(candles) < 1000:
                break
                
        except Exception as e:
            logger.warning(f"Fetch error: {e}. Retrying...")
            import time
            time.sleep(1)
            continue
    
    if not all_candles:
        raise ValueError(f"No data retrieved for {symbol} {timeframe}")
    
    # Convert to DataFrame
    df = pd.DataFrame(
        all_candles,
        columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
    )
    
    # Convert timestamp to datetime
    df['time'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df = df.drop(columns=['timestamp'])
    
    # Remove duplicates and sort
    df = df.drop_duplicates(subset=['time']).sort_values('time').reset_index(drop=True)
    
    logger.info(f"✅ Fetched {len(df)} candles for {symbol} {timeframe}")
    logger.info(f"   Date range: {df['time'].min()} to {df['time'].max()}")
    
    return df


# =============================================================================
# MASTER STORE BUILDER
# =============================================================================

def build_master_store(
    timeframes: Optional[List[str]] = None,
    years: int = DEFAULT_YEARS,
    atr_period: int = DEFAULT_ATR_PERIOD,
    output_dir: Optional[Path] = None,
    dry_run: bool = False
) -> dict:
    """
    Build the Master Data Store for BTC backtesting.
    
    Fetches OHLCV data, cleans it, calculates ATR, and saves to Parquet files.
    
    Args:
        timeframes: List of timeframes to fetch (default: ['1h', '4h'])
        years: Years of historical data
        atr_period: ATR calculation period
        output_dir: Output directory (default: data/processed/BTC/)
        dry_run: If True, only fetch 30 days of data for testing
        
    Returns:
        Dictionary with status and file paths
    """
    timeframes = timeframes or DEFAULT_TIMEFRAMES
    output_dir = output_dir or DATA_DIR
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'success': True,
        'files': {},
        'stats': {}
    }
    
    # Adjust years for dry run
    if dry_run:
        years = 0.1  # ~36 days
        logger.info("🧪 DRY RUN MODE: Fetching only 30 days of data")
    
    for tf in timeframes:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {tf} timeframe")
        logger.info(f"{'='*60}")
        
        try:
            # Fetch data
            df = fetch_btc_ohlcv(timeframe=tf, years=years)
            
            # Clean data
            df = clean_ohlcv(df)
            
            # Calculate ATR
            df['ATR'] = calculate_atr(df, period=atr_period)
            
            # Drop initial NaN ATR values
            df = df.dropna(subset=['ATR'])
            
            # Reorder columns for consistency
            column_order = ['time', 'Open', 'High', 'Low', 'Close', 'Volume', 'ATR']
            df = df[column_order]
            
            # Save to Parquet
            output_path = output_dir / f"{tf}_master.parquet"
            df.to_parquet(output_path, index=False)
            
            results['files'][tf] = str(output_path)
            results['stats'][tf] = {
                'rows': len(df),
                'date_range': (
                    df['time'].min().strftime('%Y-%m-%d'),
                    df['time'].max().strftime('%Y-%m-%d')
                ),
                'atr_range': (df['ATR'].min(), df['ATR'].max())
            }
            
            logger.info(f"✅ Saved {tf} master data: {len(df)} rows")
            logger.info(f"   File: {output_path}")
            logger.info(f"   ATR range: {df['ATR'].min():.2f} - {df['ATR'].max():.2f}")
            
        except Exception as e:
            logger.error(f"❌ Failed to build {tf} master data: {e}")
            results['success'] = False
            results['files'][tf] = None
            results['stats'][tf] = {'error': str(e)}
    
    return results


def load_master_data(
    timeframe: str,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    data_dir: Optional[Path] = None
) -> pd.DataFrame:
    """
    Load data from the Master Data Store.
    
    Args:
        timeframe: Timeframe to load ('1h' or '4h')
        start_time: Optional start filter
        end_time: Optional end filter
        data_dir: Data directory (default: data/processed/BTC/)
        
    Returns:
        DataFrame with OHLCV + ATR data
    """
    data_dir = data_dir or DATA_DIR
    file_path = data_dir / f"{timeframe}_master.parquet"
    
    if not file_path.exists():
        raise FileNotFoundError(
            f"Master data not found: {file_path}\n"
            f"Run: python -m src.data_engine to build the Master Data Store"
        )
    
    df = pd.read_parquet(file_path)
    
    # Ensure time column is datetime
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
    
    # Apply filters
    if start_time:
        start_time = pd.Timestamp(start_time)
        df = df[df['time'] >= start_time]
    
    if end_time:
        end_time = pd.Timestamp(end_time)
        df = df[df['time'] <= end_time]
    
    return df


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    """Command-line interface for building the Master Data Store."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Build the Master Data Store for BTC backtesting"
    )
    parser.add_argument(
        '--timeframes', '-t',
        nargs='+',
        default=['1h', '4h'],
        help='Timeframes to fetch (default: 1h 4h)'
    )
    parser.add_argument(
        '--years', '-y',
        type=float,
        default=5,
        help='Years of historical data (default: 5)'
    )
    parser.add_argument(
        '--atr-period', '-a',
        type=int,
        default=14,
        help='ATR calculation period (default: 14)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Fetch only 30 days of data for testing'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        default=None,
        help='Output directory (default: data/processed/BTC/)'
    )
    
    args = parser.parse_args()
    
    # Configure logging (handle both loguru and standard logging)
    try:
        # Loguru-specific configuration
        logger.remove()
        logger.add(
            sys.stderr,
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
            level="INFO"
        )
    except AttributeError:
        # Standard logging is already configured at module level
        pass
    
    logger.info("="*60)
    logger.info("🏗️  MASTER DATA STORE BUILDER")
    logger.info("="*60)
    
    results = build_master_store(
        timeframes=args.timeframes,
        years=args.years,
        atr_period=args.atr_period,
        output_dir=args.output_dir,
        dry_run=args.dry_run
    )
    
    logger.info("\n" + "="*60)
    if results['success']:
        logger.info("✅ Master Data Store build complete!")
        for tf, path in results['files'].items():
            if path:
                stats = results['stats'][tf]
                logger.info(f"   {tf}: {stats['rows']} rows ({stats['date_range'][0]} to {stats['date_range'][1]})")
    else:
        logger.error("❌ Build completed with errors. Check logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
