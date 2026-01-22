"""
Data Loading Utility for QML Trading System
=============================================
Unified data loading interface supporting:
- Local parquet/CSV files
- CCXT exchange API (Binance)
- Automatic caching for fetched data
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, List, Tuple
import pandas as pd
from loguru import logger


# =============================================================================
# CONSTANTS
# =============================================================================

# Common trading pairs
AVAILABLE_SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "ADA/USDT", "DOGE/USDT", "AVAX/USDT", "DOT/USDT", "LINK/USDT"
]

# Supported timeframes
AVAILABLE_TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]

# Data directories to search (in order of priority)
DATA_DIRS = [
    Path("data/processed"),
    Path("data/raw"),
    Path("data"),
]

# Cache directory for fetched data
CACHE_DIR = Path("data/cache")


# =============================================================================
# PUBLIC API
# =============================================================================

def get_available_symbols() -> List[str]:
    """Return list of common trading pairs."""
    return AVAILABLE_SYMBOLS.copy()


def get_available_timeframes() -> List[str]:
    """Return list of supported timeframes."""
    return AVAILABLE_TIMEFRAMES.copy()


def load_ohlcv(
    symbol: str,
    timeframe: str,
    days: int = 90,
    source: str = "auto"
) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Load OHLCV data from best available source.

    Args:
        symbol: Trading pair (e.g., "BTC/USDT" or "BTCUSDT")
        timeframe: Candle timeframe (e.g., "1h", "4h", "1d")
        days: Number of days of historical data
        source: Data source - "auto", "database", "exchange", "cache"

    Returns:
        Tuple of (DataFrame or None, source_description)
        DataFrame has columns: time, open, high, low, close, volume
    """
    # Normalize symbol format
    symbol_clean = symbol.replace("/", "")  # BTC/USDT -> BTCUSDT
    symbol_slash = symbol if "/" in symbol else f"{symbol[:3]}/{symbol[3:]}" if len(symbol) >= 6 else symbol

    logger.info(f"Loading {symbol_clean} {timeframe} ({days} days), source={source}")

    # Try sources in order based on preference
    if source == "auto":
        # 1. Try local database/files first
        df, src = _load_from_database(symbol_clean, timeframe, days)
        if df is not None and len(df) > 0:
            return df, src

        # 2. Try cache
        df, src = _load_from_cache(symbol_clean, timeframe, days)
        if df is not None and len(df) > 0:
            return df, src

        # 3. Try exchange API
        df, src = _fetch_from_exchange(symbol_slash, timeframe, days)
        if df is not None and len(df) > 0:
            # Cache the fetched data
            _cache_data(df, symbol_clean, timeframe)
            return df, src

        return None, "No data found"

    elif source == "database":
        return _load_from_database(symbol_clean, timeframe, days)

    elif source == "exchange":
        df, src = _fetch_from_exchange(symbol_slash, timeframe, days)
        if df is not None:
            _cache_data(df, symbol_clean, timeframe)
        return df, src

    elif source == "cache":
        return _load_from_cache(symbol_clean, timeframe, days)

    else:
        logger.error(f"Unknown source: {source}")
        return None, f"Unknown source: {source}"


# =============================================================================
# INTERNAL FUNCTIONS
# =============================================================================

def _load_from_database(symbol: str, timeframe: str, days: int) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Load data from local parquet/CSV files.

    Searches data directories for matching files.
    """
    # Patterns to search for
    patterns = [
        f"{symbol}/{timeframe}*.parquet",
        f"{symbol}_{timeframe}*.parquet",
        f"*{symbol}*{timeframe}*.parquet",
        f"{symbol}/{timeframe}*.csv",
        f"{symbol}_{timeframe}*.csv",
        f"*{symbol}*{timeframe}*.csv",
    ]

    for data_dir in DATA_DIRS:
        if not data_dir.exists():
            continue

        for pattern in patterns:
            files = list(data_dir.glob(pattern))
            if files:
                # Use most recently modified file
                file_path = max(files, key=lambda f: f.stat().st_mtime)
                try:
                    df = _read_file(file_path)
                    if df is not None:
                        df = _filter_to_days(df, days)
                        df = _standardize_columns(df)
                        logger.info(f"Loaded {len(df)} candles from {file_path}")
                        return df, f"Local file: {file_path.name}"
                except Exception as e:
                    logger.warning(f"Failed to read {file_path}: {e}")
                    continue

    logger.debug(f"No local files found for {symbol} {timeframe}")
    return None, "No local data"


def _load_from_cache(symbol: str, timeframe: str, days: int) -> Tuple[Optional[pd.DataFrame], str]:
    """Load data from cache directory."""
    cache_file = CACHE_DIR / f"{symbol}_{timeframe}.parquet"

    if cache_file.exists():
        # Check if cache is fresh enough (less than 1 hour old)
        cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        if cache_age < timedelta(hours=1):
            try:
                df = pd.read_parquet(cache_file)
                df = _filter_to_days(df, days)
                df = _standardize_columns(df)
                logger.info(f"Loaded {len(df)} candles from cache ({cache_age.seconds//60} min old)")
                return df, f"Cache ({cache_age.seconds//60}m old)"
            except Exception as e:
                logger.warning(f"Failed to read cache: {e}")

    return None, "No cached data"


def _fetch_from_exchange(symbol: str, timeframe: str, days: int) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Fetch fresh data from Binance via CCXT.

    Requires ccxt to be installed.
    """
    try:
        import ccxt
    except ImportError:
        logger.warning("CCXT not installed. Run: pip install ccxt")
        return None, "CCXT not installed"

    try:
        # Initialize exchange
        exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })

        # Calculate start timestamp
        since = datetime.now(timezone.utc) - timedelta(days=days)
        since_ms = int(since.timestamp() * 1000)

        # Timeframe to milliseconds mapping
        tf_ms = {
            '1m': 60_000, '5m': 300_000, '15m': 900_000, '30m': 1_800_000,
            '1h': 3_600_000, '4h': 14_400_000, '1d': 86_400_000
        }

        all_candles = []
        current_since = since_ms
        max_candles_per_request = 1000

        logger.info(f"Fetching {symbol} {timeframe} from Binance...")

        while True:
            candles = exchange.fetch_ohlcv(
                symbol,
                timeframe,
                since=current_since,
                limit=max_candles_per_request
            )

            if not candles:
                break

            all_candles.extend(candles)

            # Check if we got less than requested (end of data)
            if len(candles) < max_candles_per_request:
                break

            # Move to next batch
            last_ts = candles[-1][0]
            current_since = last_ts + tf_ms.get(timeframe, 3_600_000)

            # Prevent infinite loop
            if len(all_candles) > days * 24 * 60:  # Safety limit
                break

        if not all_candles:
            return None, "No data from exchange"

        # Convert to DataFrame
        df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['time'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df = df.drop(columns=['timestamp'])
        df = df.drop_duplicates(subset=['time']).sort_values('time').reset_index(drop=True)

        logger.info(f"Fetched {len(df)} candles from Binance")
        return df, f"Binance API ({len(df)} candles)"

    except ccxt.NetworkError as e:
        logger.error(f"Network error: {e}")
        return None, f"Network error: {e}"
    except ccxt.ExchangeError as e:
        logger.error(f"Exchange error: {e}")
        return None, f"Exchange error: {e}"
    except Exception as e:
        logger.error(f"Error fetching from exchange: {e}")
        return None, f"Error: {e}"


def _cache_data(df: pd.DataFrame, symbol: str, timeframe: str) -> bool:
    """Cache fetched data for future use."""
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = CACHE_DIR / f"{symbol}_{timeframe}.parquet"
        df.to_parquet(cache_file, index=False)
        logger.debug(f"Cached data to {cache_file}")
        return True
    except Exception as e:
        logger.warning(f"Failed to cache data: {e}")
        return False


def _read_file(file_path: Path) -> Optional[pd.DataFrame]:
    """Read parquet or CSV file."""
    if file_path.suffix == '.parquet':
        return pd.read_parquet(file_path)
    elif file_path.suffix == '.csv':
        return pd.read_csv(file_path)
    else:
        return None


def _filter_to_days(df: pd.DataFrame, days: int) -> pd.DataFrame:
    """Filter DataFrame to last N days."""
    time_col = None
    for col in ['time', 'timestamp', 'date', 'datetime']:
        if col in df.columns:
            time_col = col
            break

    if time_col is None:
        # Try using index
        if isinstance(df.index, pd.DatetimeIndex):
            cutoff = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=days)
            return df[df.index >= cutoff]
        return df

    # Ensure datetime type
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])

    # Calculate cutoff
    cutoff = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=days)

    # Handle timezone-aware vs naive
    if df[time_col].dt.tz is None:
        cutoff = cutoff.tz_localize(None)

    return df[df[time_col] >= cutoff].copy()


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to standard format."""
    # Lowercase all columns
    df.columns = df.columns.str.lower()

    # Rename common variations
    renames = {
        'timestamp': 'time',
        'date': 'time',
        'datetime': 'time',
        'o': 'open',
        'h': 'high',
        'l': 'low',
        'c': 'close',
        'v': 'volume',
    }

    df = df.rename(columns={k: v for k, v in renames.items() if k in df.columns})

    # Ensure required columns exist
    required = ['time', 'open', 'high', 'low', 'close']
    for col in required:
        if col not in df.columns:
            logger.warning(f"Missing required column: {col}")

    # Add volume if missing
    if 'volume' not in df.columns:
        df['volume'] = 0

    # Sort by time
    if 'time' in df.columns:
        df = df.sort_values('time').reset_index(drop=True)

    return df


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_load(symbol: str = "BTC/USDT", timeframe: str = "4h", days: int = 90) -> pd.DataFrame:
    """
    Quick load with sensible defaults. Returns empty DataFrame on failure.

    Example:
        df = quick_load()  # Loads BTC/USDT 4h, 90 days
        df = quick_load("ETH/USDT", "1h", 30)
    """
    df, _ = load_ohlcv(symbol, timeframe, days, source="auto")
    if df is None:
        logger.warning(f"Failed to load {symbol} {timeframe}")
        return pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    return df


def check_data_availability() -> dict:
    """
    Check what local data is available.

    Returns dict mapping symbol/timeframe to file info.
    """
    available = {}

    for data_dir in DATA_DIRS:
        if not data_dir.exists():
            continue

        for parquet_file in data_dir.glob("**/*.parquet"):
            key = parquet_file.stem
            available[key] = {
                'path': str(parquet_file),
                'size_mb': parquet_file.stat().st_size / (1024 * 1024),
                'modified': datetime.fromtimestamp(parquet_file.stat().st_mtime).isoformat()
            }

    return available
