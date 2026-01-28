"""
Funding Rate Fetcher for QML Trading System
============================================
Fetches funding rates from Bybit via ccxt.

Used for Phase 9.7 funding rate filter validation.

Filter rules:
- Reject LONG if funding > +0.01% (0.0001) - Too many longs paying
- Reject SHORT if funding < -0.01% (-0.0001) - Too many shorts paying

Usage:
    from src.data.funding_rates import FundingRateFetcher

    fetcher = FundingRateFetcher()

    # Get current funding rate
    rate = fetcher.get_current_funding('BTC/USDT')
    print(f"Current funding: {rate['funding_rate']:.4%}")

    # Get historical funding
    df = fetcher.get_historical_funding('BTC/USDT', days=365)

    # Check if trade should be filtered
    should_filter = fetcher.should_filter_trade('BTC/USDT', 'LONG', threshold=0.0001)
"""

import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd

try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    ccxt = None


PROJECT_ROOT = Path(__file__).parent.parent.parent


@dataclass
class FundingRateData:
    """Funding rate snapshot."""
    symbol: str
    timestamp: datetime
    funding_rate: float
    predicted_rate: Optional[float] = None
    next_funding_time: Optional[datetime] = None


@dataclass
class FundingRateFetcherConfig:
    """Configuration for funding rate fetcher."""
    exchange: str = "bybit"
    default_threshold: float = 0.0001  # 0.01%
    cache_ttl_seconds: int = 300  # 5 minutes
    rate_limit_delay: float = 0.2  # seconds between API calls
    data_dir: Path = PROJECT_ROOT / "data" / "funding_rates"


class FundingRateFetcher:
    """
    Fetches funding rates from Bybit via ccxt.

    Supports:
    - Current funding rate
    - Historical funding rates
    - Trade filtering based on funding direction

    IMPORTANT DATA LIMITATION:
    --------------------------
    This fetcher returns REALIZED (settled) funding rates, not PREDICTED rates.

    - Realized rate: The rate that was actually charged at the last funding time
    - Predicted rate: The estimated rate for the NEXT funding interval

    For backtesting, we use the most recent REALIZED rate published BEFORE trade entry.
    This is the rate that was just settled, reflecting market positioning 0-8 hours prior.

    In LIVE TRADING, you should use the PREDICTED rate instead:
    - Bybit API: /v5/market/tickers (fundingRate field)
    - CCXT: exchange.fetch_ticker(symbol)['info']['fundingRate']

    This difference may affect filter performance. Document in research journal.
    """

    def __init__(self, config: Optional[FundingRateFetcherConfig] = None):
        """
        Initialize funding rate fetcher.

        Args:
            config: Configuration settings
        """
        if not CCXT_AVAILABLE:
            raise ImportError(
                "ccxt is required for funding rate fetching. "
                "Install with: pip install ccxt"
            )

        self.config = config or FundingRateFetcherConfig()
        self._exchange = None
        self._cache: Dict[str, tuple[float, Any]] = {}  # symbol -> (timestamp, data)

        # Create data directory
        self.config.data_dir.mkdir(parents=True, exist_ok=True)

    @property
    def exchange(self):
        """Lazy-load exchange connection."""
        if self._exchange is None:
            if self.config.exchange == "bybit":
                self._exchange = ccxt.bybit({
                    'enableRateLimit': True,
                    'options': {
                        'defaultType': 'swap',  # Perpetual futures
                    }
                })
            else:
                raise ValueError(f"Unsupported exchange: {self.config.exchange}")
        return self._exchange

    def _normalize_symbol(self, symbol: str) -> str:
        """
        Normalize symbol to CCXT format for perpetual futures.

        Input formats: 'BTC/USDT', 'BTCUSDT', 'BTC-USDT', 'BTC/USDT:USDT'
        Output: 'BTC/USDT:USDT' (CCXT perpetual format)
        """
        # Remove any settlement suffix first
        symbol = symbol.replace(':USDT', '').replace(':USD', '')

        # Handle different input formats
        if '/' in symbol:
            base, quote = symbol.split('/')
        elif '-' in symbol:
            base, quote = symbol.split('-')
        elif 'USDT' in symbol:
            base = symbol.replace('USDT', '')
            quote = 'USDT'
        elif 'USD' in symbol:
            base = symbol.replace('USD', '')
            quote = 'USD'
        else:
            raise ValueError(f"Cannot parse symbol: {symbol}")

        # Return CCXT perpetual format
        return f"{base}/{quote}:{quote}"

    def _check_cache(self, cache_key: str) -> Optional[Any]:
        """Check if cached data is still valid."""
        if cache_key in self._cache:
            cached_time, data = self._cache[cache_key]
            if time.time() - cached_time < self.config.cache_ttl_seconds:
                return data
        return None

    def _set_cache(self, cache_key: str, data: Any):
        """Set cache with timestamp."""
        self._cache[cache_key] = (time.time(), data)

    def get_current_funding(self, symbol: str) -> Optional[Dict]:
        """
        Get current funding rate for a symbol.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT' or 'BTCUSDT')

        Returns:
            Dict with 'funding_rate', 'timestamp', 'next_funding_time'
            or None if unavailable
        """
        normalized = self._normalize_symbol(symbol)
        cache_key = f"current_{normalized}"

        # Check cache
        cached = self._check_cache(cache_key)
        if cached is not None:
            return cached

        try:
            # Fetch funding rate info
            funding_info = self.exchange.fetch_funding_rate(normalized)

            result = {
                'symbol': symbol,
                'funding_rate': funding_info.get('fundingRate', 0),
                'timestamp': datetime.fromtimestamp(funding_info.get('timestamp', 0) / 1000),
                'next_funding_time': datetime.fromtimestamp(
                    funding_info.get('fundingTimestamp', 0) / 1000
                ) if funding_info.get('fundingTimestamp') else None,
                'predicted_rate': funding_info.get('nextFundingRate'),
            }

            self._set_cache(cache_key, result)
            return result

        except Exception as e:
            print(f"Error fetching funding rate for {symbol}: {e}")
            return None

    def get_historical_funding(
        self,
        symbol: str,
        days: int = 365,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Get historical funding rates with pagination.

        Bybit provides funding every 8 hours (3x per day).
        365 days = ~1095 data points. Bybit returns max 200 records per call.

        Args:
            symbol: Trading pair
            days: Number of days to fetch (if start/end not provided)
            start_time: Start of period (overrides days)
            end_time: End of period (defaults to now)

        Returns:
            DataFrame with columns: timestamp, funding_rate, symbol
        """
        normalized = self._normalize_symbol(symbol)

        # Calculate time range
        if end_time is None:
            end_time = datetime.now()
        if start_time is None:
            start_time = end_time - timedelta(days=days)

        # Check if we have cached data on disk
        cache_file = self.config.data_dir / f"{symbol.replace('/', '_')}_funding.parquet"
        if cache_file.exists():
            df = pd.read_parquet(cache_file)
            df_start = df['timestamp'].min()
            df_end = df['timestamp'].max()

            # If cached data covers our range, return it
            if df_start <= start_time and df_end >= end_time - timedelta(hours=8):
                mask = (df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)
                return df[mask].reset_index(drop=True)

        # Fetch from API with pagination (going backward from end)
        all_records = []
        current_end = int(end_time.timestamp() * 1000)
        target_start = int(start_time.timestamp() * 1000)

        print(f"Fetching historical funding for {symbol} from {start_time.date()} to {end_time.date()}...")

        try:
            while current_end > target_start:
                # CCXT method for funding rate history with endTime for pagination
                records = self.exchange.fetch_funding_rate_history(
                    symbol=normalized,
                    since=None,  # Let exchange determine
                    limit=200,
                    params={'endTime': current_end}
                )

                if not records:
                    break

                all_records.extend(records)

                # Move end time to before oldest record
                oldest_ts = min(r['timestamp'] for r in records)
                current_end = oldest_ts - 1

                print(f"  Fetched {len(records)} records, total: {len(all_records)}")

                # Rate limiting
                time.sleep(self.config.rate_limit_delay)

                # Stop if we've gone past our target
                if oldest_ts < target_start:
                    break

        except Exception as e:
            print(f"  Error during pagination: {e}")
            if not all_records:
                return pd.DataFrame(columns=['timestamp', 'funding_rate', 'symbol'])

        if not all_records:
            return pd.DataFrame(columns=['timestamp', 'funding_rate', 'symbol'])

        # Convert to DataFrame
        df = pd.DataFrame([{
            'symbol': symbol,
            'timestamp': pd.to_datetime(r['timestamp'], unit='ms', utc=True),
            'funding_rate': r['fundingRate'],
            'funding_timestamp': r['timestamp']
        } for r in all_records])

        # Remove duplicates, sort by time
        df = df.drop_duplicates(subset=['funding_timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Filter to requested date range
        df = df[df['timestamp'] >= pd.Timestamp(start_time, tz='UTC')]

        if len(df) > 0:
            # Save to disk cache
            df.to_parquet(cache_file)
            print(f"  Final dataset: {len(df)} records from {df['timestamp'].min()} to {df['timestamp'].max()}")

        return df

    def should_filter_trade(
        self,
        symbol: str,
        direction: str,
        threshold: Optional[float] = None,
        funding_rate: Optional[float] = None,
    ) -> tuple:
        """
        Determine if trade should be filtered based on funding rate.

        Filter rules (include equality as per DeepSeek):
        - Reject LONG if funding >= +threshold (too many longs paying shorts)
        - Reject SHORT if funding <= -threshold (too many shorts paying longs)

        Args:
            symbol: Trading pair
            direction: 'LONG' or 'SHORT'
            threshold: Funding rate threshold (default 0.01% = 0.0001)
            funding_rate: Pre-fetched funding rate (optional)

        Returns:
            Tuple of (should_filter: bool, reason: str)
        """
        if threshold is None:
            threshold = self.config.default_threshold

        # Get funding rate if not provided
        if funding_rate is None:
            current = self.get_current_funding(symbol)
            if current is None:
                return True, "MISSING_DATA: Could not fetch funding rate"
            funding_rate = current['funding_rate']

        if funding_rate is None:
            return True, "MISSING_DATA: funding_rate is None"

        direction = direction.upper()

        # Filter logic (include equality as per DeepSeek)
        if direction == 'LONG' and funding_rate >= threshold:
            return True, f"LONG_OVERCROWDED: funding {funding_rate:.6f} >= {threshold}"
        elif direction == 'SHORT' and funding_rate <= -threshold:
            return True, f"SHORT_OVERCROWDED: funding {funding_rate:.6f} <= {-threshold}"
        else:
            return False, f"PASSED: funding {funding_rate:.6f} within threshold"

    def get_funding_at_time(
        self,
        symbol: str,
        target_time: datetime,
        df_funding: Optional[pd.DataFrame] = None,
    ) -> Optional[float]:
        """
        Get the funding rate at a specific time.

        Funding rates are published every 8 hours. This returns the most recent
        funding rate as of the target time.

        Args:
            symbol: Trading pair
            target_time: Target timestamp
            df_funding: Pre-loaded funding DataFrame (for efficiency in backtests)

        Returns:
            Funding rate at the time, or None if not available
        """
        if df_funding is None:
            # Load from disk cache
            cache_file = self.config.data_dir / f"{symbol.replace('/', '_')}_funding.parquet"
            if not cache_file.exists():
                return None
            df_funding = pd.read_parquet(cache_file)

        # Find most recent funding rate before target_time
        df_before = df_funding[df_funding['timestamp'] <= target_time]
        if len(df_before) == 0:
            return None

        return df_before.iloc[-1]['funding_rate']

    def load_funding_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load cached funding data from disk."""
        cache_file = self.config.data_dir / f"{symbol.replace('/', '_')}_funding.parquet"
        if cache_file.exists():
            return pd.read_parquet(cache_file)
        return None

    def get_available_symbols(self) -> List[str]:
        """Get list of symbols with cached funding data."""
        if not self.config.data_dir.exists():
            return []

        symbols = []
        for f in self.config.data_dir.glob("*_funding.parquet"):
            # Convert filename back to symbol
            symbol = f.stem.replace("_funding", "").replace("_", "/")
            symbols.append(symbol)

        return sorted(symbols)


def create_funding_filter(
    fetcher: FundingRateFetcher,
    threshold: float = 0.0001,
) -> callable:
    """
    Create a filter function for use with FeatureValidator.

    Args:
        fetcher: FundingRateFetcher instance
        threshold: Funding rate threshold

    Returns:
        Filter function that takes a trade dict and returns True to KEEP
    """
    def filter_func(trade: dict) -> bool:
        """Returns True if trade should be KEPT (not filtered)."""
        symbol = trade.get('symbol', '')
        direction = trade.get('direction', '')
        timestamp = trade.get('entry_time') or trade.get('timestamp')

        if not symbol or not direction:
            return True  # Can't filter without info

        # Get funding rate at trade time
        funding = fetcher.get_funding_at_time(symbol, timestamp)

        if funding is None:
            return False  # Skip trades without funding data (conservative)

        # Check if trade should be filtered
        should_filter = fetcher.should_filter_trade(
            symbol, direction, threshold=threshold, funding_rate=funding
        )

        return not should_filter  # Return True to KEEP

    return filter_func
