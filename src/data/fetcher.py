"""
Data Fetcher for QML Trading System
====================================
Fetches OHLCV data from exchanges via CCXT with robust error handling,
rate limiting, and automatic data synchronization.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import ccxt
import pandas as pd
from loguru import logger
from tqdm import tqdm

from config.settings import settings
from src.data.database import DatabaseManager, get_database


class DataFetcher:
    """
    Fetches cryptocurrency OHLCV data from exchanges.
    
    Features:
    - Multi-exchange support via CCXT
    - Automatic rate limiting
    - Incremental data fetching (only new candles)
    - Error handling with retries
    - Progress tracking
    """
    
    # CCXT timeframe mapping
    TIMEFRAME_MAP = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1h",
        "4h": "4h",
        "1d": "1d",
    }
    
    # Timeframe to milliseconds
    TIMEFRAME_MS = {
        "1m": 60 * 1000,
        "5m": 5 * 60 * 1000,
        "15m": 15 * 60 * 1000,
        "30m": 30 * 60 * 1000,
        "1h": 60 * 60 * 1000,
        "4h": 4 * 60 * 60 * 1000,
        "1d": 24 * 60 * 60 * 1000,
    }
    
    def __init__(
        self,
        exchange_id: str = "binance",
        db: Optional[DatabaseManager] = None,
        rate_limit: bool = True
    ):
        """
        Initialize data fetcher.
        
        Args:
            exchange_id: CCXT exchange ID (e.g., 'binance', 'coinbase')
            db: Database manager instance
            rate_limit: Enable rate limiting
        """
        self.exchange_id = exchange_id
        self.db = db or get_database()
        self.rate_limit = rate_limit
        
        # Initialize exchange
        self._exchange: Optional[ccxt.Exchange] = None
        self._markets_loaded = False
    
    @property
    def exchange(self) -> ccxt.Exchange:
        """Get or create exchange instance."""
        if self._exchange is None:
            exchange_class = getattr(ccxt, self.exchange_id)
            
            config: Dict[str, Any] = {
                "enableRateLimit": self.rate_limit,
                "options": {
                    "defaultType": "spot",  # Use spot market
                    "adjustForTimeDifference": True,
                }
            }
            
            # Add API keys if available (for higher rate limits)
            if settings.exchange.api_key and settings.exchange.secret:
                config["apiKey"] = settings.exchange.api_key
                config["secret"] = settings.exchange.secret
            
            # Use testnet if configured
            if settings.exchange.testnet:
                config["sandbox"] = True
            
            self._exchange = exchange_class(config)
            logger.info(f"Initialized {self.exchange_id} exchange connection")
        
        return self._exchange
    
    def load_markets(self) -> Dict[str, Any]:
        """
        Load exchange markets (required before trading operations).
        
        Returns:
            Dictionary of market info
        """
        if not self._markets_loaded:
            markets = self.exchange.load_markets()
            self._markets_loaded = True
            logger.info(f"Loaded {len(markets)} markets from {self.exchange_id}")
            return markets
        return self.exchange.markets
    
    def get_available_symbols(self, quote_currency: str = "USDT") -> List[str]:
        """
        Get list of available trading pairs for a quote currency.
        
        Args:
            quote_currency: Quote currency (e.g., 'USDT', 'BTC')
            
        Returns:
            List of symbol strings
        """
        self.load_markets()
        symbols = [
            symbol for symbol, market in self.exchange.markets.items()
            if market.get("quote") == quote_currency
            and market.get("active", True)
            and market.get("spot", True)
        ]
        return sorted(symbols)
    
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from exchange.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Candle timeframe (e.g., '1h', '4h', '1d')
            since: Start datetime
            until: End datetime
            limit: Maximum candles per request
            
        Returns:
            DataFrame with OHLCV data
        """
        self.load_markets()
        
        if symbol not in self.exchange.markets:
            raise ValueError(f"Symbol {symbol} not found on {self.exchange_id}")
        
        ccxt_timeframe = self.TIMEFRAME_MAP.get(timeframe, timeframe)
        timeframe_ms = self.TIMEFRAME_MS.get(timeframe, 3600000)
        
        # Convert datetime to timestamp
        since_ms = int(since.timestamp() * 1000) if since else None
        until_ms = int(until.timestamp() * 1000) if until else None
        
        all_candles = []
        current_since = since_ms
        
        # Determine end point
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        end_ms = min(until_ms, now_ms) if until_ms else now_ms
        
        # Fetch in batches
        while True:
            try:
                candles = self.exchange.fetch_ohlcv(
                    symbol,
                    ccxt_timeframe,
                    since=current_since,
                    limit=limit
                )
                
                if not candles:
                    break
                
                all_candles.extend(candles)
                
                # Get timestamp of last candle
                last_timestamp = candles[-1][0]
                
                # Check if we've reached the end
                if last_timestamp >= end_ms or len(candles) < limit:
                    break
                
                # Move to next batch
                current_since = last_timestamp + timeframe_ms
                
                # Small delay to respect rate limits
                if self.rate_limit:
                    asyncio.get_event_loop().run_until_complete(asyncio.sleep(0.1))
                
            except ccxt.NetworkError as e:
                logger.warning(f"Network error fetching {symbol}: {e}. Retrying...")
                asyncio.get_event_loop().run_until_complete(asyncio.sleep(1))
                continue
            except ccxt.ExchangeError as e:
                logger.error(f"Exchange error fetching {symbol}: {e}")
                break
        
        if not all_candles:
            return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
        
        # Convert to DataFrame
        df = pd.DataFrame(
            all_candles,
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        
        # Convert timestamp to datetime
        df["time"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.drop(columns=["timestamp"])
        
        # Filter to requested range
        if until:
            df = df[df["time"] <= pd.Timestamp(until, tz="UTC")]
        
        # Remove duplicates and sort
        df = df.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
        
        logger.debug(f"Fetched {len(df)} candles for {symbol} {timeframe}")
        return df
    
    def sync_symbol(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        force_full: bool = False
    ) -> int:
        """
        Synchronize data for a single symbol/timeframe.
        
        Fetches only new data since last stored candle.
        
        Args:
            symbol: Trading pair
            timeframe: Candle timeframe
            start_date: Start date for initial fetch
            force_full: Force full re-fetch from start_date
            
        Returns:
            Number of new candles stored
        """
        # Determine start point
        if force_full or start_date:
            since = start_date or datetime(2020, 1, 1, tzinfo=timezone.utc)
        else:
            # Get latest timestamp from database
            latest = self.db.get_latest_timestamp(symbol, timeframe, self.exchange_id)
            if latest:
                # Start from next candle
                since = latest + timedelta(milliseconds=self.TIMEFRAME_MS.get(timeframe, 3600000))
            else:
                # Default start date
                since = datetime(2020, 1, 1, tzinfo=timezone.utc)
        
        logger.info(f"Syncing {symbol} {timeframe} from {since}")
        
        # Fetch data
        df = self.fetch_ohlcv(symbol, timeframe, since=since)
        
        if df.empty:
            logger.info(f"No new data for {symbol} {timeframe}")
            return 0
        
        # Store in database
        count = self.db.insert_ohlcv(df, symbol, timeframe, self.exchange_id)
        logger.info(f"Stored {count} new candles for {symbol} {timeframe}")
        
        return count
    
    def sync_all_symbols(
        self,
        symbols: Optional[List[str]] = None,
        timeframes: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        force_full: bool = False,
        show_progress: bool = True
    ) -> Dict[str, int]:
        """
        Synchronize data for multiple symbols and timeframes.
        
        Args:
            symbols: List of symbols (defaults to configured symbols)
            timeframes: List of timeframes (defaults to configured timeframes)
            start_date: Start date for initial fetch
            force_full: Force full re-fetch
            show_progress: Show progress bar
            
        Returns:
            Dictionary mapping symbol/timeframe to candle count
        """
        symbols = symbols or settings.detection.symbols
        timeframes = timeframes or settings.detection.timeframes
        
        results: Dict[str, int] = {}
        total_tasks = len(symbols) * len(timeframes)
        
        iterator = tqdm(
            [(s, tf) for s in symbols for tf in timeframes],
            desc="Syncing data",
            disable=not show_progress
        )
        
        for symbol, timeframe in iterator:
            iterator.set_postfix({"symbol": symbol, "timeframe": timeframe})
            
            try:
                count = self.sync_symbol(symbol, timeframe, start_date, force_full)
                results[f"{symbol}_{timeframe}"] = count
            except Exception as e:
                logger.error(f"Failed to sync {symbol} {timeframe}: {e}")
                results[f"{symbol}_{timeframe}"] = -1
        
        # Summary
        total_candles = sum(v for v in results.values() if v > 0)
        failed = sum(1 for v in results.values() if v < 0)
        
        logger.info(f"Sync complete: {total_candles} candles stored, {failed} failures")
        
        return results
    
    def get_data(
        self,
        symbol: str,
        timeframe: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
        auto_sync: bool = True
    ) -> pd.DataFrame:
        """
        Get OHLCV data, automatically syncing if needed.
        
        This is the main entry point for getting price data.
        
        Args:
            symbol: Trading pair
            timeframe: Candle timeframe
            start_time: Start of date range
            end_time: End of date range
            limit: Maximum candles (from most recent)
            auto_sync: Automatically sync new data
            
        Returns:
            DataFrame with OHLCV data
        """
        # Auto-sync if requested
        if auto_sync:
            self.sync_symbol(symbol, timeframe)
        
        # Retrieve from database
        df = self.db.get_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
            exchange=self.exchange_id
        )
        
        return df
    
    def get_multiple_symbols(
        self,
        symbols: List[str],
        timeframe: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        auto_sync: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Get OHLCV data for multiple symbols.
        
        Args:
            symbols: List of trading pairs
            timeframe: Candle timeframe
            start_time: Start of date range
            end_time: End of date range
            auto_sync: Automatically sync new data
            
        Returns:
            Dictionary mapping symbol to DataFrame
        """
        results: Dict[str, pd.DataFrame] = {}
        
        for symbol in symbols:
            try:
                df = self.get_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_time=start_time,
                    end_time=end_time,
                    auto_sync=auto_sync
                )
                results[symbol] = df
            except Exception as e:
                logger.error(f"Failed to get data for {symbol}: {e}")
                results[symbol] = pd.DataFrame()
        
        return results
    
    def calculate_atr(
        self,
        df: pd.DataFrame,
        period: int = 14
    ) -> pd.Series:
        """
        Calculate Average True Range (ATR).
        
        Args:
            df: OHLCV DataFrame
            period: ATR period
            
        Returns:
            Series with ATR values
        """
        high = df["high"]
        low = df["low"]
        close = df["close"]
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate ATR (Wilder's smoothing)
        atr = true_range.ewm(span=period, adjust=False).mean()
        
        return atr
    
    def close(self) -> None:
        """Close exchange connection."""
        if self._exchange:
            # CCXT doesn't require explicit closing, but clean up
            self._exchange = None
            self._markets_loaded = False
            logger.info("Exchange connection closed")


# Factory function
def create_data_fetcher(
    exchange_id: str = "binance",
    db: Optional[DatabaseManager] = None
) -> DataFetcher:
    """
    Create a data fetcher instance.
    
    Args:
        exchange_id: Exchange to use
        db: Database manager
        
    Returns:
        DataFetcher instance
    """
    return DataFetcher(exchange_id=exchange_id, db=db)

