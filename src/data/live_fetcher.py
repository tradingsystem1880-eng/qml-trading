"""
Live Data Fetcher - Clean CCXT API
===================================
Lightweight data fetcher with in-memory caching and optional SQLite storage.

Design Philosophy:
- CCXT for live data fetching
- LRU cache for recent data (fast access)
- Optional SQLite for persistence (when needed)
- Updates every 10-15 minutes (configurable)
- Minimal overhead, maximum performance

Usage:
    fetcher = LiveDataFetcher()
    df = fetcher.get_ohlcv("BTC/USDT", "4h", limit=500)  # Cached
    fetcher.refresh()  # Force refresh from exchange
"""

from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Optional, Dict, Any, List
import hashlib
import time

import ccxt
import pandas as pd
from loguru import logger


class LiveDataFetcher:
    """
    Clean, fast CCXT data fetcher with intelligent caching.
    
    Features:
    - LRU cache for recent queries (instant access)
    - Optional SQLite persistence
    - Background refresh every N minutes
    - Automatic rate limiting via CCXT
    """
    
    def __init__(
        self,
        exchange_id: str = "binance",
        cache_size: int = 100,
        cache_ttl_minutes: int = 15,
        use_storage: bool = False,
        storage_path: Optional[Path] = None
    ):
        """
        Initialize live data fetcher.
        
        Args:
            exchange_id: CCXT exchange ID (default: binance)
            cache_size: LRU cache size (default: 100 queries)
            cache_ttl_minutes: Cache validity in minutes (default: 15)
            use_storage: Enable SQLite persistence (default: False)
            storage_path: SQLite database path (default: data/cache.db)
        """
        self.exchange_id = exchange_id
        self.cache_ttl = timedelta(minutes=cache_ttl_minutes)
        self.use_storage = use_storage
        self.storage_path = storage_path or Path("data/cache.db")
        
        # Initialize CCXT exchange
        self.exchange = self._init_exchange()
        
        # Cache metadata
        self._cache_timestamps: Dict[str, datetime] = {}
        
        # Storage (optional)
        if use_storage:
            self._init_storage()
        
        logger.info(f"LiveDataFetcher initialized: {exchange_id}")
        logger.info(f"  Cache: {cache_size} queries, TTL: {cache_ttl_minutes}min")
        logger.info(f"  Storage: {'enabled' if use_storage else 'disabled'}")
    
    def _init_exchange(self) -> ccxt.Exchange:
        """Initialize CCXT exchange with rate limiting."""
        exchange_class = getattr(ccxt, self.exchange_id)
        exchange = exchange_class({
            "enableRateLimit": True,  # Built-in rate limiting
            "options": {
                "defaultType": "spot",
                "adjustForTimeDifference": True,
            }
        })
        return exchange
    
    def _init_storage(self):
        """Initialize SQLite storage (optional)."""
        try:
            from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime
            from sqlalchemy.ext.declarative import declarative_base
            from sqlalchemy.orm import sessionmaker
            
            # Create tables if needed
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            engine = create_engine(f"sqlite:///{self.storage_path}")
            
            Base = declarative_base()
            
            class CachedOHLCV(Base):
                __tablename__ = 'cached_ohlcv'
                id = Column(Integer, primary_key=True)
                symbol = Column(String, index=True)
                timeframe = Column(String, index=True)
                timestamp = Column(DateTime, index=True)
                open = Column(Float)
                high = Column(Float)
                low = Column(Float)
                close = Column(Float)
                volume = Column(Float)
                fetched_at = Column(DateTime, default=datetime.utcnow)
            
            Base.metadata.create_all(engine)
            Session = sessionmaker(bind=engine)
            self.db_session = Session()
            self.CachedOHLCV = CachedOHLCV
            
            logger.info(f"SQLite storage initialized: {self.storage_path}")
        except ImportError:
            logger.warning("SQLAlchemy not installed. Storage disabled.")
            self.use_storage = False
    
    def _make_cache_key(self, symbol: str, timeframe: str, limit: int) -> str:
        """Generate cache key for query."""
        key_str = f"{symbol}_{timeframe}_{limit}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self._cache_timestamps:
            return False
        
        cached_time = self._cache_timestamps[cache_key]
        elapsed = datetime.now(timezone.utc) - cached_time
        
        return elapsed < self.cache_ttl
    
    @lru_cache(maxsize=100)
    def _fetch_from_exchange(
        self,
        symbol: str,
        timeframe: str,
        limit: int,
        _cache_buster: int  # Used to force cache refresh
    ) -> pd.DataFrame:
        """
        Fetch data from exchange (cached).
        
        Note: _cache_buster parameter allows forcing refresh
        by changing the value (e.g., timestamp).
        """
        logger.debug(f"Fetching {symbol} {timeframe} from {self.exchange_id}")
        
        try:
            ohlcv = self.exchange.fetch_ohlcv(
                symbol,
                timeframe,
                limit=limit
            )
            
            if not ohlcv:
                return pd.DataFrame()
            
            df = pd.DataFrame(
                ohlcv,
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["time"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df = df.drop(columns=["timestamp"])
            
            # Reorder columns
            df = df[["time", "open", "high", "low", "close", "volume"]]
            
            logger.debug(f"Fetched {len(df)} candles for {symbol} {timeframe}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch {symbol} {timeframe}: {e}")
            return pd.DataFrame()
    
    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str = "4h",
        limit: int = 500,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Get OHLCV data with intelligent caching.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Candle timeframe (e.g., '1h', '4h', '1d')
            limit: Number of candles to fetch
            force_refresh: Bypass cache and fetch fresh data
            
        Returns:
            DataFrame with OHLCV data
            
        Example:
            >>> fetcher = LiveDataFetcher()
            >>> df = fetcher.get_ohlcv("BTC/USDT", "4h", limit=200)
            >>> print(df.tail())
        """
        cache_key = self._make_cache_key(symbol, timeframe, limit)
        
        # Check if cache is valid
        if not force_refresh and self._is_cache_valid(cache_key):
            # Use cached data (LRU cache handles this)
            cache_buster = 0
        else:
            # Force fresh fetch by changing cache_buster
            cache_buster = int(time.time())
            self._cache_timestamps[cache_key] = datetime.now(timezone.utc)
        
        # Fetch (from cache or exchange)
        df = self._fetch_from_exchange(symbol, timeframe, limit, cache_buster)
        
        # Optionally store to SQLite
        if self.use_storage and not df.empty:
            self._store_to_db(df, symbol, timeframe)
        
        return df
    
    def _store_to_db(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """Store data to SQLite (optional)."""
        if not self.use_storage:
            return
        
        try:
            for _, row in df.iterrows():
                record = self.CachedOHLCV(
                    symbol=symbol,
                    timeframe=timeframe,
                    timestamp=row['time'],
                    open=row['open'],
                    high=row['high'],
                    low=row['low'],
                    close=row['close'],
                    volume=row['volume']
                )
                self.db_session.merge(record)
            
            self.db_session.commit()
            logger.debug(f"Stored {len(df)} candles to SQLite")
        except Exception as e:
            logger.warning(f"Failed to store to DB: {e}")
            self.db_session.rollback()
    
    def refresh(self, symbol: Optional[str] = None, timeframe: Optional[str] = None):
        """
        Force refresh data from exchange.
        
        Args:
            symbol: Specific symbol to refresh (default: all cached)
            timeframe: Specific timeframe to refresh (default: all)
        """
        if symbol and timeframe:
            # Refresh specific query
            cache_key = self._make_cache_key(symbol, timeframe, 500)
            if cache_key in self._cache_timestamps:
                del self._cache_timestamps[cache_key]
            logger.info(f"Refreshed cache for {symbol} {timeframe}")
        else:
            # Clear all cache timestamps
            self._cache_timestamps.clear()
            # Clear LRU cache
            self._fetch_from_exchange.cache_clear()
            logger.info("Cleared all cache")
    
    def get_multiple(
        self,
        symbols: List[str],
        timeframe: str = "4h",
        limit: int = 500
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols efficiently.
        
        Args:
            symbols: List of trading pairs
            timeframe: Candle timeframe
            limit: Number of candles per symbol
            
        Returns:
            Dictionary mapping symbol to DataFrame
        """
        results = {}
        
        for symbol in symbols:
            df = self.get_ohlcv(symbol, timeframe, limit)
            results[symbol] = df
        
        return results
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache statistics."""
        cache_info = self._fetch_from_exchange.cache_info()
        
        return {
            "cache_size": cache_info.currsize,
            "cache_maxsize": cache_info.maxsize,
            "cache_hits": cache_info.hits,
            "cache_misses": cache_info.misses,
            "hit_rate": cache_info.hits / (cache_info.hits + cache_info.misses) if (cache_info.hits + cache_info.misses) > 0 else 0,
            "cached_queries": len(self._cache_timestamps),
            "ttl_minutes": self.cache_ttl.total_seconds() / 60
        }


# Factory function for convenience
def create_live_fetcher(
    exchange_id: str = "binance",
    cache_ttl_minutes: int = 15,
    use_storage: bool = False
) -> LiveDataFetcher:
    """
    Create a live data fetcher instance.
    
    Args:
        exchange_id: Exchange to use (default: binance)
        cache_ttl_minutes: Cache validity in minutes (default: 15)
        use_storage: Enable SQLite persistence (default: False)
        
    Returns:
        LiveDataFetcher instance
        
    Example:
        >>> fetcher = create_live_fetcher(cache_ttl_minutes=10)
        >>> df = fetcher.get_ohlcv("BTC/USDT", "4h")
    """
    return LiveDataFetcher(
        exchange_id=exchange_id,
        cache_ttl_minutes=cache_ttl_minutes,
        use_storage=use_storage
    )
