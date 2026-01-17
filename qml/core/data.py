"""
Data Loading Module
==================
Unified data loading using existing infrastructure.

Wraps the existing data interfaces for clean access.
"""

import pandas as pd
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta
from loguru import logger


class DataLoader:
    """
    Unified data loading interface.
    
    Wraps existing data infrastructure:
    - MarketData (Parquet files)
    - LiveDataFetcher (CCXT + cache)
    """
    
    def __init__(self, config=None):
        """Initialize data loader."""
        self.config = config
        self._live_fetcher = None
        self._market_data = None
    
    @property
    def live_fetcher(self):
        """Lazy-load live data fetcher."""
        if self._live_fetcher is None:
            try:
                from src.data.live_fetcher import create_live_fetcher
                self._live_fetcher = create_live_fetcher()
            except Exception:
                logger.warning("Live fetcher unavailable")
        return self._live_fetcher
    
    @property
    def market_data(self):
        """Lazy-load market data interface."""
        if self._market_data is None:
            try:
                from src.data.market import MarketData
                self._market_data = MarketData()
            except Exception:
                logger.warning("MarketData unavailable")
        return self._market_data
    
    def load_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        days: int = 365,
        use_live: bool = True
    ) -> pd.DataFrame:
        """
        Load OHLCV data.
        
        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            timeframe: Candle timeframe (e.g., "4h")
            days: Days of history
            use_live: Try live fetcher first
            
        Returns:
            DataFrame with OHLCV data
        """
        # Try live fetcher first (cached)
        if use_live and self.live_fetcher:
            try:
                return self.live_fetcher.get_ohlcv(symbol, timeframe, limit=days * 6)
            except Exception as e:
                logger.warning(f"Live fetch failed: {e}")
        
        # Fall back to stored data
        if self.market_data:
            try:
                return self.market_data.load_ohlcv(symbol, timeframe)
            except Exception as e:
                logger.warning(f"Market data load failed: {e}")
        
        # Try legacy data engine
        try:
            from src.data_engine import load_master_data
            return load_master_data(timeframe, symbol=symbol)
        except Exception as e:
            logger.error(f"All data sources failed: {e}")
            raise
    
    def get_available_symbols(self) -> list:
        """Get list of available symbols."""
        if self.market_data:
            return list(self.market_data.get_available_data().keys())
        return ["BTC/USDT", "ETH/USDT"]
    
    def get_available_timeframes(self, symbol: str = None) -> list:
        """Get available timeframes."""
        if self.market_data:
            data = self.market_data.get_available_data()
            if symbol and symbol in data:
                return data[symbol]
        return ["1h", "4h", "1d"]
