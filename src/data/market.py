"""
Market Data Interface
====================
Unified interface for OHLCV market data access.

Follows industry best practices:
- Parquet for bulk storage (fast, compressed, columnar)
- Clear API for loading and updating data
- Automatic data validation and cleaning
- Support for multiple symbols and timeframes

Usage:
    from src.data.market import MarketData
    
    market = MarketData()
    df = market.load_ohlcv("BTC/USDT", "4h")
    market.update_ohlcv("BTC/USDT", "4h")  # Fetch missing candles
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, List
import pandas as pd
from loguru import logger

from src.data_engine import (
    fetch_ohlcv,
    clean_ohlcv,
    calculate_atr,
    normalize_symbol,
    get_symbol_data_dir
)


class MarketData:
    """
    Unified interface for market data (OHLCV).
    
    Single source of truth for all market data access.
    Uses Parquet files for fast, compressed storage.
    """
    
    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize market data manager.
        
        Args:
            base_dir: Base directory for market data (default: data/market)
        """
        self.base_dir = base_dir or Path("data/market")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Create OHLCV subdirectory
        self.ohlcv_dir = self.base_dir / "ohlcv"
        self.ohlcv_dir.mkdir(exist_ok=True)
        
        logger.info(f"MarketData initialized: {self.base_dir}")
    
    def _get_symbol_path(self, symbol: str) -> Path:
        """Get directory path for symbol data."""
        normalized = normalize_symbol(symbol)
        symbol_dir = self.ohlcv_dir / normalized
        symbol_dir.mkdir(exist_ok=True)
        return symbol_dir
    
    def _get_parquet_path(self, symbol: str, timeframe: str) -> Path:
        """Get parquet file path for symbol/timeframe."""
        symbol_dir = self._get_symbol_path(symbol)
        return symbol_dir / f"{timeframe}.parquet"
    
    def load_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Load OHLCV data for symbol/timeframe.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Candle timeframe (e.g., '1h', '4h')
            start_time: Optional start filter
            end_time: Optional end filter
            
        Returns:
            DataFrame with OHLCV data
            
        Example:
            >>> market = MarketData()
            >>> df = market.load_ohlcv("BTC/USDT", "4h")
            >>> print(df.tail())
        """
        parquet_path = self._get_parquet_path(symbol, timeframe)
        
        if not parquet_path.exists():
            raise FileNotFoundError(
                f"No data found for {symbol} {timeframe}\n"
                f"Expected: {parquet_path}\n"
                f"\n"
                f"To fetch this data, run:\n"
                f"  market = MarketData()\n"
                f"  market.update_ohlcv('{symbol}', '{timeframe}')"
            )
        
        # Load from parquet
        df = pd.read_parquet(parquet_path)
        
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
        
        logger.debug(f"Loaded {len(df)} candles for {symbol} {timeframe}")
        return df
    
    def update_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        years: int = 5,
        atr_period: int = 14
    ) -> pd.DataFrame:
        """
        Fetch and update OHLCV data from exchange.
        
        Args:
            symbol: Trading pair
            timeframe: Candle timeframe
            years: Years of historical data to fetch
            atr_period: ATR calculation period
            
        Returns:
            Updated DataFrame
        """
        logger.info(f"Updating {symbol} {timeframe} data...")
        
        # Fetch from exchange
        df = fetch_ohlcv(symbol=symbol, timeframe=timeframe, years=years)
        
        # Clean data
        df = clean_ohlcv(df)
        
        # Calculate ATR
        df['atr'] = calculate_atr(df, period=atr_period)
        df = df.dropna(subset=['atr'])
        
        # Normalize column names (lowercase)
        df.columns = df.columns.str.lower()
        
        # Save to parquet
        parquet_path = self._get_parquet_path(symbol, timeframe)
        df.to_parquet(parquet_path, index=False)
        
        logger.info(f"✅ Saved {len(df)} candles to {parquet_path}")
        return df
    
    def get_available_data(self) -> Dict[str, List[str]]:
        """
        Get available symbols and timeframes.
        
        Returns:
            Dictionary mapping symbol to list of available timeframes
        """
        available = {}
        
        for symbol_dir in self.ohlcv_dir.iterdir():
            if not symbol_dir.is_dir():
                continue
            
            symbol = symbol_dir.name
            timeframes = []
            
            for parquet_file in symbol_dir.glob("*.parquet"):
                timeframe = parquet_file.stem
                timeframes.append(timeframe)
            
            if timeframes:
                available[symbol] = sorted(timeframes)
        
        return available
    
    def get_date_range(self, symbol: str, timeframe: str) -> tuple[datetime, datetime]:
        """
        Get date range for symbol/timeframe.
        
        Returns:
            Tuple of (start_date, end_date)
        """
        df = self.load_ohlcv(symbol, timeframe)
        return (df['time'].min(), df['time'].max())
    
    def get_stats(self) -> Dict:
        """Get statistics about stored market data."""
        available = self.get_available_data()
        
        stats = {
            'symbols': len(available),
            'total_timeframes': sum(len(tfs) for tfs in available.values()),
            'symbols_detail': {}
        }
        
        for symbol, timeframes in available.items():
            stats['symbols_detail'][symbol] = {}
            for tf in timeframes:
                try:
                    start, end = self.get_date_range(symbol, tf)
                    df = self.load_ohlcv(symbol, tf)
                    stats['symbols_detail'][symbol][tf] = {
                        'candles': len(df),
                        'start': start.isoformat(),
                        'end': end.isoformat()
                    }
                except Exception as e:
                    logger.warning(f"Error loading {symbol} {tf}: {e}")
        
        return stats


# Convenience function
def get_market_data() -> MarketData:
    """Get global MarketData instance."""
    return MarketData()
