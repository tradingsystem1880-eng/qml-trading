"""
Enhanced Data Fetcher with Funding Rates, Open Interest, and Liquidations
==========================================================================
Extends base fetcher with crypto-specific data that improves pattern quality.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import ccxt
import pandas as pd
from loguru import logger

from config.settings import settings
from src.data.fetcher import DataFetcher


class EnhancedDataFetcher(DataFetcher):
    """
    Enhanced data fetcher with perpetual futures data.
    
    Additional data sources:
    - Funding rates (sentiment indicator)
    - Open interest (positioning indicator)
    - Liquidation data (squeeze detection)
    - Long/short ratios (crowd positioning)
    """
    
    def __init__(self, exchange_id: str = "binance", **kwargs):
        super().__init__(exchange_id, **kwargs)
        self._futures_exchange: Optional[ccxt.Exchange] = None
    
    @property
    def futures_exchange(self) -> ccxt.Exchange:
        """Get futures exchange for perpetual data."""
        if self._futures_exchange is None:
            exchange_class = getattr(ccxt, self.exchange_id)
            config = {
                "enableRateLimit": True,
                "options": {"defaultType": "future"}
            }
            if settings.exchange.api_key:
                config["apiKey"] = settings.exchange.api_key
                config["secret"] = settings.exchange.secret
            self._futures_exchange = exchange_class(config)
        return self._futures_exchange
    
    def fetch_funding_rate(
        self,
        symbol: str,
        since: Optional[datetime] = None,
        limit: int = 500
    ) -> pd.DataFrame:
        """
        Fetch historical funding rates.
        
        Funding rate is a key sentiment indicator:
        - Positive: Longs pay shorts (bullish sentiment)
        - Negative: Shorts pay longs (bearish sentiment)
        - Extreme values often precede reversals
        """
        try:
            # Convert symbol to futures format
            futures_symbol = symbol.replace("/", "") + ":USDT"
            
            since_ms = int(since.timestamp() * 1000) if since else None
            
            rates = self.futures_exchange.fetch_funding_rate_history(
                futures_symbol,
                since=since_ms,
                limit=limit
            )
            
            if not rates:
                return pd.DataFrame()
            
            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df = df.rename(columns={"fundingRate": "funding_rate"})
            
            return df[["time", "funding_rate"]].sort_values("time").reset_index(drop=True)
            
        except Exception as e:
            logger.warning(f"Failed to fetch funding rate for {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_open_interest(
        self,
        symbol: str,
        timeframe: str = "1h",
        since: Optional[datetime] = None,
        limit: int = 500
    ) -> pd.DataFrame:
        """
        Fetch historical open interest.
        
        Open interest indicates total outstanding contracts:
        - Rising OI + Rising Price = Strong trend
        - Rising OI + Falling Price = Potential reversal
        - Falling OI = Position unwinding
        """
        try:
            futures_symbol = symbol.replace("/", "") + ":USDT"
            since_ms = int(since.timestamp() * 1000) if since else None
            
            # Binance specific endpoint
            if self.exchange_id == "binance":
                oi_data = self.futures_exchange.fapiPublicGetOpenInterestHist({
                    "symbol": symbol.replace("/", ""),
                    "period": timeframe,
                    "limit": limit,
                    "startTime": since_ms
                })
                
                if not oi_data:
                    return pd.DataFrame()
                
                df = pd.DataFrame(oi_data)
                df["time"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                df["open_interest"] = df["sumOpenInterest"].astype(float)
                df["oi_value"] = df["sumOpenInterestValue"].astype(float)
                
                return df[["time", "open_interest", "oi_value"]].sort_values("time").reset_index(drop=True)
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.warning(f"Failed to fetch open interest for {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_long_short_ratio(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 500
    ) -> pd.DataFrame:
        """
        Fetch long/short ratio (top traders).
        
        Indicates how top traders are positioned:
        - Ratio > 1: More longs
        - Ratio < 1: More shorts
        - Extreme values often signal reversals
        """
        try:
            if self.exchange_id != "binance":
                return pd.DataFrame()
            
            data = self.futures_exchange.fapiPublicGetTopLongShortAccountRatio({
                "symbol": symbol.replace("/", ""),
                "period": timeframe,
                "limit": limit
            })
            
            if not data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            df["time"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df["long_short_ratio"] = df["longShortRatio"].astype(float)
            df["long_account"] = df["longAccount"].astype(float)
            df["short_account"] = df["shortAccount"].astype(float)
            
            return df[["time", "long_short_ratio", "long_account", "short_account"]].sort_values("time").reset_index(drop=True)
            
        except Exception as e:
            logger.warning(f"Failed to fetch long/short ratio for {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_enhanced_data(
        self,
        symbol: str,
        timeframe: str,
        since: Optional[datetime] = None,
        limit: int = 500
    ) -> pd.DataFrame:
        """
        Fetch OHLCV with all enhanced data merged.
        
        Returns DataFrame with:
        - Standard OHLCV
        - Funding rate (interpolated to timeframe)
        - Open interest
        - Long/short ratio
        """
        # Fetch base OHLCV
        df = self.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        
        if df.empty:
            return df
        
        # Fetch funding rates
        funding_df = self.fetch_funding_rate(symbol, since=since, limit=limit)
        if not funding_df.empty:
            funding_df = funding_df.set_index("time")
            df = df.set_index("time")
            df = df.join(funding_df, how="left")
            df["funding_rate"] = df["funding_rate"].ffill()
            df = df.reset_index()
        
        # Fetch open interest
        oi_df = self.fetch_open_interest(symbol, timeframe, since=since, limit=limit)
        if not oi_df.empty:
            oi_df = oi_df.set_index("time")
            df = df.set_index("time")
            df = df.join(oi_df, how="left")
            df[["open_interest", "oi_value"]] = df[["open_interest", "oi_value"]].ffill()
            df = df.reset_index()
        
        # Fetch long/short ratio
        ls_df = self.fetch_long_short_ratio(symbol, timeframe, limit=limit)
        if not ls_df.empty:
            ls_df = ls_df.set_index("time")
            df = df.set_index("time")
            df = df.join(ls_df[["long_short_ratio"]], how="left")
            df["long_short_ratio"] = df["long_short_ratio"].ffill()
            df = df.reset_index()
        
        logger.info(f"Fetched enhanced data for {symbol}: {len(df)} rows")
        
        return df


def create_enhanced_fetcher(exchange_id: str = "binance") -> EnhancedDataFetcher:
    """Factory for enhanced data fetcher."""
    return EnhancedDataFetcher(exchange_id=exchange_id)

