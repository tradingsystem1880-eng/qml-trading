#!/usr/bin/env python3
"""
Historical Data Sync Script
============================
Fetches and stores historical OHLCV data in TimescaleDB.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from loguru import logger

from src.data.fetcher import DataFetcher
from src.data.database import DatabaseManager
from src.utils.logging import setup_logging


def sync_historical_data(
    symbols: list[str],
    timeframes: list[str],
    days_back: int = 365,
    batch_size: int = 1000
):
    """
    Sync historical data for given symbols and timeframes.
    
    Args:
        symbols: List of trading pairs
        timeframes: List of timeframes
        days_back: Number of days of history to fetch
        batch_size: Number of candles per API request
    """
    setup_logging("INFO")
    
    fetcher = DataFetcher()
    db = DatabaseManager()
    
    logger.info(f"Starting historical data sync")
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Timeframes: {timeframes}")
    logger.info(f"Days back: {days_back}")
    
    total_synced = 0
    
    for symbol in symbols:
        for timeframe in timeframes:
            logger.info(f"Syncing {symbol} {timeframe}...")
            
            try:
                # Calculate how many candles we need
                tf_minutes = {
                    '1m': 1, '5m': 5, '15m': 15, '30m': 30,
                    '1h': 60, '2h': 120, '4h': 240,
                    '1d': 1440, '1w': 10080
                }
                
                minutes_per_candle = tf_minutes.get(timeframe, 60)
                total_candles = (days_back * 24 * 60) // minutes_per_candle
                
                logger.info(f"  Need ~{total_candles} candles for {days_back} days")
                
                # Fetch in batches
                all_data = []
                since = int((datetime.now() - timedelta(days=days_back)).timestamp() * 1000)
                
                while True:
                    try:
                        # Fetch batch
                        ohlcv = fetcher.exchange.fetch_ohlcv(
                            symbol,
                            timeframe,
                            since=since,
                            limit=batch_size
                        )
                        
                        if not ohlcv:
                            break
                        
                        all_data.extend(ohlcv)
                        
                        # Update since to last candle + 1
                        since = ohlcv[-1][0] + 1
                        
                        logger.info(f"  Fetched {len(all_data)} candles...")
                        
                        # Rate limiting
                        time.sleep(0.5)
                        
                        # Check if we've caught up to current time
                        if ohlcv[-1][0] > (datetime.now().timestamp() * 1000 - minutes_per_candle * 60 * 1000):
                            break
                            
                        # Safety limit
                        if len(all_data) >= total_candles * 1.5:
                            break
                            
                    except Exception as e:
                        logger.warning(f"  Batch fetch error: {e}")
                        time.sleep(2)
                        continue
                
                if all_data:
                    # Convert to DataFrame
                    df = pd.DataFrame(
                        all_data,
                        columns=['time', 'open', 'high', 'low', 'close', 'volume']
                    )
                    df['time'] = pd.to_datetime(df['time'], unit='ms')
                    df = df.drop_duplicates(subset=['time'])
                    df = df.sort_values('time')
                    
                    # Store in database
                    stored = db.insert_ohlcv(df, symbol, timeframe)
                    total_synced += stored
                    
                    logger.success(f"  ✅ Stored {stored} candles for {symbol} {timeframe}")
                else:
                    logger.warning(f"  No data fetched for {symbol} {timeframe}")
                    
            except Exception as e:
                logger.error(f"  Error syncing {symbol} {timeframe}: {e}")
                continue
            
            # Rate limiting between symbols
            time.sleep(1)
    
    logger.success(f"\n{'='*50}")
    logger.success(f"Sync complete! Total candles stored: {total_synced}")
    logger.success(f"{'='*50}")
    
    return total_synced


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Sync historical data")
    parser.add_argument(
        "--symbols",
        type=str,
        default="BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT,XRP/USDT",
        help="Comma-separated list of symbols"
    )
    parser.add_argument(
        "--timeframes",
        type=str,
        default="1h,4h,1d",
        help="Comma-separated list of timeframes"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Number of days of history to fetch"
    )
    
    args = parser.parse_args()
    
    symbols = [s.strip() for s in args.symbols.split(",")]
    timeframes = [t.strip() for t in args.timeframes.split(",")]
    
    sync_historical_data(symbols, timeframes, args.days)


if __name__ == "__main__":
    main()

