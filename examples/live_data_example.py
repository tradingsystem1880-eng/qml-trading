"""
Live Data Fetcher Example
=========================
Demonstrates how to use the new LiveDataFetcher for clean, fast data access.
"""

from src.data.live_fetcher import LiveDataFetcher, create_live_fetcher


def example_basic_usage():
    """Basic usage - fetch data with automatic caching."""
    print("\n" + "="*60)
    print("Example 1: Basic Usage")
    print("="*60)
    
    # Create fetcher (updates every 15 minutes by default)
    fetcher = create_live_fetcher()
    
    # Fetch BTC data
    df = fetcher.get_ohlcv("BTC/USDT", "4h", limit=200)
    
    print(f"\n✅ Fetched {len(df)} candles for BTC/USDT 4h")
    print(f"\nLatest price: ${df['close'].iloc[-1]:,.2f}")
    print(f"Date range: {df['time'].min()} to {df['time'].max()}")
    
    # Fetch again - will use cache (instant)
    df2 = fetcher.get_ohlcv("BTC/USDT", "4h", limit=200)
    print(f"\n✅ Second fetch used cache (instant)")
    
    # View cache stats
    stats = fetcher.get_cache_info()
    print(f"\n📊 Cache Stats:")
    print(f"  Hit rate: {stats['hit_rate']:.1%}")
    print(f"  Cached queries: {stats['cached_queries']}")
    print(f"  TTL: {stats['ttl_minutes']} minutes")


def example_multiple_symbols():
    """Fetch multiple symbols efficiently."""
    print("\n" + "="*60)
    print("Example 2: Multiple Symbols")
    print("="*60)
    
    fetcher = create_live_fetcher()
    
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    data = fetcher.get_multiple(symbols, timeframe="1h", limit=100)
    
    print(f"\n✅ Fetched data for {len(symbols)} symbols:")
    for symbol, df in data.items():
        latest_price = df['close'].iloc[-1]
        print(f"  {symbol}: ${latest_price:,.2f} ({len(df)} candles)")


def example_with_storage():
    """Use optional SQLite storage for persistence."""
    print("\n" + "="*60)
    print("Example 3: With SQLite Storage")
    print("="*60)
    
    # Enable storage for historical persistence
    fetcher = LiveDataFetcher(
        exchange_id="binance",
        cache_ttl_minutes=10,
        use_storage=True  # Enable SQLite
    )
    
    df = fetcher.get_ohlcv("BTC/USDT", "4h", limit=500)
    
    print(f"\n✅ Fetched {len(df)} candles")
    print(f"✅ Data persisted to SQLite database")
    print(f"  Location: {fetcher.storage_path}")


def example_force_refresh():
    """Force refresh to bypass cache."""
    print("\n" + "="*60)
    print("Example 4: Force Refresh")
    print("="*60)
    
    fetcher = create_live_fetcher(cache_ttl_minutes=15)
    
    # Normal fetch (cached)
    df1 = fetcher.get_ohlcv("BTC/USDT", "4h")
    
    # Force refresh (bypass cache)
    df2 = fetcher.get_ohlcv("BTC/USDT", "4h", force_refresh=True)
    
    print(f"\n✅ Forced refresh completed")
    print(f"  Cached price: ${df1['close'].iloc[-1]:,.2f}")
    print(f"  Fresh price: ${df2['close'].iloc[-1]:,.2f}")


def example_custom_ttl():
    """Custom cache TTL for different use cases."""
    print("\n" + "="*60)
    print("Example 5: Custom Cache TTL")
    print("="*60)
    
    # Frequent updates (5 minutes)
    fast_fetcher = create_live_fetcher(cache_ttl_minutes=5)
    
    # Infrequent updates (30 minutes)
    slow_fetcher = create_live_fetcher(cache_ttl_minutes=30)
    
    print("\n✅ Created fetchers with different TTL:")
    print(f"  Fast: {fast_fetcher.cache_ttl.total_seconds() / 60}min TTL")
    print(f"  Slow: {slow_fetcher.cache_ttl.total_seconds() / 60}min TTL")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 LIVE DATA FETCHER EXAMPLES")
    print("="*60)
    
    # Run examples
    example_basic_usage()
    example_multiple_symbols()
    example_force_refresh()
    example_custom_ttl()
    
    # Optional: Uncomment to test SQLite storage
    # example_with_storage()
    
    print("\n" + "="*60)
    print("✅ All examples completed!")
    print("="*60 + "\n")
