# Fetch Data

Fetch fresh price data from Binance.

## Usage
```
/fetch-data [symbol] [timeframes] [years]
```

## Examples
- `/fetch-data` - Fetch BTC/USDT 4h for 2 years
- `/fetch-data ETH/USDT` - Fetch ETH 4h for 2 years
- `/fetch-data SOL/USDT 1h,4h,1d 3` - Fetch SOL on multiple timeframes for 3 years

## Instructions

When the user invokes this skill:

1. Parse arguments:
   - symbol: Default "BTC/USDT" (use / format for ccxt)
   - timeframes: Default ["4h"], can be comma-separated
   - years: Default 2

2. Run the data fetch:
   ```python
   python -c "from src.data_engine import build_master_store; build_master_store('{symbol}', {timeframes}, years={years})"
   ```

3. Verify the parquet file was created in `data/` directory

4. Report:
   - Number of candles fetched
   - Date range
   - File location
