# Pandas Analysis Skill

Time series and financial data manipulation with Pandas.

## When to Use
- OHLCV data manipulation
- Time series resampling
- Financial calculations
- Data cleaning and preparation

## OHLCV Data Handling

### Loading Data

```python
import pandas as pd

# Load parquet (preferred format)
df = pd.read_parquet("data/BTCUSDT_4h.parquet")

# Ensure datetime index
df.index = pd.to_datetime(df.index)
df = df.sort_index()

# Verify OHLCV columns
required_cols = ['open', 'high', 'low', 'close', 'volume']
assert all(col in df.columns for col in required_cols)
```

### Resampling Timeframes

```python
def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Resample OHLCV data to different timeframe.

    Args:
        df: DataFrame with OHLCV columns
        timeframe: Target timeframe ('1h', '4h', '1d', '1w')
    """
    return df.resample(timeframe).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

# Example: Convert 1h to 4h
df_4h = resample_ohlcv(df_1h, '4h')
```

### Technical Indicators

```python
def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add common technical indicators."""
    df = df.copy()

    # ATR (Average True Range)
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()

    # SMA
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()

    # EMA
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df['bb_mid'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']

    return df
```

### Returns Calculation

```python
def calculate_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add various return calculations."""
    df = df.copy()

    # Simple returns
    df['return'] = df['close'].pct_change()

    # Log returns (better for compounding)
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))

    # Multi-period returns
    df['return_5'] = df['close'].pct_change(5)
    df['return_20'] = df['close'].pct_change(20)

    # Cumulative returns
    df['cum_return'] = (1 + df['return']).cumprod() - 1

    return df
```

## Time Series Operations

### Rolling Window Analysis

```python
def rolling_analysis(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Calculate rolling statistics."""
    df = df.copy()

    # Rolling mean and std
    df['rolling_mean'] = df['close'].rolling(window).mean()
    df['rolling_std'] = df['close'].rolling(window).std()

    # Rolling min/max
    df['rolling_high'] = df['high'].rolling(window).max()
    df['rolling_low'] = df['low'].rolling(window).min()

    # Rolling percentile rank
    df['pct_rank'] = df['close'].rolling(window).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1]
    )

    return df
```

### Shift and Lag Operations

```python
# Past values (lag)
df['close_prev'] = df['close'].shift(1)   # Previous close
df['close_5ago'] = df['close'].shift(5)   # 5 bars ago

# Future values (lead) - ONLY for labeling, not features!
df['close_next'] = df['close'].shift(-1)  # Next close (look-ahead!)

# Difference from past
df['close_change'] = df['close'] - df['close'].shift(1)
df['close_change_5'] = df['close'] - df['close'].shift(5)
```

### Date/Time Features

```python
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features."""
    df = df.copy()

    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['day_of_month'] = df.index.day
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter

    # Trading session (for crypto)
    df['session'] = pd.cut(
        df.index.hour,
        bins=[0, 8, 16, 24],
        labels=['asia', 'europe', 'us'],
        ordered=False
    )

    # Is weekend
    df['is_weekend'] = df.index.dayofweek >= 5

    return df
```

## Data Quality

### Missing Data Handling

```python
def check_and_fix_gaps(df: pd.DataFrame, freq: str = '4h') -> pd.DataFrame:
    """Detect and handle data gaps."""
    # Check for gaps
    expected_index = pd.date_range(df.index[0], df.index[-1], freq=freq)
    missing = expected_index.difference(df.index)

    if len(missing) > 0:
        print(f"Found {len(missing)} gaps")

        # Option 1: Forward fill (limited)
        df = df.reindex(expected_index).ffill(limit=3)

        # Option 2: Drop rows with gaps
        # df = df[df.index.isin(expected_index)]

    return df

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle NaN values in OHLCV data."""
    # Check for NaNs
    nan_counts = df.isna().sum()
    if nan_counts.any():
        print(f"NaN counts:\n{nan_counts}")

    # Forward fill prices (reasonable for short gaps)
    price_cols = ['open', 'high', 'low', 'close']
    df[price_cols] = df[price_cols].ffill(limit=2)

    # Fill volume with 0 or median
    df['volume'] = df['volume'].fillna(0)

    return df
```

### Outlier Detection

```python
def detect_outliers(df: pd.DataFrame, cols: list = None, z_threshold: float = 4.0) -> pd.DataFrame:
    """Flag outliers using z-score."""
    cols = cols or ['close', 'volume']
    df = df.copy()

    for col in cols:
        z_scores = (df[col] - df[col].mean()) / df[col].std()
        df[f'{col}_outlier'] = abs(z_scores) > z_threshold

    return df

def remove_price_spikes(df: pd.DataFrame, max_change_pct: float = 0.2) -> pd.DataFrame:
    """Remove unrealistic price spikes."""
    df = df.copy()

    pct_change = df['close'].pct_change().abs()
    spikes = pct_change > max_change_pct

    if spikes.any():
        print(f"Found {spikes.sum()} price spikes > {max_change_pct:.0%}")
        # Option: Interpolate or forward fill
        df.loc[spikes, ['open', 'high', 'low', 'close']] = np.nan
        df = df.ffill()

    return df
```

## Groupby Operations

### Per-Symbol Analysis

```python
def analyze_by_symbol(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate trade statistics by symbol."""
    return trades_df.groupby('symbol').agg({
        'pnl': ['count', 'sum', 'mean'],
        'pnl_r': ['mean', 'std'],
        'duration_bars': 'mean'
    }).round(2)

# Flatten column names
result.columns = ['_'.join(col).strip() for col in result.columns]
```

### Per-Period Analysis

```python
def monthly_performance(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate monthly P&L."""
    trades_df['month'] = pd.to_datetime(trades_df['exit_time']).dt.to_period('M')

    return trades_df.groupby('month').agg({
        'pnl': 'sum',
        'pnl_r': ['count', 'sum', 'mean'],
    })
```

## Merge Operations

### Join Multiple Timeframes

```python
def merge_timeframes(df_1h: pd.DataFrame, df_4h: pd.DataFrame) -> pd.DataFrame:
    """
    Merge 1h data with 4h indicators.
    4h values are forward-filled to 1h bars.
    """
    # Add suffix to 4h columns
    df_4h = df_4h.add_suffix('_4h')

    # Merge on index (1h gets latest 4h value)
    merged = df_1h.merge(
        df_4h,
        left_index=True,
        right_index=True,
        how='left'
    ).ffill()

    return merged
```

### Join Trades with Price Data

```python
def enrich_trades(trades_df: pd.DataFrame, ohlcv_df: pd.DataFrame) -> pd.DataFrame:
    """Add market context to trades."""
    # Get ATR at entry time
    trades_df = trades_df.merge(
        ohlcv_df[['atr', 'rsi', 'volume']].rename(columns=lambda x: f'{x}_at_entry'),
        left_on='entry_time',
        right_index=True,
        how='left'
    )
    return trades_df
```

## Performance Tips

```python
# Use categorical for repeated strings
df['symbol'] = df['symbol'].astype('category')
df['direction'] = df['direction'].astype('category')

# Use float32 for memory savings
df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].astype('float32')

# Use query for filtering (faster on large DataFrames)
df.query('close > 50000 and volume > 1000')

# Use vectorized operations
# ❌ Slow
for i in range(len(df)):
    df.iloc[i, col_idx] = df.iloc[i, col_idx] * 2

# ✅ Fast
df['col'] = df['col'] * 2

# Use .to_numpy() for numerical operations
prices = df['close'].to_numpy()
returns = np.diff(prices) / prices[:-1]
```
