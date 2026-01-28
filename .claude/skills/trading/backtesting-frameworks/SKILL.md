# Backtesting Frameworks Skill

Prevent common backtesting biases and implement robust validation.

## When to Use
- Designing backtesting pipelines
- Reviewing backtest results for validity
- Identifying potential biases
- Validating QML pattern detection

## Common Biases

### 1. Look-Ahead Bias

Using future information that wouldn't be available at trade time.

```python
# BAD: Using future data
def detect_pattern_bad(df, i):
    # Using bars AFTER detection point
    future_volatility = df['close'].iloc[i:i+20].std()  # LOOK-AHEAD!
    return future_volatility > threshold

# GOOD: Only use past data
def detect_pattern_good(df, i):
    # Only use bars BEFORE or AT detection point
    past_volatility = df['close'].iloc[max(0,i-20):i+1].std()
    return past_volatility > threshold
```

**Check Script:**
```python
def check_lookahead_bias(feature_func, df):
    """Verify feature doesn't use future data."""
    # Compute feature on full data
    full_result = feature_func(df, len(df)-1)

    # Compute on truncated data (shouldn't change)
    truncated = df.iloc[:-50]
    partial_result = feature_func(truncated, len(truncated)-1)

    if full_result != partial_result:
        print("WARNING: Look-ahead bias detected!")
        return False
    return True
```

### 2. Survivorship Bias

Only testing on assets that still exist today.

```python
# BAD: Only current top coins
symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]  # Survivors only

# GOOD: Include delisted/failed coins
symbols = [
    "BTCUSDT", "ETHUSDT",
    "LUNAUSDT",  # Collapsed May 2022
    "FTMUSDT",   # Dropped from top 20
    "CELOUSDT",  # Declined significantly
]

# Or use historical snapshots
def get_top_coins_at_date(date):
    """Get top N coins by market cap at historical date."""
    historical_rankings = load_historical_rankings()
    return historical_rankings[date][:30]
```

### 3. Selection Bias

Cherry-picking time periods or parameters.

```python
# BAD: Test only on 2021 bull market
df = df["2021-01-01":"2021-12-31"]

# GOOD: Test across multiple market regimes
periods = [
    ("Bull 2020-21", "2020-10-01", "2021-11-10"),
    ("Bear 2022", "2021-11-10", "2022-11-21"),
    ("Recovery 2023", "2022-11-21", "2023-12-31"),
    ("New cycle 2024+", "2024-01-01", None),
]

for name, start, end in periods:
    results = backtest(df[start:end])
    print(f"{name}: PF={results.pf:.2f}, WR={results.wr:.1%}")
```

### 4. Data Snooping / Overfitting

Optimizing parameters on same data used for testing.

```python
# BAD: Optimize and test on same data
params = optimize(full_data)
results = backtest(full_data, params)  # SNOOPING!

# GOOD: Proper train/test split
train_data = df[:"2024-06-30"]
test_data = df["2024-07-01":]

params = optimize(train_data)
results = backtest(test_data, params)  # Unseen data
```

## Validation Framework

### Walk-Forward Analysis

```python
def walk_forward_validation(df, n_folds=5, train_pct=0.7):
    """Rolling train/test validation."""
    results = []
    fold_size = len(df) // n_folds

    for i in range(n_folds):
        # Define fold boundaries
        fold_start = i * fold_size
        fold_end = (i + 1) * fold_size
        train_end = fold_start + int(fold_size * train_pct)

        # Split data
        train = df.iloc[fold_start:train_end]
        test = df.iloc[train_end:fold_end]

        # Optimize on train
        params = optimize(train)

        # Test on held-out period
        fold_result = backtest(test, params)
        fold_result.fold = i
        fold_result.train_period = (df.index[fold_start], df.index[train_end])
        fold_result.test_period = (df.index[train_end], df.index[fold_end])
        results.append(fold_result)

    return results
```

### Purged Cross-Validation

```python
def purged_kfold(df, n_splits=5, purge_bars=10, embargo_bars=5):
    """Time-series CV with purge and embargo gaps."""
    indices = np.arange(len(df))
    fold_size = len(df) // n_splits

    for i in range(n_splits):
        test_start = i * fold_size
        test_end = min((i + 1) * fold_size, len(df))

        # Train: all data EXCEPT test + purge + embargo
        train_mask = np.ones(len(df), dtype=bool)
        # Remove test period
        train_mask[test_start:test_end] = False
        # Remove purge period (before test)
        train_mask[max(0, test_start-purge_bars):test_start] = False
        # Remove embargo period (after test)
        train_mask[test_end:min(len(df), test_end+embargo_bars)] = False

        train_idx = indices[train_mask]
        test_idx = indices[test_start:test_end]

        yield train_idx, test_idx
```

## QML-Specific Validation

```python
def validate_qml_backtest(trades: list, df: pd.DataFrame):
    """QML-specific validation checks."""
    issues = []

    # Check 1: Patterns detected on valid swings only
    for trade in trades:
        p5_idx = df.index.get_loc(trade.p5_time)
        if p5_idx >= len(df) - 5:
            issues.append(f"Pattern {trade.id} detected too close to data end")

    # Check 2: Entry after P5 (no look-ahead)
    for trade in trades:
        if trade.entry_time <= trade.p5_time:
            issues.append(f"Trade {trade.id} entry before P5 completion")

    # Check 3: SL/TP levels set at entry time only
    for trade in trades:
        entry_idx = df.index.get_loc(trade.entry_time)
        atr_at_entry = df['atr'].iloc[entry_idx]
        expected_sl_dist = atr_at_entry * trade.sl_atr_mult
        actual_sl_dist = abs(trade.entry_price - trade.stop_loss)
        if abs(actual_sl_dist - expected_sl_dist) / expected_sl_dist > 0.01:
            issues.append(f"Trade {trade.id} SL not based on entry-time ATR")

    # Check 4: No same-bar entry and exit
    for trade in trades:
        if trade.entry_time == trade.exit_time:
            issues.append(f"Trade {trade.id} entered and exited same bar")

    return issues
```

## Realistic Simulation

```python
class RealisticBacktester:
    """Backtest with realistic execution assumptions."""

    def __init__(self):
        self.slippage_bps = 5  # 5 basis points
        self.commission_bps = 10  # Maker + taker
        self.fill_delay_bars = 1  # Enter on next bar's open

    def simulate_entry(self, signal, df):
        """Realistic entry simulation."""
        signal_bar = df.index.get_loc(signal.time)
        entry_bar = signal_bar + self.fill_delay_bars

        if entry_bar >= len(df):
            return None  # Can't fill at data end

        # Enter at next bar's open with slippage
        base_price = df['open'].iloc[entry_bar]
        slippage = base_price * (self.slippage_bps / 10000)

        if signal.direction == "LONG":
            fill_price = base_price + slippage
        else:
            fill_price = base_price - slippage

        return fill_price, df.index[entry_bar]

    def apply_costs(self, pnl: float, position_value: float):
        """Apply trading costs."""
        round_trip_cost = position_value * (self.commission_bps / 10000) * 2
        return pnl - round_trip_cost
```

## Validation Checklist

| Check | How to Verify |
|-------|---------------|
| No look-ahead | Feature only uses `iloc[:i+1]` |
| Train/test split | Test data never seen during optimization |
| Survivorship | Include delisted assets |
| Realistic fills | Slippage, delay, partial fills |
| Costs included | Commission + funding + slippage |
| Multiple regimes | Bull, bear, sideways tested |
| Walk-forward | OOS results similar to IS |
