# Debugging Guide Skill

Systematic debugging workflow for trading system issues.

## When to Use
- Investigating unexpected trade outcomes
- Debugging pattern detection issues
- Troubleshooting API connectivity
- Analyzing backtest discrepancies

## Debugging Workflow

### Step 1: Reproduce the Issue

```python
# Create minimal reproduction case
def reproduce_issue():
    """Minimal case that reproduces the bug."""
    # Load specific data that caused issue
    df = pd.read_parquet("data/BTCUSDT_4h.parquet")
    df = df.loc["2026-01-15":"2026-01-20"]  # Narrow time window

    # Run detection with same params
    detector = get_detector("atr")
    signals = detector.detect(df, config)

    # Verify issue occurs
    assert len(signals) == 0, "Expected no signals but got some"
```

### Step 2: Gather Context

```python
# Debugging utilities
import json
from datetime import datetime

def debug_snapshot(context: dict, filename: str = None):
    """Save debug state for analysis."""
    snapshot = {
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "context": context
    }
    filename = filename or f"debug_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(f"logs/{filename}", "w") as f:
        json.dump(snapshot, f, indent=2, default=str)
    return filename

# Usage in detection
def detect_with_debug(df, config):
    try:
        signals = detector.detect(df, config)
        return signals
    except Exception as e:
        debug_snapshot({
            "error": str(e),
            "df_shape": df.shape,
            "df_range": [str(df.index[0]), str(df.index[-1])],
            "config": config.__dict__,
            "last_10_rows": df.tail(10).to_dict()
        })
        raise
```

### Step 3: Binary Search Isolation

```python
def binary_search_bug(df, detect_func, works_before, fails_after):
    """Find exact point where behavior changes."""
    left, right = works_before, fails_after

    while right - left > 1:
        mid = (left + right) // 2
        test_df = df.iloc[:mid]

        try:
            result = detect_func(test_df)
            works = validate_result(result)
        except:
            works = False

        if works:
            left = mid
        else:
            right = mid

    print(f"Bug introduced between row {left} and {right}")
    print(f"Last good: {df.index[left]}")
    print(f"First bad: {df.index[right]}")
    return left, right
```

## Common Trading System Bugs

### Bug: Wrong Trade Direction

```python
def debug_direction(signal, df):
    """Verify signal direction matches pattern type."""
    print(f"Signal direction: {signal.direction}")
    print(f"Pattern type: {signal.pattern_type}")

    # BULLISH QML → LONG
    # BEARISH QML → SHORT
    if signal.pattern_type == "BULLISH" and signal.direction != "LONG":
        print("BUG: Bullish pattern should be LONG")

    # Check prior trend
    p1_idx = df.index.get_loc(signal.p1_time)
    prior_bars = df.iloc[max(0, p1_idx-20):p1_idx]
    trend = "DOWN" if prior_bars['close'].iloc[-1] < prior_bars['close'].iloc[0] else "UP"
    print(f"Prior trend: {trend}")
```

### Bug: Stop Loss Hit Too Early

```python
def debug_sl_hit(trade, df):
    """Analyze why SL was hit."""
    entry_idx = df.index.get_loc(trade.entry_time)
    exit_idx = df.index.get_loc(trade.exit_time)
    trade_bars = df.iloc[entry_idx:exit_idx+1]

    print(f"Entry: {trade.entry_price}")
    print(f"SL: {trade.stop_loss}")
    print(f"Direction: {trade.direction}")

    if trade.direction == "LONG":
        min_low = trade_bars['low'].min()
        print(f"Minimum low during trade: {min_low}")
        if min_low < trade.stop_loss:
            print(f"SL correctly hit at bar {trade_bars['low'].idxmin()}")
        else:
            print("BUG: SL hit but price never reached SL level")

    # Check for gap
    for i in range(1, len(trade_bars)):
        if trade_bars['open'].iloc[i] < trade_bars['low'].iloc[i-1]:
            print(f"GAP DOWN at {trade_bars.index[i]}")
```

### Bug: Unrealistic Backtest Results

```python
def sanity_check_results(trades: list):
    """Flag suspicious backtest results."""
    issues = []

    # Check 1: Win rate too high
    win_rate = sum(1 for t in trades if t.pnl > 0) / len(trades)
    if win_rate > 0.75:
        issues.append(f"Win rate {win_rate:.1%} suspiciously high")

    # Check 2: No losing streaks
    max_consec_loss = 0
    current_streak = 0
    for t in trades:
        if t.pnl < 0:
            current_streak += 1
            max_consec_loss = max(max_consec_loss, current_streak)
        else:
            current_streak = 0
    if max_consec_loss < 3 and len(trades) > 50:
        issues.append(f"Max losing streak only {max_consec_loss} - look-ahead bias?")

    # Check 3: Average win too large
    wins = [t.pnl_r for t in trades if t.pnl > 0]
    if wins and sum(wins)/len(wins) > 5:
        issues.append(f"Avg win {sum(wins)/len(wins):.1f}R unusually high")

    # Check 4: Profit factor unrealistic
    gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
    gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    if pf > 4:
        issues.append(f"PF {pf:.1f} may indicate data leakage")

    return issues
```

## Debugging Tools

### Interactive Debugger

```python
def debug_breakpoint():
    """Drop into debugger at this point."""
    import pdb; pdb.set_trace()
    # Or for IPython: import IPython; IPython.embed()

# Conditional breakpoint
def detect_signal(df, config):
    signal = _internal_detect(df, config)
    if signal and signal.validity_score < 0.5:
        import pdb; pdb.set_trace()  # Break on low quality signals
    return signal
```

### Visual Debugging

```python
def plot_debug_chart(df, signal, filename="debug_chart.html"):
    """Generate debug chart with annotations."""
    import plotly.graph_objects as go

    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close']
    )])

    # Add signal points
    if signal:
        for i, (time, price) in enumerate(signal.swing_points):
            fig.add_annotation(x=time, y=price, text=f"P{i+1}")

        # Add entry/SL/TP lines
        fig.add_hline(y=signal.entry, line_color="cyan", annotation_text="Entry")
        fig.add_hline(y=signal.stop_loss, line_color="red", annotation_text="SL")
        fig.add_hline(y=signal.take_profit, line_color="green", annotation_text="TP")

    fig.write_html(filename)
    print(f"Debug chart saved to {filename}")
```

### Log Analysis

```bash
# Find all errors in last hour
grep '"level":"ERROR"' logs/trading.jsonl | tail -100

# Find specific correlation ID
grep 'a1b2c3d4' logs/trading.jsonl

# Count errors by type
jq -r 'select(.level=="ERROR") | .error_type' logs/trading.jsonl | sort | uniq -c

# Find slow operations (>5s)
jq 'select(.duration_ms > 5000)' logs/trading.jsonl
```

## Debugging Checklist

| Check | Command/Action |
|-------|----------------|
| Data loaded correctly? | `print(df.shape, df.index[[0,-1]])` |
| Config as expected? | `print(config.__dict__)` |
| Timezone issues? | `print(df.index.tz, signal.time.tz)` |
| NaN values? | `print(df.isna().sum())` |
| Duplicate indices? | `print(df.index.duplicated().sum())` |
| Sorted correctly? | `assert df.index.is_monotonic_increasing` |
| Price scale right? | `print(df['close'].describe())` |
