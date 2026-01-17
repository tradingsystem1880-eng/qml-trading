# QML Backtesting.py Strategy Implementation Report

**Date:** 2026-01-07  
**Author:** Antigravity (AI Agent)  
**Purpose:** Knowledge transfer documentation for continuation by DeepSeek AI

---

## Executive Summary

Adapted the existing QML (Quasimodo) pattern detection logic from `src/detection/v2_atr.py` into a self-contained `backtesting.py` Strategy class. This replaces the custom backtest engine approach with a standard framework compatible with `Backtest.optimize()`.

---

## Files Created/Modified

### New File: `src/strategies/qml_backtestingpy.py`

A 330-line Python module containing:

| Component | Description |
|-----------|-------------|
| `atr()` function | Pure numpy ATR calculation (no TA-Lib dependency) |
| `QMLStrategy` class | `backtesting.Strategy` subclass with stateful swing detection |

### Modified: `requirements.txt`

Added dependency:
```
Backtesting>=0.3.3
```

---

## Technical Implementation

### Strategy Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     QMLStrategy                              │
├─────────────────────────────────────────────────────────────┤
│ Class Parameters (optimizable):                              │
│   atr_period=14, min_depth_ratio=0.5, max_depth_ratio=3.0   │
│   stop_loss_atr=1.5, take_profit_rr=3.0, shoulder_tolerance │
├─────────────────────────────────────────────────────────────┤
│ init():                                                      │
│   - Pre-computes ATR series via self.I(atr, ...)            │
│   - Initializes stateful tracking vars:                      │
│     _up_move, _pend_max, _pend_min, _extremes               │
├─────────────────────────────────────────────────────────────┤
│ next():                                                      │
│   - ATR Directional Change swing detection                   │
│   - Confirms swing HIGH when price drops 1 ATR from peak    │
│   - Confirms swing LOW when price rises 1 ATR from trough   │
│   - Calls _check_pattern() when new extreme confirmed       │
├─────────────────────────────────────────────────────────────┤
│ _check_pattern():                                            │
│   - Validates last 3 extremes form QML pattern              │
│   - Bullish: HIGH → LOW → HIGH (head is the low)            │
│   - Bearish: LOW → HIGH → LOW (head is the high)            │
│   - Executes self.buy()/self.sell() with SL/TP              │
└─────────────────────────────────────────────────────────────┘
```

### Core Logic Adapted From

The original detection logic in `src/detection/v2_atr.py`:
- `ATRDirectionalChange` class → State variables in `init()`, logic in `next()`
- `LocalExtreme` dataclass → Simple tuples `(type, index, price)`
- `_validate_bullish_pattern()` / `_validate_bearish_pattern()` → Methods on strategy

### Critical Design Decisions

1. **No Look-Ahead Bias**: The `next()` method only accesses data up to current bar via `self.data.High[-1]` etc.

2. **Stateful Tracking**: Swing detection state persists across bars:
   ```python
   self._up_move = True      # Direction being tracked
   self._pend_max = np.nan   # Pending swing high price
   self._pend_min = np.nan   # Pending swing low price
   self._extremes = []       # Confirmed (type, index, price) tuples
   ```

3. **Pattern Deduplication**: Uses `_traded_patterns` set to avoid re-trading same pattern.

4. **SL/TP Sanity Checks**: Added validation to ensure:
   - Long trades: `stop_loss < entry < take_profit`
   - Short trades: `take_profit < entry < stop_loss`

---

## Verification Results

### Basic Backtest (4h BTC data, ~5 years)

```
Return [%]: -74.53
Sharpe Ratio: -0.51
Win Rate [%]: 35.53
# Trades: 273
Max. Drawdown [%]: -78.54
```

### With Parameter Optimization

```python
bt.optimize(atr_period=range(12,18,2), min_depth_ratio=[0.5, 1.0])
```

```
Best ATR Period: 14
Best Min Depth Ratio: 1.0
Return [%]: +16.67
# Trades: 338
```

---

## Usage Example

```python
from backtesting import Backtest
from src.strategies.qml_backtestingpy import QMLStrategy
from src.data_engine import load_master_data

# Load data (requires time→Date rename for backtesting.py)
df = load_master_data(timeframe='4h')
df = df.rename(columns={'time': 'Date'}).set_index('Date')

# Run backtest
bt = Backtest(df, QMLStrategy, cash=1000000, commission=0.001)
stats = bt.run()

# Optimize parameters
stats = bt.optimize(
    atr_period=range(10, 20),
    min_depth_ratio=[0.5, 0.75, 1.0, 1.5],
    stop_loss_atr=[1.0, 1.5, 2.0],
    maximize='Sharpe Ratio'
)

# Generate HTML report
bt.plot(filename='qml_backtest.html')
```

---

## Known Issues / Future Work

1. **Margin Warnings**: With default $100k cash and BTC prices ~$37k+, many trades get rejected for insufficient margin. Use higher cash or enable margin trading.

2. **Poor Default Performance**: Default parameters yield -74% return. Optimization is required.

3. **Short Selling**: Strategy supports both long and short trades. If exchange doesn't support shorts, modify `_check_pattern()` to skip bearish patterns.

4. **Parameter Sensitivity**: The `shoulder_tolerance` and `max_depth_ratio` parameters significantly affect trade frequency.

---

## Relationship to Existing Codebase

| New Component | Replaces/Complements |
|---------------|---------------------|
| `QMLStrategy` | Alternative to `BacktestEngine` + `QMLDetector` pipeline |
| `atr()` function | Inline version of `_calculate_atr()` from `BaseDetector` |
| Stateful swing tracking | Adapted from `ATRDirectionalChange` class |

The existing `src/detection/v2_atr.py` and `src/backtest/engine.py` remain unchanged and can still be used independently.

---

## File References

- Strategy: [qml_backtestingpy.py](file:///Users/hunternovotny/Desktop/QML_SYSTEM/src/strategies/qml_backtestingpy.py)
- Original detector: [v2_atr.py](file:///Users/hunternovotny/Desktop/QML_SYSTEM/src/detection/v2_atr.py)
- Data loader: [data_engine.py](file:///Users/hunternovotny/Desktop/QML_SYSTEM/src/data_engine.py)
