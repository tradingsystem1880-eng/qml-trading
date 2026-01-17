# QMLStrategy v2.0 Rebuild Report

**Date:** 2026-01-07  
**Purpose:** Knowledge transfer for DeepSeek AI - Strategy paradigm correction

---

## Problem Fixed

The previous `QMLStrategy` had incorrect position tracking:
- Used manual `_traded_patterns` set instead of framework's `self.position`
- Trades were opening but not being recorded as closed
- Result: `# Trades: 0` despite equity changes

---

## Solution: Rebuilt Strategy

Complete rewrite of `src/strategies/qml_backtestingpy.py`:

### Key Changes

| Before | After |
|--------|-------|
| Manual `_traded_patterns` tracking | Uses `if self.position: return` |
| `take_profit_rr = 3.0` | `take_profit_rr = 2.0` (faster closes) |
| Custom `verbose` flag | Removed |
| Complex position management | Framework handles via SL/TP |

### Core Logic Fix

```python
def next(self):
    # Skip if already in a position (framework manages exit)
    if self.position:
        return
    
    # ... pattern detection ...
    
    # Execute trade - framework tracks position automatically
    self.buy(sl=sl, tp=tp)  # or self.sell()
```

---

## Verification Results

```
Duration:        1824 days
# Trades:        76          ← NOW NON-ZERO!
Win Rate:        30.26%
Return:          -28.33%
Max Drawdown:    -56.59%
Sharpe Ratio:    -0.17
Profit Factor:   0.93
```

Sample trades:
```
                   EntryTime  Size  EntryPrice  ExitPrice       PnL
71 2024-05-04 20:00:00+00:00     1    63921.17   54963.48  -9076.57
72 2024-07-11 20:00:00+00:00     1    57378.00   61338.51   3841.79
73 2024-07-16 12:00:00+00:00     1    63746.11   55562.28  -8303.14
74 2024-08-09 04:00:00+00:00     1    61358.39   80161.61  18661.70
75 2025-03-11 00:00:00+00:00    -1    78595.86   87347.86  -8917.95
```

---

## Usage

```python
from backtesting import Backtest
from src.strategies.qml_backtestingpy import QMLStrategy, prepare_backtesting_data
from src.data_engine import load_master_data

df = load_master_data(timeframe='4h')
df = prepare_backtesting_data(df)

bt = Backtest(df, QMLStrategy, cash=100000, commission=0.001)
stats = bt.run()
print(stats)
```

---

## File Reference

[qml_backtestingpy.py](file:///Users/hunternovotny/Desktop/QML_SYSTEM/src/strategies/qml_backtestingpy.py)

---

## Notes for Future Work

1. **Win rate is low (30%)** - Parameters need optimization
2. **Return is negative** - Default params aren't profitable on BTC 4h
3. **Use `bt.optimize()`** to find better parameter combinations
