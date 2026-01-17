# VRD Validation Bridge Script Report

**Date:** 2026-01-07  
**Purpose:** Knowledge transfer for DeepSeek AI - VRD 2.0 bridge integration

---

## Objective

Created `cli/run_vrd_validation.py` - the bridge connecting `backtesting.py` to VRD 2.0 analytics.

---

## File Created

[cli/run_vrd_validation.py](file:///Users/hunternovotny/Desktop/QML_SYSTEM/cli/run_vrd_validation.py)

---

## Pipeline Steps

| Step | Description |
|------|-------------|
| 1 | Load OHLCV data via `load_master_data()` |
| 2 | Initialize `Backtest` with `QMLStrategy` |
| 3 | Run parameter optimization (`bt.optimize()`) |
| 4 | Final backtest with best parameters |
| 5 | Run VRD validation (Permutation, Monte Carlo, Bootstrap) |
| 6 | Generate HTML dossier report |

---

## Key Functions

### `extract_trade_returns(bt_stats)` → `np.ndarray`
Extracts percentage returns from backtesting.py stats.

### `convert_trades_for_dossier(bt_stats)` → `pd.DataFrame`
Converts trade format (renames columns, adds side/symbol).

### `create_backtest_result_dict(bt_stats, config)` → `Dict`
Creates VRD-compatible result dictionary with equity curve.

### `run_vrd_validation(symbol, timeframe, ...)` → `Dict`
Main orchestration function.

---

## Successful Test Run

```bash
python -m cli.run_vrd_validation --quick
```

**Output:**
```
Symbol:    BTC/USDT
Timeframe: 4h
# Trades:      76
Win Rate:      30.26%
Return:        -28.33%
Sharpe Ratio:  -0.17

🧪 VRD VALIDATION SUITE: FAIL
  ❌ permutation_test: FAIL (p=0.65 >= 0.05)
  ✅ monte_carlo: PASS (Risk of Ruin: 0.0%)
  ⚠️ bootstrap: WARN (Sharpe CI spans zero)

📋 Report saved: results/QMLStrategy (BTC/USDT 4h)/vrd_..._dossier.html
```

---

## Usage

```bash
# Quick mode (skip optimization)
python -m cli.run_vrd_validation --quick

# Full mode with optimization
python -m cli.run_vrd_validation

# Custom symbol
python -m cli.run_vrd_validation --symbol ETH/USDT --timeframe 4h
```

---

## Notes

- The strategy shows **NO statistical edge** with default parameters (p=0.65)
- Monte Carlo shows low ruin risk despite negative returns
- Bootstrap confidence interval spans zero = uncertain profitability
- Use `bt.optimize()` to find better parameter combinations

---

## File Reference

[run_vrd_validation.py](file:///Users/hunternovotny/Desktop/QML_SYSTEM/cli/run_vrd_validation.py)
