# Multi-Symbol Data Pipeline Refactor Report

**Date:** 2026-01-07  
**Purpose:** Knowledge transfer for DeepSeek AI - Multi-symbol support

---

## Objective

Refactored the data pipeline to support multiple cryptocurrencies (not just BTC/USDT).

---

## Files Modified

### 1. `src/data_engine.py`

#### New Helper Functions

```python
def normalize_symbol(symbol: str) -> str:
    """Convert 'BTC/USDT' -> 'BTCUSDT', 'ETH/USDT' -> 'ETHUSDT'"""
    return symbol.replace("/", "").replace("-", "").upper()

def get_symbol_data_dir(symbol: str = "BTC/USDT") -> Path:
    """Get data directory: data/processed/BTCUSDT/"""
    # Has backward compat: BTC/USDT checks legacy 'BTC' folder first
```

#### Updated Functions

| Function | Change |
|----------|--------|
| `fetch_ohlcv()` | Renamed from `fetch_btc_ohlcv`, symbol is now first parameter |
| `build_master_store()` | Added `symbol` parameter (default: `"BTC/USDT"`) |
| `load_master_data()` | Added `symbol` parameter (default: `"BTC/USDT"`) |

#### CLI Updates

```bash
# New --symbol flag
python -m src.data_engine --symbol ETH/USDT --timeframes 4h --years 2

# Default still works (BTC/USDT)
python -m src.data_engine
```

### 2. `cli/run_backtest.py`

Updated `load_data()` to construct symbol-specific paths:

```python
# Old: data/processed/BTC/4h_master.parquet
# New: data/processed/BTCUSDT/4h_master.parquet (or ETHUSDT, etc.)
```

---

## Directory Structure

**Before:**
```
data/processed/
└── BTC/
    ├── 1h_master.parquet
    └── 4h_master.parquet
```

**After:**
```
data/processed/
├── BTC/                  # Legacy (still works for BTC/USDT)
├── BTCUSDT/              # New format
├── ETHUSDT/
└── SOLUSDT/
```

---

## Backward Compatibility

✅ **Existing BTC data works** - `get_symbol_data_dir("BTC/USDT")` checks legacy `BTC/` folder first

```python
# These all work with existing data:
load_master_data('4h')                      # Default BTC/USDT
load_master_data('4h', symbol='BTC/USDT')   # Explicit BTC/USDT
```

---

## Usage Examples

```python
from src.data_engine import load_master_data, build_master_store

# Load existing BTC data
df_btc = load_master_data('4h')

# Load ETH data (after fetching)
df_eth = load_master_data('4h', symbol='ETH/USDT')

# Fetch new symbol data
build_master_store(symbol='ETH/USDT', timeframes=['4h'], years=2)
```

---

## Verification

```
=== Testing Backward Compatibility ===
  get_symbol_data_dir("BTC/USDT") -> data/processed/BTC  ✅
  load_master_data("4h") -> 10950 rows ✅
  load_master_data("4h", symbol="BTC/USDT") -> 10950 rows ✅
```

---

## File References

- [data_engine.py](file:///Users/hunternovotny/Desktop/QML_SYSTEM/src/data_engine.py)
- [run_backtest.py](file:///Users/hunternovotny/Desktop/QML_SYSTEM/cli/run_backtest.py)
