# Data Migration and Deterministic Paths Report

**Date:** 2026-01-07  
**Purpose:** Knowledge transfer for DeepSeek AI - Migration to deterministic paths

---

## Problem Fixed

The previous multi-symbol implementation had non-deterministic fallback logic:
```python
# BAD: Fallback logic makes data source unpredictable
if normalized == 'BTCUSDT':
    legacy_path = DATA_DIR_BASE / 'BTC'
    if legacy_path.exists():
        return legacy_path  # Sometimes this, sometimes new path
```

---

## Solution: One-Time Migration + Deterministic Paths

### 1. Created Migration Script

**File:** `scripts/migrate_data_to_symbol_folders.py`

```bash
# Preview changes
python scripts/migrate_data_to_symbol_folders.py --dry-run

# Run migration
python scripts/migrate_data_to_symbol_folders.py
```

**Migration Result:**
```
Files to migrate (2):
  • 4h_master.parquet (0.6 MB)
  • 1h_master.parquet (2.4 MB)

✅ MIGRATION COMPLETE!
New data location: data/processed/BTCUSDT
```

### 2. Fixed `get_symbol_data_dir()` - Now Deterministic

```python
def get_symbol_data_dir(symbol: str = "BTC/USDT") -> Path:
    """DETERMINISTIC: Always returns normalized path, no fallback."""
    normalized = normalize_symbol(symbol)
    return DATA_DIR_BASE / normalized  # Always BTCUSDT, never BTC
```

### 3. Improved Error Messages

```python
# Before (unhelpful)
raise FileNotFoundError("Master data not found: ...")

# After (actionable)
raise FileNotFoundError(
    f"Data for {symbol} not found.\n"
    f"Expected directory: {data_dir}\n"
    f"\n"
    f"To fetch this data, run:\n"
    f"  python -m src.data_engine --symbol {symbol} --timeframes {timeframe}"
)
```

---

## Directory Structure After Migration

```
data/processed/
├── BTCUSDT/              # ← Migrated from BTC/
│   ├── 1h_master.parquet
│   └── 4h_master.parquet
├── ETHUSDT/              # ← Future symbols
└── SOLUSDT/
```

Note: Legacy `BTC/` folder was **deleted** after migration.

---

## Verification

```
get_symbol_data_dir("BTC/USDT") -> data/processed/BTCUSDT  ✅
load_master_data("4h") -> 10950 rows  ✅

Error message for missing symbol:
  "Data for ETH/USDT not found.
   Expected directory: data/processed/ETHUSDT
   To fetch this data, run:
     python -m src.data_engine --symbol ETH/USDT --timeframes 4h"
```

---

## Philosophy Applied

> "Switching from single-asset with hardcoded paths to multi-asset with explicit, versioned contracts. No going back."

- **No fallback logic** = Predictable behavior
- **One-time migration** = Clean slate for new format
- **Helpful errors** = Users know exactly how to fix issues

---

## Files Reference

- Migration script: [migrate_data_to_symbol_folders.py](file:///Users/hunternovotny/Desktop/QML_SYSTEM/scripts/migrate_data_to_symbol_folders.py)
- Updated: [data_engine.py](file:///Users/hunternovotny/Desktop/QML_SYSTEM/src/data_engine.py)
