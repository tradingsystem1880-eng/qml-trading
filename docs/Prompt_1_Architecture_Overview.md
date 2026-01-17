# Prompt 1 - Architecture Overview

> **Purpose**: Comprehensive analysis of QML_SYSTEM codebase for AI systems to understand data flow, fragility points, performance bottlenecks, and strategy organization.
> **Generated**: 2026-01-07
> **Target Audience**: DeepSeek AI / Advanced AI Coding Assistants

---

## 1. Main Data Flow: Loading → Detection → Signal Generation

```
┌─────────────────────────────────────────────────────────────────────┐
│  CLI Entry Point: python -m cli.run_backtest                       │
└─────────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│  📥 DATA LOADING (cli/run_backtest.py → load_data())               │
│  - Reads parquet: data/processed/BTC/4h_master.parquet             │
│  - Standardizes column names (Open→open, High→high, etc.)          │
│  - Filters by date range                                           │
│  - Returns: pd.DataFrame with [time, open, high, low, close, volume]│
└─────────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│  🩺 DATA VALIDATION (src/data/integrity.py → DataValidator)        │
│  - Checks for gaps, null values, schema compliance                 │
│  - Health report: pass/warn/fail                                   │
└─────────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│  🧠 DETECTOR FACTORY (src/detection/__init__.py → get_detector())  │
│  - Selects detector based on config: "atr" → ATRDetector           │
│  - Injects config (min_validity_score, atr_lookback, etc.)         │
└─────────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│  🔍 PATTERN DETECTION (src/detection/v2_atr.py)                    │
│                                                                    │
│  ATRDirectionalChange.update() - Called bar-by-bar:                │
│    1. Calculate rolling ATR (14-period)                            │
│    2. Track pending maximum/minimum prices                         │
│    3. Confirm swing when price reverses by 1 ATR                   │
│    4. Return LocalExtreme on confirmation                          │
│                                                                    │
│  ATRDetector._find_patterns_from_extremes():                       │
│    1. Triggered when 3+ extremes confirmed                         │
│    2. Bullish: HIGH → LOW → HIGH pattern                           │
│    3. Bearish: LOW → HIGH → LOW pattern                            │
│    4. Validate: shoulder symmetry (<10%), head depth (0.5-3.0 ATR) │
│    5. Calculate validity_score (0.0-1.0)                           │
│                                                                    │
│  Output: List[Signal] with entry_price, stop_loss, take_profit     │
└─────────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│  ⚙️ BACKTEST ENGINE (cli/run_backtest.py → BacktestEngine.run())   │
│  - Iterates bars chronologically                                   │
│  - Matches signals to bar timestamps                               │
│  - Opens trades with slippage modeling                             │
│  - Checks SL/TP on each bar (high/low wicks)                       │
│  - Tracks equity curve                                             │
│                                                                    │
│  Output: Dict with trades, metrics, equity_curve                   │
└─────────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│  📼 FLIGHT RECORDER                                                │
│  - ExperimentLogger → SQLite (results/experiments.db)              │
│  - DossierGenerator → HTML (results/atr/{run_id}_dossier.html)     │
│  - Optional: XGBoostPredictor training                             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Most Complex/Fragile Parts

### 2.1 High-Risk Components

| Component | Location | Risk Level | Issue |
|-----------|----------|------------|-------|
| **ATRDirectionalChange** | `src/detection/v2_atr.py:80-266` | 🔴 HIGH | Mutable state (`_up_move`, `_pend_max`, `_pend_min`) is complex. Off-by-one errors in index handling are easy to introduce. Rolling ATR calculation (lines 184-211) manually manages state. |
| **Pattern Matching Logic** | `src/detection/v2_atr.py:408-472` | 🔴 HIGH | `_find_patterns_from_extremes()` relies on exact ordering of `LocalExtreme` objects. If extremes don't arrive in strict alternating HIGH/LOW order, patterns can be missed. |
| **Signal-to-Bar Matching** | `cli/run_backtest.py:124-130` | 🟡 MEDIUM | Uses timestamp equality for matching signals to bars. Timezone mismatches or microsecond differences will silently drop signals. |
| **Duplicate Backtest Engines** | `cli/run_backtest.py` vs `src/backtest/engine.py` | 🟡 MEDIUM | Two separate `BacktestEngine` implementations exist with different APIs (`Signal` vs `QMLPattern`). Risk of drift and inconsistent behavior. |
| **Column Name Standardization** | `cli/run_backtest.py:434-449` | 🟡 MEDIUM | Hardcoded column mapping is fragile if data sources change format. |

### 2.2 Fragile State Machine Pattern

```python
# v2_atr.py:223-238 - State machine is easy to corrupt
if self._up_move:
    if high[i] > self._pend_max:
        self._pend_max = high[i]  # Update pending
        self._pend_max_i = i
    elif low[i] < self._pend_max - atr:
        new_extreme = self._create_extreme(...)  # Confirm
        self._up_move = False  # State flip!
```

---

## 3. Performance Bottlenecks in Backtesting

| Bottleneck | Location | Impact | Root Cause |
|------------|----------|--------|------------|
| **Bar-by-bar iteration** | `cli/run_backtest.py:133-157` | 🔴 HIGH | Python `for idx, row in df.iterrows()` is extremely slow for large datasets. Should use vectorized operations. |
| **Signal map lookup** | `cli/run_backtest.py:148` | 🟡 MEDIUM | `if bar_time in signal_map` creates a dict lookup per bar - O(n) checks for n bars. Could use merge/join or sorted bisect. |
| **ATR recalculation** | `src/detection/v2_atr.py:362-363` | 🟡 MEDIUM | `full_atr = self._calculate_atr(...)` computes ATR for entire dataset, then `ATRDirectionalChange.update()` recalculates it bar-by-bar redundantly. |
| **Window DataFrame copy** | `src/detection/v2_atr.py:384-385` | 🟡 MEDIUM | `window_df = df.iloc[...].copy().reset_index(drop=True)` creates new DataFrame on every extreme - expensive for large data. |
| **Pattern deduplication** | `src/detection/v2_atr.py:400-404` | 🟢 LOW | String key generation + set lookup is efficient, but could use tuple keys. |

### Estimated Performance Impact

- **Dataset size**: 4 years of 4h data ≈ 8,760 bars
- **Current runtime**: ~2-5 seconds for full backtest
- **With vectorization**: <0.5 seconds achievable

---

## 4. Strategy Logic Organization & Parameterization

### 4.1 Configuration Hierarchy

```
YAML config (config/default.yaml)
       ↓
CLI args (--detector, --min-validity, etc.)
       ↓
Config dataclass (BacktestConfig, ATRDetectorConfig)
       ↓
Detector/Engine instances
```

### 4.2 YAML Configuration (Single Source of Truth)

```yaml
# config/default.yaml
detection:
  method: atr_directional_change
  atr_period: 14
  qml:
    min_depth_ratio: 0.5
    max_depth_ratio: 1.0

backtest:
  risk:
    stop_loss_atr_mult: 1.5
    take_profit_atr_mult: 3.0
```

### 4.3 Code-Side Dataclasses

```python
# src/detection/v2_atr.py:34-56
@dataclass
class ATRDetectorConfig(DetectorConfig):
    name: str = "atr_directional_change"
    version: str = "2.0.0"
    atr_lookback: int = 14
    min_head_depth_atr: float = 0.5
    max_head_depth_atr: float = 3.0

# cli/run_backtest.py:45-73  
@dataclass
class BacktestConfig:
    detector_method: str = "atr"
    min_validity_score: float = 0.7
    commission_pct: float = 0.1
```

### 4.4 Current Issues with Parameterization

1. **Scattered defaults**: Some defaults in YAML, others in dataclasses - not fully DRY
2. **No validation**: Parameters aren't range-checked (e.g., `min_validity_score > 1.0` would pass)
3. **Manual wiring**: CLI must manually map args to config fields (lines 631-641)

---

## 5. Key File Reference

| When you need to... | Look at... |
|---------------------|------------|
| Run a backtest | `cli/run_backtest.py` |
| Add a new detector | `src/detection/base.py` → implement `BaseDetector` |
| Modify pattern validation | `src/detection/v2_atr.py` → `_validate_bullish_pattern()` |
| Add new features | `src/features/engineer.py` |
| Change validation params | `config/default.yaml` → `validation:` section |
| Query past experiments | `src/reporting/storage.py` → `ExperimentLogger` |
| Customize HTML reports | `src/reporting/dossier.py` |
| Understand data format | `src/schemas.py` |

---

## 6. Priority Improvements

| Priority | Area | Recommendation |
|----------|------|----------------|
| 🔴 | **Performance** | Replace `df.iterrows()` with vectorized backtest using `numpy` broadcast |
| 🔴 | **Fragility** | Add unit tests for `ATRDirectionalChange` state machine edge cases |
| 🟡 | **DRY** | Unify the two `BacktestEngine` implementations |
| 🟡 | **Robustness** | Add timezone-aware signal/bar matching with tolerance |
| 🟢 | **Config** | Add Pydantic validation for config dataclasses |

---

## 7. Command Reference

```bash
# Data
python -m src.data_engine                    # Build master store

# Backtest
python -m cli.run_backtest                   # Default run
python -m cli.run_backtest --detector atr --min-validity 0.7
python -m cli.run_grid_search                # Param search

# Validation
python -m cli.run_validation                 # Full VRD

# Tests
pytest tests/ -v                             # All tests
pytest tests/test_detection.py -v            # Detection only
```

---

*End of Prompt 1 - Architecture Overview*
