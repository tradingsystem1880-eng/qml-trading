# Prompt 2 - Crypto Data Integration

> **Purpose**: Comprehensive analysis of data fetching and management system in QML_SYSTEM for AI systems to understand data sources, storage formats, quality issues, and extension limitations.
> **Generated**: 2026-01-07
> **Target Audience**: DeepSeek AI / Advanced AI Coding Assistants

---

## 1. Integrated Crypto Exchanges and Data Sources

### 1.1 Exchange Integration via CCXT

```
┌─────────────────────────────────────────────────────────────────────┐
│  PRIMARY: CCXT Python Library (ccxt.exchange)                      │
│  Location: src/data_engine.py, src/data/fetcher.py                 │
└─────────────────────────────────────────────────────────────────────┘
```

| Exchange | Status | Implementation |
|----------|--------|----------------|
| **Binance** | 🟢 PRIMARY | Hardcoded default in `data_engine.py:170`, `fetcher.py:55` |
| **Any CCXT Exchange** | 🟡 SUPPORTED | Factory pattern via `exchange_id` parameter |
| **Coinbase** | 🟡 UNTESTED | Supported by CCXT, not actively used |

### 1.2 Data Source Configuration

```python
# src/data_engine.py:38-45
DEFAULT_SYMBOL = "BTC/USDT"
DEFAULT_TIMEFRAMES = ["1h", "4h"]
DEFAULT_YEARS = 5
DEFAULT_ATR_PERIOD = 14

# config/default.yaml:79-83
data:
  default_symbol: BTCUSDT
  default_timeframe: 4h
  lookback_days: 1460  # 4 years
  source: binance
```

### 1.3 Enhanced Data Sources (Futures-Specific)

| Data Type | Source | Location | Status |
|-----------|--------|----------|--------|
| **Funding Rates** | Binance Futures API | `enhanced_fetcher.py:49-86` | 🟢 Implemented |
| **Open Interest** | Binance FAPI | `enhanced_fetcher.py:88-130` | 🟢 Implemented |
| **Long/Short Ratio** | Binance FAPI | `enhanced_fetcher.py:132-169` | 🟢 Implemented |
| **Liquidations** | — | — | 🔴 Not implemented |

### 1.4 Exchange Connection Flow

```
┌─────────────────────────┐     ┌─────────────────────────┐
│  DataFetcher.__init__() │────▶│  ccxt.binance({        │
│  exchange_id="binance"  │     │    enableRateLimit: True│
│                         │     │    defaultType: "spot"  │
│                         │     │  })                     │
└─────────────────────────┘     └─────────────────────────┘
           │
           ▼
┌─────────────────────────┐
│  fetch_ohlcv()          │
│  - Batch requests       │
│  - 1000 candles/request │
│  - Auto rate limiting   │
└─────────────────────────┘
```

---

## 2. Historical Data Storage and Access

### 2.1 Storage Architecture

```
data/
├── processed/                    # 📦 PRIMARY: Master Data Store
│   └── BTC/
│       ├── 1h_master.parquet     # ~2.5 MB, ~43,000 rows
│       └── 4h_master.parquet     # ~618 KB, ~10,000 rows
│
├── raw/                          # 📥 Raw API downloads (currently empty)
│
└── samples/                      # 🧪 Test fixtures
```

### 2.2 File Formats

| Format | Location | Usage | Schema |
|--------|----------|-------|--------|
| **Parquet** | `data/processed/BTC/*.parquet` | PRIMARY storage for OHLCV | `[time, Open, High, Low, Close, Volume, ATR]` |
| **SQLite** | `results/experiments.db` | Experiment logging | Experiments table |
| **TimescaleDB** | `src/data/database.py` | Production DB (optional) | OHLCV, swing_points, qml_patterns tables |
| **CSV** | `data/*.csv` | Ad-hoc exports | Various |

### 2.3 Parquet Schema (Master Data Store)

```python
# Output from build_master_store() - src/data_engine.py:340-341
column_order = ['time', 'Open', 'High', 'Low', 'Close', 'Volume', 'ATR']

# Data types:
# time: datetime64[ns, UTC]
# Open, High, Low, Close, Volume: float64
# ATR: float64 (14-period Wilder smoothing)
```

### 2.4 Data Access Patterns

```
┌─────────────────────────────────────────────────────────────────────┐
│  Method 1: load_master_data() - Preferred                          │
│  Location: src/data_engine.py:370-412                              │
│                                                                     │
│  from src.data_engine import load_master_data                       │
│  df = load_master_data(timeframe='4h', start_time=..., end_time=...)│
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  Method 2: Direct Parquet Read                                     │
│  Location: cli/run_backtest.py:396-465                             │
│                                                                     │
│  df = pd.read_parquet('data/processed/BTC/4h_master.parquet')       │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  Method 3: DataFetcher with Database                               │
│  Location: src/data/fetcher.py:338-377                             │
│                                                                     │
│  fetcher = DataFetcher(exchange_id='binance', db=db)                │
│  df = fetcher.get_data(symbol='BTC/USDT', timeframe='4h')           │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.5 Database Schema (TimescaleDB - Optional)

```sql
-- From src/data/database.py INSERT queries

-- OHLCV Hypertable
CREATE TABLE ohlcv (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    exchange TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    open FLOAT,
    high FLOAT,
    low FLOAT,
    close FLOAT,
    volume FLOAT,
    quote_volume FLOAT,
    trades INT,
    PRIMARY KEY (time, symbol, exchange, timeframe)
);

-- Swing Points
CREATE TABLE swing_points (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT,
    timeframe TEXT,
    swing_type TEXT,  -- 'high' or 'low'
    price FLOAT,
    significance FLOAT,
    atr_at_point FLOAT,
    confirmed BOOLEAN
);

-- QML Patterns
CREATE TABLE qml_patterns (
    id SERIAL,
    detection_time TIMESTAMPTZ NOT NULL,
    symbol TEXT,
    timeframe TEXT,
    pattern_type TEXT,
    -- ... 20+ columns for pattern geometry
    validity_score FLOAT,
    ml_confidence FLOAT,
    status TEXT,  -- 'forming', 'active', 'triggered', 'completed'
    outcome TEXT  -- 'win', 'loss', 'breakeven'
);
```

---

## 3. Data Quality Issues

### 3.1 Data Validator Checks

```
┌─────────────────────────────────────────────────────────────────────┐
│  DataValidator.check_health() - src/data/integrity.py:108-272      │
└─────────────────────────────────────────────────────────────────────┘
```

| Check | Severity | Description |
|-------|----------|-------------|
| Empty DataFrame | 🔴 FAIL | `len(df) == 0` |
| Missing columns | 🔴 FAIL | Required: `open, high, low, close` |
| NaN values | 🔴 FAIL | Any NaN in OHLCV columns |
| Zero prices | 🔴 FAIL | Any `price == 0` |
| Negative prices | 🔴 FAIL | Any `price < 0` |
| High < Low | 🟡 WARN | OHLC inconsistency |
| Timestamp gaps | 🟡 WARN | Missing candles (>10% tolerance) |

### 3.2 Known Data Quality Issues

| Issue | Location | Impact | Mitigation |
|-------|----------|--------|------------|
| **Timestamp gaps** | Exchange maintenance windows | Missing signals | Forward-fill with warning |
| **Duplicate timestamps** | API pagination overlap | Double-counting | `drop_duplicates(subset=['time'])` |
| **High < Low anomalies** | Flash crashes/data errors | Bad ATR calculation | Swap values: `df.loc[mask, [h, l]] = df.loc[mask, [l, h]].values` |
| **UTC timezone inconsistency** | Multiple data sources | Signal timing errors | Force UTC: `pd.to_datetime(..., utc=True)` |

### 3.3 Data Cleaning Pipeline

```python
# src/data_engine.py:89-159 - clean_ohlcv()

Cleaning Steps:
1. Remove duplicate timestamps
2. Sort by time
3. Forward-fill missing OHLCV values
4. Drop rows with NaN in critical columns
5. Swap High/Low where High < Low
```

### 3.4 Gap Detection Algorithm

```python
# src/data/integrity.py:207-236

TIMEFRAME_INTERVALS = {
    '1h': timedelta(hours=1),
    '4h': timedelta(hours=4),
    '1d': timedelta(days=1),
    # ...
}

# Gap = interval > expected * 1.1 (10% tolerance)
time_diffs = time_series.diff().dropna()
tolerance = expected_interval * 1.1
gap_mask = time_diffs > tolerance
gaps_found = gap_mask.sum()
```

---

## 4. Limitations for Adding New Coins or Timeframes

### 4.1 Current Hardcoded Limitations

| Limitation | Location | Current Value | Required Change |
|------------|----------|---------------|-----------------|
| **Symbol locked to BTC** | `data_engine.py:39` | `DEFAULT_SYMBOL = "BTC/USDT"` | Need parameterization |
| **Output path locked** | `data_engine.py:45` | `DATA_DIR = .../BTC` | Symbol-based directory needed |
| **Timeframes limited** | `data_engine.py:40` | `["1h", "4h"]` | Extend list |
| **Timeframe parsing** | `data_engine.py:210-213` | Only `1h`, `4h` in `tf_ms` dict | Add more intervals |

### 4.2 Symbol Extension Checklist

To add a new coin (e.g., ETH/USDT):

```
Step 1: ❌ data_engine.py has no symbol parameter in build_master_store()
        → Need to add symbol parameter and dynamic output path

Step 2: ❌ Output directory is hardcoded to /BTC/
        → Need: data/processed/{SYMBOL}/

Step 3: ✅ DataFetcher.fetch_ohlcv() accepts any symbol
        → Works with CCXT's universal API

Step 4: ❌ load_master_data() assumes BTC directory
        → Need symbol parameter

Step 5: ❌ cli/run_backtest.py load_data() has hardcoded BTC fallback
        → possible_paths includes only BTC paths
```

### 4.3 Timeframe Extension Checklist

To add a new timeframe (e.g., `15m`):

```
Step 1: ✅ CCXT supports any timeframe
        → '1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '1w'

Step 2: ⚠️ data_engine.py:210-213 has limited tf_ms mapping
        → Only '1h' and '4h' mapped for batch calculation

Step 3: ✅ DataValidator.TIMEFRAME_INTERVALS is comprehensive
        → Already includes '15m', '30m', etc.

Step 4: ❌ Pattern detection may need retuning
        → ATR lookback (14 bars) means different time spans per timeframe
```

### 4.4 Multi-Symbol Architecture Gap

```
Current State:
┌─────────────────┐
│  data/processed │
│  └── BTC/       │  ← Single-symbol design
│      ├── 1h     │
│      └── 4h     │
└─────────────────┘

Required for Multi-Symbol:
┌─────────────────┐
│  data/processed │
│  ├── BTC/       │
│  │   ├── 1h     │
│  │   └── 4h     │
│  ├── ETH/       │  ← New
│  │   ├── 1h     │
│  │   └── 4h     │
│  └── SOL/       │  ← New
│      └── ...    │
└─────────────────┘
```

---

## 5. Key File Reference

| When you need to... | Look at... |
|---------------------|------------|
| Fetch new data from exchange | `src/data_engine.py` → `fetch_btc_ohlcv()` |
| Build master parquet files | `python -m src.data_engine` |
| Load data for backtesting | `src/data_engine.py` → `load_master_data()` |
| Add enhanced futures data | `src/data/enhanced_fetcher.py` |
| Validate data quality | `src/data/integrity.py` → `DataValidator` |
| Store data in TimescaleDB | `src/data/database.py` → `DatabaseManager` |
| Sync incremental updates | `src/data/fetcher.py` → `sync_symbol()` |

---

## 6. Command Reference

```bash
# Build Master Data Store (5 years BTC data)
python -m src.data_engine

# Dry run (only 30 days)
python -m src.data_engine --dry-run

# Custom timeframes
python -m src.data_engine --timeframes 1h 4h 1d

# Custom years
python -m src.data_engine --years 3

# Validate existing data
python -c "
from src.data_engine import load_master_data
from src.data.integrity import DataValidator

df = load_master_data('4h')
validator = DataValidator()
report = validator.check_health(df, timeframe='4h')
print(report)
"
```

---

## 7. Priority Improvements for Data System

| Priority | Area | Recommendation |
|----------|------|----------------|
| 🔴 | **Multi-Symbol** | Refactor `data_engine.py` to accept symbol parameter, use `data/processed/{symbol}/` paths |
| 🔴 | **Timeframe Flexibility** | Extend `tf_ms` dict in `fetch_btc_ohlcv()` for all CCXT timeframes |
| 🟡 | **Incremental Updates** | Use `DataFetcher.sync_symbol()` instead of full refetch |
| 🟡 | **Data Freshness** | Add CLI flag `--update` to append new candles to existing parquet |
| 🟢 | **Caching** | Add timestamp check to skip fetch if data is recent |

---

*End of Prompt 2 - Crypto Data Integration*
