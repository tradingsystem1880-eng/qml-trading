# QML Trading System - System Context

> **Last Updated**: 2026-01-04 12:07:41
> **Purpose**: This file serves as the "memory" for AI agents working on this codebase.

---

## 🎯 Project Overview

**QML (Quasimodo-Like) Pattern Trading System** is an algorithmic trading research platform focused on detecting and trading specific chart patterns on BTC/USDT.

### Core Objectives (VRD 2.0)
1. **Validate Reality of Edge** — Statistical proof that patterns predict price movement
2. **Understand Why It Works** — Feature analysis of winning vs losing trades
3. **Assess Durability** — Walk-forward and regime analysis
4. **Identify Failure Modes** — Drawdown and stress testing
5. **Define Deployment Rules** — Risk limits and position sizing

---

## 📁 Directory Structure

```
QML_SYSTEM/
├── src/                    # Core library (importable modules)
│   ├── core/               # Fundamental abstractions (models, config, exceptions)
│   ├── detection/          # Pattern detection algorithms
│   │   └── legacy/         # Archived detection versions (v1.0.0, v1.1.0, v2.0.0)
│   ├── backtest/           # Backtesting engine
│   ├── validation/         # Statistical validation (permutation, monte carlo, bootstrap)
│   ├── reporting/          # HTML dossier and chart generation
│   │   └── templates/      # Jinja2 HTML templates
│   ├── data/               # Data fetching and loading
│   ├── deployment/         # Production utilities (gatekeeper, paper trader)
│   ├── pipeline/           # Orchestration logic
│   ├── strategies/         # Strategy adapters
│   └── dashboard/          # Web dashboard
│
├── cli/                    # Command-line entry points
│   ├── run_backtest.py     # Primary backtest command
│   ├── run_validation.py   # VRD validation suite
│   └── run_detection.py    # Pattern detection command
│
├── data/                   # Data storage
│   ├── raw/                # Raw API downloads
│   ├── processed/          # Clean parquet files (BTC 1h, 4h)
│   └── samples/            # Sample data for tests
│
├── results/                # Output artifacts
│   ├── experiments.db      # SQLite database for dashboard
│   ├── charts/             # Generated visualizations
│   └── reports/            # HTML dossiers
│
├── config/                 # Configuration
│   ├── default.yaml        # Default parameters
│   └── strategies/         # Strategy-specific configs
│
├── tests/                  # Test suite
│   ├── unit/
│   └── integration/
│
├── notebooks/              # Research notebooks
├── docs/                   # Documentation
├── archive/                # Legacy code (reference only)
└── _incoming_refactor/     # Temporary staging for refactoring
```

---

## 🔧 Key Components

### Detection Module (`src/detection/`)
- **Primary Algorithm**: ATR Directional Change (v2.0.0)
- **Pattern Type**: QML Bullish (5-point pattern: P1→P2→P3→P4→P5)
- **Entry Signal**: P5 confirmation with ATR-based SL/TP

### Validation Module (`src/validation/`)
- **Permutation Test**: Shuffle returns to test edge significance
- **Monte Carlo**: Simulate equity paths for risk analysis
- **Bootstrap**: Confidence intervals on performance metrics
- **Walk-Forward**: Out-of-sample validation

### Reporting Module (`src/reporting/`)
- **Dossier**: HTML report generator (Strategy Autopsy Report)
- **Visuals**: Equity curves, drawdown charts, MC cones

---

## 📊 Data Contract

### OHLCV Parquet Schema
| Column | Type | Description |
|--------|------|-------------|
| timestamp | datetime64 | UTC timestamp |
| open | float64 | Open price |
| high | float64 | High price |
| low | float64 | Low price |
| close | float64 | Close price |
| volume | float64 | Volume |

### Trade Record Schema
| Column | Type | Description |
|--------|------|-------------|
| entry_time | datetime64 | Entry timestamp |
| exit_time | datetime64 | Exit timestamp |
| entry_price | float64 | Entry price |
| exit_price | float64 | Exit price |
| side | str | 'LONG' or 'SHORT' |
| pnl_pct | float64 | PnL percentage |
| result | str | 'WIN', 'LOSS', 'BREAKEVEN' |

---

## 🚀 Quick Start Commands

```bash
# Run pattern detection
python -m cli.run_detection --symbol BTCUSDT --timeframe 4h

# Run backtest with validation
python -m cli.run_backtest --config config/strategies/qml_bullish.yaml

# Run full VRD validation
python -m cli.run_validation --trades results/trades.csv

# Start dashboard
python -m src.dashboard.app
```

---

## ⚠️ Important Notes

1. **Legacy Detection Code**: Old versions are preserved in `src/detection/legacy/` for reference
2. **Data Location**: Primary BTC data is in `data/processed/BTC/`
3. **Results Database**: `results/experiments.db` tracks all experiment runs
4. **Configuration**: All tunable parameters should be in YAML configs, not hardcoded

---

## 🔗 Related Files

- [config/default.yaml](config/default.yaml)
- [README.md](README.md)

---

## 🏗️ System Architecture v2.0 (January 2026)

### The Brain-Body-Recorder Pattern

```
┌─────────────────────────────────────────────────────────────┐
│                     python -m cli.run_backtest              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  📂 DATA LOADER                                              │
│  data/processed/BTC/4h_master.parquet → DataFrame           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  🧠 BRAIN: src/detection/                                    │
│  ├── base.py        → BaseDetector ABC                      │
│  ├── v2_atr.py      → ATRDetector (primary)                 │
│  ├── v1_rolling.py  → RollingWindowDetector (legacy)        │
│  └── factory.py     → get_detector("atr") → ATRDetector     │
│                                                              │
│  Output: List[Signal] with price, SL, TP, validity_score    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  🏋️ BODY: cli/run_backtest.py → BacktestEngine              │
│  ├── Consumes signals, opens/closes trades                  │
│  ├── Tracks equity curve and drawdowns                      │
│  └── Calculates Sharpe, Win Rate, Profit Factor, etc.       │
│                                                              │
│  Output: Dict with metrics + List[Trade]                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  📼 FLIGHT RECORDER: src/reporting/                          │
│  ├── storage.py     → ExperimentLogger (SQLite)             │
│  └── dossier.py     → DossierGenerator (HTML + Plotly)      │
│                                                              │
│  Output:                                                     │
│  ├── results/experiments.db (queryable history)             │
│  └── results/{strategy}/{run_id}_dossier.html               │
└─────────────────────────────────────────────────────────────┘
```

### Module Reference

| Module | Purpose | Key Classes |
|--------|---------|-------------|
| `src/core/models.py` | Data structures | `Candle`, `Signal`, `Trade`, `SwingPoint` |
| `src/detection/` | Pattern detection | `BaseDetector`, `ATRDetector`, `get_detector()` |
| `src/reporting/` | Logging & reports | `ExperimentLogger`, `DossierGenerator` |
| `cli/run_backtest.py` | Backtest runner | `BacktestEngine`, `BacktestConfig` |

### How to Run

```bash
# Default backtest (BTCUSDT 4h, ATR detector)
python -m cli.run_backtest

# Custom parameters
python -m cli.run_backtest --symbol BTCUSDT --timeframe 4h --detector atr --min-validity 0.7

# Query past experiments
python -c "
from src.reporting import ExperimentLogger
logger = ExperimentLogger()
for run in logger.get_top_runs(metric='pnl_percent', limit=5):
    print(f'{run[\"run_id\"]}: {run[\"pnl_percent\"]:+.2f}%')
"
```

### Key File Locations

| Path | Purpose |
|------|---------|
| `data/processed/BTC/4h_master.parquet` | Primary price data |
| `results/experiments.db` | SQLite experiment log |
| `results/{strategy}/{run_id}_dossier.html` | HTML reports |
| `config/default.yaml` | Default parameters |
| `archive/legacy_source_2025/` | Pre-refactor code (reference only) |
