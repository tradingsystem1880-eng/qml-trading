# QML Forensic Trading System

> **VRD 2.0 Compliant** | Pattern Detection & Validation for BTC/USDT

A professional algorithmic trading research platform for detecting and validating QML (Quasimodo-Like) chart patterns.

---

## 🚀 Quick Start

```bash
# Run a backtest with default parameters
python -m cli.run_backtest

# Custom parameters
python -m cli.run_backtest --symbol BTCUSDT --timeframe 4h --detector atr --min-validity 0.7
```

**Output:**
- Console metrics (P&L, Win Rate, Sharpe, etc.)
- SQLite log: `results/experiments.db`
- HTML report: `results/atr/{run_id}_dossier.html`

---

## 📁 Project Structure

```
QML_SYSTEM/
├── cli/                    # Command-line entry points
│   └── run_backtest.py     # Main backtest runner
│
├── src/                    # Core library
│   ├── core/               # Data models (Candle, Signal, Trade)
│   ├── detection/          # Pattern detection (ATR, Rolling)
│   └── reporting/          # Flight Recorder (SQLite + HTML)
│
├── data/                   
│   └── processed/BTC/      # Price data (parquet)
│
├── results/                
│   ├── experiments.db      # All runs logged here
│   └── {strategy}/         # HTML dossiers per strategy
│
├── config/                 
│   └── default.yaml        # Tunable parameters
│
└── archive/                # Legacy code (reference only)
```

---

## 🧠 Architecture

```
           ┌──────────────┐
           │  📂 Data     │  data/processed/BTC/4h_master.parquet
           └──────┬───────┘
                  ▼
           ┌──────────────┐
           │  🧠 Brain    │  src/detection/ → List[Signal]
           └──────┬───────┘
                  ▼
           ┌──────────────┐
           │  🏋️ Body     │  BacktestEngine → Results
           └──────┬───────┘
                  ▼
           ┌──────────────┐
           │  📼 Recorder │  SQLite + HTML Dossier
           └──────────────┘
```

---

## 📊 Key Features

| Feature | Description |
|---------|-------------|
| **ATR Directional Change** | Price-action driven swing detection |
| **Validity Scoring** | Pattern quality from 0.0 to 1.0 |
| **Flight Recorder** | Every run logged to SQLite |
| **HTML Dossiers** | Standalone reports with Plotly charts |
| **VRD 2.0 Compliant** | Full reproducibility and forensic analysis |

---

## 🔧 Configuration

Default parameters in `config/default.yaml`:

```yaml
detection:
  method: atr_directional_change
  min_validity_score: 0.7
  atr_period: 14
  
risk:
  stop_loss_atr_mult: 0.5
  take_profit_atr_mult: [1.0, 2.0, 3.0]
```

---

## 📚 Documentation

- **[SYSTEM_CONTEXT.md](SYSTEM_CONTEXT.md)** — AI agent memory & architecture
- **[config/default.yaml](config/default.yaml)** — Parameter reference

---

## 🔬 Query Past Experiments

```python
from src.reporting import ExperimentLogger

logger = ExperimentLogger()

# Best runs by P&L
best = logger.get_top_runs(symbol='BTCUSDT', metric='pnl_percent', limit=10)

# Recent runs
recent = logger.get_recent_runs(limit=20)

# Get specific run config
run = logger.get_run('abc12345')
```

---

## 📜 License

Proprietary — For research use only.
