# QML Trading System

**Institutional-Grade Quasimodo (QML) Pattern Detection & ML Trading System for Crypto Swing Trading**

## Overview

The QML Trading System is a comprehensive algorithmic trading platform designed to detect Quasimodo (QML) reversal patterns in cryptocurrency markets. It combines technical pattern recognition with machine learning to identify high-probability swing trading opportunities on 1H, 4H, and 1D timeframes.

### Key Features

- **Robust Pattern Detection**: ATR-adaptive swing point identification with timeframe-specific parameters
- **Market Structure Analysis**: Automated HH/HL/LH/LL classification and trend detection
- **CHoCH/BoS Detection**: Change of Character and Break of Structure identification with volume confirmation
- **ML-Enhanced Scoring**: XGBoost-based pattern quality scoring with probability calibration
- **Multi-Timeframe Analysis**: Concurrent analysis across 1H, 4H, and 1D timeframes
- **Walk-Forward Validation**: Rigorous backtesting with purged walk-forward analysis
- **Real-Time Alerts**: Telegram notifications and TradingView visualization
- **Production Dashboard**: Streamlit-based monitoring and analysis interface

## Architecture

```
QML_SYSTEM/
├── config/                 # Configuration management
│   └── settings.py         # Pydantic settings with env vars
├── src/
│   ├── data/               # Data pipeline
│   │   ├── fetcher.py      # CCXT-based data fetching
│   │   ├── database.py     # TimescaleDB operations
│   │   └── models.py       # Data models (Pydantic)
│   ├── detection/          # Pattern detection engine
│   │   ├── swing.py        # Swing point detection
│   │   ├── structure.py    # Market structure analysis
│   │   ├── choch.py        # CHoCH detection
│   │   ├── bos.py          # BoS detection
│   │   └── detector.py     # Main QML detector
│   ├── features/           # Feature engineering
│   │   └── engineer.py     # Feature calculation
│   ├── ml/                 # Machine learning
│   │   ├── model.py        # XGBoost model
│   │   └── trainer.py      # Training pipeline
│   ├── backtest/           # Backtesting framework
│   │   └── engine.py       # VectorBT-based backtester
│   ├── alerts/             # Alert system
│   │   └── telegram.py     # Telegram bot
│   ├── dashboard/          # Streamlit UI
│   │   └── app.py          # Dashboard application
│   └── utils/              # Utilities
│       ├── indicators.py   # Technical indicators
│       └── logging.py      # Logging configuration
├── docker/
│   └── init-db.sql         # Database initialization
├── models/                 # Trained model storage
├── logs/                   # Application logs
├── tests/                  # Test suite
├── docker-compose.yml      # Docker orchestration
├── Dockerfile              # Application container
└── pyproject.toml          # Python dependencies
```

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Poetry (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   cd /path/to/QML_SYSTEM
   ```

2. **Install dependencies**
   ```bash
   poetry install
   ```

3. **Configure environment**
   ```bash
   # Copy example config and edit
   cp .env.example .env
   # Edit .env with your settings
   ```

4. **Start database**
   ```bash
   docker-compose up -d timescaledb
   ```

5. **Sync market data**
   ```bash
   poetry run python -m src.main sync --start-date 2020-01-01
   ```

6. **Run pattern detection**
   ```bash
   poetry run python -m src.main detect
   ```

7. **Start dashboard**
   ```bash
   poetry run python -m src.main dashboard
   ```

### Docker Deployment

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f qml_app

# Access dashboard at http://localhost:8501
```

## Configuration

All configuration is managed via environment variables or the `config/settings.py` file.

### Key Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `POSTGRES_USER` | Database user | `qml_user` |
| `POSTGRES_PASSWORD` | Database password | Required |
| `BINANCE_API_KEY` | Binance API key | Optional |
| `TELEGRAM_BOT_TOKEN` | Telegram bot token | Optional |
| `LOG_LEVEL` | Logging level | `INFO` |

### Detection Parameters

```python
# Timeframe-specific ATR multipliers for swing detection
swing_atr_multiplier_1h: 0.5   # More sensitive for 1H
swing_atr_multiplier_4h: 1.0   # Standard for 4H
swing_atr_multiplier_1d: 1.5   # Less sensitive for 1D

# Pattern validation thresholds
min_pattern_validity_score: 0.7  # 70% minimum
min_head_depth_atr: 0.5          # Minimum head depth
max_head_depth_atr: 3.0          # Maximum head depth
```

## Usage

### Data Synchronization

```bash
# Sync all configured symbols
python -m src.main sync

# Sync specific symbols
python -m src.main sync --symbols BTC/USDT,ETH/USDT

# Force full resync from date
python -m src.main sync --start-date 2020-01-01 --force
```

### Pattern Detection

```bash
# Run detection on all symbols
python -m src.main detect

# Specific symbols and timeframes
python -m src.main detect --symbols BTC/USDT --timeframes 4h,1d
```

### Backtesting

```bash
# Full backtest
python -m src.main backtest --start-date 2022-01-01 --end-date 2024-01-01

# Specific symbols
python -m src.main backtest --symbols BTC/USDT,ETH/USDT
```

### Dashboard

```bash
# Start on default port
python -m src.main dashboard

# Custom port
python -m src.main dashboard --port 8080
```

## QML Pattern Definition

### Pattern Structure

A valid QML (Quasimodo) pattern consists of:

1. **Established Trend**: Clear HH/HL (uptrend) or LH/LL (downtrend) sequence
2. **CHoCH (Change of Character)**: First break of trend structure
3. **Head Formation**: Deep retracement beyond the initial shoulder
4. **BoS (Break of Structure)**: Confirmation of reversal direction
5. **Right Shoulder**: Retest of the neckline area (entry zone)

### Bullish QML

```
Price moves in a downtrend (LH/LL)
→ CHoCH: Price breaks above recent LH
→ Head: Price makes deeper low (sweep)
→ BoS: Price breaks above CHoCH level
→ Entry: Retest of demand zone
```

### Bearish QML

```
Price moves in an uptrend (HH/HL)
→ CHoCH: Price breaks below recent HL
→ Head: Price makes higher high (sweep)
→ BoS: Price breaks below CHoCH level
→ Entry: Retest of supply zone
```

## Performance Metrics

Target performance criteria (out-of-sample):

| Metric | Target |
|--------|--------|
| Win Rate | > 55% |
| Sharpe Ratio | > 1.5 |
| Sortino Ratio | > 2.0 |
| Max Drawdown | < 20% |
| Profit Factor | > 1.5 |

## Development

### Running Tests

```bash
poetry run pytest tests/ -v --cov=src
```

### Code Quality

```bash
# Format code
poetry run black src/

# Lint
poetry run ruff check src/

# Type checking
poetry run mypy src/
```

## Risk Disclaimer

This software is for educational and research purposes only. Trading cryptocurrencies involves substantial risk of loss. Past performance does not guarantee future results. Always conduct your own research and never trade with money you cannot afford to lose.

## License

Proprietary - All rights reserved.

