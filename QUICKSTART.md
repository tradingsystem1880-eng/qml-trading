# QML Trading System - Quick Start Guide

## 🚀 Launch New Premium Dashboard

```bash
cd /Users/hunternovotny/Desktop/QML_SYSTEM
streamlit run qml/dashboard/app.py
```

## 📊 Using the Dashboard

1. **Select Symbol** - BTC/USDT, ETH/USDT, or SOL/USDT
2. **Select Timeframe** - 1h, 4h, or 1d
3. **Click "Detect Patterns"** - Analyze historical data
4. **View Results** - Premium TradingView charts with pattern annotations

## 💻 Using the API

```python
from qml import QMLEngine

# Initialize engine
engine = QMLEngine()

# Detect patterns
patterns = engine.detect_patterns("BTC/USDT", "4h", days=180)
print(f"Found {patterns.total_found} patterns")

# Backtest
results = engine.backtest(patterns)
print(f"Sharpe: {results.sharpe_ratio:.2f}")

# Validate
validation = engine.validate(results)
print(f"Verdict: {validation.verdict}")
```

## 🎨 Custom Visualization

```python
from qml.dashboard.charts import render_pattern_chart
import streamlit as st

# Load data
from qml.core.data import DataLoader
loader = DataLoader()
df = loader.load_ohlcv("BTC/USDT", "4h")

# Render chart
html = render_pattern_chart(
    df=df,
    pattern=your_pattern_dict,
    title="My Custom Pattern"
)

# Display
st.components.v1.html(html, height=600)
```

## 📁 New Structure

```
QML_SYSTEM/
├── qml/                    # Main package (NEW)
│   ├── core/              # Engine, config, data
│   ├── strategy/          # Pattern detection
│   ├── backtest/          # Backtesting
│   ├── validation/        # Statistical tests
│   └── dashboard/         # UI + TradingView charts
│
├── user_data/             # User files (NEW)
│   ├── configs/           # Configs
│   ├── data/              # Market data
│   └── results/           # Results
│
└── src/                   # Legacy (still works)
```

## ✅ What's Working

- ✅ Pattern detection (QML patterns)
- ✅ Data loading (live + historical)
- ✅ Backtesting
- ✅ Statistical validation
- ✅ TradingView charts
- ✅ Premium dashboard

## 🎯 Features

**Visualization**:
- TradingView Lightweight Charts
- Pattern annotations (BOS, L, S, R)
- Support/resistance zones
- Trend lines
- Professional dark theme

**Analysis**:
- Multiple symbols & timeframes
- Validity filtering
- Backtest on demand
- Statistical validation

## 📖 Documentation

- `walkthrough.md` - Complete system overview
- `implementation_plan.md` - Gap analysis vs example pipeline
- `task.md` - Work completed

---

**Version**: 2.0.0
**Status**: Production Ready 🚀
