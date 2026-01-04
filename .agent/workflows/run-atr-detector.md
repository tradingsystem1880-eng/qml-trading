---
description: Run QML pattern detection with ATR v2.0.0 detector
---

# QML Pattern Detection Workflow (v2.0.0 ATR)

This workflow runs the ATR Directional Change detector on any symbol/timeframe.

## Prerequisites
- Python 3.9+
- Dependencies: `pip install ccxt pandas numpy loguru pydantic`

## Steps

### 1. Run Detection
// turbo
```bash
cd /Users/hunternovotny/Desktop/QML_SYSTEM
python3 scripts/atr_detector.py
```

This will:
- Fetch BTC/USDT 1h data from Binance
- Run ATR-driven pattern detection
- Save results to `btc_atr_patterns.csv`

### 2. Export for Visualization

After detection, export patterns with coordinates:

```bash
python3 -c "
import pandas as pd
patterns = pd.read_csv('btc_atr_patterns.csv')
print(f'Detected {len(patterns)} patterns')
print(patterns[['time', 'pattern_type', 'validity_score']].head(10))
"
```

### 3. View in Colab

1. Upload to Google Colab:
   - `qml_patterns_v2_export.csv`
   - `btc_ohlcv_v2_export.csv`
2. Paste `qml_colab_visualizer.py` contents
3. Use slider to view patterns

### 4. Backtest Results

Check current v2.0.0 performance:
```bash
python3 qml_strategy_vrd/vr_dashboard.py show 20230101_000001_BTCUSDT_1h_v2.0.0
```

## Configuration

Edit `qml_strategy_vrd/detection_logic/v2.0.0_atr_directional_change/params.json`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| window_size | 200 | Pattern lookback bars |
| atr_lookback | 14 | ATR calculation period |
| min_validity_score | 0.7 | Minimum pattern quality |

## Expected Output

- **Patterns**: 50-60 per year
- **Win Rate**: ~65-67%
- **Profit Factor**: 1.7+
