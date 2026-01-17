# Experiment 02: ATR Directional Change Detection v2

## Method
ATR-driven pattern detection using directional change confirmation.
Replaced naive rolling window with price-action-driven approach.

## Files
- `patterns.csv` - Detected QML patterns
- `ohlcv.csv` - BTC/USDT OHLCV data
- `atr_detection.csv` - ATR-based detection intermediates

## Algorithm
Uses ATR Directional Change to confirm swing highs/lows before pattern matching.
