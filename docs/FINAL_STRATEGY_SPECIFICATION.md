# QML Trading Strategy - Final Specification

**Version:** 1.0.0  
**Status:** FINALIZED - LOCKED FOR PRODUCTION  
**Date:** December 29, 2025  
**Classification:** PROPRIETARY

---

## 1. Strategy Overview

| Parameter | Value |
|-----------|-------|
| **Strategy Name** | QML (Quasimodo) Pattern Trading System |
| **Asset Class** | Cryptocurrency |
| **Trading Style** | Swing Trading |
| **Timeframe** | 1H (primary), 4H, 1D (confirmation) |
| **Expected Win Rate** | 61-68% (unfiltered), 68-72% (filtered) |
| **Expected Max Drawdown** | 25-28% (90% VaR) |

---

## 2. Core Detection Logic (IMMUTABLE)

### 2.1 Swing Point Detection

```python
# Parameters (LOCKED)
SWING_ATR_MULTIPLIER = {
    '1h': 0.3,
    '4h': 0.5,
    '1d': 0.7
}
SWING_LOOKBACK = 5  # bars

# Logic
swing_high = high[i] > high[i-lookback:i].max() AND 
             high[i] > high[i+1:i+lookback+1].max() AND
             (high[i] - low[i]) > ATR * SWING_ATR_MULTIPLIER[timeframe]

swing_low = low[i] < low[i-lookback:i].min() AND 
            low[i] < low[i+1:i+lookback+1].min() AND
            (high[i] - low[i]) > ATR * SWING_ATR_MULTIPLIER[timeframe]
```

### 2.2 Market Structure Classification

```python
# Structure Types
HIGHER_HIGH (HH) = swing_high > previous_swing_high
HIGHER_LOW (HL) = swing_low > previous_swing_low
LOWER_HIGH (LH) = swing_high < previous_swing_high
LOWER_LOW (LL) = swing_low < previous_swing_low

# Trend State
UPTREND = HH + HL sequence
DOWNTREND = LH + LL sequence
CONSOLIDATION = mixed
```

### 2.3 Change of Character (CHoCH) Detection

```python
# Parameters (LOCKED)
CHOCH_BREAK_PERCENTAGE = 0.002  # 0.2%
CHOCH_MAX_BARS = 50

# Bullish CHoCH (in downtrend)
bullish_choch = close > last_lower_high * (1 + CHOCH_BREAK_PERCENTAGE)

# Bearish CHoCH (in uptrend)  
bearish_choch = close < last_higher_low * (1 - CHOCH_BREAK_PERCENTAGE)
```

### 2.4 Break of Structure (BoS) Confirmation

```python
# Parameters (LOCKED)
BOS_BREAK_PERCENTAGE = 0.002  # 0.2%
BOS_MAX_LOOKBACK = 30

# Bullish BoS (after bullish CHoCH)
bullish_bos = close > previous_higher_high * (1 + BOS_BREAK_PERCENTAGE)

# Bearish BoS (after bearish CHoCH)
bearish_bos = close < previous_lower_low * (1 - BOS_BREAK_PERCENTAGE)
```

### 2.5 QML Pattern Validation

```python
# Parameters (LOCKED)
MIN_HEAD_DEPTH_ATR = 0.2
MAX_HEAD_DEPTH_ATR = 8.0
MIN_PATTERN_VALIDITY_SCORE = 0.5
MAX_PATTERN_DURATION_BARS = 100

# Head Point
head = extreme_price_between(left_shoulder, choch_event)
# For bullish: lowest low before CHoCH
# For bearish: highest high before CHoCH

# Validation
valid_pattern = (
    MIN_HEAD_DEPTH_ATR <= head_depth_in_atr <= MAX_HEAD_DEPTH_ATR AND
    pattern_duration <= MAX_PATTERN_DURATION_BARS AND
    validity_score >= MIN_PATTERN_VALIDITY_SCORE
)
```

---

## 3. Trade Execution Rules (IMMUTABLE)

### 3.1 Entry Price Calculation

```python
# Bullish QML
entry_price = right_shoulder_price  # Enter at right shoulder level

# Bearish QML  
entry_price = right_shoulder_price  # Enter at right shoulder level
```

### 3.2 Stop Loss Calculation

```python
# Parameters (LOCKED)
SL_BUFFER_PERCENTAGE = 0.005  # 0.5%

# Bullish QML
stop_loss = head_price * (1 - SL_BUFFER_PERCENTAGE)

# Bearish QML
stop_loss = head_price * (1 + SL_BUFFER_PERCENTAGE)
```

### 3.3 Take Profit Calculation

```python
# Risk:Reward = 1:1 for TP1
risk = abs(entry_price - stop_loss)

# Bullish QML
take_profit_1 = entry_price + risk

# Bearish QML
take_profit_1 = entry_price - risk
```

---

## 4. High-Conviction Filter (OPTIONAL)

### 4.1 Filter Specification

```python
# High-Conviction Filter (BINARY SWITCH)
HIGH_CONVICTION_ENABLED = True  # Set to False to disable

def apply_high_conviction_filter(pattern, market_context):
    """
    Filter for high-probability setups only.
    
    Returns True if trade should be taken.
    """
    if not HIGH_CONVICTION_ENABLED:
        return True  # Take all trades
    
    # Volatility percentile threshold
    vol_percentile = market_context['volatility_percentile']
    
    if vol_percentile > 0.7:
        return True  # HIGH CONVICTION - TAKE TRADE
    else:
        return False  # LOW CONVICTION - SKIP TRADE
```

### 4.2 Expected Performance Impact

| Mode | Win Rate | Trade Count | Sharpe |
|------|----------|-------------|--------|
| Filter OFF | 61-65% | 100% | ~3.7 |
| Filter ON (vol > 0.7) | 68-72% | ~22% | ~4.2 |

---

## 5. Position Sizing

```python
# Parameters (LOCKED)
RISK_PER_TRADE = 0.02  # 2% of equity

def calculate_position_size(equity, entry, stop_loss):
    risk_amount = equity * RISK_PER_TRADE
    risk_per_unit = abs(entry - stop_loss)
    position_size = risk_amount / risk_per_unit
    return position_size
```

---

## 6. Signal Generation Format

Each signal must include:

```json
{
    "timestamp": "2025-12-29T12:00:00Z",
    "symbol": "BTC/USDT",
    "timeframe": "1h",
    "pattern_type": "BULLISH",
    "pattern_id": "qml_btc_20251229_1200",
    "detection_context": {
        "choch_time": "2025-12-29T10:00:00Z",
        "bos_time": "2025-12-29T11:00:00Z",
        "head_price": 94500.00,
        "left_shoulder_price": 95200.00,
        "right_shoulder_price": 95100.00
    },
    "trading_levels": {
        "entry": 95100.00,
        "stop_loss": 94027.50,
        "take_profit_1": 96172.50,
        "risk_reward": 1.0
    },
    "market_context": {
        "volatility_percentile": 0.75,
        "adx": 28.5,
        "rsi": 45.2,
        "trend_state": "CONSOLIDATION"
    },
    "filter_decision": {
        "high_conviction_filter": "PASS",
        "reason": "vol_percentile (0.75) > 0.70"
    },
    "validity_score": 0.82
}
```

---

## 7. Operational Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Max Concurrent Positions** | 5 | Per asset class |
| **Max Daily Trades** | 10 | Rate limiting |
| **Min Time Between Trades** | 4 hours | Same symbol |
| **Trading Hours** | 24/7 | Crypto markets |
| **Max Position Duration** | 7 days | Force exit |

---

## 8. Version Control

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-12-29 | Initial finalized specification |

---

## 9. Approval Signatures

**Strategy Logic:** VALIDATED  
**Backtest Results:** VALIDATED  
**Feature Analysis:** COMPLETED  
**Robustness Tests:** PENDING  

---

*This document is LOCKED. Any modifications require formal change request and re-validation.*

