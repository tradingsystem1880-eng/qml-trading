# Phase 7.5 Detection Verification Report

Generated: 2026-01-22 21:29:28

Data: BTCUSDT 4h
Date Range: 2024-01-18 16:00:00+00:00 to 2026-01-17 12:00:00+00:00
Total Bars: 4,380

---

## 1. Historical Swing Detector Results

**Swings Detected:** 141
- Swing Highs: 75
- Swing Lows: 66
- Mean Significance (ATR): 4.502
- Mean Z-Score: 1.292
- Z-Score Range: [0.50, 8.55]

**Z-Score Distribution:**
- < 0: 0
- 0-1: 70
- 1-2: 51
- 2-3: 15
- > 3: 5

---

## 2. v2_atr Detector Comparison

**Signals Detected:** 839

**Overlap Analysis:**
- Overlap Count: N/A (no bar indices in old signals)
- Overlap %: N/A
- Note: Could not compare - old signals lack bar_index metadata

---

## 3. QML Pattern Detection

**Patterns Found:** 8
**Valid Patterns:** 8

**Tier Distribution:**
- Tier A (Excellent): 1
- Tier B (Good): 2
- Tier C (Acceptable): 4
- Rejected: 1

**Score Statistics:**
- Mean Score: 0.558
- Median Score: 0.559
- Min Score: 0.263
- Max Score: 0.823

**Sample Patterns (Top 5):**

1. **BULLISH** @ bar 1157
   - Score: 0.575 (Tier C)
   - Head Extension: 1.51 ATR
   - BOS Efficiency: 0.54

2. **BEARISH** @ bar 2165
   - Score: 0.653 (Tier B)
   - Head Extension: 1.50 ATR
   - BOS Efficiency: 0.72

3. **BULLISH** @ bar 2205
   - Score: 0.501 (Tier C)
   - Head Extension: 1.69 ATR
   - BOS Efficiency: 0.22

4. **BULLISH** @ bar 2918
   - Score: 0.544 (Tier C)
   - Head Extension: 1.07 ATR
   - BOS Efficiency: 0.03

5. **BEARISH** @ bar 3025
   - Score: 0.823 (Tier A)
   - Head Extension: 1.32 ATR
   - BOS Efficiency: 0.68

---

## 4. Backtest Integration Check

**Component Availability:**
- QMLPattern_exists: Yes
- Signal_exists: Yes
- BacktestEngine_exists: No
- factory_integration: Yes

**Integration Issues:**
- ValidationResult uses HistoricalSwingPoint, QMLPattern expects separate fields
- Need adapter: ValidationResult.p1.price -> QMLPattern.left_shoulder_price
- BacktestEngine not found

**Adapter Code Needed:**
```python
def validation_result_to_signal(vr: ValidationResult, df: pd.DataFrame) -> Signal:
    """Convert ValidationResult to Signal for backtest compatibility."""
    atr = df['atr'].iloc[vr.p5.bar_index] if 'atr' in df.columns else vr.atr_p5
    
    if vr.direction == PatternDirection.BULLISH:
        # Short setup
        entry = vr.p5.price + (0.1 * atr)
        sl = vr.p3.price + (0.5 * atr)
        signal_type = SignalType.SELL
    else:
        # Long setup
        entry = vr.p5.price - (0.1 * atr)
        sl = vr.p3.price - (0.5 * atr)
        signal_type = SignalType.BUY
    
    risk = abs(entry - sl)
    tp1 = entry + (1.5 * risk) if signal_type == SignalType.BUY else entry - (1.5 * risk)
    
    return Signal(
        timestamp=vr.p5.timestamp,
        signal_type=signal_type,
        price=entry,
        stop_loss=sl,
        take_profit=tp1,
        strategy_name='QML_Phase75',
        validity_score=score_result.total_score,
    )
```

---

## 5. Mini-Backtest Results

**Patterns Tested:** 7
**Trades Executed:** 7

**Performance Metrics:**
- Win Rate: 42.86%
- Profit Factor: 1.81
- Total Return: 2.12%
- Max Drawdown: 2.24%
- Sharpe Ratio: 3.17

**Winning/Losing:** 3W / 4L

---

## 6. Summary & Next Steps

**Verification Status:**
- Historical Detector: WORKING
- Pattern Validator: WORKING
- Pattern Scorer: WORKING
- Factory Integration: WORKING

**Recommended Next Steps:**
1. Create adapter function for Signal conversion
2. Run full backtest with new detector
3. Compare win rate / PF with old detector
4. Begin ML optimization of config parameters