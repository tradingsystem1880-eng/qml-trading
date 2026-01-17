# QML Dashboard - User Guide

## 🚀 Quick Start

```bash
streamlit run qml/dashboard/app.py
```

## 📋 Features

### 1. Dashboard Page (📊)
- Quick overview of active patterns
- Recent pattern summary
- System statistics
- Real-time updates

### 2. Scanner Page (🔍)
**Multi-Symbol Pattern Scanner**

Steps:
1. Select symbols (BTC/USDT, ETH/USDT, etc.)
2. Choose timeframe (1h, 4h, 1d)
3. Set history period (days)
4. Set minimum validity threshold
5. Click "Start Scan"

Results show:
- Symbol
- Pattern type (Bullish/Bearish)
- Validity score
- Entry price
- Risk:Reward ratio
- Timeframe

### 3. Pattern Analyzer Page (📈)
**Detailed Pattern Analysis with TradingView Charts**

Features:
- Symbol/timeframe selection
- Pattern detection
- Advanced chart options:
  - ✅ Moving Averages (EMA 20, 50, 200)
  - ✅ Fibonacci retracement levels
  - ✅ Volume profile with POC
- Pattern metrics display
- Interactive TradingView charts

Chart Annotations:
- Swing point labels (P1-P5)
- BOS (Break of Structure) markers
- Entry arrows
- Support/resistance zones
- Stop Loss & Take Profit levels

### 4. Settings Page (⚙️)
**System Configuration**

Configure:
- Detection parameters (ATR period, swing lookback)
- Default validity threshold
- Chart theme & height
- Default timeframe

## 🎨 Chart Features

### Moving Averages
- EMA 20 (Yellow/Gold)
- EMA 50 (Orange)
- EMA 200 (Red)
- Toggle on/off in Pattern Analyzer

### Fibonacci Retracement
- Auto-detects swing high/low
- 7 standard levels: 0%, 23.6%, 38.2%, 50%, 61.8%, 78.6%, 100%
- Color-coded: Gold for key levels, Gray for others

### Volume Profile
- Aggregates volume by price
- Shows POC (Point of Control)
- Purple zone for highest volume area

## 💡 Tips

1. **Scanner Best Practices**:
   - Start with 2-3 symbols for faster scans
   - Use 4h or 1d timeframes for quality signals
   - Set min validity ≥ 0.5 for reliable patterns

2. **Pattern Analysis**:
   - Enable MAs to see trend context
   - Use Fibonacci to identify support/resistance
   - Check volume profile for price acceptance

3. **Performance**:
   - Dashboard uses caching for speed
   - Charts render client-side (fast)
   - Scans process in parallel when possible

## 🔧 Keyboard Shortcuts

- `r` - Refresh current page
- `←` / `→` - Navigate between patterns
- `Ctrl + Click` - Open pattern in new tab

## 📊 Understanding Pattern Scores

**Validity Score (0-100%)**:
- 70%+ : High confidence, strong pattern
- 50-70% : Moderate confidence
- <50% : Low confidence, use with caution

**Risk:Reward Ratio**:
- 2.0+ : Excellent
- 1.5-2.0 : Good
- <1.5 : Marginal

## 🐛 Troubleshooting

**Charts not loading**:
- Clear browser cache
- Check internet connection (CDN required for TradingView lib)

**Scanner timeout**:
- Reduce number of symbols
- Decrease history period

**No patterns found**:
- Lower min validity threshold
- Increase history period
- Try different symbols/timeframes

## 🔗 Resources

- [TradingView Lightweight Charts Docs](https://tradingview.github.io/lightweight-charts/)
- [QML Pattern Specification](../docs/QML_PATTERN_EXPLAINED.md)
- [System Architecture](QUICKSTART.md)

---

**Version**: 2.0.0
**Last Updated**: 2026-01-10
