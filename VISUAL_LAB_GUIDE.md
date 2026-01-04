# 🎯 QML Pattern Visualization Hub - User Guide

## Overview

`visual_lab.ipynb` is your professional visualization hub for analyzing QML pattern backtests. Clean, organized, and reusable for any backtest results.

---

## 📋 Notebook Structure

### **Step 1: Configure Data Files** (Cell 2)
- Set paths to your pattern and OHLCV CSV files
- Simple one-line configuration for switching between backtests

### **Step 2: Load Dependencies & Data** (Cell 3)
- Auto-loads all required libraries
- Imports and validates your data
- Shows summary statistics (pattern count, date range, distribution)

### **Step 3: Load Visualization Engine** (Cell 4)
- Loads the premium TradingView-style chart generator
- Ready to visualize any pattern

### **Step 4: Interactive Pattern Selector** (Cell 5)
- **Dropdown menu** with all patterns sorted by validity
- **One-click visualization** via "📊 Visualize" button
- **Auto-displays** highest validity pattern on load

### **Advanced Section** (Cell 6)
- Quick lookup functions for programmatic access
- `show_top_patterns(n)` - View top N patterns
- `quick_plot(id)` - Instantly plot any pattern by ID

### **Documentation** (Cell 7)
- Complete file format specifications
- Instructions for loading new backtests

---

## 🚀 Quick Start

### First Time Setup
```
1. Open visual_lab.ipynb
2. Run Cell 1 (title/overview - optional)
3. Run Cell 2 (configuration)
4. Run Cell 3 (load data)
5. Run Cell 4 (visualization engine)
6. Run Cell 5 (interactive selector)
```

**That's it!** The interactive selector will automatically display the highest validity pattern.

### Daily Use
- Just click the dropdown to browse patterns
- Select any pattern and click "📊 Visualize"
- Use the advanced section for batch analysis

---

## 🔄 Loading New Backtests

### Method 1: Update Configuration (Recommended)
Edit Cell 2:
```python
PATTERNS_CSV = Path('./my_new_backtest_patterns.csv')
OHLCV_CSV = Path('./my_new_ohlcv_data.csv')
```

Then run cells 2-5 in sequence.

### Method 2: Keep Multiple Versions
Save copies like:
- `visual_lab_v1.ipynb` (baseline backtest)
- `visual_lab_v2.ipynb` (parameter test A)
- `visual_lab_v3.ipynb` (parameter test B)

Each configured with different data files.

---

## 📊 What You Get

### Visual Features
- **TradingView-quality charts** with candlesticks
- **Pattern structure overlay** (gray trend line + blue QML zig-zag)
- **Labeled points** (LS, H, LL, RS)
- **Trade zones** (green = take profit zone, red = stop loss zone)
- **Entry marker** (blue triangle)
- **Info box** with entry, SL, TP, and Risk:Reward ratio
- **60+ bars context** before and after pattern

### Data Access
- Full pattern dataframe: `patterns_df`
- Full OHLCV dataframe: `ohlcv_df`
- Quick functions: `show_top_patterns()`, `quick_plot()`

---

## 📁 Required File Formats

### Pattern CSV
**Required columns:**
- `pattern_id`, `pattern_type`, `validity_score`
- `entry_price`, `stop_loss`, `take_profit`
- `TS_Date`, `TS_Price` (Trend Start)
- `P1_Date`, `P1_Price` (Base)
- `P2_Date`, `P2_Price` (Left Shoulder)
- `P3_Date`, `P3_Price` (Head)
- `P4_Date`, `P4_Price` (Lower Low / Higher High)
- `P5_Date`, `P5_Price` (Right Shoulder / Entry)

### OHLCV CSV
**Required columns:**
- `time` (timestamp, must cover pattern date ranges)
- `Open`, `High`, `Low`, `Close`, `Volume`

**Date format:** ISO format recommended (`2023-01-15 09:00:00`)

---

## 💡 Pro Tips

1. **Fastest Navigation**: Use the interactive dropdown (sorted by validity)
2. **Batch Analysis**: Use `show_top_patterns(20)` to view metrics at a glance
3. **Quick Checks**: Use `quick_plot(id)` when you know the pattern ID
4. **Compare Backtests**: Keep multiple notebook copies with different configs
5. **Export Charts**: Right-click any chart → Save Image

---

## 🎨 Customization

All visualization settings are in Cell 4 (`plot_pattern` function):
- Chart size: `figsize=(18, 10)`
- Bar padding: `start_idx = max(0, ts_idx - 60)`
- Colors: `#26a69a` (green), `#ef5350` (red), `#2962ff` (blue)
- Label offset: `label_offset = price_range * 0.04`

---

## ✅ Benefits

### Professional
- Clean, organized structure
- Clear step-by-step workflow
- Production-ready documentation

### Flexible
- Easy to switch between backtests
- Reusable for any parameter tests
- Programmatic access when needed

### Efficient
- Interactive GUI for quick browsing
- Auto-sorted by quality (validity)
- One-click visualization

### Complete
- Full pattern structure visualization
- Trade zones and risk metrics
- Historical context (60+ bars)

---

## 📞 Need Help?

Check Cell 7 in the notebook for:
- File format specifications
- Loading instructions
- Common troubleshooting

---

**Version:** 2.0 - Professional Edition  
**Last Updated:** December 31, 2025  
**Status:** Production Ready ✅
