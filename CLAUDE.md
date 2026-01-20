# QML Trading System

## Project Overview
Quantitative validation framework for Quasimodo (QML) chart pattern detection in crypto markets.

## Current State
- Framework: PRODUCTION-READY
- Detection Logic: PLACEHOLDER - needs iteration and optimization (DO NOT FIX YET)
- Data: Fixed, parquet files working
- Dashboard v2: ✅ FULLY WORKING - JARVIS theme, all HTML rendering fixed (2026-01-18)

## Key Directories
- cli/ - Command line entry points (run_backtest.py is main runner)
- src/ - Core library (detection, reporting, core data models)
- qml/ - QML specific code (needs consolidation with src/)
- config/ - YAML configurations
- data/ - Price data (parquet files)
- results/ - Experiment outputs
- archive/ - Legacy code (can ignore)

## User Context
- Beginner Python, learning as we go
- Wants to understand the system, not just run it
- Goal: Validate QML pattern detection with rigorous statistical testing
- Manual trade execution (not automated)
- Crypto focus: Hyperliquid/Bybit
- Timeframes: 1H, 4H, Daily

## Current Priority
1. ~~Configure Claude Code properly~~ DONE
2. ~~Document and understand the system~~ DONE
3. ~~Verify full validation pipeline works~~ DONE
4. ~~Dashboard salvage assessment~~ DONE - Improved!
5. Detection logic iteration (NEXT - system is ready)

## Commands Reference
- Run backtest: `python -m cli.run_backtest --symbol BTCUSDT --timeframe 4h`
- Fetch fresh data: `python -c "from src.data_engine import build_master_store; build_master_store('BTC/USDT', ['4h'], years=2)"`
- Config location: config/default.yaml
- Results location: results/experiments.db (SQLite) + results/atr/*.html (reports)

## Do NOT
- Change detection logic parameters yet (waiting for full system verification)
- Delete anything in archive/ without asking
- Assume I understand - explain everything

---

## Session Quick Reference

When starting a new session, tell Claude:
> "Read CLAUDE.md, we're working on the QML trading system"

## System Verification Checklist

| Component | Status | Notes |
|-----------|--------|-------|
| Data Pipeline | ✅ VERIFIED | Fetches from Binance, saves parquet |
| Backtest Engine | ✅ VERIFIED | cli/run_backtest.py works |
| Flight Recorder | ✅ VERIFIED | SQLite + HTML dossiers |
| Validation Pipeline | ✅ VERIFIED | Fixed bugs in validator.py (see Fixes section) |
| Dashboard (qml/) | ✅ VERIFIED | **USE app_v2.py** - JARVIS theme, HTML fully fixed |
| Dashboard (src/) | ✅ VERIFIED | Works but fewer features (32KB) |
| ML Training | ✅ VERIFIED | XGBoost predictor works (see ML section) |

## Validation Pipeline Details

All validation components tested and working:

1. **PermutationTest** - Tests statistical significance via shuffle testing
2. **MonteCarloSim** - Risk analysis via equity path simulation (VaR, CVaR, Risk of Ruin)
3. **BootstrapResample** - Confidence intervals on metrics
4. **PurgedWalkForwardEngine** - Rolling train/test with purge/embargo gaps
5. **StrategyValidator** - High-level orchestrator combining all above
6. **run_validation_suite()** - Convenience function for running all validators

### Usage Example
```python
from src.validation import run_validation_suite

# After running backtest
suite = run_validation_suite(results, trades=trade_list)
for r in suite.results:
    print(f'{r.validator_name}: {r.status}')
```

### Fixes Applied (2026-01-18)
- Fixed import errors in `src/validation/validator.py` (MonteCarloSimulator → MonteCarloSim, BlockBootstrap → BootstrapResample)
- Fixed constructor calls to use config dicts instead of keyword args
- Fixed API calls from `.run()` to `.validate()`
- Fixed attribute access for ValidationResult (p_value, metrics dict)

## Dashboard Comparison

Both dashboards verified working. **Recommendation: Use qml/dashboard**

| Feature | qml/dashboard | src/dashboard |
|---------|---------------|---------------|
| Size | ~1900 lines | 867 lines |
| VRD Validation Reports | ✅ Full | ⚠️ Basic |
| Backtest Runner | ✅ Real Engine | ❌ No |
| Neuro Lab (ML) | ✅ Yes | ❌ No |
| Paper Trading | ✅ Real Detection | ✅ Yes |
| TradingView Charts | ✅ Yes | ✅ Yes |
| Pattern Scanner | ✅ Yes | ✅ Yes |
| Settings Persistence | ✅ YAML | ❌ No |

### Dashboard Improvements (2026-01-18)
1. **Settings Page** - Now loads/saves to `config/default.yaml`
2. **Backtest Runner** - Uses real CLI backtest engine (not random simulation)
   - Proper signal detection via `get_detector("atr")`
   - Bar-by-bar SL/TP simulation
   - Equity curve visualization
3. **Paper Trading** - Uses real pattern detection
   - Scans for actual QML patterns
   - Shows validity, entry/SL/TP, R:R
   - Track wins/losses manually
4. **Fixed Bugs** - Removed incorrect imports in Neuro-Lab

### Launch Commands
```bash
# NEW: Professional Dashboard v2 (TradingView-grade charts)
cd /Users/hunternovotny/Desktop/QML_SYSTEM
streamlit run qml/dashboard/app_v2.py

# Test chart component standalone
streamlit run qml/dashboard/test_chart.py

# Original Dashboard (legacy)
streamlit run qml/dashboard/app.py

# Alternative: SRC Dashboard (lighter)
streamlit run src/dashboard/app.py
```

## Dashboard v5.0 - JARVIS-Style Ultra Premium (2026-01-18)

Complete JARVIS-style (Iron Man) dashboard overhaul with premium holographic UI.

### Critical Fix: HTML Rendering Issue (2026-01-18) - FULLY RESOLVED

**Problem**: Raw HTML code (like `</div>` or full div tags) was displaying as text instead of rendering properly.

**Root Cause #1 - Split HTML**: Streamlit does NOT support opening a `<div>` in one `st.markdown()` call and closing it in another.

**Root Cause #2 - Multi-line f-strings**: Triple-quoted f-strings with leading whitespace/indentation cause Streamlit to render HTML as literal text. This was the main culprit.

**Bad Pattern** (multi-line f-string with indentation):
```python
st.markdown(f"""
    <div style="background: #1a1a2e;">
        <div style="color: #00d4aa;">{value}</div>
    </div>
""", unsafe_allow_html=True)  # BROKEN - shows raw HTML
```

**Fixed Pattern** (single-line string concatenation):
```python
html = '<div style="background: #1a1a2e;">'
html += f'<div style="color: #00d4aa;">{value}</div>'
html += '</div>'
st.markdown(html, unsafe_allow_html=True)  # WORKS
```

**All Sections Fixed**:
- `render_header()` - Page headers
- `render_metric_card()` - Metric display cards
- `render_circular_metric()` - Win rate circle
- `render_monthly_heatmap()` - Monthly P&L grid
- `render_stat_row()` - Key metrics rows
- `render_trade_row()` - Trade history rows
- `render_mini_sparkline()` - Profit trend sparkline
- WIN RATIO panel - Circle + wins/losses
- KEY METRICS panel - Sharpe, expectancy, etc.
- MONTHLY P&L panel - Heatmap + YTD
- PROFIT TREND panel - Sparkline + value
- TRADE HISTORY panel - Full trade table
- SETUP ANALYSIS panel - Stats grid
- Scanner status bar - Ready indicator + results count
- Scanner results banner - Patterns detected
- Validation executive summary - Verdict banner
- Backtest status indicators - Engine ready

### New JARVIS Theme Features
- Deep space color palette (`#020408` to `#0c141e`)
- Neon accent colors (`#00ffcc` cyan, `#ff4757` red)
- CSS animations: pulse-glow, scanline, shimmer, data-flow
- Animated grid background
- Holographic card effects with glows
- Premium typography: Inter, JetBrains Mono, Space Grotesk, Orbitron

### New Components
- `qml/dashboard/core/design_system.py` - Professional color palette, typography, spacing
- `qml/dashboard/core/pro_chart.py` - Chart component with position boxes & annotations
- `qml/dashboard/app_v2.py` - New dashboard entry point
- `qml/dashboard/test_chart.py` - Chart component test page

### Features
1. **Position Boxes** - TP/SL zones extending to chart right edge
   - Green TP zones (TP1 darker, TP2 lighter)
   - Red SL zone
   - Labels inside boxes
   - Works for both LONG and SHORT positions

2. **Pattern Annotations**
   - Numbered swing points (1, 2, 3, 4, 5)
   - Blue dashed connection lines
   - QM-ZONE shading

3. **Professional Styling**
   - Dark JARVIS theme (matches Iron Man aesthetic)
   - Proper candle colors (green bullish, red bearish)
   - Clean, minimal grid with subtle glow

---

## Pattern Visualization - FINAL SPECIFICATION (2026-01-20)

**STATUS: LOCKED IN - DO NOT MODIFY WITHOUT EXPLICIT USER REQUEST**

The pattern visualization system is complete and finalized. This section documents the exact specification.

### Visual Components

1. **Pattern Swing Points (P1-P5)** - Blue numbered markers
   - Placed at ACTUAL candle highs/lows (not interpolated)
   - Uses `find_swing_points()` algorithm with lookback comparison
   - Numbered 1-5 in chronological order (sorted by time)
   - Blue circles with white text labels

2. **Pattern Connection Line** - Blue dashed zigzag
   - Connects P1 → P2 → P3 → P4 → P5
   - Color: `#2962FF` (TradingView blue)
   - Style: Dashed line, 2px width

3. **Prior Trend Line** - Orange solid line
   - Shows the trend that preceded the pattern
   - For BULLISH QML: Shows prior downtrend (Lower Highs → Lower Lows)
   - For BEARISH QML: Shows prior uptrend (Higher Highs → Higher Lows)
   - Color: `#f59e0b` (amber/orange)
   - Markers labeled: HH, HL, LH, LL as appropriate

4. **Position Box** - Trade visualization from entry to outcome
   - Uses TradingView **Baseline Series** (not area series)
   - **Green zone**: Profit area (entry to TP) - `rgba(34, 197, 94, 0.3)`
   - **Red zone**: Risk area (entry to SL) - `rgba(239, 68, 68, 0.3)`
   - SL zone is BOUNDED at actual SL price (does not extend to chart bottom)
   - Baseline (pivot point) is set at entry price

5. **Price Lines** - Horizontal reference lines
   - Entry: Cyan solid line (`#0ea5e9`)
   - Stop Loss: Red dashed line (`#ef4444`)
   - Take Profit: Green dotted line (`#22c55e`)

### Display Window Rules

- **Minimum candles**: 50 before pattern + enough after to show TP/SL hit
- **Auto-extend**: If TP or SL is hit, window extends to show the outcome
- **Default extension**: 20 candles after detection if no outcome found

### Key Files

| File | Purpose |
|------|---------|
| `qml/dashboard/app.py` | Main dashboard, contains `find_swing_points()` and `map_to_geometry()` |
| `src/dashboard/components/tradingview_chart.py` | Chart HTML generation with `_generate_chart_html()` |
| `qml/dashboard/test_chart.py` | Standalone test page with synthetic data generators |

### Critical Functions

**`find_swing_points(df, lookback=3)`** in `app.py`:
- Detects swing highs/lows by comparing to surrounding candles
- Returns list of `{time, price, type, idx}` dicts sorted by time

**`map_to_geometry(pattern, df)`** in `app.py`:
- Maps detector output to 5 swing points on actual candle data
- Handles timezone conversion (removes tz for comparison)
- Returns chronologically ordered P1-P5 coordinates

**`_generate_chart_html()`** in `tradingview_chart.py`:
- Builds complete HTML with TradingView Lightweight Charts v4.1.0
- Handles trend line, pattern line, position box, and markers
- Uses baseline series for bounded position zones

### DO NOT CHANGE

- Marker numbering order (1-5 chronological)
- Color scheme (blue pattern, orange trend, green/red position)
- Swing point detection algorithm
- Baseline series approach for position box
- Display window calculation logic

Any changes to visualization require explicit user approval.

## ML Training Details

XGBoost-based trade outcome predictor. Trains on historical trades to predict win probability.

### Training Command
```bash
python3 -m cli.run_backtest --symbol BTCUSDT --timeframe 4h --train-ml
```

### Features Used
- validity_score, entry_price, atr_at_entry
- sl_distance_pct, tp_distance_pct, risk_reward_ratio
- hour_of_day, day_of_week
- volatility_percentile, trend_strength, bars_since_last_trade

### Model Output
- Model: `results/models/xgb_latest.json`
- Metadata: `results/models/xgb_latest.meta.json`
- Minimum trades needed: 20

### Usage Example
```python
from src.ml.predictor import XGBoostPredictor

predictor = XGBoostPredictor('results/models/xgb_latest.json')
win_prob = predictor.predict(signal_dict)
```

## What's Next
All components verified and dashboard improved. Ready to iterate on detection logic when desired.

### To Start Detection Logic Work
1. Review current detector: `src/detection/v2_atr.py`
2. Run backtest to see current performance: `python -m cli.run_backtest --symbol BTCUSDT --timeframe 4h`
3. Experiment with parameters in `config/default.yaml` (or via dashboard Settings)
4. Focus areas: validity scoring, pattern geometry rules, SL/TP placement
