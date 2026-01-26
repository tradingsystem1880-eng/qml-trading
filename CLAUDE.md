# QML Trading System

## Project Overview
Quantitative validation framework for Quasimodo (QML) chart pattern detection in crypto markets.

## Current State
- Framework: PRODUCTION-READY
- Detection Logic: ✅ Phase 7.9 COMPLETE - Verified edge (PF 1.23, DSR 0.986)
- ML Meta-Labeling: ❌ Phase 8.0 FAILED - ML has no predictive power (AUC 0.53)
- **Recommendation**: Use Phase 7.9 baseline with fixed 1% position sizing
- Data: Fixed, parquet files working (30 symbols × 3 timeframes)
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
5. ~~Phase 7.6: Hierarchical swing detection + data expansion~~ DONE
6. ~~Phase 7.7: Extended optimization (56 hours, 6 objectives)~~ DONE (2026-01-26)
7. ~~Phase 7.8: Add volume/momentum/regime filters~~ DONE (2026-01-26)
8. ~~Phase 7.9: Optimization with filters~~ DONE - Verified edge (PF 1.23, DSR 0.986)
9. ~~Phase 8.0: ML Meta-Labeling~~ FAILED - ML has no predictive power (AUC 0.53)
10. **Phase 9.0: Paper trading with Phase 7.9 baseline** (NEXT)

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
# Main Dashboard (TradingView-grade charts)
streamlit run qml/dashboard/app_v2.py

# Test chart component standalone
streamlit run qml/dashboard/test_chart.py

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

## Dashboard Polish & Real Data Connection (2026-01-22)

Final polish pass connecting all dashboard pages to real backtest data.

### Changes Made

1. **Analytics Page** - Now displays real backtest metrics
   - Reads from `results/experiments.db` SQLite database
   - Shows actual equity curve, trade statistics, drawdown analysis
   - Falls back to demo data if no experiments exist

2. **Dashboard Page** - Connected to real data
   - Loads metrics from latest backtest experiments
   - Shows real trade history when available
   - Mock data fallback for new users

3. **Settings Page** - Fixed hardcoded paths
   - Now uses `PROJECT_ROOT` constant for portability
   - Works correctly regardless of working directory

4. **Dead Code Removal**
   - Removed `qml/dashboard/components/backtest.py` (260 lines duplicate)
   - Moved `src/dashboard/pattern_lab/` to archive
   - Cleaned unused imports in `pattern_lab_page.py`

5. **New Files Added**
   - `src/data/schemas.py` - Data model definitions
   - `src/data/sqlite_manager.py` - Database utilities
   - `tests/test_pattern_lab_integration.py` - Integration tests
   - `docs/planning/` - Project planning documents

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
| `qml/dashboard/app_v2.py` | Main dashboard, contains `find_swing_points()` and `map_to_geometry()` |
| `src/dashboard/components/tradingview_chart.py` | Chart HTML generation with `_generate_chart_html()` |
| `qml/dashboard/test_chart.py` | Standalone test page with synthetic data generators |

### Critical Functions

**`find_swing_points(df, lookback=3)`** in `app_v2.py`:
- Detects swing highs/lows by comparing to surrounding candles
- Returns list of `{time, price, type, idx}` dicts sorted by time

**`map_to_geometry(pattern, df)`** in `app_v2.py`:
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

---

## Phase 7.6: Detection Logic Overhaul (2026-01-23)

Improved swing detection and expanded data coverage.

### Changes
- **Hierarchical Swing Detection** - 3-layer system (geometry → significance → context)
- **Prior Trend Validator** - Ensures meaningful trend before pattern
- **Data Expansion** - 30 symbols × 3 timeframes (1h, 4h, 1d)

### New Files
- `src/detection/hierarchical_swing.py` - 3-layer swing detector
- `src/detection/trend_validator.py` - Prior trend validation + TrendRegimeValidator
- `src/optimization/parallel_runner.py` - Parallel detection across symbols

### Results
- 654 patterns detected across 32 symbols
- Mean quality: 61.5%

---

## Phase 7.7: Extended Optimization (2026-01-26) ✅ COMPLETE

**56-hour Bayesian optimization across 6 objective functions.**

### Optimization Results

| Objective | Best Score | Notes |
|-----------|------------|-------|
| COUNT_QUALITY | 0.9138 | ✅ Excellent pattern detection |
| SHARPE | 0.0366 | ⚠️ Barely positive |
| EXPECTANCY | 0.0460 | ⚠️ Low avg profit per trade |
| PROFIT_FACTOR | 0.2402 | ⚠️ Below breakeven |
| MAX_DRAWDOWN | 0.3295 | ⚠️ High drawdowns |
| COMPOSITE | 0.3348 | Combined score |

### Best COMPOSITE Parameters
```
Patterns: 1,583 across 22 symbols
Quality: 55.5%
Win Rate: 41.9%
Sharpe: -0.06
Profit Factor: 0.88
TP/SL: 4.53 / 1.68 ATR
Risk:Reward: 1.13
```

### Key Files Created
- `src/optimization/trade_simulator.py` - MAE/MFE tracking, trailing stops
- `src/optimization/objectives.py` - 6 objective functions
- `src/optimization/extended_runner.py` - Walk-forward + cluster validation
- `scripts/run_phase77_optimization.py` - Main optimization CLI

### Results Location
```
results/phase77_optimization/
├── all_objectives_summary.json    # Full results
├── count_quality/final_results.json
├── sharpe/final_results.json
├── expectancy/final_results.json
├── profit_factor/final_results.json
├── max_drawdown/final_results.json
└── composite/final_results.json
```

### Key Insight
**Pattern detection works well** (0.91 quality score), but **profitability is marginal**.
Pure price-action QML patterns hover around breakeven. Need additional filters.

---

## Phase 7.8: Volume/Momentum/Regime Filters (2026-01-26) ✅ COMPLETE

Addressed marginal profitability by adding filtering layers to pattern scoring.

### Changes Made

1. **Activated Phase 7.6 Metrics** (Previously unused)
   - Wired `df` and `trend_result` to scorer in parallel_runner, extended_runner, multi_symbol_detection
   - Volume spike, path efficiency, and trend strength scores now calculate properly

2. **Made 7.6 Weights Optimizable**
   - volume_spike_weight, path_efficiency_weight, trend_strength_weight now tunable (0.05 - 0.10)

3. **Added Regime Suitability Scoring** (Main feature)
   - Hard rejection: TRENDING regime with ADX > 35 → pattern rejected
   - Soft scoring: Regime affects quality score (RANGING=1.0, VOLATILE=0.6, EXTREME=0.5, TRENDING=0.2)

4. **Rebalanced Scoring Weights** (8 components, sum = 1.0)
   - head_extension: 22%, bos_efficiency: 18%, shoulder: 12% (fixed)
   - swing: 8% (auto-calculated), volume: 10%, path: 10%, trend: 10%, regime: 10%

### Files Modified
- `src/detection/config.py` - Regime scoring params, rebalanced weights
- `src/detection/pattern_scorer.py` - `_calculate_regime_suitability()`, updated `score()`
- `src/optimization/parallel_runner.py` - Wired df/trend_result/regime_result
- `src/optimization/extended_runner.py` - Same wiring
- `scripts/run_phase77_optimization.py` - 6 optimizable weight params
- `scripts/multi_symbol_detection.py` - Added regime detection

### Commands
```bash
# Test multi-symbol detection with Phase 7.8
python scripts/multi_symbol_detection.py --symbols ETHUSDT,BNBUSDT,SOLUSDT

# Run short optimization test
python scripts/run_phase77_optimization.py --objective composite --iterations 100
```

---

## Phase 7.9: Optimization with Filters (2026-01-26) ✅ COMPLETE

**Ran optimization with Phase 7.8 regime filters. Achieved verified edge.**

### Results
| Metric | Value | Notes |
|--------|-------|-------|
| Profit Factor | 1.23 | ✅ Above breakeven! |
| Sharpe Ratio | +0.08 | ✅ Positive |
| DSR | 0.986 | ✅ Statistically significant |
| Expectancy | +0.18 R | ✅ Positive per trade |
| Win Rate | 42.1% | Typical for trend-following |

### Key Insight
Phase 7.8 filters (especially regime filtering) pushed profitability above breakeven.
The strategy now has a **statistically verified edge** ready for ML enhancement.

---

## Phase 8.0: ML Meta-Labeling for Position Sizing (2026-01-26) ❌ ML REJECTED

**ML attempted but FAILED validation. Using Phase 7.9 baseline.**

### What Happened
1. Initial run showed PF 7.11 (unrealistic)
2. Investigation revealed **train/test data leakage** (model tested on same data it trained on)
3. Proper validation with 70/30 time-split showed **AUC 0.53** (random)
4. **ML has NO predictive power** - features don't predict trade outcomes

### Key Findings

| Issue | Discovery |
|-------|-----------|
| Initial PF 7.11 | FAKE - train/test leakage |
| Proper Test AUC | 0.53 (essentially random) |
| CV Fold Variance | 0.29 - 0.69 (unstable) |
| Real Decision | **USE_BASELINE** |

### Root Cause
The original pipeline trained on ALL trades, then tested on those SAME trades.
The model memorized outcomes, not learned patterns.

### Lessons Learned
1. **Always split train/test by TIME** (not random)
2. **AUC > 0.55** required before trusting predictions
3. **PF > 3.0 is almost always fake** (Renaissance avg 2-3)
4. **Check that test data was never seen in training**

### Approach (Minimal Changes)
Based on López de Prado's meta-labeling framework:
1. Train binary classifier to predict trade outcome (TP hit before SL)
2. Use ML confidence to scale position sizes via Kelly criterion
3. Production gate: ML must improve PF by 10%+ or fallback to baseline

### New Files Created

| File | Purpose |
|------|---------|
| `scripts/validate_objective_robustness.py` | Walk-forward validation of profit_factor objective |
| `src/ml/meta_trainer.py` | MetaTrainer with purged CV and feature pruning |
| `src/ml/kelly_sizer.py` | Kelly criterion position sizing |
| `src/ml/production_gate.py` | ML vs baseline comparison gate |
| `scripts/run_phase80_ml.py` | Orchestration CLI for full pipeline |

### Key Classes

**MetaTrainer** (`src/ml/meta_trainer.py`):
```python
from src.ml import MetaTrainer, MetaTrainerConfig

config = MetaTrainerConfig(n_folds=5, max_features=15, min_auc_threshold=0.55)
trainer = MetaTrainer(config)
result = trainer.train(features_df, labels)
# result.model, result.selected_features, result.mean_auc
```

**KellySizer** (`src/ml/kelly_sizer.py`):
```python
from src.ml import KellySizer, KellyConfig

kelly = KellySizer(KellyConfig(kelly_fraction=0.5))  # Half-Kelly for safety
result = kelly.calculate_position_size(
    ml_confidence=0.72,
    win_rate=0.42,
    avg_win_r=2.0,
    avg_loss_r=1.0,
    account_equity=100_000,
)
# result.risk_pct, result.action ('FULL', 'HALF', 'MINIMUM', 'SKIP')
```

**ProductionGate** (`src/ml/production_gate.py`):
```python
from src.ml import ProductionGate

gate = ProductionGate()
result = gate.run_gate(trades, ml_confidences, position_sizes, baseline_metrics)
# result.decision: 'DEPLOY_ML' or 'USE_BASELINE'
```

### Position Sizing Tiers
| Confidence | Action | Kelly Multiplier |
|------------|--------|------------------|
| ≥ 70% | FULL | 100% |
| ≥ 55% | HALF | 50% |
| ≥ 45% | MINIMUM | 25% |
| < 45% | SKIP | 0% |

### Commands
```bash
# Run FIXED ML pipeline (with proper train/test split)
python scripts/run_phase80_ml_fixed.py --full-pipeline

# Run diagnostic checks
python scripts/check_temporal_integrity.py
python scripts/check_lookahead_bias.py
```

### Diagnostic Scripts Created
| Script | Purpose |
|--------|---------|
| `check_temporal_integrity.py` | Verify data splits and symbol matching |
| `check_lookahead_bias.py` | Detect future data leakage in features |
| `run_phase80_ml_fixed.py` | Proper pipeline with 70/30 time split |

### Critical Principle
**ML must OUTPERFORM Phase 7.9 baseline** on HELD-OUT data.
Phase 8.0 ML failed this test (AUC 0.53 = random).

---

## Current State (2026-01-26)

**USE PHASE 7.9 BASELINE** - ML meta-labeling provides no benefit.

| System | PF | Status |
|--------|-----|--------|
| Phase 7.9 Baseline | 1.23 | ✅ USE THIS |
| Phase 8.0 ML | N/A | ❌ REJECTED |

The baseline with fixed 1% position sizing is the correct approach.

---

## What's Next

**Phase 8.0 ML rejected. Next steps:**

1. **Paper trading** with Phase 7.9 baseline (fixed 1% sizing)
2. **Forward validation** - monitor performance on new data
3. **Feature engineering** - try different features if ML attempted again
4. **Alternative approaches:**
   - Higher timeframe alignment
   - Volume confirmation filters
   - Sentiment/funding rate integration

The Phase 7.9 baseline (PF 1.23, DSR 0.986) is ready for live testing.
