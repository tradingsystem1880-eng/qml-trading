# QML TRADING SYSTEM - MASTER PROJECT PLAN
## Compiled from Planning Conversations | January 2026

---

# TABLE OF CONTENTS
1. [Project Overview](#1-project-overview)
2. [Architecture & Current State](#2-architecture--current-state)
3. [Critical Constraints](#3-critical-constraints)
4. [Dashboard Redesign Specification](#4-dashboard-redesign-specification)
5. [Chart Visualization System (LOCKED)](#5-chart-visualization-system-locked)
6. [Color Scheme - Arctic Pro](#6-color-scheme---arctic-pro)
7. [Validation Framework](#7-validation-framework)
8. [Feature Engineering](#8-feature-engineering)
9. [Prop Firm Rules - Breakout](#9-prop-firm-rules---breakout)
10. [Recommended Libraries](#10-recommended-libraries)
11. [Implementation Phases](#11-implementation-phases)
12. [Tab Structure](#12-tab-structure)
13. [DeepSeek Research Insights](#13-deepseek-research-insights)
14. [Claude Code Workflow](#14-claude-code-workflow)

---

# 1. PROJECT OVERVIEW

## Goal
Build a quantitative trading system focused on validating QML (Quasimodo) chart patterns in cryptocurrency markets, with rigorous statistical validation before deployment on a Breakout prop firm funded account ($100K).

## Philosophy
- **Validation First**: Build the measurement system before optimizing what's being measured
- **90% Rule**: Prefer simplicity over complexity - choose 90% of goals with clean code over 100% with messy code
- **No Spaghetti**: Keep code clean and structure clean
- **Understanding Over Execution**: Understand trading logic and results rather than treating implementation as black box

## Current Focus
1. Dashboard redesign - see clean visuals of trades + metrics
2. Detection logic iteration - test different swing point algorithms
3. Results analysis - data-driven refinement
4. Deploy on prop firm - with Kelly sizing within their rules

---

# 2. ARCHITECTURE & CURRENT STATE

## VRD 2.0 Framework Assessment
The existing framework was assessed as **SALVAGEABLE** with solid, production-quality architecture:

| Component | Quality | Status |
|-----------|---------|--------|
| VRD 2.0 Framework Design | ⭐⭐⭐⭐⭐ | Keep - institutional-grade |
| Purged Walk-Forward | Proper López de Prado | Keep |
| Statistical Suite | Industry standard | Keep |
| 170+ Feature Library | Comprehensive | Keep |
| Regime Detection (HMM) | Correct approach | Keep |
| QML Strategy (61-68% WR) | Passed validation | Revisit |
| Paper Trading System | Operational | Keep |

## Current Files Structure
**Files to Keep:**
- `qml/dashboard/app_v2.py` → Main dashboard file
- `qml/dashboard/core/` → Design system and chart components
- `src/dashboard/components/tradingview_chart.py` → Chart generation

**Files to Archive:**
- `src/dashboard/pattern_lab/` → Dash version (move to archive)
- `qml/dashboard/app.py` → Old version

## Technology Stack
- **Dashboard**: Streamlit (already working, simpler than Dash)
- **Charts**: TradingView Lightweight Charts via `st.components.html()`
- **Backend**: Python
- **Validation**: VRD 2.0 framework

---

# 3. CRITICAL CONSTRAINTS

## 🔒 LOCKED - DO NOT CHANGE
The TradingView Lightweight Charts pattern visualization with:
- Prior trend lines
- P1-P5 numbered swing points
- Position boxes (TP/SL zones)
- Entry/Stop/Take-Profit horizontal lines

**This is LOCKED. Claude Code must be explicitly told NOT to touch this.**

## Detection Logic Status
- Current swing/pivot/rolling window detection is **INTENTIONALLY INCOMPLETE**
- Pattern detection/strategy logic is **INTENTIONALLY INCOMPLETE**
- Dashboard must be built first to visualize and iterate on detection logic

## Code Principles
1. No spaghetti code
2. No unnecessarily complex code
3. Use existing libraries over custom implementations
4. 90% goal achievement with clean code > 100% with messy code

---

# 4. DASHBOARD REDESIGN SPECIFICATION

## Layout: Hybrid Approach
**Command Bar (Always Visible) + Tabbed Content**

```
┌─────────────────────────────────────────────────────────────────────────┐
│  COMMAND BAR (Always Visible - Never Scrolls)                          │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐    [BTC: $XX,XXX]    │
│  │Win% │ │Sharp│ │P.F. │ │MaxDD│ │Expct│ │Kelly│    [System: ● LIVE]  │
│  │61.4%│ │1.82 │ │2.14x│ │-12% │ │$127 │ │2.3% │    [Patterns: 3 🔔]  │
│  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘                       │
├─────────────────────────────────────────────────────────────────────────┤
│  TABS: [📊 Dashboard] [🔬 Pattern Lab] [⚡ Backtest] [📈 Analytics]    │
│        [🧪 Experiments] [🧠 ML Training] [⚙️ Settings]                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│                    Tab Content Changes Here                             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Why Hybrid Approach
| Approach | Pros | Cons |
|----------|------|------|
| Single Page | Everything visible | Cluttered, overwhelming |
| Pure Tabs | Clean, focused | Lose sight of key metrics |
| **Hybrid** ✅ | Best of both | Slightly more complex |

## Key Design Goals
- Professional-grade interface rivaling TradingView, Binance Pro
- Information density - many small cards vs few large panels
- Clean, minimal annotations
- Key metrics ALWAYS visible in command bar

---

# 5. CHART VISUALIZATION SYSTEM (LOCKED)

## Position Box Specifications

### Long Position Box
```
    Price
      ↑
 TP2 ─┼────┬─────────────────────────────────────────┐
      │    │           TP2 ZONE (lighter green)      │ ← "TP 2" label
 TP1 ─┼────┼─────────────────────────────────────────┤
      │    │           TP1 ZONE (green)              │ ← "TP 1" label
Entry─┼────┼═════════════════════════════════════════┤ ← Entry line (blue)
      │    │         STOP LOSS ZONE (red)            │ ← "SL" label
  SL ─┼────┴─────────────────────────────────────────┘
      └──────────────────────────────────────────────→ Time
```

### Position Box Colors (EXACT)
```typescript
const POSITION_COLORS = {
  // Profit zones
  TP_PRIMARY: 'rgba(38, 166, 91, 0.25)',      // TP1 - More visible green
  TP_SECONDARY: 'rgba(38, 166, 91, 0.15)',    // TP2, TP3 - Lighter green
  TP_BORDER: 'rgba(38, 166, 91, 0.5)',
  
  // Stop loss zone
  SL_FILL: 'rgba(231, 76, 60, 0.25)',         // Red with transparency
  SL_BORDER: 'rgba(231, 76, 60, 0.5)',
  
  // Entry line
  ENTRY_LINE: '#3498db',                       // Blue
};
```

### Swing Point Markers
```
For NUMBERED points (1, 2, 3, 4, 5) - QML pattern:
           2                           4
          /\                          /\
         /  \                        /  \
        /    \          3           /    \        5
       /      \        /\          /      \      /
      1        \      /  \        /        \    /
                \    /    \      /          \  /
                 \  /      \    /            \/

- Numbers positioned ABOVE swing highs, BELOW swing lows
- Small offset from high/low (5-8 pixels)
- Font: 11px, semi-bold, monospace
- Color: #3498db (blue)
```

### Connection Line Styles
```typescript
const LINE_STYLES = {
  PATTERN_CONNECTION: {
    stroke: '#3498db',
    strokeWidth: 1.5,
    strokeDasharray: '6, 4',  // Dashed line
    opacity: 0.8,
  },
  BOS_LINE: {
    stroke: '#f39c12',  // Orange
    strokeWidth: 1,
    strokeDasharray: 'none',  // Solid
  },
};
```

### Chart Display Requirements
- Load minimum 500 bars of data
- Display 100-150 bars in view (condensed like TradingView)
- Allow zoom from 50 to 300 bars visible
- Pattern centered with equal context on both sides
- Subtle grid lines (opacity 0.05-0.1)
- Price axis on right, time axis on bottom

---

# 6. COLOR SCHEME - ARCTIC PRO

## Confirmed Palette
```typescript
const ARCTIC_PRO = {
  // Backgrounds
  background: '#0B1426',      // Deep navy
  cardBg: '#162032',          // Slate blue
  
  // Accent
  accent: '#3B82F6',          // Electric blue
  
  // Semantic
  success: '#10B981',         // Emerald green
  danger: '#EF4444',          // Red
  warning: '#F59E0B',         // Amber
  
  // Text
  textPrimary: '#F8FAFC',
  textMuted: '#64748B',
  
  // Chart specific
  bullish: '#26a69a',         // TradingView green
  bearish: '#ef5350',         // TradingView red
};
```

## Typography
```typescript
const typography = {
  fontFamily: {
    sans: "'Inter', -apple-system, sans-serif",
    mono: "'JetBrains Mono', 'Fira Code', monospace",
  },
  
  // RULE: All numerical data (prices, percentages) MUST use mono
  // RULE: All labels and UI text use sans
};
```

---

# 7. VALIDATION FRAMEWORK

## Statistical Testing Requirements

### Permutation Testing
- Shuffle market returns 1,000-10,000 times
- Target p-value < 0.05 (preferably < 0.01)
- Minimum 50 trades for validity, 200+ desirable

### Walk-Forward Analysis
- 70/30 split (train/test)
- Roll window forward through history
- Target Walk-Forward Efficiency (WFE) > 50-60%
- Run 5-10 walk-forward periods minimum

### Monte Carlo Simulation
- Shuffle trade order 1,000+ times
- Expect max drawdown 30-50% worse than backtest
- Expect 8-9 consecutive losses (vs 5-6 historical)
- Use 95% confidence intervals (99% for conservative)

### Lopez de Prado Tests
- **Probability of Backtest Overfitting (PBO)**: Target < 5%
- **Deflated Sharpe Ratio (DSR)**: Corrects for selection bias

## Key Thresholds
| Metric | Minimum | Target |
|--------|---------|--------|
| Permutation p-value | < 0.05 | < 0.01 |
| Walk-Forward Efficiency | > 50% | > 60% |
| PBO | < 10% | < 5% |
| Min Trades | 50 | 200+ |

---

# 8. FEATURE ENGINEERING

## Tier 1: Pattern Geometry (MUST HAVE)
| Feature | Formula | Rationale |
|---------|---------|-----------|
| `head_extension_atr` | `(P3 - P1) / ATR` | Head extends beyond shoulder |
| `bos_depth_atr` | `(P2 - P4) / ATR` | Break of structure depth |
| `shoulder_symmetry` | `abs(P5 - P1) / ATR` | Right shoulder vs left |
| `amplitude_ratio` | `(P1-P2) / (P3-P4)` | Geometric proportion |
| `time_ratio` | `bars(P1→P3) / bars(P3→P5)` | Temporal symmetry |
| `fib_retracement_p5` | Fib level at P5 | Fibonacci confluence |

## Tier 2: Market Context (HIGH PRIORITY)
| Feature | Rationale |
|---------|-----------|
| `htf_trend_alignment` | Pattern with/against trend |
| `distance_to_sr_atr` | Key level confluence |
| `volatility_percentile` | High/low vol context |
| `regime_state` | Trending vs Ranging |
| `rsi_divergence` | Momentum confirmation |

## Tier 3: Volume Profile (HIGH PRIORITY)
| Feature | Rationale |
|---------|-----------|
| `volume_spike_p3` | Climax volume at head |
| `volume_spike_p4` | Volume on BOS |
| `volume_trend_p1_p5` | Participation trend |

## Detection Parameters to A/B Test
| Parameter | Values to Test |
|-----------|---------------|
| Swing Lookback | 3, 5, 7, 10 bars |
| Smoothing Window | 3, 5, 7 (Savitzky-Golay) |
| Min Head Extension | 0.3, 0.5, 1.0 ATR |
| BOS Requirement | 1 BOS, 2 BOS |
| Shoulder Tolerance | 0.3, 0.5, 1.0 ATR |

**Total combinations: ~210,000** → Need proper WFA and multiple testing correction

---

# 9. PROP FIRM RULES - BREAKOUT

## Account Details
- **Firm**: Breakout
- **Account Size**: $100K evaluation → $100K funded if pass

## Constraints (Typical - verify current rules)
| Rule | Impact |
|------|--------|
| Max Daily Drawdown (~5%) | Real-time DD tracker with alerts |
| Max Total Drawdown (~10%) | Trailing from peak equity |
| Profit Target (~8-10%) | Progress tracker needed |
| Consistency Rule | No single day > 30% of profits |
| Min Trading Days | 5-10 days |

## Kelly Criterion Adaptation
- Standard Kelly too aggressive for prop firm rules
- Use **Half-Kelly or less**
- Never risk > 1% per trade
- Daily stop-out at 50% of allowable drawdown limit

## Dashboard Requirements for Prop Firm
- Real-time daily DD tracker with alerts
- Cumulative DD from starting balance
- Progress tracker to profit target
- Consistency rule compliance display
- Kelly position sizer capped to rules

---

# 10. RECOMMENDED LIBRARIES

## Backtesting
- **VectorBT** - High-performance vectorized backtesting

## Technical Analysis
- **pandas-ta** - Comprehensive indicator coverage

## Machine Learning
- **XGBoost/LightGBM** - For tabular trading data
- **tsfresh** - Time series feature extraction

## Statistical Validation
- **mlfinlab** - Lopez de Prado's financial ML methods

## Signal Processing
- **PyWavelets** - Wavelet analysis for swing detection
- **scipy.signal** - Savitzky-Golay smoothing

## Dashboard
- **Streamlit** - Already using, keep it
- **lightweight-charts** - TradingView's open-source charting

## Data
- **ccxt** - Crypto exchange data fetching
- **parquet** - Efficient data storage

---

# 11. IMPLEMENTATION PHASES

## Phase 1: Foundation (Day 1)
- [ ] Audit and consolidate dashboard files
- [ ] Create new unified layout structure
- [ ] Implement command bar with live metrics
- [ ] Build KPI cards row component

## Phase 2: Core Panels (Day 2)
- [ ] Main chart area with TradingView integration
- [ ] Pattern scanner sidebar
- [ ] Monthly P&L heatmap
- [ ] Recent trades list

## Phase 3: Pipeline Integration (Day 3)
- [ ] Connect backtest runner properly
- [ ] VRD validation status display
- [ ] Pattern registry integration
- [ ] Settings persistence

## Phase 4: Polish (Day 4)
- [ ] Arctic Pro theme implementation
- [ ] Loading states and error handling
- [ ] Responsive design tweaks

## Phase 5: Prop Firm Module (Week 2)
- [ ] Breakout rules configuration
- [ ] Daily/trailing DD tracker
- [ ] Position sizer with Kelly
- [ ] Challenge pass probability calculator

## Phase 6: Experiment Lab (Week 2-3)
- [ ] Parameter grid manager
- [ ] Automated backtesting queue
- [ ] Results comparison dashboard
- [ ] Statistical significance flagging

## Phase 7: ML Pipeline (Week 3-4)
- [ ] Purged train/val/test splitter
- [ ] XGBoost training pipeline
- [ ] Feature selection automation

## Phase 8: Detection Logic Iteration (Week 4+)
- [ ] Wavelet-enhanced swing detection
- [ ] Multiple BOS variations
- [ ] Best parameter identification

---

# 12. TAB STRUCTURE

## Confirmed 7 Tabs
1. **📊 Dashboard** - Home/overview with key metrics
2. **🔬 Pattern Lab** - Locked chart visualization + pattern details
3. **⚡ Backtest** - Run backtests
4. **📈 Analytics** - Deep dive metrics
5. **🧪 Experiments** - A/B testing parameters
6. **🧠 ML Training** - Feature engineering + model training
7. **⚙️ Settings** - Strategy params + prop firm config

## Primary Assets
- BTC/USDT
- ETH/USDT
- Additional altcoins as needed for pattern minimums

## Primary Timeframe
- Start with 4H
- Later expand to 1H, Daily

---

# 13. DEEPSEEK RESEARCH INSIGHTS

## From DeepSeek Prompt #1 (Pattern Detection)
- Use wavelet-based swing detection for multi-scale confirmation
- Implement Bayesian P(QML | Data) instead of binary detection
- Geometric feature engineering (Fibonacci relationships, symmetry)

## From DeepSeek Prompt #2 (Position Sizing)
- Kelly adaptation formula with DD constraints
- Monte Carlo design for prop firm simulation
- Optimal stopping rules

## From DeepSeek Prompt #3 (ML Architecture)
- Multi-scale CNN architecture consideration
- Bidirectional LSTM for temporal patterns
- Multi-task output heads

**Note**: These are research insights. Implementation should use existing libraries where possible, not custom deep learning unless necessary.

---

# 14. CLAUDE CODE WORKFLOW

## Recommended Approach
1. **Single Claude Code session** for related changes
2. **Source of truth document** (this file) uploaded to project knowledge
3. **Sequential prompts** - each builds on the last
4. **Explicit protection** of locked components in every prompt

## Claude Code Prompt Template
```
Read the QML_PROJECT_MASTER_PLAN.md for full context.

CRITICAL: Do NOT modify the chart visualization system (TradingView 
Lightweight Charts with P1-P5 swing points, position boxes, trend lines).
This is LOCKED.

Task: [Specific task here]

Follow the Arctic Pro color scheme.
Keep code clean - no spaghetti.
Use existing libraries over custom code.
```

## Key Prompts to Generate
1. Dashboard foundation + command bar
2. Arctic Pro theme implementation
3. Tab structure setup
4. Prop firm module
5. Experiment lab
6. Pipeline integration
7. ML training setup

---

# APPENDIX: REFERENCE LINKS

## Past Chat Links
- [Dashboard redesign and feature planning](https://claude.ai/chat/0014cf05-60fa-469a-a4c8-a93c47306720)
- [v2 Trading system requirements](https://claude.ai/chat/2013105f-1d4f-4f7a-8c78-ab03d608e8a1)
- [Trading system requirements and planning](https://claude.ai/chat/5d78b2d0-8e75-4e78-84d0-8ddcb7e7c507)
- [Trading dashboard redesign with Claude agents](https://claude.ai/chat/54956f99-758f-492b-95a4-9db1810ad7f0)

## External References
- Delta Trend Trading YouTube: https://www.youtube.com/@deltatrendtrading
- TradingView Lightweight Charts: https://github.com/tradingview/lightweight-charts
- mlfinlab documentation: https://mlfinlab.readthedocs.io/

---

*Last Updated: January 20, 2026*
*Compiled from planning conversations in QML Trading System project*
