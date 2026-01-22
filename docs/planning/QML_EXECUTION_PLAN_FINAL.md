# QML TRADING SYSTEM - COMPLETE EXECUTION PLAN
## Final Implementation Guide with Claude Code Prompts

*Created: January 20, 2026*
*Version: 1.0 FINAL*

---

# PART 1: MASTER PHASE OVERVIEW

## Complete Phase List (11 Phases)

```
FOUNDATION (Week 1)
├── Phase 1: Codebase Audit & Cleanup
├── Phase 2: Dashboard Foundation + Arctic Pro Theme
├── Phase 3: Tab Structure Implementation
└── Phase 4: Pipeline Integration & Polish

CORE FEATURES (Week 2-3)
├── Phase 5: Prop Firm Module (Breakout Rules)
├── Phase 6: Backtest Tab + Results Visualization
└── Phase 7: Analytics Tab + Statistical Metrics

ADVANCED FEATURES (Week 3-4)
├── Phase 8: Experiment Lab (A/B Testing)
├── Phase 9: ML Training Tab + Feature Pipeline
└── Phase 10: Detection Logic Iteration

FUTURE (Week 5+)
└── Phase 11: Market Context Features (Funding, OI, Liquidations)
```

## Timeline Summary

| Week | Phases | Focus |
|------|--------|-------|
| 1 | 1-4 | Dashboard complete, working, pretty |
| 2 | 5-6 | Prop firm rules, backtesting works |
| 3 | 7-8 | Analytics, experiment framework |
| 4 | 9-10 | ML pipeline, detection iteration |
| 5+ | 11 | Market context (only if baseline proven) |

---

# PART 2: WORKFLOW STRUCTURE

## Recommended Approach

### Claude.ai Chat Organization
```
PROJECT: QML Trading System
│
├── Chat 1: "Phase 1-2 Foundation" (this chat or new)
│   └── Covers: Audit, cleanup, dashboard foundation, theme
│
├── Chat 2: "Phase 3-4 Tabs & Integration"
│   └── Covers: Tab structure, pipeline connection, polish
│
├── Chat 3: "Phase 5-6 Prop Firm & Backtest"
│   └── Covers: Breakout rules, backtest tab
│
├── Chat 4: "Phase 7-8 Analytics & Experiments"
│   └── Covers: Analytics tab, A/B testing framework
│
└── Chat 5: "Phase 9-10 ML & Detection"
    └── Covers: ML pipeline, detection logic iteration
```

**Why separate chats?**
- Each chat stays focused (less context pollution)
- Easier to reference back to specific phase
- Won't hit context limits mid-phase
- Create checkpoint docs at end of each chat

### Claude Code Session Strategy

```
For each Phase:
┌─────────────────────────────────────────────────┐
│  1. Start Claude Code session                   │
│  2. Give it the full phase prompt (below)       │
│  3. Let it work through all tasks               │
│  4. Review output, iterate if needed            │
│  5. Commit working code to git                  │
│  6. End session                                 │
└─────────────────────────────────────────────────┘
```

### Using Claude Code Agents?

**Recommendation: NO agents initially**

| Approach | When to Use |
|----------|-------------|
| Single Claude Code session | Phases 1-7 (most work) |
| Parallel agents | Only Phase 8+ if you need speed |

**Why no agents yet:**
- Your codebase needs understanding first
- Agents can conflict/overwrite each other
- Single session maintains consistency
- You're learning - easier to follow one thread

**When to consider agents:**
- Running multiple A/B test backtests simultaneously
- Training multiple ML models in parallel
- After you understand the codebase well

---

# PART 3: CLAUDE CODE PROMPTS

## Pre-Flight Checklist (Before ANY Phase)

Give Claude Code this context command first:

```
Before starting, read these files to understand the project:
1. QML_PROJECT_MASTER_PLAN.md (in project knowledge)
2. QML_PLAN_ADDENDUM_MARKET_CONTEXT.md (in project knowledge)

Key constraints to remember:
- DO NOT modify the chart visualization (TradingView Lightweight Charts, P1-P5 swing points, position boxes) - it is LOCKED
- Use Arctic Pro color scheme (bg: #0B1426, accent: #3B82F6, success: #10B981, danger: #EF4444)
- Follow 90% rule: clean code over feature completeness
- Use existing libraries over custom implementations
- Detection logic is intentionally incomplete - don't try to "fix" it yet

Confirm you understand these constraints before proceeding.
```

---

## PHASE 1: Codebase Audit & Cleanup

### Prompt 1.1 - Initial Audit

```
# Phase 1.1: Codebase Audit

## Objective
Analyze the current QML trading system codebase and create an audit report.

## Tasks
1. List all Python files in the project with their purpose
2. Identify the main entry points (dashboard, scripts, etc.)
3. Map the current file structure
4. Identify duplicate/redundant files
5. Check for broken imports or dependencies
6. List all external dependencies being used

## Output Required
Create a file `docs/CODEBASE_AUDIT.md` containing:
- File tree with annotations
- Dependency list
- Issues found
- Recommended files to archive/delete
- Recommended file structure changes

## Constraints
- DO NOT modify any code yet
- DO NOT touch anything in the chart visualization components
- This is READ-ONLY analysis
```

### Prompt 1.2 - Cleanup Execution

```
# Phase 1.2: Codebase Cleanup

## Objective
Execute the cleanup plan from the audit.

## Tasks
1. Create an `archive/` folder at project root
2. Move deprecated files to archive:
   - `src/dashboard/pattern_lab/` → `archive/pattern_lab/`
   - `qml/dashboard/app.py` → `archive/old_dashboard/`
   - Any other deprecated files identified in audit
3. Consolidate the dashboard to single entry point: `qml/dashboard/app_v2.py`
4. Fix any broken imports in remaining files
5. Update any hardcoded paths
6. Verify the dashboard still runs after cleanup

## Constraints
- DO NOT modify chart visualization components
- DO NOT change detection logic
- Keep all archived files (don't delete)
- Test that dashboard launches before and after

## Verification
Run the dashboard and confirm it loads without errors.
```

---

## PHASE 2: Dashboard Foundation + Arctic Pro Theme

### Prompt 2.1 - Arctic Pro Theme Implementation

```
# Phase 2.1: Arctic Pro Theme Implementation

## Objective
Replace the current theme with Arctic Pro color scheme across the entire dashboard.

## Arctic Pro Color Palette
```python
ARCTIC_PRO = {
    # Backgrounds
    'bg_primary': '#0B1426',      # Deep navy - main background
    'bg_card': '#162032',         # Slate blue - card backgrounds
    'bg_hover': '#1E2A3F',        # Hover states
    
    # Accent
    'accent': '#3B82F6',          # Electric blue
    'accent_hover': '#2563EB',
    
    # Semantic
    'success': '#10B981',         # Emerald green (profit)
    'danger': '#EF4444',          # Red (loss)
    'warning': '#F59E0B',         # Amber
    
    # Text
    'text_primary': '#F8FAFC',
    'text_secondary': '#94A3B8',
    'text_muted': '#64748B',
    
    # Chart (DO NOT CHANGE CHART VISUALIZATION LOGIC)
    'bullish': '#10B981',         # Match success
    'bearish': '#EF4444',         # Match danger
    
    # Borders
    'border': '#1E293B',
    'border_hover': '#334155',
}
```

## Tasks
1. Create `qml/dashboard/theme.py` with all color constants
2. Create CSS/styling that uses these colors
3. Update `app_v2.py` to import and use the theme
4. Style all existing components with new theme
5. Ensure consistent typography:
   - Sans-serif for labels: 'Inter', -apple-system, sans-serif
   - Monospace for numbers: 'JetBrains Mono', monospace

## Constraints
- DO NOT modify chart visualization logic or colors for annotations
- Chart candlestick colors can be updated to match theme
- Keep all existing functionality working

## Verification
Dashboard should look professional with consistent dark blue theme.
```

### Prompt 2.2 - Command Bar Implementation

```
# Phase 2.2: Command Bar Implementation

## Objective
Create a persistent command bar at the top of the dashboard that shows key metrics.

## Design
```
┌─────────────────────────────────────────────────────────────────────────┐
│  QML COMMAND CENTER                                                     │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌────────┐│
│  │ WIN %   │ │ SHARPE  │ │ PROFIT  │ │ MAX DD  │ │ EXPECTANCY│ │ KELLY ││
│  │  ---%   │ │  --.-   │ │ FACTOR  │ │  ---%   │ │   $---   │ │  --%  ││
│  │         │ │         │ │  --.-x  │ │         │ │          │ │       ││
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘ └────────┘│
│                                                    [BTC: $--,---] [●]  │
└─────────────────────────────────────────────────────────────────────────┘
```

## Tasks
1. Create `qml/dashboard/components/command_bar.py`
2. Implement 6 KPI cards in a row:
   - Win Rate (%)
   - Sharpe Ratio
   - Profit Factor
   - Max Drawdown (%)
   - Expectancy ($)
   - Kelly % (position size recommendation)
3. Add live BTC price display (right side)
4. Add system status indicator (green dot = connected)
5. Make command bar sticky (always visible at top)
6. Cards should show placeholder values for now (---)
7. Cards should have subtle hover effect

## Styling
- Use Arctic Pro theme colors
- Cards: bg_card background, subtle border
- Numbers: monospace font, text_primary color
- Labels: small, text_muted color
- Success metrics (win rate, profit factor): success color if positive
- Risk metrics (max DD): danger color

## Constraints
- Command bar must not scroll with page content
- Must be responsive (cards can stack on mobile)
- DO NOT connect to real data yet (placeholders only)
```

---

## PHASE 3: Tab Structure Implementation

### Prompt 3.1 - Tab Navigation Setup

```
# Phase 3.1: Tab Navigation Structure

## Objective
Implement the 7-tab navigation structure below the command bar.

## Tab Structure
1. 📊 Dashboard - Home/overview
2. 🔬 Pattern Lab - Chart visualization (LOCKED component lives here)
3. ⚡ Backtest - Run and view backtests
4. 📈 Analytics - Deep statistical analysis
5. 🧪 Experiments - A/B parameter testing
6. 🧠 ML Training - Feature engineering & model training
7. ⚙️ Settings - Configuration

## Tasks
1. Create `qml/dashboard/components/tab_navigation.py`
2. Implement tab bar with icons and labels
3. Create placeholder pages for each tab:
   - `qml/dashboard/pages/dashboard_page.py`
   - `qml/dashboard/pages/pattern_lab_page.py`
   - `qml/dashboard/pages/backtest_page.py`
   - `qml/dashboard/pages/analytics_page.py`
   - `qml/dashboard/pages/experiments_page.py`
   - `qml/dashboard/pages/ml_training_page.py`
   - `qml/dashboard/pages/settings_page.py`
4. Integrate tabs into main app_v2.py
5. Tab switching should be smooth, no page reload

## Tab Bar Styling
- Background: bg_card
- Active tab: accent color underline, text_primary
- Inactive tab: text_muted
- Hover: text_secondary
- Icons: Use emoji or simple unicode symbols

## Placeholder Content
Each page should show:
- Page title
- Brief description of what will go there
- "Coming soon" or placeholder content

## Constraints
- Pattern Lab page MUST include the existing chart visualization
- Move the LOCKED chart component into Pattern Lab page
- DO NOT modify the chart visualization code itself
```

### Prompt 3.2 - Dashboard Home Page

```
# Phase 3.2: Dashboard Home Page Content

## Objective
Build out the main Dashboard (home) tab with overview content.

## Layout
```
┌─────────────────────────────────────────────────────────────────────┐
│  [Command Bar - already built]                                      │
├─────────────────────────────────────────────────────────────────────┤
│  [Tab Navigation - Dashboard selected]                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────┐  ┌─────────────────────────────┐  │
│  │     EQUITY CURVE            │  │    MONTHLY P&L HEATMAP      │  │
│  │     [Placeholder chart]     │  │    [12-month grid]          │  │
│  │                             │  │                             │  │
│  └─────────────────────────────┘  └─────────────────────────────┘  │
│                                                                     │
│  ┌─────────────────────────────┐  ┌─────────────────────────────┐  │
│  │     RECENT TRADES           │  │    PATTERN SCANNER          │  │
│  │     [Table: last 10]        │  │    [Active patterns list]   │  │
│  │                             │  │                             │  │
│  └─────────────────────────────┘  └─────────────────────────────┘  │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │     SYSTEM STATUS                                            │   │
│  │     Data Feed: ● | Last Update: --- | Patterns Today: 0     │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Tasks
1. Create 2x2 grid of main panels
2. Equity Curve panel:
   - Placeholder line chart (can use Plotly)
   - Title: "Equity Curve"
   - Shows cumulative P&L over time
3. Monthly P&L Heatmap:
   - 12-month calendar grid
   - Green for profit months, red for loss
   - Placeholder data for now
4. Recent Trades table:
   - Columns: Date, Symbol, Direction, Entry, Exit, P&L, R-Multiple
   - Last 10 trades
   - Placeholder data
5. Pattern Scanner panel:
   - List of currently detected patterns
   - Symbol, Timeframe, Direction, Confidence
   - Placeholder data
6. System Status bar at bottom:
   - Connection status
   - Last data update time
   - Pattern count today

## Styling
- All panels: bg_card background, rounded corners, subtle border
- Tables: alternating row colors
- Use success/danger colors for P&L values
```

---

## PHASE 4: Pipeline Integration & Polish

### Prompt 4.1 - Data Connection Layer

```
# Phase 4.1: Data Connection Layer

## Objective
Create the connection layer between dashboard and existing backend systems.

## Tasks
1. Create `qml/dashboard/data/data_manager.py`:
   - Central class for all data operations
   - Methods to fetch:
     - Historical OHLCV data
     - Detected patterns
     - Backtest results
     - System metrics
   
2. Create `qml/dashboard/data/mock_data.py`:
   - Generate realistic mock data for testing
   - Mock patterns, trades, metrics
   - Use this until real pipeline connected

3. Create data models in `qml/dashboard/data/models.py`:
   ```python
   @dataclass
   class Pattern:
       id: str
       symbol: str
       timeframe: str
       direction: str  # 'BULLISH' or 'BEARISH'
       confidence: float
       timestamp: datetime
       p1_price: float
       p2_price: float
       p3_price: float
       p4_price: float
       p5_price: float
       entry_price: float
       stop_loss: float
       take_profit: float
   
   @dataclass
   class Trade:
       id: str
       pattern_id: str
       symbol: str
       direction: str
       entry_price: float
       exit_price: float
       entry_time: datetime
       exit_time: datetime
       pnl: float
       r_multiple: float
       status: str  # 'OPEN', 'CLOSED', 'CANCELLED'
   
   @dataclass
   class BacktestResult:
       id: str
       name: str
       params: dict
       start_date: datetime
       end_date: datetime
       total_trades: int
       win_rate: float
       sharpe_ratio: float
       profit_factor: float
       max_drawdown: float
       total_return: float
   ```

4. Wire mock data to dashboard components:
   - Command bar reads from data_manager
   - Dashboard page panels read from data_manager
   - Pattern Lab shows patterns from data_manager

## Constraints
- Keep mock data realistic but clearly fake
- Structure for easy swap to real data later
- DO NOT modify existing detection/backtest code yet
```

### Prompt 4.2 - Polish & Loading States

```
# Phase 4.2: Polish & Loading States

## Objective
Add polish, loading states, and error handling throughout the dashboard.

## Tasks
1. Loading States:
   - Add spinner/skeleton for all data-dependent components
   - Show "Loading..." text while fetching
   - Graceful handling when data unavailable
   
2. Error States:
   - Friendly error messages (not stack traces)
   - "No data available" placeholders
   - Retry buttons where appropriate

3. Empty States:
   - "No patterns detected" for empty pattern list
   - "No trades yet" for empty trade table
   - "Run a backtest to see results" for empty backtest

4. Visual Polish:
   - Consistent spacing (use 8px grid)
   - Smooth transitions on tab switches
   - Hover effects on interactive elements
   - Subtle shadows on cards

5. Responsive Design:
   - Dashboard should work on different screen sizes
   - Cards stack vertically on narrow screens
   - Tables scroll horizontally if needed

6. Performance:
   - Lazy load tab content (don't render hidden tabs)
   - Cache data where appropriate
   - Debounce any real-time updates

## Verification
- Dashboard loads quickly
- No errors in console
- All states (loading, error, empty, populated) look good
- Works on different window sizes
```

---

## PHASE 5: Prop Firm Module (Breakout Rules)

### Prompt 5.1 - Settings Page: Prop Firm Config

```
# Phase 5.1: Prop Firm Configuration

## Objective
Build the Settings page with prop firm rule configuration.

## Breakout Prop Firm Rules (Default Values)
```python
BREAKOUT_RULES = {
    'account_size': 100000,           # $100K account
    'max_daily_drawdown_pct': 4.0,    # 4% max daily DD
    'max_total_drawdown_pct': 8.0,    # 8% max total DD (trailing)
    'profit_target_pct': 8.0,         # 8% to pass challenge
    'min_trading_days': 5,            # Minimum days to trade
    'max_position_size_pct': 2.0,     # Max 2% risk per trade
    'consistency_rule': True,          # No single day > 30% of profits
    'consistency_max_pct': 30.0,      # Max % of profits in single day
}
```

## Settings Page Layout
```
┌─────────────────────────────────────────────────────────────────────┐
│  SETTINGS                                                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  PROP FIRM CONFIGURATION                                     │   │
│  │                                                              │   │
│  │  Prop Firm:        [Breakout    ▼]                          │   │
│  │  Account Size:     [$100,000    ]                           │   │
│  │  Max Daily DD:     [4.0        ]%                           │   │
│  │  Max Total DD:     [8.0        ]%                           │   │
│  │  Profit Target:    [8.0        ]%                           │   │
│  │  Min Trading Days: [5          ]                            │   │
│  │  Max Position Size:[2.0        ]%                           │   │
│  │  Consistency Rule: [✓]                                      │   │
│  │                                                              │   │
│  │  [Save Configuration]                                        │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  TRADING PARAMETERS                                          │   │
│  │                                                              │   │
│  │  Primary Symbols:  [BTC/USDT, ETH/USDT]                     │   │
│  │  Primary Timeframe:[4H         ▼]                           │   │
│  │  Kelly Fraction:   [0.5        ] (Half-Kelly)               │   │
│  │  Max Concurrent:   [3          ] positions                  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  DETECTION PARAMETERS                                        │   │
│  │                                                              │   │
│  │  Swing Lookback:   [5          ] bars                       │   │
│  │  Min Move ATR:     [0.5        ]                            │   │
│  │  BOS Requirement:  [1          ] break(s)                   │   │
│  │  Shoulder Tolerance:[0.5       ] ATR                        │   │
│  │                                                              │   │
│  │  [Save Parameters]                                           │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Tasks
1. Create Settings page with three config sections
2. Implement form inputs for all parameters
3. Save to local config file (JSON or YAML)
4. Load saved config on startup
5. Validate inputs (e.g., no negative percentages)
6. Show confirmation on save

## Files to Create
- `qml/dashboard/pages/settings_page.py`
- `qml/dashboard/config/prop_firm_rules.py`
- `qml/dashboard/config/user_config.json`

## Constraints
- Settings must persist between sessions
- Validate all numeric inputs
- Show current vs default values
```

### Prompt 5.2 - Prop Firm Tracking Dashboard

```
# Phase 5.2: Prop Firm Tracking in Command Bar

## Objective
Add real-time prop firm compliance tracking to the dashboard.

## Enhanced Command Bar
```
┌─────────────────────────────────────────────────────────────────────────┐
│  QML COMMAND CENTER                                    BREAKOUT $100K   │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌────────┐│
│  │ WIN %   │ │ SHARPE  │ │ DAILY DD│ │TOTAL DD │ │ PROFIT  │ │P(PASS) ││
│  │  62.3%  │ │  1.84   │ │ -1.2%   │ │ -3.4%   │ │ +5.2%   │ │  74%   ││
│  │         │ │         │ │ of -4%  │ │ of -8%  │ │ of +8%  │ │        ││
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘ └────────┘│
│                          ▓▓▓▓▓▓▓░░░  ▓▓▓▓░░░░░░  ▓▓▓▓▓▓░░░░            │
│                          30% used    42% used    65% to target         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Tasks
1. Replace some KPI cards with prop firm metrics:
   - Daily Drawdown (with progress bar to limit)
   - Total Drawdown (with progress bar to limit)
   - Profit Progress (with progress bar to target)
   - P(Pass) - probability of passing challenge

2. Add visual progress bars under each:
   - Color changes as approaching limits
   - Green → Yellow → Red for drawdown
   - Empty → Filling → Green for profit

3. Calculate P(Pass) using simple Monte Carlo:
   - Based on current win rate, avg R, trades remaining
   - Update in real-time as metrics change

4. Add prop firm badge (top right):
   - Shows firm name and account size
   - Status indicator (On Track / Warning / Violated)

## Color Coding
- Daily DD: green < 50% used, yellow 50-75%, red > 75%
- Total DD: same thresholds
- Profit: red < 25%, yellow 25-75%, green > 75%
- P(Pass): red < 30%, yellow 30-60%, green > 60%

## Files to Modify
- `qml/dashboard/components/command_bar.py`
- Add: `qml/dashboard/utils/prop_firm_calculator.py`
```

---

## PHASE 6: Backtest Tab + Results Visualization

### Prompt 6.1 - Backtest Configuration Panel

```
# Phase 6.1: Backtest Configuration Panel

## Objective
Build the Backtest tab with configuration and execution UI.

## Layout
```
┌─────────────────────────────────────────────────────────────────────┐
│  BACKTEST                                                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌───────────────────────────┐  ┌───────────────────────────────┐  │
│  │  BACKTEST CONFIGURATION   │  │  RESULTS                       │  │
│  │                           │  │                                │  │
│  │  Symbol:    [BTC/USDT ▼]  │  │  [Equity Curve Chart]         │  │
│  │  Timeframe: [4H       ▼]  │  │                                │  │
│  │  Start:     [2024-01-01]  │  │                                │  │
│  │  End:       [2024-12-31]  │  │                                │  │
│  │                           │  ├────────────────────────────────┤  │
│  │  ─── Detection ───        │  │  METRICS                       │  │
│  │  Swing Lookback: [5  ]    │  │  Total Trades:    ---          │  │
│  │  Min Move ATR:   [0.5]    │  │  Win Rate:        ---%         │  │
│  │  BOS Required:   [1  ]    │  │  Profit Factor:   ---          │  │
│  │                           │  │  Sharpe Ratio:    ---          │  │
│  │  ─── Risk ───             │  │  Max Drawdown:    ---%         │  │
│  │  Risk Per Trade: [1.0]%   │  │  Total Return:    ---%         │  │
│  │  Stop Loss ATR:  [1.5]    │  │  Avg R-Multiple:  ---          │  │
│  │  Take Profit R:  [2.0]    │  │                                │  │
│  │                           │  ├────────────────────────────────┤  │
│  │  [▶ Run Backtest]         │  │  TRADE LIST                    │  │
│  │  [Save Parameters]        │  │  [Sortable table of trades]    │  │
│  └───────────────────────────┘  └────────────────────────────────┘  │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  SAVED BACKTESTS                                             │   │
│  │  [Table: Name | Date | Trades | Win% | Sharpe | PF | Return] │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Tasks
1. Create left panel with all configuration options
2. Create right panel for results display:
   - Equity curve chart (Plotly line chart)
   - Key metrics in grid
   - Trade list table
3. Create bottom panel for saved backtest history
4. Implement "Run Backtest" button:
   - Shows loading spinner while running
   - Populates results when complete
   - For now, use mock backtest (2-3 second delay)
5. Implement "Save Parameters" to store presets

## Files to Create
- `qml/dashboard/pages/backtest_page.py`
- `qml/dashboard/components/backtest_config.py`
- `qml/dashboard/components/backtest_results.py`
- `qml/dashboard/utils/mock_backtest.py`
```

### Prompt 6.2 - Connect Real Backtest Engine

```
# Phase 6.2: Connect Real Backtest Engine

## Objective
Connect the backtest UI to the actual VRD 2.0 backtest engine.

## Tasks
1. Identify the existing backtest runner in the codebase
2. Create adapter in `qml/dashboard/data/backtest_adapter.py`:
   - Translate UI parameters to engine format
   - Run the actual backtest
   - Parse results into dashboard format

3. Wire the "Run Backtest" button to real engine:
   - Pass configuration from UI
   - Show progress (if possible)
   - Handle errors gracefully

4. Store backtest results:
   - Save to JSON file for history
   - Include all parameters and results
   - Timestamp each run

5. Load historical backtests on page load

## Constraints
- DO NOT modify the VRD 2.0 engine itself
- Create adapter layer only
- Handle cases where engine fails
- Show meaningful error messages
```

---

## PHASE 7: Analytics Tab + Statistical Metrics

### Prompt 7.1 - Analytics Page

```
# Phase 7.1: Analytics Deep Dive Page

## Objective
Build comprehensive analytics tab with statistical validation metrics.

## Layout
```
┌─────────────────────────────────────────────────────────────────────┐
│  ANALYTICS                                                          │
├─────────────────────────────────────────────────────────────────────┤
│  Select Backtest: [Latest - 2024-12-15 ▼]  [Compare Mode ☐]        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────────────────────┐ ┌──────────────────────────┐ │
│  │  RETURNS DISTRIBUTION            │ │  DRAWDOWN ANALYSIS       │ │
│  │  [Histogram of R-multiples]      │ │  [Drawdown chart]        │ │
│  │                                  │ │  Underwater curve        │ │
│  │  Mean: 0.42R  Median: 0.31R     │ │  Max: -12.3%             │ │
│  │  Skew: 0.34   Kurtosis: 2.1     │ │  Avg: -3.2%              │ │
│  └──────────────────────────────────┘ └──────────────────────────┘ │
│                                                                     │
│  ┌──────────────────────────────────┐ ┌──────────────────────────┐ │
│  │  WIN/LOSS STREAKS               │ │  TIME ANALYSIS           │ │
│  │  Max Win Streak:  7             │ │  [Heatmap: Hour x Day]   │ │
│  │  Max Loss Streak: 4             │ │                          │ │
│  │  Avg Win Streak:  2.3           │ │  Best Day:   Tuesday     │ │
│  │  Avg Loss Streak: 1.8           │ │  Best Hour:  14:00 UTC   │ │
│  └──────────────────────────────────┘ └──────────────────────────┘ │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  STATISTICAL VALIDATION                                      │   │
│  │                                                              │   │
│  │  Permutation Test p-value:     0.023 ✓  (< 0.05 required)   │   │
│  │  Walk-Forward Efficiency:      62.3% ✓  (> 50% required)    │   │
│  │  Probability of Overfitting:   4.2%  ✓  (< 10% required)    │   │
│  │  Deflated Sharpe Ratio:        1.42  ✓  (> 1.0 required)    │   │
│  │                                                              │   │
│  │  Overall Assessment: STATISTICALLY VALID ✓                   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Tasks
1. Create analytics page with 4 visualization panels
2. Returns Distribution:
   - Histogram of R-multiple outcomes
   - Show mean, median, skew, kurtosis
3. Drawdown Analysis:
   - Underwater equity curve
   - Max drawdown, average drawdown
   - Duration statistics
4. Win/Loss Streaks:
   - Calculate streak statistics
   - Visualize streak distribution
5. Time Analysis:
   - Heatmap of performance by hour/day
   - Identify best/worst trading times
6. Statistical Validation panel:
   - Show key validation metrics
   - Green checkmark if passing, red X if failing
   - Clear thresholds displayed

## Files to Create
- `qml/dashboard/pages/analytics_page.py`
- `qml/dashboard/components/analytics/returns_chart.py`
- `qml/dashboard/components/analytics/drawdown_chart.py`
- `qml/dashboard/components/analytics/time_heatmap.py`
- `qml/dashboard/utils/statistical_tests.py`

## Note on Validation Metrics
For now, calculate what's possible from trade data:
- Basic statistics always available
- Monte Carlo (streak analysis) from trades
- Permutation test: can be approximated
- WFE, PBO, DSR: may need full VRD integration
Mark "needs full validation run" if not available
```

---

## PHASE 8: Experiment Lab (A/B Testing)

### Prompt 8.1 - Experiment Framework

```
# Phase 8.1: Experiment Lab Framework

## Objective
Build A/B testing framework for parameter optimization.

## Layout
```
┌─────────────────────────────────────────────────────────────────────┐
│  EXPERIMENT LAB                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │  CREATE EXPERIMENT                                             │ │
│  │                                                                │ │
│  │  Name: [Swing Lookback Test____________]                      │ │
│  │                                                                │ │
│  │  Parameter to Test: [Swing Lookback ▼]                        │ │
│  │  Values to Test:    [3, 5, 7, 10______]                       │ │
│  │                                                                │ │
│  │  Fixed Parameters:                                            │ │
│  │  Symbol: BTC/USDT    Timeframe: 4H                           │ │
│  │  Date Range: 2024-01-01 to 2024-12-31                        │ │
│  │                                                                │ │
│  │  [▶ Run Experiment]                                           │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │  EXPERIMENT RESULTS: Swing Lookback Test                      │ │
│  │                                                                │ │
│  │  ┌──────────┬───────┬───────┬────────┬─────────┬───────────┐ │ │
│  │  │ Value    │Trades │Win %  │Sharpe  │Profit F │Significant│ │ │
│  │  ├──────────┼───────┼───────┼────────┼─────────┼───────────┤ │ │
│  │  │ 3        │ 145   │ 58.2% │ 1.21   │ 1.82    │ ⚠️ Low N  │ │ │
│  │  │ 5 ★      │ 89    │ 64.1% │ 1.87   │ 2.34    │ ✓         │ │ │
│  │  │ 7        │ 62    │ 61.3% │ 1.65   │ 2.12    │ ⚠️ Low N  │ │ │
│  │  │ 10       │ 34    │ 55.9% │ 1.12   │ 1.54    │ ✗ p>0.1   │ │ │
│  │  └──────────┴───────┴───────┴────────┴─────────┴───────────┘ │ │
│  │                                                                │ │
│  │  ★ = Best performing (statistically valid)                    │ │
│  │                                                                │ │
│  │  [Bar chart comparing metrics across values]                  │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │  EXPERIMENT HISTORY                                            │ │
│  │  [Table of past experiments with results]                     │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Tasks
1. Create experiment configuration UI:
   - Select parameter to vary
   - Enter values to test (comma-separated)
   - Fixed parameters inherited from settings
   
2. Experiment runner:
   - Run backtest for each parameter value
   - Collect results in standardized format
   - Calculate statistical significance
   
3. Results comparison:
   - Table showing all variants
   - Highlight best performer
   - Flag statistical significance issues:
     * "Low N" if < 50 trades
     * "Not significant" if p > 0.05
     * "Overfitting risk" if best by small margin
   
4. Visualization:
   - Bar chart comparing key metrics
   - Confidence intervals on bars
   
5. History:
   - Save all experiments
   - Recall and compare past results

## Files to Create
- `qml/dashboard/pages/experiments_page.py`
- `qml/dashboard/utils/experiment_runner.py`
- `qml/dashboard/utils/significance_tests.py`

## Parameters Available for Testing
- Swing Lookback (3, 5, 7, 10)
- Min Move ATR (0.3, 0.5, 0.7, 1.0)
- BOS Requirement (1, 2)
- Shoulder Tolerance (0.3, 0.5, 1.0)
- Stop Loss ATR (1.0, 1.5, 2.0)
- Take Profit R (1.5, 2.0, 2.5, 3.0)
```

---

## PHASE 9: ML Training Tab

### Prompt 9.1 - ML Training Interface

```
# Phase 9.1: ML Training Tab

## Objective
Build interface for feature engineering and model training.

## Layout
```
┌─────────────────────────────────────────────────────────────────────┐
│  ML TRAINING                                                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌───────────────────────────┐  ┌───────────────────────────────┐  │
│  │  FEATURE ENGINEERING      │  │  MODEL STATUS                  │  │
│  │                           │  │                                │  │
│  │  Available Features: 28   │  │  Current Model: XGBoost v3    │  │
│  │  Selected Features:  12   │  │  Trained: 2024-12-10          │  │
│  │                           │  │  Features: 12                  │  │
│  │  ☑ head_extension_atr    │  │  CV Score: 0.68 AUC           │  │
│  │  ☑ bos_depth_atr         │  │                                │  │
│  │  ☑ shoulder_symmetry     │  │  [View Training Report]        │  │
│  │  ☑ time_ratio            │  │                                │  │
│  │  ☐ volume_spike_p3       │  │  ┌────────────────────────┐   │  │
│  │  ☐ rsi_divergence        │  │  │ Feature Importance    │   │  │
│  │  ...                      │  │  │ [Horizontal bar chart]│   │  │
│  │                           │  │  └────────────────────────┘   │  │
│  │  [Select All] [Clear]    │  │                                │  │
│  └───────────────────────────┘  └────────────────────────────────┘  │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  TRAINING CONFIGURATION                                      │   │
│  │                                                              │   │
│  │  Model Type:     [XGBoost ▼]    Validation: [Walk-Forward ▼]│   │
│  │  Train Period:   [2022-01-01] to [2024-06-30]               │   │
│  │  Test Period:    [2024-07-01] to [2024-12-31]               │   │
│  │                                                              │   │
│  │  Walk-Forward Folds: [5]    Purge Gap: [10] bars            │   │
│  │                                                              │   │
│  │  [▶ Train New Model]  [Export Model]  [Load Model]          │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  TRAINING LOG                                                │   │
│  │  [Live output from training process]                        │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Tasks
1. Feature Selection Panel:
   - List all available features with checkboxes
   - Group by category (Geometry, Context, Volume, etc.)
   - Show feature descriptions on hover
   - Quick select: All, None, Recommended
   
2. Model Status Panel:
   - Show current trained model info
   - Feature importance visualization
   - Link to detailed training report
   
3. Training Configuration:
   - Model type selector (XGBoost, LightGBM)
   - Date range for train/test split
   - Walk-forward fold configuration
   - Purge gap setting
   
4. Training Execution:
   - Run training with progress updates
   - Stream log output to UI
   - Save model when complete
   - Generate training report

5. Model Management:
   - Export trained model
   - Load previous model
   - Compare models

## Files to Create
- `qml/dashboard/pages/ml_training_page.py`
- `qml/dashboard/components/feature_selector.py`
- `qml/dashboard/components/training_log.py`
- `qml/dashboard/utils/ml_trainer.py` (adapter to existing ML pipeline)

## Feature List (from Master Plan)
Tier 1 - Pattern Geometry:
- head_extension_atr
- bos_depth_atr
- shoulder_symmetry
- amplitude_ratio
- time_ratio
- fib_retracement_p5

Tier 2 - Market Context:
- htf_trend_alignment
- distance_to_sr_atr
- volatility_percentile
- regime_state
- rsi_divergence

Tier 3 - Volume:
- volume_spike_p3
- volume_spike_p4
- volume_trend_p1_p5
```

---

## PHASE 10: Detection Logic Iteration

### Prompt 10.1 - Pattern Lab Enhancement

```
# Phase 10.1: Pattern Lab Enhancement for Detection Tuning

## Objective
Enhance Pattern Lab tab to support visual detection logic iteration.

## Layout Enhancement
```
┌─────────────────────────────────────────────────────────────────────┐
│  PATTERN LAB                                                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Symbol: [BTC/USDT ▼]  TF: [4H ▼]  [◀ Prev Pattern] [Next ▶]      │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                                                              │   │
│  │              [LOCKED CHART VISUALIZATION]                    │   │
│  │                                                              │   │
│  │     Shows: P1-P5 points, trend lines, position boxes        │   │
│  │                                                              │   │
│  │                                                              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌───────────────────────────┐  ┌───────────────────────────────┐  │
│  │  PATTERN DETAILS          │  │  DETECTION PARAMETERS         │  │
│  │                           │  │                                │  │
│  │  ID: QML-BTC-4H-001      │  │  Swing Lookback:  [5 ▼]       │  │
│  │  Direction: BULLISH      │  │  Min Move ATR:    [0.5___]    │  │
│  │  Confidence: 78%         │  │  BOS Required:    [1 ▼]       │  │
│  │                           │  │  Shoulder Tol:    [0.5___]    │  │
│  │  P1: $94,230 (2024-12-01)│  │                                │  │
│  │  P2: $96,450             │  │  [Apply & Re-detect]           │  │
│  │  P3: $92,100 (Head)      │  │                                │  │
│  │  P4: $95,800             │  │  Patterns Found: 3             │  │
│  │  P5: $93,900 (Entry)     │  │  Current: 1 of 3              │  │
│  │                           │  │                                │  │
│  │  Entry: $93,900          │  │  ──────────────────────────    │  │
│  │  Stop:  $91,800 (-2.2%)  │  │  MANUAL LABEL                  │  │
│  │  TP1:   $97,100 (+3.4%)  │  │  [✓ Valid] [✗ Invalid]        │  │
│  │  TP2:   $99,200 (+5.6%)  │  │  [💰 Winner] [❌ Loser]        │  │
│  └───────────────────────────┘  │  [💾 Save Label]              │  │
│                                  └───────────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Tasks
1. Add navigation controls:
   - Previous/Next pattern buttons
   - Pattern counter (X of Y)
   - Filter by symbol, timeframe
   
2. Pattern Details panel:
   - Show all P1-P5 coordinates
   - Calculated entry, stop, targets
   - Confidence score
   
3. Detection Parameters (live tuning):
   - Editable parameters
   - "Apply & Re-detect" button
   - Shows updated pattern count
   - Allows visual iteration
   
4. Manual Labeling:
   - Mark patterns as Valid/Invalid
   - Mark outcomes as Winner/Loser
   - Save labels to file
   - Use for training data

## Constraints
- DO NOT modify the chart visualization component
- Only add panels AROUND the existing chart
- Chart component receives pattern data as props
```

---

## PHASE 11: Market Context (Future)

### Prompt 11.1 - Placeholder (Do Not Implement Yet)

```
# Phase 11: Market Context Features

## STATUS: PLANNED - DO NOT IMPLEMENT UNTIL PHASES 1-10 COMPLETE

## Prerequisites
- [ ] Baseline system validated
- [ ] 200+ backtest trades completed
- [ ] Phase 9 manual error analysis shows context value
- [ ] Phase 10 rule-based overlay proves value

## Features to Add
- Funding Rate integration
- Open Interest tracking
- Liquidation level estimates

## See: QML_PLAN_ADDENDUM_MARKET_CONTEXT.md for full specification

## DO NOT START THIS PHASE until prerequisites met.
```

---

# PART 4: CHECKPOINT SYSTEM

## After Each Phase, Create Checkpoint

At the end of each phase, ask Claude Code:

```
Create a checkpoint document summarizing what was completed:

1. Files created/modified (with brief description)
2. Current working features
3. Known issues or TODOs
4. Next steps

Save to: docs/checkpoints/PHASE_X_CHECKPOINT.md
```

## After Major Milestones, Update Master Plan

After completing Phases 1-4, 5-7, 8-10:

```
Review the current state of the project and update the Master Plan document 
with:
1. Completed items (mark as ✅)
2. Any deviations from original plan
3. New discoveries or requirements
4. Updated timeline if needed
```

---

# PART 5: QUICK REFERENCE

## Critical Constraints (Include in Every Prompt)
```
CRITICAL CONSTRAINTS:
1. DO NOT modify chart visualization (P1-P5 points, position boxes, trend lines) - LOCKED
2. Use Arctic Pro colors (bg: #0B1426, accent: #3B82F6, success: #10B981, danger: #EF4444)
3. Follow 90% rule: prefer simple/clean code over complex/complete
4. Use existing libraries over custom implementations
5. Detection logic is intentionally incomplete - don't "fix" it
```

## File Structure Goal
```
qml/
├── dashboard/
│   ├── app_v2.py                 # Main entry point
│   ├── theme.py                  # Arctic Pro colors
│   ├── components/
│   │   ├── command_bar.py
│   │   ├── tab_navigation.py
│   │   ├── backtest_config.py
│   │   ├── backtest_results.py
│   │   ├── feature_selector.py
│   │   └── analytics/
│   │       ├── returns_chart.py
│   │       ├── drawdown_chart.py
│   │       └── time_heatmap.py
│   ├── pages/
│   │   ├── dashboard_page.py
│   │   ├── pattern_lab_page.py
│   │   ├── backtest_page.py
│   │   ├── analytics_page.py
│   │   ├── experiments_page.py
│   │   ├── ml_training_page.py
│   │   └── settings_page.py
│   ├── data/
│   │   ├── data_manager.py
│   │   ├── mock_data.py
│   │   ├── models.py
│   │   └── backtest_adapter.py
│   ├── utils/
│   │   ├── prop_firm_calculator.py
│   │   ├── statistical_tests.py
│   │   ├── experiment_runner.py
│   │   └── ml_trainer.py
│   └── config/
│       ├── prop_firm_rules.py
│       └── user_config.json
├── docs/
│   ├── checkpoints/
│   └── CODEBASE_AUDIT.md
└── archive/
    └── [deprecated files]
```

---

# APPENDIX: Session Commands

## Starting a Claude Code Session
```bash
cd /path/to/qml-project
claude
```

## Useful Claude Code Commands
```
# Check current directory
pwd

# List files
ls -la

# Read a file
cat filename.py

# Run the dashboard
streamlit run qml/dashboard/app_v2.py

# Install dependencies
pip install package_name

# Git commit
git add . && git commit -m "Phase X complete"
```

---

*Document Complete. Save to project knowledge and reference when executing phases.*
