# VRD 2.0: Versioned Research Database & Institutional-Grade Validation Framework

## Executive Summary

VRD 2.0 is a forensic-grade strategy validation framework designed to answer one critical question: **"Is this trading edge real, or just luck?"**

The system provides statistical proof of edge before deploying capital by combining:
- **Reproducibility** (versioned experiments with git hashes)
- **Leakage prevention** (purged walk-forward validation)
- **Statistical rigor** (permutation tests, Monte Carlo, bootstrap)
- **Explainability** (170+ features, regime analysis)
- **Deployment readiness** (stress testing, position sizing)

---

## The Problem VRD 2.0 Solves

Traditional backtesting suffers from critical flaws:

1. **Data Leakage**: Train/test contamination inflates results
2. **Overfitting**: Parameters tuned to historical noise, not signal
3. **Luck vs Skill**: No statistical proof that results aren't random
4. **Regime Blindness**: Performance aggregated across different market conditions
5. **Irreproducibility**: Results can't be recreated or compared

VRD 2.0 addresses all five through a 10-module architecture.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     VRD 2.0 FRAMEWORK                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────┐ │
│  │  Module 1: VRD  │    │  Module 2: WF   │    │  Module 3:  │ │
│  │  Experiment     │───▶│  Purged Walk-   │───▶│  Statistical│ │
│  │  Tracking       │    │  Forward        │    │  Suite      │ │
│  └─────────────────┘    └─────────────────┘    └─────────────┘ │
│                                                       │         │
│  ┌─────────────────┐    ┌─────────────────┐          ▼         │
│  │  Module 6:      │    │  Module 5:      │    ┌─────────────┐ │
│  │  Feature Engine │───▶│  Regime         │───▶│  Module 4:  │ │
│  │  (170+ features)│    │  Detection      │    │  Sensitivity│ │
│  └─────────────────┘    └─────────────────┘    └─────────────┘ │
│                                                       │         │
│  ┌─────────────────┐    ┌─────────────────┐          ▼         │
│  │  Module 7:      │    │  Module 8:      │    ┌─────────────┐ │
│  │  Data Integrity │───▶│  Advanced       │───▶│  Module 9:  │ │
│  │  Audit          │    │  Diagnostics    │    │  Reporting  │ │
│  └─────────────────┘    └─────────────────┘    └─────────────┘ │
│                                                       │         │
│                                                       ▼         │
│                                              ┌─────────────────┐│
│                                              │  Module 10:     ││
│                                              │  Deployment     ││
│                                              │  Gatekeeper     ││
│                                              └─────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

---

## Module 1: VRD 2.0 - Versioned Research Database

**Purpose**: Experiment tracking and reproducibility

**Implementation**: `src/validation/tracker.py`, `src/validation/database.py`

### How It Works

Every backtest run is automatically stamped with:
- **Git commit hash** - Code version
- **Timestamp** - When it ran
- **Parameter hash** - MD5 of all strategy parameters
- **Random seeds** - For reproducibility
- **Data range** - Start/end dates used

### Directory Structure
```
experiments/
├── 20260104_111627_QML_Strategy_dc363a04/
│   ├── config.json          # Full configuration
│   ├── trades.csv           # Trade log
│   ├── metrics.json         # Performance metrics
│   └── charts/              # Visualizations
```

### SQLite Schema
```sql
CREATE TABLE experiments (
    id TEXT PRIMARY KEY,
    strategy_name TEXT,
    git_hash TEXT,
    timestamp DATETIME,
    params_hash TEXT,
    oos_sharpe REAL,
    total_trades INTEGER,
    verdict TEXT
);
```

### Why This Design

1. **Immutability**: Results can never be accidentally modified
2. **Queryability**: Compare experiments by metrics, parameters, regimes
3. **Reproducibility**: Re-run any experiment with exact same conditions

---

## Module 2: Purged Walk-Forward Engine

**Purpose**: Eliminate information leakage between train/test sets

**Implementation**: `src/validation/walk_forward.py`

### How It Works

```
Timeline: ──────────────────────────────────────────────────────────▶

Traditional Split (FLAWED):
├── Train ──────────────────┤├── Test ──────────────┤
                            ↑
                     Leakage at boundary!

Purged Walk-Forward (VRD 2.0):
├── Train ──────────┤ P ├── Test ────────┤ <- Fold 1
        ├── Train ──────────┤ P ├── Test ────────┤ <- Fold 2
                ├── Train ──────────┤ P ├── Test ────────┤ <- Fold 3
                        
P = Purge gap (5-10 bars) - prevents lookahead
```

### Key Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `n_folds` | 8-10 | Statistical power |
| `purge_gap` | 5-10 bars | Prevent signal bleed |
| `embargo` | 2-3 bars | Extra safety margin |

### Why This Design

1. **Temporal Integrity**: Never train on data from the future
2. **Multiple Samples**: 8-10 folds provide statistical confidence
3. **Parameter Stability**: Track if parameters change across folds (overfit signal)

---

## Module 3: Statistical Robustness Suite

**Purpose**: Distinguish skill from luck

**Implementation**: `src/validation/permutation.py`, `src/validation/monte_carlo.py`, `src/validation/bootstrap.py`

### 3A. Permutation Testing

**Problem**: A Sharpe ratio of 2.0 looks great, but could it happen by luck?

**Solution**: Shuffle the trade sequence 10,000+ times and compare

```python
# Pseudo-code
actual_sharpe = calculate_sharpe(trades)
null_distribution = []
for i in range(10000):
    shuffled_trades = random_shuffle(trades)
    null_distribution.append(calculate_sharpe(shuffled_trades))

p_value = sum(null_distribution >= actual_sharpe) / 10000
```

**Interpretation**:
- `p < 0.05` → Statistically significant (unlikely due to luck)
- `p > 0.10` → Cannot distinguish from random

### 3B. Monte Carlo Simulation

**Problem**: What's the range of possible outcomes?

**Solution**: Resample trades 50,000 times to build distribution

**Outputs**:
- **VaR 95%**: 95% of paths have max drawdown below this
- **VaR 99%**: 99% of paths have max drawdown below this
- **Expected Shortfall**: Average loss in worst 5% of cases
- **Kill Switch Probability**: Chance of hitting 20%+ drawdown
- **Time to Recovery**: How long to recover from drawdowns

### 3C. Block Bootstrap

**Problem**: Standard confidence intervals assume independence

**Solution**: Block bootstrap preserves autocorrelation structure

```python
# Block bootstrap (preserves temporal structure)
block_size = 5  # Group trades in blocks
for _ in range(5000):
    # Sample blocks with replacement
    resampled = sample_blocks(trades, block_size)
    metrics.append(calculate_sharpe(resampled))

ci_lower, ci_upper = np.percentile(metrics, [2.5, 97.5])
```

### Why This Design

1. **Permutation Test**: Answers "Is this skill or luck?"
2. **Monte Carlo**: Answers "What could happen?"
3. **Bootstrap**: Answers "How confident are we in these estimates?"

---

## Module 4: Parameter & Sensitivity Analysis

**Purpose**: Identify robust vs overfit parameter regions

**Implementation**: `src/analysis/sensitivity.py`

### How It Works

1. **Range Testing**: Test each parameter at ±40% of base value
2. **Grid Scanning**: Test pairs of parameters (2D grids)
3. **Surface Plots**: Visualize Sharpe ratio as 3D surface

### Plateau vs Cliff Detection

```
Good (Plateau):                Bad (Cliff/Peak):
                               
   ████████████                    ████
  ████████████████                ██████
 ██████████████████              ████████
████████████████████            ██████████
                    
Robust - small changes          Overfit - small changes
don't hurt performance          destroy performance
```

### Why This Design

Plateaus indicate robust parameter regions where small errors don't matter. Cliffs/peaks indicate overfitting where the model found a specific historical pattern that won't repeat.

---

## Module 5: Regime-Explicit Analysis

**Purpose**: Stratify performance by market conditions

**Implementation**: `src/analysis/regimes.py`

### Regime Classification

| Regime | Volatility | Trend (ADX) | Description |
|--------|------------|-------------|-------------|
| Bull Quiet | Low | High | Steady uptrend |
| Bull Volatile | High | High | Strong rally with swings |
| Bear Quiet | Low | High | Slow decline |
| Bear Volatile | High | High | Crash/capitulation |
| Ranging | Low | Low | Sideways consolidation |

### Outputs

1. **Performance by Regime**: Sharpe/WR broken down by market condition
2. **Transition Matrix**: Probability of regime changes
3. **Regime Filter**: When to trade vs avoid

### Why This Design

A strategy that works only in bull markets isn't robust. Regime analysis reveals when the edge exists and allows for adaptive position sizing.

---

## Module 6: Feature-Performance Correlation Engine

**Purpose**: Explain WHY the edge exists

**Implementation**: `src/features/library.py`

### Feature Categories (170+ Features)

| Category | Examples | Count |
|----------|----------|-------|
| Technical Indicators | RSI, MACD, Bollinger, ATR, ADX | 30+ |
| Volume/Microstructure | Relative volume, VWAP, bid-ask spread | 20+ |
| Volatility | ATR percentile, realized vol, vol-of-vol | 15+ |
| Momentum | Returns, Z-scores, mean reversion | 20+ |
| Temporal | Hour, day-of-week, month (cyclical) | 15+ |
| Candle Morphology | Hammer, engulfing, doji, gap | 20+ |
| Cross-Asset | BTC/SPY correlation, funding rates | 10+ |

### Edge Attribution

For each trade, we compute 170+ features at entry time and correlate with outcome:

```python
# Top 10 features explaining 80% of variance
1. ATR percentile (0.32 correlation)
2. RSI divergence (0.28 correlation)
3. Volume spike ratio (0.24 correlation)
...
```

### Why This Design

Knowing WHY a strategy works helps predict if it will continue. If the edge comes from a fundamental pattern, it's likely persistent. If it's noise, it will decay.

---

## Module 7: Data Integrity & Leakage Prevention

**Purpose**: Forensic auditing to ensure no contamination

**Implementation**: `src/validation/validator.py`

### Checks Performed

1. **Look-Ahead Detection**: Verify features only use past data
2. **Label Validation**: Confirm targets don't use future info
3. **Temporal Shuffle Test**: Shuffle timestamps and check if edge disappears

### Why This Design

Even subtle leakage (like using close price to predict close) can inflate results 10x. Forensic auditing catches these before deployment.

---

## Module 8: Advanced Diagnostics

**Purpose**: Deep-dive into edge dynamics

**Implementation**: `src/analysis/diagnostics.py`

### Analyses

| Analysis | What It Reveals |
|----------|-----------------|
| Volatility Expansion | Does the strategy predict volatility? |
| Rolling Correlation | How correlated is performance with BTC/SPY? |
| Drawdown Decomposition | Are losses from streaks or single trades? |
| Temporal Performance | Best/worst hours, days, months |

### Why This Design

These diagnostics reveal hidden dependencies and help design filters for when NOT to trade.

---

## Module 9: Comprehensive Reporting System

**Purpose**: Generate actionable "Deployment Dossier"

**Implementation**: `src/reporting/dossier.py`, `src/reporting/visuals.py`

### HTML Dossier Contents

1. **Executive Summary**: One-page verdict with confidence score (0-100)
2. **Key Metrics**: Sharpe, drawdown, trades, p-value
3. **Statistical Card**: Permutation test, CIs, percentile rank
4. **Visual Analysis**: Embedded charts (equity, Monte Carlo, drawdown)
5. **Risk Dashboard**: VaR, expected shortfall, ruin probability
6. **Recommendations**: Position sizing, regime filters

### Visualization Types

| Chart | Purpose |
|-------|---------|
| Equity Curve | Regime-colored performance over time |
| Monte Carlo Cones | 95%/99% confidence bands on 50,000 paths |
| Permutation Histogram | Actual Sharpe vs null distribution |
| Drawdown Chart | Time in drawdown and recovery |
| 3D Sensitivity Surface | Parameter robustness |

### Why This Design

Professional HTML output with embedded charts makes results shareable and auditable. No external dependencies - single self-contained file.

---

## Module 10: Deployment Readiness Checks

**Purpose**: Validate live trading viability

**Implementation**: `src/deployment/gatekeeper.py`

### 8 Deployment Gates

| Check | Threshold | Rationale |
|-------|-----------|-----------|
| OOS Sharpe | > 1.0 | Minimum risk-adjusted return |
| VaR 95% | < 25% | Max acceptable drawdown |
| Kill Switch Prob | < 15% | Ruin risk |
| Statistical Significance | p < 0.05 | Not luck |
| Regime Robustness | > 75% | Works in most conditions |
| Parameter Stability | > 0.6 | Not overfit |
| Trade Count | > 50 | Sufficient sample |
| IS/OOS Ratio | < 2.0 | Not massively overfit |

### Verdict Logic

```python
if all_gates_pass:
    verdict = "DEPLOY"
elif pass_rate >= 0.6:
    verdict = "CAUTION"
else:
    verdict = "REJECT"
```

### Why This Design

Multiple independent gates prevent any single flaw from causing deployment. Requires unanimous pass or explicit acknowledgment of risks.

---

## Pipeline Integration

### ValidationOrchestrator

**Implementation**: `src/pipeline/orchestrator.py`

Chains all modules together:

```python
from src.pipeline.orchestrator import ValidationOrchestrator

# Run full validation
result = orchestrator.run(
    price_data=df,
    strategy=MyStrategy(),
    config=OrchestratorConfig(
        n_folds=8,
        n_monte_carlo=50000,
        n_permutations=10000
    )
)

# Generate HTML dossier
dossier.generate(result, output_dir="results/")
```

---

## Entry Points

| Script | Purpose |
|--------|---------|
| `strategy_autopsy.py` | Full autopsy with markdown report + 4 charts |
| `run_full_validation_on_saved_trades.py` | HTML dossier from saved trade CSV |
| `ValidationOrchestrator` | Programmatic pipeline access |

---

## Key Design Principles

1. **Skepticism First**: Assume the edge is fake until proven otherwise
2. **Multiple Evidence**: Require statistical, regime, and stability confirmation
3. **Transparency**: Full audit trail with versioning
4. **Actionability**: Clear DEPLOY/CAUTION/REJECT verdict
5. **Reproducibility**: Any result can be exactly recreated

---

## File Structure

```
src/
├── validation/
│   ├── tracker.py          # Module 1: Experiment tracking
│   ├── database.py         # Module 1: SQLite storage
│   ├── walk_forward.py     # Module 2: Purged walk-forward
│   ├── permutation.py      # Module 3A: Permutation test
│   ├── monte_carlo.py      # Module 3B: Monte Carlo
│   ├── bootstrap.py        # Module 3C: Block bootstrap
│   └── validator.py        # Module 7: Data integrity
├── analysis/
│   ├── sensitivity.py      # Module 4: Parameter sweep
│   ├── regimes.py          # Module 5: Regime detection
│   └── diagnostics.py      # Module 8: Advanced diagnostics
├── features/
│   └── library.py          # Module 6: 170+ features
├── reporting/
│   ├── dossier.py          # Module 9: HTML generator
│   └── visuals.py          # Module 9: Chart generation
├── deployment/
│   └── gatekeeper.py       # Module 10: Deployment gates
└── pipeline/
    └── orchestrator.py     # Full pipeline orchestration
```

---

## Summary

VRD 2.0 answers the five critical questions for any trading strategy:

| Question | Answer Via |
|----------|------------|
| **Is the edge REAL?** | Permutation test p-value |
| **WHY does it work?** | Feature attribution, regime analysis |
| **Will it CONTINUE?** | Parameter stability, regime persistence |
| **How can it FAIL?** | Monte Carlo worst-case, regime vulnerabilities |
| **How should we TRADE it?** | Position sizing, regime filters, risk limits |

The system is designed to prevent capital deployment on lucky backtests and ensure only statistically validated strategies go live.
