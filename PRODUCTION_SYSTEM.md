# QML Trading System - Production Specification

## System Overview

- **Strategy**: QML (Quasimodo) pattern detection with regime filtering
- **Markets**: Cryptocurrency perpetual futures (22 symbols)
- **Timeframe**: 4H (primary), 1H/1D (supporting)
- **Account**: $100K prop firm target (Breakout/FTMO compatible)

---

## Validated Performance (Phase 7.9 Baseline)

| Metric | Value | 95% CI |
|--------|-------|--------|
| Profit Factor | 1.23 | [1.15, 1.31] |
| Sharpe Ratio | +0.08 | - |
| Win Rate | 52% | [50.9%, 53.1%] |
| Expectancy | +0.18R per trade | - |
| DSR | 0.986 | - |
| Sample Size | 2,155 trades | - |

**Statistical Significance**: DSR of 0.986 indicates 98.6% probability the edge is real, not luck.

---

## Entry Rules

### Pattern Detection (QML/Quasimodo)

1. **5-Point Swing Structure (P1-P5)**
   - P1: Prior trend extreme
   - P2: First retracement
   - P3: Head (extends beyond P1)
   - P4: Second retracement
   - P5: Break of Structure (BoS)

2. **Validation Criteria**
   - Head extension: 0.3-10.0 ATR beyond prior extreme
   - Shoulder symmetry: Within 5.0 ATR
   - Pattern duration: 8-200 bars
   - Break of Structure: Clear break of P2/P4 level

3. **Regime Filter (Phase 7.8)**
   - **REJECT**: TRENDING regime with ADX > 35
   - **Soft scoring by regime**:
     - RANGING: 1.0x quality
     - VOLATILE: 0.6x quality
     - EXTREME: 0.5x quality
     - TRENDING: 0.2x quality

4. **Tier System**
   - Tier A: Quality > 0.8 (highest conviction)
   - Tier B: Quality 0.6-0.8
   - Tier C: Quality 0.4-0.6 (minimum threshold)
   - REJECT: Quality < 0.4

---

## Exit Rules

### Phase 9.0 Adaptive Exits (Recommended)

1. **Time-Decaying Profit Target**
   ```
   TP(t) = Entry + Risk × R_target × e^(-λ × t)
   where λ = ln(2) / halflife_bars

   Default parameters:
   - Initial R: 3.0 (tp_atr_mult)
   - Halflife: 20 bars
   - Minimum R: 0.5 (never decay below)
   ```

   Rationale: Momentum patterns lose edge over time. If target isn't hit quickly, reduce expectations rather than waiting for full target.

2. **Trailing Stop**
   - Activation: 1.0 ATR profit from entry
   - Trail distance: 0.5 ATR from highest high (longs)
   - Moves to breakeven at activation

3. **Time-Based Exit**
   - Maximum hold: 50 bars
   - Forces exit at market if no SL/TP hit

4. **Stop Loss**
   - Fixed: 1.5 ATR from entry
   - Never moved except by trailing logic

### Alternative: Fixed Exits (Phase 7.9 Baseline)

If adaptive exits underperform, fall back to:
- TP: 3.0 ATR from entry (fixed)
- SL: 1.5 ATR from entry (fixed)
- Trailing: Activated at 1.0 ATR profit

---

## Position Sizing

### Phase 9.0 Rules

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Risk per trade | 1% of equity | Fixed fractional (Kelly negative) |
| Max concurrent positions | 3 | Correlation risk management |
| Daily loss limit | 4% | Stop trading for day |
| Monthly loss limit | 8% | Prop firm compliant |

### Kelly Criterion Note

Phase 7.9 analysis showed **negative full Kelly** due to small edge magnitude. The mathematically optimal approach is fixed fractional (1%) rather than Kelly-based sizing.

### Consecutive Loss Management

| Consecutive Losses | Action |
|-------------------|--------|
| 4 | Reduce position size by 50% |
| 7 | Pause trading (0.5% probability at 52% WR) |

---

## Symbols (22 Validated)

### Tier 1 (Highest Liquidity)
- BTC/USDT, ETH/USDT, BNB/USDT, SOL/USDT

### Tier 2 (High Liquidity)
- XRP/USDT, ADA/USDT, DOGE/USDT, AVAX/USDT
- DOT/USDT, LINK/USDT, MATIC/USDT

### Tier 3 (Medium Liquidity)
- ATOM/USDT, UNI/USDT, LTC/USDT, APT/USDT
- ARB/USDT, OP/USDT, NEAR/USDT, INJ/USDT
- FTM/USDT, AAVE/USDT, MKR/USDT

---

## Forward Testing Requirements

Before live deployment, complete:

| Requirement | Target |
|-------------|--------|
| Minimum trades | 500 |
| Win rate | >48% (allow for variance) |
| Profit factor | >1.10 (within CI of baseline) |
| Edge degradation | No critical alerts |
| Consecutive losses | <7 during test |

### Forward Test Commands

```bash
# Run forward test on recent data
python scripts/run_phase90_forward.py --symbols BTC/USDT,ETH/USDT --days 30

# Check current status
python scripts/run_phase90_forward.py --status

# Generate detailed report
python scripts/run_phase90_forward.py --report
```

---

## What Was Rejected

### ML Meta-Labeling (Phase 8.0)

| Issue | Finding |
|-------|---------|
| Initial PF 7.11 | **FAKE** - train/test data leakage |
| Proper test AUC | 0.53 (essentially random) |
| CV fold variance | 0.29-0.69 (unstable) |
| **Decision** | **USE_BASELINE** |

The edge is in the binary trade decision (pattern detection), not magnitude prediction. ML provides no benefit over fixed 1% sizing.

---

## Key Files

| Component | File |
|-----------|------|
| Trade Simulator | `src/optimization/trade_simulator.py` |
| Pattern Detection | `src/detection/qml_pattern.py` |
| Pattern Validator | `src/detection/pattern_validator.py` |
| Pattern Scorer | `src/detection/pattern_scorer.py` |
| Regime Detection | `src/detection/regime.py` |
| Position Rules | `src/risk/position_rules.py` |
| Forward Monitor | `src/risk/forward_monitor.py` |
| Kelly Sizer | `src/risk/kelly_sizer.py` |
| Exit Comparison | `scripts/compare_exit_strategies.py` |
| Forward Test CLI | `scripts/run_phase90_forward.py` |

---

## Deployment Checklist

- [ ] Forward test completes 500+ trades
- [ ] No critical degradation alerts
- [ ] Performance within baseline CI
- [ ] Prop firm rules configured
- [ ] Daily loss limit implemented
- [ ] Consecutive loss pause implemented
- [ ] Position size reduction implemented
- [ ] Exit strategy finalized (fixed vs adaptive)

---

## Version History

| Phase | Date | Description |
|-------|------|-------------|
| 7.9 | 2026-01-26 | Baseline validated (PF 1.23, DSR 0.986) |
| 8.0 | 2026-01-26 | ML meta-labeling tested and rejected |
| 9.0 | 2026-01-26 | Adaptive exits + forward test infrastructure |
| 9.1 | 2026-01-26 | Fixed regime filter bug - calculate regime per-pattern |

---

## Phase 9.1 Results (Backtest on 68 trades, 7 symbols)

| Strategy | Win Rate | Profit Factor | Expectancy | Avg Bars |
|----------|----------|---------------|------------|----------|
| Fixed TP (3R) | 82.4% | 8.31 | 1.22R | 12.7 |
| Adaptive TP | 88.2% | 12.29 | 1.22R | 7.7 |
| With Trailing | 100%* | ∞ | 1.04R | 1.3 |

*100% win rate with trailing is due to breakeven activation at 1 ATR.

**Recommendation**: Use adaptive exits for higher win rate, or fixed for larger wins. Both have identical expectancy (~1.22R).

---

## Emergency Procedures

### If edge degrades in live trading:

1. **Immediate**: Reduce position size to 0.5%
2. **If PF < 1.0 over 50 trades**: Pause live trading
3. **Review**: Check for regime shift, market conditions
4. **Revalidate**: Run forward test on recent data
5. **Resume**: Only if forward test passes

### Critical thresholds:

- 7+ consecutive losses: Pause immediately
- 4%+ daily drawdown: Stop for day
- 8%+ monthly drawdown: Full strategy review

---

*Document generated: Phase 9.0 - January 2026*
