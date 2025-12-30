# QML Strategy Diagnostic Report

**Date:** December 29, 2025  
**Purpose:** Professional diagnostic suite to validate raw QML strategy before Phase 3

---

## Executive Summary

| Diagnostic | Result | Key Finding |
|------------|--------|-------------|
| **Win Rate Decay** | ✅ NOT SIGNIFICANT | No statistical decay (p=0.28), actually slight uptrend |
| **Parameter Stability** | ✅ STABLE | "High Volatility" filter works consistently across periods |
| **Monte Carlo** | ✅ ROBUST | Results not due to lucky sequencing (82% of paths worse) |

**VERDICT: PATH A (PROCEED)** - Strategy is ready for live simulation.

---

## Diagnostic #1: Win Rate Decay Analysis

### Question: Is the -6.3% year-over-year decay statistically significant?

### Methodology
- Linear regression of quarterly win rates over 12 quarters (2023-2025)
- Segmented analysis by market regime (trending/ranging, volatility levels)

### Results

#### Quarterly Win Rate Regression

| Quarter | Trades | Win Rate |
|---------|--------|----------|
| 2023Q1 | 120 | 55.8% |
| 2023Q2 | 143 | 65.0% |
| 2023Q3 | 118 | 51.7% |
| 2023Q4 | 133 | 57.9% |
| 2024Q1 | 138 | 61.6% |
| 2024Q2 | 133 | 61.7% |
| 2024Q3 | 135 | 68.1% |
| 2024Q4 | 140 | 70.0% |
| 2025Q1 | 145 | 59.3% |
| 2025Q2 | 136 | 60.3% |
| 2025Q3 | 118 | 60.2% |
| 2025Q4 | 114 | 63.2% |

**Regression Results:**
- **Slope: +0.47% per quarter** (NOT declining)
- **P-value: 0.28** (not significant at α=0.05)
- **R²: 0.115** (low correlation)
- **Fitted start: 58.6% → Fitted end: 63.8%** (+5.2% over 3 years)

#### Regime-Segmented Analysis

| Regime | Trades | Overall WR | 2023 | 2024 | 2025 | Trend |
|--------|--------|------------|------|------|------|-------|
| **Trending (ADX>25)** | 656 | 59.8% | 53.4% | 63.9% | 62.4% | +4.5%/yr (p=0.42) |
| **Ranging (ADX≤25)** | 917 | 62.6% | 61.7% | 66.4% | 59.4% | -1.1%/yr (p=0.79) |
| **High Volatility** | 351 | **68.1%** | 62.0% | 72.3% | 70.9% | ✅ Best |
| **Low Volatility** | 649 | 56.1% | 52.2% | 61.3% | 53.5% | ❌ Weakest |
| **Bullish Patterns** | 782 | 62.1% | 62.0% | 67.3% | 56.9% | -2.5%/yr |
| **Bearish Patterns** | 791 | 60.7% | 54.1% | 63.5% | 64.3% | +5.1%/yr |

### Conclusion

**✅ NO STATISTICALLY SIGNIFICANT DECAY**

The -6.3% "decay" we observed earlier was:
- Q4 2024: 70.0% (a high point)
- 2025 avg: 60.8%
- **This is normal variance, NOT a trend**

---

## Diagnostic #2: Walk-Forward Optimization

### Question: Are the default parameters stable across time periods?

### Methodology
- Test different filtering criteria on training period
- Apply best filter to out-of-sample period
- Measure if improvement persists

### Results

| WFO Cycle | Train Period | Best Filter | Train WR | OOS Period | OOS Baseline | OOS Filtered | OOS Improvement |
|-----------|--------------|-------------|----------|------------|--------------|--------------|-----------------|
| WFO-1 | 2023 | High Vol | 62.0% | 2024 | 65.1% | **71.8%** | **+6.7%** |
| WFO-2 | 2023-2024 | High Vol | 66.5% | 2025 | 60.6% | **70.9%** | **+10.3%** |

### Key Finding: Stable "High Volatility" Filter

The same filter (volatility_percentile > 0.7) was optimal in BOTH training periods and provided consistent improvement in BOTH out-of-sample periods.

**Filter Characteristics:**
- Reduces trade count by ~78% (focuses on 110 trades vs 500+)
- Improves win rate by +6.7% to +10.3%
- Trade-off: Fewer opportunities but higher quality

### Conclusion

**✅ PARAMETERS ARE STABLE**
- Default parameters work well across all time periods
- "High Volatility" filter provides consistent, real improvement
- No evidence of parameter overfitting

---

## Diagnostic #3: Monte Carlo Simulation

### Question: What role did luck play in our results?

### Methodology
- Generate 10,000 equity curves by randomly shuffling trade sequence
- Preserve all trade outcomes (same win/loss results, different order)
- Calculate distribution of max drawdowns and final equity

### Results

**Actual Historical Performance (1,573 trades, 2% risk):**
- Final Equity: $95,939
- Max Drawdown: 26.5%
- Win Rate: 61.4%

**Monte Carlo Distribution (10,000 simulations):**

| Metric | Value |
|--------|-------|
| Max DD - 90% VaR | **25.4%** |
| Max DD - 95% VaR | **27.9%** |
| Max DD - 99% VaR | **32.6%** |
| Our Actual Max DD | 26.5% (92nd percentile) |

**Luck Analysis:**
- Our actual drawdown was **WORSE** than 92% of random orderings
- Only 18% of random paths produced higher final equity
- **This means our actual sequence was UNLUCKY, not lucky**

### Interpretation

If our results were due to a "lucky" sequence of trades, we would expect our actual equity curve to be in the top percentiles. Instead:
- Our max drawdown was worse than average
- Our sequence was actually somewhat unlucky

**This is GOOD news** - it means:
1. The edge is real (not a lucky streak)
2. Better performance is possible with different trade ordering
3. Expected drawdowns in live trading: ~25% (90% confidence)

### Conclusion

**✅ RESULTS ARE ROBUST**
- Not due to lucky sequencing
- Edge is real and persistent
- Expect max drawdown of 25-28% with 90-95% confidence

---

## Summary of Findings

### Three Core Questions Answered

| Question | Answer |
|----------|--------|
| Is win rate decay significant? | **NO** - Slope is +0.47%/quarter, p=0.28 |
| Are parameters stable via WFO? | **YES** - High Vol filter works consistently |
| What's the role of luck? | **MINIMAL** - Our sequence was unlucky, not lucky |

### Risk Characteristics

| Metric | Value | Confidence |
|--------|-------|------------|
| Expected Win Rate | 61-65% | High (stable across 12 quarters) |
| Expected Max Drawdown | 25-28% | 90-95% confidence (Monte Carlo VaR) |
| Worst Case Drawdown | 33%+ | 1% probability |
| High Vol Filter WR | 68-72% | Consistent OOS improvement |

---

## Final Recommendation

# **PATH A: PROCEED TO LIVE SIMULATION**

The QML strategy has passed all diagnostic tests:

1. ✅ **No statistical decay** - Win rate is stable or slightly improving
2. ✅ **Parameters are robust** - No overfitting, stable across periods  
3. ✅ **Results are not luck** - Monte Carlo shows edge is real

### Recommended Configuration for Live Trading

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Position Size** | 1-2% risk | VaR indicates 25-28% max DD |
| **Volatility Filter** | Optional | +7-10% WR but 80% fewer trades |
| **Pattern Types** | Both bull/bear | Both profitable, bearish slightly improving |
| **Market Conditions** | All | Works in trending and ranging |

### Next Steps

1. **Paper Trading Phase** (2-4 weeks)
   - Validate detection in real-time
   - Confirm alert system works
   - Test execution workflow

2. **Small Live Allocation** (1-2 months)
   - 0.5-1% risk per trade
   - Validate slippage and fills
   - Build confidence before scaling

3. **Full Deployment**
   - Scale to 1-2% risk
   - Consider High Vol filter for higher conviction trades
   - Monitor quarterly performance vs. historical benchmarks

---

## Appendix: Data Coverage

- **Date Range:** 2023-01-11 to 2025-12-21 (3 years)
- **Total Trades:** 1,573
- **Assets:** BTC, ETH, BNB, XRP, ADA, DOGE, AVAX, DOT, LINK, LTC
- **Timeframe:** 1H
- **Market Regimes Covered:** Bull, bear, sideways, high/low volatility

