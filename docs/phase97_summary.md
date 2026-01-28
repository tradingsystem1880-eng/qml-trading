# Phase 9.7: Funding Rate Filter Validation - Summary

**Date:** 2026-01-28
**Status:** FAILED
**Verdict:** Do not deploy funding filter. Paper trade BASE system.

---

## Executive Summary

Phase 9.7 tested whether extreme funding rates could predict poor trade outcomes for QML patterns. The filter **passed statistical tests** but **failed economic validation**.

**Key Finding:** The filter removes more winning trades (60%) than losing trades (40%), resulting in a net loss of 128.5R. Despite a 14.5% PF improvement, this is survivorship bias - not predictive power.

---

## Hypothesis

> Extreme funding rates (±0.010%) predict poor trade outcomes due to overcrowded positioning.

**Rationale:** When funding is extremely positive, too many traders are long; when extremely negative, too many are short. These crowded positions may lead to squeezes that hurt pattern-based entries.

---

## Methodology

Used DeepSeek-style multi-layer validation:

1. **Permutation Test (10,000 iterations)** - Statistical significance
2. **Walk-Forward Validation (5 folds)** - Temporal consistency
3. **Economic Significance** - Must improve net R, not just PF
4. **Mandatory Criteria** - Filter must remove more losers than winners

---

## Full Results

### Statistical Tests

| Test | Value | Threshold | Result |
|------|-------|-----------|--------|
| Permutation p-value | 0.0111 | < 0.05 | PASS |
| Walk-forward folds improved | 4/5 | ≥ 3/5 | PASS |
| Weighted score | 3/6 | ≥ 4/6 | FAIL |

### Performance Comparison

| Metric | Baseline | Filtered | Change |
|--------|----------|----------|--------|
| Profit Factor | 8.27 | 9.47 | +14.5% |
| Trades | 358 | 290 | -19.0% |
| Trade Reduction | - | 68 | - |

### Trade Attribution (Critical Finding)

| Category | Count | Percentage |
|----------|-------|------------|
| Winners Removed | 41 | 60.3% |
| Losers Removed | 27 | 39.7% |

### R-Multiple Impact

| Category | R Impact |
|----------|----------|
| Winners Lost | -155.8R |
| Losers Avoided | +27.3R |
| **Net Impact** | **-128.5R** |

---

## Why PF Improved Despite FAIL

This is the key insight that prevents a false positive:

**Survivorship Bias in PF:** Removing ANY trade with below-average R/R will mechanically improve PF, regardless of whether the filter has predictive power.

### Example:

Consider 100 trades with avg R/R = 2.0:
- Remove 20 trades with R/R = 1.5 (below average)
- PF mathematically increases
- BUT if those 20 included 15 winners, you lose more R than you save

### What Happened:

The funding filter removed trades during extreme funding periods. These periods happened to include more winning trades than losing trades - possibly because extreme funding often occurs during strong trends, which QML patterns profit from.

---

## Edge Cases Triggered

1. **REMOVED_MORE_WINNERS_THAN_LOSERS** - 41 winners vs 27 losers
2. **NET_R_NEGATIVE** - Lost 128.5R compared to baseline

These edge cases were specifically designed to catch filters that "look good" on PF but hurt actual returns.

---

## Key Learnings

1. **PF improvement alone is insufficient** - Must verify net R impact
2. **Statistical significance can be misleading** - p < 0.05 doesn't mean economically beneficial
3. **Always check trade attribution** - Filter must remove more losers than winners
4. **Consider inverse hypothesis** - If filter removes winners, maybe the signal helps rather than hurts

---

## Decision Rationale

| Factor | Assessment |
|--------|------------|
| Statistical significance | PASS - but misleading |
| Walk-forward consistency | PASS - but PF-based |
| Economic benefit | FAIL - net negative R |
| Selective power | FAIL - removes more winners |
| Risk reduction | FAIL - no DD improvement |

**Verdict:** FAIL - Do not deploy.

---

## Next Steps

1. **Paper trade BASE system** - No funding filter
2. **Priority 2 backlog** - Test inverse hypothesis:
   > Extreme funding rates indicate momentum that HELPS trend-following QML patterns

3. **Future research** - Consider directional filtering:
   - Positive funding + LONG pattern → maybe BETTER, not worse
   - Negative funding + SHORT pattern → maybe BETTER, not worse

---

## Files Reference

| File | Purpose |
|------|---------|
| `research/journal.json` | Complete experiment record |
| `src/data/funding_rates.py` | Funding rate data fetcher |
| `src/research/feature_validator.py` | DeepSeek-style validation framework |
| `scripts/validate_funding_filter.py` | Validation CLI |

---

## Validation Results Location

Full validation output saved to: `results/phase97_funding_validation/`

---

*Generated: 2026-01-28*
