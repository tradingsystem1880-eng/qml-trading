# Feature Priority Queue

Based on DeepSeek analysis: With PF 4.49, adding features is more likely to hurt than help.
Only test features with strong theoretical backing and clear edge potential.

## Current Queue

### Priority 1: Funding Rate Filter
- **Status**: IN PROGRESS
- **Hypothesis**: Filtering against extreme funding rates will reduce adverse selection
- **Rationale**: When funding is extremely positive, longs are crowded - fade longs. Vice versa for shorts.
- **Expected Impact**: Small WR improvement, potential trade reduction 10-20%
- **Risk**: May reject too many trades, reducing sample size

### Priority 2: Open Interest Divergence (HOLD)
- **Status**: NOT STARTED
- **Hypothesis**: Price/OI divergence indicates potential reversals
- **Rationale**: Rising OI with falling price = shorts entering, potential squeeze target
- **Dependency**: Requires OI data collection infrastructure
- **Risk**: Data quality issues, exchange-specific

### Priority 3: Liquidation Cascade Detection (HOLD)
- **Status**: NOT STARTED
- **Hypothesis**: Large liquidation events create short-term opportunities
- **Rationale**: Cascading liquidations cause temporary dislocations
- **Dependency**: Requires real-time liquidation feed
- **Risk**: Latency-sensitive, may not work for 4H timeframe

---

## Rejected Ideas (DO NOT TEST)

### ML Meta-Labeling (Phase 8.0)
- **Tested**: 2026-01-26
- **Result**: FAILED (AUC 0.53 = random)
- **Lesson**: Price-based features don't predict trade outcomes

### Complex TP Schemes
- **Reason**: Simpler fixed TP (4.6R) already optimal
- **Risk**: Curve-fitting, parameter explosion

---

## Testing Protocol

1. **One feature at a time** - No confounding
2. **Full validation pipeline** - FeatureValidator with all 4 checks
3. **Log everything** - ResearchJournal
4. **Paper trade BASE system** - No enhancements until proven

## Success Criteria (from plan)

| Check | Threshold |
|-------|-----------|
| Sample size | >= 100 trades |
| Trade reduction | < 30% |
| Walk-forward | All folds PF > 1.0 |
| Permutation test | p < 0.05 |
| PF degradation | None (filtered >= baseline) |
