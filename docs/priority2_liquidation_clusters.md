# Priority 2 Research: Liquidation Clusters

**Status:** NOT STARTED
**Timebox:** 20 hours maximum
**Priority:** Secondary (30% time allocation during paper trading)

---

## Hypothesis

> QML patterns forming near liquidation clusters have higher success rates due to:
> 1. Liquidation cascades providing momentum fuel for pattern breakouts
> 2. Market maker activity around cluster zones creating cleaner price action
> 3. Reduced counter-trend risk when large liquidation pools are in direction of trade

---

## Data Sources

### Primary: Velo.xyz (Free Tier)
- Liquidation heatmaps
- Historical liquidation levels
- API access (check rate limits)
- https://velo.xyz

### Alternative: Coinglass
- Liquidation data aggregator
- May require paid tier for historical data
- https://coinglass.com

### Backup: Hyblock Capital
- Another liquidation data provider
- https://hyblock.co

---

## Implementation Plan

### Phase 1: Data Collection (4-6 hours)
- [ ] Evaluate Velo.xyz API capabilities
- [ ] Build liquidation data fetcher
- [ ] Store historical liquidation levels
- [ ] Align with existing price data timestamps

### Phase 2: Feature Engineering (4-6 hours)
- [ ] Define "near liquidation cluster" metric
- [ ] Calculate distance to nearest cluster
- [ ] Determine cluster size/density scores
- [ ] Create direction alignment score (cluster in trade direction)

### Phase 3: Validation (6-8 hours)
- [ ] Apply DeepSeek-style validation (same as Phase 9.7)
- [ ] Permutation test for statistical significance
- [ ] Walk-forward consistency check
- [ ] Economic impact (net R, not just PF)
- [ ] Trade attribution (must remove more losers than winners)

### Phase 4: Decision (2 hours)
- [ ] Document results in research journal
- [ ] PASS/FAIL decision
- [ ] If PASS: integrate into production
- [ ] If FAIL: document learnings, move to next priority

---

## Success Criteria

| Criterion | Threshold |
|-----------|-----------|
| Permutation p-value | < 0.05 |
| Walk-forward folds | ≥ 3/5 improved |
| Net R impact | > 0 (positive) |
| Trade attribution | Removes more losers than winners |
| Economic benefit | True |

**All mandatory criteria must pass.** Learned from Phase 9.7: PF improvement alone is insufficient.

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Data quality issues | Cross-validate with multiple sources |
| API rate limits | Cache aggressively, batch requests |
| Overfitting to BTC/ETH | Test on full 32-symbol universe |
| Time sink | Hard 20-hour timebox, stop if not promising |

---

## Do NOT Start Until

1. ✅ Phase 9.7 documented and closed
2. ⏳ Paper trading Phase 1 launched and stable
3. ⏳ At least 10 paper trades completed
4. ⏳ No critical issues in paper trading

---

## Notes

- This is a **stub document** - do not implement yet
- Focus remains on paper trading the BASE system
- Allocate max 30% of research time to Priority 2
- Can be deprioritized if paper trading needs attention

---

*Created: 2026-01-28*
*Status: PLANNED*
