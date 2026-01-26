# Phase 7.8: Volume/Momentum/Regime Filters - COMPLETE

**Date**: 2026-01-26
**Duration**: ~3 hours

## Overview

Phase 7.7 optimization (56 hours, 3,000 iterations) revealed that pattern detection works well (0.91 quality score) but profitability was marginal (Sharpe 0.04, PF 0.88, Win Rate 41.9%). Phase 7.8 adds filters to improve trade selection.

## Key Changes

### 1. Activated Phase 7.6 Metrics (Previously Unused)

Phase 7.6 added volume_spike_score, path_efficiency_score, and trend_strength_score to the PatternScorer, but no call sites were passing `df` or `trend_result`. Fixed by wiring these parameters through:

- `src/optimization/parallel_runner.py` - Now passes `df`, `trend_result`, `regime_result` to scorer
- `src/optimization/extended_runner.py` - Same wiring
- `scripts/multi_symbol_detection.py` - Now passes `df` and `regime_result` to scorer

### 2. Made Phase 7.6 Weights Optimizable

Previously fixed at 10% each, now tunable via Bayesian optimization:

| Weight Parameter | Range | Default |
|-----------------|-------|---------|
| volume_spike_weight | 0.05 - 0.10 | 0.10 |
| path_efficiency_weight | 0.05 - 0.10 | 0.10 |
| trend_strength_weight | 0.05 - 0.10 | 0.10 |

### 3. Added Regime Suitability Scoring (Main Feature)

**Layered Filtering Approach:**

1. **Hard Rejection**: Patterns in TRENDING regime with ADX > 35 are rejected (returns score=0, tier=REJECT)
2. **Soft Scoring**: Regime affects quality score via new component

**Regime Scores:**
| Regime | Base Score | Rationale |
|--------|------------|-----------|
| RANGING | 1.0 | Ideal for QML (mean-reversion patterns) |
| VOLATILE | 0.6 | Acceptable with caution |
| EXTREME | 0.5 | High risk/reward |
| TRENDING | 0.2 | Poor for QML (trend-following dominates) |

Score is modulated by regime confidence: `final_score = base_score * confidence`

### 4. Rebalanced Scoring Weights

Updated default weights to sum to 1.0 with 8 components:

| Component | Old Weight | New Weight | Notes |
|-----------|------------|------------|-------|
| head_extension | 25% | 22% | Core geometry |
| bos_efficiency | 20% | 18% | Core geometry |
| shoulder_symmetry | 15% | 12% | Fixed |
| swing_significance | 10% | 8% | Auto-calculated |
| volume_spike | 10% | 10% | Now optimizable |
| path_efficiency | 10% | 10% | Now optimizable |
| trend_strength | 10% | 10% | Now optimizable |
| **regime_suitability** | 0% | **10%** | NEW |

## Files Modified

| File | Changes |
|------|---------|
| `src/detection/config.py` | Added regime scoring params, rebalanced weights |
| `src/detection/pattern_scorer.py` | Added `_calculate_regime_suitability()`, updated `score()` |
| `src/optimization/parallel_runner.py` | Wired df/trend_result/regime_result, weight extraction |
| `src/optimization/extended_runner.py` | Same wiring as parallel_runner |
| `scripts/run_phase77_optimization.py` | Added 6 optimizable weight params |
| `scripts/multi_symbol_detection.py` | Added regime detection |

## New Configuration Parameters

Added to `PatternScoringConfig`:

```python
# Phase 7.8 regime suitability
regime_ranging_score: float = 1.0
regime_volatile_score: float = 0.6
regime_extreme_score: float = 0.5
regime_trending_score: float = 0.2
regime_hard_reject_adx: float = 35.0
regime_suitability_weight: float = 0.10
```

## Optimization Space

Total parameters increased from 23 to 29:
- 6 new optimizable scoring weights (Phase 7.8)

Weight bounds constrained to ensure sum <= 0.86 (leaving room for fixed shoulder=0.12 and min swing=0.02).

## Verification

All tests pass:
1. PatternScorer instantiation with regime_result
2. MarketRegimeDetector regime classification
3. PatternScoringConfig weight validation (sum = 1.0)
4. ParallelDetectionRunner weight extraction

Multi-symbol detection test run:
- 17 patterns detected across 3 symbols
- 11 trades, 72.7% win rate

## Expected Outcomes

Running a new optimization with Phase 7.8 changes should:
- Improve win rate (41.9% → 45%+) by filtering low-probability setups
- Improve profit factor (0.88 → 1.0+) by avoiding trending markets
- Reduce max drawdown by skipping high-risk regime trades
- May reduce pattern count 10-20% (acceptable for quality)

## Next Steps

1. Run short optimization to verify new params work: `python scripts/run_phase77_optimization.py --objective composite --iterations 100`
2. If promising, run extended optimization (6-12 hours)
3. Consider additional filters: volume confirmation at P3/P4, momentum divergence, higher-TF alignment

## Commands

```bash
# Test multi-symbol detection with Phase 7.8
python scripts/multi_symbol_detection.py --symbols ETHUSDT,BNBUSDT,SOLUSDT

# Run short optimization test
python scripts/run_phase77_optimization.py --objective composite --iterations 100

# Run full optimization
python scripts/run_phase77_optimization.py --objective composite --iterations 500
```
