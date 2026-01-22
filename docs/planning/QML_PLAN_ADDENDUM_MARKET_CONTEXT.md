# QML PROJECT ADDENDUM: Market Microstructure Context Features
## Funding Rates, Open Interest, Liquidations

*Added: January 20, 2026*
*Status: PLANNED - Phase 9-11 (NOT current priority)*

---

## Executive Summary

Market microstructure data (funding rates, open interest, liquidations) can significantly improve QML pattern probability assessment by revealing the "fuel" behind potential reversals. However, this is **Phase 9-11 work** - do not implement until baseline system is validated.

---

## The Metrics Explained

### 1. Funding Rates
**What**: Cost to hold perpetual futures positions
- Positive funding = longs pay shorts (market is long-heavy)
- Negative funding = shorts pay longs (market is short-heavy)

**Signal for QML**:
| Scenario | Interpretation |
|----------|----------------|
| Bearish QML + High positive funding | Strong signal - crowded longs = fuel for drop |
| Bullish QML + High negative funding | Strong signal - crowded shorts = fuel for squeeze |
| Pattern + Neutral funding | Weaker signal - no positioning extreme |

**Data Source**: Binance API (free), 8-hour intervals

### 2. Open Interest (OI)
**What**: Total outstanding derivative contracts (money at stake)

**Signal for QML**:
| Scenario | Interpretation |
|----------|----------------|
| Pattern + Rising OI | New money entering, stronger conviction |
| Pattern + Falling OI | Profit-taking, weaker move expected |
| Pattern + High stagnant OI | Trapped positions, higher reversal potential |

**Data Source**: Binance API, Coinglass (free tier)

### 3. Liquidation Levels
**What**: Estimated price levels where leveraged positions get force-closed

**Signal for QML**:
| Scenario | Interpretation |
|----------|----------------|
| BOS targets liquidation cluster | Higher probability of volatile move |
| Pattern forms near liquidity void | Less "fuel" for the move |

**Data Source**: 
- Historical: Coinglass Pro (paid)
- Real-time estimates: Velo.xyz, Coinglass (proprietary algorithms)

---

## Implementation Phases

### Phase 9: Manual Error Analysis (PREREQUISITE)
**Time**: 2-3 hours
**Goal**: Validate if context data correlates with pattern success/failure

**Process**:
1. Export 100 losing trades from backtest
2. Export 100 winning trades from backtest
3. For each trade, manually check:
   - Funding rate at pattern completion (Binance or Coinglass)
   - OI change over prior 24h
   - Proximity to visible liquidation clusters (Velo)
4. Create spreadsheet comparing winners vs losers
5. Look for statistically significant differences

**Decision Gate**: If no clear pattern emerges, STOP. Context features won't help.

### Phase 10: Rule-Based Context Overlay
**Time**: 1-2 days
**Goal**: Test simple rules before ML complexity

**Implementation**:
```python
# Simple context adjustment - NO ML YET
class ContextOverlay:
    def __init__(self):
        self.funding_threshold = 0.01  # 1% funding rate
        self.oi_change_threshold = 0.05  # 5% OI change
    
    def adjust_signal(self, pattern, market_context):
        """
        Adjust pattern confidence based on market context.
        Returns multiplier (0.5 to 1.5)
        """
        multiplier = 1.0
        
        # Funding alignment
        if pattern.direction == 'BEARISH':
            if market_context.funding_rate > self.funding_threshold:
                multiplier *= 1.2  # Crowded longs = bearish fuel
            elif market_context.funding_rate < -self.funding_threshold:
                multiplier *= 0.8  # Crowded shorts = risky bearish
        
        elif pattern.direction == 'BULLISH':
            if market_context.funding_rate < -self.funding_threshold:
                multiplier *= 1.2  # Crowded shorts = bullish fuel
            elif market_context.funding_rate > self.funding_threshold:
                multiplier *= 0.8  # Crowded longs = risky bullish
        
        # OI confirmation
        if market_context.oi_change_24h > self.oi_change_threshold:
            multiplier *= 1.1  # Rising OI = conviction
        elif market_context.oi_change_24h < -self.oi_change_threshold:
            multiplier *= 0.9  # Falling OI = weak
        
        return max(0.5, min(1.5, multiplier))
```

**Backtest**: Compare baseline vs rule-adjusted performance
**Decision Gate**: If no improvement, don't proceed to Phase 11

### Phase 11: ML Context Integration
**Time**: 1-2 weeks
**Goal**: Full neural network integration

**Architecture** (only if Phase 10 succeeds):
```
┌─────────────────────────────────────────────────────────────┐
│                    FUSION ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Stream A: Pattern (existing)     Stream B: Context (new)  │
│  ┌─────────────────────────┐     ┌─────────────────────┐   │
│  │  OHLCV Window (50 bars) │     │  Context Vector     │   │
│  │         ↓               │     │  - Funding (z-score)│   │
│  │      CNN Layers         │     │  - OI change 24h    │   │
│  │         ↓               │     │  - OI change 7d     │   │
│  │      LSTM/Attention     │     │  - Dist to liq      │   │
│  │         ↓               │     │         ↓           │   │
│  │   Feature Vector (128)  │     │   Dense (32)        │   │
│  └───────────┬─────────────┘     └──────────┬──────────┘   │
│              │                              │               │
│              └──────────┬───────────────────┘               │
│                         ↓                                   │
│                   Concatenate (160)                         │
│                         ↓                                   │
│                   Dense (64) + Dropout                      │
│                         ↓                                   │
│              Classification/Regression Head                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Context Feature Vector**:
| Feature | Normalization | Source |
|---------|---------------|--------|
| `funding_rate` | Z-score (30-day) | Binance API |
| `funding_rate_change_8h` | Raw delta | Calculated |
| `oi_change_24h` | Percentage | Binance/Coinglass |
| `oi_change_7d` | Percentage | Binance/Coinglass |
| `oi_percentile_30d` | 0-100 | Calculated |
| `distance_to_liq_cluster` | ATR-normalized | Coinglass (if available) |

---

## Data Pipeline Requirements

### Free Data Sources
```python
# Binance Funding Rate
GET /fapi/v1/fundingRate?symbol=BTCUSDT&limit=1000

# Binance Open Interest
GET /fapi/v1/openInterest?symbol=BTCUSDT

# Historical OI
GET /futures/data/openInterestHist?symbol=BTCUSDT&period=5m&limit=500
```

### Paid Data Sources
- **Coinglass Pro**: $50-100/month - historical liquidations, aggregated OI
- **Velo Data API**: Contact for pricing - liquidation heatmaps

### Data Alignment Challenge
```
CRITICAL: All context data must be point-in-time accurate

Pattern completes at: 2025-01-15 14:00 UTC
You must use:
- Funding rate from 08:00 UTC (last settlement before pattern)
- OI as of 14:00 UTC exactly
- NOT future data that wasn't available at decision time

This requires careful timestamp management in your data pipeline.
```

---

## Dashboard Integration

### New Cards for Command Bar (Phase 10+)
```
┌─────────┐ ┌─────────┐ ┌─────────┐
│ FUNDING │ │   OI    │ │  LIQ    │
│ +0.021% │ │ +3.2%   │ │ $95.2K  │
│ 🔴 High │ │ Rising  │ │ ↓ Near  │
└─────────┘ └─────────┘ └─────────┘
```

### Pattern Card Enhancement
```
┌────────────────────────────────────────┐
│  BEARISH QML - BTCUSDT 4H              │
│  Base Confidence: 72%                  │
│  ─────────────────────────────         │
│  Context Adjustments:                  │
│  • Funding +0.021% (crowded longs) +8% │
│  • OI rising 3.2% (conviction)     +4% │
│  • Near liq cluster $94.8K         +5% │
│  ─────────────────────────────         │
│  ADJUSTED CONFIDENCE: 89%              │
└────────────────────────────────────────┘
```

---

## Risk & Warnings

### 1. Look-Ahead Bias
Using future funding/OI data that wasn't available at pattern completion = fatal error. All data must be point-in-time.

### 2. Regime Dependence
High funding = top signal worked in 2021. In 2024-2025, markets can trend with high funding for weeks. Model may overfit to historical regime.

### 3. Data Quality
- Liquidation estimates are proprietary algorithms, not ground truth
- Exchange OI varies wildly - need to aggregate or pick one source
- Historical data gaps common

### 4. Complexity Creep
Each feature adds potential for bugs, overfitting, and maintenance burden. Violates 90% rule if baseline isn't working.

---

## Decision Checklist

Before starting Phase 9:
- [ ] Dashboard complete and working
- [ ] Baseline detection logic finalized
- [ ] 200+ backtest trades available for analysis
- [ ] Baseline metrics documented (win rate, Sharpe, etc.)

Before starting Phase 10:
- [ ] Phase 9 manual analysis shows clear correlation
- [ ] At least 70% of losing trades had identifiable context issues
- [ ] Data sources identified and accessible

Before starting Phase 11:
- [ ] Phase 10 rule-based overlay shows measurable improvement
- [ ] Win rate improved by >5% with rules
- [ ] Sharpe improved by >0.2 with rules

---

## References

- Velo.xyz: https://velo.xyz/futures/BTC
- Coinglass: https://www.coinglass.com/
- Binance Futures API: https://binance-docs.github.io/apidocs/futures/en/
- DeepSeek analysis on market microstructure integration (January 2026)

---

*This is a PLANNING document. Implementation should not begin until Phase 1-8 are complete.*
