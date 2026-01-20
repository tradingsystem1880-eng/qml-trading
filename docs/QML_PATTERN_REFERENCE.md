# Quasimodo (QML) Pattern - Technical Analysis Reference

## 1. Pattern Definition
The Quasimodo is a reversal chart pattern resembling a hunchback, consisting of three parts:
- **Left Shoulder**: A small peak/high point.
- **Head**: A higher peak forming the "hump".
- **Right Shoulder**: A smaller peak similar to the left shoulder.

It indicates a potential trend reversal and is used across all markets (forex, crypto, stocks, etc.).

## 2. Core Concepts for Detection
- **Timeframe**: Works on all timeframes (1min to yearly). Primary analysis often uses 1-hour for setup, 15-minute for entry.
- **Market Trend**: Identify existing trend (uptrend: higher highs/lows; downtrend: lower highs/lows; sideways).
- **Break of Structure (BOS)**: When price violates established trend boundaries.
  - Uptrend BOS: Breaks last high upward.
  - Downtrend BOS: Breaks last low downward.
- **Impulsive Move/Imbalance**: Rapid, high-volume price movement indicating order flow imbalance.
- **Order Block**: Area where large capital entered, causing sharp price movement. Drawn as thick zones. Bullish order blocks form at range lows/support; bearish at range highs/resistance. Typically one-time use.
- **Fair Value Gap (FVG)**: Temporary price gap from buyer/seller imbalance, appearing as an "unadjusted" area on chart.
- **Change of Character (CHoCH)**: Shift in market momentum/structure signaling potential reversal. Must be validated with breakout above/below major high/low combined with demand/supply zone.
- **Liquidity**: Zones where stop losses, buy/sell orders cluster (especially at highs/lows). Market often "sweeps" liquidity before reversing.
- **Pullback**: Temporary reversal/correction within a trend before continuation.

## 3. Trading Setup Logic

### Bullish QML Setup Steps:
1. On 1-hour chart, mark recent swing low.
2. Wait for price to break last low downward (first BOS), then break structure upward (initial reversal sign).
3. Wait for deep pullback downward, then second break of structure upward (confirms strength).
4. Mark the pullback low as liquidity area.
5. Mark the order block that formed the first BOS and the "protected low" (swing low before first BOS).
6. Wait for price to sweep the pullback low liquidity (liquidity grab).
7. Enter long after sweep, targeting previous swing high.
8. Stop loss placed slightly below protected low.
9. Target 1:3 risk:reward; move to breakeven at 1:1.

### Bearish QML Setup Steps:
1. On 1-hour chart, mark recent swing high.
2. Wait for price to break last high upward (first BOS), then break structure downward.
3. Identify liquidity at pullback highs (e.g., double tops).
4. Mark order block that formed the first BOS and the "protected high".
5. Enter short after liquidity sweep, targeting previous swing low.
6. Stop loss placed slightly above protected high.
7. Target 1:3 risk:reward; move to breakeven at 1:1.

## 4. Key Validation Rules
- Requires **two breaks of structure** (first establishes potential reversal, second confirms).
- Must see **liquidity sweep** of pullback high/low before entry.
- **Order block** must be present near first BOS.
- Trade only after **pullback** and second BOS.
- Higher timeframe structure dominance: HTF highs/lows matter more than LTF.

## 5. Risk & Discipline Rules
- Risk only what you can afford to lose.
- Maximum 15 pips stop loss.
- Target 1:2 or 1:3+ risk:reward.
- Maximum 2 trades per day, 5 currency pairs.
- No trading without stop loss/take profit.
- No revenge trading or chasing markets.
- No trades without quality setup confirmation.

## 6. Pattern Characteristics for Code Detection
1. Look for three-peak structure with middle peak highest.
2. Identify trend before pattern formation.
3. Detect break of structure events (price beyond previous high/low).
4. Identify order blocks (sharp price movement zones).
5. Detect liquidity sweeps (price tapping previous high/low before reversal).
6. Validate with two BOS sequences and pullback.
7. Confirm with volume/impulsive move characteristics.

---

## Visual Elements for Chart Annotation

When displaying detected patterns, the chart should show:

| Element | Description | Color |
|---------|-------------|-------|
| Trend Line | Initial trend before pattern | Green (up) / Red (down) |
| Swing Points 1-5 | Key structure points | Blue circles with numbers |
| BOS Line 1 | First break of structure (trend continuation) | Dashed, trend color |
| BOS Line 2 | Second break (reversal confirmation) | Dashed, opposite color |
| Order Block | Zone where first BOS originated | Semi-transparent box |
| Liquidity Sweep | Where price swept before entry | Small marker/arrow |
| Entry Level | Entry price | Blue solid line |
| Stop Loss Zone | SL area | Red shaded box |
| Take Profit Zone(s) | TP area(s) | Green shaded box(es) |

## Swing Point Mapping

### Bullish QML (LONG):
- P1: Swing low in downtrend
- P2: Swing high (gets broken = first BOS down)
- P3: Lower low (the "head" - liquidity sweep target)
- P4: Breaks above P2 (second BOS - confirms bullish)
- P5: Retest of zone = ENTRY

### Bearish QML (SHORT):
- P1: Swing high in uptrend
- P2: Swing low (gets broken = first BOS up)
- P3: Higher high (the "head" - liquidity sweep target)
- P4: Breaks below P2 (second BOS - confirms bearish)
- P5: Retest of zone = ENTRY
