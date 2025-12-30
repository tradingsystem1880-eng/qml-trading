# QML (Quasimodo Level) Strategy Documentation

## 1. Introduction
The QML (Quasimodo Level) strategy is a reversal pattern strategy applicable to all markets (crypto, stocks, commodities, forex, synthetic indices). It relies on identifying specific market structures that indicate a shift in momentum from buyers to sellers (or vice versa).

**Important Notice:** Before applying this strategy to a live account, it is crucial to backtest thoroughly (recommended: at least 100 backtests) to understand performance under various conditions.

## 2. General Trading Rules
Adherence to these rules is mandatory for the strategy's success:
1.  **Patience:** Wait for the exact setup criteria; do not rush.
2.  **Risk Management:** Do not overtrade or over-risk.
3.  **Psychology:** Avoid revenge trading. Stay objective.
4.  **Consistency:** Avoid flipping accounts; treat trading as a business.
5.  **Discipline:** Don't force a trade if criteria aren't met.
6.  **Focus:** Specialize in a few pairs/assets.
7.  **R:R Ratio:** Target a minimum Risk-to-Reward of 1:2 or 1:3.
8.  **Trend:** Do not counter-trade blindly; follow the higher timeframe trend.

## 3. Key Concepts & Definitions

### 3.1 Quasimodo Pattern (QM)
A technical analysis reversal pattern resembling a "Head and Shoulders" but with specific structural requirements.
* **Structure:**
    1.  **Left Shoulder:** A peak/low.
    2.  **Head:** A higher peak (in bearish setup) or lower valley (in bullish setup).
    3.  **Right Shoulder:** A retracement that tests the level of the Left Shoulder.

### 3.2 Timeframes
* **Short-term:** 1m, 5m, 15m, 30m, 1h (Scalping/Intraday).
* **Medium-term:** 4h, Daily, Weekly.
* **Long-term:** Monthly, Yearly.
* **Note:** Higher Timeframe (HTF) structure carries more weight than Lower Timeframe (LTF).

### 3.3 Market Trends
* **Uptrend:** Higher Highs (HH) and Higher Lows (HL).
* **Downtrend:** Lower Highs (LH) and Lower Lows (LL).
* **Sideways/Consolidation:** No clear direction; price moves within a range.

### 3.4 Break of Structure (BOS)
Occurs when price closes beyond a significant structural point (Swing High or Swing Low) in the direction of the trend.
* **Uptrend BOS:** Breaking the previous High to the upside.
* **Downtrend BOS:** Breaking the previous Low to the downside.

### 3.5 Impulsive Move / Imbalance
A sudden, significant price movement driven by institutional orders, leaving "gaps" or inefficient pricing.
* **Characteristics:** Rapid movement, increased volatility, large candles.

### 3.6 Order Block (OB)
A specific candle or zone where "Smart Money" (institutions) entered the market, causing an impulsive move.
* **Bullish OB:** The last sell candle (down move) before a strong impulsive upward move. Found at the bottom of a range/support.
* **Bearish OB:** The last buy candle (up move) before a strong impulsive downward move. Found at the top of a range/resistance.
* **Usage:** Price often returns to "retest" these zones. Unlike support/resistance, OBs are typically "one-time use."

### 3.7 Fair Value Gap (FVG)
An imbalance between buyers and sellers visible as a gap between the wick of the first candle and the wick of the third candle in a 3-candle sequence. It acts as a magnet for price.

### 3.8 Change of Character (CHoCH)
The first sign of a potential trend reversal.
* **Bullish CHoCH:** In a downtrend (lower highs, lower lows), price breaks the most recent Lower High (LH) to the upside.
* **Bearish CHoCH:** In an uptrend (higher highs, higher lows), price breaks the most recent Higher Low (HL) to the downside.

### 3.9 Liquidity
Zones where large pools of Stop Losses or Limit Orders exist. "Smart Money" targets these areas to fill large orders.
* **Forms:** Equal Highs (EQH / $$$), Equal Lows (EQL / $$$), Trendline Liquidity.
* **Rule:** The market seeks liquidity to generate momentum.

## 4. Trading The QML Strategy (Step-by-Step)

### 4.1 Bullish Setup (Long)
**Goal:** Catch the reversal from a downtrend to an uptrend.

1.  **Identify Swing Low (HTF):** On the 1H timeframe, mark the recent major Swing Low.
2.  **First Break of Structure (BOS):** Wait for price to break a structural high to the upside (Initial sign of strength).
3.  **Wait for Pullback & Second BOS:** Wait for a deep pullback, followed by a *second* break of structure to the upside.
4.  **Mark the Protected Low:** The lowest point of this structure is your "Protected Low" (Invalidation point).
5.  **Identify the Order Block:** Mark the Order Block (demand zone) that caused the initial impulsive move/break.
6.  **Identify Liquidity:** Look for a "Pullback Low" or Equal Lows ($$$) resting *above* your Order Block.
7.  **Wait for Liquidity Sweep:** Price must come down and "sweep" (break) this liquidity.
8.  **Entry:** Enter Long after the liquidity sweep taps into the Order Block.
9.  **Stop Loss:** A few pips below the Protected Low.
10. **Target (TP):** The Swing High. Aim for 1:3 RR. Move to breakeven at 1:1.

### 4.2 Bearish Setup (Short)
**Goal:** Catch the reversal from an uptrend to a downtrend.

1.  **Identify Swing High (HTF):** On the 1H timeframe, mark the recent major Swing High.
2.  **First Break of Structure (BOS):** Wait for price to break a structural low to the downside (Initial sign of weakness).
3.  **Identify Liquidity:** Look for liquidity (e.g., Double Top/Trendline) forming during the pullback.
4.  **Mark the Order Block:** Identify the Bearish Order Block (supply zone) responsible for the break.
5.  **Mark Protected High:** The highest point is the "Protected High".
6.  **Wait for Liquidity Sweep:** Price must move up to sweep the liquidity resting below the Order Block.
7.  **Entry:** Enter Short (Sell Limit) at the Order Block after the sweep.
8.  **Stop Loss:** A few pips above the Protected High / Swing High.
9.  **Target (TP):** The Swing Low. Aim for 1:3 RR.

## 5. Risk & Money Management Checklist
* **Risk per Trade:** Only risk what you can afford to lose.
* **Drawdown:** Have a set daily limit.
* **Daily Target:** Stop trading once the daily profit target is reached.
* **Execution:**
    * Target 1:2 or 1:3+ Risk/Reward.
    * Move to Breakeven after 1:1.
    * Take partials at 1:1.
    * Maximum Stop Loss: 15 pips (adjust based on asset volatility).
* **Discipline:**
    * Trade max 5 pairs.
    * Max 2 trades per day.
    * No confirmation = No trade.
    * Never trade without a Stop Loss.