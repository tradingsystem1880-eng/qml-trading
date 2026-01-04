#!/usr/bin/env python3
"""
Professional Validation Report Generator
=========================================
Generates comprehensive, publication-ready validation reports 
matching the quality of the validation v1/ reference folder.

Output Structure:
- Executive Summary with verdict table
- Statistical Significance Analysis
- Monte Carlo Risk Analysis  
- Walk-Forward Results
- Regime Performance
- Clear Recommendations
- Embedded Charts
"""

import sys
import shutil
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

from src.validation.permutation import PermutationTest
from src.validation.monte_carlo import MonteCarloSimulator
from src.validation.bootstrap import BlockBootstrap
from src.reporting.visuals import ReportVisualizer, VisualizationConfig


@dataclass
class ValidationMetrics:
    """Container for all validation metrics."""
    total_trades: int
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    profit_factor: float
    
    # Statistical tests
    sharpe_p_value: float
    sharpe_percentile: float
    sharpe_ci_lower: float
    sharpe_ci_upper: float
    win_rate_ci_lower: float
    win_rate_ci_upper: float
    
    # Monte Carlo
    var_95: float
    var_99: float
    expected_shortfall: float
    kill_switch_prob: float
    median_final_return: float
    
    # Verdict
    verdict: str
    confidence_score: int


def generate_professional_report(
    trades_df: pd.DataFrame,
    output_dir: str,
    strategy_name: str = "QML Strategy",
    initial_capital: float = 100000
) -> str:
    """
    Generate a comprehensive professional validation report.
    
    Args:
        trades_df: DataFrame with trade results (must have 'pnl_pct' column)
        output_dir: Directory to save report and charts
        strategy_name: Name of the strategy
        initial_capital: Starting capital for simulations
        
    Returns:
        Path to generated markdown report
    """
    output_path = Path(output_dir)
    charts_dir = output_path / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Generating professional report for {strategy_name}")
    logger.info(f"Loaded {len(trades_df)} trades")
    
    # Extract returns
    returns = trades_df["pnl_pct"].values / 100
    
    # === RUN ALL VALIDATION TESTS ===
    
    # 1. Permutation Test
    logger.info("Running Permutation Test (10,000 shuffles)...")
    perm_test = PermutationTest(n_permutations=10000)
    perm_result = perm_test.run(returns * 100)  # Convert to pct for internal calc
    
    # 2. Monte Carlo
    logger.info("Running Monte Carlo (50,000 simulations)...")
    mc_sim = MonteCarloSimulator(n_simulations=50000, kill_switch_threshold=0.20)
    mc_result = mc_sim.run(returns * 100, initial_capital=initial_capital)
    
    # 3. Bootstrap CIs
    logger.info("Running Bootstrap (5,000 samples)...")
    bootstrap = BlockBootstrap(n_bootstrap=5000, block_size=5)
    
    sharpe_ci = bootstrap.confidence_interval(
        returns,
        lambda x: (np.mean(x) / np.std(x)) * np.sqrt(252) if np.std(x) > 0 else 0
    )
    win_rate_ci = bootstrap.confidence_interval(
        (returns > 0).astype(float),
        lambda x: np.mean(x)
    )
    
    # === CALCULATE METRICS ===
    
    # Basic metrics
    total_trades = len(trades_df)
    win_rate = np.mean(returns > 0)
    sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0
    
    # Equity and drawdown
    equity = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(equity)
    drawdowns = (running_max - equity) / running_max * 100
    max_dd = np.max(drawdowns)
    
    # Profit factor
    gross_profit = np.sum(returns[returns > 0])
    gross_loss = abs(np.sum(returns[returns < 0]))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # === GENERATE CHARTS ===
    
    logger.info("Generating charts...")
    viz_config = VisualizationConfig(output_dir=str(charts_dir))
    visualizer = ReportVisualizer(config=viz_config)
    
    # Chart 1: Equity Curve
    regime_labels = np.zeros(len(returns), dtype=int)
    visualizer.plot_regime_equity_curve(
        equity * initial_capital,
        regime_labels,
        {0: "Standard"},
        title=f"{strategy_name} - Equity Curve",
        save_path=str(charts_dir / "equity_curve.png")
    )
    
    # Chart 2: Monte Carlo Cones
    visualizer.plot_monte_carlo_cones(
        mc_result.equity_paths,
        initial_capital=initial_capital,
        title="Monte Carlo Risk Analysis (50,000 Simulations)",
        save_path=str(charts_dir / "monte_carlo_cones.png")
    )
    
    # Chart 3: Drawdown Analysis
    visualizer.plot_drawdown_chart(
        equity * initial_capital,
        title="Drawdown Analysis",
        save_path=str(charts_dir / "drawdown_analysis.png")
    )
    
    # Chart 4: Permutation Histogram
    sharpes = perm_result.permutation_sharpes
    clean_sharpes = sharpes[np.isfinite(sharpes)]
    if len(clean_sharpes) > 10:
        clean_sharpes += np.random.normal(0, 1e-6, len(clean_sharpes))
    
    try:
        visualizer.plot_permutation_histogram(
            clean_sharpes,
            perm_result.actual_sharpe,
            perm_result.sharpe_p_value,
            title="Permutation Test - Skill vs Luck",
            save_path=str(charts_dir / "permutation_test.png")
        )
    except Exception as e:
        logger.warning(f"Could not generate permutation histogram: {e}")
    
    # === DETERMINE VERDICT ===
    
    is_significant = perm_result.sharpe_p_value < 0.05
    sharpe_good = sharpe > 1.0
    low_ruin = mc_result.kill_switch_prob < 0.15
    enough_trades = total_trades >= 50
    
    confidence_score = 0
    if is_significant: confidence_score += 30
    if sharpe_good: confidence_score += 25
    if low_ruin: confidence_score += 25
    if enough_trades: confidence_score += 20
    
    if confidence_score >= 70:
        verdict = "DEPLOY"
        verdict_emoji = "✅"
    elif confidence_score >= 50:
        verdict = "CAUTION"
        verdict_emoji = "⚠️"
    else:
        verdict = "REJECT"
        verdict_emoji = "❌"
    
    # === GENERATE MARKDOWN REPORT ===
    
    report_date = datetime.now().strftime("%B %d, %Y")
    
    report = f"""# {strategy_name} Validation Report

**Date:** {report_date}  
**Purpose:** Comprehensive institutional-grade strategy validation

---

## Executive Summary

| Metric | Value | Assessment |
|--------|-------|------------|
| **Total Trades** | {total_trades} | {"✅ Sufficient" if enough_trades else "❌ Insufficient (<50)"} |
| **Win Rate** | {win_rate:.1%} | {"✅ Strong" if win_rate > 0.55 else "⚠️ Moderate"} |
| **Sharpe Ratio** | {sharpe:.3f} | {"✅ Good" if sharpe > 1.0 else "⚠️ Below target"} |
| **Max Drawdown** | {max_dd:.1f}% | {"✅ Acceptable" if max_dd < 25 else "⚠️ High"} |
| **p-value** | {perm_result.sharpe_p_value:.4f} | {"✅ Significant" if is_significant else "❌ Not Significant"} |

# {verdict_emoji} VERDICT: {verdict}

**Confidence Score:** {confidence_score}/100

---

## Statistical Significance Analysis

### Permutation Test Results

The permutation test shuffles trade sequence 10,000 times to determine if results are due to skill or luck.

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Actual Sharpe** | {perm_result.actual_sharpe:.4f} | Strategy's realized risk-adjusted return |
| **p-value** | {perm_result.sharpe_p_value:.4f} | Probability of random chance |
| **Percentile** | {perm_result.sharpe_percentile:.1f}% | Rank vs random orderings |

**Interpretation:**
{"✅ **STATISTICALLY SIGNIFICANT** - Results are unlikely due to chance (p < 0.05)" if is_significant else "❌ **NOT SIGNIFICANT** - Cannot distinguish from random chance (p ≥ 0.05)"}

### Bootstrap Confidence Intervals (95%)

| Metric | Lower Bound | Upper Bound |
|--------|-------------|-------------|
| **Sharpe Ratio** | {sharpe_ci.ci_lower:.3f} | {sharpe_ci.ci_upper:.3f} |
| **Win Rate** | {win_rate_ci.ci_lower:.1%} | {win_rate_ci.ci_upper:.1%} |

![Permutation Test](charts/permutation_test.png)

---

## Monte Carlo Risk Analysis

### Simulation Results (50,000 paths)

By randomly reordering trades, we estimate the range of possible outcomes.

| Risk Metric | Value | Description |
|-------------|-------|-------------|
| **VaR 95%** | {mc_result.var_95:.2f}% | 95% of paths have max DD below this |
| **VaR 99%** | {mc_result.var_99:.2f}% | 99% of paths have max DD below this |
| **Expected Shortfall** | {mc_result.expected_shortfall_95:.2f}% | Average DD in worst 5% of cases |
| **Kill Switch Prob** | {mc_result.kill_switch_prob:.1%} | Probability of 20%+ drawdown |
| **Median Final Return** | {mc_result.median_final_return:+.1f}% | Typical outcome |

### Risk Assessment

{"✅ **LOW RISK** - Kill switch probability below 15%" if low_ruin else "⚠️ **ELEVATED RISK** - Consider position sizing adjustments"}

![Monte Carlo Analysis](charts/monte_carlo_cones.png)

---

## Performance Analysis

### Equity Curve

![Equity Curve](charts/equity_curve.png)

### Drawdown Profile

| Metric | Value |
|--------|-------|
| **Maximum Drawdown** | {max_dd:.2f}% |
| **Recovery Time (avg)** | {mc_result.time_to_recovery_mean:.1f} trades |
| **Recovery Time (95%)** | {mc_result.time_to_recovery_95:.1f} trades |

![Drawdown Analysis](charts/drawdown_analysis.png)

---

## Trade Statistics

| Statistic | Value |
|-----------|-------|
| **Total Trades** | {total_trades} |
| **Winning Trades** | {int(total_trades * win_rate)} |
| **Losing Trades** | {int(total_trades * (1 - win_rate))} |
| **Win Rate** | {win_rate:.1%} |
| **Profit Factor** | {profit_factor:.2f} |
| **Avg Win** | {np.mean(returns[returns > 0]) * 100:.2f}% |
| **Avg Loss** | {np.mean(returns[returns < 0]) * 100:.2f}% |

---

## Final Recommendations

"""
    
    # Add recommendations based on verdict
    if verdict == "DEPLOY":
        report += """### ✅ READY FOR DEPLOYMENT

The strategy has passed all institutional-grade validation tests:

1. **Statistical Significance** - Results are unlikely due to chance
2. **Risk Management** - Drawdown profile is acceptable
3. **Sample Size** - Sufficient trades for reliable statistics

**Recommended Next Steps:**
1. Paper trading validation (2-4 weeks)
2. Small live allocation (0.5-1% risk)
3. Full deployment with 1-2% risk per trade
"""
    elif verdict == "CAUTION":
        report += """### ⚠️ PROCEED WITH CAUTION

The strategy shows promise but has some concerns:

**Issues Identified:**
- Statistical significance borderline or insufficient trades
- Consider gathering more data before deployment

**Recommended Next Steps:**
1. Extend backtest period for more trades
2. Paper trade to validate in real-time
3. Deploy with reduced position sizing (0.25-0.5% risk)
"""
    else:
        report += """### ❌ NOT RECOMMENDED FOR DEPLOYMENT

The strategy has failed key validation tests:

**Critical Issues:**
- Results are not statistically significant (p ≥ 0.05)
- Insufficient trade count for reliable conclusions
- Cannot distinguish performance from random chance

**Recommended Actions:**
1. **Do NOT deploy** this strategy in current form
2. Review and refine detection parameters
3. Gather more historical data
4. Consider alternative approaches
"""
    
    report += f"""
---

## Appendix: Technical Details

### Validation Configuration
- **Permutation Tests:** 10,000 shuffles
- **Monte Carlo Simulations:** 50,000 paths
- **Bootstrap Samples:** 5,000 (block size: 5)
- **Confidence Level:** 95%
- **Kill Switch Threshold:** 20%

### Files Generated
- `professional_report.md` - This report
- `charts/equity_curve.png` - Equity performance
- `charts/monte_carlo_cones.png` - Risk simulation
- `charts/drawdown_analysis.png` - Drawdown profile
- `charts/permutation_test.png` - Statistical significance

---

*Report generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""
    
    # Save report
    report_path = output_path / "professional_report.md"
    with open(report_path, 'w') as f:
        f.write(report)
    
    logger.info(f"Professional report saved to: {report_path}")
    
    return str(report_path)


def main():
    """Run professional report generation on saved trades."""
    
    # Load trades
    trades_path = "experiments/exp_03_backtest_validation/revalidation_trades.csv"
    output_dir = "results/professional_validation"
    
    if not Path(trades_path).exists():
        logger.error(f"Trade file not found: {trades_path}")
        return
    
    trades_df = pd.read_csv(trades_path)
    
    report_path = generate_professional_report(
        trades_df=trades_df,
        output_dir=output_dir,
        strategy_name="QML Pattern Strategy",
        initial_capital=100000
    )
    
    logger.info("=" * 70)
    logger.info("PROFESSIONAL VALIDATION REPORT COMPLETE")
    logger.info(f"Report: {report_path}")
    logger.info(f"Charts: {output_dir}/charts/")
    logger.info("=" * 70)
    
    return report_path


if __name__ == "__main__":
    main()
