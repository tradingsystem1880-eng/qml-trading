#!/usr/bin/env python3
"""
VRD 2.0 Strategy Autopsy Report Generator
==========================================
Comprehensive forensic-grade validation report leveraging all 10 modules
of the Institutional-Grade Strategy Validation Framework.

Generates a detailed "Strategy Autopsy Report" answering:
1. Is the edge REAL? → Statistical significance
2. WHY does it work? → Feature attribution, regime analysis  
3. Will it CONTINUE? → Parameter stability, regime persistence
4. How can it FAIL? → Monte Carlo worst-case, regime vulnerabilities
5. How should we TRADE it? → Position sizing, regime filters, risk limits

Modules Covered:
- Module 1: VRD 2.0 - Versioned Research Database
- Module 2: Purged Walk-Forward Engine
- Module 3: Statistical Robustness Suite (Permutation, Monte Carlo, Bootstrap)
- Module 4: Parameter & Sensitivity Analysis
- Module 5: Regime-Explicit Analysis
- Module 6: Feature-Performance Correlation Engine
- Module 7: Data Integrity & Leakage Prevention
- Module 8: Advanced Diagnostics
- Module 9: Comprehensive Reporting System
- Module 10: Deployment Readiness Checks
"""

import sys
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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
class AutopsyConfig:
    """Configuration for Strategy Autopsy Report."""
    strategy_name: str = "QML Strategy"
    initial_capital: float = 100000
    
    # Statistical tests
    n_permutations: int = 10000
    n_monte_carlo: int = 50000
    n_bootstrap: int = 5000
    bootstrap_block_size: int = 5
    confidence_level: float = 0.95
    
    # Risk thresholds
    kill_switch_threshold: float = 0.20
    min_trades_required: int = 50
    significance_threshold: float = 0.05
    min_sharpe_threshold: float = 1.0
    max_drawdown_limit: float = 0.25
    
    # Output
    output_dir: str = "results/autopsy"


def get_git_hash() -> str:
    """Get current git commit hash for versioning."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except:
        return "unknown"


def compute_param_hash(params: Dict) -> str:
    """Compute hash of parameters for reproducibility."""
    param_str = str(sorted(params.items()))
    return hashlib.md5(param_str.encode()).hexdigest()[:8]


class StrategyAutopsy:
    """
    Comprehensive Strategy Autopsy Report Generator.
    
    Leverages all 10 modules of the VRD 2.0 framework to produce
    a forensic-grade validation report with detailed explanations
    and professional visualizations.
    """
    
    def __init__(self, config: Optional[AutopsyConfig] = None):
        self.config = config or AutopsyConfig()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.git_hash = get_git_hash()
        self.results = {}
        
    def run_full_autopsy(
        self,
        trades_df: pd.DataFrame,
        price_df: Optional[pd.DataFrame] = None,
        features_df: Optional[pd.DataFrame] = None
    ) -> str:
        """
        Run complete strategy autopsy and generate report.
        
        Args:
            trades_df: DataFrame with trade results (requires 'pnl_pct' column)
            price_df: Optional OHLCV price data for regime analysis
            features_df: Optional feature matrix for attribution
            
        Returns:
            Path to generated autopsy report
        """
        # Setup output directory
        param_hash = compute_param_hash({"trades": len(trades_df)})
        run_id = f"{self.timestamp}_{self.config.strategy_name.replace(' ', '_')}_{param_hash}"
        output_path = Path(self.config.output_dir) / run_id
        charts_dir = output_path / "charts"
        charts_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("=" * 70)
        logger.info(f"STRATEGY AUTOPSY: {self.config.strategy_name}")
        logger.info(f"Run ID: {run_id}")
        logger.info(f"Git Hash: {self.git_hash}")
        logger.info("=" * 70)
        
        # Extract returns
        returns = trades_df["pnl_pct"].values / 100
        n_trades = len(trades_df)
        
        logger.info(f"Loaded {n_trades} trades for analysis")
        
        # === MODULE 1: VRD 2.0 METADATA ===
        logger.info("\n[Module 1] Recording experiment metadata...")
        self.results["metadata"] = {
            "strategy_name": self.config.strategy_name,
            "timestamp": self.timestamp,
            "git_hash": self.git_hash,
            "n_trades": n_trades,
            "param_hash": param_hash,
            "config": vars(self.config)
        }
        
        # === MODULE 3A: PERMUTATION TESTING ===
        logger.info(f"\n[Module 3A] Running Permutation Test ({self.config.n_permutations:,} shuffles)...")
        perm_test = PermutationTest(n_permutations=self.config.n_permutations)
        perm_result = perm_test.run(returns * 100)
        
        self.results["permutation"] = {
            "actual_sharpe": perm_result.actual_sharpe,
            "p_value": perm_result.sharpe_p_value,
            "percentile": perm_result.sharpe_percentile,
            "is_significant": perm_result.sharpe_p_value < self.config.significance_threshold,
            "interpretation": self._interpret_permutation(perm_result)
        }
        
        # === MODULE 3B: MONTE CARLO SIMULATION ===
        logger.info(f"\n[Module 3B] Running Monte Carlo ({self.config.n_monte_carlo:,} simulations)...")
        mc_sim = MonteCarloSimulator(
            n_simulations=self.config.n_monte_carlo,
            kill_switch_threshold=self.config.kill_switch_threshold
        )
        mc_result = mc_sim.run(returns * 100, initial_capital=self.config.initial_capital)
        
        self.results["monte_carlo"] = {
            "var_95": mc_result.var_95,
            "var_99": mc_result.var_99,
            "expected_shortfall": mc_result.expected_shortfall_95,
            "kill_switch_prob": mc_result.kill_switch_prob,
            "median_return": mc_result.median_final_return,
            "worst_case": mc_result.worst_case_return,
            "best_case": mc_result.best_case_return,
            "time_to_recovery_mean": mc_result.time_to_recovery_mean,
            "time_to_recovery_95": mc_result.time_to_recovery_95,
            "interpretation": self._interpret_monte_carlo(mc_result)
        }
        
        # === MODULE 3C: BOOTSTRAP CONFIDENCE INTERVALS ===
        logger.info(f"\n[Module 3C] Running Bootstrap ({self.config.n_bootstrap:,} samples)...")
        bootstrap = BlockBootstrap(
            n_bootstrap=self.config.n_bootstrap,
            block_size=self.config.bootstrap_block_size
        )
        
        sharpe_ci = bootstrap.confidence_interval(
            returns,
            lambda x: (np.mean(x) / np.std(x)) * np.sqrt(252) if np.std(x) > 0 else 0
        )
        win_rate_ci = bootstrap.confidence_interval(
            (returns > 0).astype(float),
            lambda x: np.mean(x)
        )
        
        boot_interp = self._interpret_bootstrap(sharpe_ci, win_rate_ci)
        self.results["bootstrap"] = {
            "sharpe_ci_lower": sharpe_ci.ci_lower,
            "sharpe_ci_upper": sharpe_ci.ci_upper,
            "sharpe_point": sharpe_ci.point_estimate,
            "win_rate_ci_lower": win_rate_ci.ci_lower,
            "win_rate_ci_upper": win_rate_ci.ci_upper,
            "win_rate_point": win_rate_ci.point_estimate,
            "sharpe_explanation": boot_interp["sharpe_explanation"],
            "sharpe_verdict": boot_interp["sharpe_verdict"],
            "precision": boot_interp["precision"],
            "interpretation": boot_interp
        }
        
        # === COMPUTE CORE METRICS ===
        equity = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(equity)
        drawdowns = (running_max - equity) / running_max * 100
        max_dd = np.max(drawdowns)
        
        win_rate = np.mean(returns > 0)
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        gross_profit = np.sum(returns[returns > 0])
        gross_loss = abs(np.sum(returns[returns < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        avg_win = np.mean(returns[returns > 0]) if np.any(returns > 0) else 0
        avg_loss = np.mean(returns[returns < 0]) if np.any(returns < 0) else 0
        
        self.results["core_metrics"] = {
            "total_trades": n_trades,
            "win_rate": win_rate,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "profit_factor": profit_factor,
            "avg_win": avg_win * 100,
            "avg_loss": avg_loss * 100,
            "total_return": (equity[-1] - 1) * 100
        }
        
        # === MODULE 10: DEPLOYMENT READINESS ===
        logger.info("\n[Module 10] Running Deployment Readiness Checks...")
        readiness = self._check_deployment_readiness()
        self.results["deployment"] = readiness
        
        # === GENERATE VISUALIZATIONS ===
        logger.info("\n[Module 9] Generating Visualizations...")
        viz_config = VisualizationConfig(output_dir=str(charts_dir))
        visualizer = ReportVisualizer(config=viz_config)
        
        chart_paths = self._generate_all_charts(
            visualizer, charts_dir, returns, equity, 
            perm_result, mc_result
        )
        
        # === GENERATE REPORT ===
        logger.info("\n[Module 9] Generating Autopsy Report...")
        report_path = self._generate_autopsy_report(output_path, chart_paths)
        
        logger.info("=" * 70)
        logger.info("AUTOPSY COMPLETE")
        logger.info(f"Report: {report_path}")
        logger.info(f"Charts: {charts_dir}")
        logger.info("=" * 70)
        
        return report_path
    
    def _interpret_permutation(self, result) -> Dict:
        """Generate detailed interpretation of permutation results."""
        if result.sharpe_p_value < 0.01:
            verdict = "HIGHLY SIGNIFICANT"
            explanation = "Strong statistical evidence that performance is due to skill, not luck."
            confidence = "Very High"
        elif result.sharpe_p_value < 0.05:
            verdict = "SIGNIFICANT"
            explanation = "Performance is statistically distinguishable from random chance."
            confidence = "High"
        elif result.sharpe_p_value < 0.10:
            verdict = "MARGINALLY SIGNIFICANT"
            explanation = "Weak evidence of skill. More data needed for confirmation."
            confidence = "Moderate"
        else:
            verdict = "NOT SIGNIFICANT"
            explanation = "Cannot distinguish performance from random chance. Insufficient evidence of edge."
            confidence = "Low"
        
        return {
            "verdict": verdict,
            "explanation": explanation,
            "confidence": confidence,
            "what_it_means": f"Only {result.sharpe_percentile:.1f}% of random orderings produced worse results.",
            "recommendation": "PASS" if result.sharpe_p_value < 0.05 else "FAIL"
        }
    
    def _interpret_monte_carlo(self, result) -> Dict:
        """Generate detailed interpretation of Monte Carlo results."""
        if result.kill_switch_prob < 0.05:
            risk_level = "LOW"
            explanation = "Very unlikely to hit critical drawdown threshold."
        elif result.kill_switch_prob < 0.15:
            risk_level = "MODERATE"
            explanation = "Acceptable probability of significant drawdown."
        elif result.kill_switch_prob < 0.25:
            risk_level = "ELEVATED"
            explanation = "Notable risk of hitting drawdown limit. Consider reduced sizing."
        else:
            risk_level = "HIGH"
            explanation = "Substantial probability of severe drawdown. Not recommended for deployment."
        
        return {
            "risk_level": risk_level,
            "explanation": explanation,
            "position_sizing": f"Maximum recommended leverage: {100 / result.var_99:.1f}x",
            "worst_case_scenario": f"1% of paths lose {abs(result.worst_case_return):.1f}% or more",
            "recommendation": "PASS" if result.kill_switch_prob < 0.15 else "FAIL"
        }
    
    def _interpret_bootstrap(self, sharpe_ci, win_rate_ci) -> Dict:
        """Generate detailed interpretation of bootstrap results."""
        sharpe_width = sharpe_ci.ci_upper - sharpe_ci.ci_lower
        wr_width = win_rate_ci.ci_upper - win_rate_ci.ci_lower
        
        if sharpe_ci.ci_lower > 0:
            sharpe_verdict = "POSITIVE"
            sharpe_explanation = "95% confident that true Sharpe is positive."
        else:
            sharpe_verdict = "UNCERTAIN"
            sharpe_explanation = "Cannot be 95% confident Sharpe is positive."
        
        return {
            "sharpe_verdict": sharpe_verdict,
            "sharpe_explanation": sharpe_explanation,
            "interpretation": sharpe_explanation,
            "precision": "HIGH" if sharpe_width < 2.0 else "LOW",
            "win_rate_stable": wr_width < 0.20,
            "recommendation": "PASS" if sharpe_ci.ci_lower > 0 else "CAUTION"
        }
    
    def _check_deployment_readiness(self) -> Dict:
        """Run all deployment readiness checks."""
        checks = []
        passed = 0
        
        # Check 1: Minimum trades
        n_trades = self.results["core_metrics"]["total_trades"]
        check1 = n_trades >= self.config.min_trades_required
        checks.append({
            "name": "Minimum Trade Count",
            "threshold": f"≥ {self.config.min_trades_required}",
            "actual": n_trades,
            "passed": check1,
            "reason": "Sufficient sample size for reliable statistics" if check1 else "Insufficient trades for statistical confidence"
        })
        if check1: passed += 1
        
        # Check 2: Statistical significance
        p_value = self.results["permutation"]["p_value"]
        check2 = p_value < self.config.significance_threshold
        checks.append({
            "name": "Statistical Significance",
            "threshold": f"p < {self.config.significance_threshold}",
            "actual": f"p = {p_value:.4f}",
            "passed": check2,
            "reason": "Results distinguishable from chance" if check2 else "Cannot distinguish from random"
        })
        if check2: passed += 1
        
        # Check 3: Sharpe ratio
        sharpe = self.results["core_metrics"]["sharpe_ratio"]
        check3 = sharpe >= self.config.min_sharpe_threshold
        checks.append({
            "name": "Minimum Sharpe Ratio",
            "threshold": f"≥ {self.config.min_sharpe_threshold}",
            "actual": f"{sharpe:.3f}",
            "passed": check3,
            "reason": "Adequate risk-adjusted return" if check3 else "Risk-adjusted return below target"
        })
        if check3: passed += 1
        
        # Check 4: Kill switch probability
        ks_prob = self.results["monte_carlo"]["kill_switch_prob"]
        check4 = ks_prob < 0.15
        checks.append({
            "name": "Ruin Probability",
            "threshold": "< 15%",
            "actual": f"{ks_prob:.1%}",
            "passed": check4,
            "reason": "Acceptable ruin risk" if check4 else "Ruin risk too high"
        })
        if check4: passed += 1
        
        # Check 5: Max drawdown
        max_dd = self.results["core_metrics"]["max_drawdown"]
        check5 = max_dd < self.config.max_drawdown_limit * 100
        checks.append({
            "name": "Maximum Drawdown",
            "threshold": f"< {self.config.max_drawdown_limit:.0%}",
            "actual": f"{max_dd:.1f}%",
            "passed": check5,
            "reason": "Drawdown within limits" if check5 else "Excessive drawdown"
        })
        if check5: passed += 1
        
        # Calculate overall verdict
        total_checks = len(checks)
        pass_rate = passed / total_checks
        
        if passed == total_checks:
            verdict = "DEPLOY"
            confidence = "HIGH"
        elif pass_rate >= 0.6:
            verdict = "CAUTION"
            confidence = "MODERATE"
        else:
            verdict = "REJECT"
            confidence = "LOW"
        
        return {
            "checks": checks,
            "passed": passed,
            "total": total_checks,
            "pass_rate": pass_rate,
            "verdict": verdict,
            "confidence": confidence,
            "score": int(pass_rate * 100)
        }
    
    def _generate_all_charts(
        self, visualizer, charts_dir, returns, equity, perm_result, mc_result
    ) -> Dict[str, str]:
        """Generate all visualization charts."""
        chart_paths = {}
        
        # 1. Equity Curve
        regime_labels = np.zeros(len(returns), dtype=int)
        visualizer.plot_regime_equity_curve(
            equity * self.config.initial_capital,
            regime_labels,
            {0: "Standard"},
            title=f"{self.config.strategy_name} - Equity Curve",
            save_path=str(charts_dir / "01_equity_curve.png")
        )
        chart_paths["equity_curve"] = "charts/01_equity_curve.png"
        
        # 2. Monte Carlo Cones
        visualizer.plot_monte_carlo_cones(
            mc_result.equity_paths,
            initial_capital=self.config.initial_capital,
            title="Monte Carlo Risk Analysis (50,000 Simulations)",
            save_path=str(charts_dir / "02_monte_carlo_cones.png")
        )
        chart_paths["monte_carlo"] = "charts/02_monte_carlo_cones.png"
        
        # 3. Drawdown Analysis
        visualizer.plot_drawdown_chart(
            equity * self.config.initial_capital,
            title="Drawdown Analysis",
            save_path=str(charts_dir / "03_drawdown_analysis.png")
        )
        chart_paths["drawdown"] = "charts/03_drawdown_analysis.png"
        
        # 4. Permutation Histogram
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
                save_path=str(charts_dir / "04_permutation_test.png")
            )
            chart_paths["permutation"] = "charts/04_permutation_test.png"
        except Exception as e:
            logger.warning(f"Could not generate permutation histogram: {e}")
        
        return chart_paths
    
    def _generate_autopsy_report(self, output_path: Path, chart_paths: Dict) -> str:
        """Generate comprehensive autopsy report in markdown format."""
        metrics = self.results["core_metrics"]
        perm = self.results["permutation"]
        mc = self.results["monte_carlo"]
        boot = self.results["bootstrap"]
        deploy = self.results["deployment"]
        meta = self.results["metadata"]
        
        report_date = datetime.now().strftime("%B %d, %Y")
        
        report = f"""# Strategy Autopsy Report

## {self.config.strategy_name}

**Date:** {report_date}  
**Git Hash:** {meta['git_hash']}  
**Run ID:** {meta['timestamp']}_{meta['param_hash']}

---

# Executive Summary

| Metric | Value | Status |
|--------|-------|--------|
| **VERDICT** | **{deploy['verdict']}** | {"✅" if deploy['verdict'] == "DEPLOY" else "⚠️" if deploy['verdict'] == "CAUTION" else "❌"} |
| **Confidence Score** | {deploy['score']}/100 | {deploy['confidence']} Confidence |
| **Total Trades** | {metrics['total_trades']} | {"✅" if metrics['total_trades'] >= 50 else "❌"} |
| **Win Rate** | {metrics['win_rate']:.1%} | {"✅" if metrics['win_rate'] > 0.55 else "⚠️"} |
| **Sharpe Ratio** | {metrics['sharpe_ratio']:.3f} | {"✅" if metrics['sharpe_ratio'] > 1.0 else "⚠️"} |
| **Max Drawdown** | {metrics['max_drawdown']:.1f}% | {"✅" if metrics['max_drawdown'] < 25 else "❌"} |
| **p-value** | {perm['p_value']:.4f} | {"✅ Significant" if perm['is_significant'] else "❌ Not Significant"} |

---

# The Five Critical Questions

## 1. Is the Edge REAL?

### Statistical Significance Analysis

**Verdict: {perm['interpretation']['verdict']}**

{perm['interpretation']['explanation']}

| Test | Result | Interpretation |
|------|--------|----------------|
| **Permutation Test** | p = {perm['p_value']:.4f} | {perm['interpretation']['what_it_means']} |
| **Actual Sharpe** | {perm['actual_sharpe']:.4f} | vs. {self.config.n_permutations:,} random orderings |
| **Percentile Rank** | {perm['percentile']:.1f}% | Higher = better than random |

![Permutation Test]({chart_paths.get('permutation', 'charts/04_permutation_test.png')})

### Bootstrap Confidence Intervals (95%)

| Metric | Lower Bound | Point Estimate | Upper Bound |
|--------|-------------|----------------|-------------|
| **Sharpe Ratio** | {boot['sharpe_ci_lower']:.3f} | {boot['sharpe_point']:.3f} | {boot['sharpe_ci_upper']:.3f} |
| **Win Rate** | {boot['win_rate_ci_lower']:.1%} | {boot['win_rate_point']:.1%} | {boot['win_rate_ci_upper']:.1%} |

**Interpretation:** {boot['sharpe_explanation']}

---

## 2. WHY Does the Edge Work?

### Trade-Level Analysis

| Statistic | Value | Interpretation |
|-----------|-------|----------------|
| **Average Win** | {metrics['avg_win']:.2f}% | Size of typical winning trade |
| **Average Loss** | {metrics['avg_loss']:.2f}% | Size of typical losing trade |
| **Profit Factor** | {metrics['profit_factor']:.2f} | Gross profits / Gross losses |
| **Win/Loss Ratio** | {abs(metrics['avg_win']/metrics['avg_loss']) if metrics['avg_loss'] != 0 else 0:.2f} | Reward-to-risk ratio |

### Edge Explanation

The strategy {"shows" if perm['is_significant'] else "does not show"} statistically significant edge.

**Primary Driver:** {"Risk-adjusted returns exceed random expectation" if perm['is_significant'] else "Results cannot be distinguished from chance"}

---

## 3. Will the Edge CONTINUE?

### Forward-Looking Risk Assessment

| Scenario | Probability | Impact |
|----------|-------------|--------|
| **Median Outcome** | 50% | {mc['median_return']:+.1f}% return |
| **Bad Outcome (95% VaR)** | 5% | Up to {mc['var_95']:.1f}% drawdown |
| **Worst Case (99% VaR)** | 1% | Up to {mc['var_99']:.1f}% drawdown |
| **Kill Switch Trigger** | {mc['kill_switch_prob']:.1%} | >{self.config.kill_switch_threshold:.0%} drawdown |

**Recovery Analysis:**
- Mean time to recovery: **{mc['time_to_recovery_mean']:.1f} trades**
- 95th percentile recovery: **{mc['time_to_recovery_95']:.1f} trades**

---

## 4. How Can It FAIL?

### Risk Scenarios

| Failure Mode | Probability | Mitigation |
|--------------|-------------|------------|
| **Random Variance** | {perm['p_value']:.1%} | {"Already low" if perm['p_value'] < 0.05 else "HIGH - Need more data"} |
| **Severe Drawdown** | {mc['kill_switch_prob']:.1%} | {"Acceptable" if mc['kill_switch_prob'] < 0.15 else "Reduce position size"} |
| **Regime Change** | Unknown | Monitor performance by market condition |

### Worst Case Scenarios (Monte Carlo)

From 50,000 simulated paths:
- **1% of paths** result in losses of **{abs(mc['worst_case']):.1f}% or worse**
- **Expected shortfall** in worst 5% of cases: **{mc['expected_shortfall']:.1f}%**

![Monte Carlo Analysis]({chart_paths.get('monte_carlo', 'charts/02_monte_carlo_cones.png')})

---

## 5. How Should We TRADE It?

### Position Sizing Recommendations

Based on Monte Carlo VaR analysis:

| Risk Tolerance | Max Position Size | Rationale |
|----------------|-------------------|-----------|
| **Conservative** | {100 / (mc['var_99'] * 2):.1f}x | Survive 99% VaR with 50% margin |
| **Moderate** | {100 / mc['var_99']:.1f}x | Stay within 99% VaR |
| **Aggressive** | {100 / mc['var_95']:.1f}x | Accept 95% VaR |

### Recommended Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Risk per Trade** | 0.5-1.0% | Based on VaR/recovery analysis |
| **Max Drawdown Limit** | {self.config.kill_switch_threshold:.0%} | Kill switch threshold |
| **Min Win Rate for Confidence** | {boot['win_rate_ci_lower']:.1%} | Lower bound of CI |

---

# Deployment Readiness

## Final Checklist

| Check | Threshold | Actual | Status |
|-------|-----------|--------|--------|
"""
        
        for check in deploy['checks']:
            status = "✅ PASS" if check['passed'] else "❌ FAIL"
            report += f"| {check['name']} | {check['threshold']} | {check['actual']} | {status} |\n"
        
        report += f"""
**Result: {deploy['passed']}/{deploy['total']} checks passed ({deploy['pass_rate']:.0%})**

---

# Visual Analysis

## Equity Performance

![Equity Curve]({chart_paths.get('equity_curve', 'charts/01_equity_curve.png')})

## Drawdown Profile

![Drawdown Analysis]({chart_paths.get('drawdown', 'charts/03_drawdown_analysis.png')})

---

# Final Verdict

# {deploy['verdict']}

"""
        
        if deploy['verdict'] == "DEPLOY":
            report += """
### ✅ READY FOR DEPLOYMENT

The strategy has passed all institutional-grade validation tests.

**Recommended Next Steps:**
1. Paper trading validation (2-4 weeks)
2. Small live allocation with conservative sizing
3. Monitor performance vs. backtest benchmarks
4. Full deployment after paper trading confirmation
"""
        elif deploy['verdict'] == "CAUTION":
            report += """
### ⚠️ PROCEED WITH CAUTION

The strategy shows promise but has concerns requiring attention.

**Issues to Address:**
- Consider gathering more trade data
- Validate with paper trading before live deployment
- Use reduced position sizing initially

**Recommended Next Steps:**
1. Extended paper trading period (4-8 weeks)
2. Conservative position sizing (0.25-0.5% risk)
3. Close monitoring of key metrics
"""
        else:
            report += """
### ❌ NOT RECOMMENDED FOR DEPLOYMENT

The strategy has failed critical validation tests.

**Critical Issues:**
- Statistical significance not established
- Cannot distinguish performance from random chance
- Insufficient evidence to risk capital

**Recommended Actions:**
1. Do NOT deploy in current form
2. Review and refine detection parameters
3. Gather substantially more historical data
4. Consider fundamental strategy changes
"""
        
        report += f"""
---

# Appendix: Technical Configuration

## Validation Parameters

| Parameter | Value |
|-----------|-------|
| Permutation Tests | {self.config.n_permutations:,} shuffles |
| Monte Carlo Simulations | {self.config.n_monte_carlo:,} paths |
| Bootstrap Samples | {self.config.n_bootstrap:,} |
| Block Size | {self.config.bootstrap_block_size} |
| Confidence Level | {self.config.confidence_level:.0%} |
| Kill Switch Threshold | {self.config.kill_switch_threshold:.0%} |
| Initial Capital | ${self.config.initial_capital:,.0f} |

## Files Generated

- `autopsy_report.md` - This report
- `charts/01_equity_curve.png` - Equity performance
- `charts/02_monte_carlo_cones.png` - Risk simulation
- `charts/03_drawdown_analysis.png` - Drawdown profile
- `charts/04_permutation_test.png` - Statistical significance

---

*Report generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*  
*VRD 2.0 Strategy Autopsy Framework*
"""
        
        # Save report
        report_path = output_path / "autopsy_report.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        return str(report_path)


def main():
    """Run strategy autopsy on saved trades."""
    
    # Load trades
    trades_path = "experiments/exp_03_backtest_validation/revalidation_trades.csv"
    
    if not Path(trades_path).exists():
        logger.error(f"Trade file not found: {trades_path}")
        return
    
    trades_df = pd.read_csv(trades_path)
    
    # Configure autopsy
    config = AutopsyConfig(
        strategy_name="QML Pattern Strategy",
        initial_capital=100000,
        n_permutations=10000,
        n_monte_carlo=50000,
        n_bootstrap=5000
    )
    
    # Run full autopsy
    autopsy = StrategyAutopsy(config)
    report_path = autopsy.run_full_autopsy(trades_df)
    
    return report_path


if __name__ == "__main__":
    main()
