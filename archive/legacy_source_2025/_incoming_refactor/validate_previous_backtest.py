#!/usr/bin/env python3
"""
Validate Previous Backtest Results
===================================
Runs the validation framework on the 67.4% win rate backtest
from experiments/exp_03_backtest_validation/revalidation_trades.csv
"""

import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.validation.permutation import PermutationTest
from src.validation.monte_carlo import MonteCarloSimulator
from src.validation.bootstrap import BlockBootstrap
from src.reporting.dossier import DossierGenerator
from src.deployment.gatekeeper import DeploymentGatekeeper


def main():
    """Run validation on the previous 67.4% win rate backtest."""
    
    logger.info("=" * 70)
    logger.info("VALIDATING PREVIOUS BACKTEST (67.4% WIN RATE)")
    logger.info("=" * 70)
    
    # Load the trades
    trades_path = "experiments/exp_03_backtest_validation/revalidation_trades.csv"
    
    if not Path(trades_path).exists():
        logger.error(f"Trade file not found: {trades_path}")
        return None
    
    trades_df = pd.read_csv(trades_path)
    logger.info(f"Loaded {len(trades_df)} trades from {trades_path}")
    
    # Extract returns
    returns = trades_df["pnl_pct"].values / 100  # Convert to decimal
    outcomes = trades_df["outcome"].values
    
    # Calculate metrics
    total_trades = len(trades_df)
    wins = np.sum(outcomes == 1)
    win_rate = wins / total_trades
    
    # Sharpe ratio (daily equivalent)
    mean_ret = np.mean(returns)
    std_ret = np.std(returns) if np.std(returns) > 0 else 0.01
    sharpe_ratio = (mean_ret / std_ret) * np.sqrt(252)
    
    # Max drawdown
    equity = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(equity)
    drawdowns = (running_max - equity) / running_max * 100
    max_dd = np.max(drawdowns)
    
    # Profit factor
    gross_profit = np.sum(returns[returns > 0])
    gross_loss = abs(np.sum(returns[returns < 0]))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else gross_profit
    
    logger.info("")
    logger.info("BACKTEST METRICS:")
    logger.info(f"  Total Trades: {total_trades}")
    logger.info(f"  Win Rate: {win_rate:.1%}")
    logger.info(f"  Sharpe Ratio: {sharpe_ratio:.3f}")
    logger.info(f"  Max Drawdown: {max_dd:.2f}%")
    logger.info(f"  Profit Factor: {profit_factor:.2f}")
    logger.info("")
    
    # Run Permutation Test
    logger.info("Running Permutation Test (10,000 shuffles)...")
    perm_test = PermutationTest(n_permutations=10000)
    perm_result = perm_test.run(returns)
    
    logger.info(f"  Actual Sharpe: {perm_result.actual_sharpe:.3f}")
    logger.info(f"  p-value: {perm_result.sharpe_p_value:.4f}")
    logger.info(f"  Percentile: {perm_result.sharpe_percentile:.1f}%")
    
    # Run Monte Carlo
    logger.info("")
    logger.info("Running Monte Carlo Simulation (50,000 paths)...")
    mc_sim = MonteCarloSimulator(n_simulations=50000, kill_switch_threshold=0.20)
    mc_result = mc_sim.run(returns, initial_capital=100000)
    
    logger.info(f"  Median Final Return: {mc_result.median_final_return:+.1f}%")
    logger.info(f"  VaR 95%: {mc_result.var_95:.2f}%")
    logger.info(f"  VaR 99%: {mc_result.var_99:.2f}%")
    logger.info(f"  Kill Switch Prob: {mc_result.kill_switch_prob:.1%}")
    
    # Profit probability (% of sims with positive return)
    probability_profitable = float(np.mean(mc_result.final_returns > 0))
    logger.info(f"  Profit Probability: {probability_profitable:.1%}")
    
    logger.info("")
    logger.info("Running Block Bootstrap (5,000 samples)...")
    bootstrap = BlockBootstrap(n_bootstrap=5000, block_size=5)
    
    sharpe_result = bootstrap.confidence_interval(
        returns, 
        lambda x: (np.mean(x) / np.std(x)) * np.sqrt(252) if np.std(x) > 0 else 0
    )
    win_rate_result = bootstrap.confidence_interval(
        outcomes.astype(float),
        lambda x: np.mean(x)
    )
    
    logger.info(f"  Sharpe 95% CI: [{sharpe_result.ci_lower:.3f}, {sharpe_result.ci_upper:.3f}]")
    logger.info(f"  Win Rate 95% CI: [{win_rate_result.ci_lower:.1%}, {win_rate_result.ci_upper:.1%}]")
    
    # Determine verdict
    logger.info("")
    logger.info("=" * 70)
    
    # Verdict logic
    is_significant = perm_result.sharpe_p_value < 0.05
    sharpe_good = sharpe_ratio > 1.0
    low_ruin_risk = mc_result.kill_switch_prob < 0.15
    
    confidence_score = 0
    reasons = []
    
    if is_significant:
        confidence_score += 30
        reasons.append("✅ Statistically significant (p < 0.05)")
    else:
        reasons.append("❌ Not statistically significant")
    
    if sharpe_good:
        confidence_score += 25
        reasons.append("✅ Strong Sharpe ratio (> 1.0)")
    elif sharpe_ratio > 0.5:
        confidence_score += 15
        reasons.append("⚠️ Moderate Sharpe ratio (0.5 - 1.0)")
    else:
        reasons.append("❌ Poor Sharpe ratio")
    
    if low_ruin_risk:
        confidence_score += 25
        reasons.append("✅ Low ruin risk (< 15%)")
    elif mc_result.kill_switch_prob < 0.25:
        confidence_score += 15
        reasons.append("⚠️ Moderate ruin risk (15-25%)")
    else:
        reasons.append("❌ High ruin risk")
    
    if win_rate > 0.60:
        confidence_score += 20
        reasons.append("✅ Strong win rate (> 60%)")
    elif win_rate > 0.50:
        confidence_score += 10
        reasons.append("⚠️ Moderate win rate (50-60%)")
    else:
        reasons.append("❌ Poor win rate")
    
    if confidence_score >= 70:
        verdict = "DEPLOY"
    elif confidence_score >= 50:
        verdict = "CAUTION"
    else:
        verdict = "REJECT"
    
    logger.info(f"VERDICT: {verdict}")
    logger.info(f"CONFIDENCE: {confidence_score}/100")
    logger.info("=" * 70)
    
    for r in reasons:
        logger.info(f"  {r}")
    
    logger.info("=" * 70)
    
    # Run Gatekeeper
    logger.info("")
    logger.info("Running Deployment Gatekeeper...")
    
    # Create result object
    class MockResult:
        pass
    
    result = MockResult()
    result.experiment_id = "previous_backtest_validation"
    result.strategy_name = "QML_BULLISH_V1 (Historical)"
    result.oos_sharpe = sharpe_ratio
    result.oos_max_dd = max_dd
    result.total_trades = total_trades
    result.sharpe_p_value = perm_result.sharpe_p_value
    result.monte_carlo_result = mc_result
    result.walk_forward_result = None
    result.regime_performance = {}
    result.overall_verdict = verdict
    result.confidence_score = confidence_score
    result.sharpe_ci = (sharpe_result.ci_lower, sharpe_result.ci_upper)
    result.win_rate_ci = (win_rate_result.ci_lower, win_rate_result.ci_upper)
    result.timestamp = datetime.now().isoformat()
    result.permutation_result = perm_result
    
    gatekeeper = DeploymentGatekeeper()
    readiness = gatekeeper.check_readiness(result)
    
    logger.info(f"Gatekeeper Result: {readiness}")
    
    # Generate dossier
    output_dir = "results/previous_backtest_validation"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    dossier = DossierGenerator()
    dossier_path = dossier.generate(result, output_dir=output_dir)
    
    logger.info("")
    logger.info(f"✅ Dossier saved to: {dossier_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Strategy: QML Bullish Pattern (Historical Backtest)")
    print(f"Data Period: 2023-01 to 2024-06")
    print(f"Total Trades: {total_trades}")
    print(f"")
    print(f"VERDICT: {verdict} | Confidence: {confidence_score}/100")
    print(f"")
    print(f"Key Metrics:")
    print(f"  Win Rate: {win_rate:.1%}")
    print(f"  Sharpe: {sharpe_ratio:.3f}")
    print(f"  Max DD: {max_dd:.2f}%")
    print(f"  p-value: {perm_result.sharpe_p_value:.4f}")
    print(f"  Kill Switch Prob: {mc_result.kill_switch_prob:.1%}")
    print("=" * 70)
    
    return result


if __name__ == "__main__":
    result = main()
