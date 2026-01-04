"""
Strategy Validator - High-Level Orchestrator
============================================
Combines all validation modules into a unified pipeline
for comprehensive strategy analysis.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.validation.database import VRDDatabase, ExperimentRecord
from src.validation.tracker import ExperimentTracker
from src.validation.walk_forward import (
    PurgedWalkForwardEngine,
    WalkForwardConfig,
    WalkForwardResult,
)
from src.validation.permutation import PermutationTest, PermutationResult
from src.validation.monte_carlo import MonteCarloSimulator, MonteCarloResult
from src.validation.bootstrap import BlockBootstrap, BootstrapResult


@dataclass
class ValidationConfig:
    """Configuration for full validation pipeline."""
    
    # Output directory
    output_dir: str = "validation_results"
    
    # Walk-forward settings
    n_folds: int = 10
    purge_bars: int = 5
    embargo_bars: int = 5
    train_ratio: float = 0.7
    
    # Statistical testing
    n_permutations: int = 10000
    n_monte_carlo: int = 50000
    n_bootstrap: int = 5000
    bootstrap_block_size: int = 5
    
    # Risk thresholds
    kill_switch_threshold: float = 0.20
    min_trades_for_validity: int = 30
    
    # Significance thresholds
    significance_level: float = 0.05
    
    # Random seed
    random_seed: Optional[int] = 42


@dataclass
class ValidationReport:
    """Complete validation report for a strategy."""
    
    # Experiment metadata
    experiment_id: str
    strategy_name: str
    timestamp: str
    
    # Walk-forward results
    walk_forward: Optional[WalkForwardResult] = None
    
    # Statistical results
    permutation: Optional[PermutationResult] = None
    monte_carlo: Optional[MonteCarloResult] = None
    bootstrap_results: Dict[str, BootstrapResult] = field(default_factory=dict)
    
    # Overall verdict
    confidence_score: float = 0.0        # 0-100
    overall_verdict: str = "UNKNOWN"     # DEPLOY, CAUTION, REJECT
    verdict_reasons: List[str] = field(default_factory=list)
    
    # Aggregated metrics
    oos_sharpe: float = 0.0
    oos_max_dd: float = 0.0
    oos_win_rate: float = 0.0
    total_trades: int = 0
    
    # Statistical significance
    is_statistically_significant: bool = False
    sharpe_p_value: float = 1.0


class StrategyValidator:
    """
    High-level orchestrator for strategy validation.
    
    Combines:
    - Purged Walk-Forward Optimization
    - Permutation Testing
    - Monte Carlo Simulation
    - Bootstrap Confidence Intervals
    
    Into a unified validation pipeline.
    """
    
    def __init__(
        self,
        config: Optional[ValidationConfig] = None,
        output_dir: Optional[str] = None
    ):
        """
        Initialize strategy validator.
        
        Args:
            config: Validation configuration
            output_dir: Override output directory
        """
        self.config = config or ValidationConfig()
        if output_dir:
            self.config.output_dir = output_dir
        
        # Create output directory
        self.output_path = Path(self.config.output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.db = VRDDatabase(str(self.output_path / "vrd.db"))
        self.tracker = ExperimentTracker(
            base_dir=str(self.output_path),
            db=self.db
        )
        
        # Walk-forward engine
        wf_config = WalkForwardConfig(
            n_folds=self.config.n_folds,
            purge_bars=self.config.purge_bars,
            embargo_bars=self.config.embargo_bars,
            train_ratio=self.config.train_ratio,
        )
        self.walk_forward_engine = PurgedWalkForwardEngine(
            config=wf_config,
            tracker=self.tracker
        )
        
        # Statistical tests
        self.permutation_test = PermutationTest(
            n_permutations=self.config.n_permutations,
            random_seed=self.config.random_seed
        )
        self.monte_carlo = MonteCarloSimulator(
            n_simulations=self.config.n_monte_carlo,
            kill_switch_threshold=self.config.kill_switch_threshold,
            random_seed=self.config.random_seed
        )
        self.bootstrap = BlockBootstrap(
            n_bootstrap=self.config.n_bootstrap,
            block_size=self.config.bootstrap_block_size,
            random_seed=self.config.random_seed
        )
        
        logger.info(f"StrategyValidator initialized at {self.output_path}")
    
    def run_full_validation(
        self,
        strategy_name: str,
        df: pd.DataFrame,
        objective_fn: Callable[[pd.DataFrame, Dict], Dict[str, float]],
        param_grid: Dict[str, List],
        optimization_metric: str = "sharpe_ratio"
    ) -> ValidationReport:
        """
        Run complete validation pipeline.
        
        Args:
            strategy_name: Name of strategy
            df: Full dataset (OHLCV)
            objective_fn: Backtest function taking (df, params) -> metrics dict
            param_grid: Parameter grid for optimization
            optimization_metric: Metric to optimize
            
        Returns:
            Complete ValidationReport
        """
        logger.info(f"Starting full validation for {strategy_name}")
        
        # Get data range
        if "time" in df.columns:
            data_start = str(df["time"].min())
            data_end = str(df["time"].max())
        else:
            data_start = str(df.index.min())
            data_end = str(df.index.max())
        
        # Start experiment tracking
        experiment_id = self.tracker.start_experiment(
            strategy_name=strategy_name,
            params={"param_grid": param_grid, "optimization_metric": optimization_metric},
            data_range=(data_start, data_end),
            seed=self.config.random_seed or 42,
            fold_count=self.config.n_folds
        )
        
        try:
            # 1. Walk-forward optimization
            logger.info("Phase 1: Walk-Forward Optimization")
            wf_result = self.walk_forward_engine.run(
                df=df,
                objective_fn=objective_fn,
                param_grid=param_grid,
                optimization_metric=optimization_metric
            )
            
            # Collect all OOS trades for statistical analysis
            all_oos_returns = self._collect_oos_returns(wf_result)
            
            if len(all_oos_returns) < self.config.min_trades_for_validity:
                logger.warning(
                    f"Insufficient OOS trades ({len(all_oos_returns)}) for "
                    f"statistical analysis (need {self.config.min_trades_for_validity})"
                )
            
            # 2. Permutation Testing
            logger.info("Phase 2: Permutation Testing")
            if len(all_oos_returns) >= 10:
                perm_result = self.permutation_test.run(all_oos_returns)
            else:
                perm_result = None
                logger.warning("Skipping permutation test: insufficient trades")
            
            # 3. Monte Carlo Simulation
            logger.info("Phase 3: Monte Carlo Simulation")
            if len(all_oos_returns) >= 10:
                mc_result = self.monte_carlo.run(all_oos_returns)
            else:
                mc_result = None
                logger.warning("Skipping Monte Carlo: insufficient trades")
            
            # 4. Bootstrap Confidence Intervals
            logger.info("Phase 4: Bootstrap Confidence Intervals")
            if len(all_oos_returns) >= 10:
                trades_df = pd.DataFrame({"pnl_pct": all_oos_returns})
                boot_results = self.bootstrap.all_metrics_ci(trades_df)
            else:
                boot_results = {}
                logger.warning("Skipping bootstrap: insufficient trades")
            
            # 5. Generate report
            logger.info("Phase 5: Generating Report")
            report = self._generate_report(
                experiment_id=experiment_id,
                strategy_name=strategy_name,
                wf_result=wf_result,
                perm_result=perm_result,
                mc_result=mc_result,
                boot_results=boot_results,
            )
            
            # Finalize experiment
            metrics = {
                "sharpe_ratio": report.oos_sharpe,
                "max_drawdown_pct": report.oos_max_dd,
                "win_rate": report.oos_win_rate,
                "total_trades": report.total_trades,
                "profit_factor": wf_result.aggregate_profit_factor if wf_result else 0,
                "total_return_pct": wf_result.aggregate_return if wf_result else 0,
            }
            
            statistical_results = {
                "sharpe_p_value": report.sharpe_p_value,
                "sharpe_percentile": perm_result.sharpe_percentile if perm_result else None,
                "var_95": mc_result.var_95 if mc_result else None,
                "var_99": mc_result.var_99 if mc_result else None,
                "kill_switch_prob": mc_result.kill_switch_prob if mc_result else None,
            }
            
            self.tracker.finalize(
                metrics=metrics,
                statistical_results=statistical_results
            )
            
            # Save report
            self._save_report(report)
            
            return report
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            self.tracker.fail(str(e))
            raise
    
    def run_statistical_validation(
        self,
        trade_returns: np.ndarray,
        strategy_name: str = "Strategy"
    ) -> ValidationReport:
        """
        Run statistical validation only (skip walk-forward).
        
        Useful when you already have trade returns from a backtest.
        
        Args:
            trade_returns: Array of trade returns (percentages)
            strategy_name: Name for labeling
            
        Returns:
            ValidationReport with statistical results
        """
        logger.info(f"Running statistical validation on {len(trade_returns)} trades")
        
        trade_returns = np.asarray(trade_returns).flatten()
        
        # Permutation test
        perm_result = self.permutation_test.run(trade_returns)
        
        # Monte Carlo
        mc_result = self.monte_carlo.run(trade_returns)
        
        # Bootstrap
        trades_df = pd.DataFrame({"pnl_pct": trade_returns})
        boot_results = self.bootstrap.all_metrics_ci(trades_df)
        
        # Generate report
        report = self._generate_report(
            experiment_id="statistical",
            strategy_name=strategy_name,
            wf_result=None,
            perm_result=perm_result,
            mc_result=mc_result,
            boot_results=boot_results,
        )
        
        return report
    
    def _collect_oos_returns(self, wf_result: WalkForwardResult) -> np.ndarray:
        """Collect all OOS trade returns from walk-forward folds."""
        all_returns = []
        
        for fold in wf_result.fold_results:
            # Assuming metrics include trade-level returns
            # For now, we'll simulate based on summary stats
            oos_metrics = fold.out_of_sample_metrics
            trades = fold.out_of_sample_trades
            
            if trades > 0 and "avg_trade_pnl" in oos_metrics:
                # Generate pseudo-returns based on win rate and avg returns
                win_rate = oos_metrics.get("win_rate", 0.5)
                avg_win = oos_metrics.get("avg_win_pct", 2.0)
                avg_loss = oos_metrics.get("avg_loss_pct", -1.0)
                
                n_wins = int(trades * win_rate)
                n_losses = trades - n_wins
                
                returns = (
                    [avg_win] * n_wins +
                    [avg_loss] * n_losses
                )
                all_returns.extend(returns)
        
        # If no granular returns, use total return divided by trades
        if not all_returns and wf_result.total_oos_trades > 0:
            avg_return = wf_result.aggregate_return / wf_result.total_oos_trades
            all_returns = [avg_return] * wf_result.total_oos_trades
        
        return np.array(all_returns)
    
    def _generate_report(
        self,
        experiment_id: str,
        strategy_name: str,
        wf_result: Optional[WalkForwardResult],
        perm_result: Optional[PermutationResult],
        mc_result: Optional[MonteCarloResult],
        boot_results: Dict[str, BootstrapResult],
    ) -> ValidationReport:
        """Generate comprehensive validation report."""
        
        report = ValidationReport(
            experiment_id=experiment_id,
            strategy_name=strategy_name,
            timestamp=datetime.now().isoformat(),
            walk_forward=wf_result,
            permutation=perm_result,
            monte_carlo=mc_result,
            bootstrap_results=boot_results,
        )
        
        # Extract metrics
        if wf_result:
            report.oos_sharpe = wf_result.aggregate_sharpe
            report.oos_max_dd = wf_result.aggregate_max_dd
            report.oos_win_rate = wf_result.aggregate_win_rate
            report.total_trades = wf_result.total_oos_trades
        
        if perm_result:
            report.sharpe_p_value = perm_result.sharpe_p_value
            report.is_statistically_significant = perm_result.sharpe_p_value < self.config.significance_level
        
        # Calculate confidence score (0-100)
        score_components = []
        reasons = []
        
        # Component 1: Statistical significance (30%)
        if perm_result:
            if perm_result.sharpe_p_value < 0.01:
                sig_score = 30
                reasons.append("Strong statistical significance (p < 0.01)")
            elif perm_result.sharpe_p_value < 0.05:
                sig_score = 20
                reasons.append("Statistical significance (p < 0.05)")
            elif perm_result.sharpe_p_value < 0.10:
                sig_score = 10
                reasons.append("Marginal significance (p < 0.10)")
            else:
                sig_score = 0
                reasons.append("NOT statistically significant (p >= 0.10)")
            score_components.append(sig_score)
        
        # Component 2: Sharpe quality (25%)
        if wf_result:
            if wf_result.aggregate_sharpe > 1.5:
                sharpe_score = 25
                reasons.append(f"Excellent Sharpe ({wf_result.aggregate_sharpe:.2f})")
            elif wf_result.aggregate_sharpe > 1.0:
                sharpe_score = 18
                reasons.append(f"Good Sharpe ({wf_result.aggregate_sharpe:.2f})")
            elif wf_result.aggregate_sharpe > 0.5:
                sharpe_score = 10
                reasons.append(f"Moderate Sharpe ({wf_result.aggregate_sharpe:.2f})")
            else:
                sharpe_score = 0
                reasons.append(f"Poor Sharpe ({wf_result.aggregate_sharpe:.2f})")
            score_components.append(sharpe_score)
        
        # Component 3: Risk (VaR, kill switch) (25%)
        if mc_result:
            if mc_result.kill_switch_prob < 0.05:
                risk_score = 25
                reasons.append(f"Low ruin probability ({mc_result.kill_switch_prob:.1%})")
            elif mc_result.kill_switch_prob < 0.15:
                risk_score = 15
                reasons.append(f"Moderate ruin probability ({mc_result.kill_switch_prob:.1%})")
            else:
                risk_score = 5
                reasons.append(f"High ruin probability ({mc_result.kill_switch_prob:.1%})")
            score_components.append(risk_score)
        
        # Component 4: Parameter stability (20%)
        if wf_result:
            if wf_result.param_stability_score > 0.8:
                stability_score = 20
                reasons.append("Highly stable parameters")
            elif wf_result.param_stability_score > 0.6:
                stability_score = 12
                reasons.append("Moderately stable parameters")
            else:
                stability_score = 5
                reasons.append("Unstable parameters (possible overfit)")
            score_components.append(stability_score)
        
        # Calculate final score
        if score_components:
            report.confidence_score = sum(score_components)
        
        # Determine verdict
        if report.confidence_score >= 70 and report.is_statistically_significant:
            report.overall_verdict = "DEPLOY"
        elif report.confidence_score >= 50:
            report.overall_verdict = "CAUTION"
        else:
            report.overall_verdict = "REJECT"
        
        report.verdict_reasons = reasons
        
        return report
    
    def _save_report(self, report: ValidationReport) -> Path:
        """Save report to disk."""
        report_dir = self.tracker.current_experiment_dir or self.output_path
        
        # Save JSON summary
        import json
        summary = {
            "experiment_id": report.experiment_id,
            "strategy_name": report.strategy_name,
            "timestamp": report.timestamp,
            "confidence_score": report.confidence_score,
            "overall_verdict": report.overall_verdict,
            "verdict_reasons": report.verdict_reasons,
            "oos_sharpe": report.oos_sharpe,
            "oos_max_dd": report.oos_max_dd,
            "oos_win_rate": report.oos_win_rate,
            "total_trades": report.total_trades,
            "sharpe_p_value": report.sharpe_p_value,
            "is_statistically_significant": report.is_statistically_significant,
        }
        
        # Add Monte Carlo stats
        if report.monte_carlo:
            summary["monte_carlo"] = {
                "var_95": report.monte_carlo.var_95,
                "var_99": report.monte_carlo.var_99,
                "kill_switch_prob": report.monte_carlo.kill_switch_prob,
            }
        
        report_path = report_dir / "validation_report.json"
        with open(report_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Report saved to {report_path}")
        
        return report_path
    
    def generate_text_report(self, report: ValidationReport) -> str:
        """Generate human-readable text report."""
        lines = [
            "=" * 70,
            "STRATEGY VALIDATION REPORT",
            "=" * 70,
            "",
            f"Strategy: {report.strategy_name}",
            f"Experiment ID: {report.experiment_id}",
            f"Timestamp: {report.timestamp}",
            "",
            "=" * 70,
            f"OVERALL VERDICT: {report.overall_verdict}",
            f"CONFIDENCE SCORE: {report.confidence_score}/100",
            "=" * 70,
            "",
            "Verdict Reasons:",
        ]
        
        for reason in report.verdict_reasons:
            lines.append(f"  • {reason}")
        
        lines.extend([
            "",
            "-" * 70,
            "OUT-OF-SAMPLE PERFORMANCE",
            "-" * 70,
            f"  Sharpe Ratio: {report.oos_sharpe:.3f}",
            f"  Max Drawdown: {report.oos_max_dd:.2f}%",
            f"  Win Rate: {report.oos_win_rate:.1%}",
            f"  Total Trades: {report.total_trades}",
        ])
        
        if report.permutation:
            lines.extend([
                "",
                "-" * 70,
                "STATISTICAL SIGNIFICANCE (Permutation Test)",
                "-" * 70,
                f"  Sharpe p-value: {report.permutation.sharpe_p_value:.4f}",
                f"  Percentile: {report.permutation.sharpe_percentile:.1f}th",
                f"  Significant at 5%: {'YES' if report.is_statistically_significant else 'NO'}",
            ])
        
        if report.monte_carlo:
            lines.extend([
                "",
                "-" * 70,
                "RISK ANALYSIS (Monte Carlo)",
                "-" * 70,
                f"  VaR 95%: {report.monte_carlo.var_95:.2f}%",
                f"  VaR 99%: {report.monte_carlo.var_99:.2f}%",
                f"  Expected Shortfall (CVaR 95%): {report.monte_carlo.expected_shortfall_95:.2f}%",
                f"  Kill Switch Probability: {report.monte_carlo.kill_switch_prob:.2%}",
            ])
        
        if report.walk_forward:
            lines.extend([
                "",
                "-" * 70,
                "WALK-FORWARD ANALYSIS",
                "-" * 70,
                f"  Folds: {len(report.walk_forward.fold_results)}",
                f"  Parameter Stability: {report.walk_forward.param_stability_score:.3f}",
                f"  IS/OOS Ratio: {report.walk_forward.is_to_oos_ratio:.3f}",
                f"  Sharpe Std (across folds): {report.walk_forward.sharpe_stability:.3f}",
            ])
        
        lines.append("=" * 70)
        
        return "\n".join(lines)


def create_validator(
    output_dir: str = "validation_results",
    n_folds: int = 10,
    n_permutations: int = 10000,
    n_monte_carlo: int = 50000
) -> StrategyValidator:
    """
    Factory function for StrategyValidator.
    
    Args:
        output_dir: Output directory for results
        n_folds: Number of walk-forward folds
        n_permutations: Number of permutation tests
        n_monte_carlo: Number of Monte Carlo simulations
        
    Returns:
        StrategyValidator instance
    """
    config = ValidationConfig(
        output_dir=output_dir,
        n_folds=n_folds,
        n_permutations=n_permutations,
        n_monte_carlo=n_monte_carlo,
    )
    return StrategyValidator(config=config)
