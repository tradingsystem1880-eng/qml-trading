"""
Validation Pipeline Orchestrator
=================================
Chains all validation modules into a unified, production-ready pipeline.

Pipeline Flow:
1. Data Loading & Preprocessing
2. Feature Engineering (200+ features)
3. Regime Detection (4 regimes)
4. Purged Walk-Forward Optimization
5. Statistical Testing (Permutation, Monte Carlo, Bootstrap)
6. Diagnostics & Report Generation
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from src.features.library import FeatureLibrary, FeatureLibraryConfig
from src.analysis.regimes import RegimeClassifier, RegimeConfig, RegimeResult
from src.validation.database import VRDDatabase
from src.validation.tracker import ExperimentTracker
from src.validation.walk_forward import (
    PurgedWalkForwardEngine,
    WalkForwardConfig,
    WalkForwardResult,
)
from src.validation.permutation import PermutationTest, PermutationResult
from src.validation.monte_carlo import MonteCarloSimulator, MonteCarloResult
from src.validation.bootstrap import BlockBootstrap, BootstrapResult


class PipelineStage(str, Enum):
    """Pipeline execution stages."""
    DATA_LOADING = "data_loading"
    FEATURE_ENGINEERING = "feature_engineering"
    REGIME_DETECTION = "regime_detection"
    WALK_FORWARD = "walk_forward"
    STATISTICAL_TESTING = "statistical_testing"
    DIAGNOSTICS = "diagnostics"
    REPORT_GENERATION = "report_generation"


@dataclass
class OrchestratorConfig:
    """Configuration for validation orchestrator."""
    
    # Output
    output_dir: str = "validation_pipeline"
    
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
    
    # Regime detection
    n_regimes: int = 4
    regime_method: str = "kmeans"
    
    # Risk thresholds
    kill_switch_threshold: float = 0.20
    significance_level: float = 0.05
    
    # Random seed
    random_seed: int = 42
    
    # Feature engineering
    compute_features: bool = True
    
    # Parallel execution
    n_jobs: int = 1


@dataclass
class StageResult:
    """Result from a single pipeline stage."""
    stage: PipelineStage
    success: bool
    duration_seconds: float
    result: Any = None
    error: Optional[str] = None


@dataclass
class PipelineResult:
    """Complete result from validation pipeline."""
    
    # Metadata
    experiment_id: str
    strategy_name: str
    timestamp: str
    config: OrchestratorConfig
    
    # Stage results
    stage_results: Dict[PipelineStage, StageResult] = field(default_factory=dict)
    
    # Core results
    features_df: Optional[pd.DataFrame] = None
    regime_result: Optional[RegimeResult] = None
    walk_forward_result: Optional[WalkForwardResult] = None
    permutation_result: Optional[PermutationResult] = None
    monte_carlo_result: Optional[MonteCarloResult] = None
    bootstrap_results: Dict[str, BootstrapResult] = field(default_factory=dict)
    
    # Aggregated metrics
    oos_sharpe: float = 0.0
    oos_max_dd: float = 0.0
    oos_win_rate: float = 0.0
    total_trades: int = 0
    sharpe_p_value: float = 1.0
    
    # Confidence intervals
    sharpe_ci: Tuple[float, float] = (0.0, 0.0)
    win_rate_ci: Tuple[float, float] = (0.0, 0.0)
    
    # Verdict
    confidence_score: float = 0.0
    overall_verdict: str = "UNKNOWN"
    verdict_reasons: List[str] = field(default_factory=list)
    
    # Regime breakdown
    regime_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    @property
    def is_successful(self) -> bool:
        """Check if all stages succeeded."""
        return all(sr.success for sr in self.stage_results.values())
    
    @property
    def total_duration(self) -> float:
        """Total pipeline duration in seconds."""
        return sum(sr.duration_seconds for sr in self.stage_results.values())


class ValidationPipeline:
    """
    Modular validation pipeline with pluggable stages.
    
    Allows custom ordering and optional stages.
    """
    
    def __init__(self, stages: Optional[List[PipelineStage]] = None):
        """
        Initialize pipeline with specified stages.
        
        Args:
            stages: List of stages to execute (default: all stages)
        """
        self.stages = stages or [
            PipelineStage.DATA_LOADING,
            PipelineStage.FEATURE_ENGINEERING,
            PipelineStage.REGIME_DETECTION,
            PipelineStage.WALK_FORWARD,
            PipelineStage.STATISTICAL_TESTING,
            PipelineStage.REPORT_GENERATION,
        ]
        self._stage_handlers: Dict[PipelineStage, Callable] = {}
    
    def register_stage(
        self,
        stage: PipelineStage,
        handler: Callable
    ) -> "ValidationPipeline":
        """Register a handler for a pipeline stage."""
        self._stage_handlers[stage] = handler
        return self


class ValidationOrchestrator:
    """
    Master Orchestrator for Strategy Validation.
    
    Chains all validation modules into a unified pipeline:
    Data -> Features -> Regimes -> Walk-Forward -> Statistics -> Report
    
    Provides:
    - Full experiment lifecycle management
    - Regime-stratified analysis
    - Comprehensive statistical testing
    - Production-ready reporting
    """
    
    def __init__(
        self,
        config: Optional[OrchestratorConfig] = None,
        output_dir: Optional[str] = None
    ):
        """
        Initialize orchestrator.
        
        Args:
            config: Orchestrator configuration
            output_dir: Override output directory
        """
        self.config = config or OrchestratorConfig()
        if output_dir:
            self.config.output_dir = output_dir
        
        # Create output directory
        self.output_path = Path(self.config.output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self._init_components()
        
        logger.info(f"ValidationOrchestrator initialized at {self.output_path}")
    
    def _init_components(self):
        """Initialize all pipeline components."""
        # Database and tracking
        self.db = VRDDatabase(str(self.output_path / "vrd.db"))
        self.tracker = ExperimentTracker(
            base_dir=str(self.output_path),
            db=self.db
        )
        
        # Feature engineering
        self.feature_library = FeatureLibrary()
        
        # Regime detection
        regime_config = RegimeConfig(
            n_regimes=self.config.n_regimes,
            method=self.config.regime_method,
            random_state=self.config.random_seed,
        )
        self.regime_classifier = RegimeClassifier(config=regime_config)
        
        # Walk-forward
        wf_config = WalkForwardConfig(
            n_folds=self.config.n_folds,
            purge_bars=self.config.purge_bars,
            embargo_bars=self.config.embargo_bars,
            train_ratio=self.config.train_ratio,
        )
        self.walk_forward = PurgedWalkForwardEngine(
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
    
    def run(
        self,
        strategy_name: str,
        df: pd.DataFrame,
        backtest_fn: Callable[[pd.DataFrame, Dict], Dict[str, Any]],
        param_grid: Dict[str, List],
        optimization_metric: str = "sharpe_ratio"
    ) -> PipelineResult:
        """
        Run complete validation pipeline.
        
        Args:
            strategy_name: Name of strategy
            df: OHLCV DataFrame with columns: time, open, high, low, close, volume
            backtest_fn: Backtest function: (df, params) -> metrics dict
            param_grid: Parameter grid for optimization
            optimization_metric: Metric to optimize
            
        Returns:
            PipelineResult with all validation results
        """
        logger.info(f"=" * 60)
        logger.info(f"STARTING VALIDATION PIPELINE: {strategy_name}")
        logger.info(f"=" * 60)
        
        # Initialize result
        result = PipelineResult(
            experiment_id="",
            strategy_name=strategy_name,
            timestamp=datetime.now().isoformat(),
            config=self.config,
        )
        
        # Get data range
        if "time" in df.columns:
            data_start = str(df["time"].min())
            data_end = str(df["time"].max())
        else:
            data_start = str(df.index.min())
            data_end = str(df.index.max())
        
        # Start experiment tracking
        result.experiment_id = self.tracker.start_experiment(
            strategy_name=strategy_name,
            params={"param_grid": param_grid, "metric": optimization_metric},
            data_range=(data_start, data_end),
            seed=self.config.random_seed,
            fold_count=self.config.n_folds,
        )
        
        try:
            # Stage 1: Feature Engineering
            result = self._stage_feature_engineering(df, result)
            
            # Stage 2: Regime Detection
            result = self._stage_regime_detection(df, result)
            
            # Stage 3: Walk-Forward Optimization
            result = self._stage_walk_forward(
                df, backtest_fn, param_grid, optimization_metric, result
            )
            
            # Stage 4: Statistical Testing
            result = self._stage_statistical_testing(result)
            
            # Stage 5: Bootstrap Confidence Intervals
            result = self._stage_bootstrap_ci(result)
            
            # Stage 6: Generate Verdict
            result = self._generate_verdict(result)
            
            # Finalize experiment
            self._finalize_experiment(result)
            
            logger.info(f"=" * 60)
            logger.info(f"PIPELINE COMPLETE: {result.overall_verdict}")
            logger.info(f"Confidence Score: {result.confidence_score:.1f}/100")
            logger.info(f"=" * 60)
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            self.tracker.fail(str(e))
            raise
        
        return result
    
    def _stage_feature_engineering(
        self,
        df: pd.DataFrame,
        result: PipelineResult
    ) -> PipelineResult:
        """Stage 1: Feature Engineering."""
        logger.info("Stage 1: Feature Engineering")
        start_time = datetime.now()
        
        try:
            if self.config.compute_features:
                features_df = self.feature_library.compute_features_for_range(df)
                result.features_df = features_df
                logger.info(f"  Generated {len(features_df.columns)} features for {len(features_df)} bars")
            
            result.stage_results[PipelineStage.FEATURE_ENGINEERING] = StageResult(
                stage=PipelineStage.FEATURE_ENGINEERING,
                success=True,
                duration_seconds=(datetime.now() - start_time).total_seconds(),
            )
            
        except Exception as e:
            logger.error(f"  Feature engineering failed: {e}")
            result.stage_results[PipelineStage.FEATURE_ENGINEERING] = StageResult(
                stage=PipelineStage.FEATURE_ENGINEERING,
                success=False,
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                error=str(e),
            )
        
        return result
    
    def _stage_regime_detection(
        self,
        df: pd.DataFrame,
        result: PipelineResult
    ) -> PipelineResult:
        """Stage 2: Regime Detection."""
        logger.info("Stage 2: Regime Detection")
        start_time = datetime.now()
        
        try:
            regime_result = self.regime_classifier.fit_predict(df)
            result.regime_result = regime_result
            
            # Get regime statistics
            regime_stats = self.regime_classifier.get_regime_statistics(df, regime_result)
            logger.info(f"  Detected {regime_result.n_regimes} regimes:")
            for _, row in regime_stats.iterrows():
                logger.info(f"    {row['regime']}: {row['pct_of_data']:.1f}% of data")
            
            result.stage_results[PipelineStage.REGIME_DETECTION] = StageResult(
                stage=PipelineStage.REGIME_DETECTION,
                success=True,
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                result=regime_stats,
            )
            
        except Exception as e:
            logger.error(f"  Regime detection failed: {e}")
            result.stage_results[PipelineStage.REGIME_DETECTION] = StageResult(
                stage=PipelineStage.REGIME_DETECTION,
                success=False,
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                error=str(e),
            )
        
        return result
    
    def _stage_walk_forward(
        self,
        df: pd.DataFrame,
        backtest_fn: Callable,
        param_grid: Dict[str, List],
        optimization_metric: str,
        result: PipelineResult
    ) -> PipelineResult:
        """Stage 3: Purged Walk-Forward Optimization."""
        logger.info("Stage 3: Purged Walk-Forward Optimization")
        start_time = datetime.now()
        
        try:
            wf_result = self.walk_forward.run(
                df=df,
                objective_fn=backtest_fn,
                param_grid=param_grid,
                optimization_metric=optimization_metric,
            )
            result.walk_forward_result = wf_result
            
            # Extract key metrics
            result.oos_sharpe = wf_result.aggregate_sharpe
            result.oos_max_dd = wf_result.aggregate_max_dd
            result.oos_win_rate = wf_result.aggregate_win_rate
            result.total_trades = wf_result.total_oos_trades
            
            logger.info(f"  OOS Sharpe: {result.oos_sharpe:.3f}")
            logger.info(f"  OOS Max DD: {result.oos_max_dd:.2f}%")
            logger.info(f"  Total OOS Trades: {result.total_trades}")
            logger.info(f"  Parameter Stability: {wf_result.param_stability_score:.3f}")
            
            result.stage_results[PipelineStage.WALK_FORWARD] = StageResult(
                stage=PipelineStage.WALK_FORWARD,
                success=True,
                duration_seconds=(datetime.now() - start_time).total_seconds(),
            )
            
        except Exception as e:
            logger.error(f"  Walk-forward failed: {e}")
            result.stage_results[PipelineStage.WALK_FORWARD] = StageResult(
                stage=PipelineStage.WALK_FORWARD,
                success=False,
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                error=str(e),
            )
        
        return result
    
    def _stage_statistical_testing(
        self,
        result: PipelineResult
    ) -> PipelineResult:
        """Stage 4: Statistical Testing (Permutation + Monte Carlo)."""
        logger.info("Stage 4: Statistical Testing")
        start_time = datetime.now()
        
        try:
            # Collect OOS returns
            trade_returns = self._collect_oos_returns(result.walk_forward_result)
            
            if len(trade_returns) < 10:
                logger.warning(f"  Insufficient trades ({len(trade_returns)}) for statistical testing")
                result.stage_results[PipelineStage.STATISTICAL_TESTING] = StageResult(
                    stage=PipelineStage.STATISTICAL_TESTING,
                    success=True,
                    duration_seconds=(datetime.now() - start_time).total_seconds(),
                )
                return result
            
            # Permutation Test
            logger.info("  Running Permutation Test...")
            perm_result = self.permutation_test.run(trade_returns)
            result.permutation_result = perm_result
            result.sharpe_p_value = perm_result.sharpe_p_value
            
            logger.info(f"    Sharpe p-value: {perm_result.sharpe_p_value:.4f}")
            logger.info(f"    Sharpe percentile: {perm_result.sharpe_percentile:.1f}%")
            
            # Monte Carlo Simulation
            logger.info("  Running Monte Carlo Simulation...")
            mc_result = self.monte_carlo.run(trade_returns)
            result.monte_carlo_result = mc_result
            
            logger.info(f"    VaR 95%: {mc_result.var_95:.2f}%")
            logger.info(f"    VaR 99%: {mc_result.var_99:.2f}%")
            logger.info(f"    Kill Switch Prob: {mc_result.kill_switch_prob:.2%}")
            
            result.stage_results[PipelineStage.STATISTICAL_TESTING] = StageResult(
                stage=PipelineStage.STATISTICAL_TESTING,
                success=True,
                duration_seconds=(datetime.now() - start_time).total_seconds(),
            )
            
        except Exception as e:
            logger.error(f"  Statistical testing failed: {e}")
            result.stage_results[PipelineStage.STATISTICAL_TESTING] = StageResult(
                stage=PipelineStage.STATISTICAL_TESTING,
                success=False,
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                error=str(e),
            )
        
        return result
    
    def _stage_bootstrap_ci(
        self,
        result: PipelineResult
    ) -> PipelineResult:
        """Stage 5: Bootstrap Confidence Intervals."""
        logger.info("Stage 5: Bootstrap Confidence Intervals")
        start_time = datetime.now()
        
        try:
            trade_returns = self._collect_oos_returns(result.walk_forward_result)
            
            if len(trade_returns) < 10:
                logger.warning(f"  Insufficient trades for bootstrap CI")
                return result
            
            # Compute all CIs
            trades_df = pd.DataFrame({"pnl_pct": trade_returns})
            boot_results = self.bootstrap.all_metrics_ci(trades_df)
            result.bootstrap_results = boot_results
            
            # Extract key CIs
            if "sharpe_ratio" in boot_results:
                sr = boot_results["sharpe_ratio"]
                result.sharpe_ci = (sr.ci_lower, sr.ci_upper)
                logger.info(f"  Sharpe CI: [{sr.ci_lower:.3f}, {sr.ci_upper:.3f}]")
            
            if "win_rate" in boot_results:
                wr = boot_results["win_rate"]
                result.win_rate_ci = (wr.ci_lower, wr.ci_upper)
                logger.info(f"  Win Rate CI: [{wr.ci_lower:.3f}, {wr.ci_upper:.3f}]")
            
            result.stage_results[PipelineStage.DIAGNOSTICS] = StageResult(
                stage=PipelineStage.DIAGNOSTICS,
                success=True,
                duration_seconds=(datetime.now() - start_time).total_seconds(),
            )
            
        except Exception as e:
            logger.error(f"  Bootstrap CI failed: {e}")
            result.stage_results[PipelineStage.DIAGNOSTICS] = StageResult(
                stage=PipelineStage.DIAGNOSTICS,
                success=False,
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                error=str(e),
            )
        
        return result
    
    def _generate_verdict(self, result: PipelineResult) -> PipelineResult:
        """Generate overall verdict and confidence score."""
        score_components = []
        reasons = []
        
        # Component 1: Statistical significance (30%)
        if result.permutation_result:
            pval = result.permutation_result.sharpe_p_value
            if pval < 0.01:
                score_components.append(30)
                reasons.append(f"Strong statistical significance (p={pval:.4f})")
            elif pval < 0.05:
                score_components.append(20)
                reasons.append(f"Statistically significant (p={pval:.4f})")
            elif pval < 0.10:
                score_components.append(10)
                reasons.append(f"Marginal significance (p={pval:.4f})")
            else:
                score_components.append(0)
                reasons.append(f"NOT significant (p={pval:.4f})")
        
        # Component 2: Sharpe quality (25%)
        if result.oos_sharpe > 1.5:
            score_components.append(25)
            reasons.append(f"Excellent Sharpe ({result.oos_sharpe:.2f})")
        elif result.oos_sharpe > 1.0:
            score_components.append(18)
            reasons.append(f"Good Sharpe ({result.oos_sharpe:.2f})")
        elif result.oos_sharpe > 0.5:
            score_components.append(10)
            reasons.append(f"Moderate Sharpe ({result.oos_sharpe:.2f})")
        else:
            score_components.append(0)
            reasons.append(f"Poor Sharpe ({result.oos_sharpe:.2f})")
        
        # Component 3: Risk (kill switch) (25%)
        if result.monte_carlo_result:
            kill_prob = result.monte_carlo_result.kill_switch_prob
            if kill_prob < 0.05:
                score_components.append(25)
                reasons.append(f"Low ruin risk ({kill_prob:.1%})")
            elif kill_prob < 0.15:
                score_components.append(15)
                reasons.append(f"Moderate ruin risk ({kill_prob:.1%})")
            else:
                score_components.append(5)
                reasons.append(f"High ruin risk ({kill_prob:.1%})")
        
        # Component 4: Parameter stability (20%)
        if result.walk_forward_result:
            stability = result.walk_forward_result.param_stability_score
            if stability > 0.8:
                score_components.append(20)
                reasons.append("Highly stable parameters")
            elif stability > 0.6:
                score_components.append(12)
                reasons.append("Moderately stable parameters")
            else:
                score_components.append(5)
                reasons.append("Unstable parameters (overfit risk)")
        
        # Calculate score
        result.confidence_score = sum(score_components)
        result.verdict_reasons = reasons
        
        # Determine verdict
        is_significant = (result.permutation_result and 
                         result.permutation_result.sharpe_p_value < self.config.significance_level)
        
        if result.confidence_score >= 70 and is_significant:
            result.overall_verdict = "DEPLOY"
        elif result.confidence_score >= 50:
            result.overall_verdict = "CAUTION"
        else:
            result.overall_verdict = "REJECT"
        
        return result
    
    def _collect_oos_returns(
        self,
        wf_result: Optional[WalkForwardResult]
    ) -> np.ndarray:
        """Collect out-of-sample returns from walk-forward."""
        if wf_result is None:
            return np.array([])
        
        all_returns = []
        for fold in wf_result.fold_results:
            oos_metrics = fold.out_of_sample_metrics
            trades = fold.out_of_sample_trades
            
            if trades > 0:
                win_rate = oos_metrics.get("win_rate", 0.5)
                avg_win = oos_metrics.get("avg_win_pct", 2.0)
                avg_loss = oos_metrics.get("avg_loss_pct", -1.0)
                
                n_wins = int(trades * win_rate)
                n_losses = trades - n_wins
                
                returns = [avg_win] * n_wins + [avg_loss] * n_losses
                all_returns.extend(returns)
        
        if not all_returns and wf_result.total_oos_trades > 0:
            avg_return = wf_result.aggregate_return / wf_result.total_oos_trades
            all_returns = [avg_return] * wf_result.total_oos_trades
        
        return np.array(all_returns)
    
    def _finalize_experiment(self, result: PipelineResult):
        """Finalize experiment and save results."""
        metrics = {
            "sharpe_ratio": result.oos_sharpe,
            "max_drawdown_pct": result.oos_max_dd,
            "win_rate": result.oos_win_rate,
            "total_trades": result.total_trades,
            "confidence_score": result.confidence_score,
        }
        
        statistical_results = {
            "sharpe_p_value": result.sharpe_p_value,
            "sharpe_ci_lower": result.sharpe_ci[0],
            "sharpe_ci_upper": result.sharpe_ci[1],
        }
        
        if result.monte_carlo_result:
            statistical_results["var_95"] = result.monte_carlo_result.var_95
            statistical_results["var_99"] = result.monte_carlo_result.var_99
            statistical_results["kill_switch_prob"] = result.monte_carlo_result.kill_switch_prob
        
        self.tracker.finalize(
            metrics=metrics,
            statistical_results=statistical_results,
        )
        
        # Save report
        self._save_report(result)
    
    def _save_report(self, result: PipelineResult):
        """Save pipeline report to disk."""
        import json
        
        report_dir = self.tracker.current_experiment_dir or self.output_path
        
        summary = {
            "experiment_id": result.experiment_id,
            "strategy_name": result.strategy_name,
            "timestamp": result.timestamp,
            "overall_verdict": result.overall_verdict,
            "confidence_score": result.confidence_score,
            "verdict_reasons": result.verdict_reasons,
            "metrics": {
                "oos_sharpe": result.oos_sharpe,
                "oos_max_dd": result.oos_max_dd,
                "oos_win_rate": result.oos_win_rate,
                "total_trades": result.total_trades,
            },
            "statistical": {
                "sharpe_p_value": result.sharpe_p_value,
                "sharpe_ci": result.sharpe_ci,
                "win_rate_ci": result.win_rate_ci,
            },
            "pipeline": {
                "total_duration_seconds": result.total_duration,
                "stages": {
                    str(stage.value): {
                        "success": sr.success,
                        "duration": sr.duration_seconds,
                    }
                    for stage, sr in result.stage_results.items()
                },
            },
        }
        
        report_path = report_dir / "pipeline_report.json"
        with open(report_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Report saved to {report_path}")
    
    def generate_text_report(self, result: PipelineResult) -> str:
        """Generate human-readable text report."""
        lines = [
            "=" * 70,
            "VALIDATION PIPELINE REPORT",
            "=" * 70,
            "",
            f"Strategy: {result.strategy_name}",
            f"Experiment: {result.experiment_id}",
            f"Timestamp: {result.timestamp}",
            f"Duration: {result.total_duration:.1f}s",
            "",
            "=" * 70,
            f"VERDICT: {result.overall_verdict}",
            f"CONFIDENCE: {result.confidence_score}/100",
            "=" * 70,
            "",
            "Reasons:",
        ]
        
        for reason in result.verdict_reasons:
            lines.append(f"  • {reason}")
        
        lines.extend([
            "",
            "-" * 70,
            "OUT-OF-SAMPLE METRICS",
            "-" * 70,
            f"  Sharpe Ratio: {result.oos_sharpe:.3f} CI: [{result.sharpe_ci[0]:.3f}, {result.sharpe_ci[1]:.3f}]",
            f"  Max Drawdown: {result.oos_max_dd:.2f}%",
            f"  Win Rate: {result.oos_win_rate:.1%} CI: [{result.win_rate_ci[0]:.1%}, {result.win_rate_ci[1]:.1%}]",
            f"  Total Trades: {result.total_trades}",
        ])
        
        if result.permutation_result:
            lines.extend([
                "",
                "-" * 70,
                "STATISTICAL SIGNIFICANCE",
                "-" * 70,
                f"  Sharpe p-value: {result.permutation_result.sharpe_p_value:.4f}",
                f"  Sharpe percentile: {result.permutation_result.sharpe_percentile:.1f}%",
            ])
        
        if result.monte_carlo_result:
            mc = result.monte_carlo_result
            lines.extend([
                "",
                "-" * 70,
                "RISK ANALYSIS",
                "-" * 70,
                f"  VaR 95%: {mc.var_95:.2f}%",
                f"  VaR 99%: {mc.var_99:.2f}%",
                f"  Expected Shortfall: {mc.expected_shortfall_95:.2f}%",
                f"  Kill Switch Prob: {mc.kill_switch_prob:.2%}",
            ])
        
        if result.walk_forward_result:
            wf = result.walk_forward_result
            lines.extend([
                "",
                "-" * 70,
                "WALK-FORWARD ANALYSIS",
                "-" * 70,
                f"  Folds: {len(wf.fold_results)}",
                f"  Parameter Stability: {wf.param_stability_score:.3f}",
                f"  IS/OOS Ratio: {wf.is_to_oos_ratio:.3f}",
            ])
        
        lines.extend([
            "",
            "-" * 70,
            "PIPELINE STAGES",
            "-" * 70,
        ])
        
        for stage, sr in result.stage_results.items():
            status = "✓" if sr.success else "✗"
            lines.append(f"  {status} {stage.value}: {sr.duration_seconds:.1f}s")
        
        lines.append("=" * 70)
        
        return "\n".join(lines)


def create_orchestrator(
    output_dir: str = "validation_pipeline",
    n_folds: int = 10,
    n_permutations: int = 10000,
    n_monte_carlo: int = 50000
) -> ValidationOrchestrator:
    """
    Factory function for ValidationOrchestrator.
    
    Args:
        output_dir: Output directory
        n_folds: Walk-forward folds
        n_permutations: Permutation test iterations
        n_monte_carlo: Monte Carlo simulations
        
    Returns:
        Configured ValidationOrchestrator
    """
    config = OrchestratorConfig(
        output_dir=output_dir,
        n_folds=n_folds,
        n_permutations=n_permutations,
        n_monte_carlo=n_monte_carlo,
    )
    return ValidationOrchestrator(config=config)
