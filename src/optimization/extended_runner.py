"""
Extended Runner for Phase 7.7 Optimization
==========================================
Extends ParallelDetectionRunner with:

1. Walk-Forward Validation - Rolling train/test splits with purge gaps
2. Symbol-Cluster Validation - Test generalization across asset categories
3. Trade Simulation Integration - Run trades as part of detection evaluation

Ensures parameters generalize across time and asset types.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import time
import json

import numpy as np
import pandas as pd

from src.optimization.parallel_runner import (
    ParallelDetectionRunner,
    AggregateResult,
    DetectionResult,
    PatternInfo,
)
from src.optimization.trade_simulator import (
    TradeSimulator,
    TradeManagementConfig,
    SimulationResult,
)
from src.optimization.objectives import (
    ObjectiveFunction,
    ObjectiveResult,
    ObjectiveConfig,
    create_objective,
    ObjectiveType,
)
from src.data_engine import load_master_data


# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent


# =============================================================================
# SYMBOL CLUSTERS
# =============================================================================

# Predefined symbol clusters by asset type
SYMBOL_CLUSTERS = {
    'store_of_value': ['BTCUSDT'],
    'smart_contract_l1': ['ETHUSDT', 'SOLUSDT', 'AVAXUSDT', 'NEARUSDT', 'APTUSDT'],
    'smart_contract_l2': ['ARBUSDT', 'OPUSDT', 'MATICUSDT'],
    'defi': ['AAVEUSDT', 'UNIUSDT', 'LINKUSDT', 'MKRUSDT', 'INJUSDT'],
    'exchange_token': ['BNBUSDT'],
    'meme': ['DOGEUSDT', 'PEPEUSDT', 'WIFUSDT'],
    'infrastructure': ['DOTUSDT', 'ATOMUSDT', 'RUNEUSDT', 'TIAUSDT'],
}

# Flattened list of all clustered symbols
ALL_CLUSTERED_SYMBOLS = [s for cluster in SYMBOL_CLUSTERS.values() for s in cluster]


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward validation."""
    # Number of folds
    n_folds: int = 5

    # Rolling window type
    # 'rolling' = fixed window size moves forward
    # 'expanding' = window grows over time
    window_type: str = 'rolling'

    # Train/test split ratio (train portion)
    train_ratio: float = 0.7

    # Purge gap in bars (prevent lookahead)
    purge_bars: int = 50

    # Embargo bars after test (prevent signal leakage)
    embargo_bars: int = 20

    # Minimum bars per fold
    min_fold_bars: int = 500


@dataclass
class WalkForwardFold:
    """Single fold in walk-forward validation."""
    fold_idx: int
    train_start_idx: int
    train_end_idx: int
    test_start_idx: int
    test_end_idx: int

    # Results
    train_result: Optional[AggregateResult] = None
    test_result: Optional[AggregateResult] = None
    train_sim_result: Optional[SimulationResult] = None
    test_sim_result: Optional[SimulationResult] = None


@dataclass
class WalkForwardResult:
    """Result of walk-forward validation."""
    folds: List[WalkForwardFold]

    # Aggregate metrics across folds
    mean_test_patterns: float = 0.0
    std_test_patterns: float = 0.0
    mean_test_quality: float = 0.0
    std_test_quality: float = 0.0

    # Trade simulation aggregates
    mean_test_sharpe: float = 0.0
    mean_test_expectancy: float = 0.0
    mean_test_win_rate: float = 0.0

    # Consistency ratio (test/train performance)
    consistency_ratio: float = 0.0

    # Stability score (low std = stable)
    stability_score: float = 0.0


@dataclass
class ClusterValidationResult:
    """Result of symbol-cluster validation."""
    cluster_results: Dict[str, AggregateResult]
    cluster_sim_results: Dict[str, SimulationResult]

    # Cross-cluster metrics
    mean_patterns_per_cluster: float = 0.0
    std_patterns_per_cluster: float = 0.0
    clusters_with_patterns: int = 0
    total_clusters: int = 0

    # Generalization score (how well params work across clusters)
    generalization_score: float = 0.0


@dataclass
class ExtendedRunnerConfig:
    """Configuration for extended runner."""
    # Walk-forward settings
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)

    # Trade management settings
    trade_management: TradeManagementConfig = field(default_factory=TradeManagementConfig)

    # Objective settings
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)

    # Timeframes to process
    timeframes: List[str] = field(default_factory=lambda: ['1h', '4h', '1d'])

    # Whether to run cluster validation
    run_cluster_validation: bool = True

    # Whether to run walk-forward validation
    run_walk_forward: bool = True


class ExtendedDetectionRunner(ParallelDetectionRunner):
    """
    Extended detection runner with validation capabilities.

    Inherits from ParallelDetectionRunner and adds:
    - Walk-forward validation
    - Symbol-cluster validation
    - Trade simulation integration
    """

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        timeframes: Optional[List[str]] = None,
        n_jobs: int = -1,
        config: Optional[ExtendedRunnerConfig] = None,
    ):
        """
        Initialize the extended runner.

        Args:
            symbols: List of symbols to process
            timeframes: List of timeframes to process
            n_jobs: Number of parallel jobs
            config: Extended runner configuration
        """
        # Use clustered symbols by default
        symbols = symbols or ALL_CLUSTERED_SYMBOLS

        super().__init__(symbols=symbols, timeframes=timeframes, n_jobs=n_jobs)

        self.config = config or ExtendedRunnerConfig()
        self.timeframes = timeframes or self.config.timeframes

        # Initialize components
        self.trade_simulator = TradeSimulator(self.config.trade_management)
        self._objective: Optional[ObjectiveFunction] = None

    def set_objective(self, objective_type: ObjectiveType) -> None:
        """Set the objective function for optimization."""
        self._objective = create_objective(objective_type, self.config.objective)

    def run_with_dict_extended(
        self,
        params: Dict[str, Any],
        objective_type: Optional[ObjectiveType] = None,
    ) -> Tuple[ObjectiveResult, AggregateResult, SimulationResult]:
        """
        Run detection and trade simulation with given parameters.

        Args:
            params: Parameter dictionary
            objective_type: Objective to evaluate (uses current if None)

        Returns:
            Tuple of (ObjectiveResult, AggregateResult, SimulationResult)
        """
        # Set objective if provided
        if objective_type:
            self.set_objective(objective_type)

        if self._objective is None:
            self.set_objective(ObjectiveType.COMPOSITE)

        # Extract trade management params
        trade_params = self._extract_trade_params(params)
        self.trade_simulator.config = TradeManagementConfig(**trade_params)

        # Run detection
        detection_result = self.run_with_dict(params)

        # Convert patterns to signals for simulation
        signals = self._patterns_to_signals(detection_result)

        # Run trade simulation across all data
        sim_result = self._run_simulation_all_symbols(signals, params)

        # Evaluate objective
        detection_dict = self._aggregate_to_dict(detection_result)
        obj_result = self._objective.evaluate(sim_result, detection_dict)

        return obj_result, detection_result, sim_result

    def run_walk_forward(
        self,
        params: Dict[str, Any],
        symbol: str = 'BTCUSDT',
        timeframe: str = '4h',
    ) -> WalkForwardResult:
        """
        Run walk-forward validation on a single symbol.

        Args:
            params: Parameter dictionary
            symbol: Symbol to validate on
            timeframe: Timeframe to use

        Returns:
            WalkForwardResult with fold-by-fold results
        """
        cfg = self.config.walk_forward

        # Load data
        cache_key = f"{symbol}_{timeframe}"
        if cache_key in self._data_cache:
            df = self._data_cache[cache_key]
        else:
            df = load_master_data(timeframe, symbol=symbol)
            df.columns = [c.lower() for c in df.columns]

        total_bars = len(df)

        # Calculate fold boundaries
        folds = self._calculate_fold_boundaries(total_bars, cfg)

        # Run each fold
        for fold in folds:
            # Train split
            train_df = df.iloc[fold.train_start_idx:fold.train_end_idx].copy()

            # Test split
            test_df = df.iloc[fold.test_start_idx:fold.test_end_idx].copy()

            # Run detection on train
            train_result = self._run_single_symbol(params, train_df, symbol, timeframe)
            fold.train_result = self._detection_to_aggregate([train_result])

            # Run detection on test
            test_result = self._run_single_symbol(params, test_df, symbol, timeframe)
            fold.test_result = self._detection_to_aggregate([test_result])

            # Run simulation on train
            train_signals = self._result_to_signals(train_result)
            fold.train_sim_result = self.trade_simulator.simulate_trades(
                train_df, train_signals, symbol, timeframe
            )

            # Run simulation on test
            test_signals = self._result_to_signals(test_result)
            fold.test_sim_result = self.trade_simulator.simulate_trades(
                test_df, test_signals, symbol, timeframe
            )

        # Calculate aggregate metrics
        return self._aggregate_walk_forward(folds)

    def run_cluster_validation(
        self,
        params: Dict[str, Any],
        timeframe: str = '4h',
    ) -> ClusterValidationResult:
        """
        Run symbol-cluster validation.

        Tests parameters across different asset categories to ensure
        generalization.

        Args:
            params: Parameter dictionary
            timeframe: Timeframe to use

        Returns:
            ClusterValidationResult
        """
        cluster_results: Dict[str, AggregateResult] = {}
        cluster_sim_results: Dict[str, SimulationResult] = {}

        for cluster_name, cluster_symbols in SYMBOL_CLUSTERS.items():
            # Filter to available symbols
            available = [s for s in cluster_symbols if s in self.symbols]

            if not available:
                continue

            # Create temporary runner for this cluster
            cluster_runner = ParallelDetectionRunner(
                symbols=available,
                timeframes=[timeframe],
                n_jobs=self.n_jobs,
            )
            cluster_runner._data_cache = self._data_cache  # Share cache

            # Run detection
            result = cluster_runner.run_with_dict(params)
            cluster_results[cluster_name] = result

            # Run simulation
            signals = self._patterns_to_signals_for_result(result, cluster_runner)
            sim_result = self._run_simulation_for_symbols(
                signals, params, available, timeframe
            )
            cluster_sim_results[cluster_name] = sim_result

        # Calculate generalization metrics
        return self._aggregate_cluster_results(cluster_results, cluster_sim_results)

    def _extract_trade_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Extract trade management parameters from full params dict."""
        trade_param_names = [
            'entry_buffer_atr', 'sl_atr_mult', 'tp_atr_mult',
            'trailing_activation_atr', 'trailing_step_atr',
            'max_bars_held', 'min_risk_reward',
            'slippage_pct', 'commission_pct',
        ]

        trade_params = {}
        defaults = TradeManagementConfig()

        for name in trade_param_names:
            trade_params[name] = params.get(name, getattr(defaults, name))

        return trade_params

    def _patterns_to_signals(
        self,
        detection_result: AggregateResult,
    ) -> List[Dict[str, Any]]:
        """Convert detection results to signal format for simulation."""
        signals = []

        for key, result in detection_result.symbol_results.items():
            if result.error or result.num_patterns == 0:
                continue

            # Use the PatternInfo objects stored in the result
            for pattern_info in result.patterns:
                signals.append({
                    'symbol': result.symbol,
                    'timeframe': result.timeframe,
                    'bar_idx': pattern_info.bar_idx,
                    'direction': pattern_info.direction,
                    'score': pattern_info.score,
                    'atr': pattern_info.atr,
                    'entry_price': pattern_info.entry_price,
                })

        return signals

    def _patterns_to_signals_for_result(
        self,
        result: AggregateResult,
        runner: ParallelDetectionRunner,
    ) -> List[Dict[str, Any]]:
        """Convert patterns from specific runner result."""
        signals = []

        for key, det_result in result.symbol_results.items():
            if det_result.error or det_result.num_patterns == 0:
                continue

            # Use the PatternInfo objects stored in the result
            for pattern_info in det_result.patterns:
                signals.append({
                    'symbol': det_result.symbol,
                    'timeframe': det_result.timeframe,
                    'bar_idx': pattern_info.bar_idx,
                    'direction': pattern_info.direction,
                    'score': pattern_info.score,
                    'atr': pattern_info.atr,
                    'entry_price': pattern_info.entry_price,
                })

        return signals

    def _run_simulation_all_symbols(
        self,
        signals: List[Dict[str, Any]],
        params: Dict[str, Any],
    ) -> SimulationResult:
        """Run simulation aggregated across all symbols."""
        all_trades = []

        for symbol in self.symbols:
            for tf in self.timeframes:
                cache_key = f"{symbol}_{tf}"
                if cache_key not in self._data_cache:
                    continue

                df = self._data_cache[cache_key]
                symbol_signals = [
                    s for s in signals
                    if s.get('symbol') == symbol and s.get('timeframe') == tf
                ]

                if symbol_signals:
                    result = self.trade_simulator.simulate_trades(
                        df, symbol_signals, symbol, tf
                    )
                    all_trades.extend(result.trades)

        # Re-aggregate all trades
        return self.trade_simulator._calculate_results(all_trades)

    def _run_simulation_for_symbols(
        self,
        signals: List[Dict[str, Any]],
        params: Dict[str, Any],
        symbols: List[str],
        timeframe: str,
    ) -> SimulationResult:
        """Run simulation for specific symbol list."""
        all_trades = []

        for symbol in symbols:
            cache_key = f"{symbol}_{timeframe}"
            if cache_key not in self._data_cache:
                continue

            df = self._data_cache[cache_key]
            symbol_signals = [
                s for s in signals
                if s.get('symbol') == symbol
            ]

            if symbol_signals:
                result = self.trade_simulator.simulate_trades(
                    df, symbol_signals, symbol, timeframe
                )
                all_trades.extend(result.trades)

        return self.trade_simulator._calculate_results(all_trades)

    def _run_single_symbol(
        self,
        params: Dict[str, Any],
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
    ) -> DetectionResult:
        """Run detection on a single symbol's data."""
        from src.detection.hierarchical_swing import HierarchicalSwingDetector, HierarchicalSwingConfig
        from src.detection.pattern_validator import PatternValidator, PatternValidationConfig
        from src.detection.trend_validator import TrendValidator, TrendValidationConfig
        from src.detection.pattern_scorer import PatternScorer, PatternScoringConfig
        from src.detection.regime import MarketRegimeDetector

        start = time.time()

        try:
            # Build configs from params
            swing_config = HierarchicalSwingConfig(
                min_bar_separation=params.get('min_bar_separation', 5),
                min_move_atr=params.get('min_move_atr', 1.0),
                forward_confirm_pct=params.get('forward_confirm_pct', 0.3),
                lookback=params.get('lookback', 5),
                lookforward=params.get('lookforward', 5),
            )

            validation_config = PatternValidationConfig(
                p3_min_extension_atr=params.get('p3_min_extension_atr', 0.5),
                p3_max_extension_atr=params.get('p3_max_extension_atr', 5.0),
                p4_min_break_atr=params.get('p4_min_break_atr', 0.1),
                p5_max_symmetry_atr=params.get('p5_max_symmetry_atr', 2.0),
                min_pattern_bars=params.get('min_pattern_bars', 10),
            )

            trend_config = TrendValidationConfig(
                min_adx=params.get('min_adx', 20.0),
                min_trend_move_atr=params.get('min_trend_move_atr', 3.0),
                min_trend_swings=params.get('min_trend_swings', 3),
            )

            # Get weights from params (Phase 7.8: now 8 components, all optimizable except shoulder)
            head_weight = params.get('head_extension_weight', 0.22)
            bos_weight = params.get('bos_efficiency_weight', 0.18)
            volume_weight = params.get('volume_spike_weight', 0.10)
            path_weight = params.get('path_efficiency_weight', 0.10)
            trend_weight = params.get('trend_strength_weight', 0.10)
            regime_weight = params.get('regime_suitability_weight', 0.10)
            shoulder_weight = 0.12

            # Adjust swing weight to ensure sum = 1.0
            swing_weight = max(0.02, 1.0 - head_weight - bos_weight - volume_weight - path_weight - trend_weight - regime_weight - shoulder_weight)

            scoring_config = PatternScoringConfig(
                head_extension_weight=head_weight,
                bos_efficiency_weight=bos_weight,
                shoulder_symmetry_weight=shoulder_weight,
                swing_significance_weight=swing_weight,
                volume_spike_weight=volume_weight,
                path_efficiency_weight=path_weight,
                trend_strength_weight=trend_weight,
                regime_suitability_weight=regime_weight,
            )

            # Initialize detectors
            swing_detector = HierarchicalSwingDetector(
                config=swing_config,
                symbol=symbol,
                timeframe=timeframe,
            )

            validator = PatternValidator(config=validation_config)
            trend_validator = TrendValidator(config=trend_config)
            scorer = PatternScorer(config=scoring_config)
            regime_detector = MarketRegimeDetector()

            # Detect swings
            swings = swing_detector.detect(df)

            # Get regime for this data (Phase 7.8)
            regime_result = regime_detector.get_regime(df)

            if len(swings) < 5:
                return DetectionResult(
                    symbol=symbol,
                    timeframe=timeframe,
                    num_patterns=0,
                    num_swings=len(swings),
                    duration_ms=(time.time() - start) * 1000,
                )

            # Find patterns
            price_data = df['close'].values
            patterns = validator.find_patterns(swings, price_data)

            # Filter and score
            valid_patterns = []
            scores = []
            pattern_infos = []
            tier_counts = {'A': 0, 'B': 0, 'C': 0, 'REJECT': 0}

            # Get ATR values
            atr_col = df['atr'] if 'atr' in df.columns else df.get('ATR', pd.Series([1.0]*len(df)))

            for pattern in patterns:
                if not pattern.is_valid:
                    continue

                trend_result = trend_validator.validate(
                    swings, pattern.p1.bar_index, df, pattern.direction
                )

                if not trend_result.is_valid:
                    continue

                score_result = scorer.score(
                    pattern,
                    df=df,
                    trend_result=trend_result,
                    regime_result=regime_result,
                )
                valid_patterns.append(pattern)
                scores.append(score_result.total_score)
                tier_counts[score_result.tier.value] += 1

                # Store pattern info for trade simulation
                p5_idx = pattern.p5.bar_index
                direction = 'SHORT' if pattern.direction.value == 'bullish' else 'LONG'
                atr_val = float(atr_col.iloc[p5_idx]) if p5_idx < len(atr_col) else 1.0
                entry_price = float(df['close'].iloc[p5_idx]) if p5_idx < len(df) else pattern.p5.price

                pattern_infos.append(PatternInfo(
                    bar_idx=p5_idx,
                    direction=direction,
                    score=score_result.total_score,
                    atr=atr_val,
                    entry_price=entry_price,
                ))

            return DetectionResult(
                symbol=symbol,
                timeframe=timeframe,
                num_patterns=len(valid_patterns),
                num_swings=len(swings),
                scores=scores,
                mean_score=float(np.mean(scores)) if scores else 0.0,
                tier_a=tier_counts['A'],
                patterns=pattern_infos,
                tier_b=tier_counts['B'],
                tier_c=tier_counts['C'],
                rejected=tier_counts['REJECT'],
                duration_ms=(time.time() - start) * 1000,
            )

        except Exception as e:
            return DetectionResult(
                symbol=symbol,
                timeframe=timeframe,
                num_patterns=0,
                num_swings=0,
                duration_ms=(time.time() - start) * 1000,
                error=str(e),
            )

    def _result_to_signals(self, result: DetectionResult) -> List[Dict[str, Any]]:
        """Convert DetectionResult to signal list."""
        signals = []

        if result.error or result.num_patterns == 0:
            return signals

        # Use the PatternInfo objects stored in the result
        for pattern_info in result.patterns:
            signals.append({
                'bar_idx': pattern_info.bar_idx,
                'direction': pattern_info.direction,
                'score': pattern_info.score,
                'atr': pattern_info.atr,
                'entry_price': pattern_info.entry_price,
            })

        return signals

    def _calculate_fold_boundaries(
        self,
        total_bars: int,
        cfg: WalkForwardConfig,
    ) -> List[WalkForwardFold]:
        """Calculate train/test boundaries for each fold."""
        folds = []

        if cfg.window_type == 'rolling':
            # Rolling window: fixed size moves forward
            fold_size = total_bars // cfg.n_folds
            train_size = int(fold_size * cfg.train_ratio)

            for i in range(cfg.n_folds):
                fold_start = i * (total_bars - fold_size) // (cfg.n_folds - 1) if cfg.n_folds > 1 else 0
                train_start = fold_start
                train_end = train_start + train_size

                # Purge gap
                test_start = train_end + cfg.purge_bars
                test_end = min(test_start + (fold_size - train_size - cfg.purge_bars), total_bars)

                if test_end - test_start < cfg.min_fold_bars // 2:
                    continue

                folds.append(WalkForwardFold(
                    fold_idx=i,
                    train_start_idx=train_start,
                    train_end_idx=train_end,
                    test_start_idx=test_start,
                    test_end_idx=test_end,
                ))
        else:
            # Expanding window: grows over time
            test_size = total_bars // (cfg.n_folds + 1)

            for i in range(cfg.n_folds):
                train_end = (i + 1) * test_size + test_size
                test_start = train_end + cfg.purge_bars
                test_end = min(test_start + test_size, total_bars)

                if test_end - test_start < cfg.min_fold_bars // 2:
                    continue

                folds.append(WalkForwardFold(
                    fold_idx=i,
                    train_start_idx=0,
                    train_end_idx=train_end,
                    test_start_idx=test_start,
                    test_end_idx=test_end,
                ))

        return folds

    def _detection_to_aggregate(self, results: List[DetectionResult]) -> AggregateResult:
        """Convert list of DetectionResults to AggregateResult."""
        symbol_results = {f"{r.symbol}_{r.timeframe}": r for r in results}

        all_scores = []
        for r in results:
            all_scores.extend(r.scores)

        symbols_with_patterns = set()
        for r in results:
            if r.num_patterns > 0:
                symbols_with_patterns.add(r.symbol)

        total_a = sum(r.tier_a for r in results)
        total_b = sum(r.tier_b for r in results)
        total_c = sum(r.tier_c for r in results)

        return AggregateResult(
            total_patterns=sum(r.num_patterns for r in results),
            unique_symbols=len(symbols_with_patterns),
            total_symbols=len(set(r.symbol for r in results)),
            mean_score=float(np.mean(all_scores)) if all_scores else 0.0,
            median_score=float(np.median(all_scores)) if all_scores else 0.0,
            tier_a_count=total_a,
            tier_b_count=total_b,
            tier_c_count=total_c,
            symbol_results=symbol_results,
        )

    def _aggregate_walk_forward(self, folds: List[WalkForwardFold]) -> WalkForwardResult:
        """Aggregate walk-forward fold results."""
        test_patterns = [f.test_result.total_patterns for f in folds if f.test_result]
        test_qualities = [f.test_result.mean_score for f in folds if f.test_result]
        train_patterns = [f.train_result.total_patterns for f in folds if f.train_result]

        test_sharpes = [f.test_sim_result.sharpe_ratio for f in folds if f.test_sim_result]
        test_expectancies = [f.test_sim_result.expectancy_r for f in folds if f.test_sim_result]
        test_win_rates = [f.test_sim_result.win_rate for f in folds if f.test_sim_result]

        # Consistency: test performance / train performance
        consistency_ratios = []
        for f in folds:
            if f.train_result and f.test_result and f.train_result.total_patterns > 0:
                ratio = f.test_result.total_patterns / f.train_result.total_patterns
                consistency_ratios.append(ratio)

        consistency = np.mean(consistency_ratios) if consistency_ratios else 0.0

        # Stability: inverse of coefficient of variation
        cv = np.std(test_patterns) / np.mean(test_patterns) if test_patterns and np.mean(test_patterns) > 0 else 1.0
        stability = 1.0 / (1.0 + cv)

        return WalkForwardResult(
            folds=folds,
            mean_test_patterns=float(np.mean(test_patterns)) if test_patterns else 0.0,
            std_test_patterns=float(np.std(test_patterns)) if test_patterns else 0.0,
            mean_test_quality=float(np.mean(test_qualities)) if test_qualities else 0.0,
            std_test_quality=float(np.std(test_qualities)) if test_qualities else 0.0,
            mean_test_sharpe=float(np.mean(test_sharpes)) if test_sharpes else 0.0,
            mean_test_expectancy=float(np.mean(test_expectancies)) if test_expectancies else 0.0,
            mean_test_win_rate=float(np.mean(test_win_rates)) if test_win_rates else 0.0,
            consistency_ratio=float(consistency),
            stability_score=float(stability),
        )

    def _aggregate_cluster_results(
        self,
        cluster_results: Dict[str, AggregateResult],
        cluster_sim_results: Dict[str, SimulationResult],
    ) -> ClusterValidationResult:
        """Aggregate cluster validation results."""
        pattern_counts = [r.total_patterns for r in cluster_results.values()]
        clusters_with_patterns = sum(1 for c in pattern_counts if c > 0)

        # Generalization score: consistency across clusters
        if pattern_counts:
            mean_patterns = np.mean(pattern_counts)
            std_patterns = np.std(pattern_counts)
            cv = std_patterns / mean_patterns if mean_patterns > 0 else 1.0

            # Higher score = more consistent across clusters
            generalization = 1.0 / (1.0 + cv)
        else:
            generalization = 0.0

        return ClusterValidationResult(
            cluster_results=cluster_results,
            cluster_sim_results=cluster_sim_results,
            mean_patterns_per_cluster=float(np.mean(pattern_counts)) if pattern_counts else 0.0,
            std_patterns_per_cluster=float(np.std(pattern_counts)) if pattern_counts else 0.0,
            clusters_with_patterns=clusters_with_patterns,
            total_clusters=len(cluster_results),
            generalization_score=float(generalization),
        )

    def _aggregate_to_dict(self, result: AggregateResult) -> Dict[str, Any]:
        """Convert AggregateResult to dictionary for objective evaluation."""
        return {
            'total_patterns': result.total_patterns,
            'mean_score': result.mean_score,
            'unique_symbols': result.unique_symbols,
            'total_symbols': result.total_symbols,
            'tier_a_count': result.tier_a_count,
            'tier_b_count': result.tier_b_count,
            'tier_c_count': result.tier_c_count,
        }
