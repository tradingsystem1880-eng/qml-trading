"""
Parallel Detection Runner
=========================
Runs pattern detection across multiple symbols in parallel using joblib.

Designed for rapid evaluation of parameter combinations during optimization.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import time

import pandas as pd
import numpy as np

try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

from src.data_engine import load_master_data, normalize_symbol, get_symbol_data_dir
from src.detection.hierarchical_swing import HierarchicalSwingDetector, HierarchicalSwingConfig
from src.detection.pattern_validator import PatternValidator, PatternValidationConfig
from src.detection.trend_validator import TrendValidator, TrendValidationConfig
from src.detection.pattern_scorer import PatternScorer, PatternScoringConfig
from src.detection.regime import MarketRegimeDetector
from src.detection.config import DetectionConfig

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent


@dataclass
class PatternInfo:
    """Minimal pattern info for trade simulation."""
    bar_idx: int  # P5 bar index (entry point)
    direction: str  # 'LONG' or 'SHORT'
    score: float
    atr: float
    entry_price: float


@dataclass
class DetectionResult:
    """Result of running detection on a single symbol/timeframe."""
    symbol: str
    timeframe: str
    num_patterns: int
    num_swings: int
    scores: List[float] = field(default_factory=list)
    mean_score: float = 0.0
    tier_a: int = 0
    tier_b: int = 0
    tier_c: int = 0
    rejected: int = 0
    duration_ms: float = 0.0
    error: Optional[str] = None
    # Pattern metadata for trade simulation (Phase 7.7)
    patterns: List[PatternInfo] = field(default_factory=list)


@dataclass
class AggregateResult:
    """Aggregated result across all symbols."""
    total_patterns: int
    unique_symbols: int
    total_symbols: int
    mean_score: float
    median_score: float
    tier_a_count: int
    tier_b_count: int
    tier_c_count: int
    symbol_results: Dict[str, DetectionResult] = field(default_factory=dict)
    duration_ms: float = 0.0


class ParallelDetectionRunner:
    """
    Runs pattern detection across multiple symbols in parallel.

    Designed for rapid evaluation during Bayesian optimization.
    Uses joblib for parallelization when available.
    """

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        timeframes: Optional[List[str]] = None,
        n_jobs: int = -1,  # Use all cores
    ):
        """
        Initialize the parallel runner.

        Args:
            symbols: List of symbols to process (normalized format like 'BTCUSDT')
            timeframes: List of timeframes to process
            n_jobs: Number of parallel jobs (-1 = all cores)
        """
        self.symbols = symbols or self._get_available_symbols()
        self.timeframes = timeframes or ['4h']
        self.n_jobs = n_jobs

        # Preload data for faster iteration
        self._data_cache: Dict[str, pd.DataFrame] = {}

    def _get_available_symbols(self) -> List[str]:
        """Get list of symbols that have data available."""
        data_dir = PROJECT_ROOT / "data" / "processed"

        if not data_dir.exists():
            return []

        symbols = []
        for path in data_dir.iterdir():
            if path.is_dir() and not path.name.startswith('.'):
                # Check if it has parquet files
                if list(path.glob("*_master.parquet")):
                    symbols.append(path.name)

        return sorted(symbols)

    def preload_data(self):
        """Preload all data into memory for faster iteration."""
        for symbol in self.symbols:
            for tf in self.timeframes:
                cache_key = f"{symbol}_{tf}"
                if cache_key not in self._data_cache:
                    try:
                        df = load_master_data(tf, symbol=symbol)
                        # Normalize column names
                        df.columns = [c.lower() for c in df.columns]
                        self._data_cache[cache_key] = df
                    except Exception:
                        pass  # Skip unavailable data

    def run(
        self,
        swing_config: Optional[HierarchicalSwingConfig] = None,
        validation_config: Optional[PatternValidationConfig] = None,
        trend_config: Optional[TrendValidationConfig] = None,
        scoring_config: Optional[PatternScoringConfig] = None,
    ) -> AggregateResult:
        """
        Run detection with given parameters across all symbols.

        Args:
            swing_config: Hierarchical swing detection config
            validation_config: Pattern validation config
            trend_config: Trend validation config
            scoring_config: Pattern scoring config

        Returns:
            AggregateResult with combined statistics
        """
        start_time = time.time()

        # Prepare configs
        swing_config = swing_config or HierarchicalSwingConfig()
        validation_config = validation_config or PatternValidationConfig()
        trend_config = trend_config or TrendValidationConfig()
        scoring_config = scoring_config or PatternScoringConfig()

        # Build tasks
        tasks = []
        for symbol in self.symbols:
            for tf in self.timeframes:
                tasks.append((symbol, tf))

        # Run detection
        if JOBLIB_AVAILABLE and len(tasks) > 1:
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self._detect_single)(
                    symbol, tf,
                    swing_config, validation_config, trend_config, scoring_config
                )
                for symbol, tf in tasks
            )
        else:
            results = [
                self._detect_single(
                    symbol, tf,
                    swing_config, validation_config, trend_config, scoring_config
                )
                for symbol, tf in tasks
            ]

        # Aggregate results
        return self._aggregate_results(results, time.time() - start_time)

    def _detect_single(
        self,
        symbol: str,
        timeframe: str,
        swing_config: HierarchicalSwingConfig,
        validation_config: PatternValidationConfig,
        trend_config: TrendValidationConfig,
        scoring_config: PatternScoringConfig,
    ) -> DetectionResult:
        """Run detection on a single symbol/timeframe."""
        start = time.time()

        try:
            # Get data
            cache_key = f"{symbol}_{timeframe}"
            if cache_key in self._data_cache:
                df = self._data_cache[cache_key]
            else:
                df = load_master_data(timeframe, symbol=symbol)
                df.columns = [c.lower() for c in df.columns]

            # Initialize components
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

            # Filter by trend validation and score
            valid_patterns = []
            scores = []
            pattern_infos = []
            tier_counts = {'A': 0, 'B': 0, 'C': 0, 'REJECT': 0}

            # Get ATR values
            atr_col = df['atr'] if 'atr' in df.columns else df.get('ATR', pd.Series([1.0]*len(df)))

            for pattern in patterns:
                if not pattern.is_valid:
                    continue

                # Validate prior trend
                trend_result = trend_validator.validate(
                    swings, pattern.p1.bar_index, df, pattern.direction
                )

                if not trend_result.is_valid:
                    continue

                # Score the pattern (pass df, trend_result, and regime_result for Phase 7.6/7.8)
                score_result = scorer.score(
                    pattern,
                    df=df,
                    trend_result=trend_result,
                    regime_result=regime_result,
                )

                valid_patterns.append(pattern)
                scores.append(score_result.total_score)
                tier_counts[score_result.tier.value] += 1

                # Store pattern info for trade simulation (Phase 7.7)
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
                tier_b=tier_counts['B'],
                tier_c=tier_counts['C'],
                rejected=tier_counts['REJECT'],
                duration_ms=(time.time() - start) * 1000,
                patterns=pattern_infos,
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

    def _aggregate_results(
        self,
        results: List[DetectionResult],
        duration: float
    ) -> AggregateResult:
        """Aggregate individual results into summary statistics."""
        symbol_results = {f"{r.symbol}_{r.timeframe}": r for r in results}

        # Collect all scores
        all_scores = []
        for r in results:
            all_scores.extend(r.scores)

        # Count symbols with patterns
        symbols_with_patterns = set()
        for r in results:
            if r.num_patterns > 0:
                symbols_with_patterns.add(r.symbol)

        # Sum tiers
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
            duration_ms=duration * 1000,
        )

    def run_with_dict(self, params: Dict[str, Any]) -> AggregateResult:
        """
        Run detection with parameters provided as a dictionary.

        Useful for optimization where parameters come from the optimizer.

        Args:
            params: Dictionary with parameter names and values

        Returns:
            AggregateResult
        """
        # Parse parameters into configs
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

        # Get weights from params (now 8 components, all optimizable except shoulder)
        head_weight = params.get('head_extension_weight', 0.22)
        bos_weight = params.get('bos_efficiency_weight', 0.18)
        volume_weight = params.get('volume_spike_weight', 0.10)
        path_weight = params.get('path_efficiency_weight', 0.10)
        trend_weight = params.get('trend_strength_weight', 0.10)
        regime_weight = params.get('regime_suitability_weight', 0.10)

        # Fixed weights
        shoulder_weight = 0.12

        # Adjust swing weight to ensure sum = 1.0
        fixed_sum = shoulder_weight
        swing_weight = 1.0 - head_weight - bos_weight - volume_weight - path_weight - trend_weight - regime_weight - fixed_sum

        # Ensure swing_weight stays positive
        swing_weight = max(0.02, swing_weight)

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

        return self.run(
            swing_config=swing_config,
            validation_config=validation_config,
            trend_config=trend_config,
            scoring_config=scoring_config,
        )
