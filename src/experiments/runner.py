"""
Experiment Runner for A/B Testing - Phase 6
============================================
Automated backtesting across parameter combinations.

Features:
- ExperimentResult: Single experiment output
- ExperimentRunner: Orchestrates grid search with callbacks
- Integrates with validation service (Phase 4)
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Any, Callable, Dict, List, Optional
import time
import traceback

from .parameter_grid import ParameterSet, GridSearchConfig, ParameterGridManager
from src.data.sqlite_manager import SQLiteManager


@dataclass
class ExperimentResult:
    """
    Result of a single backtesting experiment.

    Contains both raw metrics and statistical analysis.
    """
    # Parameters tested
    params: ParameterSet
    param_hash: str

    # Core metrics
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    total_return: float = 0.0

    # Additional metrics
    avg_r_multiple: float = 0.0
    expectancy: float = 0.0
    sortino_ratio: float = 0.0

    # Validation (from Phase 4 ValidationService)
    validation_passed: Optional[bool] = None
    pbo_score: Optional[float] = None
    walk_forward_efficiency: Optional[float] = None
    prop_firm_pass_rate: Optional[float] = None

    # Statistical significance
    is_significant: bool = False
    p_value: Optional[float] = None

    # Metadata
    symbol: str = ''
    timeframe: str = ''
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    duration_seconds: float = 0.0
    completed_at: datetime = field(default_factory=datetime.now)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'params': self.params.to_dict(),
            'param_hash': self.param_hash,
            'total_trades': self.total_trades,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'total_return': self.total_return,
            'avg_r_multiple': self.avg_r_multiple,
            'expectancy': self.expectancy,
            'sortino_ratio': self.sortino_ratio,
            'validation_passed': self.validation_passed,
            'pbo_score': self.pbo_score,
            'walk_forward_efficiency': self.walk_forward_efficiency,
            'prop_firm_pass_rate': self.prop_firm_pass_rate,
            'is_significant': self.is_significant,
            'p_value': self.p_value,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'start_date': self.start_date.isoformat() if self.start_date else None,
            'end_date': self.end_date.isoformat() if self.end_date else None,
            'duration_seconds': self.duration_seconds,
            'completed_at': self.completed_at.isoformat(),
            'error': self.error,
        }


class ExperimentRunner:
    """
    Runs backtesting experiments across parameter grid.

    Integrates with:
    - SQLiteManager for result storage
    - ValidationService (Phase 4) for statistical validation
    - ParameterGridManager for deduplication

    Usage:
        def my_backtest(params: ParameterSet, symbol: str, tf: str,
                        start: date, end: date) -> Dict:
            # Run backtest and return metrics dict
            return {'total_trades': 50, 'win_rate': 0.55, ...}

        runner = ExperimentRunner(db, backtest_func=my_backtest)

        # Run single experiment
        result = runner.run_single(
            params=ParameterSet(swing_lookback=5),
            symbol='BTCUSDT',
            timeframe='4h',
            start_date=date(2024, 1, 1),
            end_date=date(2024, 6, 30)
        )

        # Run grid search
        results = runner.run_grid(
            config=GridSearchConfig.small(),
            symbol='BTCUSDT',
            timeframe='4h',
            start_date=date(2024, 1, 1),
            end_date=date(2024, 6, 30),
            max_experiments=100
        )
    """

    def __init__(
        self,
        db: SQLiteManager,
        backtest_func: Optional[Callable] = None,
        validation_service: Optional[Any] = None,
        on_progress: Optional[Callable[[int, int, ExperimentResult], None]] = None,
        on_complete: Optional[Callable[[List[ExperimentResult]], None]] = None,
    ):
        """
        Initialize experiment runner.

        Args:
            db: SQLiteManager for result storage
            backtest_func: Function(params, symbol, tf, start, end) -> Dict
            validation_service: Optional ValidationService from Phase 4
            on_progress: Callback(current, total, result) for progress updates
            on_complete: Callback(results) when grid search completes
        """
        self.db = db
        self.backtest_func = backtest_func
        self.validation_service = validation_service
        self.on_progress = on_progress
        self.on_complete = on_complete
        self.grid_manager = ParameterGridManager(db)

    def run_single(
        self,
        params: ParameterSet,
        symbol: str,
        timeframe: str,
        start_date: date,
        end_date: date,
        run_validation: bool = False,
    ) -> ExperimentResult:
        """
        Run a single backtesting experiment.

        Args:
            params: Parameter configuration to test
            symbol: Trading symbol (e.g., 'BTCUSDT')
            timeframe: Timeframe (e.g., '4h')
            start_date: Backtest start date
            end_date: Backtest end date
            run_validation: Whether to run Phase 4 validation

        Returns:
            ExperimentResult with all metrics
        """
        start_time = time.time()
        param_hash = params.to_hash()

        result = ExperimentResult(
            params=params,
            param_hash=param_hash,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
        )

        try:
            if self.backtest_func is None:
                raise ValueError("No backtest function provided")

            # Run backtest
            metrics = self.backtest_func(
                params=params,
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
            )

            # Extract metrics
            result.total_trades = metrics.get('total_trades', 0)
            result.win_rate = metrics.get('win_rate', 0.0)
            result.profit_factor = metrics.get('profit_factor', 0.0)
            result.sharpe_ratio = metrics.get('sharpe_ratio', 0.0)
            result.max_drawdown = metrics.get('max_drawdown', 0.0)
            result.total_return = metrics.get('total_return', 0.0)
            result.avg_r_multiple = metrics.get('avg_r_multiple', 0.0)
            result.expectancy = metrics.get('expectancy', 0.0)
            result.sortino_ratio = metrics.get('sortino_ratio', 0.0)

            # Run validation if requested
            if run_validation and self.validation_service and result.total_trades >= 30:
                validation = self._run_validation(metrics)
                result.validation_passed = validation.get('passed', None)
                result.pbo_score = validation.get('pbo_score', None)
                result.walk_forward_efficiency = validation.get('walk_forward_efficiency', None)
                result.prop_firm_pass_rate = validation.get('prop_firm_pass_rate', None)

            # Register params as tested
            self.grid_manager.mark_tested(params)

            # Update best params if sharpe is good
            if result.sharpe_ratio and result.sharpe_ratio > 0:
                self.db.update_best_params(
                    param_hash=param_hash,
                    sharpe=result.sharpe_ratio,
                    experiment_id=f"exp_{param_hash}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                )

        except Exception as e:
            result.error = f"{type(e).__name__}: {str(e)}"
            # Still mark as tested to avoid re-running broken configs
            self.grid_manager.mark_tested(params)

        result.duration_seconds = time.time() - start_time
        result.completed_at = datetime.now()

        return result

    def run_grid(
        self,
        config: GridSearchConfig,
        symbol: str,
        timeframe: str,
        start_date: date,
        end_date: date,
        max_experiments: Optional[int] = None,
        run_validation: bool = False,
        skip_tested: bool = True,
    ) -> List[ExperimentResult]:
        """
        Run grid search across parameter combinations.

        Args:
            config: GridSearchConfig defining search space
            symbol: Trading symbol
            timeframe: Timeframe
            start_date: Backtest start date
            end_date: Backtest end date
            max_experiments: Max experiments to run (None = all)
            run_validation: Whether to run Phase 4 validation
            skip_tested: Skip already-tested parameters

        Returns:
            List of ExperimentResult
        """
        results = []
        total = config.total_combinations() if not skip_tested else None

        # Get parameter iterator
        if skip_tested:
            param_iter = self.grid_manager.get_untested(config, limit=max_experiments)
        else:
            param_iter = self.grid_manager.generate_grid(config)

        count = 0
        for params in param_iter:
            if max_experiments and count >= max_experiments:
                break

            result = self.run_single(
                params=params,
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                run_validation=run_validation,
            )
            results.append(result)
            count += 1

            # Progress callback
            if self.on_progress:
                self.on_progress(count, max_experiments or count, result)

        # Complete callback
        if self.on_complete:
            self.on_complete(results)

        return results

    def _run_validation(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run Phase 4 validation on backtest results.

        Args:
            metrics: Backtest metrics including trades list

        Returns:
            Dict with validation results
        """
        if not self.validation_service:
            return {}

        try:
            trades = metrics.get('trades', [])
            equity_curve = metrics.get('equity_curve', [])

            if not trades:
                return {'passed': None}

            # Run validation service
            validation_result = self.validation_service.run_quick_validation(
                trades=trades,
                equity_curve=equity_curve
            )

            return {
                'passed': validation_result.passed if hasattr(validation_result, 'passed') else None,
                'pbo_score': validation_result.pbo if hasattr(validation_result, 'pbo') else None,
                'walk_forward_efficiency': validation_result.walk_forward_efficiency if hasattr(validation_result, 'walk_forward_efficiency') else None,
                'prop_firm_pass_rate': validation_result.prop_firm_pass_rate if hasattr(validation_result, 'prop_firm_pass_rate') else None,
            }
        except Exception as e:
            return {'passed': None, 'error': str(e)}

    def get_best_results(
        self,
        results: List[ExperimentResult],
        metric: str = 'sharpe_ratio',
        min_trades: int = 30,
        top_n: int = 10,
    ) -> List[ExperimentResult]:
        """
        Get top performing results sorted by metric.

        Args:
            results: List of experiment results
            metric: Metric to sort by
            min_trades: Minimum trades required
            top_n: Number of top results to return

        Returns:
            Top N results sorted by metric (descending)
        """
        # Filter by minimum trades
        filtered = [r for r in results if r.total_trades >= min_trades and not r.error]

        # Sort by metric
        sorted_results = sorted(
            filtered,
            key=lambda r: getattr(r, metric, 0) or 0,
            reverse=True
        )

        return sorted_results[:top_n]

    def summarize_results(self, results: List[ExperimentResult]) -> Dict[str, Any]:
        """
        Generate summary statistics for experiment results.

        Args:
            results: List of experiment results

        Returns:
            Summary dict with aggregate metrics
        """
        if not results:
            return {'count': 0}

        successful = [r for r in results if not r.error]
        with_trades = [r for r in successful if r.total_trades >= 30]

        sharpes = [r.sharpe_ratio for r in with_trades if r.sharpe_ratio]
        win_rates = [r.win_rate for r in with_trades if r.win_rate]

        return {
            'count': len(results),
            'successful': len(successful),
            'with_min_trades': len(with_trades),
            'errors': len(results) - len(successful),
            'avg_sharpe': sum(sharpes) / len(sharpes) if sharpes else 0,
            'max_sharpe': max(sharpes) if sharpes else 0,
            'min_sharpe': min(sharpes) if sharpes else 0,
            'avg_win_rate': sum(win_rates) / len(win_rates) if win_rates else 0,
            'positive_sharpe_pct': len([s for s in sharpes if s > 0]) / len(sharpes) * 100 if sharpes else 0,
            'total_duration_seconds': sum(r.duration_seconds for r in results),
        }
