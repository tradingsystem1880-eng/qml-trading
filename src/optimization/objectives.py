"""
Objective Functions for Phase 7.7 Optimization
===============================================
Six objective functions for comprehensive strategy optimization:

1. CountQuality - Pattern count × quality (from Phase 7.6)
2. Sharpe - Risk-adjusted returns
3. Expectancy - Expected R per trade
4. ProfitFactor - Gross profit / gross loss
5. Composite - Weighted combination of all metrics
6. MaxDrawdown - Minimize maximum drawdown

Each objective can be run independently with 500 iterations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

from src.optimization.trade_simulator import SimulationResult


class ObjectiveType(str, Enum):
    """Available objective function types."""
    COUNT_QUALITY = "count_quality"
    SHARPE = "sharpe"
    EXPECTANCY = "expectancy"
    PROFIT_FACTOR = "profit_factor"
    COMPOSITE = "composite"
    MAX_DRAWDOWN = "max_drawdown"
    # Phase 7.9 - Profit-focused objectives
    PROFIT_FACTOR_PENALIZED = "profit_factor_penalized"
    EXPECTANCY_FOCUSED = "expectancy_focused"


@dataclass
class ObjectiveResult:
    """Result from evaluating an objective function."""
    score: float  # Negative for minimization (gp_minimize minimizes)
    raw_value: float  # Raw metric value before negation
    objective_type: ObjectiveType
    is_valid: bool = True
    penalty_reason: Optional[str] = None

    # Component scores for composite objective
    components: Dict[str, float] = None

    def __post_init__(self):
        if self.components is None:
            self.components = {}


@dataclass
class ObjectiveConfig:
    """Configuration for objective function evaluation."""
    # Minimum thresholds to avoid degenerate solutions
    min_trades: int = 30  # Need at least this many trades
    min_win_rate: float = 0.35  # Minimum 35% win rate
    min_symbols: int = 10  # Diversity requirement

    # Targets for normalization
    target_trades: int = 500
    target_sharpe: float = 1.5
    target_expectancy: float = 0.3  # 0.3R expected per trade
    target_profit_factor: float = 2.0
    target_quality: float = 0.7

    # Composite weights
    weight_count: float = 0.15
    weight_quality: float = 0.20
    weight_sharpe: float = 0.20
    weight_expectancy: float = 0.20
    weight_profit_factor: float = 0.15
    weight_drawdown: float = 0.10

    # Penalties
    penalty_insufficient_trades: float = 1000.0
    penalty_low_win_rate: float = 500.0
    penalty_low_diversity: float = 300.0


class ObjectiveFunction(ABC):
    """Abstract base class for objective functions."""

    def __init__(self, config: Optional[ObjectiveConfig] = None):
        self.config = config or ObjectiveConfig()

    @property
    @abstractmethod
    def objective_type(self) -> ObjectiveType:
        """Return the type of this objective."""
        pass

    @abstractmethod
    def evaluate(
        self,
        sim_result: SimulationResult,
        detection_result: Dict[str, Any],
    ) -> ObjectiveResult:
        """
        Evaluate the objective function.

        Args:
            sim_result: Trade simulation results
            detection_result: Pattern detection results with keys:
                - total_patterns: int
                - mean_score: float
                - unique_symbols: int
                - total_symbols: int
                - tier_a_count: int
                - tier_b_count: int
                - tier_c_count: int

        Returns:
            ObjectiveResult with score (negative for minimization)
        """
        pass

    def _check_validity(
        self,
        sim_result: SimulationResult,
        detection_result: Dict[str, Any],
    ) -> Tuple[bool, Optional[str], float]:
        """
        Check if results meet minimum validity thresholds.

        Returns:
            (is_valid, reason, penalty_score)
        """
        cfg = self.config

        # Check minimum trades
        if sim_result.total_trades < cfg.min_trades:
            return False, f"insufficient_trades_{sim_result.total_trades}", cfg.penalty_insufficient_trades

        # Check win rate
        if sim_result.win_rate < cfg.min_win_rate:
            return False, f"low_win_rate_{sim_result.win_rate:.2%}", cfg.penalty_low_win_rate

        # Check symbol diversity
        unique_symbols = detection_result.get('unique_symbols', 0)
        if unique_symbols < cfg.min_symbols:
            return False, f"low_diversity_{unique_symbols}", cfg.penalty_low_diversity

        return True, None, 0.0


class CountQualityObjective(ObjectiveFunction):
    """
    Objective: Maximize pattern count × mean quality.

    This is the original Phase 7.6 objective.
    Balances finding many patterns with maintaining quality.
    """

    @property
    def objective_type(self) -> ObjectiveType:
        return ObjectiveType.COUNT_QUALITY

    def evaluate(
        self,
        sim_result: SimulationResult,
        detection_result: Dict[str, Any],
    ) -> ObjectiveResult:
        is_valid, reason, penalty = self._check_validity(sim_result, detection_result)

        if not is_valid:
            return ObjectiveResult(
                score=penalty,
                raw_value=0.0,
                objective_type=self.objective_type,
                is_valid=False,
                penalty_reason=reason,
            )

        cfg = self.config
        total_patterns = detection_result.get('total_patterns', 0)
        mean_quality = detection_result.get('mean_score', 0.0)
        unique_symbols = detection_result.get('unique_symbols', 0)
        total_symbols = detection_result.get('total_symbols', 1)

        # Normalize components
        count_score = min(total_patterns, cfg.target_trades) / cfg.target_trades
        quality_score = mean_quality / cfg.target_quality
        diversity_score = unique_symbols / total_symbols

        # Combined score
        raw_value = (
            0.3 * count_score +
            0.5 * quality_score +
            0.2 * diversity_score
        )

        return ObjectiveResult(
            score=-raw_value,  # Negative for minimization
            raw_value=raw_value,
            objective_type=self.objective_type,
            components={
                'count_score': count_score,
                'quality_score': quality_score,
                'diversity_score': diversity_score,
            },
        )


class SharpeObjective(ObjectiveFunction):
    """
    Objective: Maximize Sharpe ratio.

    Risk-adjusted returns: mean(R) / std(R)
    Penalizes volatile strategies even if profitable.
    """

    @property
    def objective_type(self) -> ObjectiveType:
        return ObjectiveType.SHARPE

    def evaluate(
        self,
        sim_result: SimulationResult,
        detection_result: Dict[str, Any],
    ) -> ObjectiveResult:
        is_valid, reason, penalty = self._check_validity(sim_result, detection_result)

        if not is_valid:
            return ObjectiveResult(
                score=penalty,
                raw_value=0.0,
                objective_type=self.objective_type,
                is_valid=False,
                penalty_reason=reason,
            )

        sharpe = sim_result.sharpe_ratio

        # Normalize to 0-1 range (target Sharpe = 1.5)
        cfg = self.config
        normalized = min(sharpe, cfg.target_sharpe * 2) / (cfg.target_sharpe * 2)

        return ObjectiveResult(
            score=-normalized,  # Negative for minimization
            raw_value=sharpe,
            objective_type=self.objective_type,
            components={
                'sharpe_ratio': sharpe,
                'normalized_sharpe': normalized,
            },
        )


class ExpectancyObjective(ObjectiveFunction):
    """
    Objective: Maximize expectancy (expected R per trade).

    Expectancy = (Win% × AvgWin) - (Loss% × AvgLoss)
    Measures long-term edge.
    """

    @property
    def objective_type(self) -> ObjectiveType:
        return ObjectiveType.EXPECTANCY

    def evaluate(
        self,
        sim_result: SimulationResult,
        detection_result: Dict[str, Any],
    ) -> ObjectiveResult:
        is_valid, reason, penalty = self._check_validity(sim_result, detection_result)

        if not is_valid:
            return ObjectiveResult(
                score=penalty,
                raw_value=0.0,
                objective_type=self.objective_type,
                is_valid=False,
                penalty_reason=reason,
            )

        expectancy = sim_result.expectancy_r

        # Normalize (target expectancy = 0.3R per trade)
        cfg = self.config
        # Scale expectancy to 0-1 range, allowing up to 2x target
        normalized = min(max(expectancy, 0), cfg.target_expectancy * 2) / (cfg.target_expectancy * 2)

        return ObjectiveResult(
            score=-normalized,  # Negative for minimization
            raw_value=expectancy,
            objective_type=self.objective_type,
            components={
                'expectancy_r': expectancy,
                'win_rate': sim_result.win_rate,
                'avg_win_r': sim_result.avg_win_r,
                'avg_loss_r': sim_result.avg_loss_r,
            },
        )


class ProfitFactorObjective(ObjectiveFunction):
    """
    Objective: Maximize profit factor.

    Profit Factor = Gross Profit / Gross Loss
    Values > 1.5 indicate robust edge.
    """

    @property
    def objective_type(self) -> ObjectiveType:
        return ObjectiveType.PROFIT_FACTOR

    def evaluate(
        self,
        sim_result: SimulationResult,
        detection_result: Dict[str, Any],
    ) -> ObjectiveResult:
        is_valid, reason, penalty = self._check_validity(sim_result, detection_result)

        if not is_valid:
            return ObjectiveResult(
                score=penalty,
                raw_value=0.0,
                objective_type=self.objective_type,
                is_valid=False,
                penalty_reason=reason,
            )

        pf = sim_result.profit_factor

        # Cap at reasonable value
        pf_capped = min(pf, 10.0)

        # Normalize (target PF = 2.0, max considered = 5.0)
        cfg = self.config
        normalized = min(pf_capped, cfg.target_profit_factor * 2.5) / (cfg.target_profit_factor * 2.5)

        return ObjectiveResult(
            score=-normalized,
            raw_value=pf,
            objective_type=self.objective_type,
            components={
                'profit_factor': pf,
                'profit_factor_capped': pf_capped,
            },
        )


class MaxDrawdownObjective(ObjectiveFunction):
    """
    Objective: Minimize maximum drawdown.

    Lower drawdown = more consistent, easier to trade psychologically.
    """

    @property
    def objective_type(self) -> ObjectiveType:
        return ObjectiveType.MAX_DRAWDOWN

    def evaluate(
        self,
        sim_result: SimulationResult,
        detection_result: Dict[str, Any],
    ) -> ObjectiveResult:
        is_valid, reason, penalty = self._check_validity(sim_result, detection_result)

        if not is_valid:
            return ObjectiveResult(
                score=penalty,
                raw_value=0.0,
                objective_type=self.objective_type,
                is_valid=False,
                penalty_reason=reason,
            )

        max_dd = sim_result.max_drawdown_r

        # We want to MINIMIZE drawdown, so lower is better
        # Convert to a score where lower drawdown = higher score
        # Assume max acceptable DD is 10R, normalize inversely
        max_acceptable_dd = 10.0
        dd_capped = min(max_dd, max_acceptable_dd)

        # Invert: 0 drawdown = 1.0, max_acceptable = 0.0
        normalized = 1.0 - (dd_capped / max_acceptable_dd)

        return ObjectiveResult(
            score=-normalized,  # Negative for minimization (so lower DD = lower score)
            raw_value=max_dd,
            objective_type=self.objective_type,
            components={
                'max_drawdown_r': max_dd,
                'drawdown_score': normalized,
            },
        )


class CompositeObjective(ObjectiveFunction):
    """
    Objective: Weighted combination of all metrics.

    Balances multiple objectives:
    - Pattern count and quality
    - Sharpe ratio
    - Expectancy
    - Profit factor
    - Drawdown

    This is the recommended objective for final optimization.
    """

    @property
    def objective_type(self) -> ObjectiveType:
        return ObjectiveType.COMPOSITE

    def evaluate(
        self,
        sim_result: SimulationResult,
        detection_result: Dict[str, Any],
    ) -> ObjectiveResult:
        is_valid, reason, penalty = self._check_validity(sim_result, detection_result)

        if not is_valid:
            return ObjectiveResult(
                score=penalty,
                raw_value=0.0,
                objective_type=self.objective_type,
                is_valid=False,
                penalty_reason=reason,
            )

        cfg = self.config

        # Calculate component scores (all 0-1 range)
        # Count score
        total_patterns = detection_result.get('total_patterns', 0)
        count_score = min(total_patterns, cfg.target_trades) / cfg.target_trades

        # Quality score
        mean_quality = detection_result.get('mean_score', 0.0)
        quality_score = mean_quality / cfg.target_quality

        # Sharpe score
        sharpe = sim_result.sharpe_ratio
        sharpe_score = min(max(sharpe, 0), cfg.target_sharpe * 2) / (cfg.target_sharpe * 2)

        # Expectancy score
        expectancy = sim_result.expectancy_r
        expectancy_score = min(max(expectancy, 0), cfg.target_expectancy * 2) / (cfg.target_expectancy * 2)

        # Profit factor score
        pf = min(sim_result.profit_factor, 10.0)
        pf_score = min(pf, cfg.target_profit_factor * 2.5) / (cfg.target_profit_factor * 2.5)

        # Drawdown score (inverted)
        max_dd = sim_result.max_drawdown_r
        dd_capped = min(max_dd, 10.0)
        dd_score = 1.0 - (dd_capped / 10.0)

        # Weighted combination
        composite = (
            cfg.weight_count * count_score +
            cfg.weight_quality * quality_score +
            cfg.weight_sharpe * sharpe_score +
            cfg.weight_expectancy * expectancy_score +
            cfg.weight_profit_factor * pf_score +
            cfg.weight_drawdown * dd_score
        )

        return ObjectiveResult(
            score=-composite,
            raw_value=composite,
            objective_type=self.objective_type,
            components={
                'count_score': count_score,
                'quality_score': quality_score,
                'sharpe_score': sharpe_score,
                'expectancy_score': expectancy_score,
                'profit_factor_score': pf_score,
                'drawdown_score': dd_score,
                'total_patterns': total_patterns,
                'mean_quality': mean_quality,
                'sharpe': sharpe,
                'expectancy': expectancy,
                'profit_factor': pf,
                'max_drawdown': max_dd,
            },
        )


class ProfitFactorPenalizedObjective(ObjectiveFunction):
    """
    Profit-focused objective with hard constraints (Phase 7.9).

    Academic basis:
    - Lexicographic optimization (constraints before optimization)
    - Constrained Sharpe maximization (Boyd et al. 2017)
    - Minimum sample requirements (n > 100 for statistical validity)

    Key difference from basic ProfitFactorObjective:
    - Uses separate if statements (not elif) for hard gates
    - Applies multiplicative soft penalties, not additive
    - Primary focus on profit factor, not pattern quality
    """

    @property
    def objective_type(self) -> ObjectiveType:
        return ObjectiveType.PROFIT_FACTOR_PENALIZED

    def evaluate(
        self,
        sim_result: SimulationResult,
        detection_result: Dict[str, Any],
    ) -> ObjectiveResult:
        # Stage 1: Hard viability gates (ALL checked, not short-circuited)
        rejection_penalty = 0
        penalty_reasons = []

        total_patterns = detection_result.get('total_patterns', 0)
        if total_patterns < 400:
            rejection_penalty += 1e9
            penalty_reasons.append(f"insufficient_patterns_{total_patterns}")

        pf = sim_result.profit_factor
        if pf < 0.7:
            rejection_penalty += 1e9
            penalty_reasons.append(f"unprofitable_pf_{pf:.2f}")

        # Use percentage-based max drawdown for prop firm compatibility
        # Approximate: max_drawdown_r / total_trades gives rough % if risking 1% per trade
        max_dd_pct = sim_result.max_drawdown_r / max(sim_result.total_trades, 1) * 100
        if max_dd_pct > 20:
            rejection_penalty += 1e9
            penalty_reasons.append(f"high_drawdown_{max_dd_pct:.1f}%")

        if rejection_penalty > 0:
            return ObjectiveResult(
                score=rejection_penalty,  # Large positive = bad for minimizer
                raw_value=0.0,
                objective_type=self.objective_type,
                is_valid=False,
                penalty_reason="; ".join(penalty_reasons),
            )

        # Stage 2: Primary objective is profit factor
        base_score = pf

        # Stage 3: Soft penalties/bonuses (multiplicative, not additive)
        multiplier = 1.0

        # Penalty for low win rate (but don't over-penalize - low WR with high R:R is valid)
        if sim_result.win_rate < 0.40:
            multiplier *= 0.85

        # Bonus for positive Sharpe (but PF remains primary)
        sharpe = sim_result.sharpe_ratio
        if sharpe > 0:
            multiplier *= (1 + 0.1 * min(sharpe, 1.0))

        # Penalty for excessive drawdown (sliding scale between 10-20%)
        if max_dd_pct > 10:
            dd_penalty = 10 / max(max_dd_pct, 10.01)
            multiplier *= dd_penalty

        # Bonus for good expectancy (validates the edge is real)
        expectancy = sim_result.expectancy_r
        if expectancy > 0.1:
            multiplier *= 1.1

        final_score = base_score * multiplier

        return ObjectiveResult(
            score=-final_score,  # Negative for minimization (higher is better)
            raw_value=final_score,
            objective_type=self.objective_type,
            components={
                'profit_factor': pf,
                'base_score': base_score,
                'multiplier': multiplier,
                'sharpe': sharpe,
                'expectancy': expectancy,
                'win_rate': sim_result.win_rate,
                'max_dd_pct': max_dd_pct,
            },
        )


class ExpectancyFocusedObjective(ObjectiveFunction):
    """
    Optimize for expected R per trade (Phase 7.9).

    E = (Win% × AvgWin_R) - (Loss% × AvgLoss_R)

    This directly measures edge per trade - the purest profitability metric.

    Key differences from basic ExpectancyObjective:
    - Uses separate if statements for hard gates
    - Scales by statistical confidence (more trades = more confidence)
    - Validates expectancy with profit factor check
    """

    @property
    def objective_type(self) -> ObjectiveType:
        return ObjectiveType.EXPECTANCY_FOCUSED

    def evaluate(
        self,
        sim_result: SimulationResult,
        detection_result: Dict[str, Any],
    ) -> ObjectiveResult:
        # Hard gates (separate if statements, not elif)
        rejection_penalty = 0
        penalty_reasons = []

        total_patterns = detection_result.get('total_patterns', 0)
        if total_patterns < 400:
            rejection_penalty += 1e9
            penalty_reasons.append(f"insufficient_patterns_{total_patterns}")

        # Use percentage-based max drawdown
        max_dd_pct = sim_result.max_drawdown_r / max(sim_result.total_trades, 1) * 100
        if max_dd_pct > 20:
            rejection_penalty += 1e9
            penalty_reasons.append(f"high_drawdown_{max_dd_pct:.1f}%")

        if rejection_penalty > 0:
            return ObjectiveResult(
                score=rejection_penalty,
                raw_value=0.0,
                objective_type=self.objective_type,
                is_valid=False,
                penalty_reason="; ".join(penalty_reasons),
            )

        # Calculate expectancy in R-multiples
        win_rate = sim_result.win_rate
        avg_win_r = sim_result.avg_win_r if sim_result.avg_win_r else 1.5
        avg_loss_r = sim_result.avg_loss_r if sim_result.avg_loss_r else 1.0

        expectancy = (win_rate * avg_win_r) - ((1 - win_rate) * avg_loss_r)

        # Must be positive to be viable
        if expectancy <= 0:
            return ObjectiveResult(
                score=1e6,  # Not as bad as hard gate failure, but still rejected
                raw_value=expectancy,
                objective_type=self.objective_type,
                is_valid=False,
                penalty_reason=f"negative_expectancy_{expectancy:.3f}",
            )

        # Scale by statistical confidence (more trades = more confidence)
        confidence_factor = min(total_patterns / 600, 1.0)

        # Bonus for reasonable profit factor (confirms expectancy isn't from outliers)
        pf_bonus = 1.0
        if sim_result.profit_factor > 1.2:
            pf_bonus = 1.15

        final_score = expectancy * confidence_factor * pf_bonus

        return ObjectiveResult(
            score=-final_score,  # Negative for minimization
            raw_value=final_score,
            objective_type=self.objective_type,
            components={
                'expectancy_r': expectancy,
                'win_rate': win_rate,
                'avg_win_r': avg_win_r,
                'avg_loss_r': avg_loss_r,
                'confidence_factor': confidence_factor,
                'pf_bonus': pf_bonus,
                'profit_factor': sim_result.profit_factor,
            },
        )


def create_objective(
    objective_type: ObjectiveType,
    config: Optional[ObjectiveConfig] = None,
) -> ObjectiveFunction:
    """
    Factory function to create objective instances.

    Args:
        objective_type: Type of objective to create
        config: Optional configuration

    Returns:
        ObjectiveFunction instance
    """
    objectives = {
        ObjectiveType.COUNT_QUALITY: CountQualityObjective,
        ObjectiveType.SHARPE: SharpeObjective,
        ObjectiveType.EXPECTANCY: ExpectancyObjective,
        ObjectiveType.PROFIT_FACTOR: ProfitFactorObjective,
        ObjectiveType.MAX_DRAWDOWN: MaxDrawdownObjective,
        ObjectiveType.COMPOSITE: CompositeObjective,
        # Phase 7.9 - Profit-focused objectives
        ObjectiveType.PROFIT_FACTOR_PENALIZED: ProfitFactorPenalizedObjective,
        ObjectiveType.EXPECTANCY_FOCUSED: ExpectancyFocusedObjective,
    }

    if objective_type not in objectives:
        raise ValueError(f"Unknown objective type: {objective_type}")

    return objectives[objective_type](config)


def get_all_objective_types() -> List[ObjectiveType]:
    """Return list of all available objective types."""
    return list(ObjectiveType)
