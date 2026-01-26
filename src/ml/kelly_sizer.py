"""
Kelly-Based Position Sizing for ML Meta-Labeling
================================================
Implements Kelly criterion position sizing adjusted by ML confidence.

Based on:
- Kelly (1956) - "A New Interpretation of Information Rate"
- Thorp (2008) - "The Kelly Criterion in Blackjack, Sports Betting and the Stock Market"

Uses ML confidence to scale position size:
- High confidence (>0.7): Full Kelly fraction
- Medium confidence (0.5-0.7): Half position
- Low confidence (<0.5): Minimum or skip
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np


@dataclass
class KellyConfig:
    """Configuration for Kelly position sizing."""
    # Kelly fraction (1.0 = full Kelly, 0.5 = half-Kelly)
    # Half-Kelly is standard for safety (reduces variance)
    kelly_fraction: float = 0.5

    # Position size bounds
    max_risk_pct: float = 0.02  # Max 2% risk per trade
    min_risk_pct: float = 0.005  # Min 0.5% risk per trade

    # Confidence thresholds
    high_confidence_threshold: float = 0.70  # Full position
    medium_confidence_threshold: float = 0.55  # Half position
    low_confidence_threshold: float = 0.45  # Minimum or skip

    # Skip trades below low threshold?
    skip_low_confidence: bool = False

    # Fallback to fixed sizing if Kelly is negative
    fallback_risk_pct: float = 0.01  # 1% fixed


@dataclass
class PositionSizeResult:
    """Result of position sizing calculation."""
    # ML confidence input
    ml_confidence: float

    # Kelly calculation
    base_kelly_fraction: float
    confidence_multiplier: float
    adjusted_kelly_fraction: float

    # Final sizing
    risk_pct: float  # Risk as % of equity
    risk_dollars: float  # Risk in dollars
    position_size_dollars: float  # Position size in dollars

    # Decision
    action: str  # 'FULL', 'HALF', 'MINIMUM', 'SKIP'
    should_skip: bool
    rationale: str


class KellySizer:
    """
    Kelly criterion position sizer with ML confidence adjustment.

    The Kelly formula: f* = (p * b - q) / b
    where:
        p = probability of winning
        q = probability of losing (1 - p)
        b = win/loss ratio (avg_win / avg_loss)

    ML confidence modulates how much of the Kelly fraction we use.
    """

    def __init__(self, config: Optional[KellyConfig] = None):
        """
        Initialize sizer.

        Args:
            config: Position sizing configuration
        """
        self.config = config or KellyConfig()

    def calculate_base_kelly(
        self,
        win_rate: float,
        avg_win_r: float,
        avg_loss_r: float,
    ) -> float:
        """
        Calculate base Kelly fraction from historical performance.

        Args:
            win_rate: Historical win rate (0-1)
            avg_win_r: Average R-multiple on wins
            avg_loss_r: Average R-multiple on losses (absolute value)

        Returns:
            Kelly fraction (can be negative if edge is negative)
        """
        if win_rate <= 0 or win_rate >= 1:
            return 0.0

        if avg_loss_r <= 0:
            return 0.0

        if avg_win_r <= 0:
            return 0.0  # No wins = no edge

        p = win_rate
        q = 1 - win_rate
        b = avg_win_r / avg_loss_r  # Win/loss ratio

        # Guard against edge case where b could be very small
        if b < 0.001:
            return 0.0

        # Kelly formula: f* = (p * b - q) / b
        kelly = (p * b - q) / b

        return kelly

    def calculate_position_size(
        self,
        ml_confidence: float,
        win_rate: float,
        avg_win_r: float,
        avg_loss_r: float,
        account_equity: float,
        stop_loss_pct: float = 0.01,  # 1% SL distance
    ) -> PositionSizeResult:
        """
        Calculate ML-adjusted position size.

        Args:
            ml_confidence: Model's predicted win probability (0-1)
            win_rate: Historical win rate
            avg_win_r: Average R-multiple on wins
            avg_loss_r: Average R-multiple on losses
            account_equity: Current account value
            stop_loss_pct: Stop loss distance as % of price

        Returns:
            PositionSizeResult with full sizing details
        """
        cfg = self.config

        # Step 1: Calculate base Kelly from historical performance
        base_kelly = self.calculate_base_kelly(win_rate, avg_win_r, avg_loss_r)

        # Step 2: Apply Kelly fraction (half-Kelly by default)
        kelly_adj = base_kelly * cfg.kelly_fraction

        # Step 3: Determine confidence tier and multiplier
        action, multiplier = self._classify_confidence(ml_confidence)

        # Step 4: Calculate final Kelly fraction
        if kelly_adj <= 0:
            # Negative Kelly = no edge, use fallback
            final_kelly = cfg.fallback_risk_pct if multiplier > 0 else 0
            rationale = f"Negative Kelly ({base_kelly:.3f}), using fallback"
        else:
            final_kelly = kelly_adj * multiplier
            rationale = f"{action}: {ml_confidence:.1%} confidence -> {multiplier:.0%} of Kelly"

        # Step 5: Clamp to bounds
        if action == 'SKIP' or (cfg.skip_low_confidence and ml_confidence < cfg.low_confidence_threshold):
            final_risk_pct = 0.0
            should_skip = True
            rationale = f"SKIP: {ml_confidence:.1%} confidence below threshold"
        else:
            final_risk_pct = max(min(final_kelly, cfg.max_risk_pct), cfg.min_risk_pct)
            should_skip = False

        # Step 6: Calculate dollar amounts
        risk_dollars = account_equity * final_risk_pct
        position_size = risk_dollars / stop_loss_pct if stop_loss_pct > 0 else 0

        return PositionSizeResult(
            ml_confidence=ml_confidence,
            base_kelly_fraction=base_kelly,
            confidence_multiplier=multiplier,
            adjusted_kelly_fraction=final_kelly,
            risk_pct=final_risk_pct,
            risk_dollars=risk_dollars,
            position_size_dollars=position_size,
            action=action,
            should_skip=should_skip,
            rationale=rationale,
        )

    def _classify_confidence(self, confidence: float) -> Tuple[str, float]:
        """
        Map ML confidence to action and Kelly multiplier.

        Returns:
            (action_name, kelly_multiplier)
        """
        cfg = self.config

        if confidence >= cfg.high_confidence_threshold:
            return 'FULL', 1.0
        elif confidence >= cfg.medium_confidence_threshold:
            return 'HALF', 0.5
        elif confidence >= cfg.low_confidence_threshold:
            return 'MINIMUM', 0.25
        else:
            return 'SKIP', 0.0

    def calculate_batch(
        self,
        confidences: np.ndarray,
        win_rate: float,
        avg_win_r: float,
        avg_loss_r: float,
        account_equity: float,
        stop_loss_pct: float = 0.01,
    ) -> np.ndarray:
        """
        Calculate position sizes for multiple signals.

        Args:
            confidences: Array of ML confidences
            Other args: Same as calculate_position_size

        Returns:
            Array of risk percentages
        """
        risk_pcts = []
        for conf in confidences:
            result = self.calculate_position_size(
                ml_confidence=conf,
                win_rate=win_rate,
                avg_win_r=avg_win_r,
                avg_loss_r=avg_loss_r,
                account_equity=account_equity,
                stop_loss_pct=stop_loss_pct,
            )
            risk_pcts.append(result.risk_pct)

        return np.array(risk_pcts)


def print_sizing_summary(result: PositionSizeResult) -> None:
    """Print human-readable sizing summary."""
    print(f"\n{'='*40}")
    print(f"POSITION SIZING RESULT")
    print(f"{'='*40}")
    print(f"ML Confidence: {result.ml_confidence:.1%}")
    print(f"Action: {result.action}")
    print(f"Base Kelly: {result.base_kelly_fraction:.3f}")
    print(f"Confidence Multiplier: {result.confidence_multiplier:.0%}")
    print(f"Adjusted Kelly: {result.adjusted_kelly_fraction:.3f}")
    print(f"Risk %: {result.risk_pct:.2%}")
    print(f"Risk $: ${result.risk_dollars:,.2f}")
    print(f"Position Size: ${result.position_size_dollars:,.2f}")
    print(f"Skip Trade: {result.should_skip}")
    print(f"Rationale: {result.rationale}")
    print(f"{'='*40}\n")
