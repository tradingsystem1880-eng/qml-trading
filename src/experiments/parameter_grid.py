"""
Parameter Grid Management for A/B Testing - Phase 6
====================================================
Manages parameter combinations for systematic backtesting experiments.

Features:
- ParameterSet: Single parameter configuration with hashing
- GridSearchConfig: Defines search space (210K+ combinations)
- ParameterGridManager: Deduplication and progress tracking
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterator, List, Optional
from itertools import product
import hashlib
import json

from src.data.sqlite_manager import SQLiteManager


@dataclass
class ParameterSet:
    """
    A single parameter configuration for backtesting.

    Detection parameters control how QML patterns are identified.
    Entry/exit parameters control trade execution.
    Filter parameters add additional confirmation requirements.
    """
    # Detection parameters
    swing_lookback: int = 5
    smoothing_window: int = 5
    min_head_extension_atr: float = 0.5
    bos_requirement: int = 1
    shoulder_tolerance_atr: float = 0.5

    # Entry/exit parameters
    entry_trigger: str = 'touch'  # 'touch', 'close', 'confirmation'
    sl_placement_atr: float = 1.0
    tp_r_multiple: float = 2.0

    # Filter parameters
    volume_filter: str = 'none'  # 'none', '1.5x', '2x'
    fvg_filter: str = 'none'     # 'none', 'required'
    ob_filter: str = 'none'      # 'none', 'required'

    def to_hash(self) -> str:
        """Generate 12-char MD5 hash of parameters."""
        params_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(params_str.encode()).hexdigest()[:12]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ParameterSet':
        """Create from dictionary."""
        # Filter to only valid fields
        valid_fields = {
            'swing_lookback', 'smoothing_window', 'min_head_extension_atr',
            'bos_requirement', 'shoulder_tolerance_atr', 'entry_trigger',
            'sl_placement_atr', 'tp_r_multiple', 'volume_filter',
            'fvg_filter', 'ob_filter'
        }
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)

    def __str__(self) -> str:
        """Human-readable representation."""
        return (f"ParameterSet(swing={self.swing_lookback}, "
                f"head_ext={self.min_head_extension_atr}, "
                f"entry={self.entry_trigger}, "
                f"tp_r={self.tp_r_multiple})")


@dataclass
class GridSearchConfig:
    """
    Configuration for parameter grid search.

    Each list defines the values to test for that parameter.
    Total combinations = product of all list lengths.

    Default configuration: ~210K combinations
    """
    # Detection parameters
    swing_lookback: List[int] = field(default_factory=lambda: [3, 5, 7, 10])
    smoothing_window: List[int] = field(default_factory=lambda: [3, 5, 7])
    min_head_extension_atr: List[float] = field(default_factory=lambda: [0.3, 0.5, 1.0])
    bos_requirement: List[int] = field(default_factory=lambda: [1, 2])
    shoulder_tolerance_atr: List[float] = field(default_factory=lambda: [0.3, 0.5, 1.0])

    # Entry/exit parameters
    entry_trigger: List[str] = field(default_factory=lambda: ['touch', 'close', 'confirmation'])
    sl_placement_atr: List[float] = field(default_factory=lambda: [0.5, 1.0, 1.5])
    tp_r_multiple: List[float] = field(default_factory=lambda: [1.0, 2.0, 3.0])

    # Filter parameters
    volume_filter: List[str] = field(default_factory=lambda: ['none', '1.5x', '2x'])
    fvg_filter: List[str] = field(default_factory=lambda: ['none', 'required'])
    ob_filter: List[str] = field(default_factory=lambda: ['none', 'required'])

    def total_combinations(self) -> int:
        """Calculate total number of parameter combinations."""
        return (
            len(self.swing_lookback) *
            len(self.smoothing_window) *
            len(self.min_head_extension_atr) *
            len(self.bos_requirement) *
            len(self.shoulder_tolerance_atr) *
            len(self.entry_trigger) *
            len(self.sl_placement_atr) *
            len(self.tp_r_multiple) *
            len(self.volume_filter) *
            len(self.fvg_filter) *
            len(self.ob_filter)
        )

    def to_dict(self) -> Dict[str, List]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def minimal(cls) -> 'GridSearchConfig':
        """Create minimal config for quick testing."""
        return cls(
            swing_lookback=[5],
            smoothing_window=[5],
            min_head_extension_atr=[0.5],
            bos_requirement=[1],
            shoulder_tolerance_atr=[0.5],
            entry_trigger=['touch'],
            sl_placement_atr=[1.0],
            tp_r_multiple=[2.0],
            volume_filter=['none'],
            fvg_filter=['none'],
            ob_filter=['none'],
        )

    @classmethod
    def small(cls) -> 'GridSearchConfig':
        """Create small config for development testing (~100 combinations)."""
        return cls(
            swing_lookback=[3, 5, 7],
            smoothing_window=[3, 5],
            min_head_extension_atr=[0.5],
            bos_requirement=[1],
            shoulder_tolerance_atr=[0.5],
            entry_trigger=['touch', 'close'],
            sl_placement_atr=[1.0],
            tp_r_multiple=[1.5, 2.0, 3.0],
            volume_filter=['none'],
            fvg_filter=['none'],
            ob_filter=['none'],
        )


class ParameterGridManager:
    """
    Manages parameter grid generation with deduplication.

    Uses SQLiteManager to track which parameters have been tested,
    preventing duplicate experiments and tracking progress.

    Usage:
        db = SQLiteManager()
        manager = ParameterGridManager(db)

        config = GridSearchConfig()
        print(f"Total combinations: {config.total_combinations()}")

        # Get untested parameters
        for params in manager.get_untested(config, limit=100):
            # Run backtest with params
            # ...
            manager.mark_tested(params)
    """

    def __init__(self, db: SQLiteManager):
        """
        Initialize grid manager.

        Args:
            db: SQLiteManager instance for tracking tested parameters
        """
        self.db = db

    def generate_grid(self, config: GridSearchConfig) -> Iterator[ParameterSet]:
        """
        Generate all parameter combinations from config.

        Args:
            config: GridSearchConfig defining search space

        Yields:
            ParameterSet for each combination
        """
        # Get all value lists
        all_values = [
            config.swing_lookback,
            config.smoothing_window,
            config.min_head_extension_atr,
            config.bos_requirement,
            config.shoulder_tolerance_atr,
            config.entry_trigger,
            config.sl_placement_atr,
            config.tp_r_multiple,
            config.volume_filter,
            config.fvg_filter,
            config.ob_filter,
        ]

        # Generate all combinations
        for combo in product(*all_values):
            yield ParameterSet(
                swing_lookback=combo[0],
                smoothing_window=combo[1],
                min_head_extension_atr=combo[2],
                bos_requirement=combo[3],
                shoulder_tolerance_atr=combo[4],
                entry_trigger=combo[5],
                sl_placement_atr=combo[6],
                tp_r_multiple=combo[7],
                volume_filter=combo[8],
                fvg_filter=combo[9],
                ob_filter=combo[10],
            )

    def get_untested(
        self,
        config: GridSearchConfig,
        limit: Optional[int] = None
    ) -> Iterator[ParameterSet]:
        """
        Generate untested parameter combinations.

        Args:
            config: GridSearchConfig defining search space
            limit: Optional max number of parameters to return

        Yields:
            ParameterSet that has not been tested yet
        """
        count = 0
        for params in self.generate_grid(config):
            if not self.db.has_been_tested(params.to_dict()):
                yield params
                count += 1
                if limit and count >= limit:
                    break

    def mark_tested(self, params: ParameterSet) -> str:
        """
        Mark parameters as tested.

        Args:
            params: ParameterSet that was tested

        Returns:
            Parameter hash
        """
        return self.db.register_params(params.to_dict())

    def has_been_tested(self, params: ParameterSet) -> bool:
        """Check if parameters have been tested."""
        return self.db.has_been_tested(params.to_dict())

    def get_progress(self, config: GridSearchConfig) -> Dict[str, Any]:
        """
        Get grid search progress statistics.

        Args:
            config: GridSearchConfig to check progress for

        Returns:
            Dict with total, tested, remaining, progress_pct
        """
        total = config.total_combinations()
        tested = 0

        # Count tested (this can be slow for large grids)
        # For performance, we could use a sampling approach
        for params in self.generate_grid(config):
            if self.db.has_been_tested(params.to_dict()):
                tested += 1
            # Early exit optimization for large grids
            if tested > 10000:
                # Estimate based on sample
                break

        remaining = total - tested
        progress_pct = (tested / total * 100) if total > 0 else 0

        return {
            'total': total,
            'tested': tested,
            'remaining': remaining,
            'progress_pct': round(progress_pct, 2),
        }

    def get_fast_progress(self, config: GridSearchConfig) -> Dict[str, Any]:
        """
        Get quick progress estimate without full grid scan.

        Uses database query to count tested parameters instead
        of iterating through entire grid.

        Args:
            config: GridSearchConfig to check progress for

        Returns:
            Dict with total, tested (estimated), remaining, progress_pct
        """
        total = config.total_combinations()

        # Count from database directly
        with self.db.connection() as conn:
            row = conn.execute("SELECT COUNT(*) as count FROM parameters").fetchone()
            tested = row['count'] if row else 0

        # Cap at total (tested might include params from other configs)
        tested = min(tested, total)
        remaining = total - tested
        progress_pct = (tested / total * 100) if total > 0 else 0

        return {
            'total': total,
            'tested': tested,
            'remaining': remaining,
            'progress_pct': round(progress_pct, 2),
            'note': 'Estimate - tested count may include other configurations'
        }
