"""
Data Schemas for QML Trading System - Phase 2
==============================================
Extended dataclasses for pattern detection, trade outcomes,
feature vectors, and experiment tracking.

These complement the existing Pydantic models in models.py.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Dict, Any, List
import json
import hashlib


@dataclass
class PatternDetection:
    """
    A detected QML pattern with full geometry.

    Stores all 5 swing points (P1-P5) plus calculated trading levels.
    """
    id: str
    symbol: str
    timeframe: str
    direction: str  # 'BULLISH' or 'BEARISH'
    detection_time: datetime

    # Swing points (P1-P5)
    p1_price: float
    p1_time: datetime
    p2_price: float
    p2_time: datetime
    p3_price: float  # Head
    p3_time: datetime
    p4_price: float
    p4_time: datetime
    p5_price: float  # Entry zone
    p5_time: datetime

    # Calculated levels
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: Optional[float] = None
    take_profit_3: Optional[float] = None

    # Detection parameters used (for A/B tracking)
    detection_params: Optional[Dict[str, Any]] = None

    # Quality metrics
    validity_score: float = 0.0
    confidence: float = 0.0

    # Status
    status: str = 'ACTIVE'  # 'ACTIVE', 'TRIGGERED', 'INVALIDATED', 'EXPIRED'

    # Database tracking
    created_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        d = asdict(self)
        # Convert datetimes to ISO strings
        for key in ['detection_time', 'p1_time', 'p2_time', 'p3_time', 'p4_time', 'p5_time', 'created_at']:
            if d.get(key):
                d[key] = d[key].isoformat() if isinstance(d[key], datetime) else str(d[key])
        if d.get('detection_params'):
            d['detection_params'] = json.dumps(d['detection_params'])
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'PatternDetection':
        """Create from dictionary."""
        # Parse datetimes
        for key in ['detection_time', 'p1_time', 'p2_time', 'p3_time', 'p4_time', 'p5_time', 'created_at']:
            if d.get(key) and isinstance(d[key], str):
                try:
                    d[key] = datetime.fromisoformat(d[key])
                except (ValueError, TypeError):
                    d[key] = None
        if d.get('detection_params') and isinstance(d['detection_params'], str):
            try:
                d['detection_params'] = json.loads(d['detection_params'])
            except (ValueError, TypeError):
                d['detection_params'] = None
        return cls(**d)


@dataclass
class TradeOutcome:
    """
    Actual trade result from a pattern.

    Tracks execution, result, and risk metrics.
    """
    id: str
    pattern_id: str
    symbol: str
    timeframe: str
    direction: str  # 'LONG' or 'SHORT'

    # Execution
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None

    # Result
    status: str = 'OPEN'  # 'OPEN', 'WIN', 'LOSS', 'BREAKEVEN'
    pnl_dollars: Optional[float] = None
    pnl_percent: Optional[float] = None
    r_multiple: Optional[float] = None

    # Risk
    position_size: float = 0.0
    risk_amount: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0

    # Metadata
    exit_reason: Optional[str] = None  # 'TP1', 'TP2', 'TP3', 'SL', 'MANUAL', 'TIME', 'END_OF_DATA'
    bars_held: Optional[int] = None

    # Database tracking
    experiment_id: Optional[str] = None
    created_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        d = asdict(self)
        for key in ['entry_time', 'exit_time']:
            if d[key]:
                d[key] = d[key].isoformat() if isinstance(d[key], datetime) else str(d[key])
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'TradeOutcome':
        """Create from dictionary."""
        for key in ['entry_time', 'exit_time', 'created_at']:
            if d.get(key) and isinstance(d[key], str):
                try:
                    d[key] = datetime.fromisoformat(d[key])
                except (ValueError, TypeError):
                    d[key] = None
        return cls(**d)


@dataclass
class FeatureVector:
    """
    Features calculated for a pattern - for ML training.

    Organized into tiers by importance:
    - Tier 1: Pattern Geometry (MUST HAVE)
    - Tier 2: Market Context (HIGH PRIORITY)
    - Tier 3: Volume (HIGH PRIORITY)
    - Tier 4: Pattern Quality (MEDIUM)
    """
    pattern_id: str
    calculation_time: datetime

    # Tier 1: Pattern Geometry (MUST HAVE)
    head_extension_atr: float = 0.0      # (P3 - P1) / ATR
    bos_depth_atr: float = 0.0           # (P2 - P4) / ATR
    shoulder_symmetry: float = 0.0       # abs(P5 - P1) / ATR
    amplitude_ratio: float = 0.0         # (P1-P2) / (P3-P4)
    time_ratio: float = 0.0              # bars(P1→P3) / bars(P3→P5)
    fib_retracement_p5: float = 0.0      # Where P5 falls on fib (0-1)

    # Tier 2: Market Context (HIGH PRIORITY)
    htf_trend_alignment: float = 0.0     # -1 (against) to 1 (with)
    distance_to_sr_atr: float = 0.0      # Distance to nearest S/R level
    volatility_percentile: float = 0.0   # 0-100 percentile rank
    regime_state: str = 'UNKNOWN'        # 'TRENDING', 'RANGING', 'VOLATILE'
    rsi_divergence: float = 0.0          # RSI divergence strength

    # Tier 3: Volume (HIGH PRIORITY)
    volume_spike_p3: float = 0.0         # Volume at head vs average
    volume_spike_p4: float = 0.0         # Volume at BoS vs average
    volume_trend_p1_p5: float = 0.0      # Volume trend across pattern

    # Tier 4: Pattern Quality (MEDIUM)
    noise_ratio: float = 0.0             # Noise within pattern
    bos_candle_strength: float = 0.0     # BoS candle body/range ratio

    # Outcome (for training labels)
    outcome: Optional[str] = None        # 'WIN', 'LOSS', 'BREAKEVEN'
    r_multiple: Optional[float] = None   # Actual R achieved

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        d = asdict(self)
        if d['calculation_time']:
            d['calculation_time'] = d['calculation_time'].isoformat() if isinstance(d['calculation_time'], datetime) else str(d['calculation_time'])
        return d

    def to_array(self) -> List[float]:
        """Convert numeric features to array for ML."""
        return [
            self.head_extension_atr,
            self.bos_depth_atr,
            self.shoulder_symmetry,
            self.amplitude_ratio,
            self.time_ratio,
            self.fib_retracement_p5,
            self.htf_trend_alignment,
            self.distance_to_sr_atr,
            self.volatility_percentile,
            self.rsi_divergence,
            self.volume_spike_p3,
            self.volume_spike_p4,
            self.volume_trend_p1_p5,
            self.noise_ratio,
            self.bos_candle_strength,
        ]

    @staticmethod
    def feature_names() -> List[str]:
        """Get feature names for ML."""
        return [
            'head_extension_atr',
            'bos_depth_atr',
            'shoulder_symmetry',
            'amplitude_ratio',
            'time_ratio',
            'fib_retracement_p5',
            'htf_trend_alignment',
            'distance_to_sr_atr',
            'volatility_percentile',
            'rsi_divergence',
            'volume_spike_p3',
            'volume_spike_p4',
            'volume_trend_p1_p5',
            'noise_ratio',
            'bos_candle_strength',
        ]


@dataclass
class ExperimentRun:
    """
    A backtest/experiment run with full tracking.

    Stores configuration, results, and validation metrics.
    """
    id: str
    name: str
    created_at: datetime

    # Configuration
    symbol: str
    timeframe: str
    start_date: datetime
    end_date: datetime

    # Parameters tested (for version tracking)
    detection_params: Dict[str, Any] = field(default_factory=dict)
    risk_params: Dict[str, Any] = field(default_factory=dict)

    # Results
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    total_return: float = 0.0
    expectancy: float = 0.0
    avg_r_multiple: float = 0.0

    # Equity curve (stored as JSON)
    equity_curve: Optional[List] = None

    # Validation metrics (Phase 4)
    walk_forward_efficiency: Optional[float] = None
    probability_of_overfitting: Optional[float] = None
    deflated_sharpe: Optional[float] = None
    permutation_p_value: Optional[float] = None

    # Prop firm metrics (Phase 5)
    max_daily_drawdown: Optional[float] = None
    consistency_score: Optional[float] = None
    pass_probability: Optional[float] = None

    # Report path
    report_path: Optional[str] = None
    git_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        d = asdict(self)
        for key in ['created_at', 'start_date', 'end_date']:
            if d[key]:
                d[key] = d[key].isoformat() if isinstance(d[key], datetime) else str(d[key])
        d['detection_params'] = json.dumps(d['detection_params'])
        d['risk_params'] = json.dumps(d['risk_params'])
        if d['equity_curve']:
            d['equity_curve'] = json.dumps(d['equity_curve'], default=str)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ExperimentRun':
        """Create from dictionary."""
        for key in ['created_at', 'start_date', 'end_date']:
            if d.get(key) and isinstance(d[key], str):
                d[key] = datetime.fromisoformat(d[key])
        if d.get('detection_params') and isinstance(d['detection_params'], str):
            d['detection_params'] = json.loads(d['detection_params'])
        if d.get('risk_params') and isinstance(d['risk_params'], str):
            d['risk_params'] = json.loads(d['risk_params'])
        if d.get('equity_curve') and isinstance(d['equity_curve'], str):
            d['equity_curve'] = json.loads(d['equity_curve'])
        return cls(**d)


def generate_id(prefix: str = '') -> str:
    """Generate a unique ID."""
    import uuid
    short_uuid = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    return f"{prefix}{timestamp}_{short_uuid}"


def hash_params(params: Dict[str, Any]) -> str:
    """Generate hash of parameters for A/B tracking."""
    params_str = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(params_str.encode()).hexdigest()[:12]
