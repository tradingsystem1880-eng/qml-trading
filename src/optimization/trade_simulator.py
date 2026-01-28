"""
Trade Simulator for Phase 7.7 Optimization
==========================================
Enhanced trade simulation engine with:
- MAE/MFE tracking (Maximum Adverse/Favorable Excursion)
- Trailing stops with configurable activation and step
- Time-based exits (max bars held)
- Entry confirmation (close beyond entry, entry buffer ATR)

Designed for rapid evaluation during Bayesian optimization.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from src.core.models import Side, TradeResult


class ExitReason(str, Enum):
    """Reason for trade exit."""
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"
    TIME_EXIT = "time_exit"
    END_OF_DATA = "end_of_data"


@dataclass
class TradeManagementConfig:
    """
    Configuration for trade management parameters.

    All parameters are ML-optimizable for Phase 7.7.
    Phase 9.0 adds time-decaying TP and volume-adjusted TP.
    Phase 9.2 adds multi-stage trailing stop to fix breakeven bug.
    """
    # Entry confirmation
    entry_buffer_atr: float = 0.1  # Buffer beyond entry for confirmation
    require_close_confirmation: bool = True  # Wait for close beyond entry

    # Stop loss / Take profit ATR multipliers
    sl_atr_mult: float = 1.5  # SL distance in ATR
    tp_atr_mult: float = 3.0  # TP distance in ATR

    # Trailing stop mode
    # "none" = no trailing (fixed SL/TP only)
    # "simple" = original breakeven activation (BUGGY - causes dust profits)
    # "multi_stage" = Phase 9.2 multi-stage trailing (recommended)
    trailing_mode: str = "multi_stage"

    # Simple trailing stop params (for trailing_mode="simple")
    trailing_activation_atr: float = 1.0  # Activate when profit >= this ATR
    trailing_step_atr: float = 0.5  # Move SL by this much when price moves by step

    # Multi-stage trailing stop params (for trailing_mode="multi_stage")
    # Stage 0: Below stage1_profit_r - keep initial stop, no adjustment
    # Stage 1: stage1_profit_r to stage2_profit_r - move to breakeven + stage1_level_r
    # Stage 2: stage2_profit_r to stage3_profit_r - loose trail at stage2_atr from high
    # Stage 3: stage3_profit_r to stage4_profit_r - medium trail at stage3_atr
    # Stage 4: Above stage4_profit_r - tight trail at stage4_atr
    #
    # Phase 9.2: Conservative defaults to let trades develop
    # Previous defaults (1.0/0.2) caused 40%+ short hold trades
    trailing_stage1_profit_r: float = 1.5   # Activate stage 1 at 1.5R profit (was 1.0)
    trailing_stage1_level_r: float = 0.5    # Move SL to +0.5R (was 0.2R)
    trailing_stage2_profit_r: float = 2.0   # Activate stage 2 at 2.0R profit (was 1.5)
    trailing_stage2_atr: float = 1.2        # Trail at 1.2 ATR from high (was 1.0)
    trailing_stage3_profit_r: float = 3.0   # Activate stage 3 at 3.0R profit (was 2.5)
    trailing_stage3_atr: float = 0.8        # Trail at 0.8 ATR from high (was 0.7)
    trailing_stage4_profit_r: float = 5.0   # Activate stage 4 at 5.0R profit (was 4.0)
    trailing_stage4_atr: float = 0.5        # Trail at 0.5 ATR from high (tight)

    # Time-based exit (0 = disabled)
    max_bars_held: int = 50  # Exit after this many bars if no SL/TP hit

    # Risk/reward ratio validation
    min_risk_reward: float = 1.5  # Skip trades with R:R below this

    # Slippage and commission
    slippage_pct: float = 0.05  # 0.05% slippage per side
    commission_pct: float = 0.1  # 0.1% commission per side

    # Phase 9.0: Time-decaying profit target
    # TP(t) = Entry + Risk × R_target × e^(-λ × t) where λ = ln(2) / halflife_bars
    tp_decay_enabled: bool = False  # Enable time-decaying TP
    tp_decay_halflife_bars: int = 20  # TP R-multiple halves every N bars
    tp_minimum_r: float = 0.5  # Never decay below this R-multiple

    # Phase 9.0: Volume-adjusted TP
    # Extend TP when volume confirms momentum
    volume_tp_enabled: bool = False  # Enable volume-adjusted TP
    volume_extension_threshold: float = 2.0  # Extend if volume > threshold × avg
    volume_extension_mult: float = 1.2  # Extend TP distance by this multiplier


@dataclass
class SimulatedTrade:
    """
    Enhanced trade record with MAE/MFE tracking.

    Stores all information needed for optimization objective calculation.
    """
    # Entry
    entry_time: datetime
    entry_price: float
    entry_bar_idx: int
    side: Side

    # Exit
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_bar_idx: Optional[int] = None
    exit_reason: Optional[ExitReason] = None

    # Position details
    stop_loss: float = 0.0
    take_profit: float = 0.0
    atr_at_entry: float = 0.0

    # Results
    pnl_pct: float = 0.0
    pnl_r: float = 0.0  # P&L in R-multiples
    result: TradeResult = TradeResult.PENDING

    # MAE/MFE tracking (in R-multiples)
    mae_r: float = 0.0  # Maximum Adverse Excursion
    mfe_r: float = 0.0  # Maximum Favorable Excursion
    mae_price: float = 0.0  # Price at MAE
    mfe_price: float = 0.0  # Price at MFE

    # Duration
    bars_held: int = 0

    # Trailing stop tracking
    trailing_activated: bool = False
    trailing_stop_price: Optional[float] = None

    # Metadata
    symbol: Optional[str] = None
    timeframe: Optional[str] = None
    pattern_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_winner(self) -> bool:
        return self.result == TradeResult.WIN

    @property
    def is_loser(self) -> bool:
        return self.result == TradeResult.LOSS

    @property
    def risk_amount(self) -> float:
        """Risk amount in price units."""
        return abs(self.entry_price - self.stop_loss)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'entry_time': self.entry_time.isoformat() if self.entry_time else None,
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'entry_bar_idx': self.entry_bar_idx,
            'exit_bar_idx': self.exit_bar_idx,
            'side': self.side.value,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'atr_at_entry': self.atr_at_entry,
            'pnl_pct': self.pnl_pct,
            'pnl_r': self.pnl_r,
            'result': self.result.value,
            'exit_reason': self.exit_reason.value if self.exit_reason else None,
            'mae_r': self.mae_r,
            'mfe_r': self.mfe_r,
            'bars_held': self.bars_held,
            'trailing_activated': self.trailing_activated,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'pattern_score': self.pattern_score,
        }


@dataclass
class SimulationResult:
    """Aggregated result of trade simulation."""
    trades: List[SimulatedTrade]

    # Summary metrics
    total_trades: int = 0
    winners: int = 0
    losers: int = 0

    # Performance
    win_rate: float = 0.0
    avg_win_r: float = 0.0
    avg_loss_r: float = 0.0
    expectancy_r: float = 0.0
    profit_factor: float = 0.0

    # Risk metrics
    max_drawdown_r: float = 0.0
    sharpe_ratio: float = 0.0

    # MAE/MFE analytics
    avg_mae_r: float = 0.0
    avg_mfe_r: float = 0.0
    mfe_to_mae_ratio: float = 0.0
    mfe_capture_ratio: float = 0.0  # avg_win_r / avg_mfe_r - how much of the move we capture

    # Duration
    avg_bars_held: float = 0.0

    # Exit distribution
    exit_by_sl: int = 0
    exit_by_tp: int = 0
    exit_by_trailing: int = 0
    exit_by_time: int = 0


class TradeSimulator:
    """
    Enhanced trade simulation engine for optimization.

    Features:
    - Bar-by-bar simulation with MAE/MFE tracking
    - Trailing stop management
    - Time-based exits
    - Entry confirmation logic

    Designed for speed in optimization loops.
    """

    def __init__(self, config: Optional[TradeManagementConfig] = None):
        """
        Initialize the trade simulator.

        Args:
            config: Trade management configuration
        """
        self.config = config or TradeManagementConfig()

    def simulate_trades(
        self,
        df: pd.DataFrame,
        signals: List[Dict[str, Any]],
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
    ) -> SimulationResult:
        """
        Simulate trades from pattern signals.

        Args:
            df: OHLCV DataFrame with 'time', 'open', 'high', 'low', 'close', 'atr'
            signals: List of signal dicts with keys:
                - bar_idx: Entry bar index
                - direction: 'LONG' or 'SHORT'
                - entry_price: Suggested entry price
                - atr: ATR at signal
                - score: Pattern quality score (optional)
            symbol: Symbol name for metadata
            timeframe: Timeframe for metadata

        Returns:
            SimulationResult with all trades and metrics
        """
        trades: List[SimulatedTrade] = []

        # Normalize columns
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]

        # Ensure ATR column exists
        if 'atr' not in df.columns:
            df['atr'] = self._calculate_atr(df)

        # Process each signal
        for signal in signals:
            trade = self._simulate_single_trade(df, signal, symbol, timeframe)
            if trade is not None:
                trades.append(trade)

        # Calculate aggregate metrics
        return self._calculate_results(trades)

    def _simulate_single_trade(
        self,
        df: pd.DataFrame,
        signal: Dict[str, Any],
        symbol: Optional[str],
        timeframe: Optional[str],
    ) -> Optional[SimulatedTrade]:
        """Simulate a single trade from signal to exit."""
        bar_idx = signal['bar_idx']
        direction = signal['direction']
        entry_price = signal.get('entry_price', df.iloc[bar_idx]['close'])
        atr = signal.get('atr', df.iloc[bar_idx]['atr'])
        score = signal.get('score', 0.0)

        if bar_idx >= len(df) - 1:
            return None  # Not enough bars for simulation

        # Determine side
        side = Side.LONG if direction.upper() == 'LONG' else Side.SHORT

        # Calculate SL/TP using ATR multiples
        cfg = self.config

        if side == Side.LONG:
            stop_loss = entry_price - (cfg.sl_atr_mult * atr)
            take_profit = entry_price + (cfg.tp_atr_mult * atr)
        else:
            stop_loss = entry_price + (cfg.sl_atr_mult * atr)
            take_profit = entry_price - (cfg.tp_atr_mult * atr)

        # Validate risk/reward
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        if risk > 0 and reward / risk < cfg.min_risk_reward:
            return None  # Skip low R:R trades

        # Apply entry buffer
        if cfg.entry_buffer_atr > 0:
            buffer = cfg.entry_buffer_atr * atr
            if side == Side.LONG:
                entry_price += buffer
            else:
                entry_price -= buffer

        # Apply slippage
        slippage = entry_price * (cfg.slippage_pct / 100)
        if side == Side.LONG:
            entry_price += slippage
        else:
            entry_price -= slippage

        # Create trade
        entry_time = df.iloc[bar_idx]['time']
        if hasattr(entry_time, 'to_pydatetime'):
            entry_time = entry_time.to_pydatetime()

        trade = SimulatedTrade(
            entry_time=entry_time,
            entry_price=entry_price,
            entry_bar_idx=bar_idx,
            side=side,
            stop_loss=stop_loss,
            take_profit=take_profit,
            atr_at_entry=atr,
            symbol=symbol,
            timeframe=timeframe,
            pattern_score=score,
        )

        # Simulate bar-by-bar from entry to exit
        self._simulate_to_exit(trade, df, bar_idx + 1)

        return trade

    def _simulate_to_exit(
        self,
        trade: SimulatedTrade,
        df: pd.DataFrame,
        start_idx: int,
    ) -> None:
        """Simulate trade bar-by-bar until exit."""
        cfg = self.config

        current_sl = trade.stop_loss
        risk = abs(trade.entry_price - trade.stop_loss)

        # Track MAE/MFE
        max_adverse = 0.0  # Worst drawdown from entry
        max_favorable = 0.0  # Best profit from entry

        # Pre-calculate original R-multiple for decay
        original_r = abs(trade.take_profit - trade.entry_price) / risk if risk > 0 else 0

        # Pre-calculate average volume for volume extension (last 20 bars before entry)
        avg_volume = None
        if cfg.volume_tp_enabled and 'volume' in df.columns:
            vol_start = max(0, trade.entry_bar_idx - 20)
            avg_volume = df.iloc[vol_start:trade.entry_bar_idx]['volume'].mean()

        for idx in range(start_idx, len(df)):
            row = df.iloc[idx]
            high = row['high']
            low = row['low']
            close = row['close']

            bars_held = idx - trade.entry_bar_idx

            # Phase 9.0: Calculate time-decayed TP
            current_tp = trade.take_profit  # Default to original
            if cfg.tp_decay_enabled and risk > 0:
                # TP(t) = Entry + Risk × R_target × e^(-λ × t)
                # λ = ln(2) / halflife_bars (so TP halves every halflife_bars)
                decay_lambda = np.log(2) / cfg.tp_decay_halflife_bars
                decay_factor = np.exp(-decay_lambda * bars_held)
                decayed_r = max(cfg.tp_minimum_r, original_r * decay_factor)

                if trade.side == Side.LONG:
                    current_tp = trade.entry_price + (decayed_r * risk)
                else:
                    current_tp = trade.entry_price - (decayed_r * risk)

            # Phase 9.0: Volume extension (boost TP on high volume)
            if cfg.volume_tp_enabled and avg_volume and avg_volume > 0:
                current_volume = row.get('volume', 0)
                if current_volume > avg_volume * cfg.volume_extension_threshold:
                    # Extend TP distance by multiplier
                    tp_distance = abs(current_tp - trade.entry_price)
                    extended_distance = tp_distance * cfg.volume_extension_mult
                    if trade.side == Side.LONG:
                        current_tp = trade.entry_price + extended_distance
                    else:
                        current_tp = trade.entry_price - extended_distance

            # Update MAE/MFE based on side
            if trade.side == Side.LONG:
                # For longs: adverse = price drop, favorable = price rise
                adverse = trade.entry_price - low
                favorable = high - trade.entry_price

                if adverse > max_adverse:
                    max_adverse = adverse
                    trade.mae_price = low
                if favorable > max_favorable:
                    max_favorable = favorable
                    trade.mfe_price = high

                # Handle trailing stop based on mode
                if cfg.trailing_mode == "multi_stage":
                    # Phase 9.2: Multi-stage trailing that lets trades develop
                    current_profit_r = favorable / risk if risk > 0 else 0
                    highest_profit_r = max_favorable / risk if risk > 0 else 0

                    new_trailing = self._get_multi_stage_trailing_stop(
                        current_profit_r=current_profit_r,
                        highest_profit_r=highest_profit_r,
                        highest_price=trade.mfe_price,
                        entry_price=trade.entry_price,
                        initial_stop=trade.stop_loss,
                        atr=trade.atr_at_entry,
                        side=Side.LONG,
                    )

                    if new_trailing is not None:
                        # Only update if new trailing is higher (tighter for longs)
                        if trade.trailing_stop_price is None or new_trailing > trade.trailing_stop_price:
                            trade.trailing_activated = True
                            trade.trailing_stop_price = new_trailing
                            current_sl = max(trade.stop_loss, trade.trailing_stop_price)

                elif cfg.trailing_mode == "simple":
                    # Original simple trailing (kept for backwards compatibility)
                    if cfg.trailing_activation_atr > 0 and not trade.trailing_activated:
                        activation_level = trade.entry_price + (cfg.trailing_activation_atr * trade.atr_at_entry)
                        if high >= activation_level:
                            trade.trailing_activated = True
                            trade.trailing_stop_price = trade.entry_price

                    if trade.trailing_activated and trade.trailing_stop_price is not None:
                        step = cfg.trailing_step_atr * trade.atr_at_entry
                        new_trailing = high - step
                        if new_trailing > trade.trailing_stop_price:
                            trade.trailing_stop_price = new_trailing
                            current_sl = max(trade.stop_loss, trade.trailing_stop_price)
                # else: trailing_mode == "none" - no trailing stop

                # Check exits (order: SL first, then TP)
                if low <= current_sl:
                    exit_price = current_sl
                    if trade.trailing_activated and trade.trailing_stop_price and current_sl >= trade.trailing_stop_price:
                        exit_reason = ExitReason.TRAILING_STOP
                    else:
                        exit_reason = ExitReason.STOP_LOSS
                elif high >= current_tp:
                    exit_price = current_tp
                    exit_reason = ExitReason.TAKE_PROFIT
                elif cfg.max_bars_held > 0 and bars_held >= cfg.max_bars_held:
                    exit_price = close
                    exit_reason = ExitReason.TIME_EXIT
                else:
                    continue  # No exit this bar

            else:  # SHORT
                # For shorts: adverse = price rise, favorable = price drop
                adverse = high - trade.entry_price
                favorable = trade.entry_price - low

                if adverse > max_adverse:
                    max_adverse = adverse
                    trade.mae_price = high
                if favorable > max_favorable:
                    max_favorable = favorable
                    trade.mfe_price = low

                # Handle trailing stop based on mode
                if cfg.trailing_mode == "multi_stage":
                    # Phase 9.2: Multi-stage trailing that lets trades develop
                    current_profit_r = favorable / risk if risk > 0 else 0
                    highest_profit_r = max_favorable / risk if risk > 0 else 0

                    new_trailing = self._get_multi_stage_trailing_stop(
                        current_profit_r=current_profit_r,
                        highest_profit_r=highest_profit_r,
                        highest_price=trade.mfe_price,  # For shorts, this is the lowest price
                        entry_price=trade.entry_price,
                        initial_stop=trade.stop_loss,
                        atr=trade.atr_at_entry,
                        side=Side.SHORT,
                    )

                    if new_trailing is not None:
                        # Only update if new trailing is lower (tighter for shorts)
                        if trade.trailing_stop_price is None or new_trailing < trade.trailing_stop_price:
                            trade.trailing_activated = True
                            trade.trailing_stop_price = new_trailing
                            current_sl = min(trade.stop_loss, trade.trailing_stop_price)

                elif cfg.trailing_mode == "simple":
                    # Original simple trailing (kept for backwards compatibility)
                    if cfg.trailing_activation_atr > 0 and not trade.trailing_activated:
                        activation_level = trade.entry_price - (cfg.trailing_activation_atr * trade.atr_at_entry)
                        if low <= activation_level:
                            trade.trailing_activated = True
                            trade.trailing_stop_price = trade.entry_price

                    if trade.trailing_activated and trade.trailing_stop_price is not None:
                        step = cfg.trailing_step_atr * trade.atr_at_entry
                        new_trailing = low + step
                        if new_trailing < trade.trailing_stop_price:
                            trade.trailing_stop_price = new_trailing
                            current_sl = min(trade.stop_loss, trade.trailing_stop_price)
                # else: trailing_mode == "none" - no trailing stop

                # Check exits
                if high >= current_sl:
                    exit_price = current_sl
                    if trade.trailing_activated and trade.trailing_stop_price and current_sl <= trade.trailing_stop_price:
                        exit_reason = ExitReason.TRAILING_STOP
                    else:
                        exit_reason = ExitReason.STOP_LOSS
                elif low <= current_tp:
                    exit_price = current_tp
                    exit_reason = ExitReason.TAKE_PROFIT
                elif cfg.max_bars_held > 0 and bars_held >= cfg.max_bars_held:
                    exit_price = close
                    exit_reason = ExitReason.TIME_EXIT
                else:
                    continue  # No exit this bar

            # Exit trade
            self._close_trade(trade, df.iloc[idx], idx, exit_price, exit_reason)

            # Store MAE/MFE in R-multiples
            if risk > 0:
                trade.mae_r = max_adverse / risk
                trade.mfe_r = max_favorable / risk

            return

        # End of data - close at last bar
        last_row = df.iloc[-1]
        last_idx = len(df) - 1
        exit_price = last_row['close']

        # Apply slippage
        slippage = exit_price * (cfg.slippage_pct / 100)
        if trade.side == Side.LONG:
            exit_price -= slippage
        else:
            exit_price += slippage

        self._close_trade(trade, last_row, last_idx, exit_price, ExitReason.END_OF_DATA)

        # Store MAE/MFE
        if risk > 0:
            trade.mae_r = max_adverse / risk
            trade.mfe_r = max_favorable / risk

    def _close_trade(
        self,
        trade: SimulatedTrade,
        row: pd.Series,
        idx: int,
        exit_price: float,
        exit_reason: ExitReason,
    ) -> None:
        """Close trade and calculate results."""
        cfg = self.config

        # Apply slippage on exit
        slippage = exit_price * (cfg.slippage_pct / 100)
        if trade.side == Side.LONG:
            exit_price -= slippage
        else:
            exit_price += slippage

        # Store exit info
        trade.exit_price = exit_price
        trade.exit_bar_idx = idx
        trade.exit_reason = exit_reason

        exit_time = row['time']
        if hasattr(exit_time, 'to_pydatetime'):
            exit_time = exit_time.to_pydatetime()
        trade.exit_time = exit_time

        # Calculate P&L
        if trade.side == Side.LONG:
            trade.pnl_pct = ((exit_price - trade.entry_price) / trade.entry_price) * 100
        else:
            trade.pnl_pct = ((trade.entry_price - exit_price) / trade.entry_price) * 100

        # Subtract commission (both sides)
        trade.pnl_pct -= (cfg.commission_pct * 2)

        # Calculate P&L in R-multiples
        risk = abs(trade.entry_price - trade.stop_loss)
        if risk > 0:
            if trade.side == Side.LONG:
                profit = exit_price - trade.entry_price
            else:
                profit = trade.entry_price - exit_price
            trade.pnl_r = profit / risk

        # Determine result
        if trade.pnl_pct > 0.01:
            trade.result = TradeResult.WIN
        elif trade.pnl_pct < -0.01:
            trade.result = TradeResult.LOSS
        else:
            trade.result = TradeResult.BREAKEVEN

        # Duration
        trade.bars_held = idx - trade.entry_bar_idx

    def _calculate_results(self, trades: List[SimulatedTrade]) -> SimulationResult:
        """Calculate aggregate metrics from trades."""
        if not trades:
            return SimulationResult(trades=[])

        winners = [t for t in trades if t.is_winner]
        losers = [t for t in trades if t.is_loser]

        # Basic stats
        total = len(trades)
        win_count = len(winners)
        loss_count = len(losers)
        win_rate = win_count / total if total > 0 else 0.0

        # Average win/loss in R
        avg_win_r = np.mean([t.pnl_r for t in winners]) if winners else 0.0
        avg_loss_r = abs(np.mean([t.pnl_r for t in losers])) if losers else 0.0

        # Expectancy
        expectancy_r = (win_rate * avg_win_r) - ((1 - win_rate) * avg_loss_r)

        # Profit factor
        gross_profit = sum(t.pnl_r for t in winners) if winners else 0.0
        gross_loss = abs(sum(t.pnl_r for t in losers)) if losers else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0

        # Sharpe ratio (using R-multiple returns)
        returns = [t.pnl_r for t in trades]
        if len(returns) > 1:
            sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0.0
        else:
            sharpe = 0.0

        # Max drawdown in R
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        max_drawdown_r = np.max(drawdowns) if len(drawdowns) > 0 else 0.0

        # MAE/MFE analytics
        avg_mae_r = np.mean([t.mae_r for t in trades])
        avg_mfe_r = np.mean([t.mfe_r for t in trades])
        mfe_to_mae = avg_mfe_r / avg_mae_r if avg_mae_r > 0 else 0.0

        # MFE capture ratio - how much of the winning move we actually capture
        # If avg_win_r is 1.5R but avg_mfe_r is 3.0R, we're only capturing 50%
        winners_mfe = [t.mfe_r for t in winners] if winners else [0]
        avg_winners_mfe = np.mean(winners_mfe) if winners else 0.0
        mfe_capture = avg_win_r / avg_winners_mfe if avg_winners_mfe > 0 else 0.0

        # Duration
        avg_bars = np.mean([t.bars_held for t in trades])

        # Exit distribution
        exit_sl = sum(1 for t in trades if t.exit_reason == ExitReason.STOP_LOSS)
        exit_tp = sum(1 for t in trades if t.exit_reason == ExitReason.TAKE_PROFIT)
        exit_trailing = sum(1 for t in trades if t.exit_reason == ExitReason.TRAILING_STOP)
        exit_time = sum(1 for t in trades if t.exit_reason == ExitReason.TIME_EXIT)

        return SimulationResult(
            trades=trades,
            total_trades=total,
            winners=win_count,
            losers=loss_count,
            win_rate=win_rate,
            avg_win_r=float(avg_win_r),
            avg_loss_r=float(avg_loss_r),
            expectancy_r=float(expectancy_r),
            profit_factor=float(profit_factor) if not np.isinf(profit_factor) else 999.0,
            max_drawdown_r=float(max_drawdown_r),
            sharpe_ratio=float(sharpe),
            avg_mae_r=float(avg_mae_r),
            avg_mfe_r=float(avg_mfe_r),
            mfe_to_mae_ratio=float(mfe_to_mae),
            mfe_capture_ratio=float(mfe_capture),
            avg_bars_held=float(avg_bars),
            exit_by_sl=exit_sl,
            exit_by_tp=exit_tp,
            exit_by_trailing=exit_trailing,
            exit_by_time=exit_time,
        )

    def _get_multi_stage_trailing_stop(
        self,
        current_profit_r: float,
        highest_profit_r: float,
        highest_price: float,
        entry_price: float,
        initial_stop: float,
        atr: float,
        side: Side,
    ) -> Optional[float]:
        """
        Calculate multi-stage trailing stop level.

        Phase 9.2: Multi-stage trailing that lets trades develop:
        - Stage 0: Below 1R - no adjustment, keep initial stop
        - Stage 1: 1-1.5R - move to breakeven + 0.2R (protect small profit)
        - Stage 2: 1.5-2.5R - loose trail at 1.0 ATR from high
        - Stage 3: 2.5-4R - medium trail at 0.7 ATR from high
        - Stage 4: >4R - tight trail at 0.5 ATR from high

        Returns:
            New trailing stop price, or None if no adjustment needed
        """
        cfg = self.config
        risk = abs(entry_price - initial_stop)
        direction = 1 if side == Side.LONG else -1

        if current_profit_r < cfg.trailing_stage1_profit_r:
            # Stage 0: Below threshold - no trailing, keep initial stop
            return None

        elif current_profit_r < cfg.trailing_stage2_profit_r:
            # Stage 1: Move to just above breakeven
            return entry_price + (cfg.trailing_stage1_level_r * risk * direction)

        elif current_profit_r < cfg.trailing_stage3_profit_r:
            # Stage 2: Loose trail
            return highest_price - (cfg.trailing_stage2_atr * atr * direction)

        elif current_profit_r < cfg.trailing_stage4_profit_r:
            # Stage 3: Medium trail
            return highest_price - (cfg.trailing_stage3_atr * atr * direction)

        else:
            # Stage 4: Tight trail
            return highest_price - (cfg.trailing_stage4_atr * atr * direction)

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR if not present."""
        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period, min_periods=1).mean()

        return atr
