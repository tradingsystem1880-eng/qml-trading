"""
Bybit Paper Trader
==================
Paper trading engine using Bybit testnet for forward validation.

Uses HierarchicalSwingDetector (Phase 7.6+) for pattern detection
and phase-based risk management from forward test protocol.

Features:
- Real-time pattern detection on live data
- Position management with SL/TP
- Phase-based risk scaling (0.5% -> 0.75% -> 1.0%)
- Progress/Pause/Shutdown triggers
- Trade logging and metrics
"""

import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import uuid

from .bybit_client import BybitTestnetClient
from .models import (
    OrderSide,
    PositionSide,
    TradeSignal,
    Position,
    CompletedTrade,
    PhaseConfig,
    ForwardTestPhase,
    ForwardTestState,
)

# Import detection pipeline
import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.hierarchical_swing import HierarchicalSwingDetector, HierarchicalSwingConfig
from src.detection.pattern_validator import PatternValidator
from src.detection.pattern_scorer import PatternScorer, PatternTier
from src.detection.backtest_adapter import BacktestAdapter
from src.detection.regime import MarketRegimeDetector
from src.detection.config import PatternValidationConfig, PatternScoringConfig
from src.risk.position_rules import PositionRulesManager

logger = logging.getLogger(__name__)


# Phase 7.9 optimized configs
SWING_CONFIG = HierarchicalSwingConfig(
    min_bar_separation=3,
    min_move_atr=0.85,
    forward_confirm_pct=0.2,
    lookback=6,
    lookforward=8,
)

PATTERN_CONFIG = PatternValidationConfig(
    p3_min_extension_atr=0.3,
    p3_max_extension_atr=5.0,
    p5_max_symmetry_atr=4.6,
    min_pattern_bars=16,
    max_pattern_bars=200,
)

SCORING_CONFIG = PatternScoringConfig()


class BybitPaperTrader:
    """
    Paper trading engine for forward testing on Bybit testnet.

    Usage:
        trader = BybitPaperTrader(
            api_key="...",
            api_secret="...",
            symbols=["BTC/USDT", "ETH/USDT"],
        )
        trader.run_scan()  # Scan for patterns
        trader.check_positions()  # Manage open positions
    """

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        symbols: List[str] = None,
        timeframe: str = "4h",
        phase: ForwardTestPhase = ForwardTestPhase.PHASE1_PAPER,
        state_file: Optional[str] = None,
        log_dir: Optional[str] = None,
    ):
        """
        Initialize paper trader.

        Args:
            api_key: Bybit testnet API key
            api_secret: Bybit testnet API secret
            symbols: List of symbols to trade
            timeframe: Detection timeframe
            phase: Current forward test phase
            state_file: File to persist state
            log_dir: Directory for trade logs
        """
        self.symbols = symbols or ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        self.timeframe = timeframe

        # Initialize exchange client
        self.client = BybitTestnetClient(
            api_key=api_key,
            api_secret=api_secret,
            testnet=True,
        )

        # Initialize detection pipeline
        self.swing_config = SWING_CONFIG
        self.pattern_config = PATTERN_CONFIG
        self.scoring_config = SCORING_CONFIG
        self.regime_detector = MarketRegimeDetector()
        self.adapter = BacktestAdapter()

        # Initialize risk management
        self.risk_manager = PositionRulesManager()

        # Initialize state
        self.phase_config = self._get_phase_config(phase)
        self.state = ForwardTestState(
            phase=phase,
            phase_config=self.phase_config,
        )

        # Persistence
        self.state_file = state_file
        self.log_dir = Path(log_dir) if log_dir else PROJECT_ROOT / "results" / "forward_test"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Load existing state if available
        if state_file and Path(state_file).exists():
            self._load_state()

        logger.info(f"BybitPaperTrader initialized: {len(self.symbols)} symbols, "
                   f"phase={phase.value}, risk={self.phase_config.risk_per_trade_pct}%")

    def _get_phase_config(self, phase: ForwardTestPhase) -> PhaseConfig:
        """Get configuration for a phase."""
        if phase == ForwardTestPhase.PHASE1_PAPER:
            return PhaseConfig.phase1()
        elif phase == ForwardTestPhase.PHASE2_MICRO:
            return PhaseConfig.phase2()
        else:
            return PhaseConfig.phase3()

    # ========== Pattern Detection ==========

    def detect_patterns(self, symbol: str) -> List[TradeSignal]:
        """
        Detect patterns on a symbol using HierarchicalSwingDetector.

        Returns:
            List of TradeSignal objects
        """
        # Fetch recent data
        try:
            df = self.client.fetch_ohlcv(symbol, self.timeframe, limit=300)
        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol}: {e}")
            return []

        if len(df) < 100:
            logger.warning(f"Insufficient data for {symbol}: {len(df)} bars")
            return []

        # Standardize column names
        df.columns = [c.lower() for c in df.columns]

        # Calculate ATR
        if 'atr' not in df.columns:
            df['atr'] = self._calculate_atr(df, period=14)

        # Run detection
        normalized = symbol.replace("/", "")
        detector = HierarchicalSwingDetector(
            config=self.swing_config,
            symbol=normalized,
            timeframe=self.timeframe,
        )
        swings = detector.detect(df)

        # Validate patterns
        validator = PatternValidator(self.pattern_config)
        patterns = validator.find_patterns(swings, df['close'].values)
        valid_patterns = [p for p in patterns if p.is_valid]

        # Score patterns with regime filtering
        scorer = PatternScorer(self.scoring_config)
        signals = []

        for p in valid_patterns:
            p5_idx = p.p5.bar_index

            # Skip patterns that are too old (more than 5 bars ago)
            if p5_idx < len(df) - 5:
                continue

            # Calculate regime at pattern time
            window_start = max(0, p5_idx - 150)
            window_df = df.iloc[window_start:p5_idx + 1].copy()
            regime_result = self.regime_detector.get_regime(window_df)

            # Score pattern
            score_result = scorer.score(p, df=df, regime_result=regime_result)

            if score_result.tier == PatternTier.REJECT:
                continue

            # Convert to signal
            converted = self.adapter.batch_convert_to_signals(
                validation_results=[p],
                scoring_results=[score_result],
                symbol=normalized,
                min_tier=PatternTier.C,
            )

            for sig in converted:
                # Get ATR at signal
                atr = df.iloc[p5_idx]['atr'] if 'atr' in df.columns else 0

                # Calculate SL and TP
                direction = sig.signal_type.value.upper().replace('BUY', 'LONG').replace('SELL', 'SHORT')
                sl_atr_mult = 1.0
                tp_atr_mult = 4.6

                if direction == "LONG":
                    sl = sig.price - (sl_atr_mult * atr)
                    tp = sig.price + (tp_atr_mult * atr)
                else:
                    sl = sig.price + (sl_atr_mult * atr)
                    tp = sig.price - (tp_atr_mult * atr)

                trade_signal = TradeSignal(
                    symbol=symbol,
                    direction=direction,
                    entry_price=sig.price,
                    stop_loss=sl,
                    take_profit=tp,
                    atr=atr,
                    score=sig.validity_score if hasattr(sig, 'validity_score') else 0.5,
                    timestamp=datetime.now(),
                    pattern_id=str(uuid.uuid4())[:8],
                    tier=score_result.tier.name,
                    validity_score=sig.validity_score if hasattr(sig, 'validity_score') else 0.5,
                )

                signals.append(trade_signal)

        return signals

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR."""
        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        return tr.rolling(window=period).mean()

    # ========== Position Management ==========

    def run_scan(self) -> List[TradeSignal]:
        """
        Scan all symbols for trading signals.

        Returns:
            List of new signals found
        """
        if self.state.is_shutdown:
            logger.warning("Forward test is shutdown. No scanning.")
            return []

        if self.state.is_paused:
            logger.warning(f"Forward test is paused: {self.state.pause_reason}")
            return []

        all_signals = []

        for symbol in self.symbols:
            try:
                signals = self.detect_patterns(symbol)
                all_signals.extend(signals)
                if signals:
                    logger.info(f"Found {len(signals)} signals for {symbol}")
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")

        return all_signals

    def can_open_position(self, signal: TradeSignal) -> bool:
        """
        Check if we can open a new position.

        Checks:
        - Not paused/shutdown
        - Max positions not reached
        - Daily loss limit not hit
        - No existing position for this symbol
        """
        if self.state.is_shutdown or self.state.is_paused:
            return False

        # Check max positions
        if len(self.state.open_positions) >= self.phase_config.max_positions:
            logger.info(f"Max positions reached ({self.phase_config.max_positions})")
            return False

        # Check daily loss limit
        if abs(self.state.daily_pnl_r) >= self.phase_config.max_daily_loss_pct:
            logger.info(f"Daily loss limit reached ({self.state.daily_pnl_r:.2f}R)")
            return False

        # Check for existing position in same symbol
        for pos in self.state.open_positions:
            if pos.symbol == signal.symbol:
                logger.info(f"Already have position in {signal.symbol}")
                return False

        return True

    def calculate_position_size(
        self,
        signal: TradeSignal,
        account_equity: float,
    ) -> float:
        """
        Calculate position size based on risk rules.

        Args:
            signal: Trade signal with entry and stop loss
            account_equity: Current account equity

        Returns:
            Position size in base currency
        """
        risk_pct = self.phase_config.risk_per_trade_pct / 100.0
        risk_usd = account_equity * risk_pct

        # Calculate risk per unit
        risk_per_unit = abs(signal.entry_price - signal.stop_loss)
        if risk_per_unit == 0:
            return 0

        # Calculate quantity
        quantity = risk_usd / risk_per_unit

        # Round to symbol precision
        quantity = self.client.round_quantity(signal.symbol, quantity)

        # Check minimum order size
        min_size = self.client.get_min_order_size(signal.symbol)
        if quantity < min_size:
            logger.warning(f"Position size {quantity} below minimum {min_size}")
            return 0

        return quantity

    def open_position(self, signal: TradeSignal) -> Optional[Position]:
        """
        Open a new position from a signal.

        Args:
            signal: Trade signal to execute

        Returns:
            Position object if successful, None otherwise
        """
        if not self.can_open_position(signal):
            return None

        # Get account balance
        try:
            balance = self.client.get_balance()
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            return None

        # Calculate position size
        quantity = self.calculate_position_size(signal, balance.total_equity)
        if quantity == 0:
            logger.warning("Position size is 0, skipping")
            return None

        # For paper trading, we simulate the order
        logger.info(f"PAPER TRADE: Opening {signal.direction} {quantity} {signal.symbol} @ {signal.entry_price}")

        # Create position record
        position = Position(
            symbol=signal.symbol,
            side=PositionSide.LONG if signal.direction == "LONG" else PositionSide.SHORT,
            entry_price=signal.entry_price,
            quantity=quantity,
            stop_loss_price=signal.stop_loss,
            take_profit_price=signal.take_profit,
            entry_time=datetime.now(),
            signal=signal,
            highest_price=signal.entry_price,
            lowest_price=signal.entry_price,
        )

        self.state.open_positions.append(position)
        self._save_state()

        return position

    def check_positions(self) -> List[CompletedTrade]:
        """
        Check open positions for SL/TP hits.

        Returns:
            List of positions that were closed
        """
        closed_trades = []

        for position in list(self.state.open_positions):
            try:
                # Get current price
                ticker = self.client.fetch_ticker(position.symbol)
                current_price = ticker['last']

                # Update trailing extremes
                position.update_trailing_extreme(current_price)
                position.bars_held += 1

                # Check stop loss
                hit_sl = False
                hit_tp = False

                if position.side == PositionSide.LONG:
                    hit_sl = current_price <= position.stop_loss_price
                    hit_tp = current_price >= position.take_profit_price
                else:
                    hit_sl = current_price >= position.stop_loss_price
                    hit_tp = current_price <= position.take_profit_price

                if hit_sl or hit_tp:
                    exit_reason = "stop_loss" if hit_sl else "take_profit"
                    trade = self._close_position(position, current_price, exit_reason)
                    if trade:
                        closed_trades.append(trade)

            except Exception as e:
                logger.error(f"Error checking position {position.symbol}: {e}")

        return closed_trades

    def _close_position(
        self,
        position: Position,
        exit_price: float,
        exit_reason: str,
    ) -> Optional[CompletedTrade]:
        """Close a position and record the trade."""
        # Calculate P&L
        if position.side == PositionSide.LONG:
            pnl_usd = (exit_price - position.entry_price) * position.quantity
        else:
            pnl_usd = (position.entry_price - exit_price) * position.quantity

        # Calculate R-multiple
        risk_per_unit = abs(position.entry_price - position.stop_loss_price)
        pnl_r = (pnl_usd / position.quantity) / risk_per_unit if risk_per_unit > 0 else 0

        # Create completed trade record
        trade = CompletedTrade(
            id=str(uuid.uuid4())[:8],
            symbol=position.symbol,
            direction=position.side.value.upper(),
            entry_price=position.entry_price,
            exit_price=exit_price,
            quantity=position.quantity,
            entry_time=position.entry_time or datetime.now(),
            exit_time=datetime.now(),
            pnl_usd=pnl_usd,
            pnl_r=pnl_r,
            exit_reason=exit_reason,
            bars_held=position.bars_held,
            signal_score=position.signal.score if position.signal else 0,
            pattern_tier=position.signal.tier if position.signal else "C",
        )

        # Update state
        self.state.open_positions.remove(position)
        self.state.completed_trades.append(trade)
        self.state.total_pnl_r += pnl_r
        self.state.daily_pnl_r += pnl_r

        # Update consecutive losses
        if pnl_r < 0:
            self.state.consecutive_losses += 1
        else:
            self.state.consecutive_losses = 0

        # Check for pause/shutdown conditions
        self._check_risk_triggers()

        # Log and save
        logger.info(f"CLOSED: {trade.symbol} {trade.direction} "
                   f"PnL={trade.pnl_r:+.2f}R ({trade.exit_reason})")

        self._log_trade(trade)
        self._save_state()

        return trade

    def _check_risk_triggers(self):
        """Check if pause or shutdown conditions are met."""
        if self.state.should_shutdown():
            self.state.is_shutdown = True
            if self.state.consecutive_losses >= self.phase_config.shutdown_consecutive_losses:
                self.state.shutdown_reason = f"Consecutive losses: {self.state.consecutive_losses}"
            else:
                self.state.shutdown_reason = f"PF below threshold: {self.state.profit_factor:.2f}"
            logger.warning(f"SHUTDOWN TRIGGERED: {self.state.shutdown_reason}")

        elif self.state.should_pause():
            self.state.is_paused = True
            self.state.pause_reason = f"PF={self.state.profit_factor:.2f}, WR={self.state.win_rate:.1%}"
            logger.warning(f"PAUSE TRIGGERED: {self.state.pause_reason}")

    # ========== Progress and Reporting ==========

    def get_status(self) -> Dict:
        """Get current forward test status."""
        return {
            "phase": self.state.phase.value,
            "trade_count": self.state.trade_count,
            "target_trades": self.phase_config.min_trades,
            "win_rate": self.state.win_rate,
            "profit_factor": self.state.profit_factor,
            "expectancy": self.state.expectancy,
            "total_pnl_r": self.state.total_pnl_r,
            "open_positions": len(self.state.open_positions),
            "consecutive_losses": self.state.consecutive_losses,
            "is_paused": self.state.is_paused,
            "pause_reason": self.state.pause_reason,
            "is_shutdown": self.state.is_shutdown,
            "shutdown_reason": self.state.shutdown_reason,
            "can_progress": self.state.should_progress(),
        }

    def print_status(self):
        """Print formatted status to console."""
        status = self.get_status()

        print("\n" + "=" * 60)
        print(f"FORWARD TEST STATUS - {status['phase'].upper()}")
        print("=" * 60)
        print(f"\nProgress: {status['trade_count']}/{status['target_trades']} trades")
        print(f"Win Rate: {status['win_rate']:.1%}")
        print(f"Profit Factor: {status['profit_factor']:.2f}")
        print(f"Expectancy: {status['expectancy']:.2f}R")
        print(f"Total P&L: {status['total_pnl_r']:+.2f}R")
        print(f"Open Positions: {status['open_positions']}")
        print(f"Consecutive Losses: {status['consecutive_losses']}")

        if status['is_shutdown']:
            print(f"\n⛔ SHUTDOWN: {status['shutdown_reason']}")
        elif status['is_paused']:
            print(f"\n⏸️  PAUSED: {status['pause_reason']}")
        elif status['can_progress']:
            print(f"\n✅ Ready to progress to next phase!")
        else:
            remaining = status['target_trades'] - status['trade_count']
            print(f"\n📊 {remaining} more trades needed")

    def advance_phase(self) -> bool:
        """
        Advance to the next forward test phase if conditions are met.

        Returns:
            True if advanced, False otherwise
        """
        if not self.state.should_progress():
            logger.warning("Cannot advance: conditions not met")
            return False

        current = self.state.phase
        if current == ForwardTestPhase.PHASE1_PAPER:
            new_phase = ForwardTestPhase.PHASE2_MICRO
        elif current == ForwardTestPhase.PHASE2_MICRO:
            new_phase = ForwardTestPhase.PHASE3_FULL
        else:
            logger.info("Already at final phase")
            return False

        # Update phase
        self.state.phase = new_phase
        self.phase_config = self._get_phase_config(new_phase)
        self.state.phase_config = self.phase_config

        # Reset daily stats but keep cumulative
        self.state.daily_pnl_r = 0
        self.state.consecutive_losses = 0
        self.state.is_paused = False
        self.state.pause_reason = None

        logger.info(f"Advanced to {new_phase.value}")
        self._save_state()

        return True

    # ========== Persistence ==========

    def _save_state(self):
        """Save current state to file."""
        if not self.state_file:
            return

        state_dict = {
            "phase": self.state.phase.value,
            "total_pnl_r": self.state.total_pnl_r,
            "consecutive_losses": self.state.consecutive_losses,
            "daily_pnl_r": self.state.daily_pnl_r,
            "is_paused": self.state.is_paused,
            "pause_reason": self.state.pause_reason,
            "is_shutdown": self.state.is_shutdown,
            "shutdown_reason": self.state.shutdown_reason,
            "completed_trades": [
                {
                    "id": t.id,
                    "symbol": t.symbol,
                    "direction": t.direction,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "quantity": t.quantity,
                    "entry_time": t.entry_time.isoformat(),
                    "exit_time": t.exit_time.isoformat(),
                    "pnl_usd": t.pnl_usd,
                    "pnl_r": t.pnl_r,
                    "exit_reason": t.exit_reason,
                    "bars_held": t.bars_held,
                }
                for t in self.state.completed_trades
            ],
            "open_positions": [
                {
                    "symbol": p.symbol,
                    "side": p.side.value,
                    "entry_price": p.entry_price,
                    "quantity": p.quantity,
                    "stop_loss_price": p.stop_loss_price,
                    "take_profit_price": p.take_profit_price,
                    "entry_time": p.entry_time.isoformat() if p.entry_time else None,
                    "bars_held": p.bars_held,
                }
                for p in self.state.open_positions
            ],
        }

        with open(self.state_file, 'w') as f:
            json.dump(state_dict, f, indent=2)

    def _load_state(self):
        """Load state from file."""
        if not self.state_file or not Path(self.state_file).exists():
            return

        with open(self.state_file, 'r') as f:
            state_dict = json.load(f)

        self.state.phase = ForwardTestPhase(state_dict.get("phase", "phase1_paper"))
        self.state.total_pnl_r = state_dict.get("total_pnl_r", 0)
        self.state.consecutive_losses = state_dict.get("consecutive_losses", 0)
        self.state.daily_pnl_r = state_dict.get("daily_pnl_r", 0)
        self.state.is_paused = state_dict.get("is_paused", False)
        self.state.pause_reason = state_dict.get("pause_reason")
        self.state.is_shutdown = state_dict.get("is_shutdown", False)
        self.state.shutdown_reason = state_dict.get("shutdown_reason")

        # Load completed trades
        for t_dict in state_dict.get("completed_trades", []):
            trade = CompletedTrade(
                id=t_dict["id"],
                symbol=t_dict["symbol"],
                direction=t_dict["direction"],
                entry_price=t_dict["entry_price"],
                exit_price=t_dict["exit_price"],
                quantity=t_dict["quantity"],
                entry_time=datetime.fromisoformat(t_dict["entry_time"]),
                exit_time=datetime.fromisoformat(t_dict["exit_time"]),
                pnl_usd=t_dict["pnl_usd"],
                pnl_r=t_dict["pnl_r"],
                exit_reason=t_dict["exit_reason"],
                bars_held=t_dict.get("bars_held", 0),
            )
            self.state.completed_trades.append(trade)

        logger.info(f"Loaded state: {len(self.state.completed_trades)} completed trades")

    def _log_trade(self, trade: CompletedTrade):
        """Log trade to file."""
        log_file = self.log_dir / "trade_log.jsonl"

        trade_dict = {
            "id": trade.id,
            "timestamp": datetime.now().isoformat(),
            "symbol": trade.symbol,
            "direction": trade.direction,
            "entry_price": trade.entry_price,
            "exit_price": trade.exit_price,
            "quantity": trade.quantity,
            "pnl_usd": trade.pnl_usd,
            "pnl_r": trade.pnl_r,
            "exit_reason": trade.exit_reason,
            "bars_held": trade.bars_held,
            "phase": self.state.phase.value,
        }

        with open(log_file, 'a') as f:
            f.write(json.dumps(trade_dict) + "\n")
