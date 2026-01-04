"""
Advanced Diagnostics Module
============================
Deep analysis of strategy behavior and edge attribution.

Includes:
1. Volatility Expansion Analysis (pre/post trade)
2. Drawdown Decomposition (source of losses)
3. Rolling Correlations (Strategy vs BTC/SPY)
4. Trade Clustering Analysis
5. Time-of-Day Performance
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from src.utils.indicators import calculate_atr


@dataclass
class VolatilityExpansionResult:
    """Result of volatility expansion analysis."""
    
    # Pre-trade volatility stats
    avg_pre_trade_vol: float
    median_pre_trade_vol: float
    
    # Post-trade volatility stats
    avg_post_trade_vol: float
    median_post_trade_vol: float
    
    # Expansion metrics
    avg_expansion_ratio: float
    expansion_win_rate: float  # % of trades where vol expanded
    
    # By trade outcome
    vol_expansion_winners: float
    vol_expansion_losers: float
    
    # Time series
    pre_trade_vol_series: np.ndarray = field(default_factory=lambda: np.array([]))
    post_trade_vol_series: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class DrawdownDecomposition:
    """Decomposition of drawdown sources."""
    
    # Total stats
    max_drawdown: float
    avg_drawdown: float
    drawdown_duration_avg: float  # bars
    
    # By regime
    drawdown_by_regime: Dict[str, float] = field(default_factory=dict)
    
    # By time
    drawdown_by_hour: Dict[int, float] = field(default_factory=dict)
    drawdown_by_dow: Dict[int, float] = field(default_factory=dict)
    
    # By trade size
    drawdown_from_large_losses: float = 0.0  # % from top 10% losses
    drawdown_from_small_losses: float = 0.0  # % from bottom 90% losses
    
    # Consecutive losses
    max_consecutive_losses: int = 0
    avg_consecutive_losses: float = 0.0
    
    # Recovery
    avg_recovery_time: float = 0.0  # bars
    max_recovery_time: float = 0.0


@dataclass
class CorrelationAnalysis:
    """Rolling correlation analysis."""
    
    # Overall correlations
    correlation_btc: float = 0.0
    correlation_spy: float = 0.0
    
    # Rolling correlations (time series)
    rolling_corr_btc: np.ndarray = field(default_factory=lambda: np.array([]))
    rolling_corr_spy: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Correlation stability
    corr_btc_std: float = 0.0
    corr_spy_std: float = 0.0
    
    # Regime-specific correlations
    corr_by_regime: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class DiagnosticsResult:
    """Complete diagnostics result."""
    
    volatility_expansion: Optional[VolatilityExpansionResult] = None
    drawdown_decomposition: Optional[DrawdownDecomposition] = None
    correlation_analysis: Optional[CorrelationAnalysis] = None
    
    # Trade clustering
    trade_clusters: Dict[str, Dict] = field(default_factory=dict)
    
    # Time-of-day performance
    hourly_performance: Dict[int, Dict[str, float]] = field(default_factory=dict)
    dow_performance: Dict[int, Dict[str, float]] = field(default_factory=dict)


class AdvancedDiagnostics:
    """
    Advanced Diagnostics for Strategy Analysis.
    
    Provides deep analysis of:
    - Volatility behavior around trades
    - Drawdown sources and decomposition
    - Cross-asset correlations
    - Temporal performance patterns
    """
    
    def __init__(
        self,
        atr_period: int = 14,
        vol_window_pre: int = 20,
        vol_window_post: int = 20,
        correlation_window: int = 50
    ):
        """
        Initialize diagnostics.
        
        Args:
            atr_period: ATR calculation period
            vol_window_pre: Bars before trade for vol analysis
            vol_window_post: Bars after trade for vol analysis
            correlation_window: Window for rolling correlations
        """
        self.atr_period = atr_period
        self.vol_window_pre = vol_window_pre
        self.vol_window_post = vol_window_post
        self.correlation_window = correlation_window
        
        logger.info("AdvancedDiagnostics initialized")
    
    def run_full_diagnostics(
        self,
        trades_df: pd.DataFrame,
        price_df: pd.DataFrame,
        btc_df: Optional[pd.DataFrame] = None,
        spy_df: Optional[pd.DataFrame] = None,
        regime_labels: Optional[np.ndarray] = None,
        regime_mapping: Optional[Dict[int, str]] = None
    ) -> DiagnosticsResult:
        """
        Run complete diagnostics suite.
        
        Args:
            trades_df: DataFrame with trade data (entry_time, exit_time, pnl_pct)
            price_df: OHLCV price data
            btc_df: Optional BTC price data for correlation
            spy_df: Optional SPY/equity data for correlation
            regime_labels: Optional regime labels for each bar
            regime_mapping: Optional regime label -> name mapping
            
        Returns:
            DiagnosticsResult with all analyses
        """
        logger.info("Running full diagnostics suite...")
        
        result = DiagnosticsResult()
        
        # 1. Volatility Expansion
        result.volatility_expansion = self.analyze_volatility_expansion(
            trades_df, price_df
        )
        
        # 2. Drawdown Decomposition
        result.drawdown_decomposition = self.decompose_drawdowns(
            trades_df, price_df, regime_labels, regime_mapping
        )
        
        # 3. Correlation Analysis
        result.correlation_analysis = self.analyze_correlations(
            trades_df, price_df, btc_df, spy_df, regime_labels, regime_mapping
        )
        
        # 4. Temporal Performance
        result.hourly_performance = self.analyze_hourly_performance(trades_df)
        result.dow_performance = self.analyze_dow_performance(trades_df)
        
        logger.info("Diagnostics complete")
        
        return result
    
    def analyze_volatility_expansion(
        self,
        trades_df: pd.DataFrame,
        price_df: pd.DataFrame
    ) -> VolatilityExpansionResult:
        """
        Analyze volatility expansion around trades.
        
        Measures whether trades tend to occur before vol expansion
        (good for breakout strategies) or after (mean reversion).
        """
        logger.info("  Analyzing volatility expansion...")
        
        # Calculate ATR for full price series
        high = price_df["high"].values
        low = price_df["low"].values
        close = price_df["close"].values
        atr = calculate_atr(high, low, close, self.atr_period)
        
        # Normalize ATR by price
        atr_pct = atr / close * 100
        
        # Get time index
        times = pd.to_datetime(price_df["time"])
        
        pre_trade_vols = []
        post_trade_vols = []
        expansion_ratios = []
        winner_expansions = []
        loser_expansions = []
        
        for _, trade in trades_df.iterrows():
            entry_time = pd.Timestamp(trade.get("entry_time", trade.get("time")))
            pnl = trade.get("pnl_pct", trade.get("pnl", 0))
            
            # Find bar index
            time_diffs = abs(times - entry_time)
            bar_idx = time_diffs.argmin()
            
            if bar_idx < self.vol_window_pre or bar_idx + self.vol_window_post >= len(atr_pct):
                continue
            
            # Pre-trade volatility
            pre_vol = np.nanmean(atr_pct[bar_idx - self.vol_window_pre:bar_idx])
            
            # Post-trade volatility
            post_vol = np.nanmean(atr_pct[bar_idx:bar_idx + self.vol_window_post])
            
            if np.isnan(pre_vol) or np.isnan(post_vol) or pre_vol == 0:
                continue
            
            pre_trade_vols.append(pre_vol)
            post_trade_vols.append(post_vol)
            
            expansion = post_vol / pre_vol
            expansion_ratios.append(expansion)
            
            if pnl > 0:
                winner_expansions.append(expansion)
            else:
                loser_expansions.append(expansion)
        
        if not pre_trade_vols:
            logger.warning("    No valid trades for volatility analysis")
            return VolatilityExpansionResult(
                avg_pre_trade_vol=0,
                median_pre_trade_vol=0,
                avg_post_trade_vol=0,
                median_post_trade_vol=0,
                avg_expansion_ratio=1,
                expansion_win_rate=0,
                vol_expansion_winners=1,
                vol_expansion_losers=1,
            )
        
        result = VolatilityExpansionResult(
            avg_pre_trade_vol=float(np.mean(pre_trade_vols)),
            median_pre_trade_vol=float(np.median(pre_trade_vols)),
            avg_post_trade_vol=float(np.mean(post_trade_vols)),
            median_post_trade_vol=float(np.median(post_trade_vols)),
            avg_expansion_ratio=float(np.mean(expansion_ratios)),
            expansion_win_rate=float(np.mean([e > 1 for e in expansion_ratios])),
            vol_expansion_winners=float(np.mean(winner_expansions)) if winner_expansions else 1.0,
            vol_expansion_losers=float(np.mean(loser_expansions)) if loser_expansions else 1.0,
            pre_trade_vol_series=np.array(pre_trade_vols),
            post_trade_vol_series=np.array(post_trade_vols),
        )
        
        logger.info(f"    Avg expansion ratio: {result.avg_expansion_ratio:.3f}")
        logger.info(f"    Expansion win rate: {result.expansion_win_rate:.1%}")
        
        return result
    
    def decompose_drawdowns(
        self,
        trades_df: pd.DataFrame,
        price_df: pd.DataFrame,
        regime_labels: Optional[np.ndarray] = None,
        regime_mapping: Optional[Dict[int, str]] = None
    ) -> DrawdownDecomposition:
        """
        Decompose drawdowns by source.
        
        Identifies what's causing losses: regime, time, trade size, etc.
        """
        logger.info("  Decomposing drawdowns...")
        
        pnl_col = "pnl_pct" if "pnl_pct" in trades_df.columns else "pnl"
        pnl = trades_df[pnl_col].values
        
        # Build equity curve
        equity = np.cumprod(1 + pnl / 100)
        running_max = np.maximum.accumulate(equity)
        drawdowns = (running_max - equity) / running_max * 100
        
        max_dd = float(np.max(drawdowns))
        avg_dd = float(np.mean(drawdowns))
        
        # Calculate drawdown durations
        in_drawdown = drawdowns > 0
        dd_starts = np.where(np.diff(in_drawdown.astype(int)) == 1)[0]
        dd_ends = np.where(np.diff(in_drawdown.astype(int)) == -1)[0]
        
        if len(dd_starts) > 0 and len(dd_ends) > 0:
            if dd_ends[0] < dd_starts[0]:
                dd_ends = dd_ends[1:]
            min_len = min(len(dd_starts), len(dd_ends))
            dd_durations = dd_ends[:min_len] - dd_starts[:min_len]
            avg_duration = float(np.mean(dd_durations)) if len(dd_durations) > 0 else 0
        else:
            avg_duration = 0
        
        # Drawdown by regime
        dd_by_regime = {}
        if regime_labels is not None and regime_mapping is not None:
            times = pd.to_datetime(price_df["time"])
            for _, trade in trades_df.iterrows():
                trade_pnl = trade[pnl_col]
                if trade_pnl >= 0:
                    continue
                
                entry_time = pd.Timestamp(trade.get("entry_time", trade.get("time")))
                bar_idx = abs(times - entry_time).argmin()
                
                if bar_idx < len(regime_labels) and not np.isnan(regime_labels[bar_idx]):
                    regime = regime_mapping.get(int(regime_labels[bar_idx]), "unknown")
                    dd_by_regime[regime] = dd_by_regime.get(regime, 0) + abs(trade_pnl)
        
        # Drawdown by hour
        dd_by_hour = {}
        if "entry_time" in trades_df.columns or "time" in trades_df.columns:
            time_col = "entry_time" if "entry_time" in trades_df.columns else "time"
            for _, trade in trades_df.iterrows():
                if trade[pnl_col] >= 0:
                    continue
                hour = pd.Timestamp(trade[time_col]).hour
                dd_by_hour[hour] = dd_by_hour.get(hour, 0) + abs(trade[pnl_col])
        
        # Drawdown by day of week
        dd_by_dow = {}
        if "entry_time" in trades_df.columns or "time" in trades_df.columns:
            time_col = "entry_time" if "entry_time" in trades_df.columns else "time"
            for _, trade in trades_df.iterrows():
                if trade[pnl_col] >= 0:
                    continue
                dow = pd.Timestamp(trade[time_col]).dayofweek
                dd_by_dow[dow] = dd_by_dow.get(dow, 0) + abs(trade[pnl_col])
        
        # Large vs small losses
        losses = pnl[pnl < 0]
        if len(losses) > 0:
            threshold = np.percentile(losses, 10)  # Top 10% of losses
            large_losses = losses[losses <= threshold]
            small_losses = losses[losses > threshold]
            
            total_loss = abs(losses.sum())
            dd_from_large = float(abs(large_losses.sum()) / total_loss * 100) if total_loss > 0 else 0
            dd_from_small = float(abs(small_losses.sum()) / total_loss * 100) if total_loss > 0 else 0
        else:
            dd_from_large = 0
            dd_from_small = 0
        
        # Consecutive losses
        is_loss = pnl < 0
        consec_losses = []
        current_streak = 0
        for loss in is_loss:
            if loss:
                current_streak += 1
            else:
                if current_streak > 0:
                    consec_losses.append(current_streak)
                current_streak = 0
        if current_streak > 0:
            consec_losses.append(current_streak)
        
        max_consec = max(consec_losses) if consec_losses else 0
        avg_consec = float(np.mean(consec_losses)) if consec_losses else 0
        
        result = DrawdownDecomposition(
            max_drawdown=max_dd,
            avg_drawdown=avg_dd,
            drawdown_duration_avg=avg_duration,
            drawdown_by_regime=dd_by_regime,
            drawdown_by_hour=dd_by_hour,
            drawdown_by_dow=dd_by_dow,
            drawdown_from_large_losses=dd_from_large,
            drawdown_from_small_losses=dd_from_small,
            max_consecutive_losses=max_consec,
            avg_consecutive_losses=avg_consec,
        )
        
        logger.info(f"    Max drawdown: {max_dd:.2f}%")
        logger.info(f"    Max consecutive losses: {max_consec}")
        logger.info(f"    DD from large losses: {dd_from_large:.1f}%")
        
        return result
    
    def analyze_correlations(
        self,
        trades_df: pd.DataFrame,
        price_df: pd.DataFrame,
        btc_df: Optional[pd.DataFrame] = None,
        spy_df: Optional[pd.DataFrame] = None,
        regime_labels: Optional[np.ndarray] = None,
        regime_mapping: Optional[Dict[int, str]] = None
    ) -> CorrelationAnalysis:
        """
        Analyze rolling correlations with benchmarks.
        """
        logger.info("  Analyzing correlations...")
        
        result = CorrelationAnalysis()
        
        # Get strategy returns
        strategy_returns = price_df["close"].pct_change().dropna()
        
        # BTC correlation
        if btc_df is not None and len(btc_df) > 0:
            btc_returns = btc_df["close"].pct_change().dropna()
            
            # Align returns
            min_len = min(len(strategy_returns), len(btc_returns))
            if min_len >= self.correlation_window:
                strat_aligned = strategy_returns.iloc[-min_len:].reset_index(drop=True)
                btc_aligned = btc_returns.iloc[-min_len:].reset_index(drop=True)
                
                # Overall correlation
                result.correlation_btc = float(strat_aligned.corr(btc_aligned))
                
                # Rolling correlation
                rolling_corr = []
                for i in range(self.correlation_window, min_len):
                    window_corr = strat_aligned.iloc[i - self.correlation_window:i].corr(
                        btc_aligned.iloc[i - self.correlation_window:i]
                    )
                    rolling_corr.append(window_corr)
                
                result.rolling_corr_btc = np.array(rolling_corr)
                result.corr_btc_std = float(np.std(rolling_corr)) if rolling_corr else 0
            
            logger.info(f"    BTC correlation: {result.correlation_btc:.3f}")
        
        # SPY correlation
        if spy_df is not None and len(spy_df) > 0:
            spy_returns = spy_df["close"].pct_change().dropna()
            
            min_len = min(len(strategy_returns), len(spy_returns))
            if min_len >= self.correlation_window:
                strat_aligned = strategy_returns.iloc[-min_len:].reset_index(drop=True)
                spy_aligned = spy_returns.iloc[-min_len:].reset_index(drop=True)
                
                result.correlation_spy = float(strat_aligned.corr(spy_aligned))
                
                rolling_corr = []
                for i in range(self.correlation_window, min_len):
                    window_corr = strat_aligned.iloc[i - self.correlation_window:i].corr(
                        spy_aligned.iloc[i - self.correlation_window:i]
                    )
                    rolling_corr.append(window_corr)
                
                result.rolling_corr_spy = np.array(rolling_corr)
                result.corr_spy_std = float(np.std(rolling_corr)) if rolling_corr else 0
            
            logger.info(f"    SPY correlation: {result.correlation_spy:.3f}")
        
        # Regime-specific correlations
        if regime_labels is not None and regime_mapping is not None and btc_df is not None:
            btc_returns = btc_df["close"].pct_change()
            
            for label, regime_name in regime_mapping.items():
                regime_mask = regime_labels == label
                
                if np.sum(regime_mask) < 20:
                    continue
                
                regime_strat = strategy_returns.values[:len(regime_mask)][regime_mask[:len(strategy_returns)]]
                regime_btc = btc_returns.values[:len(regime_mask)][regime_mask[:len(btc_returns)]]
                
                if len(regime_strat) > 10 and len(regime_btc) > 10:
                    min_len = min(len(regime_strat), len(regime_btc))
                    corr = np.corrcoef(regime_strat[:min_len], regime_btc[:min_len])[0, 1]
                    if not np.isnan(corr):
                        result.corr_by_regime[regime_name] = {"btc_corr": float(corr)}
        
        return result
    
    def analyze_hourly_performance(
        self,
        trades_df: pd.DataFrame
    ) -> Dict[int, Dict[str, float]]:
        """Analyze performance by hour of day."""
        logger.info("  Analyzing hourly performance...")
        
        pnl_col = "pnl_pct" if "pnl_pct" in trades_df.columns else "pnl"
        time_col = "entry_time" if "entry_time" in trades_df.columns else "time"
        
        if time_col not in trades_df.columns:
            return {}
        
        hourly = {}
        for hour in range(24):
            trades_df["_hour"] = pd.to_datetime(trades_df[time_col]).dt.hour
            hour_trades = trades_df[trades_df["_hour"] == hour]
            
            if len(hour_trades) == 0:
                continue
            
            pnl = hour_trades[pnl_col].values
            hourly[hour] = {
                "n_trades": len(pnl),
                "win_rate": float(np.mean(pnl > 0)),
                "avg_pnl": float(np.mean(pnl)),
                "total_pnl": float(np.sum(pnl)),
            }
        
        if "_hour" in trades_df.columns:
            trades_df.drop("_hour", axis=1, inplace=True)
        
        return hourly
    
    def analyze_dow_performance(
        self,
        trades_df: pd.DataFrame
    ) -> Dict[int, Dict[str, float]]:
        """Analyze performance by day of week."""
        logger.info("  Analyzing day-of-week performance...")
        
        pnl_col = "pnl_pct" if "pnl_pct" in trades_df.columns else "pnl"
        time_col = "entry_time" if "entry_time" in trades_df.columns else "time"
        
        if time_col not in trades_df.columns:
            return {}
        
        dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        dow_perf = {}
        
        for dow in range(7):
            trades_df["_dow"] = pd.to_datetime(trades_df[time_col]).dt.dayofweek
            dow_trades = trades_df[trades_df["_dow"] == dow]
            
            if len(dow_trades) == 0:
                continue
            
            pnl = dow_trades[pnl_col].values
            dow_perf[dow] = {
                "name": dow_names[dow],
                "n_trades": len(pnl),
                "win_rate": float(np.mean(pnl > 0)),
                "avg_pnl": float(np.mean(pnl)),
                "total_pnl": float(np.sum(pnl)),
            }
        
        if "_dow" in trades_df.columns:
            trades_df.drop("_dow", axis=1, inplace=True)
        
        return dow_perf
    
    def generate_report(self, result: DiagnosticsResult) -> str:
        """Generate text report of diagnostics."""
        lines = [
            "=" * 60,
            "ADVANCED DIAGNOSTICS REPORT",
            "=" * 60,
            "",
        ]
        
        # Volatility Expansion
        if result.volatility_expansion:
            ve = result.volatility_expansion
            lines.extend([
                "VOLATILITY EXPANSION ANALYSIS",
                "-" * 60,
                f"  Pre-trade avg vol:  {ve.avg_pre_trade_vol:.4f}%",
                f"  Post-trade avg vol: {ve.avg_post_trade_vol:.4f}%",
                f"  Avg expansion ratio: {ve.avg_expansion_ratio:.3f}",
                f"  Expansion win rate: {ve.expansion_win_rate:.1%}",
                f"  Winners expansion: {ve.vol_expansion_winners:.3f}",
                f"  Losers expansion: {ve.vol_expansion_losers:.3f}",
                "",
            ])
        
        # Drawdown Decomposition
        if result.drawdown_decomposition:
            dd = result.drawdown_decomposition
            lines.extend([
                "DRAWDOWN DECOMPOSITION",
                "-" * 60,
                f"  Max drawdown: {dd.max_drawdown:.2f}%",
                f"  Avg drawdown: {dd.avg_drawdown:.2f}%",
                f"  Max consecutive losses: {dd.max_consecutive_losses}",
                f"  Avg consecutive losses: {dd.avg_consecutive_losses:.2f}",
                f"  From large losses (top 10%): {dd.drawdown_from_large_losses:.1f}%",
                "",
            ])
            
            if dd.drawdown_by_regime:
                lines.append("  Drawdown by regime:")
                for regime, dd_val in sorted(dd.drawdown_by_regime.items(), key=lambda x: -x[1]):
                    lines.append(f"    {regime}: {dd_val:.2f}")
        
        # Correlations
        if result.correlation_analysis:
            ca = result.correlation_analysis
            lines.extend([
                "",
                "CORRELATION ANALYSIS",
                "-" * 60,
                f"  BTC correlation: {ca.correlation_btc:.3f} (std: {ca.corr_btc_std:.3f})",
                f"  SPY correlation: {ca.correlation_spy:.3f} (std: {ca.corr_spy_std:.3f})",
            ])
            
            if ca.corr_by_regime:
                lines.append("  Correlation by regime:")
                for regime, corrs in ca.corr_by_regime.items():
                    lines.append(f"    {regime}: BTC={corrs.get('btc_corr', 0):.3f}")
        
        # Temporal Performance
        if result.hourly_performance:
            lines.extend([
                "",
                "HOURLY PERFORMANCE (best hours):",
                "-" * 60,
            ])
            sorted_hours = sorted(
                result.hourly_performance.items(),
                key=lambda x: x[1].get("avg_pnl", 0),
                reverse=True
            )[:5]
            for hour, stats in sorted_hours:
                lines.append(
                    f"  Hour {hour:02d}: avg={stats['avg_pnl']:.3f}%, "
                    f"win_rate={stats['win_rate']:.0%}, n={stats['n_trades']}"
                )
        
        if result.dow_performance:
            lines.extend([
                "",
                "DAY OF WEEK PERFORMANCE:",
                "-" * 60,
            ])
            for dow in sorted(result.dow_performance.keys()):
                stats = result.dow_performance[dow]
                lines.append(
                    f"  {stats['name']}: avg={stats['avg_pnl']:.3f}%, "
                    f"win_rate={stats['win_rate']:.0%}, n={stats['n_trades']}"
                )
        
        lines.append("=" * 60)
        
        return "\n".join(lines)


def create_diagnostics(
    atr_period: int = 14,
    vol_window: int = 20,
    correlation_window: int = 50
) -> AdvancedDiagnostics:
    """Factory function for AdvancedDiagnostics."""
    return AdvancedDiagnostics(
        atr_period=atr_period,
        vol_window_pre=vol_window,
        vol_window_post=vol_window,
        correlation_window=correlation_window
    )
