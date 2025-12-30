"""
Telegram Alerts for QML Trading System
=======================================
Sends pattern alerts and notifications via Telegram bot.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from loguru import logger

try:
    from telegram import Bot
    from telegram.constants import ParseMode
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    logger.warning("python-telegram-bot not installed, Telegram alerts disabled")

from config.settings import settings
from src.data.models import PatternType, QMLPattern


@dataclass
class AlertMessage:
    """Alert message content."""
    
    title: str
    symbol: str
    timeframe: str
    pattern_type: str
    
    entry: float
    stop_loss: float
    take_profit: float
    
    validity_score: float
    ml_confidence: Optional[float] = None
    
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_telegram_message(self) -> str:
        """Format as Telegram message with markdown."""
        
        direction = "🟢 LONG" if self.pattern_type == "bullish" else "🔴 SHORT"
        confidence = f"{self.ml_confidence:.1%}" if self.ml_confidence else "N/A"
        
        message = f"""
*{self.title}*
━━━━━━━━━━━━━━━━━━

{direction} *{self.symbol}* ({self.timeframe})

📊 *Pattern Quality*
├ Validity: `{self.validity_score:.1%}`
└ ML Confidence: `{confidence}`

💰 *Trading Levels*
├ Entry: `{self.entry:.8f}`
├ Stop Loss: `{self.stop_loss:.8f}`
└ Take Profit: `{self.take_profit:.8f}`

⏰ {self.timestamp.strftime('%Y-%m-%d %H:%M UTC')}
"""
        return message.strip()


class TelegramAlerts:
    """
    Sends alerts via Telegram bot.
    
    Usage:
        alerts = TelegramAlerts()
        await alerts.send_pattern_alert(pattern)
    """
    
    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None
    ):
        """
        Initialize Telegram alerts.
        
        Args:
            bot_token: Telegram bot token
            chat_id: Chat ID to send messages to
        """
        self.bot_token = bot_token or settings.telegram.bot_token
        self.chat_id = chat_id or settings.telegram.chat_id
        self.bot: Optional[Bot] = None
        self.enabled = TELEGRAM_AVAILABLE and bool(self.bot_token) and bool(self.chat_id)
        
        if self.enabled:
            self.bot = Bot(token=self.bot_token)
            logger.info("Telegram alerts initialized")
        else:
            logger.warning("Telegram alerts disabled (missing token or chat_id)")
    
    async def send_message(self, text: str, parse_mode: str = "Markdown") -> bool:
        """
        Send a text message.
        
        Args:
            text: Message text
            parse_mode: Message format ('Markdown' or 'HTML')
            
        Returns:
            True if sent successfully
        """
        if not self.enabled or not self.bot:
            logger.debug("Telegram alerts disabled, skipping message")
            return False
        
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=text,
                parse_mode=parse_mode
            )
            logger.debug("Telegram message sent")
            return True
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False
    
    async def send_pattern_alert(self, pattern: QMLPattern) -> bool:
        """
        Send alert for a detected pattern.
        
        Args:
            pattern: QML pattern to alert
            
        Returns:
            True if sent successfully
        """
        if not pattern.trading_levels:
            logger.warning("Pattern has no trading levels, skipping alert")
            return False
        
        message = AlertMessage(
            title="🎯 QML Pattern Detected",
            symbol=pattern.symbol,
            timeframe=pattern.timeframe,
            pattern_type=pattern.pattern_type.value,
            entry=pattern.trading_levels.entry,
            stop_loss=pattern.trading_levels.stop_loss,
            take_profit=pattern.trading_levels.take_profit_3,
            validity_score=pattern.validity_score,
            ml_confidence=pattern.ml_confidence,
            timestamp=pattern.detection_time
        )
        
        return await self.send_message(message.to_telegram_message())
    
    async def send_daily_summary(
        self,
        total_patterns: int,
        high_quality_patterns: int,
        best_opportunities: list
    ) -> bool:
        """
        Send daily summary of patterns.
        
        Args:
            total_patterns: Total patterns detected
            high_quality_patterns: High confidence patterns
            best_opportunities: Top opportunities
            
        Returns:
            True if sent successfully
        """
        opportunities_text = ""
        for i, opp in enumerate(best_opportunities[:5], 1):
            opportunities_text += f"\n{i}. {opp['symbol']} ({opp['timeframe']}): {opp['score']:.1%}"
        
        no_patterns = "\nNo high-quality patterns today"
        message = f"""
*📊 Daily QML Summary*
━━━━━━━━━━━━━━━━━━

📈 *Pattern Statistics*
├ Total Detected: `{total_patterns}`
└ High Quality: `{high_quality_patterns}`

🏆 *Top Opportunities*{opportunities_text if opportunities_text else no_patterns}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}
"""
        return await self.send_message(message.strip())
    
    def send_sync(self, text: str) -> bool:
        """Synchronous send for non-async contexts."""
        return asyncio.run(self.send_message(text))
    
    def send_pattern_alert_sync(self, pattern: QMLPattern) -> bool:
        """Synchronous pattern alert for non-async contexts."""
        return asyncio.run(self.send_pattern_alert(pattern))


def create_telegram_alerts() -> TelegramAlerts:
    """Factory function for TelegramAlerts."""
    return TelegramAlerts()

