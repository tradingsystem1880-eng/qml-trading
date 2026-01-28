"""
Bybit Testnet Client
====================
CCXT wrapper for Bybit testnet connectivity.

Provides:
- Account balance fetching
- Market data (OHLCV, ticker)
- Order placement and management
- Position management

Note: This uses SANDBOX mode for paper trading safety.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd

try:
    import ccxt
except ImportError:
    raise ImportError("ccxt is required. Install with: pip install ccxt")

from .models import (
    OrderSide,
    OrderType,
    OrderStatus,
    PositionSide,
    Order,
    Position,
    AccountBalance,
)

logger = logging.getLogger(__name__)


class BybitTestnetClient:
    """
    CCXT-based client for Bybit testnet.

    Usage:
        client = BybitTestnetClient(api_key, api_secret)
        balance = client.get_balance()
        candles = client.fetch_ohlcv("BTC/USDT", "4h", limit=100)
    """

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        testnet: bool = True,
    ):
        """
        Initialize Bybit client.

        Args:
            api_key: Bybit API key (required for trading)
            api_secret: Bybit API secret (required for trading)
            testnet: Use testnet (default True for safety)
        """
        self.testnet = testnet

        # Initialize CCXT exchange
        self.exchange = ccxt.bybit({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'linear',  # USDT perpetuals
            },
        })

        # Enable testnet mode
        if testnet:
            self.exchange.set_sandbox_mode(True)
            logger.info("Bybit client initialized in TESTNET mode")
        else:
            logger.warning("Bybit client initialized in LIVE mode - use with caution!")

        self._markets_loaded = False

    def _ensure_markets_loaded(self):
        """Load markets if not already loaded."""
        if not self._markets_loaded:
            self.exchange.load_markets()
            self._markets_loaded = True

    # ========== Market Data Methods ==========

    def fetch_ticker(self, symbol: str) -> Dict:
        """
        Fetch current ticker data.

        Returns:
            Dict with: last, bid, ask, high, low, volume, timestamp
        """
        self._ensure_markets_loaded()
        ticker = self.exchange.fetch_ticker(symbol)
        return {
            'symbol': symbol,
            'last': ticker['last'],
            'bid': ticker['bid'],
            'ask': ticker['ask'],
            'high': ticker['high'],
            'low': ticker['low'],
            'volume': ticker['baseVolume'],
            'timestamp': datetime.fromtimestamp(ticker['timestamp'] / 1000),
        }

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "4h",
        limit: int = 100,
        since: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV candle data.

        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            timeframe: Candle interval (e.g., "1h", "4h", "1d")
            limit: Number of candles to fetch
            since: Start timestamp in milliseconds

        Returns:
            DataFrame with columns: time, open, high, low, close, volume
        """
        self._ensure_markets_loaded()

        ohlcv = self.exchange.fetch_ohlcv(
            symbol,
            timeframe=timeframe,
            since=since,
            limit=limit,
        )

        df = pd.DataFrame(
            ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        df['time'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.drop('timestamp', axis=1)

        return df[['time', 'open', 'high', 'low', 'close', 'volume']]

    def fetch_funding_rate(self, symbol: str) -> Dict:
        """
        Fetch current funding rate.

        Returns:
            Dict with: rate, next_funding_time
        """
        self._ensure_markets_loaded()

        try:
            funding = self.exchange.fetch_funding_rate(symbol)
            return {
                'symbol': symbol,
                'rate': funding.get('fundingRate', 0),
                'next_funding_time': funding.get('fundingTimestamp'),
            }
        except Exception as e:
            logger.warning(f"Failed to fetch funding rate for {symbol}: {e}")
            return {'symbol': symbol, 'rate': 0, 'next_funding_time': None}

    # ========== Account Methods ==========

    def get_balance(self) -> AccountBalance:
        """
        Fetch account balance.

        Returns:
            AccountBalance with equity, available, margin info
        """
        self._ensure_markets_loaded()

        balance = self.exchange.fetch_balance()

        # For USDT linear perpetuals
        usdt = balance.get('USDT', {})

        return AccountBalance(
            total_equity=usdt.get('total', 0) or 0,
            available_balance=usdt.get('free', 0) or 0,
            margin_used=usdt.get('used', 0) or 0,
            unrealized_pnl=0,  # Would need to sum from positions
            currency='USDT',
            timestamp=datetime.now(),
        )

    def get_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """
        Fetch open positions.

        Args:
            symbol: Optional symbol filter

        Returns:
            List of Position objects
        """
        self._ensure_markets_loaded()

        if symbol:
            positions = self.exchange.fetch_positions([symbol])
        else:
            positions = self.exchange.fetch_positions()

        result = []
        for pos in positions:
            if pos['contracts'] == 0:
                continue

            side = PositionSide.LONG if pos['side'] == 'long' else PositionSide.SHORT

            result.append(Position(
                symbol=pos['symbol'],
                side=side,
                entry_price=pos['entryPrice'] or 0,
                quantity=abs(pos['contracts'] or 0),
                unrealized_pnl=pos['unrealizedPnl'] or 0,
                leverage=pos['leverage'] or 1,
                liquidation_price=pos.get('liquidationPrice'),
                mark_price=pos.get('markPrice'),
                margin_type=pos.get('marginMode', 'isolated'),
            ))

        return result

    # ========== Order Methods ==========

    def create_market_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        reduce_only: bool = False,
    ) -> Order:
        """
        Create a market order.

        Args:
            symbol: Trading pair
            side: BUY or SELL
            quantity: Order size in base currency
            reduce_only: Only reduce position (for closing)

        Returns:
            Order object
        """
        self._ensure_markets_loaded()

        params = {'reduceOnly': reduce_only}

        order = self.exchange.create_order(
            symbol=symbol,
            type='market',
            side=side.value,
            amount=quantity,
            params=params,
        )

        return self._parse_order(order)

    def create_limit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
        reduce_only: bool = False,
    ) -> Order:
        """
        Create a limit order.

        Args:
            symbol: Trading pair
            side: BUY or SELL
            quantity: Order size in base currency
            price: Limit price
            reduce_only: Only reduce position

        Returns:
            Order object
        """
        self._ensure_markets_loaded()

        params = {'reduceOnly': reduce_only}

        order = self.exchange.create_order(
            symbol=symbol,
            type='limit',
            side=side.value,
            amount=quantity,
            price=price,
            params=params,
        )

        return self._parse_order(order)

    def create_stop_loss(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        stop_price: float,
    ) -> Order:
        """
        Create a stop loss order.

        Args:
            symbol: Trading pair
            side: SELL for long position, BUY for short position
            quantity: Order size
            stop_price: Stop trigger price

        Returns:
            Order object
        """
        self._ensure_markets_loaded()

        # Bybit uses conditional orders for SL/TP
        params = {
            'stopPrice': stop_price,
            'reduceOnly': True,
        }

        order = self.exchange.create_order(
            symbol=symbol,
            type='market',
            side=side.value,
            amount=quantity,
            params=params,
        )

        return self._parse_order(order)

    def create_take_profit(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        take_profit_price: float,
    ) -> Order:
        """
        Create a take profit order.

        Args:
            symbol: Trading pair
            side: SELL for long position, BUY for short position
            quantity: Order size
            take_profit_price: Take profit trigger price

        Returns:
            Order object
        """
        self._ensure_markets_loaded()

        params = {
            'takeProfitPrice': take_profit_price,
            'reduceOnly': True,
        }

        order = self.exchange.create_order(
            symbol=symbol,
            type='market',
            side=side.value,
            amount=quantity,
            params=params,
        )

        return self._parse_order(order)

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """
        Cancel an open order.

        Returns:
            True if cancelled successfully
        """
        self._ensure_markets_loaded()

        try:
            self.exchange.cancel_order(order_id, symbol)
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    def cancel_all_orders(self, symbol: str) -> int:
        """
        Cancel all open orders for a symbol.

        Returns:
            Number of orders cancelled
        """
        self._ensure_markets_loaded()

        try:
            result = self.exchange.cancel_all_orders(symbol)
            return len(result) if isinstance(result, list) else 1
        except Exception as e:
            logger.error(f"Failed to cancel orders for {symbol}: {e}")
            return 0

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """
        Fetch open orders.

        Returns:
            List of Order objects
        """
        self._ensure_markets_loaded()

        orders = self.exchange.fetch_open_orders(symbol)
        return [self._parse_order(o) for o in orders]

    def get_order(self, order_id: str, symbol: str) -> Optional[Order]:
        """
        Fetch a specific order by ID.

        Returns:
            Order object or None if not found
        """
        self._ensure_markets_loaded()

        try:
            order = self.exchange.fetch_order(order_id, symbol)
            return self._parse_order(order)
        except Exception as e:
            logger.error(f"Failed to fetch order {order_id}: {e}")
            return None

    def _parse_order(self, order: Dict) -> Order:
        """Parse CCXT order dict to Order object."""
        status_map = {
            'open': OrderStatus.OPEN,
            'closed': OrderStatus.FILLED,
            'canceled': OrderStatus.CANCELED,
            'rejected': OrderStatus.REJECTED,
            'expired': OrderStatus.EXPIRED,
        }

        type_map = {
            'market': OrderType.MARKET,
            'limit': OrderType.LIMIT,
        }

        return Order(
            id=order['id'],
            symbol=order['symbol'],
            side=OrderSide.BUY if order['side'] == 'buy' else OrderSide.SELL,
            order_type=type_map.get(order['type'], OrderType.MARKET),
            price=order.get('price'),
            quantity=order['amount'],
            status=status_map.get(order['status'], OrderStatus.OPEN),
            created_at=datetime.fromtimestamp(order['timestamp'] / 1000) if order.get('timestamp') else datetime.now(),
            filled_at=datetime.fromtimestamp(order['lastTradeTimestamp'] / 1000) if order.get('lastTradeTimestamp') else None,
            filled_price=order.get('average'),
            filled_quantity=order.get('filled', 0),
            fee=order.get('fee', {}).get('cost', 0) if order.get('fee') else 0,
            fee_currency=order.get('fee', {}).get('currency', 'USDT') if order.get('fee') else 'USDT',
            client_order_id=order.get('clientOrderId', ''),
        )

    # ========== Utility Methods ==========

    def get_min_order_size(self, symbol: str) -> float:
        """Get minimum order size for a symbol."""
        self._ensure_markets_loaded()

        market = self.exchange.market(symbol)
        return market.get('limits', {}).get('amount', {}).get('min', 0.001)

    def get_price_precision(self, symbol: str) -> int:
        """Get price precision (decimal places) for a symbol."""
        self._ensure_markets_loaded()

        market = self.exchange.market(symbol)
        return market.get('precision', {}).get('price', 2)

    def get_quantity_precision(self, symbol: str) -> int:
        """Get quantity precision (decimal places) for a symbol."""
        self._ensure_markets_loaded()

        market = self.exchange.market(symbol)
        return market.get('precision', {}).get('amount', 3)

    def round_price(self, symbol: str, price: float) -> float:
        """Round price to symbol's precision."""
        precision = self.get_price_precision(symbol)
        return round(price, precision)

    def round_quantity(self, symbol: str, quantity: float) -> float:
        """Round quantity to symbol's precision."""
        precision = self.get_quantity_precision(symbol)
        return round(quantity, precision)

    def is_connected(self) -> bool:
        """Check if exchange connection is working."""
        try:
            self.exchange.fetch_time()
            return True
        except Exception:
            return False
