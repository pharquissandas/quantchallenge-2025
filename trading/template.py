"""
Quant Challenge 2025
Algorithmic strategy template
"""
from enum import Enum
from typing import Optional, Dict
import time
import math

class Side(Enum):
    BUY = 0
    SELL = 1

class Ticker(Enum):
    # TEAM_A (home team)
    TEAM_A = 0

def place_market_order(side: Side, ticker: Ticker, quantity: float) -> None:
    """Place a market order.
    
    Parameters
    ----------
    side
        Side of order to place
    ticker
        Ticker of order to place
    quantity
        Quantity of order to place
    """
    return

def place_limit_order(side: Side, ticker: Ticker, quantity: float, price: float, ioc: bool = False) -> int:
    """Place a limit order.
    
    Parameters
    ----------
    side
        Side of order to place
    ticker
        Ticker of order to place
    quantity
        Quantity of order to place
    price
        Price of order to place
    ioc
        Immediate or cancel flag (FOK)

    Returns
    -------
    order_id
        Order ID of order placed
    """
    return 0

def cancel_order(ticker: Ticker, order_id: int) -> bool:
    """Cancel an order.
    
    Parameters
    ----------
    ticker
        Ticker of order to cancel
    order_id
        Order ID of order to cancel

    Returns
    -------
    success
        True if order was cancelled, False otherwise
    """
    return False

class Strategy:
    """Template for a strategy."""

    def reset_state(self) -> None:
        """Reset the state of the strategy to the start of game position.
        
        Since the sandbox execution can start mid-game, we recommend creating a
        function which can be called from __init__ and on_game_event_update (END_GAME).

        Note: In production execution, the game will start from the beginning
        and will not be replayed.
        """
        self.bestBid: Optional[float] = None
        self.bestAsk: Optional[float] = None
        self.midPrice: Optional[float] = None

        self.capitalRemaining: float=100000.0
        self.position: float = 0.0
        self.averageEntryPrice: Optional[float] = None
        self.untappedPNL: float = 0.0
        self.openLimitOrders: Dict[int,Dict] = {}
        self.homeScore: int=0
        self.awayScore: int=0
        self.timeRemaining: Optional[float] = None
        self.gameLength: Optional[float] = None

        # Conservative parameters

        self.maxcapitalperSide: float=0.05
        self.edgeThreshold: float=4.0
        self.maxEdge: float=12.0
        self.minTradeIn: float=50.0
        self.minContractsperTrade: int=1
        self.maxContractsperTrade: int=500
        self.tickSize: float=0.1
        self.cancelTimeLimitOrder: float=5.0
        self.checkStaleOrders: float=1.0
        self.lastCancelCheck: float=time.time()

        self.exitPosition: float=3.0
        self.maxTradesperDay: int=200

        self.tradeNo: int=0
        self.logPrefix = "[Conservative]"
        self.logText("Started conservative strategy")

    def __init__(self) -> None:
        """Your initialization code goes here."""
        self.reset_state()

    def logText(self, message:str) -> None:
        print(f"{self.logPrefix} {message}")

    def updateMidprice(self) -> None:
        if (self.bestBid is not None) and (self.bestAsk is not None):
            self.midPrice = max(0.0, min(100.0, (self.bestBid+self.bestAsk)/2.0))
        elif self.bestBid is not None:
            self.midPrice = self.bestBid
        elif self.bestAsk is not None:
            self.midPrice = self.bestAsk
        else:
            self.midPrice=None 

    def calculateWinProb(self, homeScore: int, awayScore: int, time: Optional[float]) -> float:
        
        # finds difference and remaining time

        return probability
    
    def allowedContracts(self, price: float) -> float:

        return ______

    def on_trade_update(
        self, ticker: Ticker, side: Side, quantity: float, price: float
    ) -> None:
        """Called whenever two orders match. Could be one of your orders, or two other people's orders.
        Parameters
        ----------
        ticker
            Ticker of orders that were matched
        side:
            Side of orders that were matched
        quantity
            Volume traded
        price
            Price that trade was executed at
        """
        print(f"Python Trade update: {ticker} {side} {quantity} shares @ {price}")

    def on_orderbook_update(
        self, ticker: Ticker, side: Side, quantity: float, price: float
    ) -> None:
        """Called whenever the orderbook changes. This could be because of a trade, or because of a new order, or both.
        Parameters
        ----------
        ticker
            Ticker that has an orderbook update
        side
            Which orderbook was updated
        price
            Price of orderbook that has an update
        quantity
            Volume placed into orderbook
        """
        pass

    def on_account_update(
        self,
        ticker: Ticker,
        side: Side,
        price: float,
        quantity: float,
        capital_remaining: float,
    ) -> None:
        """Called whenever one of your orders is filled.
        Parameters
        ----------
        ticker
            Ticker of order that was fulfilled
        side
            Side of order that was fulfilled
        price
            Price that order was fulfilled at
        quantity
            Volume of order that was fulfilled
        capital_remaining
            Amount of capital after fulfilling order
        """
        pass

    def on_game_event_update(self,
                           event_type: str,
                           home_away: str,
                           home_score: int,
                           away_score: int,
                           player_name: Optional[str],
                           substituted_player_name: Optional[str],
                           shot_type: Optional[str],
                           assist_player: Optional[str],
                           rebound_type: Optional[str],
                           coordinate_x: Optional[float],
                           coordinate_y: Optional[float],
                           time_seconds: Optional[float]
        ) -> None:
        """Called whenever a basketball game event occurs.
        Parameters
        ----------
        event_type
            Type of event that occurred
        home_score
            Home team score after event
        away_score
            Away team score after event
        player_name (Optional)
            Player involved in event
        substituted_player_name (Optional)
            Player being substituted out
        shot_type (Optional)
            Type of shot
        assist_player (Optional)
            Player who made the assist
        rebound_type (Optional)
            Type of rebound
        coordinate_x (Optional)
            X coordinate of shot location in feet
        coordinate_y (Optional)
            Y coordinate of shot location in feet
        time_seconds (Optional)
            Game time remaining in seconds
        """

        print(f"{event_type} {home_score} - {away_score}")

        if event_type == "END_GAME":
            # IMPORTANT: Highly recommended to call reset_state() when the
            # game ends. See reset_state() for more details.
            self.reset_state()
            return

