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

        self.homeScore: int=0
        self.awayScore: int=0
        self.timeRemaining: Optional[float] = None
        self.gameLength: Optional[float] = 2400.0

        # Conservative parameters

        self.maxcapitalperSide: float=0.50
        self.baseTrade: float=500.0
        self.minContractsperTrade: int=1
        self.maxContractsperTrade: int=5000
        self.maxPosition: float=0.9*self.capitalRemaining
        self.tickSize: float=0.1

        self.minEdge: float=0.5
        self.kellyShrink: float=0.60
        self.maxKellyFraction: float=0.70
        self.lateGameTime: float=120.0
        self.lateGameMultiplier: float=6.0

        self.layers = [0.0,-0.2,0.2,-0.5,0.5]
        self.openLimitOrders: Dict[int,Dict] = {}
        self.syntheticNextOID = 1

        self.initialCapital = self.capitalRemaining
        self.maxDrawdown = 0.6
        self.stopTradingFlag = False

        self.tradeNo: int=0
        self.logPrefix: str = "[High Leverage]"
        self.logText("Started aggressive strategy")

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

    def updatePNL(self) -> None:
        if self.position == 0 or self.midPrice is None or self.averageEntryPrice is None:
            self.untappedPNL=0.0
            return
        self.untappedPNL = (self.midPrice-self.averageEntryPrice)*self.position

    def calculateWinProb(self, homeScore: int, awayScore: int, timeSeconds: Optional[float]) -> float:
        difference = homeScore - awayScore
        if timeSeconds is None:
            timeSeconds=self.gameLength
        timeFraction = max(1e-6, timeSeconds/self.gameLength)
        weighting = 1.0/(0.15+(timeFraction*3.5))
        raw=0.6*difference*weighting
        probability = 50.0+(30.0*math.tanh(raw/8.0))
        return max(0.0, min(100.0,probability))
    
    def allowedContracts(self, price: float) -> float:
        allowedMoney = max(0.0,self.capitalRemaining*self.maxcapitalperSide)
        if price<=0.0:
            return self.minContractsperTrade
        maxCapital = int(math.floor(allowedMoney/price))
        maxCapital = max(self.minContractsperTrade, min(self.maxContractsperTrade, maxCapital))
        return maxCapital

    def kellyFraction(self, edge: float) -> float:
        if edge<=0:
            return 0.0
        eV = edge/100.0
        rawF = eV/0.5
        fraction = rawF*self.kellyShrink
        return max(0.0, min(self.maxKellyFraction,fraction))
    
    def computeOrderQTY(self, edge:float, price:float) -> int:
        if price <= 0:
            return self.minContractsperTrade
        fraction = self.kellyFraction(edge)
        if (self.timeRemaining<=self.lateGameTime) and (self.timeRemaining is not None):
            fraction = fraction*self.lateGameMultiplier
            fraction = min(fraction, self.maxKellyFraction*self.lateGameMultiplier)
        allowedCapital = min(self.capitalRemaining*self.maxcapitalperSide, self.maxPosition)
        money = max(self.baseTrade, allowedCapital*fraction)
        quantity = int(math.floor(money/price))
        quantity = max(self.minContractsperTrade, min(self.maxContractsperTrade,quantity))
        if self.capitalRemaining < quantity*price:
            quantity = int(max(self.minContractsperTrade, math.floor(self.capitalRemaining/max(price,1e-6))))
        return int(quantity)
    
    def layerLimits(self, side: Side, totalQuantity: int) -> None:
        if self.midPrice is None or totalQuantity<=0:
            return
        perOrder = max(1, int(math.ceil(totalQuantity/len(self.layers))))
        for offset in self.layers:
            price = round((self.midPrice+offset)/self.tickSize)*self.tickSize
            price = max(0.0, min(100.0,price))
            try:
                oid = place_limit_order(side, Ticker.TEAM_A, float(perOrder), float(price), ioc=False)
            except Exception:
                oid = 0
            if not oid:
                oid = self.syntheticNextOID
                self.syntheticNextOID=self.syntheticNextOID+1
            self.openLimitOrders[oid] = {"side": side, "quantity": perOrder, "price": price, "time":time.time()}

    def cancelAllLimits(self) -> None:
        for oid in list(self.openLimitOrders.keys()):
            try:
                cancel_order(Ticker.TEAM_A,oid)
            except Exception:
                pass
            self.openLimitOrders.pop(oid,None)

    def on_trade_update(self, ticker: Ticker, side: Side, quantity: float, price: float) -> None:
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
        self.logText(f"Trade update incoming: {ticker} {side} {quantity} at {price}")

    def on_orderbook_update(self, ticker: Ticker, side: Side, quantity: float, price: float) -> None:
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
        try:
            if side==Side.SELL:
                if quantity>0 and (self.bestAsk is None or price<self.bestAsk):
                    self.bestAsk=price
                elif self.bestAsk==price and quantity==0:
                    self.bestAsk=0
            elif side==Side.BUY:
                if quantity>0 and (self.bestBid is None or price>self.bestBid):
                    self.bestBid=price
                elif self.bestBid==price and quantity==0:
                    self.bestBid=None
            self.updateMidprice()
            self.updatePNL()
        except Exception as a:
            self.logText(f"Error in orderbook update: {a}")
    
    def on_account_update(self,ticker: Ticker, side: Side, price: float, quantity: float, capital_remaining: float) -> None:
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
        try:
            previousPosition = self.position
            if side == Side.BUY:
                self.position=self.position+quantity
                if previousPosition==0 or self.averageEntryPrice is None:
                    self.averageEntryPrice=price
                else:
                    self.averageEntryPrice=((self.averageEntryPrice*abs(previousPosition))+(price*quantity))/(abs(previousPosition)+quantity)
            else:
                self.position=self.position-quantity
                if self.position==0:
                    self.averageEntryPrice=None
                elif self.position<0:
                    self.averageEntryPrice=price
            self.capitalRemaining=capital_remaining
            self.updatePNL()
            if self.capitalRemaining<self.initialCapital*(1.0-self.maxDrawdown):
                self.stopTradingFlag=True
                self.logText("stop trading fla one drawdown breach")
        except Exception as a:
            self.logText(f"on account update error: {a}")

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
                           time_seconds: Optional[float]) -> None:
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
        try:
            self.homeScore = home_score
            self.awayScore = away_score
            self.timeRemaining = time_seconds

            if event_type == "END_GAME":
                if self.position != 0:
                    side = Side.SELL if self.position>0 else Side.BUY
                    quantity = int(abs(self.position))
                    try:
                        place_market_order(side, Ticker.TEAM_A, float(quantity))
                    except Exception:
                        pass
                self.cancelAllLimits()
                self.reset_state()
                return
            
            if self.stopTradingFlag:
                return
            if self.midPrice is None:
                return
            
            fairProbability = self.calculateWinProb(home_score, away_score, time_seconds)
            edge = fairProbability-self.midPrice
            absEdge = abs(edge)

            if absEdge>=max(self.minEdge*2.0,1.0):
                quantity=self.computeOrderQTY(edge,self.midPrice)
                if quantity<=0:
                    return
                side = Side.BUY if edge > 0 else Side.SELL
                expectedExposure = abs((self.position+(quantity if side == Side.BUY else -quantity)))*self.midPrice
                if expectedExposure<=max(self.maxPosition,self.capitalRemaining*self.maxcapitalperSide):
                    if side == Side.BUY and self.capitalRemaining>=quantity*self.midPrice:
                        place_market_order(side, Ticker.TEAM_A, float(quantity))
                        self.tradeNo+=1
                        self.logText(f"aggressive market {side.name} quantity={quantity} mid={self.midPrice:.2f} fair={fairProbability:.2f} edge={edge:.2f}")
                    elif side == Side.SELL:
                        if self.position>0:
                            place_market_order(side,Ticker.TEAM_A, float(min(quantity, int(self.position))))
                            self.tradeNo+=1
                            self.logText(f"aggressive market {side.name} quantity={quantity} mid={self.midPrice:.2f} fair={fairProbability:.2f} edge={edge:.2f}")
                else:
                    self.logText("aggressive strategy prevented by exposure cap")

            elif absEdge>=self.minEdge:
                quantity=self.computeOrderQTY(edge, self.midPrice)
                side = Side.BUY if edge>0 else Side.SELL
                self.logText(f"layered limits {side.name} quantity={quantity} mid={self.midPrice:.2f} fair={fairProbability:.2f} edge={edge:.2f}")
                self.layerLimits(side, quantity)
            else:
                microQuantity= max(1, int(self.baseTrade//max(self.midPrice,1.0)))
                side = Side.BUY if edge>0 else Side.SELL
                self.layerLimits(side, min(microQuantity,3))
            
            currentTime = time.time()
            stale = [oid for oid, info in self.openLimitOrders.items() if currentTime - info.get("time", currentTime)>8.0]
            for oid in stale:
                try:
                    cancel_order(Ticker.TEAM_A,oid)
                except Exception:
                    pass
                self.openLimitOrders.pop(oid,None)

        except Exception as a:
            self.logText(f"Error in on game event: {a}")
        
strategy = Strategy()