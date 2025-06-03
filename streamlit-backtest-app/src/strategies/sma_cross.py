from backtesting import Strategy
from backtesting.lib import crossover
from backtesting.test import SMA, GOOG

class SmaCross(Strategy):
    n1 = 10
    n2 = 20

    def init(self):
        close = self.data.Close
        self.sma1 = self.I(SMA, close, self.n1)
        self.sma2 = self.I(SMA, close, self.n2)

    def next(self):
        if crossover(self.sma1, self.sma2):
            if self.position:
                self.position.close()
            self.buy(size=0.95)  # Use 95% of available cash to avoid margin issues
        elif crossover(self.sma2, self.sma1):
            if self.position:
                self.position.close()