from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA, GOOG
from strategies.sma_cross import SmaCross
import pandas as pd


def run_backtest(data, strategy_class=SmaCross, cash=10000, commission=0.002):
    """Run backtest and return results"""
    bt = Backtest(data, strategy_class, cash=cash, commission=commission)
    stats = bt.run()
    return stats, bt


def prepare_stats_for_display(stats):
    """Convert stats to display-friendly format"""
    # Convert to dict and handle problematic types
    stats_dict = {}
    for key, value in stats.items():
        if isinstance(value, pd.Timedelta):
            stats_dict[key] = str(value)
        elif hasattr(value, 'strftime'):  # datetime objects
            stats_dict[key] = value.strftime('%Y-%m-%d')
        else:
            stats_dict[key] = value
    
    return pd.DataFrame(list(stats_dict.items()), columns=['Metric', 'Value'])