"""
Moving Average Crossover Strategy
"""
import pandas as pd
import numpy as np


class MovingAverageStrategy:
    """
    Simple Moving Average Crossover Strategy
    
    Generates buy signals when short MA crosses above long MA
    Generates sell signals when short MA crosses below long MA
    """
    
    def __init__(self, short_window=20, long_window=50):
        """
        Initialize the Moving Average strategy
        
        Parameters:
        -----------
        short_window : int
            Period for short-term moving average
        long_window : int
            Period for long-term moving average
        """
        self.short_window = short_window
        self.long_window = long_window
        self.name = f"MA_Cross_{short_window}_{long_window}"
        
    def generate_signals(self, data):
        """
        Generate trading signals based on moving average crossover
        
        Parameters:
        -----------
        data : pandas.DataFrame
            OHLCV data with columns ['Open', 'High', 'Low', 'Close', 'Volume']
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with additional columns:
            - 'short_ma': Short-term moving average
            - 'long_ma': Long-term moving average  
            - 'signal': Trading signal (1=buy, -1=sell, 0=hold)
            - 'position': Current position (1=long, 0=no position)
        """
        df = data.copy()
        
        # Calculate moving averages
        df['short_ma'] = df['Close'].rolling(window=self.short_window).mean()
        df['long_ma'] = df['Close'].rolling(window=self.long_window).mean()
        
        # Initialize signal column
        df['signal'] = 0
        
        # Generate signals where we have enough data
        valid_idx = self.long_window
        
        # Buy signal: short MA crosses above long MA
        df.loc[valid_idx:, 'signal'] = np.where(
            (df['short_ma'][valid_idx:] > df['long_ma'][valid_idx:]) & 
            (df['short_ma'][valid_idx:].shift(1) <= df['long_ma'][valid_idx:].shift(1)), 
            1, 0
        )
        
        # Sell signal: short MA crosses below long MA
        df.loc[valid_idx:, 'signal'] = np.where(
            (df['short_ma'][valid_idx:] < df['long_ma'][valid_idx:]) & 
            (df['short_ma'][valid_idx:].shift(1) >= df['long_ma'][valid_idx:].shift(1)), 
            -1, df.loc[valid_idx:, 'signal']
        )
        
        # Calculate positions
        df['position'] = 0
        current_position = 0
        
        for i in range(valid_idx, len(df)):
            if df.iloc[i]['signal'] == 1:  # Buy signal
                current_position = 1
            elif df.iloc[i]['signal'] == -1:  # Sell signal
                current_position = 0
            
            df.iloc[i, df.columns.get_loc('position')] = current_position
        
        # Add strategy metadata
        df['strategy'] = self.name
        
        return df
    
    def get_parameters(self):
        """Return strategy parameters"""
        return {
            'short_window': self.short_window,
            'long_window': self.long_window,
            'strategy_type': 'Moving Average Crossover'
        }
    
    def __str__(self):
        return f"MovingAverageStrategy(short={self.short_window}, long={self.long_window})"
    
    def __repr__(self):
        return self.__str__()
