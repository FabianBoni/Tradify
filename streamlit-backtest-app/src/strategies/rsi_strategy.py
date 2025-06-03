"""
RSI (Relative Strength Index) Trading Strategy
"""
import pandas as pd
import numpy as np


class RSIStrategy:
    """
    RSI-based Trading Strategy
    
    Generates buy signals when RSI is oversold (< oversold_threshold)
    Generates sell signals when RSI is overbought (> overbought_threshold)
    """
    
    def __init__(self, rsi_period=14, oversold=30, overbought=70):
        """
        Initialize the RSI strategy
        
        Parameters:
        -----------
        rsi_period : int
            Period for RSI calculation (default: 14)
        oversold : float
            RSI threshold for oversold condition (default: 30)
        overbought : float
            RSI threshold for overbought condition (default: 70)
        """
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.name = f"RSI_{rsi_period}_{oversold}_{overbought}"
        
    def calculate_rsi(self, prices, period=14):
        """
        Calculate RSI (Relative Strength Index)
        
        Parameters:
        -----------
        prices : pandas.Series
            Price series (typically closing prices)
        period : int
            Period for RSI calculation
            
        Returns:
        --------
        pandas.Series
            RSI values
        """
        # Calculate price changes
        delta = prices.diff()
        
        # Separate gains and losses
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        # Calculate RS (Relative Strength)
        rs = gain / loss
        
        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def generate_signals(self, data):
        """
        Generate trading signals based on RSI
        
        Parameters:
        -----------
        data : pandas.DataFrame
            OHLCV data with columns ['Open', 'High', 'Low', 'Close', 'Volume']
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with additional columns:
            - 'rsi': RSI values
            - 'signal': Trading signal (1=buy, -1=sell, 0=hold)
            - 'position': Current position (1=long, 0=no position)
        """
        df = data.copy()
        
        # Calculate RSI
        df['rsi'] = self.calculate_rsi(df['Close'], self.rsi_period)
        
        # Initialize signal column
        df['signal'] = 0
        
        # Generate signals where we have enough data
        valid_idx = self.rsi_period
        
        # Buy signal: RSI crosses above oversold threshold from below
        df.loc[valid_idx:, 'signal'] = np.where(
            (df['rsi'][valid_idx:] > self.oversold) & 
            (df['rsi'][valid_idx:].shift(1) <= self.oversold), 
            1, 0
        )
        
        # Sell signal: RSI crosses below overbought threshold from above
        df.loc[valid_idx:, 'signal'] = np.where(
            (df['rsi'][valid_idx:] < self.overbought) & 
            (df['rsi'][valid_idx:].shift(1) >= self.overbought), 
            -1, df.loc[valid_idx:, 'signal']
        )
        
        # Alternative: Simple threshold strategy
        # Uncomment these lines for a simpler approach:
        # df.loc[valid_idx:, 'signal'] = np.where(df['rsi'][valid_idx:] < self.oversold, 1, 0)
        # df.loc[valid_idx:, 'signal'] = np.where(df['rsi'][valid_idx:] > self.overbought, -1, df.loc[valid_idx:, 'signal'])
        
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
            'rsi_period': self.rsi_period,
            'oversold_threshold': self.oversold,
            'overbought_threshold': self.overbought,
            'strategy_type': 'RSI'
        }
    
    def get_rsi_levels(self):
        """Return current RSI levels for analysis"""
        return {
            'oversold': self.oversold,
            'overbought': self.overbought,
            'period': self.rsi_period
        }
    
    def __str__(self):
        return f"RSIStrategy(period={self.rsi_period}, oversold={self.oversold}, overbought={self.overbought})"
    
    def __repr__(self):
        return self.__str__()
