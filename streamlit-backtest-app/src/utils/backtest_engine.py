import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import warnings
warnings.filterwarnings('ignore')


class SmaCross(Strategy):
    """
    Simple Moving Average Crossover Strategy
    """
    n1 = 10  # Short period
    n2 = 20  # Long period
    
    def init(self):
        close = self.data.Close
        self.sma1 = self.I(lambda x: pd.Series(x).rolling(self.n1).mean(), close)
        self.sma2 = self.I(lambda x: pd.Series(x).rolling(self.n2).mean(), close)
    
    def next(self):
        if crossover(self.sma1, self.sma2):
            self.buy()
        elif crossover(self.sma2, self.sma1):
            self.position.close()


class MLStrategy(Strategy):
    """
    Machine Learning based trading strategy
    """
    
    def init(self):
        # Get model info from parameters
        self.model_info = getattr(self._broker._data, 'model_info', None)
        self.confidence_threshold = getattr(self._broker._data, 'confidence_threshold', 0.6)
        self.signals = None
        
        if self.model_info is not None:
            try:
                # Import here to avoid circular imports
                from .ml_engine import generate_ml_signals, create_comprehensive_features
                
                # Create features for the entire dataset
                df_with_features = create_comprehensive_features(self.data.df)
                
                # Generate ML signals
                self.signals = generate_ml_signals(df_with_features, self.model_info, self.confidence_threshold)
                
                print(f"Generated {len(self.signals)} ML signals")
                print(f"Signal distribution: {self.signals['signal'].value_counts().to_dict()}")
                
            except Exception as e:
                print(f"Error generating ML signals: {e}")
                self.signals = None
        
    def next(self):
        if self.signals is None:
            return
            
        current_idx = len(self.data) - 1
        if current_idx >= len(self.signals):
            return
            
        signal = self.signals.iloc[current_idx]['signal']
        confidence = self.signals.iloc[current_idx]['confidence']
        
        # Only trade with sufficient confidence
        if confidence < self.confidence_threshold:
            return
            
        if signal == 1 and not self.position:
            # Buy signal
            self.buy()
        elif signal == -1 and self.position:
            # Sell signal
            self.position.close()


def run_backtest(data, strategy_class, initial_cash=10000, commission=0.002, **kwargs):
    """
    Run backtest with given strategy and parameters
    """
    try:
        # Ensure data has the correct format for backtesting
        if data.empty:
            raise Exception("No data provided for backtesting")
        
        # Validate required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise Exception(f"Missing required columns: {missing_columns}")
        
        # Remove any NaN values
        data_clean = data.dropna()
        
        if len(data_clean) < 10:
            raise Exception(f"Insufficient data for backtesting: {len(data_clean)} points")
        
        print(f"📊 Running backtest on {len(data_clean)} data points")
        print(f"📅 Period: {data_clean.index.min().date()} to {data_clean.index.max().date()}")
        
        # Attach additional parameters to data for ML strategy
        if 'model_info' in kwargs:
            data_clean.model_info = kwargs['model_info']
        if 'confidence_threshold' in kwargs:
            data_clean.confidence_threshold = kwargs['confidence_threshold']
        
        # Initialize backtest with enhanced parameters
        bt = Backtest(
            data_clean, 
            strategy_class,
            cash=initial_cash,
            commission=commission,
            exclusive_orders=True
        )
        
        # Run backtest
        stats = bt.run()
        
        print(f"✅ Backtest completed successfully")
        return stats, bt
    
    except Exception as e:
        raise Exception(f"Backtest failed: {str(e)}")


def prepare_stats_for_display(stats):
    """
    Prepare backtest statistics for display in Streamlit
    """
    try:
        # Convert stats to DataFrame for display
        stats_dict = {
            'Metric': [],
            'Value': []
        }
        
        # Key metrics to display
        key_metrics = [
            'Return [%]',
            'Buy & Hold Return [%]',
            'Sharpe Ratio',
            'Max. Drawdown [%]',
            'Win Rate [%]',
            '# Trades',
            'Avg. Trade [%]',
            'Max. Trade [%]',
            'Min. Trade [%]'
        ]
        
        for metric in key_metrics:
            if metric in stats:
                value = stats[metric]
                if isinstance(value, float):
                    if '%' in metric:
                        stats_dict['Value'].append(f"{value:.2f}%")
                    else:
                        stats_dict['Value'].append(f"{value:.3f}")
                else:
                    stats_dict['Value'].append(str(value))
                stats_dict['Metric'].append(metric)
        
        return pd.DataFrame(stats_dict)
    
    except Exception as e:
        print(f"Error preparing stats: {e}")
        return pd.DataFrame({'Metric': ['Error'], 'Value': [str(e)]})