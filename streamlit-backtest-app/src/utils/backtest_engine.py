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


class BacktestEngine:
    """
    Enhanced backtesting engine for strategy evaluation
    """
    
    def __init__(self, initial_capital=10000, commission=0.001):
        """
        Initialize the backtest engine
        
        Parameters:
        -----------
        initial_capital : float
            Starting capital for backtesting
        commission : float
            Commission rate per trade (e.g., 0.001 = 0.1%)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        
    def run_backtest(self, data, strategy):
        """
        Run backtest using the provided strategy
        
        Parameters:
        -----------
        data : pandas.DataFrame
            OHLCV price data
        strategy : Strategy object
            Trading strategy with generate_signals method
            
        Returns:
        --------
        pandas.DataFrame
            Backtest results with portfolio performance
        """
        try:
            print(f"🔄 Running backtest with {strategy.name}")
            print(f"📊 Data period: {data.index[0]} to {data.index[-1]}")
            print(f"💰 Initial capital: ${self.initial_capital:,}")
            print(f"📈 Commission: {self.commission*100:.3f}%")
            
            # Generate strategy signals
            df_with_signals = strategy.generate_signals(data)
            
            # Initialize portfolio tracking
            df_with_signals['cash'] = self.initial_capital
            df_with_signals['shares'] = 0.0
            df_with_signals['portfolio_value'] = self.initial_capital
            df_with_signals['trade_value'] = 0.0
            df_with_signals['commission_paid'] = 0.0
            
            current_cash = self.initial_capital
            current_shares = 0.0
            total_commission = 0.0
            
            # Process each trading signal
            for i in range(1, len(df_with_signals)):
                signal = df_with_signals.iloc[i]['signal']
                price = df_with_signals.iloc[i]['Close']
                prev_position = df_with_signals.iloc[i-1]['position']
                current_position = df_with_signals.iloc[i]['position']
                
                trade_value = 0.0
                commission_cost = 0.0
                
                # Execute trades based on position changes
                if current_position != prev_position:
                    if current_position == 1 and prev_position == 0:
                        # Buy: Use all available cash
                        shares_to_buy = current_cash / price
                        commission_cost = shares_to_buy * price * self.commission
                        
                        if current_cash > commission_cost:
                            current_shares = (current_cash - commission_cost) / price
                            trade_value = current_shares * price
                            current_cash = 0.0
                            total_commission += commission_cost
                            
                            print(f"  📈 BUY at {price:.4f}: {current_shares:.2f} shares, commission: ${commission_cost:.2f}")
                    
                    elif current_position == 0 and prev_position == 1:
                        # Sell: Sell all shares
                        if current_shares > 0:
                            trade_value = current_shares * price
                            commission_cost = trade_value * self.commission
                            current_cash = trade_value - commission_cost
                            current_shares = 0.0
                            total_commission += commission_cost
                            
                            print(f"  📉 SELL at {price:.4f}: ${trade_value:.2f}, commission: ${commission_cost:.2f}")
                
                # Update portfolio values
                df_with_signals.iloc[i, df_with_signals.columns.get_loc('cash')] = current_cash
                df_with_signals.iloc[i, df_with_signals.columns.get_loc('shares')] = current_shares
                df_with_signals.iloc[i, df_with_signals.columns.get_loc('trade_value')] = trade_value
                df_with_signals.iloc[i, df_with_signals.columns.get_loc('commission_paid')] = commission_cost
                
                # Calculate total portfolio value
                portfolio_value = current_cash + (current_shares * price)
                df_with_signals.iloc[i, df_with_signals.columns.get_loc('portfolio_value')] = portfolio_value
            
            # Calculate performance metrics
            final_value = df_with_signals['portfolio_value'].iloc[-1]
            total_return = ((final_value - self.initial_capital) / self.initial_capital) * 100
            
            # Count trades
            trades = df_with_signals['signal'].abs().sum()
            
            print(f"\n📊 Backtest Results:")
            print(f"   Final portfolio value: ${final_value:,.2f}")
            print(f"   Total return: {total_return:.2f}%")
            print(f"   Total trades: {trades}")
            print(f"   Total commission paid: ${total_commission:.2f}")
            
            return df_with_signals
            
        except Exception as e:
            print(f"❌ Backtest failed: {e}")
            return None


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