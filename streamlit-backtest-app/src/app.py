import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.backtest_engine import SmaCross, run_backtest, prepare_stats_for_display

st.set_page_config(
    page_title="Trading Strategy Backtester",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Trading Strategy Backtester")
st.markdown("Test your trading strategies with historical data")

# Sidebar for parameters
with st.sidebar:
    st.header("Configuration")
    
    # Stock selection
    symbol = st.text_input("Stock Symbol", value="AAPL", help="Enter a valid stock ticker")
    
    # Date range
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=365))
    with col2:
        end_date = st.date_input("End Date", value=datetime.now())
    
    # Strategy parameters
    st.subheader("Strategy Parameters")
    short_ma = st.slider("Short MA Period", min_value=5, max_value=50, value=10)
    long_ma = st.slider("Long MA Period", min_value=20, max_value=200, value=20)
    
    # Backtest parameters
    st.subheader("Backtest Parameters")
    initial_cash = st.number_input("Initial Cash ($)", min_value=1000, value=10000, step=1000)
    commission = st.slider("Commission (%)", min_value=0.0, max_value=1.0, value=0.2, step=0.1) / 100
    
    run_backtest_btn = st.button("Run Backtest", type="primary")

# Main content area
if run_backtest_btn:
    try:
        # Download data
        with st.spinner(f"Downloading {symbol} data..."):
            data = yf.download(symbol, start=start_date, end=end_date, auto_adjust=False)
        
        if data.empty:
            st.error(f"No data found for symbol {symbol}")
            st.stop()
        
        # Prepare data for backtesting
        data = data.dropna()
        
        # Ensure we have the expected column structure
        if isinstance(data.columns, pd.MultiIndex):
            # Flatten MultiIndex columns if present
            data.columns = [col[0] if col[1] == symbol else col[1] for col in data.columns]
        
        # Create custom strategy class with user parameters
        class CustomSmaCross(SmaCross):
            n1 = short_ma
            n2 = long_ma
        
        # Run backtest
        with st.spinner("Running backtest..."):
            stats, bt = run_backtest(data, CustomSmaCross, initial_cash, commission)
        
        # Display results
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📊 Backtest Results")
            
            # Create price chart with signals
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=('Price & Moving Averages', 'Portfolio Value'),
                row_heights=[0.7, 0.3]
            )
            
            # Price and moving averages
            fig.add_trace(
                go.Scatter(x=data.index, y=data['Close'], name='Close Price', line=dict(color='blue')),
                row=1, col=1
            )
            
            # Calculate MAs for display
            data[f'SMA_{short_ma}'] = data['Close'].rolling(short_ma).mean()
            data[f'SMA_{long_ma}'] = data['Close'].rolling(long_ma).mean()
            
            fig.add_trace(
                go.Scatter(x=data.index, y=data[f'SMA_{short_ma}'], name=f'SMA {short_ma}', line=dict(color='orange')),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=data.index, y=data[f'SMA_{long_ma}'], name=f'SMA {long_ma}', line=dict(color='red')),
                row=1, col=1
            )
            
            # Portfolio value (if available in stats)
            if hasattr(stats, '_equity_curve'):
                equity_curve = stats._equity_curve['Equity']
                fig.add_trace(
                    go.Scatter(x=equity_curve.index, y=equity_curve, name='Portfolio Value', line=dict(color='green')),
                    row=2, col=1
                )
            
            fig.update_layout(height=600, title=f"{symbol} - Strategy Performance")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📈 Performance Metrics")
            
            # Prepare and display stats
            stats_df = prepare_stats_for_display(stats)
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        # Additional metrics in expandable sections
        with st.expander("📋 Detailed Analysis"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Return", f"{stats['Return [%]']:.2f}%")
                st.metric("Buy & Hold Return", f"{stats['Buy & Hold Return [%]']:.2f}%")
            
            with col2:
                st.metric("Sharpe Ratio", f"{stats['Sharpe Ratio']:.3f}")
                st.metric("Max Drawdown", f"{stats['Max. Drawdown [%]']:.2f}%")
            
            with col3:
                st.metric("Win Rate", f"{stats['Win Rate [%]']:.1f}%")
                st.metric("# Trades", f"{stats['# Trades']}")
        
        # Show trades if available
        if hasattr(stats, '_trades') and not stats._trades.empty:
            with st.expander("🔍 Trade Details"):
                trades_df = stats._trades.copy()
                # Convert datetime columns to strings to avoid Arrow errors
                for col in trades_df.columns:
                    if trades_df[col].dtype == 'datetime64[ns]':
                        trades_df[col] = trades_df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                    elif trades_df[col].dtype == 'timedelta64[ns]':
                        trades_df[col] = trades_df[col].astype(str)
                
                st.dataframe(trades_df, use_container_width=True)
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please check your inputs and try again.")

else:
    # Default view when no backtest has been run
    st.info("👈 Configure your strategy parameters in the sidebar and click 'Run Backtest' to start.")
    
    # Show sample data
    st.subheader("📖 How to use this app:")
    st.markdown("""
    1. **Select a stock symbol** (e.g., AAPL, GOOGL, TSLA)
    2. **Choose your date range** for the backtest
    3. **Adjust strategy parameters** (Moving Average periods)
    4. **Set backtest parameters** (initial cash, commission)
    5. **Click 'Run Backtest'** to see results
    
    The strategy uses a Simple Moving Average crossover:
    - **Buy signal**: When short MA crosses above long MA
    - **Sell signal**: When long MA crosses above short MA
    """)