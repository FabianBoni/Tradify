import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta, date
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.backtest_engine import SmaCross, MLStrategy, run_backtest, prepare_stats_for_display
from utils.ml_engine import (
    download_market_data, create_comprehensive_features, train_ml_model,
    fetch_training_data, run_training_pipeline, load_trained_model, 
    train_xgboost_model, prepare_ml_dataset, fetch_cryptocompare_data
)
from utils.data_fetcher import DataFetcher, fetch_cryptocompare_data

st.set_page_config(
    page_title="Trading Strategy Backtester",
    page_icon="📈",
    layout="wide"
)

# Add tabs for different functionalities
tab1, tab2 = st.tabs(["📈 Backtest", "🧠 Modellentwicklung"])

with tab1:
    st.title("📈 Trading Strategy Backtester")
    st.markdown("Test your trading strategies with historical data")

    # Main configuration in content area instead of sidebar
    st.subheader("🎛️ Configuration")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # API Configuration
        with st.expander("🔧 API Settings"):
            api_key = st.text_input("CryptoCompare API Key (Optional)", type="password", 
                                   help="Get free API key from cryptocompare.com")
            use_crypto_data = st.checkbox("Force CryptoCompare API", 
                                        help="Use CryptoCompare even for traditional assets")
        
        # Enhanced Asset Selection
        st.markdown("**💰 Asset Selection**")
        
        asset_col1, asset_col2 = st.columns(2)
        
        with asset_col1:
            # Asset category selection
            asset_category = st.selectbox(
                "Asset Category",
                ["🪙 Crypto", "📈 Stocks", "💰 Forex", "⚡ Custom"],
                help="Choose category for pre-defined options"
            )
            
            # Crypto symbols mapping
            crypto_symbols = {
                "Bitcoin (BTC)": "BTC",
                "Ethereum (ETH)": "ETH", 
                "Ripple (XRP)": "XRP",
                "Cardano (ADA)": "ADA",
                "Dogecoin (DOGE)": "DOGE",
                "Litecoin (LTC)": "LTC"
            }
            
            # Stock symbols mapping
            stock_symbols = {
                "Apple (AAPL)": "AAPL",
                "Microsoft (MSFT)": "MSFT",
                "Google (GOOGL)": "GOOGL", 
                "Tesla (TSLA)": "TSLA",
                "Amazon (AMZN)": "AMZN",
                "Netflix (NFLX)": "NFLX"
            }
            
            if asset_category == "🪙 Crypto":
                symbol_display = st.selectbox("Select Cryptocurrency", list(crypto_symbols.keys()))
                symbol = crypto_symbols[symbol_display]
            elif asset_category == "📈 Stocks":
                symbol_display = st.selectbox("Select Stock", list(stock_symbols.keys()))
                symbol = stock_symbols[symbol_display]
            elif asset_category == "💰 Forex":
                symbol = st.selectbox("Select Forex Pair", ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"])
            else:  # Custom
                symbol = st.text_input("Custom Symbol", value="BTC", 
                                     help="Enter any symbol (e.g., BTC, AAPL, EURUSD)")
        
        with asset_col2:
            # Date range selection
            st.markdown("**📅 Date Range**")
            
            # Preset date ranges
            date_preset = st.selectbox(
                "Quick Preset",
                ["Custom", "1 Month", "3 Months", "6 Months", "1 Year", "2 Years", "Max Available"]
            )
            
            if date_preset == "Custom":
                start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=365))
                end_date = st.date_input("End Date", value=datetime.now())
            else:
                end_date = datetime.now()
                if date_preset == "1 Month":
                    start_date = end_date - timedelta(days=30)
                elif date_preset == "3 Months":
                    start_date = end_date - timedelta(days=90)
                elif date_preset == "6 Months":
                    start_date = end_date - timedelta(days=180)
                elif date_preset == "1 Year":
                    start_date = end_date - timedelta(days=365)
                elif date_preset == "2 Years":
                    start_date = end_date - timedelta(days=730)
                else:  # Max Available
                    start_date = datetime(2010, 1, 1)
                
                st.info(f"📅 Selected: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            
            # Interval selection
            interval = st.selectbox("Data Interval", ["1d", "1h", "4h", "1wk"], 
                                  help="Higher frequency = more detailed analysis")
    
    with col2:
        # Strategy Configuration
        st.markdown("**🎯 Strategy Configuration**")
        
        strategy_type = st.selectbox(
            "Strategy Type",
            ["Simple Moving Average", "Machine Learning", "ML with Pre-trained Model"],
            help="Choose your trading strategy"
        )
        
        if strategy_type == "Simple Moving Average":
            st.markdown("**📊 SMA Parameters**")
            short_window = st.slider("Short MA Period", min_value=5, max_value=50, value=20)
            long_window = st.slider("Long MA Period", min_value=20, max_value=200, value=50)
            
            if short_window >= long_window:
                st.warning("⚠️ Short MA period should be less than Long MA period")
                
        elif strategy_type == "Machine Learning":
            st.markdown("**🤖 ML Training Parameters**")
            train_start = st.date_input("Training Start", 
                                      value=datetime.now() - timedelta(days=1095), 
                                      key="ml_train_start")
            train_end = st.date_input("Training End", 
                                    value=datetime.now() - timedelta(days=30), 
                                    key="ml_train_end")
            
            # Validate training dates
            date_validation_errors = []
            if train_start >= train_end:
                date_validation_errors.append("Training start must be before training end")
            if (train_end - train_start).days < 30:
                date_validation_errors.append("Training period should be at least 30 days")
            if train_end >= datetime.now().date():
                date_validation_errors.append("Training end should be in the past")
                
            if date_validation_errors:
                for error in date_validation_errors:
                    st.error(f"❌ {error}")
            
            confidence_threshold = st.slider("Confidence Threshold", min_value=0.5, max_value=0.9, value=0.6, step=0.05)
            
            # ML model parameters
            st.markdown("**⚙️ Model Parameters**")
            target_type = st.selectbox("Target Type", ["price_diff_pct", "price_direction"], key="backtest_target")
            horizon = st.slider("Prediction Horizon", min_value=1, max_value=12, value=1, key="backtest_horizon")
            model_type = st.selectbox("Model Type", ["regression", "classification"], key="backtest_model_type")
            
        else:  # ML with Pre-trained Model
            st.markdown("**📁 Pre-trained Model**")
            model_file = st.file_uploader("Upload Model File", type=['pkl', 'joblib'])
            confidence_threshold = st.slider("Confidence Threshold", min_value=0.5, max_value=0.9, value=0.6, step=0.05)
            
            # Option to use session state model
            if 'trained_model' in st.session_state:
                use_session_model = st.checkbox("Use Last Trained Model", 
                                               help="Use the model from the ML Training tab")
            else:
                use_session_model = False
            date_validation_errors = []  # No validation errors for pre-trained model
        
        # Backtest parameters
        st.markdown("**💰 Backtest Parameters**")
        initial_cash = st.number_input("Initial Cash ($)", min_value=1000, value=10000, step=1000)
        commission = st.slider("Commission (%)", min_value=0.0, max_value=1.0, value=0.2, step=0.1) / 100
        
        # Disable run button if there are validation errors for ML strategy
        can_run_backtest = True
        if strategy_type == "Machine Learning" and date_validation_errors:
            can_run_backtest = False
            
        run_backtest_btn = st.button("🚀 Run Backtest", type="primary", disabled=not can_run_backtest)

    # Add enhanced data coverage information
    with st.expander("📊 Data Coverage & Quality Information"):
        st.markdown("""
        **Data Sources & Coverage:**
        - **CryptoCompare API**: 
          - ✅ Comprehensive crypto data (BTC from 2013, most alts from 2017+)
          - ✅ Multiple intervals: 1h, 1d, 1wk
          - ✅ High-quality OHLCV data
          - ⚠️ Rate limits: 100k calls/month (free), 100 calls/second

        - **Yahoo Finance**: 
          - ✅ Stocks, indices, ETFs from 1970s+
          - ✅ Some major crypto pairs (BTC-USD, ETH-USD)
          - ✅ No rate limits
          - ⚠️ Limited crypto coverage

        **Optimization Tips:**
        - **Symbols**: Use major symbols for maximum historical depth
        - **Intervals**: '1d' for maximum history, '1h' for recent detailed data
                
        **New Features:**
        - Automatic pagination retrieves complete historical datasets
        - Intelligent fallback between CryptoCompare and Yahoo Finance
        - Enhanced coverage analysis and validation
        """)

# Tab 2: Modellentwicklung
with tab2:
    try:
        from components.model_development import render_model_development
        render_model_development()
    except ImportError as e:
        st.error(f"Fehler beim Laden des Modellentwicklungs-Moduls: {e}")
        st.markdown("""
        Das erweiterte Modellentwicklungs-Interface ist nicht verfügbar.
        
        **Benötigte Komponenten:**
        - `components/model_development.py`
        
        **Funktionen des erweiterten Interfaces:**
        - 🤖 Vollautomatische ML-Pipeline
        - 📊 Echtzeit-Monitoring mit Fortschrittsbalken
        - 🔧 Intelligente Feature-Engineering
        - 📈 Experiment-Tracking und Vergleich
        - 📝 Automatische Dokumentation
        """)

# Run backtest logic
if run_backtest_btn and can_run_backtest:
    with st.spinner("Running backtest..."):
        try:
            # Initialize data fetcher
            fetcher = DataFetcher(api_key=api_key if api_key else None)
            
            # Fetch data
            if use_crypto_data or symbol in ['BTC', 'ETH', 'XRP', 'ADA', 'DOGE', 'LTC']:
                df = fetcher.fetch_cryptocompare_data(symbol, start_date, end_date, interval)
            else:
                df = fetcher.fetch_yahoo_data(symbol, start_date, end_date, interval)
            
            if df is None or df.empty:
                st.error("❌ Failed to fetch data. Please check symbol and date range.")
                st.stop()
            
            # Display data info
            st.success(f"✅ Fetched {len(df)} data points for {symbol}")
            with st.expander("📊 Data Preview"):
                st.dataframe(df.head(10))
                
                # Data quality metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Records", len(df))
                with col2:
                    missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
                    st.metric("Missing Data", f"{missing_pct:.1f}%")
                with col3:
                    date_range_days = (df.index[-1] - df.index[0]).days
                    st.metric("Date Range", f"{date_range_days} days")
            
            # Prepare strategy
            if strategy_type == "Simple Moving Average":
                strategy = SmaCross(short_window=short_window, long_window=long_window)
                
            elif strategy_type == "Machine Learning":
                # Train ML model
                st.info("🔄 Training ML model...")
                
                # Convert dates for training
                train_start_dt = datetime.combine(train_start, datetime.min.time())
                train_end_dt = datetime.combine(train_end, datetime.min.time())
                
                # Train model
                model_info = run_training_pipeline(
                    symbol=symbol,
                    start_date=train_start_dt,
                    end_date=train_end_dt,
                    interval=interval,
                    target_type=target_type,
                    horizon=horizon,
                    model_type=model_type,
                    api_key=api_key
                )
                
                if model_info is None:
                    st.error("❌ Model training failed")
                    st.stop()
                
                strategy = MLStrategy(
                    model_info=model_info,
                    confidence_threshold=confidence_threshold
                )
                
                st.success(f"✅ Model trained successfully! Score: {model_info['metrics'].get('test_r2', model_info['metrics'].get('test_accuracy', 0)):.3f}")
                
            elif strategy_type == "ML with Pre-trained Model":
                if use_session_model and 'trained_model' in st.session_state:
                    model_info = st.session_state.trained_model
                    st.info("Using model from session state")
                elif model_file:
                    # Load uploaded model
                    model_info = load_trained_model(model_file)
                    if model_info is None:
                        st.error("❌ Failed to load model file")
                        st.stop()
                else:
                    st.error("❌ Please upload a model file or train a model first")
                    st.stop()
                
                strategy = MLStrategy(
                    model_info=model_info,
                    confidence_threshold=confidence_threshold
                )
            
            # Run backtest
            results = run_backtest(df, strategy, initial_cash, commission)
            
            if results is None:
                st.error("❌ Backtest execution failed")
                st.stop()
            
            # Display results
            st.subheader("📊 Backtest Results")
            
            # Prepare stats for display
            stats = prepare_stats_for_display(results, initial_cash)
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Return", f"{stats['total_return']:.1%}", 
                         f"{stats['total_return_abs']:.1%}")
            with col2:
                st.metric("Sharpe Ratio", f"{stats['sharpe_ratio']:.2f}")
            with col3:
                st.metric("Max Drawdown", f"{stats['max_drawdown']:.1%}")
            with col4:
                st.metric("Win Rate", f"{stats['win_rate']:.1%}")
            
            # Performance chart
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=('Price & Signals', 'Portfolio Value'),
                row_heights=[0.7, 0.3]
            )
            
            # Price and signals
            fig.add_trace(
                go.Scatter(x=df.index, y=df['Close'], name='Price', line=dict(color='black')),
                row=1, col=1
            )
            
            # Add buy/sell signals if available
            if 'position' in results.columns:
                buy_signals = results[results['position'].diff() == 1]
                sell_signals = results[results['position'].diff() == -1]
                
                if not buy_signals.empty:
                    fig.add_trace(
                        go.Scatter(x=buy_signals.index, y=buy_signals['Close'], 
                                 mode='markers', name='Buy', 
                                 marker=dict(symbol='triangle-up', color='green', size=10)),
                        row=1, col=1
                    )
                
                if not sell_signals.empty:
                    fig.add_trace(
                        go.Scatter(x=sell_signals.index, y=sell_signals['Close'], 
                                 mode='markers', name='Sell',
                                 marker=dict(symbol='triangle-down', color='red', size=10)),
                        row=1, col=1
                    )
            
            # Portfolio value
            fig.add_trace(
                go.Scatter(x=results.index, y=results['portfolio_value'], 
                         name='Portfolio', line=dict(color='blue')),
                row=2, col=1
            )
            
            fig.update_layout(height=600, title=f"{symbol} Backtest Results")
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed stats table
            with st.expander("📈 Detailed Statistics"):
                stats_df = pd.DataFrame(list(stats.items()), columns=['Metric', 'Value'])
                st.dataframe(stats_df, hide_index=True)
            
            # Store results in session state for potential use in other tabs
            st.session_state.last_backtest = {
                'results': results,
                'stats': stats,
                'symbol': symbol,
                'strategy_type': strategy_type
            }
                
        except Exception as e:
            st.error(f"Backtest failed: {str(e)}")
            st.exception(e)
