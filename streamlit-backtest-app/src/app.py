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
tab1, tab2, tab3, tab4 = st.tabs(["📈 Backtest", "🤖 ML Training", "🔧 Model Optimization", "📊 Model Analysis"])

with tab1:
    st.title("📈 Trading Strategy Backtester")
    st.markdown("Test your trading strategies with historical data")

    # Sidebar for parameters
    with st.sidebar:
        st.header("Configuration")
        
        # API Configuration
        with st.expander("🔧 API Settings"):
            api_key = st.text_input("CryptoCompare API Key (Optional)", type="password", 
                                   help="Get free API key from cryptocompare.com")
            use_crypto_data = st.checkbox("Force CryptoCompare API", 
                                        help="Use CryptoCompare even for traditional assets")
        
        # Stock selection
        symbol = st.text_input("Asset Symbol", value="BTC", help="Enter crypto (BTC, ETH) or stock ticker (AAPL, GOOGL)")
        
        # Data interval
        data_interval = st.selectbox("Data Interval", ["1h", "1d"], help="1h for crypto, 1d for stocks")
        
        # Backtesting date range
        st.subheader("📅 Backtesting Period")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Backtest Start", value=datetime.now() - timedelta(days=90), key="backtest_start")
        with col2:
            end_date = st.date_input("Backtest End", value=datetime.now(), key="backtest_end")
        
        # Strategy selection
        st.subheader("Strategy Selection")
        strategy_options = ["Simple MA Cross", "Machine Learning", "ML with Pre-trained Model"]
        strategy_type = st.selectbox("Choose Strategy", strategy_options)
        
        # Strategy parameters
        if strategy_type == "Simple MA Cross":
            st.subheader("Strategy Parameters")
            short_ma = st.slider("Short MA Period", min_value=5, max_value=50, value=10)
            long_ma = st.slider("Long MA Period", min_value=20, max_value=200, value=20)
            
        elif strategy_type == "Machine Learning":
            st.subheader("🎯 Training Data Period")
            
            # Training date range with validation
            train_col1, train_col2 = st.columns(2)
            with train_col1:
                train_start = st.date_input(
                    "Training Start", 
                    value=datetime.now() - timedelta(days=730), 
                    key="ml_train_start",
                    help="Must be before backtest start date"
                )
            with train_col2:
                train_end = st.date_input(
                    "Training End", 
                    value=start_date - timedelta(days=1), 
                    key="ml_train_end",
                    help="Must be before backtest start date"
                )
            
            # Validation checks
            date_validation_errors = []
            
            if train_start >= start_date:
                date_validation_errors.append("Training start must be before backtest start")
            
            if train_end >= start_date:
                date_validation_errors.append("Training end must be before backtest start")
            
            if train_start >= train_end:
                date_validation_errors.append("Training start must be before training end")
            
            if (train_end - train_start).days < 30:
                date_validation_errors.append("Training period should be at least 30 days")
            
            if date_validation_errors:
                for error in date_validation_errors:
                    st.error(f"❌ {error}")
            else:
                st.success("✅ Training dates are valid")
                
                # Show training period info
                training_days = (train_end - train_start).days
                st.info(f"📊 Training period: {training_days} days")
            
            st.subheader("ML Strategy Parameters")
            confidence_threshold = st.slider("Confidence Threshold", min_value=0.5, max_value=0.9, value=0.6, step=0.05)
            
            # ML model parameters
            st.subheader("Model Parameters")
            target_type = st.selectbox("Target Type", ["price_diff_pct", "price_direction"], key="backtest_target")
            horizon = st.slider("Prediction Horizon", min_value=1, max_value=12, value=1, key="backtest_horizon")
            model_type = st.selectbox("Model Type", ["regression", "classification"], key="backtest_model_type")
            
        else:  # ML with Pre-trained Model
            st.subheader("Pre-trained Model")
            model_file = st.file_uploader("Upload Model File", type=['pkl', 'joblib'])
            confidence_threshold = st.slider("Confidence Threshold", min_value=0.5, max_value=0.9, value=0.6, step=0.05)
            
            # Option to use session state model
            if 'trained_model' in st.session_state:
                use_session_model = st.checkbox("Use Last Trained Model", 
                                               help="Use the model from the ML Training tab")
            else:
                use_session_model = False
        
        # Backtest parameters
        st.subheader("Backtest Parameters")
        initial_cash = st.number_input("Initial Cash ($)", min_value=1000, value=10000, step=1000)
        commission = st.slider("Commission (%)", min_value=0.0, max_value=1.0, value=0.2, step=0.1) / 100
        
        # Disable run button if there are validation errors for ML strategy
        can_run_backtest = True
        if strategy_type == "Machine Learning" and date_validation_errors:
            can_run_backtest = False
            
        run_backtest_btn = st.button("Run Backtest", type="primary", disabled=not can_run_backtest)

    # Main content area
    if run_backtest_btn:
        try:
            # Convert date objects to datetime objects for proper handling
            start_date_dt = datetime.combine(start_date, datetime.min.time()) if isinstance(start_date, date) else start_date
            end_date_dt = datetime.combine(end_date, datetime.min.time()) if isinstance(end_date, date) else end_date
            
            # Download data using enhanced data fetcher
            with st.spinner(f"Downloading {symbol} data..."):
                fetcher = DataFetcher(cryptocompare_api_key=api_key)
                
                if strategy_type in ["Machine Learning", "ML with Pre-trained Model"]:
                    if strategy_type == "Machine Learning":
                        # Convert training dates to datetime objects
                        train_start_dt = datetime.combine(train_start, datetime.min.time()) if isinstance(train_start, date) else train_start
                        train_end_dt = datetime.combine(train_end, datetime.min.time()) if isinstance(train_end, date) else train_end
                        
                        # Download training data plus backtest data with enhanced fetcher
                        data_start = min(train_start_dt, start_date_dt - timedelta(days=7))
                        data = fetcher.fetch_historical_data(symbol, data_start, end_date_dt, data_interval)
                        
                        # Convert dates to pandas timestamps for proper comparison
                        train_start_ts = pd.Timestamp(train_start_dt)
                        train_end_ts = pd.Timestamp(train_end_dt)
                        backtest_start_ts = pd.Timestamp(start_date_dt)
                        backtest_end_ts = pd.Timestamp(end_date_dt)
                        
                        # Validate we have enough data
                        training_data = data[(data.index >= train_start_ts) & (data.index <= train_end_ts)]
                        backtest_data_check = data[(data.index >= backtest_start_ts) & (data.index <= backtest_end_ts)]
                        
                        if len(training_data) < 100:
                            st.error(f"❌ Insufficient training data: {len(training_data)} points. Need at least 100.")
                            
                            # Show data availability info
                            if not data.empty:
                                available_start = data.index.min().date()
                                available_end = data.index.max().date()
                                
                                st.warning(f"""
                                **📊 Data Availability:**
                                - Available: {available_start} to {available_end} ({len(data)} points)
                                - Requested training: {train_start} to {train_end}
                                - Requested backtest: {start_date} to {end_date}
                                
                                **💡 Solutions:**
                                1. **Get API Key**: Free CryptoCompare API key provides more historical data
                                2. **Adjust dates**: Use available data range shown above
                                3. **Try stocks**: Traditional assets (AAPL, GOOGL) have longer history
                                4. **Use daily data**: Switch to '1d' interval for maximum history
                                """)
                            else:
                                st.error("❌ No data available for this symbol/period combination")
                            
                            st.stop()
                        
                        if len(backtest_data_check) < 10:
                            st.error(f"❌ Insufficient backtest data: {len(backtest_data_check)} points. Need at least 10.")
                            st.stop()
                        
                        st.success(f"✅ Enhanced data fetch: {len(training_data)} training points, {len(backtest_data_check)} backtest points")
                        
                    else:
                        # Just download backtest period data for pre-trained model
                        data = fetcher.fetch_historical_data(symbol, start_date_dt, end_date_dt, data_interval)
                else:
                    # Traditional strategy - use enhanced data fetcher
                    data = fetcher.fetch_historical_data(symbol, start_date_dt, end_date_dt, data_interval)
            
            if data.empty:
                st.error(f"No data found for symbol {symbol}")
                st.stop()
            
            # Prepare data for backtesting
            data = data.dropna()
            
            # Strategy execution
            if strategy_type == "Simple MA Cross":
                # Create custom strategy class with user parameters
                class CustomSmaCross(SmaCross):
                    n1 = short_ma
                    n2 = long_ma
                
                # Run backtest
                with st.spinner("Running backtest..."):
                    backtest_start_ts = pd.Timestamp(start_date_dt)
                    backtest_end_ts = pd.Timestamp(end_date_dt)
                    backtest_data = data[(data.index >= backtest_start_ts) & (data.index <= backtest_end_ts)]
                    stats, bt = run_backtest(backtest_data, CustomSmaCross, initial_cash, commission)
            
            elif strategy_type == "Machine Learning":
                # Split data into training and backtesting periods
                train_start_ts = pd.Timestamp(train_start_dt)
                train_end_ts = pd.Timestamp(train_end_dt)
                backtest_start_ts = pd.Timestamp(start_date_dt)
                backtest_end_ts = pd.Timestamp(end_date_dt)
                
                training_data = data[(data.index >= train_start_ts) & (data.index <= train_end_ts)]
                backtest_data = data[(data.index >= backtest_start_ts) & (data.index <= backtest_end_ts)]
                
                # Train model on training data
                with st.spinner("Training XGBoost model on historical data..."):
                    st.info(f"🔄 Training on {len(training_data)} data points from {train_start} to {train_end}")
                    X, y, features = prepare_ml_dataset(training_data, target_type=target_type, horizon=horizon)
                    model_info = train_xgboost_model(X, y, features, model_type=model_type)
                
                # Run ML backtest on separate period
                with st.spinner("Running ML backtest on out-of-sample data..."):
                    st.info(f"🎯 Backtesting on {len(backtest_data)} data points from {start_date} to {end_date}")
                    stats, bt = run_backtest(
                        backtest_data, 
                        MLStrategy, 
                        initial_cash, 
                        commission,
                        model_info=model_info,
                        confidence_threshold=confidence_threshold
                    )
                
                # Display ML model performance
                st.subheader("🤖 ML Model Performance")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    if 'train_r2' in model_info['metrics']:
                        st.metric("Training R²", f"{model_info['metrics']['train_r2']:.3f}")
                    else:
                        st.metric("Training Acc", f"{model_info['metrics'].get('train_accuracy', 0):.3f}")
                with col2:
                    if 'test_r2' in model_info['metrics']:
                        st.metric("Test R²", f"{model_info['metrics']['test_r2']:.3f}")
                    else:
                        st.metric("Test Acc", f"{model_info['metrics'].get('test_accuracy', 0):.3f}")
                with col3:
                    st.metric("Features", len(model_info['features']))
                with col4:
                    if 'overfitting_score' in model_info['metrics']:
                        st.metric("Overfitting", f"{model_info['metrics']['overfitting_score']:.3f}")
                
                # Show data split information
                with st.expander("📊 Data Split Information"):
                    split_col1, split_col2 = st.columns(2)
                    with split_col1:
                        st.metric("Training Period", f"{train_start} to {train_end}")
                        st.metric("Training Data Points", len(training_data))
                    with split_col2:
                        st.metric("Backtest Period", f"{start_date} to {end_date}")
                        st.metric("Backtest Data Points", len(backtest_data))
            
            else:  # ML with Pre-trained Model
                model_info = None
                
                if use_session_model and 'trained_model' in st.session_state:
                    # Use model from session state
                    model_info = st.session_state.trained_model
                    st.success("✅ Using model from training session")
                    
                elif model_file is not None:
                    # Load uploaded model
                    with st.spinner("Loading pre-trained model..."):
                        import tempfile
                        import os
                        
                        # Save uploaded file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
                            tmp_file.write(model_file.getvalue())
                            tmp_path = tmp_file.name
                        
                        try:
                            model_info = load_trained_model(tmp_path)
                            os.unlink(tmp_path)  # Clean up temp file
                            st.success("✅ Pre-trained model loaded successfully")
                        except Exception as e:
                            os.unlink(tmp_path)
                            raise e
                
                if model_info is None:
                    st.error("❌ No model available. Please upload a model file or train a model first.")
                    st.stop()
                
                # Run ML backtest with pre-trained model
                with st.spinner("Running ML backtest with pre-trained model..."):
                    backtest_start_ts = pd.Timestamp(start_date_dt)
                    backtest_end_ts = pd.Timestamp(end_date_dt)
                    backtest_data = data[(data.index >= backtest_start_ts) & (data.index <= backtest_end_ts)]
                    stats, bt = run_backtest(
                        backtest_data, 
                        MLStrategy, 
                        initial_cash, 
                        commission,
                        model_info=model_info,
                        confidence_threshold=confidence_threshold
                    )
                
                # Display model info
                st.subheader("🔍 Model Information")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.info(f"**Model Type:** {model_info['model_type']}")
                with col2:
                    st.info(f"**Features:** {len(model_info['features'])}")
                with col3:
                    if 'test_r2' in model_info['metrics']:
                        st.info(f"**Test R²:** {model_info['metrics']['test_r2']:.3f}")
                    elif 'test_accuracy' in model_info['metrics']:
                        st.info(f"**Test Acc:** {model_info['metrics']['test_accuracy']:.3f}")
        
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
                
                if strategy_type == "Simple MA Cross":
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
            
            # Enhanced error handling with pagination info
            if "No data found" in str(e) or "Error fetching" in str(e):
                st.info("""
                **💡 Enhanced Data Fetching Tips:**
                - **Crypto symbols**: BTC, ETH, XRP, DOGE (now with full historical coverage via pagination)
                - **Stock symbols**: AAPL, GOOGL, TSLA (extensive Yahoo Finance history)
                - **API Key**: Get free CryptoCompare key for best historical coverage
                - **Intervals**: '1d' for maximum history, '1h' for recent detailed data
                
                **New Features:**
                - Automatic pagination retrieves complete historical datasets
                - Intelligent fallback between CryptoCompare and Yahoo Finance
                - Enhanced coverage analysis and validation
                """)

    else:
        # Default view when no backtest has been run
        st.info("👈 Configure your strategy parameters in the sidebar and click 'Run Backtest' to start.")
        
        # Show sample data
        st.subheader("📖 How to use this app:")
        st.markdown("""
        **Available Strategies:**
        
        1. **Simple MA Cross**: Traditional moving average crossover
           - Buy when short MA crosses above long MA
           - Sell when long MA crosses above short MA
        
        2. **Machine Learning**: Train a new XGBoost model
           - Set training period (must be before backtest period)
           - Uses comprehensive feature engineering
           - Predicts price movements or direction
           - Trains on historical data, tests on future data
        
        3. **ML with Pre-trained Model**: Use existing models
           - Upload your own trained model files
           - Use models from the ML Training tab
           - Perfect for testing different model configurations
        
        **Important for ML Strategies:**
        - Training data must be from BEFORE the backtesting period
        - This prevents data leakage and ensures realistic results
        - Minimum 30 days training period recommended
        
        **Getting Started:**
        1. Select an asset symbol (crypto or stock)
        2. Set your backtesting date range
        3. For ML: Set training dates (before backtest dates)
        4. Choose your strategy type and configure parameters
        5. Run backtest and analyze results
        """)

with tab2:
    st.title("🤖 ML Model Training")
    st.markdown("Train XGBoost models for price prediction")
    
    # API Configuration for training
    with st.expander("🔧 Data Source Settings"):
        train_api_key = st.text_input("CryptoCompare API Key", type="password", key="train_api")
        st.info("💡 Get a free API key from cryptocompare.com for better rate limits and more historical data")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Training Configuration")
        
        # Training parameters
        train_symbol = st.text_input("Training Symbol", value="BTC", key="train_symbol", 
                                   help="Crypto: BTC, ETH, DOGE | Stocks: AAPL, GOOGL")
        
        # Training date range
        train_col1, train_col2 = st.columns(2)
        with train_col1:
            train_start = st.date_input("Training Start", value=datetime.now() - timedelta(days=730), key="train_start")
        with train_col2:
            train_end = st.date_input("Training End", value=datetime.now(), key="train_end")
        
        # Model parameters
        st.subheader("Model Parameters")
        target_type = st.selectbox("Target Type", ["price_diff_pct", "price_diff", "price_direction"])
        horizon = st.slider("Prediction Horizon", min_value=1, max_value=24, value=1)
        model_type = st.selectbox("Model Type", ["regression", "classification"])
        interval = st.selectbox("Data Interval", ["1h", "1d", "4h"])
        
        # Training controls
        test_size = st.slider("Test Size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
        
        train_model_btn = st.button("🚀 Train Model", type="primary")
        
    with col2:
        st.subheader("Model Management")
        
        # Save/Load options
        save_model = st.checkbox("Save Trained Model")
        if save_model:
            model_name = st.text_input("Model Name", value=f"{train_symbol}_{target_type}_{horizon}h")
        
        # Quick training presets
        st.subheader("Quick Presets")
        if st.button("📈 Price Direction (1h)"):
            st.session_state.preset_config = {
                'target_type': 'price_direction', 
                'horizon': 1, 
                'model_type': 'classification',
                'interval': '1h'
            }
        
        if st.button("💰 Price Change % (4h)"):
            st.session_state.preset_config = {
                'target_type': 'price_diff_pct', 
                'horizon': 4, 
                'model_type': 'regression',
                'interval': '4h'
            }

    # Training execution
    if train_model_btn:
        try:
            # Convert date objects to datetime objects
            train_start_dt = datetime.combine(train_start, datetime.min.time()) if isinstance(train_start, date) else train_start
            train_end_dt = datetime.combine(train_end, datetime.min.time()) if isinstance(train_end, date) else train_end
            
            with st.spinner("🔄 Running complete training pipeline..."):
                # Determine save path
                save_path = None
                if save_model:
                    save_path = f"models/{model_name}.pkl"
                    
                # Run training pipeline with API key
                model_info, training_data = run_training_pipeline(
                    symbol=train_symbol,
                    start_date=train_start_dt,
                    end_date=train_end_dt,
                    interval=interval,
                    target_type=target_type,
                    horizon=horizon,
                    model_type=model_type,
                    save_model_path=save_path,
                    api_key=train_api_key
                )
                
                # Store in session state
                st.session_state.trained_model = model_info
                st.session_state.training_data = training_data
                
                st.success("✅ Model training completed!")
                
                # Display results
                st.subheader("📊 Training Results")
                
                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                with metrics_col1:
                    if model_type == 'regression':
                        st.metric("Training R²", f"{model_info['metrics']['train_r2']:.4f}")
                        st.metric("Test R²", f"{model_info['metrics']['test_r2']:.4f}")
                    else:
                        st.metric("Training Accuracy", f"{model_info['metrics']['train_accuracy']:.4f}")
                        st.metric("Test Accuracy", f"{model_info['metrics']['test_accuracy']:.4f}")
                
                with metrics_col2:
                    if model_type == 'regression':
                        st.metric("Training RMSE", f"{model_info['metrics']['train_rmse']:.4f}")
                        st.metric("Test RMSE", f"{model_info['metrics']['test_rmse']:.4f}")
                        st.metric("Test MAE", f"{model_info['metrics']['test_mae']:.4f}")
                
                with metrics_col3:
                    st.metric("Features Used", len(model_info['features']))
                    st.metric("Training Samples", len(training_data))
                
                # Feature importance
                if hasattr(model_info, 'feature_importance'):
                    st.subheader("🎯 Feature Importance")
                    top_features = model_info['feature_importance'].head(10)
                    
                    import plotly.express as px
                    fig = px.bar(top_features, x='importance', y='feature', orientation='h')
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Training failed: {str(e)}")
            
            # Show helpful error messages
            if "API" in str(e) or "rate limit" in str(e).lower():
                st.warning("🔑 Consider getting a CryptoCompare API key for better access to historical data")
            elif "symbol" in str(e).lower():
                st.info("💡 Try different symbols: BTC, ETH for crypto or AAPL, GOOGL for stocks")

with tab3:
    st.title("🔧 Model Optimization")
    st.markdown("Optimize model performance through hyperparameter tuning and feature engineering")
    
    # Check if we have training data or model to optimize
    if 'trained_model' not in st.session_state and 'training_data' not in st.session_state:
        st.info("🎯 **No model available for optimization.**")
        st.markdown("""
        **To get started:**
        1. Go to the **ML Training** tab
        2. Train a baseline model first
        3. Return here to optimize it
        
        **What this tab offers:**
        - 🎛️ **Hyperparameter Tuning**: Optimize XGBoost parameters
        - 🔍 **Feature Selection**: Find the best feature combinations
        - 📊 **Model Comparison**: Compare different configurations
        - 🚀 **Advanced Models**: Try Random Forest, Neural Networks
        """)
    else:
        # Optimization controls
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🎯 Optimization Strategy")
            
            optimization_type = st.selectbox(
                "Choose Optimization Method",
                ["Hyperparameter Tuning", "Feature Selection", "Model Comparison", "Advanced Engineering"]
            )
            
            if optimization_type == "Hyperparameter Tuning":
                st.markdown("**Optimize XGBoost hyperparameters using grid search or random search**")
                
                tuning_method = st.selectbox("Tuning Method", ["Grid Search", "Random Search", "Bayesian Optimization"])
                
                # XGBoost parameters to tune
                st.subheader("Parameters to Optimize")
                
                tune_n_estimators = st.checkbox("n_estimators", value=True)
                if tune_n_estimators:
                    n_est_range = st.slider("n_estimators range", 50, 1000, (100, 500), step=50)
                
                tune_max_depth = st.checkbox("max_depth", value=True)
                if tune_max_depth:
                    depth_range = st.slider("max_depth range", 3, 15, (4, 8))
                
                tune_learning_rate = st.checkbox("learning_rate", value=True)
                if tune_learning_rate:
                    lr_range = st.slider("learning_rate range", 0.01, 0.3, (0.05, 0.2), step=0.01)
                
                tune_subsample = st.checkbox("subsample", value=False)
                if tune_subsample:
                    subsample_range = st.slider("subsample range", 0.6, 1.0, (0.8, 1.0), step=0.1)
                
                # Advanced parameters
                with st.expander("🔧 Advanced Parameters"):
                    tune_colsample = st.checkbox("colsample_bytree")
                    if tune_colsample:
                        colsample_range = st.slider("colsample_bytree range", 0.6, 1.0, (0.8, 1.0), step=0.1)
                    
                    tune_gamma = st.checkbox("gamma (regularization)")
                    if tune_gamma:
                        gamma_range = st.slider("gamma range", 0.0, 5.0, (0.0, 2.0), step=0.5)
                
                cv_folds = st.slider("Cross-validation folds", 3, 10, 5)
                n_trials = st.slider("Number of trials", 10, 100, 20) if tuning_method != "Grid Search" else None
                
            elif optimization_type == "Feature Selection":
                st.markdown("**Find the optimal feature combination**")
                
                selection_method = st.selectbox(
                    "Selection Method", 
                    ["Recursive Feature Elimination", "Feature Importance", "Correlation Analysis", "Sequential Selection"]
                )
                
                if selection_method == "Recursive Feature Elimination":
                    n_features_to_select = st.slider("Target number of features", 5, 50, 15)
                elif selection_method == "Feature Importance":
                    importance_threshold = st.slider("Importance threshold", 0.01, 0.1, 0.02, step=0.01)
                elif selection_method == "Correlation Analysis":
                    correlation_threshold = st.slider("Correlation threshold", 0.7, 0.95, 0.8, step=0.05)
                
                # Feature engineering options
                st.subheader("Feature Engineering")
                add_polynomial = st.checkbox("Add polynomial features (degree 2)")
                add_interactions = st.checkbox("Add feature interactions")
                add_rolling_features = st.checkbox("Add more rolling window features")
                
                if add_rolling_features:
                    rolling_windows = st.multiselect(
                        "Rolling windows", 
                        [5, 10, 20, 50, 100, 200],
                        default=[5, 10, 20]
                    )
                
            elif optimization_type == "Model Comparison":
                st.markdown("**Compare different model types**")
                
                models_to_compare = st.multiselect(
                    "Models to compare",
                    ["XGBoost", "Random Forest", "LightGBM", "CatBoost", "Neural Network"],
                    default=["XGBoost", "Random Forest"]
                )
                
                # Ensemble options
                st.subheader("Ensemble Methods")
                create_ensemble = st.checkbox("Create ensemble model")
                if create_ensemble:
                    ensemble_method = st.selectbox("Ensemble method", ["Voting", "Stacking", "Blending"])
                
            else:  # Advanced Engineering
                st.markdown("**Advanced feature engineering and model architectures**")
                
                # Technical indicators
                st.subheader("Technical Indicators")
                add_rsi = st.checkbox("RSI (Relative Strength Index)", value=True)
                add_macd = st.checkbox("MACD", value=True)
                add_bollinger = st.checkbox("Bollinger Bands", value=True)
                add_stochastic = st.checkbox("Stochastic Oscillator")
                add_williams = st.checkbox("Williams %R")
                
                # Time-based features
                st.subheader("Time Features")
                add_time_features = st.checkbox("Hour/Day/Week features", value=True)
                add_seasonal = st.checkbox("Seasonal decomposition")
                add_fourier = st.checkbox("Fourier transform features")
                
                # Market regime features
                st.subheader("Market Regime")
                add_volatility_regime = st.checkbox("Volatility regime detection")
                add_trend_regime = st.checkbox("Trend regime detection")
        
        with col2:
            st.subheader("📊 Current Model Performance")
            
            if 'trained_model' in st.session_state:
                model_info = st.session_state.trained_model
                
                # Current metrics
                metrics_col1, metrics_col2 = st.columns(2)
                with metrics_col1:
                    if 'test_r2' in model_info['metrics']:
                        st.metric("Current Test R²", f"{model_info['metrics']['test_r2']:.4f}")
                        st.metric("Current Train R²", f"{model_info['metrics']['train_r2']:.4f}")
                    else:
                        st.metric("Current Test Acc", f"{model_info['metrics'].get('test_accuracy', 0):.4f}")
                        st.metric("Current Train Acc", f"{model_info['metrics'].get('train_accuracy', 0):.4f}")
                
                with metrics_col2:
                    if 'test_rmse' in model_info['metrics']:
                        st.metric("Current RMSE", f"{model_info['metrics']['test_rmse']:.4f}")
                        st.metric("Current MAE", f"{model_info['metrics'].get('test_mae', 0):.4f}")
                    
                    st.metric("Features Used", len(model_info['features']))
                
                # Performance analysis
                current_r2 = model_info['metrics'].get('test_r2', 0)
                overfitting = model_info['metrics'].get('train_r2', 0) - current_r2
                
                if current_r2 < 0.1:
                    st.warning("⚠️ **Low R² Score**: Model has weak predictive power")
                    st.info("""
                    **Recommendations:**
                    1. Try hyperparameter tuning
                    2. Add more technical indicators
                    3. Consider ensemble methods
                    4. Check for data leakage
                    """)
                
                if overfitting > 0.05:
                    st.warning("⚠️ **Overfitting Detected**: Large gap between train/test performance")
                    st.info("""
                    **Solutions:**
                    1. Increase regularization
                    2. Reduce model complexity
                    3. Add more training data
                    4. Use cross-validation
                    """)
            
            # Optimization targets
            st.subheader("🎯 Optimization Targets")
            
            target_r2 = st.slider("Target R² Score", 0.1, 0.8, 0.3, step=0.05)
            max_overfitting = st.slider("Max Overfitting Gap", 0.01, 0.1, 0.03, step=0.01)
            max_training_time = st.slider("Max Training Time (minutes)", 1, 60, 10)
        
        # Run optimization button
        st.markdown("---")
        run_optimization = st.button("🚀 Run Optimization", type="primary", use_container_width=True)
        
        if run_optimization:
            with st.spinner("🔄 Running optimization..."):
                try:
                    # Get base data
                    if 'training_data' in st.session_state:
                        training_data = st.session_state.training_data
                    else:
                        st.error("No training data available")
                        st.stop()
                    
                    # Get base model parameters
                    base_model = st.session_state.trained_model
                    target_type = "price_diff_pct"  # Get from model info if available
                    horizon = 1
                    model_type = base_model.get('model_type', 'regression')
                    
                    optimization_results = []
                    
                    if optimization_type == "Hyperparameter Tuning":
                        st.info(f"🎛️ Running {tuning_method} hyperparameter optimization...")
                        
                        # Import optimization utilities
                        from utils.ml_engine import optimize_xgboost_hyperparameters
                        
                        # Prepare parameter grid
                        param_grid = {}
                        if tune_n_estimators:
                            param_grid['n_estimators'] = list(range(n_est_range[0], n_est_range[1]+1, 50))
                        if tune_max_depth:
                            param_grid['max_depth'] = list(range(depth_range[0], depth_range[1]+1))
                        if tune_learning_rate:
                            param_grid['learning_rate'] = [round(x, 2) for x in np.arange(lr_range[0], lr_range[1]+0.01, 0.02)]
                        
                        # Run optimization
                        best_params, cv_results = optimize_xgboost_hyperparameters(
                            training_data, 
                            target_type, 
                            horizon, 
                            param_grid, 
                            cv_folds,
                            tuning_method.lower().replace(' ', '_')
                        )
                        
                        st.success("✅ Hyperparameter optimization completed!")
                        
                        # Display results
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("🏆 Best Parameters")
                            for param, value in best_params.items():
                                st.metric(param, value)
                        
                        with col2:
                            st.subheader("📈 Performance Improvement")
                            improvement = cv_results['best_score'] - current_r2
                            st.metric("R² Improvement", f"+{improvement:.4f}")
                            st.metric("Best CV Score", f"{cv_results['best_score']:.4f}")
                        
                        # Store optimized model
                        if st.button("💾 Use Optimized Parameters"):
                            # Retrain with best parameters
                            from utils.ml_engine import prepare_ml_dataset, train_xgboost_model
                            X, y, features = prepare_ml_dataset(training_data, target_type, horizon)
                            
                            optimized_model = train_xgboost_model(
                                X, y, features, model_type, 
                                custom_params=best_params
                            )
                            
                            st.session_state.trained_model = optimized_model
                            st.success("✅ Model updated with optimized parameters!")
                            st.rerun()
                    
                    elif optimization_type == "Feature Selection":
                        st.info(f"🔍 Running {selection_method}...")
                        
                        from utils.ml_engine import optimize_feature_selection, create_comprehensive_features
                        
                        # Enhanced feature engineering
                        enhanced_data = create_comprehensive_features(training_data)
                        
                        if add_polynomial:
                            st.info("Adding polynomial features...")
                            # Add polynomial features for top features
                            numeric_cols = enhanced_data.select_dtypes(include=[np.number]).columns[:10]  # Top 10 features
                            for col in numeric_cols:
                                enhanced_data[f"{col}_squared"] = enhanced_data[col] ** 2
                        
                        if add_interactions:
                            st.info("Adding interaction features...")
                            # Add key interactions
                            enhanced_data['volume_price_interaction'] = enhanced_data['volume'] * enhanced_data['close']
                            enhanced_data['high_low_spread'] = enhanced_data['high'] - enhanced_data['low']
                        
                        # Run feature selection
                        selected_features, selection_results = optimize_feature_selection(
                            enhanced_data, target_type, horizon, selection_method, n_features_to_select
                        )
                        
                        st.success("✅ Feature selection completed!")
                        
                        # Display results
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("📋 Selected Features")
                            st.write(f"Reduced from {len(enhanced_data.columns)} to {len(selected_features)} features")
                            st.dataframe(pd.DataFrame(selected_features, columns=['Feature']))
                        
                        with col2:
                            st.subheader("📈 Performance")
                            st.metric("New R² Score", f"{selection_results['score']:.4f}")
                            improvement = selection_results['score'] - current_r2
                            st.metric("Improvement", f"+{improvement:.4f}")
                    
                    elif optimization_type == "Model Comparison":
                        st.info("🔄 Training and comparing multiple models...")
                        
                        from utils.ml_engine import compare_models
                        
                        comparison_results = compare_models(
                            training_data, 
                            target_type, 
                            horizon, 
                            models_to_compare
                        )
                        
                        st.success("✅ Model comparison completed!")
                        
                        # Display comparison
                        st.subheader("🏆 Model Performance Comparison")
                        comparison_df = pd.DataFrame(comparison_results).T
                        comparison_df = comparison_df.sort_values('test_r2', ascending=False)
                        
                        st.dataframe(comparison_df, use_container_width=True)
                        
                        # Best model selection
                        best_model_name = comparison_df.index[0]
                        st.success(f"🥇 Best performing model: **{best_model_name}**")
                        
                        if st.button("🔄 Switch to Best Model"):
                            # Implementation would depend on having saved models
                            st.info("Feature coming soon: Auto-switch to best performing model")
                    
                    else:  # Advanced Engineering
                        st.info("🔬 Applying advanced feature engineering...")
                        
                        from utils.ml_engine import advanced_feature_engineering
                        
                        # Apply advanced features
                        advanced_data = advanced_feature_engineering(
                            training_data,
                            add_technical_indicators=True,
                            add_time_features=add_time_features,
                            add_regime_features=add_volatility_regime
                        )
                        
                        # Train with advanced features
                        from utils.ml_engine import prepare_ml_dataset, train_xgboost_model
                        X, y, features = prepare_ml_dataset(advanced_data, target_type, horizon)
                        
                        advanced_model = train_xgboost_model(X, y, features, model_type)
                        
                        st.success("✅ Advanced feature engineering completed!")
                        
                        # Compare performance
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Original R²", f"{current_r2:.4f}")
                            st.metric("Original Features", len(base_model['features']))
                        
                        with col2:
                            new_r2 = advanced_model['metrics']['test_r2']
                            st.metric("Advanced R²", f"{new_r2:.4f}")
                            st.metric("Advanced Features", len(advanced_model['features']))
                        
                        improvement = new_r2 - current_r2
                        if improvement > 0:
                            st.success(f"🎉 Improvement: +{improvement:.4f} R²")
                            if st.button("💾 Use Advanced Model"):
                                st.session_state.trained_model = advanced_model
                                st.success("✅ Model updated!")
                                st.rerun()
                        else:
                            st.warning("⚠️ No significant improvement with advanced features")
                
                except Exception as e:
                    st.error(f"Optimization failed: {str(e)}")
                    st.info("💡 Try reducing the complexity or check your data quality")

with tab4:  # Update tab4 to be tab4 instead of tab3
    st.title("📊 Model Analysis")
    st.markdown("Analyze and compare trained models")
    
    if 'trained_model' in st.session_state:
        model_info = st.session_state.trained_model
        
        st.subheader("🔍 Model Details")
        
        # Model info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**Model Type:** {model_info['model_type']}")
            st.info(f"**Features:** {len(model_info['features'])}")
        with col2:
            if 'test_r2' in model_info['metrics']:
                st.info(f"**Test R²:** {model_info['metrics']['test_r2']:.4f}")
            if 'test_accuracy' in model_info['metrics']:
                st.info(f"**Test Accuracy:** {model_info['metrics']['test_accuracy']:.4f}")
        with col3:
            st.info(f"**Test Samples:** {len(model_info['X_test'])}")
        
        # Feature list
        with st.expander("📋 Model Features"):
            features_df = pd.DataFrame(model_info['features'], columns=['Feature'])
            st.dataframe(features_df, use_container_width=True)
        
        # Predictions vs Actual
        if 'y_test' in model_info and 'y_pred' in model_info:
            st.subheader("🎯 Predictions vs Actual")
            
            pred_df = pd.DataFrame({
                'Actual': model_info['y_test'],
                'Predicted': model_info['y_pred']
            })
            
            import plotly.express as px
            fig = px.scatter(pred_df, x='Actual', y='Predicted', 
                           title='Predictions vs Actual Values')
            fig.add_trace(go.Scatter(x=[pred_df['Actual'].min(), pred_df['Actual'].max()],
                                   y=[pred_df['Actual'].min(), pred_df['Actual'].max()],
                                   mode='lines', name='Perfect Prediction'))
            st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.info("No trained model available. Train a model in the ML Training tab first.")