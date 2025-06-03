import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import ta
import xgboost as xgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

def download_market_data(symbol, start_date, end_date, interval='1d', api_key=None):
    """
    Download market data with enhanced error handling and fallback options
    """
    try:
        # Import the new data fetcher
        from .data_fetcher import DataFetcher
        
        print(f"📊 Downloading {symbol} data from {start_date} to {end_date}")
        
        # Use the enhanced data fetcher with pagination
        fetcher = DataFetcher(cryptocompare_api_key=api_key)
        data = fetcher.fetch_historical_data(symbol, start_date, end_date, interval)
        
        if data.empty:
            raise ValueError(f"No data available for {symbol}")
        
        print(f"✅ Successfully downloaded {len(data)} data points")
        return data
        
    except Exception as e:
        print(f"❌ Error downloading data: {e}")
        raise e

def fetch_cryptocompare_data(symbol, start_date, end_date, interval='1d', api_key=None):
    """
    Enhanced CryptoCompare data fetcher with pagination support
    """
    try:
        from .data_fetcher import fetch_cryptocompare_data as fetch_data
        return fetch_data(symbol, start_date, end_date, interval, api_key)
    except Exception as e:
        print(f"Error fetching CryptoCompare data: {e}")
        # Fallback to yfinance
        import yfinance as yf
        symbol_yf = f"{symbol}-USD" if not symbol.endswith('-USD') else symbol
        data = yf.download(symbol_yf, start=start_date, end=end_date, interval=interval, auto_adjust=False, progress=False)
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]
        
        return data

def fetch_traditional_asset_data(symbol, start_date, end_date, interval='1d'):
    """
    Fallback to Alpha Vantage or other APIs for traditional assets
    """
    try:
        # Try Alpha Vantage for stocks
        import yfinance as yf
        print(f"Falling back to Yahoo Finance for {symbol}...")
        
        data = yf.download(symbol, start=start_date, end=end_date, auto_adjust=False, progress=False)
        
        if data.empty:
            raise ValueError(f"No data found for symbol {symbol}")
        
        # Handle MultiIndex columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]
        
        return data.dropna()
        
    except Exception as e:
        raise Exception(f"Error fetching traditional asset data: {str(e)}")

def fetch_training_data(symbol, start_date, end_date, interval='1h', save_path=None, api_key=None):
    """
    Fetch and optionally save training data with multiple data sources and fallbacks
    """
    try:
        print(f"Fetching data for {symbol} from {start_date} to {end_date}...")
        
        # Determine if it's crypto or traditional asset
        crypto_symbols = ['BTC', 'ETH', 'ADA', 'DOT', 'LINK', 'UNI', 'AAVE', 'DOGE', 'XRP', 'LTC']
        is_crypto = any(symbol.upper().startswith(crypto) for crypto in crypto_symbols) or 'USD' in symbol.upper()
        
        data = None
        
        if is_crypto:
            try:
                data = fetch_cryptocompare_data(symbol, start_date, end_date, interval, api_key)
                print(f"✅ CryptoCompare data: {len(data)} points")
            except Exception as cc_error:
                print(f"❌ CryptoCompare failed: {cc_error}")
                
                # Fallback to Yahoo Finance for crypto
                try:
                    print(f"🔄 Trying Yahoo Finance as fallback for {symbol}...")
                    import yfinance as yf
                    
                    # Try different symbol formats for crypto on Yahoo Finance
                    yf_symbols = [f"{symbol}-USD", symbol, f"{symbol}USD"]
                    
                    for yf_symbol in yf_symbols:
                        try:
                            yf_interval = '1h' if interval in ['hour', '1h'] else '1d'
                            data = yf.download(yf_symbol, start=start_date, end=end_date, 
                                             interval=yf_interval, auto_adjust=False, progress=False)
                            
                            if not data.empty:
                                # Handle MultiIndex columns if present
                                if isinstance(data.columns, pd.MultiIndex):
                                    data.columns = [col[0] for col in data.columns]
                                print(f"✅ Yahoo Finance ({yf_symbol}): {len(data)} points")
                                break
                        except:
                            continue
                    
                    if data is None or data.empty:
                        raise Exception("Yahoo Finance fallback also failed")
                        
                except Exception as yf_error:
                    print(f"❌ Yahoo Finance fallback failed: {yf_error}")
                    raise Exception(f"Both CryptoCompare and Yahoo Finance failed for {symbol}")
        else:
            # Traditional assets - use Yahoo Finance directly
            yf_interval = '1h' if interval in ['hour', '1h'] else '1d'
            data = fetch_traditional_asset_data(symbol, start_date, end_date, yf_interval)
        
        if data is None or data.empty:
            raise ValueError(f"No data found for symbol {symbol}")
        
        # Ensure required columns are present
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Clean data
        data = data.dropna()
        
        print(f"Final dataset: {len(data)} data points from {data.index.min()} to {data.index.max()}")
        
        # Save if path provided
        if save_path:
            data.to_csv(save_path)
            print(f"Data saved to {save_path}")
        
        return data
    
    except Exception as e:
        raise Exception(f"Error downloading data: {str(e)}")

def create_comprehensive_features(data):
    """
    Create comprehensive feature engineering pipeline with 14 categories
    """
    df = data.copy()
    
    # Safety function for division
    def safe_divide(a, b, default=0.0):
        return np.where(np.abs(b) < 0.0001, default, a / b)
    
    # 1. Basic Price Action Features
    df['price_change_pct'] = df['Close'].pct_change() * 100
    df['candle_body'] = abs(df['Close'] - df['Open'])
    df['candle_upper_shadow'] = df['High'] - np.maximum(df['Open'], df['Close'])
    df['candle_lower_shadow'] = np.minimum(df['Open'], df['Close']) - df['Low']
    
    # Range-based ratios
    candle_range = df['High'] - df['Low']
    df['body_to_range'] = safe_divide(df['candle_body'], candle_range)
    df['upper_shadow_to_range'] = safe_divide(df['candle_upper_shadow'], candle_range)
    df['lower_shadow_to_range'] = safe_divide(df['candle_lower_shadow'], candle_range)
    
    # Price ratios with clipping
    df['hl_ratio'] = np.clip(safe_divide(df['High'], df['Low'], 1.0), 0.5, 2.0)
    df['oc_ratio'] = np.clip(safe_divide(df['Open'], df['Close'], 1.0), 0.8, 1.2)
    df['vol_price_ratio'] = safe_divide(df['Volume'], df['Close'])
    
    # 2. Moving Averages (SMA & EMA)
    windows = [5, 10, 20, 50, 100, 200]
    for window in windows:
        # SMA
        df[f'sma_{window}'] = df['Close'].rolling(window).mean()
        df[f'dist_sma_{window}'] = df['Close'] - df[f'sma_{window}']
        df[f'dist_sma_{window}_pct'] = np.clip(
            safe_divide(df[f'dist_sma_{window}'], df[f'sma_{window}']) * 100, -50, 50
        )
        
        # EMA
        df[f'ema_{window}'] = df['Close'].ewm(span=window).mean()
        df[f'dist_ema_{window}'] = df['Close'] - df[f'ema_{window}']
        df[f'dist_ema_{window}_pct'] = np.clip(
            safe_divide(df[f'dist_ema_{window}'], df[f'ema_{window}']) * 100, -50, 50
        )
    
    # 3. Momentum Indicators
    rsi_windows = [7, 14, 21, 30]
    for window in rsi_windows:
        df[f'rsi_{window}'] = ta.momentum.RSIIndicator(df['Close'], window).rsi()
    
    # MACD
    macd_indicator = ta.trend.MACD(df['Close'])
    df['macd'] = macd_indicator.macd()
    df['macd_signal'] = macd_indicator.macd_signal()
    df['macd_histogram'] = macd_indicator.macd_diff()
    
    # Stochastic
    stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    
    # Williams %R
    df['williams_r'] = ta.momentum.WilliamsRIndicator(df['High'], df['Low'], df['Close']).williams_r()
    
    # Ultimate Oscillator
    df['ui'] = ta.momentum.UltimateOscillator(df['High'], df['Low'], df['Close']).ultimate_oscillator()
    
    # Rate of Change
    roc_periods = [5, 10, 20]
    for period in roc_periods:
        df[f'roc_{period}'] = ta.momentum.ROCIndicator(df['Close'], period).roc()
    
    # 4. Volatility Indicators
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df['Close'])
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_middle'] = bb.bollinger_mavg()
    df['bb_width'] = df['bb_upper'] - df['bb_lower']
    df['bb_position'] = safe_divide(df['Close'] - df['bb_lower'], df['bb_width'])
    
    # ATR
    df['atr'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
    df['atr_ratio'] = safe_divide(df['atr'], df['Close'])
    
    # Donchian Channels
    dc = ta.volatility.DonchianChannel(df['High'], df['Low'], df['Close'])
    df['dc_upper'] = dc.donchian_channel_hband()
    df['dc_lower'] = dc.donchian_channel_lband()
    df['dc_middle'] = dc.donchian_channel_mband()
    df['dc_wband'] = dc.donchian_channel_wband()
    df['dc_pband'] = dc.donchian_channel_pband()
    
    # Rolling volatility
    vol_windows = [5, 10, 20, 50]
    for window in vol_windows:
        df[f'volatility_{window}d'] = df['Close'].rolling(window).std()
    
    # 5. Volume Indicators
    df['vpt'] = ta.volume.VolumePriceTrendIndicator(df['Close'], df['Volume']).volume_price_trend()
    df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
    
    vol_windows = [5, 10, 20, 50]
    for window in vol_windows:
        df[f'vol_sma_{window}'] = df['Volume'].rolling(window).mean()
        df[f'vol_ratio_{window}'] = safe_divide(df['Volume'], df[f'vol_sma_{window}'])
    
    # 6. Trend Indicators
    # ADX
    adx = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'])
    df['adx'] = adx.adx()
    df['adx_pos'] = adx.adx_pos()
    df['adx_neg'] = adx.adx_neg()
    
    # CCI
    df['cci'] = ta.trend.CCIIndicator(df['High'], df['Low'], df['Close']).cci()
    
    # Parabolic SAR
    psar = ta.trend.PSARIndicator(df['High'], df['Low'], df['Close'])
    df['psar'] = psar.psar()
    df['psar_up'] = psar.psar_up()
    df['psar_down'] = psar.psar_down()
    
    # 7. Lag Features
    lag_periods = [1, 2, 3, 5, 10]
    for lag in lag_periods:
        df[f'close_lag_{lag}'] = df['Close'].shift(lag)
        df[f'vol_lag_{lag}'] = df['Volume'].shift(lag)
        df[f'return_lag_{lag}'] = df['price_change_pct'].shift(lag)
    
    # 8. Time-Based Features
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['day_of_month'] = df.index.day
    df['month'] = df.index.month
    if hasattr(df.index, 'minute'):
        df['minute'] = df.index.minute
    
    # 9. Cross-over Signals
    df['sma_cross_20_50'] = np.where(df['sma_20'] > df['sma_50'], 1, 0)
    df['price_above_sma200'] = np.where(df['Close'] > df['sma_200'], 1, 0)
    df['rsi_oversold'] = np.where(df['rsi_14'] < 30, 1, 0)
    df['rsi_overbought'] = np.where(df['rsi_14'] > 70, 1, 0)
    
    # 10. Support/Resistance Levels
    periods = [10, 20, 50]
    for period in periods:
        df[f'local_min_{period}'] = df['Low'].rolling(period, center=True).min()
        df[f'local_max_{period}'] = df['High'].rolling(period, center=True).max()
    
    # 11. Fibonacci Retracements
    fib_levels = [236, 382, 500, 618]
    period = 50
    rolling_high = df['High'].rolling(period).max()
    rolling_low = df['Low'].rolling(period).min()
    fib_range = rolling_high - rolling_low
    
    for level in fib_levels:
        fib_value = rolling_low + (fib_range * level / 1000)
        df[f'fib_{level}'] = fib_value
        df[f'dist_fib_{level}'] = abs(df['Close'] - fib_value)
    
    # 12. Rolling Statistics
    stat_windows = [10, 20, 50]
    for window in stat_windows:
        df[f'close_mean_{window}'] = df['Close'].rolling(window).mean()
        df[f'close_std_{window}'] = df['Close'].rolling(window).std()
        df[f'close_min_{window}'] = df['Close'].rolling(window).min()
        df[f'close_max_{window}'] = df['Close'].rolling(window).max()
    
    # 13. Create Prediction Labels
    thresholds = {'strict': 1.5, 'medium': 0.8, 'loose': 0.4}
    horizons = [1, 4, 12, 24]
    
    for threshold_name, threshold_val in thresholds.items():
        for horizon in horizons:
            future_return = df['Close'].shift(-horizon).pct_change(horizon) * 100
            
            df[f'label_up_{threshold_name}_{horizon}'] = np.where(future_return > threshold_val, 1, 0)
            df[f'label_down_{threshold_name}_{horizon}'] = np.where(future_return < -threshold_val, 1, 0)
            df[f'label_neutral_{threshold_name}_{horizon}'] = np.where(
                (future_return >= -threshold_val) & (future_return <= threshold_val), 1, 0
            )
    
    # 14. Add next price change for immediate predictions
    df['next_price_change_pct'] = df['Close'].shift(-1).pct_change() * 100
    df['price_change_1step'] = df['Close'].pct_change() * 100
    
    # Clean up infinite and NaN values
    df = df.replace([np.inf, -np.inf], np.nan)
    
    return df

def create_price_difference_target(df, horizon=1):
    """
    Create price difference target for regression - Fixed for data leakage
    """
    # Calculate price difference for next candle(s)
    future_close = df['Close'].shift(-horizon)
    current_close = df['Close']
    
    # Price difference (absolute)
    price_diff = future_close - current_close
    
    # Price difference percentage
    price_diff_pct = ((future_close - current_close) / current_close) * 100
    
    # Price direction (classification target)
    price_direction = np.where(price_diff > 0, 1, np.where(price_diff < 0, -1, 0))
    
    return {
        'price_diff': price_diff,
        'price_diff_pct': price_diff_pct, 
        'price_direction': price_direction
    }

def select_top_features():
    """
    Return features that don't include future data - Fixed for data leakage
    """
    return [
        'price_change_pct', 'atr_ratio', 'volatility_5d', 'volatility_10d', 
        'hl_ratio', 'oc_ratio', 'dc_wband', 'dist_sma_20_pct', 'dist_sma_50_pct', 
        'vpt', 'vol_sma_20', 'roc_5', 'rsi_14', 'macd_histogram',
        'bb_position', 'adx', 'vol_ratio_20', 'stoch_k', 'williams_r',
        'close_lag_1', 'close_lag_2', 'vol_lag_1', 'return_lag_1'
    ]

def prepare_ml_dataset(df, target_type='price_diff_pct', horizon=1):
    """
    Prepare dataset for ML training with strict data leakage prevention
    """
    print(f"Creating features for {len(df)} data points...")
    
    # Create features first
    df_features = create_comprehensive_features(df)
    
    print(f"Features created, dataset now has {len(df_features)} rows")
    
    # Create targets
    targets = create_price_difference_target(df_features, horizon)
    df_features['target'] = targets[target_type]
    
    # Remove future-looking features that could cause data leakage
    forbidden_features = [
        'next_price_change_pct',  # This is literally future data
        'price_change_1step'      # This might be current step
    ]
    
    # Select features
    feature_cols = select_top_features()
    available_features = [col for col in feature_cols if col in df_features.columns and col not in forbidden_features]
    
    print(f"Available features: {len(available_features)}")
    print(f"Features: {available_features}")
    
    if len(available_features) < 5:
        # Fallback to very basic features
        basic_features = ['price_change_pct', 'hl_ratio', 'oc_ratio', 'vol_price_ratio', 'rsi_14']
        available_features = [col for col in basic_features if col in df_features.columns]
        print(f"Using basic features: {available_features}")
    
    # Prepare final dataset and remove rows with future data
    feature_data = df_features[available_features + ['target']].copy()
    
    # Remove the last 'horizon' rows as they don't have valid targets
    feature_data = feature_data.iloc[:-horizon] if horizon > 0 else feature_data
    
    # Remove rows with NaN values
    clean_data = feature_data.dropna()
    
    print(f"Clean dataset size: {len(clean_data)} (removed {len(feature_data) - len(clean_data)} rows with NaN)")
    
    X = clean_data[available_features]
    y = clean_data['target']
    
    # Check for potential data leakage by examining target distribution
    print(f"Target statistics: mean={y.mean():.4f}, std={y.std():.4f}, min={y.min():.4f}, max={y.max():.4f}")
    
    return X, y, available_features

def train_xgboost_model(X, y, features, model_type='regression', test_size=0.2, save_path=None):
    """
    Train XGBoost model with more realistic parameters and validation
    """
    print(f"Training XGBoost {model_type} model with {len(features)} features...")
    print(f"Dataset size: {len(X)} samples")
    
    # Check for potential data leakage indicators
    if model_type == 'regression':
        y_std = y.std()
        if y_std < 0.1:
            print(f"⚠️  WARNING: Very low target variance (std={y_std:.6f}) - possible data leakage!")
    
    # Use TimeSeriesSplit for proper time series validation
    tscv = TimeSeriesSplit(n_splits=3)
    
    # Split data (keeping time order) - this is crucial for time series
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Training set: {len(X_train)}, Test set: {len(X_test)}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Configure XGBoost with more conservative parameters
    if model_type == 'regression':
        model = xgb.XGBRegressor(
            n_estimators=100,        # Reduced from 200
            max_depth=4,             # Reduced from 6
            learning_rate=0.05,      # Reduced from 0.1
            subsample=0.7,           # Reduced from 0.8
            colsample_bytree=0.7,    # Reduced from 0.8
            random_state=42,
            early_stopping_rounds=15,
            eval_metric='rmse',
            reg_alpha=0.1,           # L1 regularization
            reg_lambda=1.0           # L2 regularization
        )
    else:  # classification
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.7,
            colsample_bytree=0.7,
            random_state=42,
            early_stopping_rounds=15,
            eval_metric='logloss',
            reg_alpha=0.1,
            reg_lambda=1.0
        )
    
    # Train model with validation
    print("Training model...")
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_train_scaled, y_train), (X_test_scaled, y_test)],
        verbose=True  # Show training progress
    )
    
    # Make predictions
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    # Calculate metrics
    if model_type == 'regression':
        train_score = r2_score(y_train, y_pred_train)
        test_score = r2_score(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        # Check for overfitting
        r2_diff = train_score - test_score
        if r2_diff > 0.1:
            print(f"⚠️  WARNING: Possible overfitting detected! R² difference: {r2_diff:.4f}")
        
        # Check for unrealistic performance
        if test_score > 0.9:
            print(f"⚠️  WARNING: Unrealistically high R² ({test_score:.4f}) - check for data leakage!")
        
        metrics = {
            'train_r2': train_score,
            'test_r2': test_score,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'overfitting_score': r2_diff
        }
    else:
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)
        
        acc_diff = train_score - test_score
        if acc_diff > 0.1:
            print(f"⚠️  WARNING: Possible overfitting detected! Accuracy difference: {acc_diff:.4f}")
        
        metrics = {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'overfitting_score': acc_diff
        }
    
    # Feature importance analysis
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 5 Most Important Features:")
    print(feature_importance.head())
    
    model_info = {
        'model': model,
        'scaler': scaler,
        'features': features,
        'model_type': model_type,
        'metrics': metrics,
        'feature_importance': feature_importance,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred_test
    }
    
    # Save model if path provided
    if save_path:
        joblib.dump(model_info, save_path)
        print(f"Model saved to {save_path}")
    
    print("Training completed!")
    print(f"Final metrics: {metrics}")
    
    return model_info

def load_trained_model(model_path):
    """
    Load a pre-trained model
    """
    try:
        model_info = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
        return model_info
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")

def train_ml_model(df, target_type='price_diff_pct', horizon=1, model_type='regression', test_size=0.2):
    """
    Complete ML training pipeline
    """
    # Prepare dataset
    X, y, features = prepare_ml_dataset(df, target_type, horizon)
    
    if len(X) < 100:
        raise ValueError("Not enough clean data for training")
    
    # Train model
    model_info = train_xgboost_model(X, y, features, model_type, test_size)
    
    return model_info

def generate_ml_signals(df, model_info, confidence_threshold=0.6):
    """
    Generate trading signals using the trained XGBoost model
    """
    model = model_info['model']
    scaler = model_info['scaler']
    features = model_info['features']
    model_type = model_info['model_type']
    
    # Prepare features
    feature_data = df[features].copy()
    
    # Handle missing values
    feature_data = feature_data.fillna(method='ffill').fillna(0)
    
    # Scale features
    try:
        X_scaled = scaler.transform(feature_data)
        
        # Generate predictions
        predictions = model.predict(X_scaled)
        
        # Create signals DataFrame
        signals = pd.DataFrame(index=df.index)
        signals['prediction'] = predictions
        
        if model_type == 'regression':
            # For regression, convert price difference to signals
            # Positive prediction = buy, negative = sell
            signals['confidence'] = np.abs(predictions) / (np.abs(predictions).max() + 0.001)
            signals['signal'] = np.where(
                (predictions > 0.5) & (signals['confidence'] > confidence_threshold), 1,
                np.where((predictions < -0.5) & (signals['confidence'] > confidence_threshold), -1, 0)
            )
        else:
            # For classification, use probabilities
            probabilities = model.predict_proba(X_scaled)
            signals['confidence'] = np.max(probabilities, axis=1)
            signals['signal'] = np.where(
                (predictions == 1) & (signals['confidence'] > confidence_threshold), 1,
                np.where((predictions == -1) & (signals['confidence'] > confidence_threshold), -1, 0)
            )
        
        return signals
    
    except Exception as e:
        print(f"Error generating signals: {e}")
        # Fallback to simple signals if ML fails
        signals = pd.DataFrame(index=df.index)
        signals['prediction'] = 0
        signals['confidence'] = 0.5
        signals['signal'] = 0
        return signals

def run_training_pipeline(symbol, start_date, end_date, interval='1h', 
                         target_type='price_diff_pct', horizon=1, 
                         model_type='regression', save_model_path=None, save_data_path=None, api_key=None):
    """
    Complete training pipeline from data fetching to model training
    """
    print("=== ML Training Pipeline ===")
    
    # Step 1: Fetch data
    data = fetch_training_data(symbol, start_date, end_date, interval, save_data_path, api_key)
    
    # Step 2: Prepare dataset
    print("Preparing ML dataset...")
    X, y, features = prepare_ml_dataset(data, target_type, horizon)
    
    # Step 3: Train model
    model_info = train_xgboost_model(X, y, features, model_type, save_path=save_model_path)
    
    print("=== Training Pipeline Complete ===")
    
    return model_info, data
