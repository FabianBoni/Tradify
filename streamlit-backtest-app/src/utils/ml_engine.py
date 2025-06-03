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
        
        # Ensure consistent column naming for ML processing
        column_mapping = {
            'Open': 'open',
            'High': 'high', 
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }
        
        # Rename columns if they exist in uppercase
        for old_col, new_col in column_mapping.items():
            if old_col in data.columns:
                data = data.rename(columns={old_col: new_col})
        
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
    Create comprehensive features from OHLCV data
    """
    try:
        df = data.copy()
        
        # Standardize column names to lowercase for consistency
        column_mapping = {
            'Open': 'open',
            'High': 'high', 
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }
        
        # Rename columns if they exist in uppercase
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df = df.rename(columns={old_col: new_col})
        
        # Ensure we have the required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        print(f"📊 Creating features from {len(df)} data points")
        print(f"📅 Date range: {df.index.min()} to {df.index.max()}")
        
        # Basic price features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Price ratios and spreads
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        df['hl_spread'] = (df['high'] - df['low']) / df['close']
        df['co_spread'] = (df['close'] - df['open']) / df['close']
        
        # Volume features
        df['volume_ma_5'] = df['volume'].rolling(5).mean()
        df['volume_ma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma_20']
        df['price_volume'] = df['close'] * df['volume']
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            df[f'sma_{window}'] = df['close'].rolling(window).mean()
            df[f'price_sma_{window}_ratio'] = df['close'] / df[f'sma_{window}']
        
        # Exponential moving averages
        for span in [12, 26]:
            df[f'ema_{span}'] = df['close'].ewm(span=span).mean()
            df[f'price_ema_{span}_ratio'] = df['close'] / df[f'ema_{span}']
        
        # Volatility features
        df['volatility_5'] = df['returns'].rolling(5).std()
        df['volatility_20'] = df['returns'].rolling(20).std()
        df['volatility_ratio'] = df['volatility_5'] / df['volatility_20']
        
        # Momentum indicators
        df['roc_5'] = df['close'].pct_change(5)
        df['roc_10'] = df['close'].pct_change(10)
        df['momentum_5'] = df['close'] / df['close'].shift(5)
        df['momentum_10'] = df['close'] / df['close'].shift(10)
        
        # Lag features
        for lag in [1, 2, 3, 5]:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
        
        # Statistical features
        df['close_rank_10'] = df['close'].rolling(10).rank(pct=True)
        df['close_rank_20'] = df['close'].rolling(20).rank(pct=True)
        df['returns_skew_10'] = df['returns'].rolling(10).skew()
        df['returns_kurt_10'] = df['returns'].rolling(10).kurt()
        
        # Remove infinite and NaN values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Get the number of features before cleaning
        features_before = len(df.columns)
        
        # Remove columns with too many NaN values (>50%)
        nan_threshold = len(df) * 0.5
        df = df.dropna(thresh=nan_threshold, axis=1)
        
        features_after = len(df.columns)
        print(f"🔧 Feature engineering completed: {features_before} → {features_after} features")
        
        return df
        
    except Exception as e:
        print(f"Error in feature engineering: {e}")
        raise e

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

def prepare_ml_dataset(data, target_type="price_diff_pct", horizon=1):
    """
    Prepare dataset for machine learning with proper target variable creation
    
    Parameters:
    -----------
    data : pandas.DataFrame
        OHLCV price data with datetime index
    target_type : str
        Type of target variable to predict:
        - "price_diff_pct": Percentage price change
        - "price_diff": Absolute price change  
        - "price_direction": Binary direction (1=up, 0=down)
    horizon : int
        Prediction horizon - how many periods ahead to predict
        Examples:
        - horizon=1: Predict next period (1 hour ahead if using hourly data)
        - horizon=4: Predict 4 periods ahead (4 hours ahead if using hourly data)
        - horizon=24: Predict 24 periods ahead (1 day ahead if using hourly data)
    
    Returns:
    --------
    X : numpy.ndarray
        Feature matrix
    y : numpy.ndarray  
        Target variable (what we're trying to predict)
    features : list
        Names of all features
        
    Example:
    --------
    If you have hourly Bitcoin data and set horizon=4:
    - At 10:00 AM, model uses current features to predict Bitcoin price at 2:00 PM
    - At 11:00 AM, model uses current features to predict Bitcoin price at 3:00 PM
    - This gives the trading strategy a 4-hour "look ahead" for decision making
    """
    try:
        # Create comprehensive features from the raw OHLCV data
        df_features = create_comprehensive_features(data)
        
        # Ensure we have 'close' column for target calculation
        if 'close' not in df_features.columns:
            raise ValueError("Missing 'close' column for target calculation")
        
        # Calculate the target variable based on horizon
        if target_type == "price_diff_pct":
            # Percentage price change over the horizon
            # Example: If horizon=4, this calculates (price_t+4 - price_t) / price_t * 100
            df_features['target'] = (
                df_features['close'].shift(-horizon) - df_features['close']
            ) / df_features['close'] * 100
            
        elif target_type == "price_diff":
            # Absolute price change over the horizon
            # Example: If horizon=4, this calculates price_t+4 - price_t
            df_features['target'] = (
                df_features['close'].shift(-horizon) - df_features['close']
            )
            
        elif target_type == "price_direction":
            # Binary direction: 1 if price goes up, 0 if down
            # Example: If horizon=4, this checks if price_t+4 > price_t
            df_features['target'] = (
                df_features['close'].shift(-horizon) > df_features['close']
            ).astype(int)
        
        # Remove rows where we can't calculate the target (last 'horizon' rows)
        # This is because we need future prices to create the target
        df_features = df_features.dropna()
        
        # Separate features and target
        feature_columns = [col for col in df_features.columns if col != 'target']
        X = df_features[feature_columns].values
        y = df_features['target'].values
        
        print(f"📊 Dataset prepared:")
        print(f"   Target type: {target_type}")
        print(f"   Prediction horizon: {horizon} periods")
        print(f"   Features: {len(feature_columns)}")
        print(f"   Samples: {len(X)}")
        print(f"   Target range: {y.min():.4f} to {y.max():.4f}")
        
        return X, y, feature_columns
        
    except Exception as e:
        print(f"Error preparing ML dataset: {e}")
        raise e

def train_xgboost_model(X, y, features, model_type='regression', custom_params=None):
    """
    Train XGBoost model with proper array handling and updated API
    """
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, accuracy_score, mean_squared_error, mean_absolute_error
    
    try:
        print(f"Training XGBoost {model_type} model with {len(features)} features...")
        print(f"Dataset size: {len(X)} samples")
        
        # Ensure we have numpy arrays
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Set default parameters with early stopping in model initialization
        if model_type == 'regression':
            default_params = {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'n_jobs': -1,
                'early_stopping_rounds': 20,  # Move to model init
                'eval_metric': 'rmse'
            }
            model = xgb.XGBRegressor(**default_params)
        else:  # classification
            default_params = {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'n_jobs': -1,
                'early_stopping_rounds': 20,  # Move to model init
                'eval_metric': 'logloss'
            }
            model = xgb.XGBClassifier(**default_params)
        
        # Override with custom parameters if provided
        if custom_params:
            # Remove early_stopping_rounds from custom_params if it exists
            # and set it separately since it needs to be in model init
            early_stopping = custom_params.pop('early_stopping_rounds', 20)
            model.set_params(**custom_params)
            model.set_params(early_stopping_rounds=early_stopping)
        
        # Train model with eval_set for early stopping
        model.fit(
            X_train, y_train, 
            eval_set=[(X_test, y_test)], 
            verbose=False
        )
        
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {}
        
        if model_type == 'regression':
            metrics['train_r2'] = r2_score(y_train, y_train_pred)
            metrics['test_r2'] = r2_score(y_test, y_test_pred)
            metrics['train_rmse'] = np.sqrt(mean_squared_error(y_train, y_train_pred))
            metrics['test_rmse'] = np.sqrt(mean_squared_error(y_test, y_test_pred))
            metrics['test_mae'] = mean_absolute_error(y_test, y_test_pred)
            metrics['overfitting_score'] = metrics['train_r2'] - metrics['test_r2']
        else:  # classification
            metrics['train_accuracy'] = accuracy_score(y_train, y_train_pred)
            metrics['test_accuracy'] = accuracy_score(y_test, y_test_pred)
            metrics['overfitting_score'] = metrics['train_accuracy'] - metrics['test_accuracy']
        
        # Get feature importance
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            importance_data = list(zip(features, model.feature_importances_))
            importance_data.sort(key=lambda x: x[1], reverse=True)
            feature_importance = pd.DataFrame(importance_data, columns=['feature', 'importance'])
        
        # Create model info dictionary
        model_info = {
            'model': model,
            'model_type': model_type,
            'features': features,
            'metrics': metrics,
            'feature_importance': feature_importance,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred': y_test_pred
        }
        
        print(f"✅ Model training completed!")
        print(f"   Features: {len(features)}")
        if model_type == 'regression':
            print(f"   Test R²: {metrics['test_r2']:.4f}")
            print(f"   Test RMSE: {metrics['test_rmse']:.4f}")
        else:
            print(f"   Test Accuracy: {metrics['test_accuracy']:.4f}")
        
        return model_info
        
    except Exception as e:
        print(f"❌ XGBoost training failed: {e}")
        raise e

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

def generate_ml_signals(data, model_info, confidence_threshold=0.6):
    """
    Generate ML trading signals with proper data handling
    """
    try:
        # Ensure we have a DataFrame
        if isinstance(data, np.ndarray):
            raise ValueError("Data must be a pandas DataFrame for signal generation")
        
        model = model_info['model']
        features = model_info['features']
        model_type = model_info['model_type']
        
        # Prepare features for prediction
        available_features = [f for f in features if f in data.columns]
        missing_features = [f for f in features if f not in data.columns]
        
        if missing_features:
            print(f"⚠️ Missing features: {missing_features[:5]}...")  # Show first 5
        
        if len(available_features) < len(features) * 0.8:  # Need at least 80% of features
            raise ValueError(f"Too many missing features: {len(missing_features)}/{len(features)}")
        
        # Use available features and fill missing ones with 0
        X_signals = np.zeros((len(data), len(features)))
        
        for i, feature in enumerate(features):
            if feature in data.columns:
                X_signals[:, i] = data[feature].fillna(0).values
            else:
                X_signals[:, i] = 0  # Fill missing features with 0
        
        # Generate predictions
        if model_type == 'regression':
            predictions = model.predict(X_signals)
            
            # Convert to signals based on prediction magnitude
            signals = np.where(predictions > np.percentile(predictions, 60), 1, 0)
            signals = np.where(predictions < np.percentile(predictions, 40), -1, signals)
            
            # Use absolute prediction as confidence
            confidence = np.abs(predictions) / (np.abs(predictions).max() + 1e-8)
            
        else:  # classification
            predictions = model.predict(X_signals)
            pred_proba = model.predict_proba(X_signals)
            
            # Direct signals from classification
            signals = np.where(predictions == 1, 1, -1)
            
            # Use prediction probability as confidence
            confidence = np.max(pred_proba, axis=1)
        
        # Create signals DataFrame
        signals_df = pd.DataFrame({
            'signal': signals,
            'confidence': confidence,
            'prediction': predictions
        }, index=data.index)
        
        print(f"✅ Generated {len(signals_df)} ML signals")
        print(f"   Signal distribution: {pd.Series(signals).value_counts().to_dict()}")
        print(f"   Average confidence: {confidence.mean():.3f}")
        
        return signals_df
        
    except Exception as e:
        print(f"❌ Signal generation failed: {e}")
        raise e

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

def optimize_xgboost_hyperparameters(data, target_type, horizon, param_grid, cv_folds=5, method='grid_search'):
    """
    Optimize XGBoost hyperparameters using various methods with updated API
    """
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    from sklearn.metrics import r2_score, mean_squared_error
    import xgboost as xgb
    
    # Ensure we have a DataFrame
    if isinstance(data, np.ndarray):
        raise ValueError("Data must be a pandas DataFrame for optimization")
    
    # Prepare data
    X, y, features = prepare_ml_dataset(data, target_type, horizon)
    
    # Split for validation
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Base XGBoost model with early stopping in init
    base_model = xgb.XGBRegressor(
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=10,
        eval_metric='rmse'
    )
    
    # Choose optimization method
    if method == 'grid_search':
        search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv_folds,
            scoring='r2',
            n_jobs=-1,
            verbose=1
        )
    elif method == 'random_search':
        search = RandomizedSearchCV(
            base_model,
            param_grid,
            n_iter=20,
            cv=cv_folds,
            scoring='r2',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
    else:  # bayesian_optimization
        # For Bayesian optimization, we'd need optuna or similar
        # Simplified version using random search for now
        search = RandomizedSearchCV(
            base_model,
            param_grid,
            n_iter=30,
            cv=cv_folds,
            scoring='r2',
            n_jobs=-1,
            random_state=42
        )
    
    # Fit the search - use fit_params for eval_set
    search.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Return results
    return search.best_params_, {
        'best_score': search.best_score_,
        'cv_results': search.cv_results_
    }

def optimize_feature_selection(data, target_type, horizon, method='rfe', n_features=15):
    """
    Optimize feature selection using various methods
    """
    from sklearn.feature_selection import RFE, SelectKBest, f_regression
    from sklearn.ensemble import RandomForestRegressor
    import xgboost as xgb
    
    # Ensure we have a DataFrame
    if isinstance(data, np.ndarray):
        raise ValueError("Data must be a pandas DataFrame for feature selection")
    
    # Prepare data
    X, y, features = prepare_ml_dataset(data, target_type, horizon)
    
    if method.lower() == 'recursive_feature_elimination' or method.lower() == 'rfe':
        # Use Random Forest for feature selection
        estimator = RandomForestRegressor(n_estimators=50, random_state=42)
        selector = RFE(estimator, n_features_to_select=n_features)
        X_selected = selector.fit_transform(X, y)
        selected_features = [features[i] for i in range(len(features)) if selector.support_[i]]
        
    elif method.lower() == 'feature_importance':
        # Use XGBoost feature importance
        model = xgb.XGBRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Get feature importance
        importance_scores = model.feature_importances_
        feature_importance = list(zip(features, importance_scores))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        # Select top features
        selected_features = [feat for feat, score in feature_importance[:n_features]]
        X_selected = X[:, [features.index(feat) for feat in selected_features]]
        
    elif method.lower() == 'correlation_analysis':
        # Remove highly correlated features
        corr_matrix = pd.DataFrame(X, columns=features).corr().abs()
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features with correlation > threshold
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.8)]
        selected_features = [feat for feat in features if feat not in to_drop]
        X_selected = pd.DataFrame(X, columns=features)[selected_features].values
        
    else:  # SelectKBest
        selector = SelectKBest(score_func=f_regression, k=n_features)
        X_selected = selector.fit_transform(X, y)
        selected_features = [features[i] for i in range(len(features)) if selector.get_support()[i]]
    
    # Evaluate selected features
    from sklearn.model_selection import cross_val_score
    model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    scores = cross_val_score(model, X_selected, y, cv=5, scoring='r2')
    
    return selected_features, {
        'score': scores.mean(),
        'score_std': scores.std(),
        'n_features': len(selected_features)
    }

def compare_models(data, target_type, horizon, model_names):
    """
    Compare performance of different model types
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.model_selection import cross_val_score
    import xgboost as xgb
    
    # Ensure we have a DataFrame
    if isinstance(data, np.ndarray):
        raise ValueError("Data must be a pandas DataFrame for model comparison")
    
    # Prepare data
    X, y, features = prepare_ml_dataset(data, target_type, horizon)
    
    # Define models
    models = {}
    
    if 'XGBoost' in model_names:
        models['XGBoost'] = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
    
    if 'Random Forest' in model_names:
        models['Random Forest'] = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
    
    if 'Neural Network' in model_names:
        models['Neural Network'] = MLPRegressor(
            hidden_layer_sizes=(100, 50),
            max_iter=500,
            random_state=42,
            early_stopping=True
        )
    
    # Compare models
    results = {}
    for name, model in models.items():
        try:
            # Cross-validation scores
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
            
            # Fit model for additional metrics
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
            
            results[name] = {
                'cv_r2_mean': cv_scores.mean(),
                'cv_r2_std': cv_scores.std(),
                'test_r2': r2_score(y_test, y_pred),
                'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'test_mae': mean_absolute_error(y_test, y_pred)
            }
            
        except Exception as e:
            results[name] = {'error': str(e)}
    
    return results

def advanced_feature_engineering(data, add_technical_indicators=True, add_time_features=True, add_regime_features=True):
    """
    Apply advanced feature engineering techniques
    """
    # Ensure we have a DataFrame
    if isinstance(data, np.ndarray):
        raise ValueError("Data must be a pandas DataFrame for advanced feature engineering")
    
    enhanced_data = data.copy()
    
    if add_technical_indicators:
        # RSI
        def calculate_rsi(prices, window=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        enhanced_data['rsi'] = calculate_rsi(enhanced_data['close'])
        
        # MACD
        ema_12 = enhanced_data['close'].ewm(span=12).mean()
        ema_26 = enhanced_data['close'].ewm(span=26).mean()
        enhanced_data['macd'] = ema_12 - ema_26
        enhanced_data['macd_signal'] = enhanced_data['macd'].ewm(span=9).mean()
        
        # Bollinger Bands
        rolling_mean = enhanced_data['close'].rolling(window=20).mean()
        rolling_std = enhanced_data['close'].rolling(window=20).std()
        enhanced_data['bb_upper'] = rolling_mean + (rolling_std * 2)
        enhanced_data['bb_lower'] = rolling_mean - (rolling_std * 2)
        enhanced_data['bb_width'] = enhanced_data['bb_upper'] - enhanced_data['bb_lower']
    
    if add_time_features:
        # Time-based features
        enhanced_data['hour'] = enhanced_data.index.hour
        enhanced_data['day_of_week'] = enhanced_data.index.dayofweek
        enhanced_data['month'] = enhanced_data.index.month
        enhanced_data['quarter'] = enhanced_data.index.quarter
    
    if add_regime_features:
        # Volatility regime (high/low volatility periods)
        enhanced_data['volatility'] = enhanced_data['close'].rolling(window=20).std()
        vol_median = enhanced_data['volatility'].median()
        enhanced_data['high_vol_regime'] = (enhanced_data['volatility'] > vol_median).astype(int)
        
        # Trend regime (bull/bear market)
        enhanced_data['trend_20'] = enhanced_data['close'].rolling(window=20).mean()
        enhanced_data['trend_50'] = enhanced_data['close'].rolling(window=50).mean()
        enhanced_data['bull_regime'] = (enhanced_data['trend_20'] > enhanced_data['trend_50']).astype(int)
    
    # Remove any NaN values created by rolling calculations
    enhanced_data = enhanced_data.dropna()
    
    return enhanced_data

def run_training_pipeline(symbol, start_date, end_date, interval='1d', target_type='price_diff_pct', 
                         horizon=1, model_type='regression', save_model_path=None, api_key=None):
    """
    Complete training pipeline with proper data handling
    """
    try:
        # Download data with enhanced data fetcher
        print(f"📊 Starting training pipeline for {symbol}")
        
        # Use the enhanced data fetcher
        training_data = download_market_data(symbol, start_date, end_date, interval, api_key)
        
        if training_data.empty:
            raise ValueError(f"No training data available for {symbol}")
        
        print(f"✅ Downloaded {len(training_data)} data points")
        
        # Prepare ML dataset
        print("🔄 Preparing ML dataset...")
        X, y, features = prepare_ml_dataset(training_data, target_type, horizon)
        
        # Train XGBoost model
        print("🤖 Training XGBoost model...")
        model_info = train_xgboost_model(X, y, features, model_type)
        
        # Save model if requested
        if save_model_path:
            import joblib
            import os
            os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
            joblib.dump(model_info, save_model_path)
            print(f"💾 Model saved to {save_model_path}")
        
        return model_info, training_data
        
    except Exception as e:
        print(f"❌ Training pipeline failed: {e}")
        raise e
