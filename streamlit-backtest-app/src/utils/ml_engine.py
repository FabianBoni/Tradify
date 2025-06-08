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
import os
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

def optimize_xgboost_hyperparameters(training_data, target_type, horizon, param_grid, cv_folds=5, method='grid_search'):
    """
    Optimize XGBoost hyperparameters using various methods
    """
    try:
        from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
        from sklearn.metrics import make_scorer, r2_score, accuracy_score
        
        # Prepare dataset
        X, y, features = prepare_ml_dataset(training_data, target_type, horizon)
        
        # Create base model
        if target_type == 'price_direction':
            from xgboost import XGBClassifier
            base_model = XGBClassifier(random_state=42)
            scoring = make_scorer(accuracy_score)
        else:
            from xgboost import XGBRegressor
            base_model = XGBRegressor(random_state=42)
            scoring = make_scorer(r2_score)
        
        # Perform hyperparameter search
        if method == 'grid_search':
            search = GridSearchCV(base_model, param_grid, cv=cv_folds, scoring=scoring, n_jobs=-1)
        else:  # random_search
            search = RandomizedSearchCV(base_model, param_grid, cv=cv_folds, scoring=scoring, n_jobs=-1, n_iter=20)
        
        search.fit(X, y)
        
        return search.best_params_, {'best_score': search.best_score_}
        
    except Exception as e:
        print(f"Hyperparameter optimization failed: {e}")
        return {}, {'best_score': 0}

def optimize_feature_selection(training_data, target_type, horizon, selection_method, n_features=15):
    """
    Optimize feature selection using various methods
    """
    try:
        from sklearn.feature_selection import RFE, SelectKBest, f_regression, f_classif
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        
        # Prepare enhanced dataset
        enhanced_data = create_comprehensive_features(training_data)
        X, y, features = prepare_ml_dataset(enhanced_data, target_type, horizon)
        
        # Select method
        if selection_method == "Recursive Feature Elimination":
            if target_type == 'price_direction':
                estimator = RandomForestClassifier(n_estimators=50, random_state=42)
            else:
                estimator = RandomForestRegressor(n_estimators=50, random_state=42)
            
            selector = RFE(estimator, n_features_to_select=n_features)
            X_selected = selector.fit_transform(X, y)
            selected_features = [features[i] for i in range(len(features)) if selector.support_[i]]
            
        elif selection_method == "Feature Importance":
            if target_type == 'price_direction':
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                scoring_func = f_classif
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                scoring_func = f_regression
            
            selector = SelectKBest(score_func=scoring_func, k=n_features)
            X_selected = selector.fit_transform(X, y)
            selected_features = [features[i] for i in range(len(features)) if selector.get_support()[i]]
        
        else:
            # Default to top features
            selected_features = features[:n_features]
            X_selected = X[:, :n_features]
        
        # Train model with selected features
        model_info = train_xgboost_model(X_selected, y, selected_features, 
                                       'classification' if target_type == 'price_direction' else 'regression')
        
        return selected_features, {'score': model_info['metrics'].get('test_r2', model_info['metrics'].get('test_accuracy', 0))}
        
    except Exception as e:
        print(f"Feature selection failed: {e}")
        return [], {'score': 0}

def compare_models(training_data, target_type, horizon, models_to_compare):
    """
    Compare different model types
    """
    try:
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score, accuracy_score
        
        # Prepare dataset
        X, y, features = prepare_ml_dataset(training_data, target_type, horizon)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        results = {}
        
        for model_name in models_to_compare:
            try:
                if model_name == "XGBoost":
                    model_info = train_xgboost_model(X, y, features, 
                                                   'classification' if target_type == 'price_direction' else 'regression')
                    results[model_name] = model_info['metrics']
                
                elif model_name == "Random Forest":
                    if target_type == 'price_direction':
                        model = RandomForestClassifier(n_estimators=100, random_state=42)
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        score = accuracy_score(y_test, y_pred)
                        results[model_name] = {'test_accuracy': score}
                    else:
                        model = RandomForestRegressor(n_estimators=100, random_state=42)
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        score = r2_score(y_test, y_pred)
                        results[model_name] = {'test_r2': score}
                
                # Add more models as needed
                
            except Exception as model_error:
                print(f"Error training {model_name}: {model_error}")
                results[model_name] = {'test_r2': 0, 'test_accuracy': 0}
        
        return results
        
    except Exception as e:
        print(f"Model comparison failed: {e}")
        return {}

def optimize_hyperparameters(X, y, features, model_type='regression', n_trials=50):
    """
    Optimize hyperparameters using simple grid search
    """
    try:
        from sklearn.model_selection import cross_val_score
        import xgboost as xgb
        
        print(f"🔧 Optimizing hyperparameters with {n_trials} trials...")
        
        # Define parameter grid
        param_grid = [
            {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 'subsample': 0.8},
            {'n_estimators': 200, 'max_depth': 4, 'learning_rate': 0.05, 'subsample': 0.9},
            {'n_estimators': 150, 'max_depth': 8, 'learning_rate': 0.08, 'subsample': 0.85},
            {'n_estimators': 300, 'max_depth': 5, 'learning_rate': 0.03, 'subsample': 0.8},
            {'n_estimators': 250, 'max_depth': 7, 'learning_rate': 0.06, 'subsample': 0.9},
            {'n_estimators': 180, 'max_depth': 3, 'learning_rate': 0.12, 'subsample': 0.85}
        ]
        
        best_score = -np.inf
        best_params = None
        best_model = None
        
        for params in param_grid[:min(len(param_grid), n_trials//10 + 1)]:
            try:
                if model_type == 'regression':
                    model = xgb.XGBRegressor(**params, random_state=42)
                    cv_scores = cross_val_score(model, X, y, cv=3, scoring='r2')
                else:
                    model = xgb.XGBClassifier(**params, random_state=42)
                    cv_scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
                
                mean_score = cv_scores.mean()
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = params
                    best_model = model
                    
            except Exception as e:
                continue
        
        if best_params:
            print(f"✅ Best hyperparameters found: {best_params}")
            print(f"✅ Best CV score: {best_score:.4f}")
            return best_params, best_score, best_model
        else:
            print("⚠️ No valid hyperparameters found, using defaults")
            return None, None, None
            
    except Exception as e:
        print(f"❌ Hyperparameter optimization failed: {e}")
        return None, None, None

def create_enhanced_features(data):
    """
    Create enhanced features with more technical indicators
    """
    try:
        df = data.copy()
        
        # Create basic features first
        df_enhanced = create_comprehensive_features(df)
        
        print("🔬 Adding enhanced technical indicators...")
        
        # Advanced momentum indicators
        df_enhanced['price_momentum_3'] = df['close'].pct_change(periods=3)
        df_enhanced['price_momentum_5'] = df['close'].pct_change(periods=5)
        df_enhanced['price_momentum_10'] = df['close'].pct_change(periods=10)
        
        # Volume analysis
        df_enhanced['volume_sma_10'] = df['volume'].rolling(10).mean()
        df_enhanced['volume_sma_20'] = df['volume'].rolling(20).mean()
        df_enhanced['volume_ratio_10_20'] = df_enhanced['volume_sma_10'] / df_enhanced['volume_sma_20']
        df_enhanced['volume_spike'] = (df['volume'] > df['volume'].rolling(20).mean() * 2).astype(int)
        
        # Price ratios and spreads
        df_enhanced['high_low_ratio'] = df['high'] / df['low']
        df_enhanced['close_open_ratio'] = df['close'] / df['open']
        df_enhanced['hl_spread'] = (df['high'] - df['low']) / df['close']
        df_enhanced['co_spread'] = (df['close'] - df['open']) / df['open']
        
        # Volatility measures
        df_enhanced['volatility_5'] = df['close'].rolling(5).std()
        df_enhanced['volatility_10'] = df['close'].rolling(10).std()
        df_enhanced['volatility_20'] = df['close'].rolling(20).std()
        df_enhanced['volatility_ratio'] = df_enhanced['volatility_5'] / df_enhanced['volatility_20']
        
        # Price position indicators
        df_enhanced['price_position_5'] = (df['close'] - df['close'].rolling(5).min()) / (df['close'].rolling(5).max() - df['close'].rolling(5).min())
        df_enhanced['price_position_20'] = (df['close'] - df['close'].rolling(20).min()) / (df['close'].rolling(20).max() - df['close'].rolling(20).min())
        
        # Clean up
        df_enhanced = df_enhanced.replace([np.inf, -np.inf], np.nan)
        df_enhanced = df_enhanced.dropna()
        
        print(f"✅ Enhanced features created. Total features: {len(df_enhanced.columns)}")
        return df_enhanced
        
    except Exception as e:
        print(f"❌ Enhanced feature creation failed: {e}")
        return create_comprehensive_features(data)  # Fallback to basic features

def compare_models(X, y, features, model_type='regression'):
    """
    Compare different model types and return the best one
    """
    try:
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        from sklearn.linear_model import LinearRegression, LogisticRegression
        from sklearn.model_selection import cross_val_score
        import xgboost as xgb
        
        print("🏆 Comparing different model types...")
        
        models = {}
        scores = {}
        
        # XGBoost (baseline)
        if model_type == 'regression':
            models['XGBoost'] = xgb.XGBRegressor(n_estimators=100, random_state=42)
            models['RandomForest'] = RandomForestRegressor(n_estimators=100, random_state=42)
            models['LinearRegression'] = LinearRegression()
            scoring = 'r2'
        else:
            models['XGBoost'] = xgb.XGBClassifier(n_estimators=100, random_state=42)
            models['RandomForest'] = RandomForestClassifier(n_estimators=100, random_state=42)
            models['LogisticRegression'] = LogisticRegression(random_state=42, max_iter=1000)
            scoring = 'accuracy'
        
        # Test each model
        for name, model in models.items():
            try:
                cv_scores = cross_val_score(model, X, y, cv=3, scoring=scoring)
                scores[name] = cv_scores.mean()
                print(f"✅ {name}: {scores[name]:.4f}")
            except Exception as e:
                print(f"❌ {name} failed: {e}")
                scores[name] = -np.inf
        
        # Find best model
        best_model_name = max(scores, key=scores.get)
        best_score = scores[best_model_name]
        best_model = models[best_model_name]
        
        print(f"🏆 Best model: {best_model_name} with score {best_score:.4f}")
        
        return best_model_name, best_model, best_score
        
    except Exception as e:
        print(f"❌ Model comparison failed: {e}")
        return 'XGBoost', None, 0

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

def save_optimized_model(model_info, symbol, optimization_mode, save_dir="models"):
    """
    Automatically save optimized model with metadata
    """
    try:
        # Ensure models directory exists
        os.makedirs(save_dir, exist_ok=True)
        
        # Generate filename with timestamp and optimization info
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_type = model_info.get('model_type', 'regression')
        performance = model_info['metrics'].get('test_r2', model_info['metrics'].get('test_accuracy', 0))
        
        filename = f"{symbol}_{model_type}_{optimization_mode}_{timestamp}_perf{performance:.3f}.pkl"
        filepath = os.path.join(save_dir, filename)
        
        # Add metadata to model_info
        model_info['save_metadata'] = {
            'symbol': symbol,
            'optimization_mode': optimization_mode,
            'save_timestamp': timestamp,
            'performance_score': performance,
            'filename': filename
        }
        
        # Save the model
        joblib.dump(model_info, filepath)
        
        print(f"✅ Optimized model saved: {filepath}")
        return filepath
        
    except Exception as e:
        print(f"❌ Error saving optimized model: {e}")
        return None

def load_optimized_model(filepath):
    """
    Load optimized model with validation
    """
    try:
        model_info = joblib.load(filepath)
        
        # Validate model structure
        required_keys = ['model', 'features', 'metrics', 'model_type']
        if not all(key in model_info for key in required_keys):
            raise ValueError("Invalid model file structure")
        
        print(f"✅ Optimized model loaded: {filepath}")
        return model_info
        
    except Exception as e:
        print(f"❌ Error loading optimized model: {e}")
        return None

def list_saved_models(save_dir="models"):
    """
    List all saved optimized models with metadata
    """
    try:
        if not os.path.exists(save_dir):
            return []
        
        models = []
        for filename in os.listdir(save_dir):
            if filename.endswith('.pkl'):
                filepath = os.path.join(save_dir, filename)
                try:
                    # Try to load metadata without loading full model
                    model_info = joblib.load(filepath)
                    if 'save_metadata' in model_info:
                        metadata = model_info['save_metadata'].copy()
                        metadata['filepath'] = filepath
                        metadata['file_size'] = os.path.getsize(filepath) / 1024  # KB
                        models.append(metadata)
                except:
                    # Skip corrupted files
                    continue
        
        return sorted(models, key=lambda x: x['save_timestamp'], reverse=True)
        
    except Exception as e:
        print(f"❌ Error listing models: {e}")
        return []
