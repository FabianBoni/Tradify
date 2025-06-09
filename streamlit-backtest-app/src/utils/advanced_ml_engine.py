"""
Advanced ML Engine für intelligente Modellentwicklung
====================================================

Dieses Modul stellt erweiterte ML-Funktionalitäten zur Verfügung:
- Automatische Hyperparameter-Optimierung
- Intelligente Feature Engineering
- Model Ensemble Methoden
- Performance Monitoring
- Experiment Tracking

Author: Trading Strategy Team
Version: 2.0
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
import json
import joblib
import warnings
warnings.filterwarnings('ignore')
import os

# ML Libraries
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import (
    TimeSeriesSplit, GridSearchCV, RandomizedSearchCV,
    cross_val_score, train_test_split
)
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix
)
from sklearn.feature_selection import (
    SelectKBest, f_classif, f_regression, RFE, SelectFromModel
)

# Optimization Libraries
try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer, Categorical
    BAYESIAN_OPT_AVAILABLE = True
except ImportError:
    BAYESIAN_OPT_AVAILABLE = False

# Technical Analysis
try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    print("Warning: ta library not available. Technical indicators will be limited.")

class AdvancedMLEngine:
    """
    Erweiterte ML-Engine mit automatischer Optimierung und Dokumentation
    """
    
    def __init__(self, experiment_name: str = None, log_level: str = "INFO"):
        """
        Initialisiert die Advanced ML Engine
        
        Args:
            experiment_name: Name des Experiments für Tracking
            log_level: Logging Level (DEBUG, INFO, WARNING, ERROR)
        """
        self.experiment_name = experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.setup_logging(log_level)
        
        # Experiment Tracking
        self.experiments = []
        self.best_model = None
        self.best_score = float('-inf')
        self.feature_importance_history = []
        
        # Model Registry
        self.models = {
            'classification': {
                'xgboost': xgb.XGBClassifier,
                'random_forest': RandomForestClassifier,
                'logistic_regression': LogisticRegression,
                'svm': SVC,
                'neural_network': MLPClassifier
            },
            'regression': {
                'xgboost': xgb.XGBRegressor,
                'random_forest': RandomForestRegressor,
                'linear_regression': LinearRegression,
                'svm': SVR,
                'neural_network': MLPRegressor
            }
        }
        
        # Hyperparameter Search Spaces
        self.param_spaces = self._get_parameter_spaces()
        
        self.logger.info(f"Initialisiert Advanced ML Engine für Experiment: {self.experiment_name}")
    
    def setup_logging(self, level: str):
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(f"AdvancedML_{self.experiment_name}")
    
    def create_comprehensive_features(self, data: pd.DataFrame, 
                                    target_type: str = "classification",
                                    horizon: int = 1) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Erweiterte Feature Engineering Pipeline
        
        Args:
            data: Preisdaten (OHLCV)
            target_type: 'classification' oder 'regression'
            horizon: Vorhersage-Horizont in Zeitperioden
            
        Returns:
            Tuple von (Features DataFrame, Target Series)
        """
        self.logger.info("🔧 Starte erweitertes Feature Engineering...")
        
        df = data.copy()
        
        # 1. Basis Technical Indicators
        df = self._add_technical_indicators(df)
        
        # 2. Advanced Statistical Features
        df = self._add_statistical_features(df)
        
        # 3. Lag Features
        df = self._add_lag_features(df)
        
        # 4. Rolling Window Features
        df = self._add_rolling_features(df)
        
        # 5. Volatility Features
        df = self._add_volatility_features(df)
        
        # 6. Market Microstructure Features
        df = self._add_microstructure_features(df)
        
        # 7. Time-based Features
        df = self._add_time_features(df)
        
        # Create Target Variable
        target = self._create_target_variable(df, target_type, horizon)
        
        # Feature Selection
        feature_cols = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
        features = df[feature_cols].copy()
        
        # Remove rows with NaN values
        valid_idx = ~(features.isna().any(axis=1) | target.isna())
        features = features[valid_idx]
        target = target[valid_idx]
        
        self.logger.info(f"✅ Feature Engineering abgeschlossen: {len(feature_cols)} Features, {len(features)} Samples")
        
        return features, target    
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fügt technische Indikatoren hinzu"""
        
        # Standardize column names to lowercase for consistent access
        df = df.copy()
        df.columns = [col.lower() for col in df.columns]
        
        if not TA_AVAILABLE:
            # Fallback to simple technical indicators without ta library
            self.logger.warning("ta library not available. Using basic technical indicators.")
            
            # Simple Moving Averages
            df['sma_10'] = df['close'].rolling(10).mean()
            df['sma_20'] = df['close'].rolling(20).mean()
            df['sma_50'] = df['close'].rolling(50).mean()
            
            # Simple Exponential Moving Averages
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()
            
            # Simple RSI calculation
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Basic Bollinger Bands
            df['bb_mid'] = df['close'].rolling(20).mean()
            bb_std = df['close'].rolling(20).std()
            df['bb_high'] = df['bb_mid'] + (2 * bb_std)
            df['bb_low'] = df['bb_mid'] - (2 * bb_std)
            df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mid']
            df['bb_position'] = (df['close'] - df['bb_low']) / (df['bb_high'] - df['bb_low'])
            
            return df
        
        # Full technical indicators with ta library
        # Trend Indicators
        df['sma_10'] = ta.trend.sma_indicator(df['close'], window=10)
        df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
        df['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
        df['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)
        
        # MACD
        df['macd'] = ta.trend.macd(df['close'])
        df['macd_signal'] = ta.trend.macd_signal(df['close'])
        df['macd_diff'] = ta.trend.macd_diff(df['close'])
        
        # Bollinger Bands
        df['bb_high'] = ta.volatility.bollinger_hband(df['close'])
        df['bb_low'] = ta.volatility.bollinger_lband(df['close'])
        df['bb_mid'] = ta.volatility.bollinger_mavg(df['close'])
        df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mid']
        df['bb_position'] = (df['close'] - df['bb_low']) / (df['bb_high'] - df['bb_low'])
        
        # Momentum Indicators
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        df['stoch'] = ta.momentum.stoch(df['high'], df['low'], df['close'])
        df['williams_r'] = ta.momentum.williams_r(df['high'], df['low'], df['close'])
        df['cci'] = ta.trend.cci(df['high'], df['low'], df['close'])
        df['roc'] = ta.momentum.roc(df['close'], window=10)
          # Volume Indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()  # Simple manual volume SMA
        df['mfi'] = ta.volume.money_flow_index(df['high'], df['low'], df['close'], df['volume'])
        df['adi'] = ta.volume.acc_dist_index(df['high'], df['low'], df['close'], df['volume'])
        df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
        
        # Volatility Indicators
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
        df['keltner_high'] = ta.volatility.keltner_channel_hband(df['high'], df['low'], df['close'])
        df['keltner_low'] = ta.volatility.keltner_channel_lband(df['high'], df['low'], df['close'])
        
        return df
    
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fügt statistische Features hinzu"""
        
        # Price-based statistics
        df['price_mean_10'] = df['close'].rolling(10).mean()
        df['price_std_10'] = df['close'].rolling(10).std()
        df['price_skew_10'] = df['close'].rolling(10).skew()
        df['price_kurt_10'] = df['close'].rolling(10).kurt()
        
        # Volume-based statistics
        df['volume_mean_10'] = df['volume'].rolling(10).mean()
        df['volume_std_10'] = df['volume'].rolling(10).std()
        df['volume_ratio'] = df['volume'] / df['volume_mean_10']
        
        # Price range statistics
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        df['body_size'] = abs(df['close'] - df['open']) / df['open']
        df['upper_shadow'] = (df['high'] - np.maximum(df['close'], df['open'])) / df['open']
        df['lower_shadow'] = (np.minimum(df['close'], df['open']) - df['low']) / df['open']
        
        return df
    
    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fügt Lag Features hinzu"""
        
        # Price lags
        for lag in [1, 2, 3, 5, 10]:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
            df[f'rsi_lag_{lag}'] = df['rsi'].shift(lag)
        
        # Price changes
        for lag in [1, 2, 3, 5, 10]:
            df[f'price_change_{lag}'] = df['close'].pct_change(lag)
            df[f'volume_change_{lag}'] = df['volume'].pct_change(lag)
        
        return df
    
    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fügt Rolling Window Features hinzu"""
        
        windows = [5, 10, 20, 50]
        
        for window in windows:
            # Price features
            df[f'price_max_{window}'] = df['close'].rolling(window).max()
            df[f'price_min_{window}'] = df['close'].rolling(window).min()
            df[f'price_position_{window}'] = (df['close'] - df[f'price_min_{window}']) / (df[f'price_max_{window}'] - df[f'price_min_{window}'])
            
            # Momentum features
            df[f'momentum_{window}'] = df['close'] / df['close'].shift(window) - 1
            
            # Volume features
            df[f'volume_max_{window}'] = df['volume'].rolling(window).max()
            df[f'volume_min_{window}'] = df['volume'].rolling(window).min()
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fügt Volatilitäts-Features hinzu"""
        
        # Historical volatility
        for window in [10, 20, 50]:
            df[f'volatility_{window}'] = df['close'].pct_change().rolling(window).std() * np.sqrt(252)
            df[f'volatility_ratio_{window}'] = df[f'volatility_{window}'] / df[f'volatility_{window}'].rolling(50).mean()
        
        # Parkinson volatility (uses high/low)
        df['parkinson_vol'] = np.sqrt(252 * np.log(df['high'] / df['low']).rolling(20).var())
        
        # Garman-Klass volatility
        df['gk_vol'] = np.sqrt(252 * (np.log(df['high'] / df['low'])**2 - 
                                     (2*np.log(2)-1) * np.log(df['close'] / df['open'])**2).rolling(20).mean())
        
        return df
    
    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fügt Market Microstructure Features hinzu"""
        
        # Price impact measures
        df['price_impact'] = abs(df['close'].pct_change()) / (df['volume'] / df['volume'].rolling(20).mean())
        
        # Bid-ask spread proxy
        df['spread_proxy'] = (df['high'] - df['low']) / df['close']
        
        # Volume-price trend
        df['vpt'] = (df['volume'] * (df['close'] - df['close'].shift(1)) / df['close'].shift(1)).cumsum()
        
        # Money flow
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['money_flow'] = df['typical_price'] * df['volume']
        df['money_flow_ratio'] = df['money_flow'] / df['money_flow'].rolling(20).mean()
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fügt zeitbasierte Features hinzu"""
        
        # Hour of day (für intraday data)
        if hasattr(df.index, 'hour'):
            df['hour'] = df.index.hour
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Day of week
        if hasattr(df.index, 'dayofweek'):
            df['day_of_week'] = df.index.dayofweek
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        return df
    
    def _create_target_variable(self, df: pd.DataFrame, target_type: str, horizon: int) -> pd.Series:
        """Erstellt die Zielvariable"""
        
        if target_type == "classification":
            # Binary classification: wird der Preis steigen?
            future_return = df['close'].shift(-horizon) / df['close'] - 1
            target = (future_return > 0).astype(int)
            
        elif target_type == "regression":
            # Regression: prozentuale Preisveränderung
            target = df['close'].shift(-horizon) / df['close'] - 1
            
        elif target_type == "price_direction":
            # Price direction: 1 for upward movement, 0 for downward
            future_return = df['close'].shift(-horizon) / df['close'] - 1
            target = (future_return > 0).astype(int)
            
        else:
            raise ValueError(f"Unbekannter target_type: {target_type}")
        
        return target
    
    def run_automated_optimization(self, 
                                 data: pd.DataFrame,
                                 target_type: str = "classification",
                                 horizon: int = 1,
                                 max_trials: int = 100,
                                 cv_folds: int = 5,
                                 test_size: float = 0.2,
                                 optimization_method: str = "bayesian",
                                 models_to_test: List[str] = None) -> Dict[str, Any]:
        """
        Führt automatische Hyperparameter-Optimierung durch
        
        Args:
            data: Eingabedaten
            target_type: 'classification' oder 'regression'
            horizon: Vorhersagehorizont
            max_trials: Maximale Anzahl Optimierungsversuche
            cv_folds: Anzahl Cross-Validation Folds
            test_size: Anteil Test-Daten
            optimization_method: 'grid', 'random', 'bayesian'
            models_to_test: Liste der zu testenden Modelle
            
        Returns:
            Dictionary mit Optimierungsergebnissen
        """
        
        self.logger.info(f"🚀 Starte automatische Optimierung für {target_type}")
        
        # Feature Engineering
        features, target = self.create_comprehensive_features(data, target_type, horizon)
        
        # Train-Test Split (zeitbasiert)
        split_idx = int(len(features) * (1 - test_size))
        X_train, X_test = features.iloc[:split_idx], features.iloc[split_idx:]
        y_train, y_test = target.iloc[:split_idx], target.iloc[split_idx:]
        
        # Feature Scaling
        scaler = RobustScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        # Models to test
        if models_to_test is None:
            models_to_test = ['xgboost', 'random_forest', 'logistic_regression'] if target_type == "classification" else ['xgboost', 'random_forest', 'linear_regression']
        
        results = {
            'experiments': [],
            'best_model': None,
            'best_score': float('-inf'),
            'feature_importance': None,
            'test_scores': {},
            'optimization_history': []
        }
        
        # Cross-validation setup
        cv = TimeSeriesSplit(n_splits=cv_folds)
        
        # Test each model
        for model_name in models_to_test:
            self.logger.info(f"🔧 Optimiere {model_name}...")
            
            model_class = self.models[target_type][model_name]
            param_space = self.param_spaces[target_type][model_name]
            
            # Choose optimization method
            if optimization_method == "bayesian" and BAYESIAN_OPT_AVAILABLE:
                search = BayesSearchCV(
                    model_class(),
                    param_space,
                    n_iter=max_trials // len(models_to_test),
                    cv=cv,
                    scoring='accuracy' if target_type == "classification" else 'neg_mean_squared_error',
                    n_jobs=-1,
                    random_state=42
                )
            elif optimization_method == "random":
                search = RandomizedSearchCV(
                    model_class(),
                    param_space,
                    n_iter=max_trials // len(models_to_test),
                    cv=cv,
                    scoring='accuracy' if target_type == "classification" else 'neg_mean_squared_error',
                    n_jobs=-1,
                    random_state=42
                )
            else:  # grid search
                search = GridSearchCV(
                    model_class(),
                    param_space,
                    cv=cv,
                    scoring='accuracy' if target_type == "classification" else 'neg_mean_squared_error',
                    n_jobs=-1
                )
            
            # Fit the search
            search.fit(X_train_scaled, y_train)
            
            # Evaluate on test set
            best_model = search.best_estimator_
            test_predictions = best_model.predict(X_test_scaled)
            
            if target_type == "classification":
                test_score = accuracy_score(y_test, test_predictions)
                test_metrics = {
                    'accuracy': accuracy_score(y_test, test_predictions),
                    'precision': precision_score(y_test, test_predictions, average='weighted'),
                    'recall': recall_score(y_test, test_predictions, average='weighted'),
                    'f1': f1_score(y_test, test_predictions, average='weighted')
                }
            else:
                test_score = -mean_squared_error(y_test, test_predictions)
                test_metrics = {
                    'mse': mean_squared_error(y_test, test_predictions),
                    'mae': mean_absolute_error(y_test, test_predictions),
                    'r2': r2_score(y_test, test_predictions)
                }
            
            # Store results
            experiment = {
                'model_name': model_name,
                'cv_score': search.best_score_,
                'test_score': test_score,
                'test_metrics': test_metrics,
                'best_params': search.best_params_,
                'model': best_model,
                'timestamp': datetime.now()
            }
            
            results['experiments'].append(experiment)
            results['test_scores'][model_name] = test_metrics
            
            # Update best model
            if test_score > results['best_score']:
                results['best_score'] = test_score
                results['best_model'] = best_model
                
                # Feature importance
                if hasattr(best_model, 'feature_importances_'):
                    importance_df = pd.DataFrame({
                        'feature': X_train.columns,
                        'importance': best_model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    results['feature_importance'] = importance_df
            
            self.logger.info(f"✅ {model_name}: CV Score = {search.best_score_:.4f}, Test Score = {test_score:.4f}")
        
        # Save best model
        self.best_model = results['best_model']
        self.best_score = results['best_score']
        
        # Save experiment results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"models/{self.experiment_name}_optimization_{timestamp}.joblib"
        os.makedirs("models", exist_ok=True)
        joblib.dump(results, filename)
        
        self.logger.info(f"🎉 Optimierung abgeschlossen! Beste Performance: {results['best_score']:.4f}")
        self.logger.info(f"💾 Ergebnisse gespeichert: {filename}")
        
        return results
    
    def _get_parameter_spaces(self) -> Dict[str, Dict[str, Dict]]:
        """Definiert die Hyperparameter-Suchräume für alle Modelle"""
        
        spaces = {
            'classification': {
                'xgboost': {
                    'n_estimators': [100, 200, 300, 500],
                    'max_depth': [3, 5, 7, 9],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0],
                    'reg_alpha': [0, 0.1, 0.5, 1.0],
                    'reg_lambda': [0, 0.1, 0.5, 1.0]
                },
                'random_forest': {
                    'n_estimators': [100, 200, 300, 500],
                    'max_depth': [None, 5, 10, 15, 20],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['auto', 'sqrt', 'log2']
                },
                'logistic_regression': {
                    'C': [0.1, 1.0, 10.0, 100.0],
                    'penalty': ['l1', 'l2', 'elasticnet'],
                    'solver': ['liblinear', 'saga'],
                    'max_iter': [1000, 2000, 5000]
                },
                'svm': {
                    'C': [0.1, 1.0, 10.0, 100.0],
                    'kernel': ['linear', 'rbf', 'poly'],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
                }
            },
            'regression': {
                'xgboost': {
                    'n_estimators': [100, 200, 300, 500],
                    'max_depth': [3, 5, 7, 9],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0],
                    'reg_alpha': [0, 0.1, 0.5, 1.0],
                    'reg_lambda': [0, 0.1, 0.5, 1.0]
                },
                'random_forest': {
                    'n_estimators': [100, 200, 300, 500],
                    'max_depth': [None, 5, 10, 15, 20],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['auto', 'sqrt', 'log2']
                },
                'linear_regression': {
                    'fit_intercept': [True, False],
                    'normalize': [True, False]
                }
            }
        }
        
        # Convert to Bayesian optimization format if available
        if BAYESIAN_OPT_AVAILABLE:
            bayesian_spaces = {}
            for task_type, models in spaces.items():
                bayesian_spaces[task_type] = {}
                for model_name, params in models.items():
                    bayesian_params = {}
                    for param_name, param_values in params.items():
                        if isinstance(param_values[0], (int, float)):
                            if isinstance(param_values[0], int):
                                bayesian_params[param_name] = Integer(min(param_values), max(param_values))
                            else:
                                bayesian_params[param_name] = Real(min(param_values), max(param_values))
                        else:
                            bayesian_params[param_name] = Categorical(param_values)
                    bayesian_spaces[task_type][model_name] = bayesian_params
            return bayesian_spaces
        
        return spaces
    
    def generate_model_report(self, results: Dict[str, Any], save_path: str = None) -> str:
        """
        Generiert einen umfassenden Modellbericht
        
        Args:
            results: Optimierungsergebnisse
            save_path: Pfad zum Speichern des Berichts
            
        Returns:
            Bericht als String
        """
        
        report = f"""
# ML Model Optimization Report
## Experiment: {self.experiment_name}
### Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Executive Summary

**Best Model Performance:** {results['best_score']:.4f}
**Total Experiments:** {len(results['experiments'])}
**Best Model Type:** {results['best_model'].__class__.__name__ if results['best_model'] else 'N/A'}

---

## Model Comparison

| Model | CV Score | Test Score | Best Parameters |
|-------|----------|------------|-----------------|
"""
        
        for exp in results['experiments']:
            params_str = str(exp['best_params'])[:50] + "..." if len(str(exp['best_params'])) > 50 else str(exp['best_params'])
            report += f"| {exp['model_name']} | {exp['cv_score']:.4f} | {exp['test_score']:.4f} | {params_str} |\n"
        
        report += "\n---\n\n## Detailed Results\n\n"
        
        for exp in results['experiments']:
            report += f"### {exp['model_name']}\n\n"
            report += f"- **Cross-Validation Score:** {exp['cv_score']:.4f}\n"
            report += f"- **Test Score:** {exp['test_score']:.4f}\n"
            report += f"- **Best Parameters:**\n"
            for param, value in exp['best_params'].items():
                report += f"  - {param}: {value}\n"
            
            report += f"- **Test Metrics:**\n"
            for metric, value in exp['test_metrics'].items():
                report += f"  - {metric}: {value:.4f}\n"
            report += "\n"
        
        # Feature Importance
        if results.get('feature_importance') is not None:
            report += "---\n\n## Feature Importance (Top 20)\n\n"
            report += "| Rank | Feature | Importance |\n"
            report += "|------|---------|------------|\n"
            
            for i, (_, row) in enumerate(results['feature_importance'].head(20).iterrows()):
                report += f"| {i+1} | {row['feature']} | {row['importance']:.4f} |\n"
        
        # Recommendations
        report += "\n---\n\n## Recommendations\n\n"
        
        best_exp = max(results['experiments'], key=lambda x: x['test_score'])
        report += f"1. **Use {best_exp['model_name']} as production model**\n"
        report += f"2. **Monitor model performance** - Current test accuracy: {best_exp['test_score']:.4f}\n"
        
        if results.get('feature_importance') is not None:
            top_features = results['feature_importance'].head(10)['feature'].tolist()
            report += f"3. **Focus on top features:** {', '.join(top_features[:5])}\n"
        
        report += "4. **Consider ensemble methods** for improved robustness\n"
        report += "5. **Regular retraining** recommended as market conditions change\n"
        
        # Save report if path provided
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
            self.logger.info(f"📄 Bericht gespeichert: {save_path}")
        
        return report

# Convenience functions for easy usage
def create_trading_model(data: pd.DataFrame, 
                        symbol: str,
                        target_type: str = "classification",
                        horizon: int = 1,
                        experiment_name: str = None) -> Tuple[Any, Dict[str, Any], str]:
    """
    Convenience function to create an optimized trading model
    
    Args:
        data: Historical price data
        symbol: Trading symbol
        target_type: 'classification' or 'regression'
        horizon: Prediction horizon
        experiment_name: Custom experiment name
        
    Returns:
        Tuple of (best_model, results_dict, report_string)
    """
    
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"{symbol}_{target_type}_{timestamp}"
    
    # Initialize engine
    engine = AdvancedMLEngine(experiment_name=experiment_name)
    
    # Run optimization
    results = engine.run_automated_optimization(
        data=data,
        target_type=target_type,
        horizon=horizon,
        max_trials=50,
        optimization_method="bayesian" if BAYESIAN_OPT_AVAILABLE else "random"
    )
    
    # Generate report
    report = engine.generate_model_report(results)
    
    return results['best_model'], results, report

if __name__ == "__main__":
    # Example usage
    print("Advanced ML Engine - Ready for intelligent model development!")
    print(f"Bayesian Optimization: {'Available' if BAYESIAN_OPT_AVAILABLE else 'Not Available'}")
    
    # Create sample data for testing
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='1H')
    sample_data = pd.DataFrame({
        'open': np.random.randn(len(dates)).cumsum() + 100,
        'high': np.random.randn(len(dates)).cumsum() + 105,
        'low': np.random.randn(len(dates)).cumsum() + 95,
        'close': np.random.randn(len(dates)).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    
    print(f"Sample data shape: {sample_data.shape}")
