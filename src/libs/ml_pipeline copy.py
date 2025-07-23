"""
ML Pipeline class for aquarium parameter prediction.
"""

import joblib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config.settings import settings
import logging

logger = logging.getLogger(__name__)


class MLPipeline:
    """ML Pipeline for aquarium parameter prediction with 30-minute horizon."""
    
    def __init__(self):
        # Model storage - organized by aquarium_id and parameter
        self.models: Dict[str, RandomForestRegressor] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.feature_columns: Dict[str, List[str]] = {}
        self.model_metadata: Dict[str, Dict] = {}
        
        # Fixed prediction horizon: 30 minutes = 6 periods of 5-minute intervals
        self.PREDICTION_HORIZON_PERIODS = 6
        self.PREDICTION_HORIZON_MINUTES = 30
        
        # Parameters to predict
        self.parameters = ['ph', 'water_temperature', 'dissolved_oxygen']
        
        # Feature engineering configuration
        self.lag_periods = [1, 2, 6, 12, 36, 72, 144, 288]  # 5min to 24h
        self.rolling_windows = [6, 12, 36, 72, 144, 288]    # 30min to 24h
        self.change_periods = [1, 2, 6, 12, 36, 72]         # 5min to 6h
        
        # Ensure model directory exists
        os.makedirs(settings.MODEL_SAVE_PATH, exist_ok=True)
        
        # Load saved models on initialization
        self.load_saved_models()
    
    def _get_model_key(self, aquarium_id: str, parameter: str) -> str:
        """Generate model key for aquarium and parameter combination."""
        return f"{aquarium_id}_{parameter}"
    
    def load_saved_models(self) -> bool:
        """
        Load previously saved models, scalers, and metadata for all aquariums.
        
        Returns:
            True if any models loaded successfully, False otherwise
        """
        logger.info(f"Loading saved models from {settings.model_save_path}")
        
        loaded_count = 0
        
        try:
            # Scan for all model files
            for model_file in settings.model_save_path.glob("*_model.joblib"):
                model_key = model_file.stem.replace('_model', '')
                scaler_file = settings.model_save_path / f"{model_key}_scaler.joblib"
                metadata_file = settings.model_save_path / f"{model_key}_metadata.json"
                
                if scaler_file.exists() and metadata_file.exists():
                    try:
                        # Load model
                        self.models[model_key] = joblib.load(model_file)
                        
                        # Load scaler
                        self.scalers[model_key] = joblib.load(scaler_file)
                        
                        # Load metadata
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                            self.model_metadata[model_key] = metadata
                            self.feature_columns[model_key] = metadata['feature_columns']
                        
                        logger.info(f"✅ Loaded model {model_key} (trained: {metadata.get('trained_at', 'unknown')})")
                        loaded_count += 1
                        
                    except Exception as e:
                        logger.error(f"❌ Error loading model {model_key}: {e}")
                else:
                    logger.warning(f"⚠️ Incomplete model files for {model_key}")
        
        except Exception as e:
            logger.error(f"Error scanning model directory: {e}")
        
        logger.info(f"Loaded {loaded_count} models total")
        return loaded_count > 0
    
    def save_model(self, aquarium_id: str, parameter: str, model: RandomForestRegressor, 
                   scaler: StandardScaler, feature_columns: List[str], 
                   performance: Dict, training_info: Dict) -> bool:
        """
        Save trained model, scaler, and metadata to disk.
        
        Args:
            aquarium_id: Aquarium identifier
            parameter: Parameter name ('ph', 'water_temperature', etc.)
            model: Trained RandomForest model
            scaler: Fitted StandardScaler
            feature_columns: List of feature column names
            performance: Model performance metrics
            training_info: Additional training information
            
        Returns:
            True if saved successfully, False otherwise
        """
        model_key = self._get_model_key(aquarium_id, parameter)
        
        try:
            # Save model
            model_file = settings.model_save_path / f"{model_key}_model.joblib"
            joblib.dump(model, model_file)
            
            # Save scaler
            scaler_file = settings.model_save_path / f"{model_key}_scaler.joblib"
            joblib.dump(scaler, scaler_file)
            
            # Prepare metadata
            metadata = {
                'aquarium_id': aquarium_id,
                'parameter': parameter,
                'feature_columns': feature_columns,
                'prediction_horizon_periods': self.PREDICTION_HORIZON_PERIODS,
                'prediction_horizon_minutes': self.PREDICTION_HORIZON_MINUTES,
                'performance': performance,
                'trained_at': datetime.now().isoformat(),
                'training_data_points': training_info.get('training_data_points', 0),
                'model_parameters': settings.model_parameters
            }
            
            # Save metadata
            metadata_file = settings.model_save_path / f"{model_key}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Update in-memory storage
            self.models[model_key] = model
            self.scalers[model_key] = scaler
            self.feature_columns[model_key] = feature_columns
            self.model_metadata[model_key] = metadata
            
            logger.info(f"✅ Saved model {model_key} with test MAE: {performance.get('test_mae', 'N/A'):.4f}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving model {model_key}: {e}")
            return False
    
    def prepare_features(self, measurements: pd.DataFrame, feeding_times: pd.DataFrame, 
                        water_changes: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features from raw data for training or prediction.
        
        Args:
            measurements: DataFrame with parameter measurements (indexed by time)
            feeding_times: DataFrame with feeding events
            water_changes: DataFrame with water change events
            
        Returns:
            DataFrame with engineered features
        """
        logger.debug("Preparing features from raw data")
        
        try:
            df = measurements.copy()
            
            if df.empty:
                raise ValueError("No measurement data provided")
            
            # 1. Temporal features
            df = self._add_temporal_features(df)
            
            # 2. Lag features
            df = self._add_lag_features(df)
            
            # 3. Rolling statistics
            df = self._add_rolling_features(df)
            
            # 4. Rate of change features
            df = self._add_change_features(df)
            
            # 5. Cross-parameter interactions
            df = self._add_interaction_features(df)
            
            # 6. Maintenance features
            df = self._add_maintenance_features(df, feeding_times, water_changes)
            
            # 7. Minute-level event features
            df = self._add_minute_event_features(df, feeding_times, water_changes)
            
            # Validate that we have enough data
            if len(df) < self.PREDICTION_HORIZON_PERIODS:
                raise ValueError(f"Insufficient data: {len(df)} rows, need at least {self.PREDICTION_HORIZON_PERIODS}")
            
            logger.debug(f"Features prepared: {len(df)} rows")
            
            return df
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            raise
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        df['hour'] = df.index.hour # type: ignore
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        return df
    
    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lag features for all parameters."""
        params = ['ph', 'water_temperature', 'room_temperature', 'dissolved_oxygen']
        
        # Map periods to human-readable labels
        period_labels = {
            1: "5min", 2: "10min", 6: "30min", 12: "1h", 
            36: "3h", 72: "6h", 144: "12h", 288: "24h"
        }
        
        for param in params:
            if param in df.columns:
                for period in self.lag_periods:
                    if period in period_labels:
                        label = period_labels[period]
                        df[f'{param}_lag_{label}'] = df[param].shift(period)
        
        return df
    
    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling statistics features."""
        params = ['ph', 'water_temperature', 'dissolved_oxygen']
        
        # Map windows to human-readable labels
        window_labels = {
            6: "30min", 12: "1h", 36: "3h", 72: "6h", 144: "12h", 288: "24h"
        }
        
        for param in params:
            if param in df.columns:
                for window in self.rolling_windows:
                    if window in window_labels:
                        label = window_labels[window]
                        
                        # Basic rolling statistics
                        df[f'{param}_rolling_mean_{label}'] = df[param].rolling(window).mean()
                        df[f'{param}_rolling_std_{label}'] = df[param].rolling(window).std()
                        
                        # Min/max for longer windows
                        if label in ["6h", "12h", "24h"]:
                            df[f'{param}_rolling_min_{label}'] = df[param].rolling(window).min()
                            df[f'{param}_rolling_max_{label}'] = df[param].rolling(window).max()
        
        return df
    
    def _add_change_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rate of change features."""
        params = ['ph', 'water_temperature', 'dissolved_oxygen']
        
        # Map periods to human-readable labels
        period_labels = {
            1: "5min", 2: "10min", 6: "30min", 12: "1h", 36: "3h", 72: "6h"
        }
        
        for param in params:
            if param in df.columns:
                for period in self.change_periods:
                    if period in period_labels:
                        label = period_labels[period]
                        
                        # Absolute change
                        df[f'{param}_change_{label}'] = df[param].diff(period)
                        
                        # Percentage change for key intervals
                        if label in ["30min", "1h", "3h"]:
                            df[f'{param}_pct_change_{label}'] = df[param].pct_change(period)
        
        return df
    
    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cross-parameter interaction features."""
        # Temperature interactions
        if 'room_temperature' in df.columns and 'water_temperature' in df.columns:
            df['temp_difference'] = df['room_temperature'] - df['water_temperature']
            df['temp_ratio'] = df['room_temperature'] / (df['water_temperature'] + 1e-6)
        
        # pH interactions
        if 'ph' in df.columns:
            df['ph_deviation_from_neutral'] = abs(df['ph'] - 7.0)
        
        # Dissolved oxygen interactions
        if 'dissolved_oxygen' in df.columns and 'water_temperature' in df.columns:
            df['do_temperature_interaction'] = df['dissolved_oxygen'] * df['water_temperature']
        
        # pH-temperature interaction
        if 'ph' in df.columns and 'water_temperature' in df.columns:
            df['ph_temp_interaction'] = df['ph'] * df['water_temperature']
        
        # Temperature gradient rate
        if 'temp_difference' in df.columns:
            df['temp_gradient_rate'] = df['temp_difference'].diff(1)
        
        return df
    
    def _add_maintenance_features(self, df: pd.DataFrame, feeding_times: pd.DataFrame, 
                                 water_changes: pd.DataFrame) -> pd.DataFrame:
        """Add maintenance event features."""
        # Hours since feeding
        df['hours_since_feeding'] = df.index.to_series().apply(
            lambda x: self._hours_since_event(x, feeding_times.get('fed_at', pd.Series(dtype='datetime64[ns]')))
        )
        
        # Hours since water change
        df['hours_since_water_change'] = df.index.to_series().apply(
            lambda x: self._hours_since_event(x, water_changes.get('changed_at', pd.Series(dtype='datetime64[ns]')))
        )
        
        # Days since water change
        df['days_since_water_change'] = df['hours_since_water_change'] / 24
        
        # Water change volume (last change)
        df['water_change_volume'] = df.index.to_series().apply(
            lambda x: self._last_water_change_volume(x, water_changes)
        )
        
        # Post-event flags
        df['is_post_water_change_6h'] = (df['hours_since_water_change'] <= 6).astype(int)
        
        return df
    
    def _add_minute_event_features(self, df: pd.DataFrame, feeding_times: pd.DataFrame, 
                                  water_changes: pd.DataFrame) -> pd.DataFrame:
        """Add minute-level event features."""
        # Minutes since feeding
        df['minutes_since_feeding'] = df.index.to_series().apply(
            lambda x: self._minutes_since_event(x, feeding_times.get('fed_at', pd.Series(dtype='datetime64[ns]')))
        )
        
        # Minutes since water change
        df['minutes_since_water_change'] = df.index.to_series().apply(
            lambda x: self._minutes_since_event(x, water_changes.get('changed_at', pd.Series(dtype='datetime64[ns]')))
        )
        
        # Event response windows - Feeding
        df['is_post_feeding_30min'] = (df['minutes_since_feeding'] <= 30).astype(int)
        df['is_post_feeding_60min'] = (df['minutes_since_feeding'] <= 60).astype(int)
        df['is_post_feeding_120min'] = (df['minutes_since_feeding'] <= 120).astype(int)
        df['is_post_feeding_240min'] = (df['minutes_since_feeding'] <= 240).astype(int)
        
        # Event response windows - Water change
        df['is_post_water_change_30min'] = (df['minutes_since_water_change'] <= 30).astype(int)
        df['is_post_water_change_60min'] = (df['minutes_since_water_change'] <= 60).astype(int)
        df['is_post_water_change_180min'] = (df['minutes_since_water_change'] <= 180).astype(int)
        df['is_post_water_change_360min'] = (df['minutes_since_water_change'] <= 360).astype(int)
        
        # Cyclical feeding response
        df['feeding_response_sin'] = np.sin(2 * np.pi * df['minutes_since_feeding'] / 120)
        df['feeding_response_cos'] = np.cos(2 * np.pi * df['minutes_since_feeding'] / 120)
        
        # Capped minute features
        df['minutes_since_feeding_capped'] = np.minimum(df['minutes_since_feeding'], 240)
        df['minutes_since_water_change_capped'] = np.minimum(df['minutes_since_water_change'], 360)
        
        return df
    
    def _hours_since_event(self, current_time: pd.Timestamp, event_times: pd.Series) -> float:
        """Calculate hours since last event."""
        if event_times.empty:
            return 999.0  # Large number if no events
        
        past_events = event_times[event_times <= current_time]
        if past_events.empty:
            return 999.0
        
        last_event = past_events.max()
        return (current_time - last_event).total_seconds() / 3600
    
    def _minutes_since_event(self, current_time: pd.Timestamp, event_times: pd.Series) -> float:
        """Calculate minutes since last event."""
        return self._hours_since_event(current_time, event_times) * 60
    
    def _last_water_change_volume(self, current_time: pd.Timestamp, water_changes: pd.DataFrame) -> float:
        """Get volume of last water change."""
        if water_changes.empty or 'changed_at' not in water_changes.columns:
            return 0.0
        
        past_changes = water_changes[water_changes['changed_at'] <= current_time]
        if past_changes.empty:
            return 0.0
        
        last_change = past_changes.loc[past_changes['changed_at'].idxmax()]
        return last_change.get('percentage_changed', 0.0) # type: ignore
    
    def train_model(self, aquarium_id: str, parameter: str, measurements: pd.DataFrame, 
                   feeding_times: pd.DataFrame, water_changes: pd.DataFrame,
                   days_back: int = 30) -> Dict:
        """
        Train a new model for the specified aquarium and parameter.
        
        Args:
            aquarium_id: Aquarium identifier
            parameter: Parameter to predict ('ph', 'water_temperature', 'dissolved_oxygen')
            measurements: DataFrame with parameter measurements
            feeding_times: DataFrame with feeding events
            water_changes: DataFrame with water change events
            days_back: Number of days represented in the data
            
        Returns:
            Dictionary with training results and performance metrics
        """
        logger.info(f"Training {parameter} model for aquarium {aquarium_id} with {days_back} days of data")
        
        try:
            # 1. Prepare features
            df_with_features = self.prepare_features(measurements, feeding_times, water_changes)
            
            # 2. Create target variable (30 minutes ahead)
            df_with_features[f'{parameter}_target'] = df_with_features[parameter].shift(-self.PREDICTION_HORIZON_PERIODS)
            
            # 3. Prepare training dataset
            # Get feature columns (exclude original parameters and target)
            original_params = ['ph', 'water_temperature', 'room_temperature', 'dissolved_oxygen']
            feature_cols = [col for col in df_with_features.columns 
                           if col not in original_params and not col.endswith('_target')]
            
            # Remove rows with missing target
            df_clean = df_with_features.dropna(subset=[f'{parameter}_target'])
            
            if len(df_clean) < 100:
                raise ValueError(f"Insufficient clean data for training: {len(df_clean)} samples")
            
            # Prepare X and y
            X = df_clean[feature_cols].fillna(method='ffill').fillna(method='bfill').fillna(0) # type: ignore
            y = df_clean[f'{parameter}_target']
            
            # 4. Time-aware train/test split (80% train, 20% test)
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # 5. Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # 6. Train model
            model = RandomForestRegressor(**settings.model_parameters)
            model.fit(X_train_scaled, y_train)
            
            # 7. Evaluate model
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)
            
            performance = {
                'train_mae': mean_absolute_error(y_train, y_pred_train),
                'test_mae': mean_absolute_error(y_test, y_pred_test),
                'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'test_r2': r2_score(y_test, y_pred_test),
                'train_samples': len(X_train),
                'test_samples': len(X_test)
            }
            
            # 8. Feature importance
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # 9. Training info
            training_info = {
                'training_data_points': len(X_train),
                'feature_count': len(feature_cols),
                'data_period_days': days_back,
                'trained_at': datetime.now().isoformat()
            }
            
            # 10. Save model
            save_success = self.save_model(
                aquarium_id=aquarium_id,
                parameter=parameter,
                model=model,
                scaler=scaler,
                feature_columns=feature_cols,
                performance=performance,
                training_info=training_info
            )
            
            if not save_success:
                logger.warning(f"Failed to save {parameter} model for aquarium {aquarium_id}")
            
            # 11. Prepare result
            result = {
                'aquarium_id': aquarium_id,
                'parameter': parameter,
                'performance': performance,
                'feature_importance': feature_importance,
                'training_info': training_info,
                'model_saved': save_success
            }
            
            logger.info(f"✅ {parameter} model trained for aquarium {aquarium_id} - Test MAE: {performance['test_mae']:.4f}, R²: {performance['test_r2']:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error training {parameter} model for aquarium {aquarium_id}: {e}")
            raise
    
    def make_prediction(self, aquarium_id: str, parameter: str, measurements: pd.DataFrame,
                       feeding_times: pd.DataFrame, water_changes: pd.DataFrame) -> Dict:
        """
        Make a 30-minute ahead prediction for the specified aquarium and parameter.
        
        Args:
            aquarium_id: Aquarium identifier
            parameter: Parameter to predict ('ph', 'water_temperature', 'dissolved_oxygen')
            measurements: DataFrame with recent parameter measurements
            feeding_times: DataFrame with recent feeding events
            water_changes: DataFrame with recent water change events
            
        Returns:
            Dictionary with prediction results
        """
        logger.debug(f"Making 30-minute prediction for {parameter} in aquarium {aquarium_id}")
        
        try:
            model_key = self._get_model_key(aquarium_id, parameter)
            
            # Check if model is loaded
            if model_key not in self.models:
                raise ValueError(f"No trained model found for {parameter} in aquarium {aquarium_id}")
            
            # 1. Prepare features
            df_with_features = self.prepare_features(measurements, feeding_times, water_changes)
            
            # 2. Get latest features
            feature_cols = self.feature_columns[model_key]
            latest_features = df_with_features[feature_cols].iloc[-1:].fillna(method='ffill').fillna(0) # type: ignore
            
            # 3. Scale features
            latest_scaled = self.scalers[model_key].transform(latest_features)
            
            # 4. Make prediction
            prediction = self.models[model_key].predict(latest_scaled)[0]
            
            # 5. Calculate confidence intervals (simplified using RMSE)
            performance = self.model_metadata[model_key]['performance']
            std_error = performance['test_rmse']
            confidence_lower = prediction - 1.96 * std_error
            confidence_upper = prediction + 1.96 * std_error
            
            # 6. Get current time and target time
            current_time = df_with_features.index[-1]
            target_time = current_time + pd.Timedelta(minutes=self.PREDICTION_HORIZON_MINUTES)
            
            # 7. Prepare result
            result = {
                'aquarium_id': aquarium_id,
                'parameter': parameter,
                'prediction': prediction,
                'confidence_lower': confidence_lower,
                'confidence_upper': confidence_upper,
                'current_value': df_with_features[parameter].iloc[-1],
                'predicted_at': current_time,
                'target_time': target_time,
                'model_performance': performance
            }
            
            logger.debug(f"✅ {parameter} prediction for aquarium {aquarium_id}: {prediction:.3f} (current: {result['current_value']:.3f})")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error making prediction for {parameter} in aquarium {aquarium_id}: {e}")
            raise
    
    def retrain_model(self, aquarium_id: str, parameter: str, measurements: pd.DataFrame,
                     feeding_times: pd.DataFrame, water_changes: pd.DataFrame,
                     actual_predictions: pd.DataFrame, days_back: int = 30) -> Dict:
        """
        Retrain model with latest data, evaluating against recent predictions.
        
        Args:
            aquarium_id: Aquarium identifier
            parameter: Parameter to retrain
            measurements: DataFrame with parameter measurements
            feeding_times: DataFrame with feeding events
            water_changes: DataFrame with water change events
            actual_predictions: DataFrame with prediction_id, predicted_value, actual_value, error
            days_back: Number of days of data used for retraining
            
        Returns:
            Dictionary with retraining results and evaluation
        """
        logger.info(f"Retraining {parameter} model for aquarium {aquarium_id} with evaluation of recent predictions")
        
        try:
            # 1. Evaluate recent predictions
            evaluation_results = self._evaluate_predictions(aquarium_id, parameter, actual_predictions)
            
            # 2. Train new model
            training_results = self.train_model(aquarium_id, parameter, measurements, feeding_times, water_changes, days_back)
            
            # 3. Compare old vs new model performance
            comparison = self._compare_model_performance(aquarium_id, parameter, training_results, evaluation_results)
            
            # 4. Prepare comprehensive result
            result = {
                'aquarium_id': aquarium_id,
                'parameter': parameter,
                'retrained_at': datetime.now().isoformat(),
                'training_results': training_results,
                'evaluation_results': evaluation_results,
                'model_comparison': comparison,
                'improvement': comparison.get('improved', False)
            }
            
            logger.info(f"✅ {parameter} model retrained for aquarium {aquarium_id} - "
                       f"New test MAE: {training_results['performance']['test_mae']:.4f}, "
                       f"Improvement: {comparison.get('improved', False)}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error retraining {parameter} model for aquarium {aquarium_id}: {e}")
            raise
    
    def get_model_info(self, aquarium_id: str) -> Dict:
        """
        Get information about trained models for a specific aquarium.
        
        Args:
            aquarium_id: Aquarium identifier
            
        Returns:
            Dictionary with model information
        """
        info = {
            "aquarium_id": aquarium_id,
            "models": {},
            "total_models": 0
        }
        
        for param in self.parameters:
            model_key = self._get_model_key(aquarium_id, param)
            if model_key in self.models:
                metadata = self.model_metadata.get(model_key, {})
                info["models"][param] = {
                    "trained": True,
                    "model_type": type(self.models[model_key]).__name__,
                    "trained_at": metadata.get('trained_at', 'unknown'),
                    "test_mae": metadata.get('performance', {}).get('test_mae', 'unknown'),
                    "test_r2": metadata.get('performance', {}).get('test_r2', 'unknown')
                }
                info["total_models"] += 1
            else:
                info["models"][param] = {"trained": False}
        
        return info
    
    def _evaluate_predictions(self, aquarium_id: str, parameter: str, actual_predictions: pd.DataFrame) -> Dict:
        """
        Evaluate predictions against actual measurements for specific aquarium and parameter.
        
        Args:
            aquarium_id: Aquarium identifier
            parameter: Parameter to evaluate
            actual_predictions: DataFrame with predicted_value, actual_value, error columns
            
        Returns:
            Dictionary with evaluation metrics
        """
        try:
            if actual_predictions.empty:
                return {'error': 'No predictions to evaluate'}
            
            # Filter for the specific aquarium and parameter
            mask = (actual_predictions.get('aquarium_id', aquarium_id) == aquarium_id) & \
                   (actual_predictions.get('target_parameter', parameter) == parameter)
            param_predictions = actual_predictions[mask]
            
            if param_predictions.empty:
                return {'error': f'No predictions found for parameter {parameter} in aquarium {aquarium_id}'}
            
            # Calculate evaluation metrics
            errors = param_predictions['error'].values if 'error' in param_predictions.columns else []
            predicted_values = param_predictions['predicted_value'].values
            actual_values = param_predictions['actual_value'].values
            
            if len(errors) == 0:
                errors = np.abs(predicted_values - actual_values) # type: ignore
            
            evaluation_metrics = {
                'aquarium_id': aquarium_id,
                'parameter': parameter,
                'evaluated_predictions': len(param_predictions),
                'mae': np.mean(errors), # type: ignore
                'rmse': np.sqrt(np.mean(errors**2)), # type: ignore
                'max_error': np.max(errors), # type: ignore
                'min_error': np.min(errors), # type: ignore
                'r2': r2_score(actual_values, predicted_values) if len(actual_values) > 1 else None # type: ignore
            }
            
            logger.info(f"Evaluated {len(param_predictions)} recent {parameter} predictions for aquarium {aquarium_id} - MAE: {evaluation_metrics['mae']:.4f}")
            
            return evaluation_metrics
            
        except Exception as e:
            logger.error(f"Error evaluating predictions for {parameter} in aquarium {aquarium_id}: {e}")
            return {'error': str(e)}
    
    def _compare_model_performance(self, aquarium_id: str, parameter: str, new_training_results: Dict, 
                                  evaluation_results: Dict) -> Dict:
        """
        Compare new model performance with previous model and recent predictions.
        
        Args:
            aquarium_id: Aquarium identifier
            parameter: Parameter being compared
            new_training_results: Results from new model training
            evaluation_results: Results from evaluating recent predictions
            
        Returns:
            Dictionary with comparison results
        """
        try:
            model_key = self._get_model_key(aquarium_id, parameter)
            new_test_mae = new_training_results['performance']['test_mae']
            new_test_r2 = new_training_results['performance']['test_r2']
            
            comparison = {
                'aquarium_id': aquarium_id,
                'parameter': parameter,
                'new_model_test_mae': new_test_mae,
                'new_model_test_r2': new_test_r2
            }
            
            # Compare with previous model if available
            if model_key in self.model_metadata:
                old_metadata = self.model_metadata[model_key]
                old_test_mae = old_metadata['performance']['test_mae']
                old_test_r2 = old_metadata['performance']['test_r2']
                
                comparison.update({
                    'old_model_test_mae': old_test_mae,
                    'old_model_test_r2': old_test_r2,
                    'mae_improvement': old_test_mae - new_test_mae,
                    'r2_improvement': new_test_r2 - old_test_r2,
                    'improved': new_test_mae < old_test_mae and new_test_r2 > old_test_r2
                })
            
            # Compare with real-world prediction performance
            if 'mae' in evaluation_results:
                real_world_mae = evaluation_results['mae']
                comparison.update({
                    'real_world_mae': real_world_mae,
                    'real_world_vs_test': real_world_mae - new_test_mae
                })
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing model performance for {parameter} in aquarium {aquarium_id}: {e}")
            return {'error': str(e)}