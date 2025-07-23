"""
ML Pipeline class for aquarium parameter prediction.
"""

import joblib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Literal
import os
from src.libs.supabase_manager import SupabaseManager
from src.libs.features_engineering.water_temperature_features import FeatureEngineeringWaterTemperature
from xgboost import XGBRegressor
import xgboost as xgb
import math
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
        self.supabase = SupabaseManager()
        self.features_water_temperature = FeatureEngineeringWaterTemperature()
        self.models: Dict[str, XGBRegressor] = {}
        self.metadata: Dict[str, Dict] = {}

        self.model_dir = settings.MODEL_SAVE_PATH
        
        # Parameters to predict
        self.parameters = ['water_temperature', 'ph', 'do']
        
        # Ensure model directory exists
        os.makedirs(settings.MODEL_SAVE_PATH, exist_ok=True)
        
        # Load saved models on initialization
        self._load_saved_models()
    
    def _load_saved_models(self) -> bool:
        """
        Load saved models and metadata from the model directory. 
        """
        try:
            for model_file in os.listdir(self.model_dir):
                if model_file.endswith('_model.joblib'):
                    model_key = model_file.replace('_model.joblib', '')
                    model_path = os.path.join(self.model_dir, model_file)
                    self.models[model_key] = joblib.load(model_path)

            for metadata_file in os.listdir(self.model_dir):
                if metadata_file.endswith('_metadata.json'):
                    model_key = metadata_file.replace('_metadata.json', '')
                    metadata_path = os.path.join(self.model_dir, metadata_file)
                    with open(metadata_path, 'r') as f:
                        self.metadata[model_key] = json.load(f)

            logger.info(f"Loaded {len(self.models)} models and {len(self.metadata)} metadata entries.")

            return True
        except Exception as e:
            logger.error(f"Error loading saved models: {e}")
            return False

    def _get_model_key(self, aquarium_id: str, parameter: Literal['water_temperature', 'ph', 'do']) -> str:
        """
        Generate a unique key for the model based on aquarium ID and parameter.
        """
        return f"{aquarium_id}_{parameter}"
    
    def _save_model(self,
                    aquarium_id: str,
                    parameter: Literal['water_temperature', 'ph', 'do'],
                    model: XGBRegressor,
                    performance: Dict,
                    training_info: Dict):
        
        model_key = self._get_model_key(aquarium_id, parameter)

        try:
            # Save the model to a file
            model_file = f"{self.model_dir}/{model_key}_model.joblib"
            joblib.dump(model, model_file)

            # Prepare metadata
            metadata = {
                'aquarium_id': aquarium_id,
                'parameter': parameter,
                'performance': performance,
                'trained_at': datetime.now().isoformat(),
                'training_info': training_info
            }

            # Save metadata
            metadata_file = f"{self.model_dir}/{model_key}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            # Update the models dictionary
            self.models[model_key] = model

            # Log the activity
            self.supabase.log_ml_activity(
                aquarium_id=aquarium_id,
                activity_type="model_save",
                status="success",
                metadata={
                    "parameter": parameter,
                    "performance": performance,
                    "training_info": training_info
                }
            )

        except Exception as e:
            logger.error(f"Error saving model for {aquarium_id} - {parameter}: {e}")
            self.supabase.log_ml_activity(
                aquarium_id=aquarium_id,
                activity_type="error",
                error_message=str(e),
                metadata={
                    "parameter": parameter,
                    "performance": performance,
                    "training_info": training_info
                    }
            )
            raise

    def train_water_temperature(self, 
                                  aquarium_id: str, 
                                  historical_data: pd.DataFrame,
                                  water_change_data: pd.DataFrame,
                                  days_back: int,
                                ) -> Dict:
        """
        Train the water temperature prediction model.
        """
        try:
            start_time = datetime.now(timezone.utc)

            # Prepare features
            df = self.features_water_temperature.prepare_all_features(
                aquarium_id,
                historical_data=historical_data[['created_at', 'water_temperature']].copy(),
                water_change_data=water_change_data
            )

            # Split data into training and testing sets
            train_size = int(len(df) * 0.8)
            train, test = df[:train_size], df[train_size:]

            FEATURES = df.columns.tolist()
            FEATURES.remove('water_temperature')  # Remove target variable from features
            TARGET = 'water_temperature'

            X_train = train[FEATURES]
            y_train = train[TARGET]

            X_test = test[FEATURES]
            y_test = test[TARGET]

            # Train the model
            model = xgb.XGBRegressor(
                n_estimators=1000,
                early_stopping_rounds=50,
                learning_rate=0.01,
                max_depth=6,
                random_state=42
            )
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_train, y_train), (X_test, y_test)],
                verbose=100
            )

            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            # Evaluate the model
            evaluation_metrics = {
                "aquarium_id": aquarium_id,
                "parameter": TARGET,
                "days_back": days_back,
                "train_mae": mean_absolute_error(y_train, y_pred_train),
                "test_mae": mean_absolute_error(y_test, y_pred_test),
                "train_rmse": np.sqrt(mean_squared_error(y_train, y_pred_train)),
                "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred_test)),
                "train_r2": r2_score(y_train, y_pred_train),
                "test_r2": r2_score(y_test, y_pred_test),
                "train_samples": len(y_train),
                "test_samples": len(y_test)
            }

            # Feature importance
            feature_importance = pd.DataFrame(
                data=model.feature_importances_,
                index=model.feature_names_in_,
                columns=['importance']
            ).sort_values('importance', ascending=False)

            # Retraining with 0.9 of the data with early stopping
            split_idx = int(len(df) * 0.9)
            X_train_split = df[FEATURES].iloc[:split_idx]
            y_train_split = df[TARGET].iloc[:split_idx]
            X_val_split = df[FEATURES].iloc[split_idx:]
            y_val_split = df[TARGET].iloc[split_idx:]

            final_model = xgb.XGBRegressor(
                n_estimators=1000,
                early_stopping_rounds=50,
                learning_rate=0.01,
                max_depth=6,
                random_state=42
            )
            final_model.fit(
                X_train_split,
                y_train_split,
                eval_set=[(X_train_split, y_train_split), (X_val_split, y_val_split)],
                verbose=100
            )

            # Final evaluation metrics
            final_y_pred_train = final_model.predict(X_train_split)
            final_y_pred_test = final_model.predict(X_val_split)

            final_evaluation_metrics = {
                "aquarium_id": aquarium_id,
                "parameter": TARGET,
                "days_back": days_back,
                "train_mae": mean_absolute_error(y_train_split, final_y_pred_train),
                "test_mae": mean_absolute_error(y_val_split, final_y_pred_test),
                "train_rmse": np.sqrt(mean_squared_error(y_train_split, final_y_pred_train)),
                "test_rmse": np.sqrt(mean_squared_error(y_val_split, final_y_pred_test)),
                "train_r2": r2_score(y_train_split, final_y_pred_train),
                "test_r2": r2_score(y_val_split, final_y_pred_test),
                "train_samples": len(y_train_split),
                "test_samples": len(y_val_split)
            }

            # Feature importance for final model
            final_feature_importance = pd.DataFrame(
                data=final_model.feature_importances_,
                index=final_model.feature_names_in_,
                columns=['importance']
            ).sort_values('importance', ascending=False)

            # Compare evaluation metrics
            if (final_evaluation_metrics['test_rmse'] < evaluation_metrics['test_rmse']):
                logger.info("Final model has better performance than initial model.")
                model = final_model
                evaluation_metrics = final_evaluation_metrics
                feature_importance = final_feature_importance
                df['prediction'] = None
                df.loc[X_val_split.index, 'prediction'] = final_y_pred_test
            else:
                logger.info("Initial model has better performance than final model. Using initial model.")
                df['prediction'] = None
                df.loc[X_test.index, 'prediction'] = y_pred_test

            # Save the model and metadata
            self._save_model(
                aquarium_id=aquarium_id,
                parameter=TARGET,
                model=model,
                performance=evaluation_metrics,
                training_info={
                    "features": FEATURES,
                    "feature_importance": feature_importance.to_dict(),
                    "processing_time": (datetime.now(timezone.utc) - start_time).total_seconds(),
                    "model_version": f"{TARGET}_{datetime.now(timezone.utc).isoformat()}"
                }
            )

            return {
                "evaluation_metrics": evaluation_metrics,
                "feature_importance": feature_importance,
                "dataframe": df,
                "success": True,
            }

        except Exception as e:
            return {
                "error": str(e),
                "success": False
            }

    def predict(self,
                aquarium_id: str,
                date_time_now: datetime):
        """
        Predict the specified parameter for the given aquarium within the date range.
        """

        try:
            predicted_water_temperature = self.predict_water_temperature(
                aquarium_id,
                date_time_now
            )

        except Exception as e:
            logger.error(f"Error predicting water temperature for aquarium {aquarium_id}: {e}")
            raise

    def predict_water_temperature(
            self,
            aquarium_id: str,
            date_time_now: datetime,
    ):
        """
        Predict the water temperature for the given aquarium at the specified date and time.
        """
        try:
            model = self.models[self._get_model_key(aquarium_id, 'water_temperature')]
            model_metadata = self.metadata[self._get_model_key(aquarium_id, 'water_temperature')]

            if not model or not model_metadata:
                raise ValueError(f"No model found for aquarium {aquarium_id} and parameter 'water_temperature'.")
            
            # Fetch the latest 30 minute historical data
            historical_data = self.supabase.get_historical_data(
                aquarium_id=aquarium_id,
                start_date=(date_time_now - timedelta(hours=6)),
                end_date=date_time_now
            )

            # Fetch water change data
            water_change_data = self.supabase.get_water_changing_data(
                aquarium_id=aquarium_id,
                start_date=(date_time_now - timedelta(hours=6)),
                end_date=date_time_now
            )

            # Get aquarium location data
            aquarium_geo = self.supabase.get_aquarium_geo(aquarium_id)
            if aquarium_geo is None:
                raise ValueError(f"Aquarium {aquarium_id} not found or geo data missing")

            # Get Wheather Forecast
            forecast = self.features_water_temperature._get_weather_forecast(
                aquarium_geo=aquarium_geo,
                start_date=(date_time_now - timedelta(hours=6)).strftime('%Y-%m-%d'),
                end_date=(date_time_now + timedelta(hours=6)).strftime('%Y-%m-%d')
            )

            # Reindex to 15-minute intervals
            historical_data['created_at'] = pd.to_datetime(historical_data['created_at'], format='ISO8601')
            historical_data.set_index('created_at', inplace=True)
            historical_data.sort_index(inplace=True)

            full_index = pd.date_range(start=historical_data.index.min(), end=historical_data.index.max(), freq='15min')
            historical_data = historical_data.reindex(full_index)
            historical_data['water_temperature'] = historical_data['water_temperature'].interpolate(method='time').ffill().bfill()
            historical_data.index.name = 'created_at'
            historical_data.reset_index(inplace=True)

            # Concate prediction_df with historical data
            if not isinstance(historical_data.index, pd.DatetimeIndex):
                historical_data['created_at'] = pd.to_datetime(historical_data['created_at'], format='ISO8601')
                historical_data.set_index('created_at', inplace=True)

            # Create new DataFrame for prediction
            index_range = pd.date_range(start=historical_data.index.min(), end=historical_data.index.max() + timedelta(hours=6), freq='15min')
            prediction_df = pd.DataFrame(index=index_range)
            prediction_df.index.name = 'created_at'
            prediction_df.index = pd.to_datetime(prediction_df.index, format='ISO8601')

            prediction_df = historical_data.reindex(index=prediction_df.index)

            # Prepare features
            FEATURES = model_metadata['training_info']['features']

            # Add prediction label for indicating if the row is a prediction
            prediction_df.loc[prediction_df['water_temperature'].isna(), 'is_prediction'] = True
            prediction_df.loc[prediction_df['water_temperature'].notna(), 'is_prediction'] = False

            # Prepare features for prediction
            prediction_df = self.features_water_temperature.prepare_features(prediction_df, dropNan=False, fillna=False)
            prediction_df = self.features_water_temperature.prepare_feature_with_weather(
                df=prediction_df,
                weather_df=forecast.get('forecast', pd.DataFrame()),
                sunset_sunrise_df=forecast.get('sunset_sunrise', pd.DataFrame()),
            )
            prediction_df = self.features_water_temperature.prepare_features_with_water_change(
                prediction_df,
                water_change_data,
            )

            # Predict water temperature
            target_prediction_index: pd.DatetimeIndex = prediction_df[prediction_df['water_temperature'].notna()].index.max() + timedelta(minutes=15)
            while target_prediction_index <= prediction_df.index.max():
                X = prediction_df[target_prediction_index:target_prediction_index][FEATURES]

                if not X.empty and model is not None:
                    prediction = model.predict(X)
                    prediction_df.loc[target_prediction_index, 'water_temperature'] = math.ceil(prediction[0] * 100) / 100.0

                self.features_water_temperature.prepare_lag_features(prediction_df)

                target_prediction_index: pd.DatetimeIndex = prediction_df[prediction_df['water_temperature'].notna()].index.max() + timedelta(minutes=15)

            prediction_df.drop(columns=FEATURES, inplace=True)

            # After all predictions are made, add confidence intervals
            predicted_rows = prediction_df[prediction_df['is_prediction'] == True]
            
            if not predicted_rows.empty:
                predictions = predicted_rows['water_temperature'].values
                test_rmse = model_metadata['performance']['test_rmse']
                
                # Calculate confidence intervals
                confidence_results = self.calculate_prediction_confidence(
                    predictions=predictions,
                    test_rmse=test_rmse,
                    confidence_level=0.95
                )
                
                # Add to DataFrame
                for i, result in enumerate(confidence_results):
                    row_idx = predicted_rows.index[i]
                    prediction_df.loc[row_idx, 'confidence_lower'] = result['confidence_lower']
                    prediction_df.loc[row_idx, 'confidence_upper'] = result['confidence_upper']
                    prediction_df.loc[row_idx, 'std_error'] = result['std_error']

            # Insert to Supabase
            self.supabase.insert_prediction(
                aquarium_id=aquarium_id,
                parameter='water_temperature',
                model_version=model_metadata['training_info']['model_version'],
                data=prediction_df[prediction_df['is_prediction'] == True].reset_index().rename(columns={'created_at': 'target_time'})
            )

            # Remove unnecessary columns
            prediction_df.drop(
                columns=['confidence_lower', 'confidence_upper', 'std_error'],
            )

            return prediction_df

        except Exception as e:
            logger.error(f"Error predicting water temperature for aquarium {aquarium_id}: {e}")
            self.supabase.log_ml_activity(
                aquarium_id=aquarium_id,
                activity_type="error",
                error_message=f"Error predicting water temperature: {e}",
                metadata={"date_time_now": date_time_now.isoformat()}
            )
            raise

    def calculate_prediction_confidence(self, predictions, test_rmse, confidence_level=0.95):
        """
        Calculate growing confidence intervals for sequential predictions
        """
        import scipy.stats as stats
        import numpy as np
        
        z_score = stats.norm.ppf(1 - (1-confidence_level)/2)
        results = []
        
        for step, pred in enumerate(predictions):
            # Uncertainty grows with prediction steps
            step_uncertainty = test_rmse * np.sqrt(step + 1)
            
            # Optional: Add small drift component
            drift_uncertainty = test_rmse * 0.03 * step  # 3% linear growth
            total_std = np.sqrt(step_uncertainty**2 + drift_uncertainty**2)
            
            lower = pred - z_score * total_std
            upper = pred + z_score * total_std
            
            results.append({
                'prediction': pred,
                'confidence_lower': round(lower, 2),
                'confidence_upper': round(upper, 2),
                'std_error': round(total_std, 3)
            })
        
        return results
