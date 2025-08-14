"""
ML Pipeline class for aquarium parameter prediction.
"""

import joblib
import json
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Literal
import os
from src.libs.supabase_manager import SupabaseManager
from src.libs.features_engineering.water_temperature_features import FeatureEngineeringWaterTemperature
from src.libs.features_engineering.ph_features import FeatureEngineeringPH
from xgboost import XGBRegressor
import xgboost as xgb
import math
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pytz
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from scipy import stats

from src.config.settings import settings
import logging

logger = logging.getLogger(__name__)

class MLPipeline:
    """ML Pipeline for aquarium parameter prediction with 30-minute horizon."""
    
    def __init__(self):
        # Model storage - organized by aquarium_id and parameter
        self.supabase = SupabaseManager()
        self.features_water_temperature = FeatureEngineeringWaterTemperature()
        self.features_ph = FeatureEngineeringPH()
        self.models: Dict[str, XGBRegressor] = {}
        self.metadata: Dict[str, Dict] = {}

        # Data configuration
        self.DATA_INTERVAL = '1h'
        self.PERIODS_IN_HOUR = 1
        self.PERIODS_IN_DAY = 24
        self.MIN_TRAINING_SAMPLES = self.PERIODS_IN_DAY * 7  # Minimum samples required for training (7 days of hourly data)

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
            # Clear existing models and metadata
            self.models.clear()
            self.metadata.clear()

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
            self.metadata[model_key] = metadata

        except Exception as e:
            logger.error(f"Error saving model for {aquarium_id} - {parameter}: {e}")
            raise

    def validate_prediction(self,
                            aquarium_id: str,
                            parameter: Literal['water_temperature', 'ph', 'do'],
                            target_time: datetime) -> bool:
        """
        Validate the prediction for a given aquarium and parameter at a specific time.
        """
        try:
            # Fetch the prediction from Supabase
            predicted = self.supabase.get_prediction(aquarium_id, parameter, target_time)
            predicted['target_time'] = pd.to_datetime(predicted['target_time'], format='ISO8601')
            predicted.set_index('target_time', inplace=True)

            # get start and end date for validation
            start_date = predicted.index.min()
            end_date = predicted.index.max()

            actual_historical_data = self.supabase.get_historical_data(aquarium_id=aquarium_id, start_date=start_date, end_date=end_date)

            if actual_historical_data.empty:
                raise ValueError(f"No historical data found for aquarium {aquarium_id} between {start_date} and {end_date}")

            actual_historical_data['created_at'] = pd.to_datetime(actual_historical_data['created_at'], format='ISO8601')
            actual_historical_data.set_index('created_at', inplace=True)

            for index, row in predicted.iterrows():
                actual_value = actual_historical_data.loc[index, parameter] if index in actual_historical_data.index else None # type: ignore
                if actual_value is not None:
                    predicted.at[index, 'actual_value'] = actual_value

            # Drop where actual_value is null
            predicted = predicted.dropna(subset=['actual_value'])

            # Calculate prediction error
            predicted['prediction_error'] = np.abs(predicted['predicted_value'] - predicted['actual_value']) 

            # update validated_at
            validated_at = datetime.now(timezone.utc).isoformat()
            predicted['validated_at'] = validated_at

            # Reset Index
            predicted.reset_index(inplace=True, drop=False)
            predicted['target_time'] = predicted['target_time'].apply(lambda x: x.isoformat())

            # Validate the prediction
            is_success = self.supabase.validate_prediction(
                data=predicted.to_dict(orient='records') # type: ignore
            )

            for _, row in predicted.iterrows():
                # Report the validation
                prediction_error = row['prediction_error']

                if prediction_error > 0.5:
                    self.report_prediction_validation(
                    aquarium_id=aquarium_id,
                    target_time=datetime.fromisoformat(row['target_time']),
                    parameter=parameter,
                    predicted_value=row['predicted_value'],
                    actual_value=row['actual_value'],
                    prediction_error=row['prediction_error'],
                )

            if is_success:
                return True
            else:
                raise ValueError("Failed to validate the prediction in Supabase")

        except Exception as e:
            logger.error(e)
            return False

    def train_models(self, aquarium_id: str):
        try:
            # Fetch aquarium model settings
            aquarium_settings = self.supabase.get_aquarium_model_settings(aquarium_id)

            if not aquarium_settings:
                raise ValueError("No model settings found for the aquarium.")

            trained_models = []

            for parameter in self.parameters:
                if parameter == 'water_temperature':
                    days_back = int(aquarium_settings['train_temp_model_days'])
                    is_success = self.train_water_temperature(aquarium_id, days_back)
                    trained_models.append(parameter) if is_success else None
                elif parameter == 'ph':
                    days_back = int(aquarium_settings['train_ph_model_days'])
                    is_success = self.train_ph(aquarium_id, days_back)
                    trained_models.append(parameter) if is_success else None
                
            if len(trained_models) == 0:
                logger.warning(f"No models were trained for aquarium {aquarium_id}. Check the settings or data availability.")
            else:
                logger.info(f"Successfully trained models for aquarium {aquarium_id}: {', '.join(trained_models)}")
        except Exception as e:
            logger.error(f"Error during model training for aquarium {aquarium_id}: {e}")
            raise

    def train_water_temperature(self, 
                                  aquarium_id: str, 
                                  days_back: int,
        ) -> bool:
        """
            Train the water temperature prediction model.
        """
        try:
            start_date = (datetime.now(timezone.utc) - pd.Timedelta(days=days_back)).replace(minute=0, second=0, microsecond=0)
            end_date = datetime.now(timezone.utc)

            # Fetch historical data
            historical_data = self.supabase.get_historical_data(
                aquarium_id,
                start_date=start_date,
                end_date=end_date
            )

            historical_data = self.convert_from_5min_to_1hour(
                historical_data,
                target_cols=['water_temperature', 'ph', 'do'],
                index_name='created_at'
            )

            if len(historical_data) < self.MIN_TRAINING_SAMPLES:
                raise ValueError(f"Not enough data to train the model. Minimum required samples: {self.MIN_TRAINING_SAMPLES}, available: {len(historical_data)}")
            
            # Fetch water change data
            water_change_data = self.supabase.get_water_changing_data(
                aquarium_id,
                start_date=start_date,
                end_date=end_date
            )

            df, features = self.features_water_temperature.prepare_all_features(
                aquarium_id=aquarium_id,
                historical_data=historical_data[['water_temperature', 'created_at']].copy(),
                water_change_data=water_change_data,
            )

            logger.info(f"Features prepared for water temperature model: {features}")

            # Train the model
            self.train_model(
                aquarium_id=aquarium_id,
                parameter='water_temperature',
                df=df,
                features=features,
                days_back=days_back
            )

            return True
        except Exception as e:
            raise RuntimeError(f"Error during training water temperature model: {e}")

    def train_ph(self,
                 aquarium_id: str,
                 days_back: int) -> bool:
        """
        Train the pH prediction model for the specified aquarium.
        """
        try:
            start_date = (datetime.now(timezone.utc) - timedelta(days=days_back)).replace(minute=0, second=0, microsecond=0)
            end_date = datetime.now(timezone.utc)

            # Fetch historical data
            historical_data =  self.supabase.get_historical_data(aquarium_id, start_date=start_date, end_date=end_date)

            if historical_data.empty:
                raise ValueError(f"No historical data found for aquarium {aquarium_id} in the last {days_back} days.")
            
            historical_data = self.convert_from_5min_to_1hour(
                df=historical_data,
                # target_col='ph',
                index_name='created_at'
            )

            # Check if we have enough data
            if len(historical_data) < self.MIN_TRAINING_SAMPLES:
                raise ValueError(f"Not enough data to train the model. Minimum required samples: {self.MIN_TRAINING_SAMPLES}, available: {len(historical_data)}")
            
            # Fetch water change data
            water_change_data = self.supabase.get_water_changing_data(
                aquarium_id=aquarium_id,
                start_date=start_date,
                end_date=end_date
            )

            # Fetch feeding data
            feeding_data = self.supabase.get_feeding_data(
                aquarium_id=aquarium_id,
                start_date=start_date,
                end_date=end_date
            )

            # Prepare features
            df, features = self.features_ph.prepare_all_features(
                historical_data=historical_data,
                water_change_data=water_change_data,
                feed_data=feeding_data
            )

            self.train_model(
                aquarium_id=aquarium_id,
                parameter='ph',
                df=df,
                features=features,
                days_back=days_back
            )

            return True
        except Exception as e:
            raise Exception(f"Error during training pH model: {e}")
   
    def predict(self,
                aquarium_id: str,
                date_time_now: datetime,
                parameters: List[Literal['water_temperature', 'ph', 'do']] = ['water_temperature', 'ph']
                ):
        """
        Predict the specified parameter for the given aquarium within the date range.
        """
        try:
            # load models and metadata
            self._load_saved_models()

            # Fetch historical data for the last 24 hours
            historical_data = self.supabase.get_historical_data(
                aquarium_id=aquarium_id,
                start_date=(date_time_now - timedelta(hours=24)),
                end_date=date_time_now
            )

            # Fetch water change data
            water_change_data = self.supabase.get_water_changing_data(
                aquarium_id=aquarium_id,
                start_date=(date_time_now - timedelta(hours=24)),
                end_date=date_time_now
            )

            # Convert historical data from 5-minute to 1-hour intervals
            historical_data = self.convert_from_5min_to_1hour(
                historical_data,
                target_cols=['water_temperature', 'ph', 'do'],
                index_name='created_at'
            )

            if not isinstance(historical_data.index, pd.DatetimeIndex):
                historical_data['created_at'] = pd.to_datetime(historical_data['created_at'], format='ISO8601')
                historical_data.set_index('created_at', inplace=True)

            # fill missing values in historical data
            historical_data = historical_data.interpolate(method='linear')

            # Create new DataFrame for prediction
            index_range = pd.date_range(start=historical_data.index.min().replace(minute=0, second=0, microsecond=0), end=historical_data.index.max() + timedelta(hours=24), freq='1h')
            prediction_df = pd.DataFrame(index=index_range)
            prediction_df.index.name = 'created_at'
            prediction_df.index = pd.to_datetime(prediction_df.index, format='ISO8601')
            prediction_df = historical_data.reindex(index=prediction_df.index)

            # Add prediction label for indicating if the row is a prediction
            prediction_df.loc[prediction_df['water_temperature'].isna(), 'is_prediction'] = True
            prediction_df.loc[prediction_df['water_temperature'].notna(), 'is_prediction'] = False

            prediction_df = self.predict_water_temperature(
                aquarium_id=aquarium_id,
                date_time_now=date_time_now,
                prediction_df=prediction_df,
                water_change_data=water_change_data
            )

            if 'ph' in parameters:
                # Fetch Feeding Data
                feeding_data = self.supabase.get_feeding_data(
                    aquarium_id=aquarium_id,
                    start_date=(date_time_now - timedelta(hours=6)),
                    end_date=date_time_now
                )
                prediction_df = self.predict_ph(
                    aquarium_id=aquarium_id,
                    date_time_now=date_time_now,
                    prediction_df=prediction_df,
                    water_change_data=water_change_data,
                    feeding_data=feeding_data
                )   

            return prediction_df
        except Exception as e:
            logger.error(f"Error during prediction for aquarium {aquarium_id}: {e}")
            raise

    def predict_water_temperature(
            self,
            aquarium_id: str,
            date_time_now: datetime,
            prediction_df: pd.DataFrame,
            water_change_data: pd.DataFrame
    ):
        """
        Predict the water temperature for the given aquarium at the specified date and time.
        """
        try:
            logger.info(f"Starting water temperature prediction for aquarium {aquarium_id} at {date_time_now.isoformat()}")
            model = self.models[self._get_model_key(aquarium_id, 'water_temperature')]
            model_metadata = self.metadata[self._get_model_key(aquarium_id, 'water_temperature')]

            if not model or not model_metadata:
                logger.error(f"No model found for aquarium {aquarium_id} and parameter 'water_temperature'.")
                raise ValueError(f"No model found for aquarium {aquarium_id} and parameter 'water_temperature'.")

            # Get aquarium location data
            aquarium_geo = self.supabase.get_aquarium_geo(aquarium_id)
            if aquarium_geo is None:
                raise ValueError(f"Aquarium {aquarium_id} not found or geo data missing")

            # Get Wheather Forecast
            forecast = self.features_water_temperature._get_weather_forecast(
                aquarium_geo=aquarium_geo,
                start_date=(date_time_now - timedelta(hours=24)).strftime('%Y-%m-%d'),
                end_date=(date_time_now + timedelta(hours=24)).strftime('%Y-%m-%d')
            )

            # Prepare features
            FEATURES = model_metadata['training_info']['features']
            logger.info(f"Features for water temperature prediction: {FEATURES}")

            # Prepare features for prediction
            prediction_df = self.features_water_temperature.prepare_features(prediction_df, dropNan=False, fillna=False)
            prediction_df = self.features_water_temperature.prepare_rolling_features(prediction_df)
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
            before_prediction_index: pd.DatetimeIndex = prediction_df.loc[prediction_df['water_temperature'].notna()].index.max()
            target_prediction_index: pd.DatetimeIndex = before_prediction_index + timedelta(hours=1)

            while target_prediction_index <= prediction_df.index.max():
                X = prediction_df[target_prediction_index:target_prediction_index][FEATURES]

                if X.empty is not None:
                    prediction = model.predict(X)
                    prediction_df.loc[target_prediction_index, 'water_temperature'] = math.floor(prediction[0] * 100) / 100.0
                    self.features_water_temperature.prepare_lag_features(prediction_df)
                    self.features_water_temperature.prepare_rolling_features(prediction_df)
                    target_prediction_index += timedelta(hours=1)
                else:
                    logger.warning(f"No features available for prediction at index {target_prediction_index}. Skipping prediction.")
                    raise ValueError(f"No features available for prediction at index {target_prediction_index}. Skipping prediction.")

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

            insertion_data = prediction_df.copy()
            insertion_data = insertion_data[insertion_data['is_prediction'] == True]
            insertion_data.reset_index(inplace=True, drop=False)
            insertion_data.rename(columns={'created_at': 'target_time'}, inplace=True)

            # Insert to Supabase
            logger.info(f"Inserting water temperature predictions for aquarium {aquarium_id} into Supabase.")
            self.supabase.insert_prediction(
                aquarium_id=aquarium_id,
                parameter='water_temperature',
                model_version=model_metadata['training_info']['model_version'],
                data=insertion_data,
                exclude_columns=['do', 'ph','is_prediction']
            )

            # Remove unnecessary columns
            prediction_df.drop(
                columns=['confidence_lower', 'confidence_upper', 'std_error'],
                inplace=True, errors='ignore'
            )

            return prediction_df

        except Exception as e:
            logger.error(f"Error in predict_water_temperature: {e}")
            self.supabase.log_ml_activity(
                aquarium_id=aquarium_id,
                activity_type="predicting_water_temperature",
                status="error",
                error_message=f"Error predicting water temperature: {e}",
                metadata={"date_time_now": date_time_now.isoformat()}
            )
            raise

    def predict_ph(
            self,
            aquarium_id: str,
            date_time_now: datetime,
            prediction_df: pd.DataFrame,
            water_change_data: pd.DataFrame,
            feeding_data: pd.DataFrame
    ):
        """
        Predict the pH for the given aquarium at the specified date and time.
        """
        try:
            model = self.models[self._get_model_key(aquarium_id, 'ph')]
            model_metadata = self.metadata[self._get_model_key(aquarium_id, 'ph')]

            if not model or not model_metadata:
                raise ValueError(f"No model found for aquarium {aquarium_id} and parameter 'ph'.")

            # Prepare features
            FEATURES = model_metadata['training_info']['features']
            logger.info(f"Features for pH prediction: {FEATURES}")

            # Prepare features for prediction
            prediction_df, _ = self.features_ph.prepare_all_features(
                historical_data=prediction_df,
                water_change_data=water_change_data,
                feed_data= feeding_data
            )

            before_prediction_index: pd.DatetimeIndex = prediction_df.loc[prediction_df['ph'].notna()].index.max()
            target_prediction_index: pd.DatetimeIndex = before_prediction_index + timedelta(hours=1)

            while target_prediction_index <= prediction_df.index.max():
                X = prediction_df[target_prediction_index:target_prediction_index][FEATURES]

                if X.empty is not None:
                    prediction = model.predict(X)
                    prediction_df.loc[target_prediction_index, 'ph'] = math.floor(prediction[0] * 100) / 100.0
                    self.features_ph.prepare_lag_features(prediction_df)
                    self.features_ph.prepare_rolling_features(prediction_df)
                    target_prediction_index += timedelta(hours=1)
                else:
                    logger.warning(f"No features available for prediction at index {target_prediction_index}. Skipping prediction.")
                    raise ValueError(f"No features available for prediction at index {target_prediction_index}. Skipping prediction.")

            prediction_df.drop(columns=FEATURES, inplace=True)

            # After all predictions are made, add confidence intervals
            predicted_rows = prediction_df[prediction_df['is_prediction'] == True]
            
            if not predicted_rows.empty:
                predictions = predicted_rows['ph'].values
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

            insertion_data = prediction_df.copy()
            insertion_data = insertion_data[insertion_data['is_prediction'] == True]
            insertion_data.reset_index(inplace=True, drop=False)
            insertion_data.rename(columns={'created_at': 'target_time'}, inplace=True)

            # Insert to Supabase
            logger.info(f"Inserting water temperature predictions for aquarium {aquarium_id} into Supabase.")
            self.supabase.insert_prediction(
                aquarium_id=aquarium_id,
                parameter='ph',
                model_version=model_metadata['training_info']['model_version'],
                data=insertion_data,
                exclude_columns=['do', 'water_temperature','is_prediction']
            )

            # Remove unnecessary columns
            prediction_df.drop(
                columns=['confidence_lower', 'confidence_upper', 'std_error'],
                inplace=True, errors='ignore'
            )

            return prediction_df

        except Exception as e:
            logger.error(f"Error in predict_ph: {e}")
            self.supabase.log_ml_activity(
                aquarium_id=aquarium_id,
                activity_type="predicting_ph",
                status="error",
                error_message=f"Error predicting pH: {e}",
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

    def convert_from_5min_to_1hour(self, 
                                   df: pd.DataFrame, 
                                   target_cols: List[str] = ['water_temperature', 'ph', 'do'],
                                   index_name: str = 'created_at') -> pd.DataFrame:
        """
        Convert a DataFrame with 5-minute intervals to 1-hour intervals.
        """
        is_reset_index = False

        if not isinstance(df.index, pd.DatetimeIndex):
            df[index_name] = pd.to_datetime(df[index_name], format='ISO8601')
            df.set_index(index_name, inplace=True)
            is_reset_index = True

        # Resample to 1-hour intervals
        full_index = pd.date_range(start=df.index.min().replace(minute=0, second=0, microsecond=0), end=df.index.max().replace(minute=0, second=0, microsecond=0), freq=self.DATA_INTERVAL)
        df = df.reindex(full_index)
        df.index.name = index_name
        df.sort_index(inplace=True)

        # Interpolate missing values for target columns
        for col in target_cols:
            if col in df.columns:
                df[col] = df[col].interpolate(method='linear')
            else:
                logger.warning(f"Column '{col}' not found in DataFrame. Skipping interpolation for this column.")

        if is_reset_index:
            df.reset_index(inplace=True)
            
        return df
        
    def train_model(self,
                    aquarium_id: str,
                    parameter: Literal['water_temperature', 'ph', 'do'],
                    df: pd.DataFrame,
                    features: list[str],
                    days_back: int
                    ):
        """
        Train a model for the specified parameter of the aquarium.
        """
        try:
            start_time = datetime.now(timezone.utc)

            # Drop rows with NaN values in all features and target
            df = df.dropna(subset=features + [parameter])

            # Split data into training and testing sets
            train_size = int(len(df) * 0.8)
            train, test = df[:train_size], df[train_size:]

            FEATURES = features

            TARGET = parameter

            X_train = train[FEATURES]
            y_train = train[TARGET]

            X_test = test[FEATURES]
            y_test = test[TARGET]

            # Train the model
            model = xgb.XGBRegressor(
                n_estimators=1000,
                early_stopping_rounds=50,
                learning_rate=0.01,
                max_depth=3,
                random_state=42
            )
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_train, y_train), (X_test, y_test)],
                verbose=False
            )

            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            # Evaluate the model
            evaluation_metrics = {
                "aquarium_id": aquarium_id,
                "parameter": TARGET,
                "train_model_days_count": days_back,
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
                max_depth=3,
                random_state=42
            )
            final_model.fit(
                X_train_split,
                y_train_split,
                eval_set=[(X_train_split, y_train_split), (X_val_split, y_val_split)],
                verbose=False
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

            init_model_diff_rmse = evaluation_metrics['train_rmse'] - evaluation_metrics['test_rmse']
            final_model_diff_rmse = final_evaluation_metrics['train_rmse'] - final_evaluation_metrics['test_rmse']

            # Compare evaluation metrics
            if (final_model_diff_rmse < init_model_diff_rmse):
                logger.info("Final model has better performance than initial model. Using final model.")
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

            try:
                # Insert the training data into Supabase
                self.supabase.log_ml_activity(
                aquarium_id=aquarium_id,
                activity_type=f"model_training_{parameter}",
                status="success",
                processing_time_seconds=int((datetime.now(timezone.utc) - start_time).total_seconds()),
                metadata={
                    "parameter": parameter,
                    "dataset_days_size": days_back,
                    "evaluation_metrics": evaluation_metrics,
                    "feature_importance": feature_importance.to_dict(),
                    "training_info": {
                        "features": FEATURES,
                        "model_version": f"{TARGET}_{datetime.now(timezone.utc).isoformat()}"
                    }
                })
            except Exception as e:
                logger.error(f"Error logging ML activity for aquarium {aquarium_id}: {e}")
                raise
        except Exception as e:
            logger.error(f"Error during model training for aquarium {aquarium_id} and parameter {parameter}: {e}")
            self.supabase.log_ml_activity(
                aquarium_id=aquarium_id,
                activity_type=f"model_training_{parameter}",
                status="error",
                error_message=str(e),
                metadata={"parameter": parameter}
            )
            raise

    def find_missing_data(
            self,
            aquarium_id: str,
            date_time_start: datetime,
            date_time_end: datetime
    ):
        """
        Find missing data for the specified aquarium within the date range.
        """
        try:
            # Fetch aquarium data
            aquarium_data = self.supabase.get_aquarium_data(aquarium_id, ['name', 'timezone', 'user_id'])

            # Fetch historical data
            historical_data = self.supabase.get_historical_data(
                aquarium_id=aquarium_id,
                start_date=date_time_start,
                end_date=date_time_end
            )

            if historical_data.empty:
                logger.warning(f"No historical data found for aquarium {aquarium_id} in the specified date range.")
                return
            
            historical_data['created_at'] = pd.to_datetime(historical_data['created_at'], format='ISO8601')
            historical_data.set_index('created_at', inplace=True)
            
            # Calculate time differences between consecutive measurements
            time_diffs = historical_data.index.to_series().diff()
            
            # Convert to minutes
            time_diffs_minutes = time_diffs.dt.total_seconds() / 60
            
            # Find gaps larger than threshold
            gaps = time_diffs_minutes > 15
            
            # Create gap summary
            gap_info: List[Dict[str, str | float]] = []
            for i, is_gap in enumerate(gaps):
                if is_gap:
                    gap_start = historical_data.index[i-1]
                    gap_end = historical_data.index[i]
                    gap_duration = time_diffs_minutes.iloc[i]
                    
                    gap_info.append({
                        'aquarium_id': aquarium_id,
                        'gap_start': gap_start.isoformat(),
                        'gap_end': gap_end.isoformat(),
                        'duration_minutes': float(gap_duration)
                    })

            if gap_info:
                # Insert gap information into Supabase
                self.supabase.insert_missing_data(
                    data=gap_info,
                    user_id=aquarium_data['user_id'], # type: ignore
                )
                logger.info(f"Missing data found for aquarium {aquarium_id}. Gaps: {gap_info}")
            else:
                logger.info(f"No significant gaps found for aquarium {aquarium_id} in the specified date range.")
        
        except Exception as e:
            logger.error(f"Error fetching historical data for aquarium {aquarium_id}: {e}")
            raise

    def report_prediction_validation(
            self,
            aquarium_id: str,
            target_time: datetime,
            parameter: Literal['water_temperature', 'ph', 'do'],
            predicted_value: float,
            actual_value: float,
            prediction_error: float
    ):
        """
           Send a report for the prediction validation.
        """
        try:
            # Prepare the data for validation
            severity = 'low'

            if prediction_error > 2.0:
                severity = 'high'
            elif prediction_error > 1.0:
                severity = 'medium'
            
            aquarium_data = self.supabase.get_aquarium_data(aquarium_id, ['name', 'timezone', 'user_id'])

            if aquarium_data is None:
                raise ValueError(f"Aquarium {aquarium_id} not found or data missing")

            aquarium_name = aquarium_data.get('name', 'Unknown Aquarium')

            # Set timezone to Istanbul
            aquarium_timezone = pytz.timezone(aquarium_data.get('timezone', 'Europe/Istanbul'))
            
            # convert target_time to human-readable format
            target_time = target_time.astimezone(aquarium_timezone)
            hour = target_time.strftime('%H')
            am_pm = 'AM' if int(hour) < 12 else 'PM'

            parameter_modified = parameter.replace('_',' ').title()

            message = f"Prediction for {aquarium_name} at {hour} {am_pm} for {parameter_modified} was {predicted_value}, but actual value was {actual_value}. Prediction error: {prediction_error:.2f}."

            # Insert the validation record into Supabase
            self.supabase.send_alert(
                user_id=aquarium_data['user_id'],
                severity=severity,
                title='Prediction Validation Alert',
                message=message,
                alert_timestamp=datetime.now(timezone.utc),
            )

        except Exception as e:
            logger.error(f"Error reporting prediction validation for aquarium {aquarium_id} and parameter {parameter}: {e}")
        
    def detect_anomalies(self, raw_data: pd.DataFrame, parameter: str, contamination: float = 0.01):
        """
        Detect anomalies in the data using Isolation Forest.
        
        :param data: DataFrame containing the features for anomaly detection.
        :param feature_columns: List of feature columns to use for anomaly detection.
        :param contamination: Proportion of outliers in the data.
        :return: DataFrame with detected anomalies.
        """
        data = raw_data.copy()

        if not isinstance(data.index, pd.DatetimeIndex):
            data['created_at'] = pd.to_datetime(data['created_at'], format='ISO8601')
            data.set_index('created_at', inplace=True)
            data = data.sort_index()

        feature_columns = [parameter,f"{parameter}_change", f"{parameter}_deviation"]

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(data[feature_columns])

        isolation_forest = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1, n_estimators=100)

        anomaly_labels = isolation_forest.fit_predict(features_scaled)
        anomaly_scores = isolation_forest.decision_function(features_scaled)

        # Convert anomaly labels to 0 for normal and 1 for anomaly
        anomaly_labels = np.where(anomaly_labels == -1, 1, 0)

        percentile_ranks = stats.rankdata(anomaly_scores) / len(anomaly_scores) * 100
        anomaly_percentages = 100 - percentile_ranks

        data['anomaly_score'] = anomaly_percentages
        data['anomaly_label'] = anomaly_labels

        # Filter anomalies
        anomalies = data[data['anomaly_label'] == 1]

        # check index type
        if not isinstance(anomalies.index, pd.DatetimeIndex):
            anomalies['created_at'] = pd.to_datetime(anomalies['created_at'], format='ISO8601')
            anomalies.set_index('created_at', inplace=True)
            
        # Prepare the final DataFrame
        anomalies_df = pd.DataFrame(columns=['datetime', 'parameter', 'anomaly_score', 'value', 'reason'])

        mean = data[parameter].mean()
        std = data[parameter].std()
        change_std = data[f"{parameter}_change"].std()

        for index, row in anomalies.iterrows():
            reason = []

            # Check if parameter is an outlier
            if abs(row[parameter] - mean) > 2 * std:
                reason.append('outlier')

            # Check if change is large
            if abs(row[f"{parameter}_change"]) > 2 * change_std:
                reason.append('Large change')

            # Check if deviation from rolling mean is high
            if row[f"{parameter}_deviation"] > 2 * data[f"{parameter}_deviation"].std():
                reason.append('High deviation')

            if not reason:
                reason.append('Pattern anomaly')

            anomalies_df.loc[len(anomalies_df)] = [
                index,
                parameter,
                row['anomaly_score'],
                row[parameter],
                ', '.join(reason),
                'high' if row['anomaly_score'] > 95 else 'medium' if row['anomaly_score'] > 80 else 'low'
            ]

        return anomalies_df
