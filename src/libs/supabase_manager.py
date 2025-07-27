from supabase import create_client, Client
from src.config.settings import settings
from datetime import datetime, timezone
import pandas as pd
import logging
from typing import Dict, Optional, Literal

logger = logging.getLogger(__name__)

class SupabaseManager:
    def __init__(self):
        self.supabase : Client = create_client(
            settings.SUPABASE_URL,
            settings.SUPABASE_SERVICE_KEY
        )

    def get_active_aquariums(self):
        """Fetch active aquariums from Supabase"""
        try:
            response = self.supabase.table("aquarium").select("*").eq("is_online", True).execute()
            return response.data if response.data else []
        except Exception as e:
            logger.error(f"Error fetching active aquariums: {e}")
            return []

    def get_historical_data(self, aquarium_id: str, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None, target_time: Optional[datetime] = None) -> pd.DataFrame:
        try:
            query = self.supabase.table("measurements").select("created_at, water_temperature, ph, do").eq("env_id", aquarium_id).order(column="created_at", desc=False)

            if start_date:
                query = query.gte("created_at", start_date.isoformat())
            if end_date:
                query = query.lte("created_at", end_date.isoformat())
            if target_time:
                query = query.eq("created_at", target_time.isoformat())

            response = query.execute()

            temp_data = pd.DataFrame(response.data)

            return temp_data

        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            raise ValueError(f"Error fetching historical data: {e}")

    def get_feeding_data(self,
                         aquarium_id: str,
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None) -> pd.DataFrame:
        try:
            query = self.supabase.table("feeding_times").select("fed_at").eq("aquarium_id", aquarium_id).order(column="fed_at", desc=False)

            if start_date:
                query = query.gte("fed_at", start_date.isoformat())
            if end_date:
                query = query.lte("fed_at", end_date.isoformat())

            response = query.execute()

            temp_data = pd.DataFrame(response.data)

            return temp_data
        except Exception as e:
            print(f"Error fetching feeding data: {e}")
            raise ValueError(f"Error fetching feeding data: {e}")

    def get_water_changing_data(self, aquarium_id: str, start_date: datetime | None, end_date: datetime | None) -> pd.DataFrame:
        try:
            query = self.supabase.table("water_changing_times").select("changed_at, percentage_changed, water_temperature_added").eq("aquarium_id", aquarium_id).order(column="changed_at", desc=False)

            if start_date:
                query = query.gte("changed_at", start_date.isoformat())
            if end_date:
                query = query.lte("changed_at", end_date.isoformat())

            response = query.execute()

            temp_data = pd.DataFrame(response.data)

            return temp_data

        except Exception as e:
            logger.error(f"Error fetching water changing data: {e}")
            raise ValueError(f"Error fetching water changing data: {e}")
        
    def get_aquarium_geo(self, aquarium_id: str) -> Dict[str, float] | None:
        try:
            response = self.supabase.table("aquarium_settings").select("latitude, longitude").eq("aquarium_id", aquarium_id).single().execute()

            # Check if latitude or longitude is None
            if response.data and 'latitude' in response.data and 'longitude' in response.data:
                latitude = response.data['latitude']
                longitude = response.data['longitude']
                
                if latitude is not None and longitude is not None:
                    return {
                        'latitude': latitude,
                        'longitude': longitude
                    }
                
            return None
        except Exception as e:
            logger.error(f"Error fetching aquarium geo data: {e}")
            return None
        
    def get_aquarium_model_settings(self, aquarium_id: str) -> Dict[str, int] | None:
        try:
            response = self.supabase.table("aquarium_settings").select("train_temp_model_days, train_ph_model_days, prediction_parameters").eq("aquarium_id", aquarium_id).single().execute()

            return response.data if response.data else None
        except Exception as e:
            logger.error(f"Error fetching aquarium model settings: {e}")
            return None

    def log_ml_activity(self, aquarium_id: Optional[str] = None, 
                       activity_type: str = "", 
                       status: str = "", 
                       error_message: str = "", 
                       processing_time_seconds: Optional[float] = None,
                       metadata: Optional[Dict] = None) -> None:
        """
        Log ML activity to Supabase
        
        Args:
            aquarium_id: Optional UUID of the aquarium
            activity_type: Type of activity (training, prediction, evaluation)
            status: Status of the activity (success, error, warning)
            error_message: Error message if status is error
            processing_time_seconds: Time taken for the activity
            metadata: Additional metadata as JSON
        """
        try:
            data = {
                'activity_type': activity_type,
                'status': status,
                'error_message': error_message,
                'created_at': datetime.now(timezone.utc).isoformat()
            }
            
            # Add optional fields
            if aquarium_id is not None:
                data['aquarium_id'] = aquarium_id
            if processing_time_seconds is not None:
                data['processing_time_seconds'] = processing_time_seconds # type: ignore
            if metadata is not None:
                data['metadata'] = metadata # type: ignore

            self.supabase.table('ml_logs').insert(data).execute()
            logger.info(f"Logged ML activity: {activity_type} - {status}")
                
        except Exception as e:
            logger.error(f"Error logging ML activity: {e}")

    def insert_prediction(
            self,
            aquarium_id: str,
            parameter: Literal['water_temperature', 'ph'],
            exclude_columns: list[Literal['is_prediction', 'ph', 'do', 'water_temperature']],
            data: pd.DataFrame,
            model_version: str
        ):
        """
        Insert prediction data into the database
        Args:
            aquarium_id: UUID of the aquarium
            parameter: Target parameter (e.g., water_temperature)
            data: DataFrame containing prediction data
        """
        try:
            # Ensure the DataFrame has the required columns
            required_columns = [parameter, 'confidence_lower', 'confidence_upper', 'target_time', 'std_error']
            if not all(col in data.columns for col in required_columns):
                raise ValueError(f"DataFrame must contain the following columns: {required_columns}, but got {data.columns.tolist()}")
            
            # Get min date  from the DataFrame
            min_date = data['target_time'].min()

            # Remove all predictions that are later than the minimum date
            self.supabase.table("predictions").delete().eq("aquarium_id", aquarium_id).eq("target_parameter", parameter).gte("target_time", min_date.isoformat()).execute()

            # Prepare the data for insertion
            data['aquarium_id'] = aquarium_id
            data['target_parameter'] = parameter
            data.rename(columns={parameter: 'predicted_value'}, inplace=True)
            data.drop(columns=exclude_columns, inplace=True, errors='ignore')  # Remove 'is_prediction' if it exists

            # Convert date to ISO format
            data['target_time'] = data['target_time'].apply(lambda x: x.isoformat() if isinstance(x, datetime) else x)

            data['model_version'] = model_version

            # Insert the data into the predictions table
            self.supabase.table("predictions").insert(data.to_dict(orient='records')).execute()
        except Exception as e:
            logger.error(f"Error inserting prediction: {e}")
            raise ValueError(f"Error inserting prediction: {e}") 
        
    def get_prediction(self,
                       aquarium_id: str,
                       parameter: Literal['water_temperature', 'ph', 'do'],
                       target_time: datetime) -> pd.DataFrame:
        """
        Get a specific prediction for an aquarium and parameter at a given time
        Args:
            aquarium_id: UUID of the aquarium
            parameter: Target parameter (e.g., water_temperature)
            target_time: Time of the prediction
        Returns:
            DataFrame containing the prediction data, or an empty DataFrame if no prediction is found
        """

        try:
            # Fetch that actual_value is null
            response = self.supabase.table("predictions").select("*").eq("aquarium_id", aquarium_id).eq("target_parameter", parameter).lte("target_time", target_time.isoformat()).order("target_time", desc=True).execute()

            df = pd.DataFrame(response.data)

            if df.empty:
                raise ValueError("No prediction found")
            
            df = df[df['actual_value'].isnull()]

            if df.empty:
                raise ValueError("No prediction found with actual_value as null")
            
            return df
        except Exception as e:
            raise ValueError(f"Failed to fetch prediction: {e}")
        
    def validate_prediction(self,
                            data: Dict) -> bool:
        """
        Validate a prediction against an actual value
        Args:
            data: Dictionary containing the prediction data to validate
                Expected keys: 'id', 'aquarium_id', 'target_parameter', 'target_time', 'predicted_value', 'actual_value', 'confidence_lower', 'confidence_upper', 'std_error'
        Returns:
            True if the prediction is valid, False otherwise
        """
        try:
            self.supabase.table("predictions").upsert(
                data
            ).execute()

            return True
        except Exception as e:
            return False
        
    