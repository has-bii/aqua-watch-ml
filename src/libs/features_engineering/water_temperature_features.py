from src.libs.supabase_manager import SupabaseManager
from src.libs.forecast import WeatherForecast
import pandas as pd
from typing import Dict, Optional

class FeatureEngineeringWaterTemperature:
    # Constants
    WATER_CHANGE_WINDOW_MINUTES = 15
    DEFAULT_MINUTES_AFTER_WATER_CHANGE = 1000
    LAG_PERIODS = [1, 2, 3]
    
    def __init__(self):
        self.weather_forecast = WeatherForecast()
        self.supabase = SupabaseManager()

    def prepare_all_features(self,
                             aquarium_id: str,
                             historical_data: pd.DataFrame, 
                             water_change_data: pd.DataFrame,
                             forecast: Optional[Dict[str, pd.DataFrame]] = None
                             ) -> pd.DataFrame:
        try:
            # Input validation
            self._validate_input_data(historical_data, water_change_data)
            
            df = historical_data.copy()

            if forecast is None:
                # Get aquarium location data
                aquarium_geo = self.supabase.get_aquarium_geo(aquarium_id)
                if aquarium_geo is None:
                    raise ValueError(f"Aquarium {aquarium_id} not found or geo data missing")
            
                # Get weather forecast data
                start_date = df['created_at'].min().strftime('%Y-%m-%d')
                end_date = df['created_at'].max().strftime('%Y-%m-%d')

                forecast = self._get_weather_forecast(aquarium_geo, start_date, end_date)
                forecast_df: pd.DataFrame = forecast["forecast"] # type: ignore
            else:
                forecast_df = forecast.get("forecast", pd.DataFrame())

            # Apply feature engineering in sequence
            df = self.prepare_features(df, dropNan=True)
            df = self.prepare_feature_with_weather(df, forecast_df, forecast.get("sunset_sunrise", pd.DataFrame()))
            df = self.prepare_features_with_water_change(df, water_change_data)

            return df
        except Exception as e:
            raise ValueError(f"Error preparing features: {e}")

    def _validate_input_data(self, historical_data: pd.DataFrame, water_change_data: pd.DataFrame) -> None:
        """Validate input DataFrames have required columns."""
        required_historical_cols = ['created_at', 'water_temperature']
        required_water_change_cols = ['changed_at', 'water_temperature_added']

        if isinstance(historical_data.index, pd.DatetimeIndex):
            required_historical_cols.remove('created_at')  # If index is datetime, no need for this column
        if isinstance(water_change_data.index, pd.DatetimeIndex):
            required_water_change_cols.remove('changed_at')
        
        if historical_data.empty:
            raise ValueError("Historical data cannot be empty")
        
        missing_hist_cols = [col for col in required_historical_cols if col not in historical_data.columns]
        if missing_hist_cols:
            raise ValueError(f"Historical data missing required columns: {missing_hist_cols}")
        
        if not water_change_data.empty:
            missing_wc_cols = [col for col in required_water_change_cols if col not in water_change_data.columns]
            if missing_wc_cols:
                raise ValueError(f"Water change data missing required columns: {missing_wc_cols}")

    def _get_weather_forecast(self, 
                              aquarium_geo: dict, 
                              start_date: str,
                              end_date: str
                              ) -> dict:
        """Get weather forecast data with proper validation."""

        if aquarium_geo['latitude'] is None or aquarium_geo['longitude'] is None:
            raise ValueError("Aquarium geo data is incomplete, latitude or longitude missing")

        forecast = self.weather_forecast.get_weather_data(
            latitude=aquarium_geo['latitude'],
            longitude=aquarium_geo['longitude'],
            start_date=start_date,
            end_date=end_date
        )

        if forecast is None:
            raise ValueError("Weather forecast data not available")
        
        return forecast

    def prepare_feature_with_weather(self, 
                                     df: pd.DataFrame, 
                                     weather_df: pd.DataFrame, 
                                     sunset_sunrise_df: pd.DataFrame,
                                     index_name: Optional[str] = 'created_at') -> pd.DataFrame:
        """Merge weather data with historical data and handle missing values."""
        # Ensure both DataFrames have datetime index for proper merging
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.set_index(index_name)

        # Merge the weather data with the main DataFrame
        df = df.merge(weather_df, how='left', left_index=True, right_index=True)

        # Fill NaN values in the weather data with interpolation and forward/backward fill
        df['outside_temperature'] = df['outside_temperature'].interpolate(method='time').ffill().bfill()

        # Add is_day feature based on sunset and sunrise times
        df['is_day'] = pd.NA  # Initialize with NaN

        ss_df = sunset_sunrise_df.copy()
        ss_df['date'] = pd.to_datetime(ss_df['date'], unit='s', utc=True)
        ss_df.set_index('date', inplace=True)

        # sunset_sunrise_df.columns = ['sunset', 'sunrise']
        # all types are already timestamp
        df['is_day'] = False  
        for index, row in ss_df.iterrows():
            date: int = index.dayofyear # type: ignore
            df.loc[((df.index.dayofyear == date) & (df.index >= row['sunrise']) & (df.index < row['sunset'])), 'is_day'] = True # type: ignore

        # Convert is_day to int
        df['is_day'] = df['is_day'].astype(int)

        return df

    def prepare_features(self, 
                         dff: pd.DataFrame, 
                         dropNan: Optional[bool] = False,
                         fillna: Optional[bool] = True,
                         index_name: str = 'created_at') -> pd.DataFrame:
        """Create time-based and lag features."""
        df = dff.copy()

        if isinstance(df.index, pd.DatetimeIndex):
            df.reset_index(inplace=True, names=index_name)
            df[index_name] = pd.to_datetime(df[index_name], format='ISO8601')
            df.sort_values(by=index_name, inplace=True)

        # Time-based features using constants
        df['hour_of_day'] = df[index_name].dt.hour
        df['minutes_of_day'] = df[index_name].dt.hour * 60 + df[index_name].dt.minute

        # Lag features using constant
        for lag in self.LAG_PERIODS:
            df[f'lag_{lag}'] = df['water_temperature'].shift(lag)

        if dropNan:
            # Drop rows with NaN values after creating lag features
            df.dropna(inplace=True)

        if fillna:
            # Fill NaN values (backward fill first, then forward fill)
            df = df.bfill().ffill()

        return df
    
    def prepare_lag_features(self,
                        df: pd.DataFrame
                        ):
        """Create lag features"""

        # Lag features using constant
        for lag in self.LAG_PERIODS:
            df[f'lag_{lag}'] = df['water_temperature'].shift(lag)

        return df

    def prepare_features_with_water_change(self, 
                                           df: pd.DataFrame, 
                                           water_change_df: pd.DataFrame,
                                           index_name: Optional[str] = 'created_at'
                                           ) -> pd.DataFrame:
        """Create features related to water changes with support for multiple events."""
        try:
            if water_change_df.empty:
                # If no water changes, initialize with default values
                df['minutes_after_water_change'] = self.DEFAULT_MINUTES_AFTER_WATER_CHANGE
                df['diff_water_temp_after_change'] = 0
                return df
        
            df_working = df.copy()
            water_change_df = water_change_df.copy()

            # Ensure we have created_at as a column for processing
            if isinstance(df_working.index, pd.DatetimeIndex):
                df_working.reset_index(inplace=True)
            
            # Ensure datetime columns are properly formatted
            df_working[index_name] = pd.to_datetime(df_working[index_name], format='ISO8601')
            water_change_df = water_change_df.reset_index() if isinstance(water_change_df.index, pd.DatetimeIndex) else water_change_df.copy()
            water_change_df['changed_at'] = pd.to_datetime(water_change_df['changed_at'], format='ISO8601')

            # Initialize features with default values
            df_working['minutes_after_water_change'] = self.DEFAULT_MINUTES_AFTER_WATER_CHANGE
            df_working['diff_water_temp_after_change'] = 0.0

            # Process each water change event
            for _, water_change in water_change_df.iterrows():
                water_change_time = water_change['changed_at']
                water_temp_added = water_change['water_temperature_added']
                
                # Calculate time differences for all records (vectorized)
                time_diffs = (df_working[index_name] - water_change_time).dt.total_seconds() / 60
                
                # Find records within the time window after this water change
                within_window = (time_diffs >= 0) & (time_diffs <= self.WATER_CHANGE_WINDOW_MINUTES)
                
                # Update features for records within the window
                # Only update if this water change is more recent than previously recorded ones
                closer_to_change = time_diffs < df_working['minutes_after_water_change']
                update_mask = within_window & closer_to_change
                
                if update_mask.any():
                    df_working.loc[update_mask, 'minutes_after_water_change'] = time_diffs[update_mask].astype(int)
                    df_working.loc[update_mask, 'diff_water_temp_after_change'] = (
                        df_working.loc[update_mask, 'water_temperature'] - water_temp_added
                    )

            # Restore the original index structure
            if isinstance(df.index, pd.DatetimeIndex):
                df_working.set_index(index_name, inplace=True)

            return df_working

        except Exception as e:
            raise ValueError(f"Error preparing features with water change data: {e}")
