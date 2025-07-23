import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
from datetime import datetime
from typing import Dict
from typing import Optional, Dict
import math
import logging
import os
from src.config.settings import settings

logger = logging.getLogger(__name__)

class WeatherForecast:
    def __init__(self):
        self.url = "https://api.open-meteo.com/v1/forecast"
        # Use the designated cache directory instead of current directory
        cache_path = os.path.join(settings.DATA_CACHE_PATH, 'weather_cache')
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        self.cache_session = requests_cache.CachedSession(cache_path, expire_after = 3600)
        self.retry_session = retry(self.cache_session, retries = 5, backoff_factor = 0.2)
        self.openmeteo = openmeteo_requests.Client(session = self.retry_session) # type: ignore

        # Default parameters for the Open Meteo API
        self.params = {
            "daily": ["sunset", "sunrise"],
            "current": "temperature_2m",
            "minutely_15": "temperature_2m",
            "timezone": "GMT",
        }

    def get_weather_data(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str
    ) -> Optional[Dict[str, pd.DataFrame]]:
        """
        Fetches weather data for the given latitude and longitude within the specified date range.
        Returns a dictionary with keys 'minutely_15' (DataFrame) and 'daily' (dict with 'sunset' and 'sunrise' Series).
        """
        try:
            responses = self.openmeteo.weather_api(self.url, params={
                **self.params,
                "latitude": latitude,
                "longitude": longitude,
                "start_date": start_date,
                "end_date": end_date
            })

            response = responses[0]

            minutely_15 = response.Minutely15()
            minutely_15_temperature_2m = minutely_15.Variables(0).ValuesAsNumpy() # type: ignore

            minutely_15_data = {"date": pd.date_range(
                start=pd.to_datetime(minutely_15.Time(), unit="s", utc=True), # type: ignore
                end=pd.to_datetime(minutely_15.TimeEnd(), unit="s", utc=True), # type: ignore
                freq=pd.Timedelta(seconds=minutely_15.Interval()), # type: ignore
                inclusive="left"
            )}

            minutely_15_data["outside_temperature"] = minutely_15_temperature_2m # type: ignore
            minutely_15_dataframe = pd.DataFrame(data=minutely_15_data)
            minutely_15_dataframe['date'] = pd.to_datetime(minutely_15_dataframe['date'], utc=True)
            minutely_15_dataframe.set_index('date', inplace=True)

            # Rounded to 2 decimal places
            minutely_15_dataframe["outside_temperature"] = minutely_15_dataframe["outside_temperature"].apply(lambda x: math.floor(x * 100) / 100)

            daily = response.Daily()
            daily_sunset = daily.Variables(0).ValuesInt64AsNumpy() # type: ignore
            daily_sunrise = daily.Variables(1).ValuesInt64AsNumpy() # type: ignore

            daily_data = {"date": pd.date_range(
                start=pd.to_datetime(daily.Time(), unit="s", utc=True), # type: ignore
                end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True), # type: ignore
                freq=pd.Timedelta(seconds=daily.Interval()), # type: ignore
                inclusive="left"
            )}

            daily_data["sunset"] = daily_sunset # type: ignore
            daily_data["sunrise"] = daily_sunrise # type: ignore

            daily_data["sunset"] = pd.to_datetime(daily_data["sunset"], unit='s', utc=True)
            daily_data["sunrise"] = pd.to_datetime(daily_data["sunrise"], unit='s', utc=True)

            daily_dataframe = pd.DataFrame(data=daily_data)
            daily_dataframe['date'] = pd.to_datetime(daily_dataframe['date'], format='s', utc=True)
            daily_dataframe.sort_values('date', inplace=True)

            return {
                "forecast": minutely_15_dataframe,
                "sunset_sunrise": daily_dataframe
            }

        except Exception as e:
            logger.error(f"Error fetching weather data: {e}")
            return None
        
    def _process_5_minutely_data(self, response) -> pd.DataFrame:
        """
        Processes the 5-minute interval data from the weather API response.
        Returns a DataFrame with the date and temperature.
        """
        try:
            minutely_15 = response.Minutely15()
            minutely_15_temperature_2m = minutely_15.Variables(0).ValuesAsNumpy() # type: ignore

            minutely_15_data = {"date": pd.date_range(
                start=pd.to_datetime(minutely_15.Time(), unit="s", utc=True), # type: ignore
                end=pd.to_datetime(minutely_15.TimeEnd(), unit="s", utc=True), # type: ignore
                freq=pd.Timedelta(seconds=minutely_15.Interval()), # type: ignore
                inclusive="left"
            )}

            minutely_15_data["temperature_2m"] = minutely_15_temperature_2m # type: ignore
            minutely_15_dataframe = pd.DataFrame(data=minutely_15_data)
            minutely_15_dataframe['date'] = pd.to_datetime(minutely_15_dataframe['date'], utc=True)

            # Change 15 minute intervals to 5 minute intervals
            minutely_5_df = minutely_15_dataframe.copy()
            index = pd.date_range(start=minutely_5_df.index.min(), end=minutely_5_df.index.max(), freq='5min')
            minutely_5_df = minutely_5_df.reindex(index)
            minutely_5_df.rename(columns={"temperature_2m": "outside_temperature"}, inplace=True)
            minutely_5_df['outside_temperature'] = minutely_5_df['outside_temperature'].interpolate(method='time').ffill().bfill()
            minutely_5_df["outside_temperature"] = minutely_5_df["outside_temperature"].round(2)
            minutely_5_df["outside_temperature"] = minutely_5_df["outside_temperature"].apply(lambda x: round(x, 2))

            return minutely_5_df
        except Exception as e:
            print(f"Error processing 5-minute data: {e}")
            return pd.DataFrame()

