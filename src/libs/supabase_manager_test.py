from supabase import create_client, Client
from src.config.settings import settings
from datetime import datetime, timedelta
import pandas as pd

class SupabaseManager:
    def __init__(self):
        self.client: Client = create_client(
            settings.SUPABASE_URL,
            settings.SUPABASE_SERVICE_KEY
        )

    def get_historical_measurements(self, aquarium_id: str) -> pd.DataFrame | None:
        try:
            # Get 7 days ago
            start_date = (datetime.now() - timedelta(days=7)).replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = datetime.now()

            # Fetch historical measurements from the database
            response = self.client.table("measurements").select("created_at, water_temperature, ph, do").eq("env_id", aquarium_id).order("created_at", desc=True).gte("created_at", start_date.isoformat()).lte("created_at", end_date.isoformat()).execute()

            if response.data:
                df = pd.DataFrame(response.data)
                df['created_at'] = pd.to_datetime(df['created_at'], format='ISO8601')
                df.set_index('created_at', inplace=True)
                df.sort_index(inplace=True)
                return df
            
            print("No historical measurements found for the given aquarium ID.")
            return None

        except Exception as e:
            print(f"Error while fetching historical measurements: {e}")
            return None
        
    def get_feeding_times(self, aquarium_id: str) -> pd.DataFrame | None:
        try:
            # Get 7 days ago
            start_date = (datetime.now() - timedelta(days=7)).replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = datetime.now()

            print(f"Fetching feeding times for aquarium ID: {aquarium_id} from {start_date} to {end_date}")

            # Fetch feeding times from the database
            response = self.client.table("feeding_times").select("fed_at").eq("aquarium_id", aquarium_id).order("fed_at", desc=True).gte("fed_at", start_date.isoformat()).lte("fed_at", end_date.isoformat()).execute()

            if response.data:
                df = pd.DataFrame(response.data)
                return df
            
            print("No feeding times found for the given aquarium ID.")
            return None

        except Exception as e:
            print(f"Error while fetching feeding times: {e}")
            return None
        
    def get_water_changing_times(self, aquarium_id: str) -> pd.DataFrame | None:
        try:
            # Get 7 days ago
            start_date = (datetime.now() - timedelta(days=7)).replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = datetime.now()

            print(f"Fetching water changing times for aquarium ID: {aquarium_id} from {start_date} to {end_date}")

            # Fetch water changing times from the database
            response = self.client.table("water_changing_times").select("changed_at, percentage_changed, water_temperature_added").eq("aquarium_id", aquarium_id).order("changed_at", desc=True).gte("changed_at", start_date.isoformat()).lte("changed_at", end_date.isoformat()).execute()

            if response.data:
                df = pd.DataFrame(response.data)
                return df
            
            print("No water changing times found for the given aquarium ID.")
            return None

        except Exception as e:
            print(f"Error while fetching water changing times: {e}")
            return None