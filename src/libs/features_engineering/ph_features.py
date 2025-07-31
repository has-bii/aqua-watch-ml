import pandas as pd
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)

class FeatureEngineeringPH:
    LAG_PERIODS = [1, 2, 3]
    
    WATER_CHANGE_WINDOW_MINUTES = 120
    AFTER_FEED_WINDOW_MINUTES = 120
    
    DEFAULT_MINUTES_AFTER_WATER_CHANGE = 9999
    DEFAULT_MINUTES_AFTER_FEED = 9999
    DEFAULT_PERCENTAGE_CHANGED = 0.0

    ROLLING_WINDOW_SIZE = [2, 3, 4]

    def prepare_all_features(self,
                             historical_data: pd.DataFrame,
                             water_change_data: pd.DataFrame,
                             feed_data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare all features for pH prediction.
        """
        try:
            features = ['water_temperature']

            # Prepare temporal features (without duplicate lag features)
            historical_data = self.prepare_temporal_features(historical_data, features)

            # Prepare lag features separately 
            historical_data = self.prepare_lag_features(historical_data, features)

            # Prepare rolling features
            historical_data = self.prepare_rolling_features(historical_data, features)

            # Prepare features related to water changes
            historical_data = self.prepare_features_with_water_change(
                historical_data, water_change_data, features, index_name='created_at',
            )

            # Prepare feed features with enhanced debugging
            historical_data = self.prepare_feed_features(historical_data, feed_data, features)

            return historical_data, features
        except Exception as e:
            logger.error(f"Error preparing all features: {e}")
            raise e

    def prepare_temporal_features(self, 
                                  df: pd.DataFrame,
                                  features: Optional[list[str]] = None) -> pd.DataFrame:
        """
        Prepare temporal features from the DataFrame (without lag features).
        """
        try:
            is_reset_index = False

            if isinstance(df.index, pd.DatetimeIndex):
                df.reset_index(inplace=True)
                df['created_at'] = pd.to_datetime(df['created_at'], format='ISO8601')
                is_reset_index = True

            # Time-based features only
            df['hour_of_day'] = df['created_at'].dt.hour

            if is_reset_index:
                df.set_index('created_at', inplace=True)

            if features:
                features.append('hour_of_day')

            return df
        except Exception as e:
            raise ValueError(f"Error preparing temporal features: {e}")
        
    def prepare_lag_features(self, 
                                df: pd.DataFrame, 
                                features: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Prepare lag features from the DataFrame.
        """
        try:
            for lag in self.LAG_PERIODS:
                df[f'lag_{lag}'] = df['ph'].shift(lag)

            if features is not None:
                features.extend([f'lag_{lag}' for lag in self.LAG_PERIODS])

            return df
        except Exception as e:
            raise ValueError(f"Error preparing lag features: {e}")
      
    def prepare_rolling_features(self,
                                 df: pd.DataFrame,
                                 features: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Prepare rolling features from the DataFrame.
        """
        try:
            for rolling in self.ROLLING_WINDOW_SIZE:
                df[f'rolling_mean_{rolling}'] = df['ph'].rolling(window=rolling).mean().shift()

            if features is not None:
                features.extend([f'rolling_mean_{rolling}' for rolling in self.ROLLING_WINDOW_SIZE])

            return df
        except Exception as e:
            raise ValueError(f"Error preparing rolling features: {e}")

    def prepare_features_with_water_change(self, 
                                       df: pd.DataFrame, 
                                       water_change_df: pd.DataFrame,
                                       features: Optional[List[str]] = None,
                                       index_name: Optional[str] = 'created_at') -> pd.DataFrame:
        """
        Create features related to water changes with support for multiple events.
        
        Args:
            df: Historical data DataFrame with pH measurements
            water_change_df: DataFrame with water change events
            index_name: Name of the datetime column/index
            
        Returns:
            DataFrame with added water change features:
            - minutes_after_water_change: Time elapsed since last water change
            - diff_water_temp_after_change: Temperature difference after water change
            - percentage_water_changed: Percentage of water changed
        """
        try:
            if water_change_df.empty:
                # Initialize with default values when no water changes exist
                df['minutes_after_water_change'] = self.DEFAULT_MINUTES_AFTER_WATER_CHANGE
                # df['diff_water_temp_after_change'] = 0.0
                df['percentage_water_changed'] = self.DEFAULT_PERCENTAGE_CHANGED
                return df
        
            # Work with copies to avoid modifying original data
            df_working = df.copy()
            water_change_df_working = water_change_df.copy()

            # Handle index management for historical data
            hist_was_indexed = isinstance(df_working.index, pd.DatetimeIndex)
            if hist_was_indexed:
                df_working.reset_index(inplace=True)
            
            # Handle index management for water change data
            water_was_indexed = isinstance(water_change_df_working.index, pd.DatetimeIndex)
            if water_was_indexed:
                water_change_df_working.reset_index(inplace=True)
            
            # Ensure datetime columns are properly formatted
            df_working[index_name] = pd.to_datetime(df_working[index_name], format='ISO8601')
            water_change_df_working['changed_at'] = pd.to_datetime(water_change_df_working['changed_at'], format='ISO8601')

            # Initialize all features with default values
            df_working['minutes_after_water_change'] = self.DEFAULT_MINUTES_AFTER_WATER_CHANGE
            # df_working['diff_water_temp_after_change'] = 0.0
            df_working['percentage_water_changed'] = self.DEFAULT_PERCENTAGE_CHANGED

            # Process each water change event
            for _, water_change in water_change_df_working.iterrows():
                water_change_time = water_change['changed_at']
                # water_temp_added = water_change['water_temperature_added']
                percentage_changed = water_change['percentage_changed']
                
                # Calculate time differences for all records (vectorized operation)
                time_diffs = (df_working[index_name] - water_change_time).dt.total_seconds() / 60
                
                # Find records within the time window after this water change
                within_window = (time_diffs >= 0) & (time_diffs <= self.WATER_CHANGE_WINDOW_MINUTES)
                
                # Update features for records within the window
                # Only update if this water change is more recent than previously recorded ones
                closer_to_change = time_diffs < df_working['minutes_after_water_change']
                update_mask = within_window & closer_to_change
                
                if update_mask.any():
                    # Update time since water change
                    df_working.loc[update_mask, 'minutes_after_water_change'] = time_diffs[update_mask].astype(int)
                    
                    # # Update temperature difference feature
                    # df_working.loc[update_mask, 'diff_water_temp_after_change'] = (
                    #     df_working.loc[update_mask, 'water_temperature'] - water_temp_added
                    # )
                    
                    # Update percentage changed feature
                    df_working.loc[update_mask, 'percentage_water_changed'] = percentage_changed

            # Restore the original index structure if needed
            if hist_was_indexed:
                df_working.set_index(index_name, inplace=True)

            if features is not None:
                features.extend([
                    'minutes_after_water_change',
                    # 'diff_water_temp_after_change',
                    'percentage_water_changed'
                ])

            return df_working

        except Exception as e:
            raise ValueError(f"Error preparing features with water change data: {e}")

    def prepare_feed_features(self,
                          historical_data: pd.DataFrame,
                          feed_data: pd.DataFrame,
                          features: Optional[List[str]]) -> pd.DataFrame:
        """
        Prepare features related to feed data with enhanced time window handling.
        
        Args:
            historical_data: DataFrame with pH measurements and timestamps
            feed_data: DataFrame with feed events and 'fed_at' timestamps
            
        Returns:
            DataFrame with added 'minutes_after_feed' feature
        """
        try:
            # Work with copies to avoid modifying original data
            df_working = historical_data.copy()
            feed_df_working = feed_data.copy()

            feed_was_indexed = isinstance(feed_df_working.index, pd.DatetimeIndex)
            if feed_was_indexed:
                feed_df_working.reset_index(inplace=True)

            if feed_df_working.empty:
                historical_data['minutes_after_feed'] = self.DEFAULT_MINUTES_AFTER_FEED
                return historical_data
            
            # Handle index and datetime conversion
            hist_was_indexed = isinstance(df_working.index, pd.DatetimeIndex)
            if hist_was_indexed:
                df_working.reset_index(inplace=True)
            
            # Ensure datetime columns are properly formatted
            df_working['created_at'] = pd.to_datetime(df_working['created_at'])
            
            # Validate and convert feed data
            if 'fed_at' not in feed_df_working.columns:
                raise ValueError("feed_data must contain 'fed_at' column")
            
            feed_df_working['fed_at'] = pd.to_datetime(feed_df_working['fed_at'])
            
            # Check for time overlap between datasets
            hist_start = df_working['created_at'].min()
            hist_end = df_working['created_at'].max()
            feed_start = feed_df_working['fed_at'].min()
            feed_end = feed_df_working['fed_at'].max()
            
            overlap = (feed_start <= hist_end) and (hist_start <= feed_end)
            
            # Initialize the feature with default values
            df_working['minutes_after_feed'] = self.DEFAULT_MINUTES_AFTER_FEED
            
            if not overlap:
                # No temporal overlap between datasets
                if hist_was_indexed:
                    df_working.set_index('created_at', inplace=True)
                return df_working
            
            # Process each feed event
            for _, feed_row in feed_df_working.iterrows():
                feed_time = feed_row['fed_at']
                
                # Calculate time differences for all records (vectorized operation)
                time_diffs = (df_working['created_at'] - feed_time).dt.total_seconds() / 60
                
                # Find records within the time window after this feed
                within_window = (time_diffs >= 0) & (time_diffs <= self.AFTER_FEED_WINDOW_MINUTES)
                
                # Update only if this feed event is closer than previously recorded ones
                closer_to_feed = time_diffs < df_working['minutes_after_feed']
                update_mask = within_window & closer_to_feed
                
                if update_mask.any():
                    df_working.loc[update_mask, 'minutes_after_feed'] = time_diffs[update_mask].astype(int)
            
            # Restore original index structure if needed
            if hist_was_indexed:
                df_working.set_index('created_at', inplace=True)

            if features is not None:
                features.append('minutes_after_feed')
            
            return df_working
        
        except Exception as e:
            raise ValueError(f"Error preparing feed features: {e}")
    